"""Transport-independent service/action facade, exercised without a ROS install."""
from __future__ import annotations

import asyncio

from .contracts import ContractError, checked_copy, digest


class RuntimeBridge:
    def __init__(self, runtime, reasoner=None, *, capture_registry=None):
        from .perception.images import CaptureRegistry
        self.runtime = runtime
        self.reasoner = reasoner
        self.capture_registry = capture_registry or CaptureRegistry()
        self.admissions = {}
        self.dispatches = {}
        self.completed = {}
        self.manual_nodes = set()
        self.used_dispatches = set()
        self.used_node_attempts = set()

    def _seal_manual_epoch(self, epoch):
        executor = self.runtime.executor
        if self.manual_nodes or not executor.safety.held_verified or self.runtime.world.snapshot().execution_epoch == epoch:
            return
        try:
            self.runtime.world.seal_epoch(epoch)
        except ContractError as exc:
            executor.safety.fault_latched = True
            executor.trace.emit("reconciliation_required", detail=str(exc))

    async def validate(self, plan, *, snapshot_id, epoch, node_id=""):
        plan = checked_copy(plan)
        world = self.runtime.world.snapshot()
        if snapshot_id != world.snapshot_id or epoch != world.execution_epoch:
            raise ContractError("Validation request state is stale")
        validator = self.runtime.validator
        key = digest(plan)
        if not node_id:
            admission = await asyncio.to_thread(validator.admit, plan)
            self.admissions[key] = (plan, admission)
            self.completed.setdefault(key, set())
            return admission
        if key not in self.admissions:
            raise ContractError("Whole plan must be admitted before individual dispatch validation")
        stored_plan, admission = self.admissions[key]
        dispatch = await asyncio.to_thread(validator.dispatch, stored_plan, node_id, admission,
                                           completed_nodes=set(self.completed[key]))
        self.dispatches[dispatch.receipt_id] = (key, dispatch)
        return dispatch

    async def execute_skill(self, request):
        """Use the same executor path; a receipt cannot be turned into another action."""
        request = checked_copy(request)
        if set(request) != {"validation_id", "task_id", "execution_epoch", "node_id", "skill_id", "args", "attempt"}:
            raise ContractError("ExecuteSkill fields do not match the closed request contract")
        executor = self.runtime.executor
        if executor._run_lock.locked() or executor._task_lock.locked() or executor._external_reasoning:
            raise ContractError("Task supervisor already owns execution")
        validation_id = request["validation_id"]
        if validation_id in self.used_dispatches or validation_id not in self.dispatches:
            raise ContractError("Dispatch receipt is absent or already consumed")
        key, dispatch = self.dispatches[validation_id]
        plan, admission = self.admissions[key]
        node = next(n for n in plan["nodes"] if n["id"] == dispatch.node_id)
        if (request["task_id"] != plan["task_id"] or request["execution_epoch"] != plan["execution_epoch"] or
                request["node_id"] != node["id"] or request["skill_id"] != node["skill"] or
                checked_copy(request["args"]) != node["args"] or type(request["attempt"]) is not int or request["attempt"] != 1
                or type(request["execution_epoch"]) is not int):
            raise ContractError("ExecuteSkill request does not match its validated node")
        attempt_key = (plan["task_id"], plan["execution_epoch"], node["id"], request["attempt"])
        if attempt_key in self.used_node_attempts:
            raise ContractError("This node attempt already consumed its execution authority")
        manual_owner = (plan["execution_epoch"], key)
        if executor._manual_plan is not None and executor._manual_plan[0] == plan["execution_epoch"] and executor._manual_plan != manual_owner:
            raise ContractError("A different manually dispatched plan already owns this epoch")
        if node["id"] in self.manual_nodes or node["id"] in self.completed[key]:
            raise ContractError("Node already active or complete")
        self.runtime.validator.verify_receipt(dispatch, plan, node["id"])
        executor._manual_plan = manual_owner
        self.used_dispatches.add(validation_id)
        self.used_node_attempts.add(attempt_key)
        self.manual_nodes.add(node["id"])
        acquired = False
        started = False
        safe_to_release = False
        watchdog = asyncio.create_task(executor.safety.watch())
        try:
            deadline = asyncio.get_running_loop().time() + self.runtime.catalog.skills[node["skill"]]["timing"]["timeout_s"]
            while True:
                executor.safety.check_dispatch(plan["execution_epoch"], skills=(node["skill"],))
                executor.resources.changed.clear()
                if executor.resources.try_acquire(node["id"], self.runtime.registry.claims(node["skill"])):
                    acquired = True
                    break
                changed = asyncio.create_task(executor.resources.changed.wait())
                stopped = asyncio.create_task(executor.safety.stop_requested.wait())
                try:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise ContractError("Resource wait deadline expired")
                    done, _ = await asyncio.wait({changed, stopped}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                    if not done:
                        raise ContractError("Resource wait deadline expired")
                finally:
                    changed.cancel()
                    stopped.cancel()
                    await asyncio.gather(changed, stopped, return_exceptions=True)
            # New validation is required if waiting or another completion changes
            # dependencies; _run_node performs that dispatch validation itself.
            executor._completed = self.completed[key]
            started = True
            result = await executor._run_node(plan, node, admission)
            safe_to_release = result.backend_quiescent
            if result.status != "succeeded":
                executor._cancel.set()
                await executor._stop_safely(result.failure_code or "SKILL_FAILED")
            if result.backend_quiescent:
                executor.resources.release(node["id"])
                acquired = False
            return result
        except asyncio.CancelledError:
            from .executor import NodeResult
            await executor.cancel("USER_CANCELLED")
            safe_to_release = not started and executor.safety.held_verified
            return NodeResult(node["id"], "cancelled", "cancelled", backend_quiescent=safe_to_release)
        except Exception as exc:
            from .executor import NodeResult
            await executor._stop_safely("DIRECT_EXECUTION_FAULT", fault=True)
            safe_to_release = not started and executor.safety.held_verified
            return NodeResult(node["id"], "failed", "runtime_error", str(exc), backend_quiescent=safe_to_release)
        finally:
            watchdog.cancel()
            await asyncio.wait({watchdog}, timeout=executor.safety.stop_timeout_s)
            self.manual_nodes.discard(node["id"])
            if acquired and safe_to_release:
                executor.resources.release(node["id"])
            self._seal_manual_epoch(plan["execution_epoch"])

    async def execute_task(self, *, task_id, task_text, plan=None):
        if self.manual_nodes:
            raise ContractError("Individual skill execution is already active")
        if task_id != self.runtime.world.snapshot().context["task_id"]:
            raise ContractError("Task ID does not match the locally initialized goal")
        if plan is not None:
            return await self.runtime.executor.run_plan(plan)
        if self.reasoner is None:
            raise ContractError("Astra is not configured")
        return await self.runtime.executor.run_task(task_text, self.reasoner)

    async def generate_plan(self, request):
        if self.reasoner is None:
            raise ContractError("Astra is not configured")
        executor = self.runtime.executor
        if executor._run_lock.locked() or executor._task_lock.locked() or executor._external_reasoning or self.manual_nodes:
            raise ContractError("Another operation owns the task/held reasoning state")
        snapshot = self.runtime.world.snapshot()
        supplied = checked_copy(request["context"])
        if (request["task_id"], request["epoch"], supplied["snapshot_id"]) != (snapshot.context["task_id"], snapshot.execution_epoch, snapshot.snapshot_id):
            raise ContractError("Reasoning request is stale or belongs to another task")
        if request["catalog_hash"] != self.runtime.catalog.hash:
            raise ContractError("Reasoning request catalog identity mismatch")
        # Use trusted current context; the caller cannot erase hazards or replace the goal.
        context = snapshot.context
        images = self.resolve_images(request.get("image_ids", []))
        async def invoke():
            return await self.reasoner.generate_plan(context, task_text=request["task_text"], images=images,
                                                    request_id=request.get("request_id"), feedback=request.get("feedback"))
        return await self._held_reasoning(snapshot, request.get("request_id"), invoke)

    async def ground_target(self, request):
        """Grounding shares the same held resource lease as task reasoning."""
        if self.reasoner is None:
            raise ContractError("Astra is not configured")
        request = checked_copy(request)
        if set(request) - {"request_id", "epoch", "entity_id", "image_ids", "query"}:
            raise ContractError("Grounding request has unknown fields")
        snapshot = self.runtime.world.snapshot()
        if type(request["epoch"]) is not int or request["epoch"] != snapshot.execution_epoch:
            raise ContractError("Grounding request uses a stale epoch")
        context = snapshot.context
        if request["entity_id"] not in {entity["entity_id"] for entity in context["entities"]}:
            raise ContractError("Grounding target has no locally registered entity")
        images = self.resolve_images(request.get("image_ids", []))
        async def invoke():
            return await self.reasoner.ground_target(context, request["entity_id"], images,
                                                     query=request.get("query", ""), request_id=request.get("request_id"))
        return await self._held_reasoning(snapshot, request.get("request_id"), invoke)

    async def _held_reasoning(self, snapshot, request_id, invoke):
        executor = self.runtime.executor
        if executor._run_lock.locked() or executor._task_lock.locked() or executor._external_reasoning or self.manual_nodes:
            raise ContractError("Another operation owns the task/held reasoning state")
        owner = "bridge-cloud:" + str(request_id or "request")
        if not executor.resources.try_acquire(owner, self.runtime.catalog.skills["observe"]["claims"]):
            raise ContractError("Cloud reasoning requires exclusive held command resources")
        executor._external_reasoning = True
        try:
            await executor._bounded(executor.safety.verify_hold(), executor.safety.stop_timeout_s)
            response = await executor._bounded(invoke(), executor._reasoning_deadline_s)
            if self.runtime.world.snapshot().execution_epoch != snapshot.execution_epoch:
                raise ContractError("Reasoning result belongs to a revoked epoch")
            return response
        finally:
            executor._external_reasoning = False
            try:
                executor.safety.held_verified = await executor._measured_quiescence()
            except Exception:
                executor.safety.held_verified = False
            if executor.safety.held_verified:
                executor.resources.release(owner)
            else:
                await executor._stop_safely("REASONING_HOLD_LOST", fault=True)

    def resolve_images(self, image_ids):
        if len(image_ids) != len(set(image_ids)):
            raise ContractError("Image does not match a locally registered minimized capture")
        try:
            return tuple(self.capture_registry.resolve(image_id)[0] for image_id in image_ids)
        except (KeyError, ValueError) as exc:
            raise ContractError("Image capture is unknown, stale or invalid") from exc

    async def stop(self, reason):
        old_epoch = self.runtime.world.snapshot().execution_epoch
        await self.runtime.executor.cancel(reason)
        if self.runtime.executor._manual_plan is not None:
            self._seal_manual_epoch(old_epoch)
        return self.runtime.world.snapshot().execution_epoch
