"""Event-driven DAG execution with catalog ownership and measured effect commits."""
from __future__ import annotations

import asyncio
import inspect
import math
from dataclasses import asdict, dataclass, field

from .contracts import ContractError, checked_copy, strict_loads, validate_schema
from .resources import ResourceManager
from .safety import ConfirmationStore, SafetyError, SafetySupervisor, action_digest
from .telemetry import TraceRecorder


@dataclass
class NodeResult:
    node_id: str
    status: str
    failure_code: str = ""
    detail: str = ""
    outputs: dict = field(default_factory=dict)
    commit_receipt: str = ""
    backend_quiescent: bool = False
    skill: str = ""
    pose_role: str = ""


@dataclass
class TaskResult:
    task_id: str
    status: str
    nodes: list[NodeResult]
    reason: str = ""
    task_replans: int = 0
    simulation_only: bool = True

    def to_dict(self):
        return asdict(self)


class DagExecutor:
    def __init__(self, catalog, registry, world, validator, backend, *, trace=None,
                 confirmations=None, confirmation_callback=None, stop_timeout_s=2.0):
        self.catalog, self.registry, self.world = catalog, registry, world
        self.validator, self.backend = validator, backend
        self.trace = trace or TraceRecorder()
        self.resources = ResourceManager(catalog.library["resources"])
        self.confirmations = confirmations or ConfirmationStore()
        self.confirmation_callback = confirmation_callback
        self.safety = SafetySupervisor(
            backend, epoch_getter=lambda: world.snapshot().execution_epoch,
            invalidate_epoch=world.cancel_epoch, trace=self.trace, stop_timeout_s=stop_timeout_s)
        self._cancel = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()
        self._task_owner = None
        self._manual_plan = None
        self._external_reasoning = False
        self._user_cancelled = False
        self._user_cancel_reason = "USER_CANCELLED"
        self._issued_motion = set()
        self._completed: set[str] = set()
        self._authorities = {}
        reasoning_policy = strict_loads((catalog.root / "config/reasoning.json").read_bytes())
        self._max_replans = reasoning_policy["max_task_replans"]
        self._reasoning_deadline_s = reasoning_policy["event_deadline_s"]
        if (type(self._max_replans) is not int or not 0 <= self._max_replans <= 3
                or type(self._reasoning_deadline_s) not in (int, float)
                or not math.isfinite(self._reasoning_deadline_s) or self._reasoning_deadline_s <= 0):
            raise ContractError("Invalid bounded task reasoning policy")
        # Only registered code receives evidence authority. No JSON boundary exposes it.
        for skill_id in registry.available_skills:
            self._authorities[skill_id] = world.authorize_source(
                "handler:" + skill_id, catalog.predicates.keys())

    async def cancel(self, reason="USER_CANCELLED") -> None:
        self._user_cancelled = True
        self._user_cancel_reason = reason
        self._cancel.set()
        self.confirmations.revoke_all()
        await self._stop_safely(reason)

    async def _bounded(self, awaitable, timeout):
        """A non-cooperative coroutine cannot extend a safety deadline forever."""
        work = asyncio.ensure_future(awaitable)
        try:
            done, _ = await asyncio.wait({work}, timeout=timeout)
        except asyncio.CancelledError:
            work.cancel()
            work.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            raise
        if not done:
            work.cancel()
            work.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            raise SafetyError("Backend operation exceeded its bounded deadline")
        return work.result()

    @property
    def simulation_only(self):
        return getattr(self.backend, "mode", None) in SafetySupervisor.SIMULATION_MODES

    def _result(self, task_id, status, nodes, reason="", task_replans=0):
        return TaskResult(task_id, status, nodes, reason, task_replans, simulation_only=self.simulation_only)

    async def _stop_safely(self, reason, *, fault=False):
        try:
            return await self._bounded(self.safety.request_stop(reason, fault=fault),
                                       2 * self.safety.stop_timeout_s)
        except BaseException as exc:
            self.safety.held_verified = False
            self.safety.fault_latched = True
            self.safety.stop_requested.set()
            self.safety.reason = "STOP_UNVERIFIED"
            self.trace.emit("fault_latched", reason="STOP_UNVERIFIED", error_type=type(exc).__name__)
            return False

    async def _measured_quiescence(self, skill_id=None):
        async def check():
            method = self.backend.quiescent if skill_id is None else self.backend.skill_quiescent
            value = method() if skill_id is None else method(skill_id)
            return bool(await value if inspect.isawaitable(value) else value)
        return await self._bounded(check(), self.safety.stop_timeout_s)

    async def _settle_worker(self, work):
        if work is None:
            return True
        if not work.done():
            done, _ = await asyncio.wait({work}, timeout=self.safety.stop_timeout_s)
            if not done:
                work.cancel()
                done, _ = await asyncio.wait({work}, timeout=self.safety.stop_timeout_s)
                if not done:
                    self.safety.fault_latched = True
                    self.safety.held_verified = False
                    work.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                    return False
        return True

    def _emit_handler(self, event=None, **fields):
        if isinstance(event, dict):
            data = dict(event)
            name = data.pop("event", "skill_feedback")
            self.trace.emit(name, **data)
        else:
            self.trace.emit(event or "skill_feedback", **fields)

    async def _confirm(self, plan, node, dispatch):
        policy = self.catalog.skills[node["skill"]]["confirmation"]
        if policy in ("none", False, None):
            return
        if policy == "image_consent_if_needed" and self.backend.mode == "simulation_fixture":
            return  # This backend produces no images and has no image egress.
        # Profile policies are local, never model-selected authorization flags.
        if policy in {"profile_policy", "recipient_confirmation_if_handover"}:
            profiles = self.world.snapshot().context["profiles"]
            profile = next((p for p in profiles if p["profile_id"] == node["args"].get("profile_id")), None)
            if profile is not None and profile.get("simulation_only"):
                return  # Explicit synthetic environment; no assent is fabricated.
        await self.safety.verify_hold()
        if self.confirmation_callback is None:
            raise SafetyError("REQUIRES_CONFIRMATION")
        digest = action_digest(node, dict(dispatch.dependencies))
        self.trace.emit("wait_started", wait_id="confirm:" + node["id"], reason="user_confirmation")
        try:
            result = self.confirmation_callback(plan, node, digest)
            confirmation_id = (await self._bounded(result, self.catalog.skills[node["skill"]]["timing"]["timeout_s"])
                               if inspect.isawaitable(result) else result)
            self.confirmations.consume(confirmation_id, task_id=plan["task_id"], epoch=plan["execution_epoch"], node_id=node["id"], digest=digest)
        finally:
            self.trace.emit("wait_ended", wait_id="confirm:" + node["id"])

    async def _run_node(self, plan, node, admission, attempt=1):
        from .handlers import ExecutionContext
        owner = node["id"]
        epoch = plan["execution_epoch"]
        operation = f"{plan['task_id']}/{epoch}/{owner}/{attempt}"
        registered = False
        committed = False
        dispatch = None
        work = None
        cancelled = None
        worker_settled = True
        try:
            dispatch = await asyncio.to_thread(self.validator.dispatch, plan, owner, admission,
                                               completed_nodes=set(self._completed))
            await self._confirm(plan, node, dispatch)
            self.safety.check_dispatch(epoch, skills=(node["skill"],))
            self.resources.assert_owned(owner, self.registry.claims(node["skill"]))
            self.validator.verify_receipt(dispatch, plan, owner)
            snapshot = self.world.snapshot()
            self.world.register_inflight(operation, epoch, dependencies=dict(dispatch.dependencies))
            registered = True
            context = ExecutionContext(task_id=plan["task_id"], node_id=owner,
                                       attempt=attempt, execution_epoch=epoch,
                                       cancel_event=self._cancel, snapshot=snapshot,
                                       emit=self._emit_handler,
                                       validation_artifact=dispatch.geometry.artifact,
                                       validation_id=dispatch.receipt_id)
            handler = self.registry.require(node["skill"])
            self.trace.emit("node_started", node_id=owner, skill=node["skill"], attempt=attempt)
            timeout_s = self.catalog.skills[node["skill"]]["timing"]["timeout_s"]
            if node["skill"] != "observe":
                self._issued_motion.add(operation)
                self.safety.held_verified = False
            # Cancellation remains cooperative so a handler can return measured partial effects.
            work = asyncio.create_task(handler.execute(checked_copy(node["args"]), context))
            cancelled = asyncio.create_task(self._cancel.wait())
            done, _ = await asyncio.wait({work, cancelled}, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
            timed_out = not done
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)
            if work not in done:
                self._cancel.set()
                await self._stop_safely("SKILL_TIMEOUT" if timed_out else "EXECUTION_CANCELLED")
                worker_settled = await self._settle_worker(work)
                if not worker_settled:
                    raise SafetyError("Handler did not terminate after verified stop")
            outcome = work.result()
            if timed_out:
                outcome.status, outcome.failure_code = "failed", "timeout"
                outcome.detail = "Skill deadline expired; measured partial effects retained"
            if not outcome.backend_quiescent or not await self._measured_quiescence(node["skill"]):
                raise SafetyError("Handler returned before owned backend commands became quiescent")
            self._issued_motion.discard(operation)
            self.safety.held_verified = await self._measured_quiescence() and not self._issued_motion
            if outcome.metric_poses and node["skill"] != "observe":
                raise ContractError("Only local observation can propose metric pose updates")
            for record in outcome.evidence:
                self.world.register_evidence(
                    record["evidence_id"], source=self._authorities[node["skill"]],
                    predicates=record["predicates"], ttl_s=record["ttl_s"], data=record.get("data"),
                    observed_at=record.get("observed_at"), dependencies=record.get("dependencies"))
            if outcome.status not in {"succeeded", "failed", "cancelled"}:
                outcome.status, outcome.failure_code = "failed", "invalid_handler_result"
                outcome.detail = "Unknown handler terminal status; retaining measured partial effects"
            if self._cancel.is_set() or epoch != self.world.snapshot().execution_epoch:
                outcome.status, outcome.failure_code = "cancelled", "cancelled"
            if outcome.status == "succeeded":
                try:
                    validate_schema(outcome.outputs, self.catalog.skills[node["skill"]]["outputs"], "skill outputs")
                    for condition in self.catalog.bind_conditions(node["skill"], node["args"], "postconditions"):
                        if not any(effect["predicate"] == condition["predicate"] and
                                   effect["args"] == condition["args"] and effect["validity"] == "true"
                                   for effect in outcome.proposed_effects):
                            raise ContractError("Handler success omitted a measured catalog postcondition")
                except (ContractError, KeyError, TypeError) as exc:
                    outcome.status, outcome.failure_code = "failed", "invalid_handler_result"
                    outcome.detail, outcome.outputs = str(exc), {}
            receipt = self.world.commit_effects(
                operation, outcome.proposed_effects, snapshot.revision, epoch,
                backend_quiescent=outcome.backend_quiescent,
                completed_node=owner if outcome.status == "succeeded" else None,
                expected_postconditions=(self.catalog.bind_conditions(node["skill"], node["args"], "postconditions")
                                         if outcome.status == "succeeded" else None),
                metric_poses=outcome.metric_poses, metric_source=self._authorities[node["skill"]])
            committed = True
            # A handler's success must be backed by all catalog postconditions after commit.
            if outcome.status == "succeeded":
                fresh = self.world.snapshot()
                for condition in self.catalog.bind_conditions(node["skill"], node["args"], "postconditions"):
                    if fresh.fact(condition["predicate"], condition["args"]) != "true":
                        raise ContractError("Handler success lacks a measured catalog postcondition")
                self._completed.add(owner)
            result = NodeResult(owner, outcome.status, outcome.failure_code or "", outcome.detail,
                                outcome.outputs, receipt.receipt_id, True)
            self.trace.emit("effects_committed", node_id=owner, receipt_id=receipt.receipt_id,
                            rebased=receipt.rebased, status=outcome.status)
            return result
        except asyncio.CancelledError:
            self._cancel.set()
            await self._stop_safely("EXECUTOR_CANCELLED")
            worker_settled = await self._settle_worker(work)
            if worker_settled:
                self._issued_motion.discard(operation)
            if (registered and worker_settled and work is not None and not work.cancelled()
                    and self.safety.held_verified):
                try:
                    outcome = work.result()
                    if not outcome.backend_quiescent or not await self._measured_quiescence(node["skill"]):
                        raise SafetyError("Cancelled worker did not establish owned-command quiescence")
                    if outcome.metric_poses and node["skill"] != "observe":
                        raise ContractError("Only local observation can propose metric pose updates")
                    for record in outcome.evidence:
                        self.world.register_evidence(record["evidence_id"], source=self._authorities[node["skill"]],
                                                     predicates=record["predicates"], ttl_s=record["ttl_s"],
                                                     data=record.get("data"), observed_at=record.get("observed_at"),
                                                     dependencies=record.get("dependencies"))
                    receipt = self.world.commit_effects(operation, outcome.proposed_effects, snapshot.revision, epoch,
                        backend_quiescent=True, metric_poses=outcome.metric_poses,
                        metric_source=self._authorities[node["skill"]])
                    committed = True
                    return NodeResult(owner, "cancelled", "cancelled", commit_receipt=receipt.receipt_id, backend_quiescent=True)
                except Exception as exc:
                    self.trace.emit("cancelled_effects_uncommitted", node_id=owner, error_type=type(exc).__name__)
            return NodeResult(owner, "cancelled", "cancelled", backend_quiescent=worker_settled and self.safety.held_verified)
        except Exception as exc:
            self._cancel.set()
            await self._stop_safely("EXECUTION_FAULT", fault=isinstance(exc, SafetyError) and str(exc) != "REQUIRES_CONFIRMATION")
            worker_settled = await self._settle_worker(work)
            if worker_settled:
                self._issued_motion.discard(operation)
            return NodeResult(owner, "failed", "runtime_error", str(exc), backend_quiescent=worker_settled and self.safety.held_verified)
        finally:
            if cancelled is not None:
                cancelled.cancel()
            if registered and not committed:
                # Evidence remains in the world even when an effect conflict prevents commit.
                # A stopped failed action is never replayed to retry a commit.
                if worker_settled and self.safety.held_verified:
                    self.world.reconcile_operation(operation, backend_quiescent=True)
            self.trace.emit("node_finished", node_id=owner)

    async def run_plan(self, plan: dict) -> TaskResult:
        if self._external_reasoning:
            raise RuntimeError("External reasoning owns the held command state")
        if self._task_lock.locked() and self._task_owner is not asyncio.current_task():
            raise RuntimeError("Task-level reasoning already owns execution")
        if self._manual_plan is not None and self._manual_plan[0] == self.world.snapshot().execution_epoch:
            raise RuntimeError("Individual skill mode owns this execution epoch")
        if self._run_lock.locked():
            raise RuntimeError("A task is already active")
        async with self._run_lock:
            plan = checked_copy(plan)
            task_id = plan.get("task_id", "unknown")
            self._completed = set()
            self._cancel.clear()
            if self._task_owner is None:
                self._user_cancelled = False
            self.trace.emit("task_started", task_id=task_id, mode=getattr(self.backend, "mode", "unknown"))
            try:
                self.safety.check_dispatch(plan.get("execution_epoch"),
                    skills=tuple(node.get("skill") for node in plan.get("nodes", [])))
                await self.safety.verify_hold()
                self.trace.emit("wait_started", wait_id="admission", reason="plan_validation")
                try:
                    admission = await asyncio.to_thread(self.validator.admit, plan)
                finally:
                    self.trace.emit("wait_ended", wait_id="admission")
            except Exception as exc:
                self.trace.emit("plan_rejected", detail=str(exc))
                return self._result(task_id, "rejected", [], str(exc))
            nodes = {node["id"]: node for node in plan["nodes"]}
            predecessors = {node: set() for node in nodes}
            for edge in plan["edges"]:
                predecessors[edge["to"]].add(edge["from"])
            pending, active, results = set(nodes), {}, []
            attempts = {node: 0 for node in nodes}
            watchdog = asyncio.create_task(self.safety.watch())
            stop_wait = asyncio.create_task(self.safety.stop_requested.wait())
            try:
                while pending or active:
                    if self.safety.stop_requested.is_set():
                        self._cancel.set()
                    if not self._cancel.is_set():
                        for node_id in admission.order:
                            if node_id not in pending or not predecessors[node_id] <= self._completed:
                                continue
                            node = nodes[node_id]
                            if self.resources.try_acquire(node_id, self.registry.claims(node["skill"])):
                                attempts[node_id] += 1
                                pending.remove(node_id)
                                active[asyncio.create_task(self._run_node(plan, node, admission, attempts[node_id]))] = node_id
                    if not active:
                        break
                    wait_set = set(active)
                    if not stop_wait.done():
                        wait_set.add(stop_wait)
                    done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                    for work in done:
                        if work is stop_wait:
                            self._cancel.set()
                            continue
                        node_id = active.pop(work)
                        result = work.result()
                        result.skill = nodes[node_id]["skill"]
                        result.pose_role = (nodes[node_id]["args"].get("target") or {}).get("pose_role", "")
                        # Release only after the coroutine has reconciled measured terminal state.
                        if result.backend_quiescent:
                            self.resources.release(node_id)
                        if result.status != "succeeded":
                            failure = next((f for f in self.catalog.skills[nodes[node_id]["skill"]]["failures"]
                                            if f["code"] == result.failure_code), None)
                            if (not self._cancel.is_set() and failure and
                                    attempts[node_id] <= failure["local_retries"]):
                                pending.add(node_id)
                                self.trace.emit("local_retry", node_id=node_id, failure_code=result.failure_code)
                                continue
                            self._cancel.set()
                            await self._stop_safely(result.failure_code or "SKILL_FAILED",
                                                   fault=result.failure_code in {"safety_fault", "slip", "tracking_lost", "recoil", "crush_risk", "excess_load", "ownership_lost"})
                        results.append(result)
            except asyncio.CancelledError:
                await self.cancel("TASK_CANCELLED")
                for work, owner in active.items():
                    settled = await self._settle_worker(work)
                    if settled and not work.cancelled():
                        try:
                            item = work.result()
                            if isinstance(item, NodeResult):
                                results.append(item)
                                if item.backend_quiescent:
                                    self.resources.release(owner)
                        except Exception:
                            self.safety.fault_latched = True
            except Exception as exc:
                self._cancel.set()
                await self._stop_safely("SCHEDULER_FAULT", fault=True)
                self.trace.emit("scheduler_fault", error_type=type(exc).__name__)
                for work, owner in active.items():
                    if await self._settle_worker(work) and self.safety.held_verified:
                        self.resources.release(owner)
            finally:
                watchdog.cancel()
                stop_wait.cancel()
                await asyncio.gather(watchdog, stop_wait, return_exceptions=True)
            if self._cancel.is_set():
                if self.safety.held_verified:
                    try:
                        self.world.seal_epoch(plan["execution_epoch"])
                    except ContractError as exc:
                        self.safety.fault_latched = True
                        self.trace.emit("reconciliation_required", detail=str(exc))
                status = "safety_fault" if self.safety.fault_latched else ("cancelled" if self._user_cancelled else "incomplete")
                reason = self._user_cancel_reason if self._user_cancelled else self.safety.reason
            elif pending:
                status, reason = "incomplete", "No executable continuation"
            elif self.world.goal_satisfied():
                status, reason = "succeeded", "Measured task goal satisfied"
            else:
                status, reason = "incomplete", "Plan completed; task goal remains unmet"
            self.trace.emit("task_finished", task_id=task_id, status=status, reason=reason)
            return self._result(task_id, status, results, reason)

    async def run_task(self, task_text: str, reasoner, *, initial_plan=None, max_replans=3):
        """Bounded task-level reasoning/recovery; cloud requests always start held."""
        if type(max_replans) is not int or not 0 <= max_replans <= self._max_replans:
            raise ContractError("Task replan cap exceeds the configured bounded policy")
        if self._task_lock.locked() or self._run_lock.locked():
            raise RuntimeError("A task already owns execution")
        if self._external_reasoning:
            raise RuntimeError("External reasoning owns the held command state")
        if self._manual_plan is not None and self._manual_plan[0] == self.world.snapshot().execution_epoch:
            raise RuntimeError("Individual skill mode owns this execution epoch")
        async with self._task_lock:
            self._task_owner = asyncio.current_task()
            self._user_cancelled = False
            self._cancel.clear()
            self.trace.emit("task_request_started", task_id=self.world.snapshot().context["task_id"],
                            mode=getattr(self.backend, "mode", "unknown"))
            try:
                return await self._run_task_owned(task_text, reasoner, initial_plan=initial_plan, max_replans=max_replans)
            except asyncio.CancelledError:
                await self.cancel("TASK_CANCELLED")
                return self._result(self.world.snapshot().context["task_id"], "cancelled", [], "TASK_CANCELLED")
            except Exception as exc:
                await self._stop_safely("TASK_RUNTIME_FAULT", fault=True)
                return self._result(self.world.snapshot().context["task_id"], "safety_fault", [], str(exc))
            finally:
                self._task_owner = None

    async def _run_task_owned(self, task_text, reasoner, *, initial_plan, max_replans):
        history = []
        feedback = repeated = None
        plan = initial_plan
        task_id = self.world.snapshot().context["task_id"]
        for replans in range(max_replans + 1):
            if self._user_cancelled:
                return self._result(task_id, "cancelled", history, self._user_cancel_reason, replans)
            if self.safety.fault_latched:
                return self._result(task_id, "safety_fault", history, self.safety.reason, replans)
            await self.safety.verify_hold()
            if self.safety.stop_requested.is_set():
                await self.safety.reset()
            self._cancel.clear()
            if plan is None:
                self.trace.emit("wait_started", wait_id=f"reasoning:{replans}", reason="cloud_reasoning")
                provider = None
                cancelled = None
                reasoning_epoch = self.world.snapshot().execution_epoch
                try:
                    images = ()
                    supplier = getattr(self, "replan_images", None)
                    if callable(supplier) and (replans > 0 or feedback is not None):
                        try:
                            images = tuple(supplier() or ())
                        except Exception as exc:                # noqa: BLE001 - a replan without a picture still runs
                            self.trace.emit("replan_images_unavailable", detail=str(exc)[:200])
                    provider = asyncio.create_task(reasoner.generate_plan(self.world.snapshot().context, task_text=task_text,
                                                                          feedback=feedback, replan=replans > 0,
                                                                          images=images))
                    cancelled = asyncio.create_task(self._cancel.wait())
                    done, _ = await asyncio.wait({provider, cancelled}, timeout=self._reasoning_deadline_s,
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if not done:
                        provider.cancel()
                        provider.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                        await self._stop_safely("REASONING_TIMEOUT")
                        if self.safety.held_verified:
                            self.world.seal_epoch(reasoning_epoch)
                        return self._result(task_id, "safety_fault" if self.safety.fault_latched else "incomplete",
                                          history, "Reasoning event deadline expired", replans)
                    if cancelled in done:
                        provider.cancel()
                        provider.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                        if self.safety.held_verified and self.world.snapshot().execution_epoch != reasoning_epoch:
                            self.world.seal_epoch(reasoning_epoch)
                        return self._result(task_id, "cancelled" if self._user_cancelled else "incomplete", history,
                                          self._user_cancel_reason if self._user_cancelled else self.safety.reason, replans)
                    response = provider.result()
                finally:
                    if cancelled is not None:
                        cancelled.cancel()
                    if provider is not None and not provider.done():
                        provider.cancel()
                    self.trace.emit("wait_ended", wait_id=f"reasoning:{replans}")
                if response.status != "OK":
                    return self._result(task_id, "incomplete", history, response.status + ": " + response.detail, replans)
                plan = response.plan
            result = await self.run_plan(plan)
            history.extend(result.nodes)
            if result.status in {"succeeded", "safety_fault"}:
                result.nodes, result.task_replans = history, replans
                return result
            if self._user_cancelled:
                return self._result(task_id, "cancelled", history, self._user_cancel_reason, replans)
            failed = [node for node in result.nodes if node.status == "failed"]
            if failed:
                last = failed[-1]
                plan_node = next((node for node in plan["nodes"] if node["id"] == last.node_id), None)
                failure = (next((policy for policy in self.catalog.skills[plan_node["skill"]]["failures"]
                                 if policy["code"] == last.failure_code), None) if plan_node else None)
                if failure is None or failure["escalation"] == "handoff":
                    return self._result(task_id, "incomplete", history,
                                      last.failure_code + ": " + (last.detail or "Handler policy requires handback"), replans)
            if failed:
                # One retry per failing idea: the same skill failing the same way
                # twice is a fact about the world or the stack, not bad luck.
                signature = (plan_node["skill"] if plan_node else None, last.failure_code, last.detail)
                if signature == repeated:
                    return self._result(task_id, "incomplete", history,
                                      f"{last.failure_code} repeated after a replan: {last.detail}"[:512], replans)
                repeated = signature
            feedback = {"previous_attempt": {"status": result.status, "reason": result.reason[:512],
                        "last_failed_node": ({"node_id": last.node_id, "skill": plan_node["skill"] if plan_node else None,
                                              "failure_code": last.failure_code, "detail": last.detail[:512]}
                                             if failed else None)}}
            # Give each new symbolic attempt a fresh command epoch. run_plan has
            # already invalidated and reconciled failed attempts.
            if self.world.snapshot().execution_epoch == plan["execution_epoch"]:
                old = plan["execution_epoch"]
                self.world.cancel_epoch()
                self.world.seal_epoch(old)
            plan = None
        return self._result(task_id, "incomplete", history,
                          "Task replan budget exhausted; last attempt: " + result.reason[:512], max_replans)
