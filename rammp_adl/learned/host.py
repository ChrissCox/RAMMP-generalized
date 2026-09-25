"""The primitive API bound to a runtime: what a learned skill's calls actually do.

ExecutorHost runs every motion primitive as a plan that the validator admits and the executor runs, under
the supervisor and the guards, exactly as a planned step; a refused or failed step comes back to the skill
as RobotError, and a latched fault or a cancel ends the skill (SkillAbort). DryRunHost moves nothing: it
appends each motion step to one growing chain and has the validator admit the whole chain, so every step's
preconditions are checked against what the earlier steps would have established; state questions are
answered from that prediction. Perception is the runtime's world (the discovered entities) until local
perception replaces it behind find.
"""
from __future__ import annotations

import asyncio
import copy
import re

from ..contracts import ContractError
from ..plans import chain_plan, profiles_by_class
from .jail import SkillAbort

#: API role -> the pose role the world measures.
ROLES = {"pregrasp": "pregrasp", "grasp": "grasp", "retract": "retract", "above": "staging"}


class PrimitiveFailed(RuntimeError):
    """A primitive that was refused or failed; the skill receives the message."""


class _Binding:
    def __init__(self, runtime, *, support_of=None, finder=None, cancel=None, log=None):
        self.runtime, self.finder, self.cancel, self.log = runtime, finder, cancel, log
        context = runtime.world.snapshot().context
        self.support_of = dict(support_of or {})
        for constraint in context.get("constraints", ()):
            if constraint.get("surface_entity_id"):
                self.support_of.setdefault(constraint["entity_id"], constraint["surface_entity_id"])
        self.steps = []

    # -- what the world holds -------------------------------------------------
    def _context(self):
        return self.runtime.world.snapshot().context

    def _entity(self, object_id):
        entity = next((e for e in self._context()["entities"] if e["entity_id"] == object_id), None)
        if entity is None or object_id == "robot":
            raise PrimitiveFailed(f"there is no object {object_id!r}; find() lists what is in view")
        return entity

    def _constraint(self, object_id):
        found = [c for c in self._context().get("constraints", ()) if c["entity_id"] == object_id]
        if not found:
            raise PrimitiveFailed(f"{object_id} has no measured way of moving; it is not a handle on a door or drawer")
        return found[0]

    def _profile(self, safety_class):
        profile = profiles_by_class(self._context()).get(safety_class)
        if profile is None:
            raise PrimitiveFailed(f"the deployment has no {safety_class} motion profile")
        return profile

    def _node(self, name, args):
        """The catalog step a motion primitive stands for."""
        if name == "move_to":
            entity = self._entity(args["object"])
            role = ROLES[args["role"]]
            if role not in (entity.get("pose_roles") or ()):
                raise PrimitiveFailed(f"{args['object']} has no {args['role']} pose; it has {', '.join(entity.get('pose_roles') or ()) or 'none'}")
            return {"skill": "move_to_pose", "args": {"target": {"entity_id": args["object"], "pose_role": role},
                                                      "profile_id": self._profile("transit")}}
        if name == "open_hand":
            return {"skill": "set_gripper", "args": {"aperture_m": args["aperture_m"], "profile_id": self._profile("gripper")}}
        if name == "grasp":
            self._entity(args["object"])
            return {"skill": "grasp", "args": {"entity_id": args["object"], "profile_id": self._profile("gripper")}}
        if name == "release":
            self._entity(args["object"])
            support = args.get("onto") or self.support_of.get(args["object"])
            if support is None:
                raise PrimitiveFailed(f"say what {args['object']} is released onto (onto=...)")
            return {"skill": "release", "args": {"entity_id": args["object"], "support_id": support,
                                                 "profile_id": self._profile("gripper")}}
        if name == "move_part":
            constraint = self._constraint(args["object"])
            if args["unit"] != constraint["unit"]:
                raise PrimitiveFailed(f"{args['object']} moves in {constraint['unit']}, not {args['unit']}")
            if not constraint["minimum"] <= args["amount"] <= constraint["maximum"]:
                raise PrimitiveFailed(f"{args['object']} moves from {constraint['minimum']} to {constraint['maximum']} {constraint['unit']}")
            return {"skill": "follow_constraint", "args": {"entity_id": args["object"], "constraint_id": constraint["constraint_id"],
                                                           "target_value": args["amount"], "target_unit": constraint["unit"],
                                                           "profile_id": self._profile("contact")}}
        raise PrimitiveFailed(f"{name} is not a plan step")

    def _step(self, node):
        self.steps.append(node)
        return {**copy.deepcopy(node), "id": f"skill_{len(self.steps)}_{node['skill']}"}

    # -- perception and state --------------------------------------------------
    def _find(self, query):
        if self.finder is not None:
            return self.finder(query)
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        snapshot = self.runtime.world.snapshot()
        hits = []
        for entity in snapshot.context["entities"]:
            if entity["entity_id"] == "robot":
                continue
            have = set(re.findall(r"[a-z0-9]+", (entity.get("label", "")+" "+entity["entity_id"]).lower()))
            if not words & have:
                continue
            pose = next((snapshot.metric_poses[(entity["entity_id"], role)] for role in ("grasp", "pregrasp", "staging")
                         if (entity["entity_id"], role) in snapshot.metric_poses), None)
            hits.append({"id": entity["entity_id"], "label": entity.get("label", ""),
                         "roles": [role for api, role in ROLES.items() if role in (entity.get("pose_roles") or ())],
                         "movable_part": any(c["entity_id"] == entity["entity_id"] for c in snapshot.context.get("constraints", ())),
                         "moves": next(({"kind": c["kind"], "unit": c["unit"], "from": c["minimum"], "to": c["maximum"]}
                                        for c in snapshot.context.get("constraints", ()) if c["entity_id"] == entity["entity_id"]), None),
                         "position_m": None if pose is None else [round(v, 4) for v in pose.position_m],
                         "match": len(words & have)})
        hits.sort(key=lambda hit: -hit.pop("match"))
        return hits

    def scene(self):
        """Everything the task's world holds, as find() reports each: what a skill writer is shown."""
        words = " ".join(e["entity_id"] for e in self._context()["entities"] if e["entity_id"] != "robot")
        return self._find(words) if words else []

    def _check_abort(self):
        if self.cancel is not None and self.cancel.is_set():
            raise SkillAbort("the task was cancelled")


class ExecutorHost(_Binding):
    """Every motion primitive as an admitted, guarded plan run by the executor."""

    async def call(self, name, args):
        self._check_abort()
        if self.log is not None and name not in ("holding", "gripper_opening"):
            self.log(f"skill: {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"[:200])
        if name == "find":
            return self._find(args["query"])
        if name == "holding":
            return self._holding()
        if name == "gripper_opening":
            return self._gripper_opening()
        if name == "check":
            return self._check(**args)
        if name == "look":
            return await self._look(args["direction"])
        if name == "go_home":
            home = getattr(self.runtime.backend, "return_home", None)
            if home is None:
                raise PrimitiveFailed("this deployment has no home pose")
            report = await home()
            if not report.get("at_home"):
                raise PrimitiveFailed(report.get("detail") or "the arm did not get home")
            return report["done"]
        return await self._run(self._step(self._node(name, args)))

    async def _run(self, node):
        executor, world = self.runtime.executor, self.runtime.world
        if executor.safety.fault_latched:
            raise SkillAbort(f"a fault is latched: {executor.safety.reason}")
        await executor.safety.verify_hold()
        if executor.safety.stop_requested.is_set():
            await executor.safety.reset()
        plan = chain_plan([node], world.snapshot().context, self.runtime.catalog)
        result = await executor.run_plan(plan)
        if result.status == "succeeded" or (result.nodes and all(n.status == "succeeded" for n in result.nodes)):
            return True                                     # the step did its part; the task's goal is the skill's to reach
        if executor.safety.fault_latched:
            raise SkillAbort(f"a fault is latched: {executor.safety.reason}")
        if world.snapshot().execution_epoch == plan["execution_epoch"]:
            world.cancel_epoch()                            # the next step gets a fresh command epoch
            world.seal_epoch(plan["execution_epoch"])
        failed = next((n for n in result.nodes if n.status == "failed"), None)
        why = f"{failed.failure_code}: {failed.detail}" if failed else result.reason
        raise PrimitiveFailed(f"{node['skill']} did not succeed ({why})"[:600])

    def _holding(self):
        held = getattr(self.runtime.backend, "holding_id", None)
        if held is not None:
            return held
        snapshot = self.runtime.world.snapshot()
        return next((e["entity_id"] for e in snapshot.context["entities"]
                     if snapshot.fact("holding", {"entity_id": e["entity_id"]}) == "true"), None)

    def _gripper_opening(self):
        backend = self.runtime.backend
        client = getattr(backend, "client", None)
        live = client.live_joints() if client is not None else None
        if live is None or live.get("knuckle_rad") is None or not hasattr(backend, "aperture"):
            raise PrimitiveFailed("the gripper's opening cannot be measured here")
        return round(float(backend.aperture.to_aperture(live["knuckle_rad"])), 4)

    def _check(self, condition, object=None, value=None):
        snapshot = self.runtime.world.snapshot()
        if condition == "holding":
            return self._holding() == object if object else self._holding() is not None
        if condition == "gripper_empty":
            return snapshot.fact("gripper_empty", {"robot_id": "robot"}) == "true"
        if condition == "part_moved":
            constraint = self._constraint(object)
            moved = getattr(self.runtime.backend, "constraint_progress", {}).get(constraint["constraint_id"])
            if moved is None:
                moved = next((a["target_value"] for key, validity in snapshot.facts.items() if validity == "true"
                              for a in [dict(key[1])] if key[0] == "constraint_goal_verified"
                              and a.get("constraint_id") == constraint["constraint_id"]), 0.)
            return float(moved) >= (value if value is not None else constraint["maximum"])-1e-6
        if condition == "goal":
            return bool(self.runtime.world.goal_satisfied())
        raise PrimitiveFailed("conditions: holding, gripper_empty, part_moved, goal")

    async def _look(self, direction):
        look = getattr(self.runtime.backend, "look", None)
        if look is None:
            raise PrimitiveFailed("this deployment cannot turn its camera")
        from ..handlers import BackendFailure, ExecutionContext
        context = self._context()
        try:
            await look(direction, ExecutionContext(task_id=context["task_id"], node_id=f"look_{direction}",
                                                   execution_epoch=context["execution_epoch"]))
        except BackendFailure as exc:
            raise PrimitiveFailed(f"look {direction}: {exc.code}: {exc}") from exc
        return True


class DryRunHost(_Binding):
    """Moves nothing: each motion step joins one chain the validator must admit whole; state is predicted."""

    def __init__(self, runtime, *, reach=True, **options):
        super().__init__(runtime, **options)
        self.held, self.refusals = None, []
        # With a planner behind the backend, each move is also planned (not flown) from where the last one ended.
        self.reach = reach and hasattr(runtime.backend, "reach_check")
        self.joints = None

    async def call(self, name, args):
        self._check_abort()
        if name == "find":
            return self._find(args["query"])
        if name == "holding":
            return self.held
        if name == "gripper_opening":
            return .085 if self.held is None else None
        if name == "check":
            if args["condition"] == "holding":
                return self.held == args.get("object") if args.get("object") else self.held is not None
            if args["condition"] == "gripper_empty":
                return self.held is None
            return True                                     # what a real run would measure is assumed here
        if name in ("look", "go_home"):
            return True
        node = self._node(name, args)
        candidate = chain_plan([{**copy.deepcopy(n), "id": f"skill_{i+1}_{n['skill']}"}
                                for i, n in enumerate(self.steps+[node])], self._context(), self.runtime.catalog)
        try:
            await asyncio.to_thread(self.runtime.validator.admit, candidate)
        except ContractError as exc:
            self.refusals.append(str(exc))
            raise PrimitiveFailed(f"{node['skill']} would be refused: {exc}"[:600]) from exc
        if self.reach and name == "move_to" and node["args"]["target"]["pose_role"] != "retract":
            from ..handlers import BackendFailure
            try:
                self.joints = await self.runtime.backend.reach_check(args["object"], node["args"]["target"]["pose_role"], self.joints)
            except BackendFailure as exc:
                self.refusals.append(str(exc))
                raise PrimitiveFailed(f"move_to would fail: {exc}"[:600]) from exc
        self.steps.append(node)
        if name == "grasp":
            self.held = args["object"]
        elif name == "release":
            self.held = None
        return True
