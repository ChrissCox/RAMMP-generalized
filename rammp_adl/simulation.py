"""Explicit discrete-event test backend and separate real MuJoCo physics replay.

Discrete simulation provides fault-injectable measured *fixture* state for
executor testing. It never advertises cuRobo, collision or contact capabilities
on hardware. Analytic joint fixtures below are protocol stimuli only, not robot
planning or a replacement for cuRobo.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
import math
import json
from pathlib import Path
import time
import uuid

from .handlers import BackendFailure, SkillOutcome
from .motion.rolling import (
    BoundaryTolerance, Candidate, JointLimits, JointState, JointTrajectory,
    MotionError, MotionIdentity, RollingController, TrajectoryPoint, TrajectoryValidator,
)


def assertion(predicate, args, validity="true"):
    return {"predicate": predicate, "args": args, "validity": validity}


@dataclass
class SimEntity:
    entity_id: str
    pose_roles: set[str] = field(default_factory=set)
    visible: bool = True
    fixed_support: str | None = None
    support_id: str | None = None
    retained: bool = False


class FixtureBackend:
    mode = "simulation_fixture"
    hardware_commands = False
    physical_capabilities = frozenset()
    # Deliberately distinct from registry capability IDs. The root must opt into
    # fixture-only execution; these never qualify a deployed robot capability.
    fixture_capabilities = frozenset({"fixture_observation", "fixture_arm", "fixture_gripper", "fixture_contact"})

    def __init__(self, context=None, *, time_scale=0.0, failures=None):
        if not 0 <= time_scale <= 10:
            raise ValueError("invalid simulation time scale")
        self.time_scale, self.context = time_scale, copy.deepcopy(context or {})
        self.entities = {e["entity_id"]: SimEntity(e["entity_id"], set(e.get("pose_roles", []))) for e in self.context.get("entities", [])}
        self.constraints = {c["constraint_id"]: {**c, "coordinate": c.get("minimum", 0)} for c in self.context.get("constraints", [])}
        self.profiles = {p["profile_id"]: p for p in self.context.get("profiles", [])}
        self.aperture_m, self.holding_id, self.current_pose = 0.04, None, None
        self.active = set()
        self.failures = {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in (failures or {}).items()}
        self.events = []
        self._stopped = set()
        self._run_tokens = {}
        for entry in self.context.get("entities", []):
            for fact in entry.get("facts", []):
                if fact.get("validity") != "true":
                    continue
                args = fact.get("args", {})
                entity = self.entities.get(args.get("entity_id"))
                if fact["predicate"] == "support_verified" and entity:
                    entity.support_id = args["support_id"]
                    if any(c["entity_id"] == entity.entity_id for c in self.constraints.values()):
                        entity.fixed_support = args["support_id"]
                if fact["predicate"] == "holding" and entity:
                    self.holding_id, entity.retained = entity.entity_id, True
                if fact["predicate"] in ("at_grasp_pose", "at_pose"):
                    self.current_pose = (args.get("entity_id"), args.get("pose_role", "grasp"))

    def simulation_capabilities(self, catalog):
        """Catalog capability IDs only for an explicitly simulation-mode registry."""
        from .contracts import CURRENT_SKILLS
        return frozenset(capability for skill_id in CURRENT_SKILLS
                         for capability in catalog.skills[skill_id]["capabilities_required"])

    def verify_artifact(self, skill_id, args, context):
        from .contracts import digest
        artifact = context.validation_artifact
        if artifact is None:
            return  # Direct fixture unit tests; the executor enforces its gate.
        if artifact.get("mode") != self.mode or artifact.get("skill") != skill_id or artifact.get("args_digest") != digest(args) or artifact.get("node_id") != context.node_id:
            raise BackendFailure("safety_fault", "fixture dispatch artifact does not match executed arguments")

    async def quiescent(self):
        return not self.active

    async def skill_quiescent(self, skill_id):
        return skill_id not in self.active

    async def stop_skill(self, skill_id, reason):
        if skill_id in self._run_tokens:
            self._stopped.add(self._run_tokens[skill_id])
        self.active.discard(skill_id)
        self.events.append({"event": "stopped", "skill": skill_id, "reason": reason})

    async def stop(self, reason="supervisor"):
        for skill in tuple(self.active):
            await self.stop_skill(skill, reason)

    def _entity(self, entity_id):
        try:
            return self.entities[entity_id]
        except KeyError as exc:
            raise BackendFailure("stale_state", "entity missing from simulated scene") from exc

    def _profile(self, profile_id):
        profile = self.profiles.get(profile_id)
        if profile is None or not profile.get("simulation_only", False):
            raise BackendFailure("safety_fault", "simulation requires an explicit simulation-only profile")
        return profile

    async def _work(self, skill, context, duration_s):
        if skill in self.active:
            raise BackendFailure("safety_fault", "duplicate backend resource owner")
        run_token = uuid.uuid4().hex
        self._run_tokens[skill] = run_token
        self.active.add(skill)
        self.events.append({"event": "started", "skill": skill, "node_id": context.node_id, "wall_s": time.monotonic(), "sim_duration_s": duration_s})
        if skill in ("move_to_pose", "follow_constraint"):
            await context.feedback(event="motion_started", skill=skill, simulated=True)
        try:
            if context.cancel_event.is_set():
                raise BackendFailure("cancelled")
            # Yield models work/concurrency; only an explicit nonzero time scale
            # introduces wall time. Production dispatch has no sleep here.
            if self.time_scale:
                try:
                    await asyncio.wait_for(context.cancel_event.wait(), timeout=duration_s*self.time_scale)
                    raise BackendFailure("cancelled")
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(0)
            if context.cancel_event.is_set() or run_token in self._stopped:
                raise BackendFailure("cancelled")
            queue = self.failures.get(context.node_id, self.failures.get(skill, []))
            if queue:
                failure = queue.pop(0)
                if failure:
                    raise BackendFailure(failure, f"injected {failure} in fixture backend")
        finally:
            if self._run_tokens.get(skill) == run_token:
                self.active.discard(skill)
                self._run_tokens.pop(skill, None)
            self.events.append({"event": "finished", "skill": skill, "node_id": context.node_id, "wall_s": time.monotonic()})
            if skill in ("move_to_pose", "follow_constraint"):
                await context.feedback(event="motion_finished", skill=skill, simulated=True)
        if context.cancel_event.is_set() or run_token in self._stopped:
            raise BackendFailure("cancelled")

    def _outcome(self, facts, *, data=None):
        evidence_id = f"sim-{uuid.uuid4().hex}"
        evidence = {"evidence_id": evidence_id, "source": self.mode, "predicates": copy.deepcopy(facts),
                    "ttl_s": 30., "data": {"simulation_only": True, "physics_validated": False, **(data or {})}}
        evidence = json.loads(json.dumps(evidence, allow_nan=False))
        effects = [{**fact, "evidence_id": evidence_id} for fact in facts]
        return SkillOutcome("succeeded", {"evidence_id": evidence_id}, [evidence], effects, True)

    async def observe(self, args, context):
        entity = self._entity(args["entity_id"])
        await self._work("observe", context, 0.3)
        if not entity.visible:
            raise BackendFailure("no_detection", "simulated object is occluded")
        facts = [assertion("observation_valid", {"entity_id": entity.entity_id, "purpose": args["purpose"]})]
        if args["purpose"] in ("pose", "grasp"):
            facts += [assertion("pose_valid", {"entity_id": entity.entity_id, "pose_role": role}) for role in sorted(entity.pose_roles)]
        return self._outcome(facts, data={"entity_visible": entity.visible, "camera": args["camera"]})

    async def move_to_pose(self, args, context):
        self._profile(args["profile_id"])
        entity = self._entity(args["target"]["entity_id"])
        role = args["target"]["pose_role"]
        if not entity.visible or role not in entity.pose_roles:
            raise BackendFailure("stale_state", "simulated role is not grounded")
        await self._work("move_to_pose", context, 1.2)
        previous = self.current_pose
        self.current_pose = (entity.entity_id, role)
        facts = [assertion("at_pose", {"entity_id": entity.entity_id, "pose_role": role})]
        if previous and previous != self.current_pose:
            facts.append(assertion("at_pose", {"entity_id": previous[0], "pose_role": previous[1]}, "false"))
            if previous[1] == "grasp":
                facts.append(assertion("at_grasp_pose", {"entity_id": previous[0]}, "false"))
        if role == "grasp":
            facts.append(assertion("at_grasp_pose", {"entity_id": entity.entity_id}))
        if role == "placement" and self.holding_id:
            carried = self._entity(self.holding_id)
            carried.support_id = entity.entity_id
            facts += [assertion("support_verified", {"entity_id": carried.entity_id, "support_id": entity.entity_id}),
                      assertion("at_release_pose", {"entity_id": carried.entity_id, "support_id": entity.entity_id})]
        return self._outcome(facts, data={"measured_symbolic_pose": self.current_pose})

    async def set_gripper(self, args, context):
        self._profile(args["profile_id"])
        if self.holding_id is not None:
            raise BackendFailure("safety_fault", "empty-gripper preshape cannot change a retained grip")
        if not math.isfinite(args["aperture_m"]) or not 0 <= args["aperture_m"] <= 0.085:
            raise BackendFailure("safety_fault", "aperture outside fixture gripper range")
        await self._work("set_gripper", context, 0.4)
        self.aperture_m = args["aperture_m"]
        return self._outcome([assertion("aperture_reached", {"aperture_m": self.aperture_m})], data={"aperture_m": self.aperture_m})

    async def grasp(self, args, context):
        self._profile(args["profile_id"])
        entity = self._entity(args["entity_id"])
        if self.holding_id or self.current_pose != (entity.entity_id, "grasp"):
            raise BackendFailure("empty_grasp", "retention requires aligned empty gripper")
        await self._work("grasp", context, 0.6)
        self.holding_id, entity.retained = entity.entity_id, True
        facts = [assertion("holding", {"entity_id": entity.entity_id}), assertion("gripper_empty", {"robot_id": "robot"}, "false")]
        if entity.fixed_support:
            facts.append(assertion("at_release_pose", {"entity_id": entity.entity_id, "support_id": entity.fixed_support}))
        return self._outcome(facts, data={"retained": entity.retained, "grasp_relation": "constrained" if entity.fixed_support else "carried"})

    async def release(self, args, context):
        self._profile(args["profile_id"])
        entity = self._entity(args["entity_id"])
        if self.holding_id != entity.entity_id or not entity.retained:
            raise BackendFailure("release_incomplete", "item is not retained")
        if entity.support_id != args["support_id"]:
            raise BackendFailure("support_lost", "support evidence missing; grip retained")
        await self._work("release", context, 0.4)
        self.holding_id, entity.retained = None, False
        return self._outcome([assertion("released", {"entity_id": entity.entity_id, "support_id": args["support_id"]}),
                              assertion("holding", {"entity_id": entity.entity_id}, "false"), assertion("gripper_empty", {"robot_id": "robot"})],
                             data={"retained": False, "supported_by": entity.support_id})

    async def follow_constraint(self, args, context):
        self._profile(args["profile_id"])
        entity = self._entity(args["entity_id"])
        constraint = self.constraints.get(args["constraint_id"])
        if not constraint or constraint["entity_id"] != entity.entity_id or constraint["unit"] != args["target_unit"]:
            raise BackendFailure("model_mismatch", "constraint identity/units disagree")
        if constraint["kind"] not in ("revolute", "prismatic"):
            raise BackendFailure("model_mismatch", "fixture supports only revolute and prismatic contact")
        if not constraint["minimum"] <= args["target_value"] <= constraint["maximum"]:
            raise BackendFailure("model_mismatch", "absolute target exceeds fixture stroke")
        if not entity.retained:
            raise BackendFailure("slip", "expected handle retention absent")
        previous = constraint["coordinate"]
        try:
            await self._work("follow_constraint", context, 1.5)
        except BackendFailure as exc:
            if exc.code in ("model_mismatch", "goal_unobserved"):
                # An injected physical mismatch happens after partial fixture
                # progress; report measured partial evidence, never target success.
                constraint["coordinate"] = previous + (args["target_value"]-previous)*0.35
                partial = self._outcome([assertion("constraint_goal_verified", {k: args[k] for k in ("constraint_id", "target_value", "target_unit")}, "false")],
                                        data={"measured_coordinate": constraint["coordinate"], "partial": True})
                exc.evidence, exc.effects = partial.evidence, partial.proposed_effects
            raise
        constraint["coordinate"] = args["target_value"]
        facts = [assertion("constraint_goal_verified", {k: args[k] for k in ("constraint_id", "target_value", "target_unit")})]
        if entity.fixed_support:
            facts.append(assertion("at_release_pose", {"entity_id": entity.entity_id, "support_id": entity.fixed_support}))
        return self._outcome(facts, data={"measured_coordinate": constraint["coordinate"], "retained": entity.retained})


class FixtureGeometry:
    """Symbolic fixture geometry only: never claims IK, collision or physics proof."""
    simulation_only = True

    def __init__(self, backend):
        if backend.mode != "simulation_fixture":
            raise ValueError("fixture geometry requires fixture backend")
        self.backend = backend

    def validate(self, node, snapshot, predicted_state, phase):
        from .contracts import ContractError, digest, fact_key
        from .validation import GeometryCheck
        skill, args = node["skill"], node["args"]
        if phase not in ("admission", "dispatch"):
            raise ContractError("invalid fixture validation phase")
        state = copy.deepcopy(predicted_state.get("geometry")) if phase == "admission" else None
        if state is None:
            state = {"pose": self.backend.current_pose, "holding_id": self.backend.holding_id,
                     "supports": {key: entity.support_id for key, entity in self.backend.entities.items()}}
        before = copy.deepcopy(state)
        established = []
        if "profile_id" in args:
            profile = self.backend.profiles.get(args["profile_id"])
            if profile is None or not profile.get("simulation_only"):
                raise ContractError("fixture requires simulation-only profile")
            established.append(assertion("motion_profile_valid", {"profile_id": args["profile_id"]}))
        if skill == "move_to_pose":
            entity_id, role = args["target"]["entity_id"], args["target"]["pose_role"]
            entity = self.backend.entities.get(entity_id)
            if entity is None or not entity.visible or role not in entity.pose_roles:
                raise ContractError("fixture target role unavailable")
            state["pose"] = (entity_id, role)
            if role == "placement" and state["holding_id"]:
                state["supports"][state["holding_id"]] = entity_id
            for parallel in predicted_state.get("parallel_nodes", ()):
                if parallel["skill"] == "set_gripper" and not 0 <= parallel["args"]["aperture_m"] <= 0.085:
                    raise ContractError("parallel preshape exceeds full fixture aperture envelope")
        elif skill == "set_gripper":
            if state["holding_id"] or not 0 <= args["aperture_m"] <= 0.085:
                raise ContractError("invalid fixture preshape")
        elif skill == "grasp":
            if tuple(state["pose"] or ()) != (args["entity_id"], "grasp") or state["holding_id"]:
                raise ContractError("fixture grasp alignment/retention invalid")
            state["holding_id"] = args["entity_id"]
        elif skill == "release":
            if state["holding_id"] != args["entity_id"] or state["supports"].get(args["entity_id"]) != args["support_id"]:
                raise ContractError("fixture release lacks actual/predicted support")
            established += [assertion("at_release_pose", {"entity_id": args["entity_id"], "support_id": args["support_id"]}),
                            assertion("support_verified", {"entity_id": args["entity_id"], "support_id": args["support_id"]})]
            state["holding_id"] = None
        elif skill == "follow_constraint":
            constraint = self.backend.constraints.get(args["constraint_id"])
            holding = state["holding_id"] == args["entity_id"]
            if not constraint or constraint["kind"] not in ("revolute", "prismatic") or not holding:
                raise ContractError("fixture contact relation unavailable")
            if constraint["entity_id"] != args["entity_id"] or constraint["unit"] != args["target_unit"] or not constraint["minimum"] <= args["target_value"] <= constraint["maximum"]:
                raise ContractError("fixture constraint identity, unit or target invalid")
            established.append(assertion("contact_ready", {k: args[k] for k in ("entity_id", "constraint_id", "profile_id")}))
            state["pose"] = (args["entity_id"], "constraint_end")
        else:
            raise ContractError("fixture geometry skill unsupported")
        artifact = {"mode": "simulation_fixture", "physics_validated": False, "node_id": node["id"],
                    "skill": skill, "args_digest": digest(args), "start_state": before, "end_state": copy.deepcopy(state),
                    "aperture_envelope_m": [0., 0.085]}
        return GeometryCheck(end_state=state, dependencies={}, artifact=artifact,
                             established_facts=tuple(established), valid_for_s=30.)


# Deliberately test-only analytic JOINT-space inputs. These functions must never
# be registered as a cuRobo planner or exposed as model tool calls.
def fixture_joint_trajectory(start, end_position, *, duration_s=2., terminal_velocity=None):
    end_position = tuple(end_position)
    zero = (0.,)*len(end_position)
    terminal = JointState(end_position, tuple(terminal_velocity or zero), zero)
    return JointTrajectory(tuple(f"joint_{i+1}" for i in range(len(end_position))),
                           (TrajectoryPoint(0., start), TrajectoryPoint(duration_s, terminal)), "test_fixture_joint_space_only")


def fixture_stop_provider(validator, *, duration_s=0.5):
    def stop(now, measured, dependencies):
        end = tuple(q + v*duration_s/2 for q, v in zip(measured.position, measured.velocity))
        trajectory = fixture_joint_trajectory(measured, end, duration_s=duration_s)
        return validator.validate(trajectory, dependencies, now=now, expires_at=now+duration_s+1.)
    return stop


def rolling_fixture_trace():
    """Deterministic moving-target handoff protocol trace, not cuRobo planning."""
    zero = JointState((0.,)*7, (0.,)*7, (0.,)*7)
    validator = TrajectoryValidator(JointLimits((-6.,)*7, (6.,)*7, (3.,)*7, (20.,)*7), lambda path, deps: deps.get("scene") == "fixture-clear")
    dependencies = {"scene": "fixture-clear", "target": "same-object", "calibration": "fixture-v1"}
    initial = fixture_joint_trajectory(zero, (0.25,)*7, duration_s=2.)
    certificate = validator.validate(initial, dependencies, now=0., expires_at=10.)
    controller = RollingController(MotionIdentity("reactive-fixture", "move", 1, 1), certificate, validator, start_at=0.,
                                   tolerance=BoundaryTolerance(0.001, 0.001, 0.005), stop_provider=fixture_stop_provider(validator), stop_budget_s=0.5)
    boundary = initial.sample(0.8)
    suffix = fixture_joint_trajectory(boundary, (0.3,)*7, duration_s=1.5)
    candidate = Candidate(controller.identity, 0, 1, 0.8, boundary, validator.validate(suffix, dependencies, now=0.5, expires_at=10.))
    controller.install(candidate, now=0.5, dependencies=dependencies)
    samples = []
    for tick in range(231):
        now = tick/100
        measured = initial.sample(now) if now < 0.8 else suffix.sample(min(now-0.8, suffix.duration_s))
        controller.tick(now=now, measured=measured, dependencies=dependencies)
        samples.append({"time_s": now, "generation": controller.generation, "q": measured.position,
                        "dq": measured.velocity, "ddq": measured.acceleration})
    return {"mode": "simulation_fixture", "physics_validated": False, "curobo_planned": False,
            "state": controller.state, "events": controller.events, "samples": samples}


class MujocoReplay:
    """Physical dynamics replay of provided joint trajectories; no IK or planner."""
    hardware_commands = False

    def __init__(self, model_path=None):
        import mujoco
        self.mujoco = mujoco
        if model_path is None:
            from .contracts import Catalog
            self.simulation_root = Catalog().root / "simulation"
            self.model_path = self.simulation_root / "vendor/kinova_gen3/scene.xml"
        else:
            self.model_path = Path(model_path)
            resolved = self.model_path.resolve()
            self.simulation_root = resolved.parent if resolved.name == "rolling_scene.xml" else resolved.parents[2]
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt))
        if self.model.nq != 7 or self.model.nv != 7 or self.model.nu != 7:
            raise MotionError("replay model must be the pinned bare seven-joint Gen3")

    def home(self):
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.mujoco.mj_forward(self.model, self.data)
        return JointState(tuple(self.data.qpos), tuple(self.data.qvel), (0.,)*7)

    def replay(self, trajectory, *, settle_s=0.5, render_path=None):
        import numpy as np
        if trajectory.joint_names != self.joint_names:
            raise MotionError("trajectory/model joint order mismatch")
        self.data.qpos[:] = trajectory.points[0].state.position
        self.data.qvel[:] = trajectory.points[0].state.velocity
        self.data.ctrl[:] = self.data.qpos
        self.mujoco.mj_forward(self.model, self.data)
        maximum_error, maximum_velocity, maximum_torque, maximum_contacts = 0., 0., 0., 0
        contact_pairs = set()
        samples = []
        ticks = math.ceil((trajectory.duration_s+settle_s)/self.model.opt.timestep)
        for tick in range(ticks+1):
            elapsed = tick*self.model.opt.timestep
            target = trajectory.sample(min(elapsed, trajectory.duration_s))
            self.data.ctrl[:] = target.position
            self.mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise MotionError("MuJoCo state became non-finite")
            error = float(np.max(np.abs(self.data.qpos-np.asarray(target.position))))
            maximum_error = max(maximum_error, error)
            maximum_velocity = max(maximum_velocity, float(np.max(np.abs(self.data.qvel))))
            maximum_torque = max(maximum_torque, float(np.max(np.abs(self.data.actuator_force))))
            maximum_contacts = max(maximum_contacts, self.data.ncon)
            for contact in self.data.contact:
                bodies = tuple(self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[geom]))
                               for geom in (contact.geom1, contact.geom2))
                contact_pairs.add(bodies)
            if tick % 25 == 0:
                samples.append({"time_s": elapsed, "q": self.data.qpos.tolist(), "dq": self.data.qvel.tolist(), "tracking_error_rad": error})
        final_error = float(np.max(np.abs(self.data.qpos-np.asarray(trajectory.points[-1].state.position))))
        if render_path:
            import cv2
            with self.mujoco.Renderer(self.model, height=480, width=640) as renderer:
                renderer.update_scene(self.data)
                rgb = renderer.render()
                destination = Path(render_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError("simulation rendering could not be saved")
        return {"mode": "mujoco_physics_replay", "mujoco_version": self.mujoco.__version__,
                "model_source_commit": "71f066ad0be9cd271f7ed58c030243ef157af9f4",
                "trajectory_provenance": trajectory.provenance, "curobo_planned": trajectory.provenance.startswith("rammp_curobo:"),
                "hardware_validated": False, "gripper_simulated": False,
                "maximum_tracking_error_rad": maximum_error, "final_tracking_error_rad": final_error,
                "maximum_joint_velocity_rad_s": maximum_velocity, "maximum_actuator_torque_nm": maximum_torque,
                "maximum_contact_count": maximum_contacts, "simulated_duration_s": ticks*self.model.opt.timestep,
                "contact_body_pairs": [list(pair) for pair in sorted(contact_pairs)],
                "model_limitations": ["Bare arm; no gripper or installed D405/tool calibration",
                                      "Project overlay excludes the adjacent base/shoulder mesh overlap; no hardware collision certification"
                                      if self.model_path.name == "rolling_scene.xml" else
                                      "Upstream base/shoulder mesh overlap produces contacts; replay does not certify collision-free motion"],
                "samples": samples}

    def replay_rolling_fixture(self):
        """Track an in-flight suffix switch in real MuJoCo rigid-body dynamics.

        Inputs are explicitly analytic joint test fixtures. Tolerances below
        describe this simulation check only and never become hardware profiles.
        """
        import numpy as np
        # Explicit project overlay removes only the known adjacent mesh overlap.
        # The upstream XML/mesh files themselves remain unchanged.
        rolling_scene = self.simulation_root / "rolling_scene.xml"
        self.model = self.mujoco.MjModel.from_xml_path(str(rolling_scene))
        self.data = self.mujoco.MjData(self.model)
        self.home()
        for _ in range(500):
            self.mujoco.mj_step(self.model, self.data)
        # Continue the existing stationary position-controller command. Actual
        # gravity/tracking error is checked against the same boundary tolerance;
        # snapping the reference to measured position would create a torque step.
        start = JointState(tuple(self.data.ctrl), (0.,)*7, (0.,)*7)
        target = list(start.position)
        target[0] += 0.1
        initial = fixture_joint_trajectory(start, target, duration_s=3.)
        validator = TrajectoryValidator(JointLimits((-7.,)*7, (7.,)*7, (3.,)*7, (30.,)*7),
                                        lambda path, deps: deps.get("scene") == "joint-fixture-bare-arm")
        dependencies = {"scene": "joint-fixture-bare-arm", "model": "gen3-vendor", "target": "same-joint-fixture"}
        certificate = validator.validate(initial, dependencies, now=0., expires_at=10.)
        controller = RollingController(MotionIdentity("physics-joint-fixture", "move", 1, 1), certificate, validator,
                                       start_at=0., tolerance=BoundaryTolerance(0.015, 0.15, 5.),
                                       stop_provider=fixture_stop_provider(validator), stop_budget_s=0.5,
                                       switch_lead_s=0.05, activation_lateness_s=0.005)
        switch_at = 1.2
        boundary = initial.sample(switch_at)
        target[0] += 0.05
        suffix = fixture_joint_trajectory(boundary, target, duration_s=2.5)
        candidate = Candidate(controller.identity, 0, 1, switch_at, boundary,
                              validator.validate(suffix, dependencies, now=0.7, expires_at=10.))
        step_dt = self.model.opt.timestep
        samples, maximum_error = [], 0.
        installed, activation_error = False, None
        ticks = math.ceil(4.2/step_dt)
        for tick in range(ticks+1):
            elapsed = tick*step_dt
            actual = JointState(tuple(self.data.qpos), tuple(self.data.qvel), tuple(self.data.qacc))
            if not installed and elapsed >= 0.7:
                controller.install(candidate, now=elapsed, dependencies=dependencies)
                installed = True
            old_generation = controller.generation
            command = controller.tick(now=elapsed, measured=actual, dependencies=dependencies)
            if old_generation != controller.generation:
                activation_error = {"position_rad": float(np.max(np.abs(np.asarray(actual.position)-np.asarray(boundary.position)))),
                                    "velocity_rad_s": float(np.max(np.abs(np.asarray(actual.velocity)-np.asarray(boundary.velocity)))),
                                    "acceleration_rad_s2": float(np.max(np.abs(np.asarray(actual.acceleration)-np.asarray(boundary.acceleration))))}
            if controller.state == "fault":
                raise MotionError("MuJoCo rolling fixture entered supervisor fault: " + str(controller.events[-1]))
            if command is not None:
                self.data.ctrl[:] = command.position
                maximum_error = max(maximum_error, float(np.max(np.abs(self.data.qpos-np.asarray(command.position)))))
            self.mujoco.mj_step(self.model, self.data)
            if tick % 25 == 0:
                samples.append({"time_s": elapsed, "q": self.data.qpos.tolist(), "dq": self.data.qvel.tolist(), "generation": controller.generation})
        return {"mode": "mujoco_rolling_joint_fixture", "curobo_planned": False, "hardware_validated": False,
                "gripper_simulated": False, "state": controller.state, "generation": controller.generation,
                "actual_activation_error": activation_error, "maximum_tracking_error_rad": maximum_error,
                "final_tracking_error_rad": float(np.max(np.abs(self.data.qpos-np.asarray(target)))),
                "events": controller.events, "samples": samples,
                "model_overlay": "simulation/rolling_scene.xml: exclude base_link/shoulder_link adjacent mesh overlap",
                "limitations": ["Joint fixtures only; no Cartesian/ADL planning", "Sampled joint checks are not swept/stopping collision proof", "Simulation tolerances are not commissioned profiles"]}
