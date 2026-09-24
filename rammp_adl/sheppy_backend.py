"""The six catalog skills executed as a client of sheppy's arm module.

Planning is the RAMMP-CuRobo container's, execution and gripping are the
kinova-gen3-ros2 container's, and this backend owns neither. It resolves a
target from the world snapshot, asks the planner for a path from the measured
joints, runs the client-side gates the driver lacks, sends one trajectory,
and commits only what it measured afterwards.

What it does not provide is stated rather than hidden. The driver executes a
static trajectory: there is no rolling replanning and no continuous handoff;
the live collision guard is the wrist depth stream when one is wired. Aperture
is mapped through a nominal 2F-85 relation until a measured table replaces it.
follow_constraint follows an installed metric constraint record as a sequence
of cuRobo plans from rest, each executed under the effort budget the record
carries; that is stop-and-go, not a constrained planner. Release is supported
only for parts the constraint record says are attached to their support.
Which of those the operator declares as capabilities is an explicit runtime
input, never a default.
"""
from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
import math
import time
import uuid

from .contracts import ContractError, SkillRegistry, checked_copy, digest, validate_schema
from .handlers import (BackendFailure, FollowConstraintHandler, GraspHandler, MoveToPoseHandler,
                       ObserveHandler, ReleaseHandler, SetGripperHandler, SkillOutcome)
from .hardware_backend import HardwareObservationBackend
from .motion.sheppy_client import (JOINTS, KNUCKLE_CLOSED_RAD, TOOL_FRAME_FROM_FLANGE_M, SheppyClientError,
                                   reversed_trajectory, scale_trajectory_time)
from .validation import GeometryCheck


MODE = "sheppy_client"
#: What this client genuinely provides. Declaring more is the operator's act.
PROVIDED_CAPABILITIES = frozenset({"local_geometry", "cloud_grounding", "curobo_transit",
                                   "trajectory_execution", "local_retention_monitor"})
#: Provided additionally when a depth-backed guard is wired.
GUARDED_CAPABILITIES = frozenset({"live_collision_guard"})
#: Provided whenever a guard factory exists: the effort guard watches contact.
EFFORT_CAPABILITIES = frozenset({"contact_monitor"})
#: Capabilities the catalog names that this client is known NOT to provide.
KNOWN_GAPS = {
    "curobo_online_replanning": "the planner returns one static trajectory per request",
    "continuous_trajectory_handoff": "the driver executes whole trajectories; no suffix replacement",
    "live_collision_guard": "no depth-backed guard is wired; nothing watches the path during motion",
    "calibrated_aperture": "aperture is a nominal 2F-85 relation, not a measured table",
    "calibrated_grasp": "grasp success is a knuckle stall, not a calibrated grip model",
    "support_detection": "no sensor establishes that a released object is supported",
    "curobo_constrained_path": "constraints are followed as cuRobo plans from rest between waypoints, not a constrained path",
    "contact_monitor": "no effort guard is wired; contact is not monitored",
}


def assertion(predicate, args, validity="true"):
    return {"predicate": predicate, "args": args, "validity": validity}


@dataclass(frozen=True)
class NominalApertureMap:
    """Robotiq 2F-85 nominal relation: 85 mm open at knuckle 0, closed at 0.8 rad.

    Datasheet geometry, not a measurement of this gripper. error_m is the
    allowance the backend reserves for that; a measured table should replace
    this map before aperture_reached is relied on for anything tight.
    """
    open_aperture_m: float = .085
    closed_knuckle_rad: float = KNUCKLE_CLOSED_RAD
    error_m: float = .008
    nominal: bool = True
    evidence_id: str = "robotiq-2f85-datasheet-nominal"

    def to_knuckle(self, aperture_m):
        if not 0. <= aperture_m <= self.open_aperture_m:
            raise BackendFailure("safety_fault", f"aperture {aperture_m} m is outside 0..{self.open_aperture_m} m")
        return (1.-aperture_m/self.open_aperture_m)*self.closed_knuckle_rad

    def to_aperture(self, knuckle_rad):
        return max(0., (1.-knuckle_rad/self.closed_knuckle_rad)*self.open_aperture_m)


class SheppyGeometry:
    """Admission geometry for the client: profiles are checked, collision is the planner's.

    Establishes motion_profile_valid for profiles the context declares as
    physical. It does not claim reachability or collision freedom; the planner
    decides both at dispatch, against the world it holds.
    """
    simulation_only = False

    def __init__(self, backend):
        if getattr(backend, "mode", None) != MODE:
            raise ValueError("sheppy geometry requires the sheppy client backend")
        self.backend = backend

    def validate(self, node, snapshot, predicted_state, phase):
        if phase not in ("admission", "dispatch"):
            raise ContractError("invalid validation phase")
        args = node["args"]
        established = []
        if "profile_id" in args:
            profile = self.backend.profiles.get(args["profile_id"])
            if profile is None or profile.get("simulation_only", True):
                raise ContractError("the client backend requires an explicit physical profile")
            established.append(assertion("motion_profile_valid", {"profile_id": args["profile_id"]}))
        if node["skill"] == "follow_constraint":
            # Readiness for guarded contact: a metric record for this exact
            # constraint, a contact profile, and an effort guard to stop on.
            record = self.backend.constraints.get(args["constraint_id"])
            if record is None or record["entity_id"] != args["entity_id"]:
                raise ContractError("no metric constraint record is installed for this constraint")
            if record["unit"] != args["target_unit"] or not record["minimum"] <= args["target_value"] <= record["maximum"]:
                raise ContractError("constraint target disagrees with the installed record")
            if self.backend.guard_factory is None or self.backend.chain is None:
                raise ContractError("constraint following needs the effort guard and the kinematic model")
            established.append(assertion("contact_ready", {key: args[key] for key in ("entity_id", "constraint_id", "profile_id")}))
        elif node["skill"] == "release":
            # A part the record says is attached to its support is supported
            # wherever it is; a free object's support is not sensed here.
            attached = any(record["entity_id"] == args["entity_id"] and record.get("surface_entity_id") == args["support_id"]
                           for record in self.backend.constraints.values())
            if attached:
                established += [assertion("at_release_pose", {"entity_id": args["entity_id"], "support_id": args["support_id"]}),
                                assertion("support_verified", {"entity_id": args["entity_id"], "support_id": args["support_id"]})]
        identities = snapshot.identities()
        dependencies = {key: identities[key] for key in ("execution_epoch", "collision_revision",
                                                         "calibration_id", "base_epoch")}
        return GeometryCheck(end_state={"geometry": predicted_state.get("geometry")},
                             dependencies=dependencies, established_facts=tuple(established),
                             valid_for_s=5., status="validated",
                             artifact={"collision_checked_by": "rammp_curobo planner world at dispatch",
                                       "reachability_checked_by": "rammp_curobo planner at dispatch",
                                       "simulation_only": False})


class SheppyArmBackend:
    mode = MODE
    hardware_commands = True
    physical_transport = MODE
    fixture_capabilities = frozenset()

    def __init__(self, catalog, world, *, client, profiles, aperture_map=None, observer=None,
                 stationary_duration_s=.5, grasp_close_knuckle_rad=KNUCKLE_CLOSED_RAD,
                 grasp_stall_margin_rad=.05, aperture_tolerance_m=.01, guard_factory=None,
                 collision_guarded=False, chain=None, constraints=None, constraint_store=None,
                 speed_scales=None, grasp_exclusion_m=.10, tool_exclusion_m=.15, look_act=True,
                 look_act_calls=5, step_limit_m=.03, total_shift_limit_m=.08, yaw_limit_deg=15.,
                 turn_tolerance_rad=.17):
        if client is None or not all(hasattr(client, name) for name in
                                     ("live_joints", "stationary", "plan_to_pose", "execute", "gripper", "cancel")):
            raise ContractError("A SheppyArmClient-shaped client is required")
        self.catalog, self.world, self.client = catalog, world, client
        self.profiles = {p["profile_id"]: p for p in profiles}
        if any(p.get("simulation_only", True) for p in self.profiles.values()):
            raise ContractError("Client backend profiles must be explicit physical profiles")
        self.aperture = aperture_map or NominalApertureMap()
        self.observation = (HardwareObservationBackend(catalog, world, observer=observer, held_check=self.quiescent)
                            if observer is not None else None)
        self.stationary_duration_s = stationary_duration_s
        self.grasp_close_knuckle_rad = grasp_close_knuckle_rad
        self.grasp_stall_margin_rad = grasp_stall_margin_rad
        self.aperture_tolerance_m = aperture_tolerance_m
        # guard_factory() returns a fresh GuardSet for one move, or None. The
        # live_collision_guard capability is only honest when this is wired.
        if guard_factory is not None and not callable(guard_factory):
            raise ContractError("guard_factory must be callable")
        self.guard_factory = guard_factory
        # An effort-only guard stops on contact; only a depth-backed guard may
        # be called a live collision guard, and the composer says which it built.
        if collision_guarded and guard_factory is None:
            raise ContractError("collision_guarded requires a guard factory")
        self.collision_guarded = bool(collision_guarded)
        # Metric constraint records the intake installed, by constraint id;
        # the kinematic model places the tool; the store keeps attempt history.
        self.chain, self.constraints, self.constraint_store = chain, dict(constraints or {}), constraint_store
        scales = {"transit": 1., "contact": 1., **(speed_scales or {})}
        if any(isinstance(v, bool) or type(v) not in (int, float) or not v >= 1. for v in scales.values()):
            raise ContractError("speed scales are slow-down factors of at least 1")
        self.speed_scales = scales
        self.grasp_exclusion_m, self.tool_exclusion_m = float(grasp_exclusion_m), float(tool_exclusion_m)
        # The scene, when the observer is one: keyframes, crops and surface
        # normals during a skill, and the reasoner bound to it for look-act.
        self.scene = observer if all(hasattr(observer, name) for name in ("wait_for_keyframe", "crop_for", "surface_normal")) else None
        self.look_act, self.look_act_calls = bool(look_act), int(look_act_calls)
        self.step_limit_m, self.total_shift_limit_m = float(step_limit_m), float(total_shift_limit_m)
        self.yaw_limit_deg, self.turn_tolerance_rad = float(yaw_limit_deg), float(turn_tolerance_rad)
        self.grasp_knuckle = None
        self.active = set()
        self.holding_id = None
        self.current_pose = None
        self.log = lambda message: None                 # the node replaces this with its logger
        self.record_root = None                         # artifacts/bench: guard trips are saved for offline replay
        self.at_contact = False                         # the tool is parked where it touches something on purpose
        # The surface the grasped part stands on (a door face under its pull, a table under a cup), kept in the
        # tool frame from the moment of contact so it stays right while the part moves with the hand.
        self.contact_support = None
        self.surface_disk_m, self.surface_protrusion_m = .14, .02
        self.align_done_m, self.align_done_deg = .008, 3.
        # The close view at the standoff re-measures the surface and the part before the final approach;
        # a correction beyond these bounds means the close view and discovery disagree too much to approach.
        self.refine_radius_m, self.refine_max_shift_m, self.refine_max_tilt_deg = .15, .06, 12.
        self.refine_max_yaw_deg, self.part_search_m = 30., .06
        self.last_trajectory = None                     # the last path flown to completion: the way out is the way in
        self.events = []

    # -- registration --------------------------------------------------------
    def handlers(self):
        handlers = {"move_to_pose": MoveToPoseHandler(self), "set_gripper": SetGripperHandler(self),
                    "grasp": GraspHandler(self), "release": ReleaseHandler(self),
                    "follow_constraint": FollowConstraintHandler(self)}
        if self.observation is not None:
            handlers["observe"] = ObserveHandler(self)
        return handlers

    @property
    def provided_capabilities(self):
        provided = set(PROVIDED_CAPABILITIES)
        if self.collision_guarded:
            provided |= GUARDED_CAPABILITIES
        if self.guard_factory is not None:
            provided |= EFFORT_CAPABILITIES
        return frozenset(provided)

    def registry(self, *, capabilities, commissioned, mode="hardware"):
        """The operator's declared capabilities; known gaps must be acknowledged by name."""
        declared = frozenset(capabilities)
        unacknowledged = sorted((declared & set(KNOWN_GAPS)) - self.provided_capabilities)
        self.declared_gaps = {name: KNOWN_GAPS[name] for name in unacknowledged}
        return SkillRegistry(self.catalog, self.handlers(), declared, mode=mode, commissioned=commissioned)

    # -- hold / quiescence -----------------------------------------------------
    async def quiescent(self):
        # Held means nothing in flight and the arm measured still. A skill that
        # owns the arm while it waits on the model is still a held arm: a stop
        # during that wait must verify, not latch a fault on bookkeeping.
        if getattr(self.client, "in_flight", False):
            return False
        try:
            # The client tracks the still window continuously, so held state
            # is answered at once: still for at least the dwell, or not held.
            # The status tick and the cloud guard allow tens of milliseconds;
            # a dwell here would trip them. A client without the window dwells.
            still_since = getattr(self.client, "still_since_s", None)
            if still_since is None:
                return bool(await self.client.stationary(duration_s=self.stationary_duration_s))
            since = still_since()
            return since is not None and time.monotonic()-since >= self.stationary_duration_s
        except Exception:                                 # noqa: BLE001 - unverified is not held
            return False

    def still_now(self):
        """The arm has been measured still for the dwell, whoever owns it; synchronous, never waits."""
        still_since = getattr(self.client, "still_since_s", None)
        if still_since is None:
            return False
        since = still_since()
        return since is not None and time.monotonic()-since >= self.stationary_duration_s

    def quiescent_now(self):
        """Held state read synchronously from the still window: safe from any thread, never waits."""
        return not getattr(self.client, "in_flight", False) and self.still_now()

    quiescence_grace_s = 1.5                             # under the supervisor's 2 s bound on this query

    async def skill_quiescent(self, skill_id):
        """Whether this skill's commands are done and the arm is at rest; a moving skill gets a moment to get there.

        The client's execute() confirms rest by sampling every 20 ms; the still window here restarts on any
        single joint-state message over the threshold, and near a contact the joints can ring for a moment
        after arriving. So a motion skill is given quiescence_grace_s to come to rest before it is called
        not quiescent, and when it does not, the log says what the joints were doing.
        """
        if skill_id in self.active:
            return False
        if skill_id not in ("move_to_pose", "grasp", "release", "set_gripper", "follow_constraint"):
            return True
        deadline = time.monotonic()+self.quiescence_grace_s
        while True:
            if await self.quiescent():
                return True
            if time.monotonic() >= deadline:
                self.log(f"{skill_id}: not at rest {self.quiescence_grace_s:.1f} s after its commands finished: {self.quiescence_report()}")
                return False
            await asyncio.sleep(.05)

    def quiescence_report(self):
        live = self.client.live_joints() if hasattr(self.client, "live_joints") else None
        still_since = getattr(self.client, "still_since_s", lambda: None)()
        velocity = None if live is None or live.get("velocity_rad_s") is None else max(abs(v) for v in live["velocity_rad_s"])
        return {"in_flight": bool(getattr(self.client, "in_flight", False)), "joint_state_fresh": live is not None,
                "still_for_s": None if still_since is None else round(time.monotonic()-still_since, 3),
                "max_joint_velocity_rad_s": velocity}

    async def stop_skill(self, skill_id, reason):
        self.events.append({"event": "stop", "skill": skill_id, "reason": reason, "at": time.monotonic()})
        if skill_id == "observe" and self.observation is not None:
            await self.observation.stop_skill(skill_id, reason)
            return
        # Cancelling an in-flight trajectory makes the driver stop and hold its
        # last reference. A gripper setpoint is latched and cannot be recalled.
        await self.client.cancel()

    async def stop(self, reason="supervisor"):
        for skill in list(self.active) or ["move_to_pose"]:
            await self.stop_skill(skill, reason)

    # -- helpers ---------------------------------------------------------------
    def _profile(self, profile_id, *, safety_class=None):
        profile = self.profiles.get(profile_id)
        if profile is None:
            raise BackendFailure("safety_fault", f"unknown physical profile {profile_id}")
        if safety_class is not None and profile.get("safety_class") != safety_class:
            raise BackendFailure("safety_fault", f"profile {profile_id} is not a {safety_class} profile")
        return profile

    def _snapshot(self, context):
        snapshot = context.snapshot if context.snapshot is not None else self.world.snapshot()
        if context.cancel_event.is_set() or context.execution_epoch != snapshot.execution_epoch:
            raise BackendFailure("cancelled", "execution epoch was revoked")
        return snapshot

    async def _own(self, skill, context):
        if skill in self.active:
            raise BackendFailure("safety_fault", "duplicate backend resource owner")
        if not self.client.motion_enabled:
            raise BackendFailure("safety_fault", "motion is not armed on the client; nothing is sent")
        self.active.add(skill)
        self.events.append({"event": "started", "skill": skill, "node_id": context.node_id, "at": time.monotonic()})

    def _release(self, skill):
        self.active.discard(skill)

    def _outcome(self, facts, *, data):
        evidence_id = f"sheppy-{uuid.uuid4().hex}"
        evidence = {"evidence_id": evidence_id, "source": self.mode, "predicates": checked_copy(facts),
                    "ttl_s": 30., "data": {"simulation_only": False, "physical_contact_state_measured": False,
                                          **checked_copy(data)}}
        return SkillOutcome("succeeded", {"evidence_id": evidence_id}, [evidence],
                            [{**fact, "evidence_id": evidence_id} for fact in facts])

    def _guard(self, **options):
        """A fresh guard for one move; factories without options still work."""
        if self.guard_factory is None:
            return None
        try:
            return self.guard_factory(**options)
        except TypeError:
            return self.guard_factory()

    def _scaled(self, trajectory, safety_class):
        return scale_trajectory_time(trajectory, self.speed_scales.get(safety_class, 1.))

    def _tool_pose(self, positions):
        """The planner's tool_frame for measured joints: (position, quaternion)."""
        if self.chain is None:
            raise BackendFailure("safety_fault", "no kinematic model is wired; the tool cannot be placed")
        from .motion.kinematics import quaternion_xyzw_from_matrix
        import numpy as np
        ee = self.chain.base_from_link(dict(zip(JOINTS, positions)), "end_effector_link")
        tool = ee @ np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., TOOL_FRAME_FROM_FLANGE_M], [0., 0., 0., 1.]])
        return tuple(float(v) for v in tool[:3, 3]), tuple(float(v) for v in quaternion_xyzw_from_matrix(tool[:3, :3]))

    async def _step_to(self, position, orientation, context, *, safety_class, exclusions=(), tool_exclusion_m=0., touch_nm=None,
                       announce=None):
        """One planned, gated, guarded move to a tool pose; the receipt or a failure."""
        if context.cancel_event.is_set():
            raise BackendFailure("cancelled", "execution epoch was revoked")
        if not await self.client.stationary(duration_s=self.stationary_duration_s):
            raise BackendFailure("stale_state", "the arm is not verifiably still before planning")
        exclusions = list(exclusions)
        if self.at_contact:
            # Leaving a place the tool touched on purpose: what it touched is
            # still beside the fingers, and is exempt for this move only.
            live = self.client.live_joints()
            if live is not None and self.chain is not None:
                tool_position, tool_orientation = self._tool_pose(live["position_rad"])
                exclusions.append((tool_position, self.grasp_exclusion_m))
                if self.contact_support is not None:
                    point, normal = self._support_in_base(tool_position, tool_orientation)
                    exclusions.append(self._surface_exclusion(point, normal, tool_position))
        try:
            trajectory, planning = await self.client.plan_to_pose(position, orientation, cancel_event=context.cancel_event)
        except SheppyClientError as exc:
            # The planner refuses a start it judges to be in collision
            # (cuRobo INVALID_START_STATE through RAMMP-CuRobo's "STATUS: error" text).
            # TODO: confirm against planner that the status name reaches the message unchanged.
            if "START" not in str(exc).upper() or not await self._back_out(context, exclusions):
                raise BackendFailure("planning_failed", str(exc)) from exc
            try:
                trajectory, planning = await self.client.plan_to_pose(position, orientation, cancel_event=context.cancel_event)
            except SheppyClientError as again:
                raise BackendFailure("planning_failed", f"after backing out: {again}") from again
        trajectory = self._scaled(trajectory, safety_class)
        options = {"exclusions": exclusions, "tool_exclusion_m": tool_exclusion_m}
        if touch_nm is not None:
            options["touch_nm"] = touch_nm
        guard = self._guard(**options)
        if announce is not None:
            await context.feedback(event="motion_started", skill=announce, duration_s=trajectory.duration_s,
                                   provenance=trajectory.provenance)
        receipt = await self.client.execute(trajectory, cancel_event=context.cancel_event, guard=guard)
        self.log(f"{context.node_id}: {safety_class} move {trajectory.duration_s:.1f} s, {len(trajectory.points)} points: "
                 f"{receipt['status']} {receipt['message'][:120]}")
        if receipt["status"] != "succeeded":
            # A cancelled goal leaves the arm decelerating. The failure is reported from a still arm:
            # returning sooner reads as a handler that left its own motion running, which latches a fault.
            settle = getattr(self.client, "settle", None)
            if settle is not None:
                await settle(timeout_s=3.)
        if receipt["status"] == "cancelled":
            raise BackendFailure("cancelled", receipt["message"])
        if receipt["status"] == "guard_trip":
            trip = receipt.get("trip") or {}
            if self.record_root is not None and trip.get("kind") == "collision" and guard is not None:
                from .perception.scene_record import save_guard_trip
                live = self.client.live_joints()
                reader = getattr(guard, "depth_reader", None)
                save_guard_trip(self.record_root, depth_frame=reader() if callable(reader) else None,
                                joints_rad=(live or {}).get("position_rad") or trajectory.points[0].state.position,
                                trajectory=trajectory, elapsed_s=float(receipt.get("progress") or 0.)*trajectory.duration_s,
                                exclusions=getattr(guard, "exclusions", ()), tool_exclusion_m=getattr(guard, "tool_exclusion_m", 0.),
                                trip=checked_copy(trip), context={"node_id": context.node_id, "task_id": context.task_id,
                                                                  "safety_class": safety_class,
                                                                  "target": {"position_m": list(position), "orientation_xyzw": list(orientation)}})
            # An obstacle on the remaining path means the collision evidence
            # the plan was admitted against is stale: stop, then replan.
            # Contact, a blind camera or lost state is a supervisor fault.
            code = "stale_state" if trip.get("kind") == "collision" else "safety_fault"
            raise BackendFailure(code, receipt["message"], evidence=[{
                "evidence_id": f"guard-{uuid.uuid4().hex}", "source": self.mode, "predicates": [],
                "ttl_s": 30., "data": {"trip": checked_copy(trip), "receipt": checked_copy(receipt)}}])
        if receipt["status"] != "succeeded":
            raise BackendFailure("safety_fault" if receipt["status"] in ("goal_not_reached", "timeout") else "planning_failed",
                                 receipt["message"])
        self.last_trajectory = trajectory
        return trajectory, planning, receipt, guard

    def _surface_exclusion(self, point, normal, target):
        """A ball that holds the measured surface under the target and at most surface_protrusion_m in front of it.

        The guard's finger spheres are coarse (3.5 cm, centred near the tool frame) and its margin is 3 cm,
        so a hand whose fingertips stop a centimetre off a door face reads the face as an obstacle. This exempts only that face: a
        disk of surface_disk_m around the target's foot, and nothing standing proud of it by more than
        the protrusion. Everything beyond, and anything on the face taller than that, is still checked.
        """
        import numpy as np
        point, normal, target = (np.asarray(v, dtype=float) for v in (point, normal, target))
        normal = normal/np.linalg.norm(normal)
        foot = target-normal*float((target-point) @ normal)
        rho, delta = self.surface_disk_m, self.surface_protrusion_m
        behind = (rho*rho-delta*delta)/(2.*delta)
        return tuple(float(v) for v in foot-normal*behind), float(behind+delta)

    async def _refine_contact(self, entity_id, position, orientation, *, centre=True):
        """From the standoff, measure the surface the part stands on and the part itself, and place the grasp from that.

        Discovery measures from 40 cm or more, where a small mount or kinematic error moves a door face by a
        centimetre or two and tilts it by degrees, and a model judging a wrist image cannot tell where along a
        pull the fingers will close. So before the final approach the depth of the close view decides: the
        surface is refitted around the target; the part is the connected cluster standing proud of it nearest
        the target, and (with centre) the grasp moves onto the middle of that cluster with the fingers closing
        across its long axis; the tool is squared to the surface at the fingertips' closed reach plus clearance
        off it. Returns (position, orientation, support, record); refuses when the close view and discovery
        disagree beyond the bounds.
        """
        import numpy as np
        from .motion.kinematics import quaternion_matrix, quaternion_xyzw_from_matrix
        from .perception.object_geometry import FINGERTIP_REACH_M, SURFACE_CLEARANCE_M, points_in_region, support_plane, to_base
        support = self._entity_support(entity_id)
        record = {"refined": False, "centred_on_part": False, "skipped": None}
        rotation = quaternion_matrix(tuple(orientation))
        if support is None or self.scene is None:
            record["skipped"] = "no measured surface under this part"
            return position, orientation, support, record
        discovered = np.asarray(support[1], dtype=float)/np.linalg.norm(support[1])
        if float(rotation[:, 2] @ -discovered) < math.cos(math.radians(30.)):
            record["skipped"] = "this grasp does not approach the surface the part stands on"
            return position, orientation, support, record
        try:
            keyframe = await self.scene.wait_for_keyframe(timeout_s=2.)
        except Exception as exc:                            # noqa: BLE001 - the discovery surface stands, and the guard with it
            record["skipped"] = f"no still close view: {exc}"
            return position, orientation, support, record
        target = np.asarray(position, dtype=float)

        def measure():
            points = to_base(points_in_region(keyframe, (0, 0, keyframe.width, keyframe.height), stride=2, max_range_m=.6),
                             keyframe.base_from_camera)
            near = points[np.linalg.norm(points-target, axis=1) < self.refine_radius_m]
            plane = support_plane(near, camera_position=keyframe.camera_position_base, min_points=100)
            if plane is None:
                return None
            origin, normal = np.asarray(plane["origin_m"]), np.asarray(plane["normal"])/np.linalg.norm(plane["normal"])
            along = (points-target) @ normal
            lateral = np.linalg.norm((points-target)-np.outer(along, normal), axis=1)
            elevation = (points-origin) @ normal
            candidates = (lateral < self.refine_radius_m) & (elevation > .008) & (elevation < .10)
            cluster = _nearest_cluster(points[candidates], normal, target, cell_m=.01, seed_within_m=self.part_search_m)
            if cluster is None or len(cluster) < 20:
                return origin, normal, float(plane["rms_m"]), None, 0, None
            height = float(np.percentile((cluster-origin) @ normal, 95))
            flat = cluster-np.outer((cluster-origin) @ normal, normal)
            middle = flat.mean(axis=0)
            spread = np.cov((flat-middle).T)
            values, vectors = np.linalg.eigh(spread)
            major = vectors[:, int(np.argmax(values))]
            major = major-(major @ normal)*normal
            major /= np.linalg.norm(major)
            ordered = np.sort(values)[::-1]
            elongation = float(np.sqrt(ordered[0]/max(ordered[1], 1e-12)))
            return origin, normal, float(plane["rms_m"]), height, int(len(cluster)), (middle, major, elongation)
        measured = await asyncio.to_thread(measure)
        if measured is None:
            record["skipped"] = "no surface around the target in the close view"
            return position, orientation, support, record
        origin, normal, rms, height, count, part = measured
        tilt = math.degrees(math.acos(max(-1., min(1., float(normal @ discovered)))))
        z = -normal
        old_x = rotation[:, 0]-(rotation[:, 0] @ normal)*normal
        old_x /= np.linalg.norm(old_x)
        if centre and part is not None:
            foot, major, elongation = part
            across = np.cross(normal, major) if elongation >= 1.8 else old_x
            across = across/np.linalg.norm(across)
            if across @ old_x < 0:
                across = -across
            record.update(centred_on_part=True, part_elongation=round(elongation, 2))
        else:
            foot, across = target-normal*float((target-origin) @ normal), old_x
        yaw = math.degrees(math.acos(max(-1., min(1., float(abs(across @ old_x))))))
        elevation = max((height or 0.)-min(.03, (height or 0.)/2.), FINGERTIP_REACH_M+SURFACE_CLEARANCE_M)
        refined = foot+normal*elevation
        shift = float(np.linalg.norm(refined-target))
        record.update(tilt_deg=round(tilt, 2), yaw_deg=round(yaw, 2), shift_m=round(shift, 4),
                      part_height_m=None if height is None else round(height, 4), part_points=count, surface_rms_m=round(rms, 4),
                      surface_point_m=[float(v) for v in origin], surface_normal=[float(v) for v in normal],
                      fingertip_clearance_m=round(elevation-FINGERTIP_REACH_M, 4))
        if tilt > self.refine_max_tilt_deg or shift > self.refine_max_shift_m or yaw > self.refine_max_yaw_deg:
            raise BackendFailure("target_changed", f"the part seen from the standoff is {tilt:.1f} degrees, {yaw:.1f} degrees about the "
                                                   f"approach and {shift*100:.1f} cm from where discovery put it; not approaching on either")
        across = across-(across @ z)*z
        across /= np.linalg.norm(across)
        record["refined"] = True
        return (tuple(float(v) for v in refined),
                tuple(float(v) for v in quaternion_xyzw_from_matrix(np.column_stack([across, np.cross(z, across), z]))),
                ([float(v) for v in origin], [float(v) for v in normal]), record)

    async def _line_up_standoff(self, position, orientation, context, *, minimum_back_m=.06):
        """Move the tool onto the grasp's approach line, so the last centimetres go straight in."""
        import numpy as np
        from .motion.kinematics import quaternion_matrix
        live = self.client.live_joints()
        if live is None or self.chain is None:
            return None
        tool_position, tool_orientation = self._tool_pose(live["position_rad"])
        rotation = quaternion_matrix(tuple(orientation))
        back = max(float((np.asarray(position)-np.asarray(tool_position)) @ rotation[:, 2]), minimum_back_m)
        standoff = np.asarray(position)-rotation[:, 2]*back
        turn = float(np.degrees(np.arccos(max(-1., min(1., (np.trace(quaternion_matrix(tuple(tool_orientation)).T @ rotation)-1.)/2.)))))
        if float(np.linalg.norm(standoff-np.asarray(tool_position))) < .005 and turn < 2.:
            return None
        await self._step_to(tuple(float(v) for v in standoff), tuple(orientation), context, safety_class="contact",
                            exclusions=[(tuple(float(v) for v in position), self.grasp_exclusion_m)])
        return {"standoff_m": [float(v) for v in standoff], "turn_deg": round(turn, 2)}

    def _entity_support(self, entity_id):
        records = getattr(self.scene, "entities", None) or {}
        support = ((records.get(entity_id) or {}).get("grasp") or {}).get("support")
        return (support["point_m"], support["normal"]) if support else None

    def _support_in_base(self, tool_position, tool_orientation):
        import numpy as np
        from .motion.kinematics import quaternion_matrix
        rotation = quaternion_matrix(tuple(tool_orientation))
        point_tool, normal_tool = self.contact_support
        return rotation @ point_tool+np.asarray(tool_position), rotation @ normal_tool

    def _remember_support(self, support):
        """Store the contact's surface in the tool frame, from where the tool measurably is."""
        import numpy as np
        from .motion.kinematics import quaternion_matrix
        live = self.client.live_joints()
        if support is None or live is None or self.chain is None:
            self.contact_support = None
            return
        position, orientation = self._tool_pose(live["position_rad"])
        rotation = quaternion_matrix(tuple(orientation))
        self.contact_support = (rotation.T @ (np.asarray(support[0], dtype=float)-np.asarray(position)),
                                rotation.T @ np.asarray(support[1], dtype=float))

    async def _back_out(self, context, exclusions=()):
        """Retrace the last completed path, empty-handed, from where it ended; True if the arm is now back at its start."""
        path = self.last_trajectory
        if path is None or self.holding_id is not None:
            return False
        self.last_trajectory = None                        # one way out per way in
        self.events.append({"event": "back_out", "node_id": context.node_id, "at": time.monotonic()})
        receipt = await self.client.execute(reversed_trajectory(path), cancel_event=context.cancel_event,
                                            guard=self._guard(exclusions=list(exclusions)))
        if receipt["status"] == "succeeded":
            self.current_pose, self.at_contact, self.contact_support = None, False, None
        return receipt["status"] == "succeeded"

    LOOK_HINTS = ("left", "right", "up", "down", "back", "closer")

    def look_target(self, hint, positions, *, pan_rad=.5, tilt_rad=.4, step_m=.10):
        """The tool pose that turns the wrist camera the way a search hint asks, from measured joints."""
        import numpy as np
        from .constraints import rotation_about
        from .motion.kinematics import quaternion_matrix, quaternion_xyzw_from_matrix
        if hint not in self.LOOK_HINTS:
            raise BackendFailure("planning_failed", f"unknown look hint {hint!r}")
        position, orientation = self._tool_pose(positions)
        position, rotation = np.asarray(position, dtype=float), quaternion_matrix(orientation)
        view = rotation[:, 2]                                   # the camera looks along the tool z axis
        up = np.array([0., 0., 1.])
        if hint in ("left", "right"):
            turn = rotation_about(up, pan_rad if hint == "left" else -pan_rad)
            rotation = turn @ rotation
        elif hint in ("up", "down"):
            left = np.cross(up, view)
            if np.linalg.norm(left) < 1e-6:
                left = rotation[:, 0]
            left /= np.linalg.norm(left)
            angle = tilt_rad
            tilted = rotation_about(left, angle) @ view
            if (tilted[2] > view[2]) != (hint == "up"):
                angle = -angle
            rotation = rotation_about(left, angle) @ rotation
        else:
            position = position-view*step_m if hint == "back" else position+view*step_m
        return tuple(float(v) for v in position), tuple(float(v) for v in quaternion_xyzw_from_matrix(rotation))

    async def look(self, hint, context, **options):
        """Turn the wrist camera as a search hint asks: one planned, guarded transit."""
        live = self.client.live_joints()
        if live is None:
            raise BackendFailure("stale_state", "no fresh joint state to place the tool")
        position, orientation = self.look_target(hint, live["position_rad"], **options)
        await self._own("look", context)
        try:
            trajectory, planning, receipt, _ = await self._step_to(position, orientation, context, safety_class="transit")
            self.current_pose, self.at_contact, self.contact_support = None, False, None
            return {"hint": hint, "position_m": list(position), "orientation_xyzw": list(orientation),
                    "duration_s": trajectory.duration_s, "planning": planning, "receipt_status": receipt["status"]}
        finally:
            self._release("look")

    async def _look_act(self, entity_id, pose, context):
        """Align the grasp target from the standoff: look, shift a little, look again.

        The model sees the wrist image and proposes a small camera-frame shift
        or says DONE; each shift moves the standoff pose by the same amount so
        the next look shows its effect, and the grasp target moves with it.
        Bounded per step and in total; the model can only nudge, never aim.
        """
        import numpy as np
        from .constraints import rotation_about
        from .motion.kinematics import quaternion_matrix, quaternion_xyzw_from_matrix
        record = {"calls": 0, "steps": [], "converged": None, "skipped": None, "shift_m": [0., 0., 0.], "yaw_rad": 0.}
        scene = self.scene
        reasoner = getattr(scene, "reasoner", None)
        if not self.look_act or scene is None or reasoner is None:
            record["skipped"] = "no scene or reasoner bound"
            return tuple(pose.position_m), tuple(pose.orientation_xyzw), record
        snapshot = self._snapshot(context)
        label = next((e["label"] for e in snapshot.context["entities"] if e["entity_id"] == entity_id), entity_id)
        position = np.asarray(pose.position_m, dtype=float)
        rotation = quaternion_matrix(tuple(pose.orientation_xyzw))
        total, yaw_total = np.zeros(3), 0.
        yaw_limit = math.radians(self.yaw_limit_deg)
        for call in range(self.look_act_calls):
            try:
                keyframe = await scene.wait_for_keyframe(timeout_s=2.)
                crop = await asyncio.to_thread(scene.crop_for, keyframe)
            except Exception as exc:                        # noqa: BLE001 - alignment is an aid, not a gate
                record["skipped"] = f"no usable keyframe: {exc}"
                break
            live = self.client.live_joints()
            if live is None:
                raise BackendFailure("stale_state", "no fresh joint state during alignment")
            tool_position, tool_orientation = self._tool_pose(live["position_rad"])
            aperture = None if live["knuckle_rad"] is None else self.aperture.to_aperture(live["knuckle_rad"])
            state = {"tool_position_m": list(tool_position), "target_position_m": position.tolist(),
                     "distance_to_target_m": float(np.linalg.norm(position-np.asarray(tool_position))),
                     "gripper_aperture_m": aperture, "shift_so_far_m": total.tolist(), "yaw_so_far_deg": math.degrees(yaw_total),
                     "calls_remaining": self.look_act_calls-call}
            result = await reasoner.correct_pose(snapshot.context, entity_id, label=label, state=state, images=[crop],
                                                 step_limit_m=self.step_limit_m, yaw_limit_deg=self.yaw_limit_deg)
            record["calls"] += 1
            self.log(f"align {entity_id} look {record['calls']}/{self.look_act_calls}: {result.status} "
                     f"{json.dumps(result.proposal) if result.proposal else ''} {result.detail[:120]}")
            if result.status == "ABORT":
                raise BackendFailure("target_changed", "the model aborted the approach: "+result.detail)
            if result.status != "OK" or result.proposal is None:
                record["skipped"] = f"{result.status}: {result.detail}"
                break
            if result.proposal["status"] == "DONE":
                record["converged"] = True
                record["steps"].append({"keyframe": keyframe.capture_id, "status": "DONE", "rationale": result.detail[:200]})
                break
            delta_camera = np.clip(np.asarray(result.proposal["delta_camera_m"], dtype=float), -self.step_limit_m, self.step_limit_m)
            if float(np.linalg.norm(delta_camera)) < self.align_done_m and abs(float(result.proposal["yaw_deg"])) < self.align_done_deg:
                # A nudge smaller than the grasp can resolve is a model saying "centred" in other words;
                # chasing it only spends the task's request budget.
                record["converged"] = True
                record["steps"].append({"keyframe": keyframe.capture_id, "status": "DONE within tolerance",
                                        "delta_camera_m": delta_camera.tolist(), "rationale": result.detail[:200]})
                break
            delta = keyframe.base_from_camera[:3, :3] @ delta_camera
            room = self.total_shift_limit_m-float(np.linalg.norm(total))
            if np.linalg.norm(delta) > room:
                delta = delta*(max(0., room)/float(np.linalg.norm(delta))) if np.linalg.norm(delta) > 0 else delta
            yaw = math.radians(float(np.clip(result.proposal["yaw_deg"], -self.yaw_limit_deg, self.yaw_limit_deg)))
            yaw = float(np.clip(yaw_total+yaw, -yaw_limit, yaw_limit))-yaw_total
            total, yaw_total = total+delta, yaw_total+yaw
            position = position+delta
            rotation = rotation @ rotation_about([0., 0., 1.], yaw)
            record["steps"].append({"keyframe": keyframe.capture_id, "status": "MOVE", "delta_camera_m": delta_camera.tolist(),
                                    "delta_base_m": [float(v) for v in delta], "yaw_rad": yaw, "rationale": result.detail[:200]})
            # Show the model its shift: the standoff pose moves by the same amount.
            standoff = np.asarray(tool_position)+delta
            standoff_rotation = quaternion_matrix(tool_orientation) @ rotation_about([0., 0., 1.], yaw)
            await self._step_to(tuple(float(v) for v in standoff), tuple(quaternion_xyzw_from_matrix(standoff_rotation)), context,
                                safety_class="contact", exclusions=[(tuple(float(v) for v in position), self.grasp_exclusion_m)])
        else:
            record["converged"] = False
            raise BackendFailure("target_changed", "alignment did not converge within the look budget")
        record["shift_m"], record["yaw_rad"] = [float(v) for v in total], float(yaw_total)
        return tuple(float(v) for v in position), tuple(float(v) for v in quaternion_xyzw_from_matrix(rotation)), record

    def _metric_pose(self, snapshot, entity_id, role):
        pose = snapshot.metric_poses.get((entity_id, role))
        if pose is None:
            raise BackendFailure("stale_state", f"no measured {role} pose for {entity_id} in the snapshot")
        now = self.world.clock()
        if not pose.captured_at <= now < pose.captured_at+pose.valid_for_s:
            raise BackendFailure("stale_state", f"the {role} pose of {entity_id} has expired")
        if pose.frame_id != self.catalog.library["frames"]["planning"]:
            raise BackendFailure("geometry_invalid", "target pose is not in the planning frame")
        return pose

    # -- skills ----------------------------------------------------------------
    async def observe(self, args, context):
        if self.observation is None:
            raise BackendFailure("no_detection", "no local observer is configured for this deployment")
        return await self.observation.observe(args, context)

    async def move_to_pose(self, args, context):
        validate_schema(args, self.catalog.skills["move_to_pose"]["arguments"], "move_to_pose args")
        profile = self._profile(args["profile_id"], safety_class="transit")
        snapshot = self._snapshot(context)
        target = args["target"]
        pose = self._metric_pose(snapshot, target["entity_id"], target["pose_role"])
        await self._own("move_to_pose", context)
        try:
            target_position, target_orientation, alignment = tuple(pose.position_m), tuple(pose.orientation_xyzw), None
            support = None
            if target["pose_role"] == "grasp":
                # The close view at the standoff places the grasp when it can find the part; the model aligns
                # only when it cannot, and the close view then still sets the depth.
                target_position, target_orientation, support, refine = await self._refine_contact(
                    target["entity_id"], target_position, target_orientation, centre=True)
                if refine["centred_on_part"]:
                    alignment = {"calls": 0, "steps": [], "converged": True, "skipped": "placed by the close view",
                                 "shift_m": [0., 0., 0.], "yaw_rad": 0., "refine": refine,
                                 "line_up": await self._line_up_standoff(target_position, target_orientation, context)}
                else:
                    target_position, target_orientation, alignment = await self._look_act(target["entity_id"], pose, context)
                    target_position, target_orientation, support, refine = await self._refine_contact(
                        target["entity_id"], target_position, target_orientation, centre=False)
                    alignment["refine"] = refine
            # Arriving at the grasp role means touching the target: depth
            # points around it are the intended contact, not an obstacle.
            exclusions = []
            if target["pose_role"] == "grasp":
                exclusions.append((tuple(target_position), self.grasp_exclusion_m))
                if support is not None:
                    exclusions.append(self._surface_exclusion(*support, target_position))
            trajectory, planning, receipt, _ = await self._step_to(target_position, target_orientation, context,
                                                                   safety_class="contact" if target["pose_role"] == "grasp" else "transit",
                                                                   exclusions=exclusions, announce="move_to_pose")
            previous, self.current_pose = self.current_pose, (target["entity_id"], target["pose_role"])
            self.at_contact = target["pose_role"] == "grasp"
            if self.at_contact:
                self._remember_support(support)
            else:
                self.contact_support = None
            facts = [assertion("at_pose", {"entity_id": target["entity_id"], "pose_role": target["pose_role"]})]
            if previous and previous != self.current_pose:
                facts.append(assertion("at_pose", {"entity_id": previous[0], "pose_role": previous[1]}, "false"))
                if previous[1] == "grasp":
                    facts.append(assertion("at_grasp_pose", {"entity_id": previous[0]}, "false"))
            if target["pose_role"] == "grasp":
                facts.append(assertion("at_grasp_pose", {"entity_id": target["entity_id"]}))
            if target["pose_role"] == "placement" and self.holding_id:
                # Arrival at the placement pose is measured; support is not.
                facts.append(assertion("at_release_pose", {"entity_id": self.holding_id,
                                                           "support_id": target["entity_id"]}))
            return self._outcome(facts, data={
                "target": {"entity_id": target["entity_id"], "pose_role": target["pose_role"],
                           "position_m": list(pose.position_m), "orientation_xyzw": list(pose.orientation_xyzw),
                           "evidence_id": pose.evidence_id},
                "commanded": {"position_m": list(target_position), "orientation_xyzw": list(target_orientation)},
                "alignment": alignment,
                "profile_id": profile["profile_id"], "planning": planning,
                "trajectory": {"digest": trajectory.digest, "duration_s": trajectory.duration_s,
                               "points": len(trajectory.points), "provenance": trajectory.provenance},
                "receipt": receipt, "support_verified": False})
        finally:
            self._release("move_to_pose")

    async def set_gripper(self, args, context):
        validate_schema(args, self.catalog.skills["set_gripper"]["arguments"], "set_gripper args")
        self._profile(args["profile_id"], safety_class="gripper")
        self._snapshot(context)
        if self.holding_id is not None:
            raise BackendFailure("safety_fault", "empty-gripper preshape cannot change a retained grip")
        knuckle = self.aperture.to_knuckle(args["aperture_m"])
        await self._own("set_gripper", context)
        try:
            outcome = await self.client.gripper(knuckle)
            if not outcome["ok"] or outcome["stalled"]:
                raise BackendFailure("safety_fault", "gripper did not reach the aperture: "+outcome["message"])
            measured = self.aperture.to_aperture(outcome["knuckle_rad"])
            if abs(measured-args["aperture_m"]) > self.aperture_tolerance_m:
                raise BackendFailure("safety_fault", f"measured aperture {measured:.4f} m misses {args['aperture_m']} m")
            return self._outcome([assertion("aperture_reached", {"aperture_m": args["aperture_m"]})],
                                 data={"knuckle_rad": outcome["knuckle_rad"], "measured_aperture_m": measured,
                                       "aperture_map": {"nominal": self.aperture.nominal,
                                                        "error_m": self.aperture.error_m,
                                                        "evidence_id": self.aperture.evidence_id}})
        finally:
            self._release("set_gripper")

    async def grasp(self, args, context):
        validate_schema(args, self.catalog.skills["grasp"]["arguments"], "grasp args")
        self._profile(args["profile_id"], safety_class="gripper")
        self._snapshot(context)
        if self.holding_id or self.current_pose != (args["entity_id"], "grasp"):
            raise BackendFailure("empty_grasp", "retention requires an empty gripper at the grasp pose")
        await self._own("grasp", context)
        try:
            outcome = await self.client.gripper(self.grasp_close_knuckle_rad)
            if not outcome["ok"]:
                raise BackendFailure("empty_grasp", "the gripper did not close: "+outcome["message"])
            closed_on_nothing = (not outcome["stalled"]
                                 or outcome["knuckle_rad"] > self.grasp_close_knuckle_rad-self.grasp_stall_margin_rad)
            # Stalled almost open: the fingers are pressing on the part, not around it.
            jammed = not closed_on_nothing and outcome["knuckle_rad"] < .15*self.grasp_close_knuckle_rad
            if closed_on_nothing or jammed:
                # Reopen, so the retry starts from the open hand the plan expects.
                reopened = await self.client.gripper(0.)
                detail = ("the fingers stalled almost open: they pressed on the part instead of closing around it"
                          if jammed else "the gripper closed fully; nothing was retained")
                raise BackendFailure("empty_grasp", f"{detail} (knuckle {outcome['knuckle_rad']:.3f} rad; "
                                                    f"reopened: {bool(reopened.get('ok'))})")
            self.holding_id = args["entity_id"]
            self.grasp_knuckle = outcome["knuckle_rad"]
            facts = [assertion("holding", {"entity_id": args["entity_id"]}),
                     assertion("gripper_empty", {"robot_id": "robot"}, "false")]
            return self._outcome(facts, data={"knuckle_rad": outcome["knuckle_rad"], "stalled": True,
                                              "retention": "knuckle stall short of closed; not a calibrated grip"})
        finally:
            self._release("grasp")

    async def release(self, args, context):
        validate_schema(args, self.catalog.skills["release"]["arguments"], "release args")
        self._profile(args["profile_id"], safety_class="gripper")
        self._snapshot(context)
        if self.holding_id != args["entity_id"]:
            raise BackendFailure("release_incomplete", "item is not retained")
        await self._own("release", context)
        try:
            outcome = await self.client.gripper(0.)
            if not outcome["ok"] or outcome["stalled"]:
                raise BackendFailure("release_incomplete", "the gripper did not open: "+outcome["message"])
            self.holding_id, self.grasp_knuckle = None, None
            attached = any(record["entity_id"] == args["entity_id"] and record.get("surface_entity_id") == args["support_id"]
                           for record in self.constraints.values())
            facts = [assertion("released", {"entity_id": args["entity_id"], "support_id": args["support_id"]}),
                     assertion("holding", {"entity_id": args["entity_id"]}, "false"),
                     assertion("gripper_empty", {"robot_id": "robot"})]
            return self._outcome(facts, data={"knuckle_rad": outcome["knuckle_rad"], "support_detected": False,
                                              "attached_part": attached, "supported_by": args["support_id"]})
        finally:
            self._release("release")

    async def follow_constraint(self, args, context):
        """Follow an installed constraint record as guarded cuRobo waypoints.

        Each waypoint is one plan from rest and one execution under the
        record's effort budget, with the tool's own neighbourhood excluded
        from the depth guard because it holds the part. Success is the tool
        having traced the arc with the grip retained; the part's own angle is
        not observed, and the outcome says so. Every attempt is recorded.
        """
        from .constraints import (PROGRESS_RUBRIC, progress_score, record_attempt, save_demonstration, turn_between,
                                  waypoints)
        validate_schema(args, self.catalog.skills["follow_constraint"]["arguments"], "follow_constraint args")
        profile = self._profile(args["profile_id"], safety_class="contact")
        self._snapshot(context)
        entity_id, constraint_id = args["entity_id"], args["constraint_id"]
        if self.holding_id != entity_id:
            raise BackendFailure("slip", "the part is not retained; there is nothing to follow")
        record = self.constraints.get(constraint_id)
        if record is None or record["entity_id"] != entity_id:
            raise BackendFailure("stale_state", f"no metric constraint record is installed for {constraint_id}")
        if args["target_unit"] != record["unit"]:
            raise BackendFailure("model_mismatch", "target unit differs from the constraint record")
        target = float(args["target_value"])
        live = self.client.live_joints()
        if live is None:
            raise BackendFailure("stale_state", "no fresh joint state to place the tool")
        position, orientation = self._tool_pose(live["position_rad"])
        plan = waypoints(record, position, orientation, target)
        import numpy as np
        from .constraints import rotation_about
        from .motion.kinematics import quaternion_matrix, quaternion_xyzw_from_matrix
        axis = np.asarray(record["axis_base"], dtype=float)/np.linalg.norm(record["axis_base"])
        # Pads closed across a part that runs along the hinge (a vertical pull on a vertical hinge) let the
        # part swivel between them, so the wrist need not turn the whole door angle: that is the reach a
        # step out of reach gets back. The effort guard still stops a grasp that binds.
        can_swivel = record["kind"] == "revolute" and abs(float(quaternion_matrix(tuple(orientation))[:, 0] @ axis)) < .3
        swivel = 0.

        def swivelled(step_orientation, value, fraction):
            if fraction == 0.:
                return step_orientation
            back = rotation_about(axis, -float(record["direction"])*value*fraction)
            return tuple(float(v) for v in quaternion_xyzw_from_matrix(back @ quaternion_matrix(tuple(step_orientation))))
        await self._own("follow_constraint", context)
        achieved, peak, trip_info, status, detail, steps = 0., 0., None, "succeeded", "", []
        measured, frames, initial_normal, verified, scores = [], [], None, None, {}
        scene = self.scene
        if scene is not None:
            initial_normal, keyframe0 = await scene.surface_normal()
            if keyframe0 is not None:
                try:
                    frames.append(await asyncio.to_thread(scene.crop_for, keyframe0))
                except Exception:                            # noqa: BLE001 - a demo frame is optional
                    pass
        try:
            for value, step_position, step_orientation in plan:
                if context.cancel_event.is_set():
                    status, detail = "cancelled", "execution epoch was revoked"
                    raise BackendFailure("cancelled", detail)
                if not await self.client.stationary(duration_s=self.stationary_duration_s):
                    status, detail = "stale_state", "the arm is not verifiably still before the next step"
                    raise BackendFailure("stale_state", detail)
                trajectory, refusal = None, None
                for fraction in [swivel]+[f for f in (.5, 1.) if can_swivel and f > swivel]:
                    try:
                        trajectory, planning = await self.client.plan_to_pose(step_position, swivelled(step_orientation, value, fraction),
                                                                              cancel_event=context.cancel_event)
                    except SheppyClientError as exc:
                        refusal = exc
                        if "IK_FAIL" not in str(exc):
                            break                           # only reach is worth another wrist; anything else stands
                        continue
                    if fraction != swivel:
                        self.log(f"follow {constraint_id}: {value:.3f} {record['unit']} out of reach with the wrist turning "
                                 f"{(1-swivel)*100:.0f}% of the door; the part swivels in the grasp, wrist at {(1-fraction)*100:.0f}%")
                        swivel = fraction
                    break
                if trajectory is None:
                    status, detail = "planning_failed", str(refusal)
                    raise BackendFailure("planning_failed", f"step to {value:.3f} {record['unit']}: {refusal}") from refusal
                trajectory = self._scaled(trajectory, "contact")
                guard = self._guard(touch_nm=float(record["contact_effort_nm"]), tool_exclusion_m=self.tool_exclusion_m)
                await context.feedback(event="constraint_step", skill="follow_constraint", value=value,
                                       unit=record["unit"], duration_s=trajectory.duration_s)
                receipt = await self.client.execute(trajectory, cancel_event=context.cancel_event, guard=guard)
                effort = getattr(guard, "effort", None)
                if effort is not None:
                    peak = max(peak, float(getattr(effort, "peak_nm", 0.)))
                if receipt["status"] != "succeeded" and getattr(self.client, "settle", None) is not None:
                    await self.client.settle(timeout_s=3.)     # report the stop from a still arm
                steps.append({"value": value, "status": receipt["status"], "planning": planning,
                              "trajectory_digest": trajectory.digest, "swivel": swivel})
                self.log(f"follow {constraint_id}: {value:.3f}/{target:.3f} {record['unit']} {receipt['status']} "
                         f"peak {peak:.1f} Nm")
                if receipt["status"] == "cancelled":
                    status, detail = "cancelled", receipt["message"]
                    raise BackendFailure("cancelled", detail)
                if receipt["status"] == "guard_trip":
                    trip_info = receipt.get("trip") or {}
                    status, detail = "tripped", receipt["message"]
                    kind = trip_info.get("kind")
                    code = "model_mismatch" if kind == "contact" else "stale_state" if kind == "collision" else "safety_fault"
                    raise BackendFailure(code, f"stopped at {achieved:.3f} of {target:.3f} {record['unit']}: {receipt['message']}",
                                         evidence=[{"evidence_id": f"guard-{uuid.uuid4().hex}", "source": self.mode,
                                                    "predicates": [], "ttl_s": 30.,
                                                    "data": {"trip": checked_copy(trip_info), "receipt": checked_copy(receipt),
                                                             "achieved": achieved, "constraint": constraint_id}}])
                if receipt["status"] != "succeeded":
                    status, detail = "failed", receipt["message"]
                    raise BackendFailure("safety_fault" if receipt["status"] in ("goal_not_reached", "timeout") else "planning_failed",
                                         detail)
                achieved = value
                live = self.client.live_joints()
                knuckle = None if live is None else live["knuckle_rad"]
                if (knuckle is not None and self.grasp_knuckle is not None
                        and knuckle > self.grasp_knuckle+self.grasp_stall_margin_rad):
                    status, detail = "slipped", "the gripper closed further than at grasp; the part slipped out"
                    raise BackendFailure("slip", detail)
                if scene is not None and record["kind"] == "revolute" and initial_normal is not None:
                    normal, keyframe = await scene.surface_normal()
                    turned = None if normal is None else turn_between(initial_normal, normal, record["axis_base"])
                    measured.append({"value": value, "turned_rad": turned, "keyframe": None if keyframe is None else keyframe.capture_id})
                    if keyframe is not None and (len(frames) < 2 or value >= target-1e-9):
                        try:
                            frames.append(await asyncio.to_thread(scene.crop_for, keyframe))
                        except Exception:                    # noqa: BLE001 - a demo frame is optional
                            pass
            # Local verification: the door face must have turned as far as the tool did.
            # A waypoint where no plane was seen says nothing; the last one that was
            # seen is compared with its own waypoint, and verifies the goal only if it is the final one.
            seen = [m for m in measured if m["turned_rad"] is not None]
            if seen:
                last = seen[-1]
                if abs(last["turned_rad"]-last["value"]) > self.turn_tolerance_rad:
                    verified = False
                    status, detail = "unverified", (f"the surface turned {last['turned_rad']:.2f} rad while the tool moved "
                                                    f"{last['value']:.2f} rad")
                    raise BackendFailure("goal_unobserved", detail)
                verified = True if last["value"] >= achieved-1e-9 else None
            scores["local"] = progress_score(achieved=achieved, target=target, grasped=True, verified=verified)
            reasoner = getattr(scene, "reasoner", None)
            if reasoner is not None and frames:
                try:
                    judged = await reasoner.verify_progress(
                        self._snapshot(context).context, task_text=f"{record['opening']} the {record['label']}",
                        rubric=PROGRESS_RUBRIC, images=[frames[-1]],
                        question=f"The tool moved {achieved:.2f} {record['unit']} of {target:.2f}. How far has the part moved?")
                    if judged.status == "OK" and judged.proposal is not None:
                        scores["model"] = judged.proposal["score"]
                        scores["model_observation"] = judged.detail[:200]
                        scores["model_confidence"] = judged.proposal["confidence"]
                except Exception as exc:                    # noqa: BLE001 - a second opinion, never a gate
                    scores["model_error"] = str(exc)[:120]
            facts = [assertion("constraint_goal_verified", {"constraint_id": constraint_id, "target_value": args["target_value"],
                                                            "target_unit": args["target_unit"]})]
            if self.current_pose is not None:
                facts.append(assertion("at_pose", {"entity_id": self.current_pose[0], "pose_role": self.current_pose[1]}, "false"))
                if self.current_pose[1] == "grasp":
                    facts.append(assertion("at_grasp_pose", {"entity_id": self.current_pose[0]}, "false"))
            self.current_pose = None
            return self._outcome(facts, data={
                "constraint": {"constraint_id": constraint_id, "kind": record["kind"], "digest": record.get("digest"),
                               "parameters_version": record.get("parameters_version")},
                "profile_id": profile["profile_id"], "target": target, "achieved": achieved, "unit": record["unit"],
                "steps": steps, "peak_effort_nm": peak, "contact_effort_nm": record["contact_effort_nm"],
                "measured": measured, "verified_locally": verified, "progress": scores,
                "evidence_basis": ("tool traced the constraint path with the grip retained; the surface normal seen by the "
                                   "wrist camera turned with it" if verified else
                                   "tool traced the constraint path with the grip retained; the part's own displacement was not measured")})
        finally:
            self._release("follow_constraint")
            try:
                if "local" not in scores:
                    scores["local"] = progress_score(achieved=achieved, target=target, grasped=self.holding_id == entity_id,
                                                     verified=verified)
                progress = {**scores, "measured_turn_rad": ([m["turned_rad"] for m in measured if m["turned_rad"] is not None] or [None])[-1],
                            "verified_locally": verified}
                record_attempt(record, task_id=context.task_id, target=target, achieved=achieved, status=status,
                               detail=detail, peak_effort_nm=peak, trip=trip_info, progress=progress)
                if self.constraint_store is not None:
                    if status == "succeeded" and frames:
                        save_demonstration(self.constraint_store, record, frames)
                    self.constraint_store.save(record)
            except Exception as exc:                        # noqa: BLE001 - bookkeeping never masks the outcome
                self.events.append({"event": "attempt_record_failed", "detail": str(exc), "at": time.monotonic()})


def _nearest_cluster(points, normal, target, *, cell_m=.01, reach_cells=2, seed_within_m=None):
    """The connected group of points (in the surface's plane, on a coarse grid) nearest the target's line.

    The group starts from the occupied cell nearest the line, which must lie within seed_within_m of it,
    and grows through every connected cell, so a part is taken whole however far it runs.
    """
    import numpy as np
    if len(points) == 0:
        return None
    normal = np.asarray(normal, dtype=float)
    a = np.cross(normal, [0., 0., 1.]) if abs(normal[2]) < .9 else np.cross(normal, [1., 0., 0.])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    relative = np.asarray(points, dtype=float)-np.asarray(target, dtype=float)
    cells = np.floor(np.column_stack([relative @ a, relative @ b])/cell_m).astype(int)
    occupied = {}
    for index, cell in enumerate(map(tuple, cells)):
        occupied.setdefault(cell, []).append(index)
    seed = min(occupied, key=lambda c: c[0]*c[0]+c[1]*c[1])
    if seed_within_m is not None and np.hypot(seed[0]+.5, seed[1]+.5)*cell_m > seed_within_m:
        return None
    seen, frontier = {seed}, [seed]
    while frontier:
        x, y = frontier.pop()
        for dx in range(-reach_cells, reach_cells+1):
            for dy in range(-reach_cells, reach_cells+1):
                neighbour = (x+dx, y+dy)
                if neighbour in occupied and neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
    return np.asarray(points)[[i for cell in seen for i in occupied[cell]]]


async def bootstrap_robot_facts(world, client, *, source_name="sheppy_client_bootstrap",
                                open_knuckle_rad=.05, closed_knuckle_rad=KNUCKLE_CLOSED_RAD-.03,
                                stationary_duration_s=.5):
    """Register the robot's initial held/empty state from measurement, not from JSON.

    held_state comes from a fresh stationary dwell. gripper_empty is asserted
    only when the context declares no attachment and the knuckle reads either
    open or at its closed stroke limit, where nothing can sit between the
    fingers; an open gripper can still hold a wide object, so the attachment
    declaration is the operator's, not this function's. An intermediate
    knuckle stays unknown.
    """
    live = client.live_joints()
    if live is None:
        raise ContractError("no fresh /joint_states; the robot's state cannot be bootstrapped")
    if not await client.stationary(duration_s=stationary_duration_s):
        raise ContractError("the arm is moving; refusing to assert a held state")
    facts = [assertion("held_state", {"robot_id": "robot"})]
    context = world.snapshot().context
    knuckle = live["knuckle_rad"]
    knuckle_state = ("unknown" if knuckle is None else "open" if knuckle <= open_knuckle_rad
                     else "closed" if knuckle >= closed_knuckle_rad else "intermediate")
    empty = knuckle_state in ("open", "closed") and context.get("attachment_id") == "empty"
    if empty:
        facts.append(assertion("gripper_empty", {"robot_id": "robot"}))
    authority = world.authorize_source(source_name, world.catalog.predicates)
    evidence_id = "bootstrap-"+digest({"joints": list(live["position_rad"]), "knuckle": live["knuckle_rad"],
                                       "at": live["received_at_monotonic_s"]})
    world.register_evidence(evidence_id, source=authority, predicates=facts,
                            ttl_s=world.max_evidence_age_s, observed_at=world.clock())
    # Evidence records a measurement; only a committed operation asserts it.
    snapshot = world.snapshot()
    key = "operation-"+evidence_id
    world.register_operation(key, snapshot.execution_epoch)
    world.commit_effects(key, [{**fact, "evidence_id": evidence_id} for fact in facts],
                         snapshot.revision, snapshot.execution_epoch, backend_quiescent=True)
    return {"evidence_id": evidence_id, "facts": facts, "knuckle_rad": live["knuckle_rad"],
            "knuckle_state": knuckle_state, "gripper_empty_asserted": empty}
