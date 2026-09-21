"""Local observation and grounded free-space commissioning composition seams.

These are project-owned Python adapters, not proposed upstream driver calls.
The observation adapter has no command port. The arm session uses the existing
cuRobo planner and guarded trajectory gateway but deliberately does not implement
the catalog's rolling ``move_to_pose`` contract or advertise physical ADL skills.
Injected evaluators and ports are trusted composition code, never model output.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import math
from types import MappingProxyType
from typing import Any

from .contracts import ContractError, SkillRegistry, canonical_json, checked_copy, digest, validate_schema
from .handlers import BackendFailure, ObserveHandler, SkillOutcome
from .motion.commissioning import measured_transport_bounds
from .motion.driver_transport import VerifiedMotionState
from .motion.leasing import drain_nonpreemptible
from .motion.rolling import JointState, MotionError, MotionIdentity
from .world import MetricPose


async def _call(function, *args, **kwargs):
    result = function(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@dataclass(frozen=True)
class ObservationMeasurement:
    """Exact assertions from a local evaluator, using the world monotonic clock.

    This does not derive poses from detections. Fresh calibrated MetricPose
    proposals are returned in metric_poses and committed with their facts by the
    sole WorldModel writer. The observer must not mutate the world mid-operation.
    Dependencies identify the input snapshot; a proposed entity revision advances
    that snapshot. Image bytes are never part of the cloud interface.
    """
    entity_id: str
    camera: str
    purpose: str
    evidence_id: str
    captured_at: float
    valid_for_s: float
    assertions: tuple[dict, ...]
    data: Any
    dependencies: dict
    metric_poses: tuple[MetricPose, ...] = ()

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in
               (self.entity_id, self.camera, self.purpose, self.evidence_id)):
            raise ContractError("Observation identity must be nonempty")
        if (any(type(value) not in (int, float) or not math.isfinite(value)
                for value in (self.captured_at, self.valid_for_s)) or self.valid_for_s <= 0):
            raise ContractError("Observation requires finite capture time and positive validity")
        # Detach from a mutable sensor callback result before the executor sees it.
        object.__setattr__(self, "assertions", tuple(checked_copy(list(self.assertions))))
        object.__setattr__(self, "data", checked_copy(self.data))
        object.__setattr__(self, "dependencies", checked_copy(self.dependencies))
        poses = tuple(self.metric_poses)
        if any(not isinstance(pose, MetricPose) for pose in poses):
            raise ContractError("Observation geometry must use canonical MetricPose records")
        object.__setattr__(self, "metric_poses", poses)


class HardwareObservationBackend:
    """Existing ObserveHandler contract backed by local measured observations.

    ``observer(args, context)`` returns ObservationMeasurement. ``held_check()``
    must verify the complete robot's held state, including retained payload if
    applicable; an idle subscription or an arm-only zero velocity is insufficient.
    This backend cannot command a stop: the deployment supervisor must retain its
    independent robot stop path. Its stop methods cancel observation work only.
    """
    mode = "local_measured_observation"
    hardware_commands = False
    fixture_capabilities = frozenset()
    physical_capabilities = frozenset()

    def __init__(self, catalog, world, *, observer, held_check):
        if not callable(observer) or not callable(held_check):
            raise ContractError("Observation requires local evaluator and verified hold reader")
        self.catalog, self.world = catalog, world
        self.observer, self.held_check = observer, held_check
        self._work = None
        self._busy = False
        self._cancelled = False

    def handlers(self):
        return {"observe": ObserveHandler(self)}

    def registry(self, *, capabilities, commissioned=False, mode="hardware"):
        """Keep catalog/commissioning gates; never infer capabilities from a port.

        Explicit ``mode='simulation'`` supports local rehearsal. It does not
        change the catalog's planned hardware status or enable robot commands.
        """
        return SkillRegistry(self.catalog, self.handlers(), capabilities,
                             mode=mode, commissioned=commissioned)

    async def quiescent(self):
        if self._busy or (self._work is not None and not self._work.done()):
            return False
        try:
            held = await _call(self.held_check)
            return held is True and not self._busy and (self._work is None or self._work.done())
        except Exception:
            return False

    async def skill_quiescent(self, skill_id):
        return skill_id == "observe" and not self._busy and (self._work is None or self._work.done())

    async def stop_skill(self, skill_id, reason):
        if skill_id != "observe":
            raise BackendFailure("safety_fault", "Observation backend has no robot command/stop port")
        self._cancelled = True
        if self._work is not None and not self._work.done():
            self._work.cancel()

    async def stop(self, reason="supervisor"):
        await self.stop_skill("observe", reason)

    def _context_current(self, context):
        snapshot = self.world.snapshot()
        if (context.cancel_event.is_set() or self._cancelled
                or context.execution_epoch != snapshot.execution_epoch):
            raise BackendFailure("cancelled", "Observation epoch was revoked")
        if context.task_id != snapshot.context["task_id"]:
            raise BackendFailure("stale_state", "Observation belongs to a different task")
        return snapshot

    def _outcome(self, measurement, args, context):
        if not isinstance(measurement, ObservationMeasurement):
            raise BackendFailure("geometry_invalid", "Local evaluator omitted a measured observation")
        if (measurement.entity_id, measurement.camera, measurement.purpose) != (
                args["entity_id"], args["camera"], args["purpose"]):
            raise BackendFailure("stale_state", "Observation target, view or purpose differs from the request")
        snapshot = self._context_current(context)
        now = self.world.clock()
        if (not measurement.captured_at <= now < measurement.captured_at + measurement.valid_for_s
                or measurement.valid_for_s > self.world.max_evidence_age_s):
            raise BackendFailure("stale_state", "Observation capture is stale or from an incompatible clock")
        identities = snapshot.identities()
        required_dependencies = {"execution_epoch", "calibration_id", "base_epoch",
                                 "entity:" + measurement.entity_id}
        if not required_dependencies <= measurement.dependencies.keys():
            raise BackendFailure("stale_state", "Observation lacks target/calibration/base/epoch provenance")
        if any(identities.get(key) != value for key, value in measurement.dependencies.items()):
            raise BackendFailure("stale_state", "Observation dependencies changed before completion")
        proposed_poses = {}
        for pose in measurement.metric_poses:
            key = (pose.entity_id, pose.pose_role)
            if (key in proposed_poses or pose.entity_id != measurement.entity_id
                    or pose.evidence_id != measurement.evidence_id
                    or pose.captured_at != measurement.captured_at
                    or pose.valid_for_s != measurement.valid_for_s
                    or pose.entity_revision != identities["entity:" + measurement.entity_id] + 1
                    or pose.calibration_id != identities["calibration_id"]
                    or pose.base_epoch != identities["base_epoch"]
                    or pose.frame_id != self.catalog.library["frames"]["planning"]):
                raise BackendFailure("geometry_invalid", "Proposed geometry differs from the measured observation")
            proposed_poses[key] = pose
        facts, keys = [], set()
        expected = {"predicate": "observation_valid", "args": {
            "entity_id": measurement.entity_id, "purpose": measurement.purpose}, "validity": "true"}
        for incoming in measurement.assertions:
            fact = checked_copy(incoming)
            if set(fact) != {"predicate", "args", "validity"} or fact["validity"] != "true":
                raise BackendFailure("geometry_invalid", "Successful observation needs explicit true assertions")
            self.catalog.validate_predicate(fact["predicate"], fact["args"])
            if fact["predicate"] not in {"observation_valid", "entity_exists", "pose_valid"}:
                raise BackendFailure("geometry_invalid", "Observation cannot establish physical skill effects")
            if fact["args"].get("entity_id") != measurement.entity_id:
                raise BackendFailure("geometry_invalid", "Observation cannot refresh another entity")
            key = digest(fact)
            if key in keys:
                raise BackendFailure("geometry_invalid", "Duplicate observation assertion")
            keys.add(key)
            if fact["predicate"] == "observation_valid" and fact != expected:
                raise BackendFailure("geometry_invalid", "Observation cannot refresh an unmeasured purpose")
            if fact["predicate"] == "pose_valid":
                pose_key = (measurement.entity_id, fact["args"]["pose_role"])
                pose = snapshot.metric_poses.get(pose_key)
                if pose_key in proposed_poses:
                    pass  # Checked above; WorldModel atomically verifies and installs it.
                elif proposed_poses:
                    raise BackendFailure("geometry_invalid", "New entity geometry cannot preserve an older pose role")
                elif (pose is None or pose.captured_at != measurement.captured_at
                        or pose.entity_revision != identities["entity:" + measurement.entity_id]
                        or measurement.valid_for_s > pose.valid_for_s
                        or not pose.captured_at <= now < pose.captured_at + pose.valid_for_s
                        or snapshot.fact("pose_valid", fact["args"]) != "true"):
                    raise BackendFailure("geometry_invalid", "Pose assertion lacks matching measured metric geometry")
            facts.append(fact)
        for entity_id, role in proposed_poses:
            if {"predicate": "pose_valid", "args": {"entity_id": entity_id, "pose_role": role},
                    "validity": "true"} not in facts:
                raise BackendFailure("geometry_invalid", "Proposed geometry lacks its exact pose assertion")
        if expected not in facts:
            raise BackendFailure("geometry_invalid", "Local evaluator did not establish the requested observation")
        if measurement.purpose in {"pose", "grasp"} and not any(f["predicate"] == "pose_valid" for f in facts):
            raise BackendFailure("geometry_invalid", "Pose observation needs explicitly measured metric pose evidence")
        evidence = {"evidence_id": measurement.evidence_id, "source": self.mode,
                    "observed_at": measurement.captured_at, "ttl_s": measurement.valid_for_s,
                    "predicates": facts, "dependencies": checked_copy(measurement.dependencies),
                    "data": {"camera": measurement.camera, "measurement": checked_copy(measurement.data)}}
        return SkillOutcome("succeeded", {"evidence_id": measurement.evidence_id}, [evidence],
                            [{**fact, "evidence_id": measurement.evidence_id} for fact in facts],
                            metric_poses=measurement.metric_poses)

    async def observe(self, args, context):
        validate_schema(args, self.catalog.skills["observe"]["arguments"], "observe args")
        if self._busy or (self._work is not None and not self._work.done()):
            raise BackendFailure("safety_fault", "Observation evaluator is already owned")
        # Reserve before the first await, including both held-state checks.
        self._busy = True
        self._cancelled = False
        work = None
        try:
            self._context_current(context)
            if await _call(self.held_check) is not True:
                raise BackendFailure("safety_fault", "Observation requires verified global hold")
            self._context_current(context)
            self._work = work = asyncio.create_task(_call(self.observer, checked_copy(args), context))
            try:
                measurement = await asyncio.shield(work)
            except asyncio.CancelledError:
                self._cancelled = True
                work.cancel()
                try:
                    await drain_nonpreemptible(work)
                finally:
                    raise
            if await _call(self.held_check) is not True:
                raise BackendFailure("safety_fault", "Verified hold was lost during observation")
            return self._outcome(measurement, args, context)
        except ContractError as exc:
            raise BackendFailure("geometry_invalid", str(exc)) from exc
        finally:
            if work is not None and work.done() and self._work is work:
                self._work = None
            self._busy = False


def observation_runtime(world, *, observer, held_check, capabilities, commissioned=False,
                        mode="hardware", confirmation_callback=None, stop_timeout_s=2.):
    """Compose the existing runtime around an already populated trusted world.

    The caller initializes ``available_skills`` to the actual registry result
    before creating its WorldModel. Evidence must be populated by local sources;
    this helper never trusts initial JSON facts or emulates capabilities. With
    today's planned catalog, hardware mode cannot expose an observation handler;
    explicit simulation mode is available for an integration rehearsal.
    """
    from .app import Runtime
    from .executor import DagExecutor
    from .telemetry import TraceRecorder
    from .validation import PlanValidator
    backend = HardwareObservationBackend(world.catalog, world, observer=observer, held_check=held_check)
    registry = backend.registry(capabilities=capabilities, commissioned=commissioned, mode=mode)
    if set(world.snapshot().context["available_skills"]) != set(registry.available_skills):
        raise ContractError("Observation world must advertise exactly the locally available registry skills")
    validator = PlanValidator(world.catalog, registry, world)
    trace = TraceRecorder()
    executor = DagExecutor(world.catalog, registry, world, validator, backend, trace=trace,
                           confirmation_callback=confirmation_callback, stop_timeout_s=stop_timeout_s)
    return Runtime(world.catalog, registry, world, validator, backend, executor, trace)


@dataclass(frozen=True, eq=False)
class PreparedCommissioningMove:
    """In-memory, single-use candidate; serialization never conveys admission."""
    identity: MotionIdentity
    pose: Any
    dependencies: Any
    trajectory: Any
    permit: Any
    prepared_at: float

    def review(self):
        return {"scope": "grounded_free_space_commissioning", "physical_adl_available": False,
                "task_id": self.identity.task_id, "node_id": self.identity.node_id,
                "execution_epoch": self.identity.execution_epoch, "entity_id": self.pose.entity_id,
                "pose_role": self.pose.pose_role, "pose_evidence_id": self.pose.evidence_id,
                "trajectory_digest": self.trajectory.digest, "duration_s": self.trajectory.duration_s}


class CommissioningArmSession:
    """Bind one measured world pose to retained cuRobo planning and exact execution.

    ``stationary_check`` must supply verified stationary dwell/uncertainty evidence
    (e.g. DriverFeedbackBuffer.stationary_start), not an instantaneous zero sample.
    ``command_check(context)`` verifies current task/epoch/resource authorization;
    no ownership is acquired/released here. The gateway must use this session's
    ``admission_check`` as its final command gate so identity/freshness changes are
    rechecked during driver admission and each monitored tracking checkpoint.
    Preparing another skill still requires the composition's planner lease.
    """
    physical_adl_skills = ()

    def __init__(self, world, planner, admission, *, stationary_check, collision_world,
                 command_check, gateway=None, clock=None):
        self.world, self.planner, self.admission = world, planner, admission
        self.stationary_check, self.command_check = stationary_check, command_check
        self.clock = clock or world.clock
        self.gateway = gateway
        if (not callable(command_check) or inspect.iscoroutinefunction(command_check)
                or not callable(stationary_check) or not callable(getattr(planner, "plan_pose", None))):
            raise MotionError("Commissioning requires a local synchronous authorization gate and measured/planner ports")
        frozen_world = checked_copy(collision_world)
        if digest(frozen_world) != admission.settings.world_digest:
            raise MotionError("Commissioned collision world differs from the planner world")
        self._collision_world_json = canonical_json(frozen_world)
        self._busy = False
        self._prepared = None
        self._consumed = False
        self._context = None
        self._active_gateway = None
        self.last_invalidation_reason = ""

    @property
    def collision_world(self):
        """A detached view cannot mutate the commissioned planning snapshot."""
        return checked_copy(self._collision_world_json)

    def _checked_gateway(self):
        gateway = self.gateway
        if gateway is None or gateway.admission_check != self.admission_check:
            raise MotionError("Gateway must use the world-bound commissioning admission gate")
        if gateway.bounds != measured_transport_bounds(self.admission.settings):
            raise MotionError("Gateway bounds differ from the measured commissioning allowances")
        return gateway

    def _stationary_state(self, state):
        if not isinstance(state, VerifiedMotionState):
            raise MotionError("Commissioning start needs verified driver feedback")
        bounds = measured_transport_bounds(self.admission.settings)
        now = self.clock()
        if (not 0 <= now-state.acquired_at_monotonic_s <= bounds.state_max_age_s
                or not 0 <= now-state.received_at_monotonic_s <= bounds.receipt_max_age_s
                or max(map(abs, state.velocity_rad_s)) > bounds.stationary_velocity_rad_s):
            raise MotionError("Stationary start feedback is stale or moving")
        return state

    @staticmethod
    def _identity(context):
        return MotionIdentity(context.task_id, context.node_id, context.attempt, context.execution_epoch)

    def _current(self, context, prepared=None):
        if context.cancel_event.is_set():
            raise MotionError("Commissioning operation cancelled")
        if self.command_check(context) is not True:
            raise MotionError("Commissioning task/resource authorization was revoked")
        snapshot = self.world.snapshot()
        if (context.execution_epoch != snapshot.execution_epoch
                or context.task_id != snapshot.context["task_id"]):
            raise MotionError("Commissioning task or execution epoch changed")
        if prepared is not None:
            if self._identity(context) != prepared.identity:
                raise MotionError("Prepared motion belongs to a different task/node/attempt/epoch")
            current = snapshot.identities()
            if any(current.get(key) != value for key, value in prepared.dependencies.items()):
                raise MotionError("Prepared motion world/target dependencies changed")
            pose = snapshot.metric_poses.get((prepared.pose.entity_id, prepared.pose.pose_role))
            if pose != prepared.pose or not pose.captured_at <= self.clock() < pose.captured_at + pose.valid_for_s:
                raise MotionError("Prepared target is stale or replaced")
            if snapshot.fact("pose_valid", {"entity_id": pose.entity_id, "pose_role": pose.pose_role}) != "true":
                raise MotionError("Prepared target lost valid measured pose evidence")
        self.admission.settings.unchanged()
        return snapshot

    def admission_check(self, permit, trajectory):
        """Use as JointTrajectoryTransport.admission_check; fail before every send."""
        if self._prepared is None or self._context is None or not self._consumed:
            raise MotionError("No actively dispatched commissioning operation")
        try:
            if self._checked_gateway() is not self._active_gateway:
                raise MotionError("Commissioning command gateway was replaced")
            self._current(self._context, self._prepared)
        except MotionError as exc:
            self.last_invalidation_reason = str(exc)
            raise
        if permit is not self._prepared.permit or trajectory.digest != self._prepared.trajectory.digest:
            raise MotionError("Trajectory differs from the world-bound commissioning candidate")
        return self.admission.check(permit, trajectory)

    async def prepare(self, entity_id, pose_role, context):
        if self._busy or self._prepared is not None or self._consumed:
            raise MotionError("Commissioning session is single-use or already leased")
        self._busy = True
        try:
            snapshot = self._current(context)
            pose = snapshot.metric_poses.get((entity_id, pose_role))
            if pose is None or snapshot.fact("pose_valid", {"entity_id": entity_id, "pose_role": pose_role}) != "true":
                raise MotionError("Commissioning target has no measured metric pose")
            identities = snapshot.identities()
            names = ("execution_epoch", "collision_revision", "base_epoch", "calibration_id",
                     "robot_config_id", "attachment_id", "grasp_state_id", "entity:" + entity_id)
            dependencies = MappingProxyType({name: identities[name] for name in names})
            binding = PreparedCommissioningMove(self._identity(context), pose, dependencies, None, None, self.clock())
            self._current(context, binding)
            start = self._stationary_state(await _call(self.stationary_check))
            # Zero derivatives describe the admitted stationary planning boundary,
            # never invented measured acceleration or a moving-start substitution.
            planning = asyncio.create_task(self.planner.plan_pose(
                position_m=list(pose.position_m), quaternion_xyzw=list(pose.orientation_xyzw),
                start=JointState(start.position_rad, (0.,)*7, (0.,)*7),
                world=checked_copy(self.collision_world), world_identity=self.admission.settings.world_digest))
            try:
                trajectory = await asyncio.shield(planning)
            except asyncio.CancelledError:
                await drain_nonpreemptible(planning)
                raise
            self._current(context, binding)
            # Validation may be expensive; retain this session's lease until its
            # nonpreemptible worker has drained even on cancellation.
            validating = asyncio.create_task(asyncio.to_thread(self.admission.validate, trajectory))
            try:
                permit = await asyncio.shield(validating)
            except asyncio.CancelledError:
                await drain_nonpreemptible(validating)
                raise
            self._current(context, binding)
            prepared = PreparedCommissioningMove(binding.identity, pose, dependencies,
                                                  permit.trajectory, permit, self.clock())
            self._prepared = prepared
            return prepared
        finally:
            self._busy = False

    async def execute(self, prepared, context):
        if (prepared is not self._prepared or prepared is None or self._consumed or self._busy
                or self.gateway is None):
            raise MotionError("Unknown, already consumed, busy or disconnected commissioning candidate")
        # Claim this candidate before any asynchronous state read. A second
        # caller must never wait through the first execution and replay it.
        self._busy = True
        try:
            gateway = self._checked_gateway()
            self._current(context, prepared)
            self.admission.check(prepared.permit, prepared.trajectory)
            start = self._stationary_state(await _call(self.stationary_check))
            if self._checked_gateway() is not gateway:
                raise MotionError("Commissioning command gateway was replaced")
            if any(abs(q-target) > tolerance for q, target, tolerance in zip(
                    start.position_rad, prepared.trajectory.points[0].state.position,
                    gateway.bounds.start_position_rad)):
                raise MotionError("Measured start changed during commissioning preparation")
            self._current(context, prepared)
            self._consumed, self._context, self._active_gateway = True, context, gateway
            receipt = await gateway.execute(prepared.trajectory, permit=prepared.permit,
                                            cancel_event=context.cancel_event)
            # Terminal ACK and measured settling remain solely the gateway's
            # responsibility. No predicted TCP/goal state is committed here.
            return receipt
        finally:
            self._busy = False
