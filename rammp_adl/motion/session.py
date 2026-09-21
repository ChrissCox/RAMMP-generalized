"""Async same-skill rolling session using NEW simulation transport contracts.

The tracking loop never waits for the planner or geometric validator. Only the
cuRobo adapter generates real candidate arm paths; injected fixture planners are
explicitly tagged in tests. This session exposes no hardware capabilities.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import time

from ..contracts import checked_copy
from .guards import TargetObservation, distance, quaternion_distance
from .leasing import drain_nonpreemptible
from .rolling import Candidate, MotionError


@dataclass(frozen=True)
class SessionWorld:
    """Locally measured versioned geometry, never model-generated metadata."""
    dependencies: dict
    planner_world: dict
    world_identity: str

    def copied(self):
        dependencies = checked_copy(self.dependencies)
        if not dependencies or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in dependencies.items()):
            raise MotionError("Rolling session requires nonempty versioned dependencies")
        if not isinstance(self.world_identity, str) or not self.world_identity:
            raise MotionError("Rolling session requires an explicit planner world identity")
        if dependencies.get("world") != self.world_identity:
            raise MotionError("Planner world identity must be bound into trajectory dependencies")
        return SessionWorld(dependencies, checked_copy(self.planner_world), self.world_identity)


@dataclass(frozen=True)
class SessionResult:
    status: str
    reason: str
    generation: int
    backend_quiescent: bool
    hardware_validated: bool = False


class RollingMotionSession:
    """Retains a single active skill's ownership until motion AND work drain.

    ``io`` is a trusted simulation adapter with hardware_commands=False and
    synchronous, bounded methods: owns(identity,resources), read_state(now),
    read_target(now), read_world(now), continuation_valid(now,state,world),
    command(state,now), supervisor_stop(reason,now), quiescent(), and
    goal_satisfied(target,state,world). Reads must be cached local observations;
    GPU/cloud/SDK waits do not belong in these callbacks. The supervisor callback
    owns the deterministic fallback after a fault or ownership loss.

    ``planner.candidate`` follows CuroboMpcAdapter's source-verified signature.
    The independent validator's callback must establish the exact interpolation
    and stopping envelope appropriate to its backend. Fixtures prove scheduling
    only. Cross-skill speculation and automatic controller fallback are absent.
    """
    hardware_commands = False
    capabilities = frozenset()

    def __init__(self, *, controller, planner, broker, guard, io, required_resources, planning_lead_s,
                 tick_period_s=.005, maximum_tick_gap_s=.05, maximum_duration_s=120.,
                 evidence_lifetime_s=5., clock=time.monotonic):
        numbers = (planning_lead_s, tick_period_s, maximum_tick_gap_s, maximum_duration_s, evidence_lifetime_s)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in numbers):
            raise MotionError("Invalid rolling session timing bounds")
        if getattr(io, "hardware_commands", None) is not False:
            raise MotionError("Only explicit simulation transports are enabled")
        required_resources = frozenset(required_resources)
        if (not {"ARM", "PLANNER"}.issubset(required_resources)
                or any(not isinstance(resource, str) or not resource for resource in required_resources)):
            raise MotionError("Full trusted skill claims must include ARM and PLANNER")
        if not tick_period_s <= maximum_tick_gap_s < planning_lead_s or maximum_duration_s > 3600.:
            raise MotionError("Rolling timing bounds cannot sustain future handoff")
        if planning_lead_s <= broker.maximum_call_s + controller.switch_lead_s + 2*tick_period_s:
            raise MotionError("Future switch must cover bounded solver/validation time and activation lead")
        if evidence_lifetime_s <= planning_lead_s + controller.stop_budget_s:
            raise MotionError("Candidate evidence lifetime cannot cover handoff and stopping")
        self.controller, self.planner, self.broker, self.guard, self.io = controller, planner, broker, guard, io
        self._required_resources = required_resources
        self.planning_lead_s, self.tick_period_s = planning_lead_s, tick_period_s
        self.maximum_tick_gap_s, self.maximum_duration_s = maximum_tick_gap_s, maximum_duration_s
        self.evidence_lifetime_s, self.clock = evidence_lifetime_s, clock
        self.events, self.running, self._planning = [], False, None
        self._desired_target, self._planned_target = guard.initial, guard.initial
        self._pending_target, self._planning_deadline = None, None
        self._request_serial, self._used = 0, False

    def _owns(self):
        if self.io.owns(self.controller.identity, self.required_resources) is not True:
            raise MotionError("Active skill no longer owns its complete admitted resource set")

    @property
    def required_resources(self):
        return self._required_resources

    def quiescent(self):
        return not self.running and (self._planning is None or self._planning.done()) and self.io.quiescent() is True

    def _observation(self, now):
        target = self.io.read_target(now)
        if not isinstance(target, TargetObservation):
            raise MotionError("Target callback did not provide typed local pose evidence")
        return TargetObservation(target.entity_id, target.pose_role, tuple(target.position_m), target.captured_at,
                                 target.uncertainty_m, target.identity_confident, tuple(target.quaternion_xyzw))

    def _same_target(self, first, second):
        return ((first.entity_id, first.pose_role) == (second.entity_id, second.pose_role)
                and distance(first.position_m, second.position_m) <= self.guard.meaningful_change_m
                and quaternion_distance(first.quaternion_xyzw, second.quaternion_xyzw) <= self.guard.meaningful_rotation_rad)

    async def _prepare(self, target, world, requested_at, switch_at, boundary, generation, serial):
        async def operation():
            self._owns()
            trajectory = await self.planner.candidate(moving_boundary=boundary,
                goal_pose=(target.position_m, target.quaternion_xyzw), world_config=world.planner_world,
                world_identity=world.world_identity)
            self._owns()
            # Independent geometry may be expensive. Keep it off the tracking
            # loop and retain the same planner lease through actual completion.
            certificate = await asyncio.to_thread(self.controller.validator.validate, trajectory, world.dependencies,
                now=requested_at, expires_at=requested_at+self.evidence_lifetime_s)
            return Candidate(self.controller.identity, generation, generation+1, switch_at, boundary, certificate)
        owner = f"{self.controller.identity.task_id}/{self.controller.identity.node_id}/{self.controller.identity.attempt}/{serial}"
        candidate = await self.broker.run(owner, operation, active_motion=True)
        return candidate, target, world

    def _launch(self, now, world):
        self._owns()
        switch_at = now+self.planning_lead_s
        if switch_at >= self.controller.start_at+self.controller.active.trajectory.duration_s:
            self.events.append({"event": "insufficient_committed_prefix", "time_s": now})
            return False
        boundary = self.controller.expected(switch_at)
        self._request_serial += 1
        self._planning_deadline = now+self.broker.maximum_call_s
        self._planning = asyncio.create_task(self._prepare(self._desired_target, world.copied(), now, switch_at,
            boundary, self.controller.generation, self._request_serial))
        self.events.append({"event": "planning_started", "time_s": now, "switch_at": switch_at,
                            "expected_generation": self.controller.generation})
        return True

    def _stop(self, reason, now, measured, world):
        self.controller.request_stop(reason, now=now, measured=measured, dependencies=world.dependencies)
        if self._planning is not None and not self._planning.done():
            self._planning.cancel()

    async def run(self, cancel_event=None):
        if self._used:
            raise MotionError("A rolling session is single-use")
        self._used, self.running = True, True
        cancel_event = cancel_event or asyncio.Event()
        task = asyncio.create_task(self._loop(cancel_event))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancel_event.set()
            try:
                await drain_nonpreemptible(task)
            finally:
                raise

    async def _loop(self, cancel_event):
        started = self.clock()
        previous = started
        reason = ""
        measured, world = None, None
        try:
            self._owns()
            while self.controller.state not in {"held", "fault"}:
                now = self.clock()
                self._owns()
                measured = self.io.read_state(now)
                world = self.io.read_world(now).copied()
                if cancel_event.is_set():
                    reason = "cancelled"
                    self._stop(reason, now, measured, world)
                elif now-started > self.maximum_duration_s or now-previous > self.maximum_tick_gap_s:
                    reason = "session_deadline" if now-started > self.maximum_duration_s else "tracking_tick_stalled"
                    self._stop(reason, now, measured, world)
                previous = now
                if self.controller.state == "running":
                    if self._pending_target is not None and self.controller.generation == self._pending_target[0]:
                        self._planned_target, self._pending_target = self._pending_target[1], None
                    target = self._observation(now)
                    if self.guard.accept(target, now=now):
                        self._desired_target = target
                    if (self.controller.pending is not None and self._pending_target is not None
                            and not self._same_target(self._pending_target[1], self._desired_target)):
                        self.controller.discard_pending("same-target pose superseded before activation")
                        self._pending_target = None
                    if self._planning is not None and self._planning.done():
                        completed, self._planning = self._planning, None
                        try:
                            candidate, planned_target, planned_world = completed.result()
                            self._owns()
                            if planned_world != world or not self._same_target(planned_target, self._desired_target):
                                raise MotionError("Candidate world or same-target observation was superseded")
                            self.controller.install(candidate, now=now, dependencies=world.dependencies)
                            self._pending_target = (candidate.next_generation, planned_target)
                        except (MotionError, asyncio.CancelledError) as exc:
                            self.events.append({"event": "planning_rejected", "time_s": now, "reason": str(exc)})
                            if self.broker.faulted:
                                reason = "planner_deadline"
                                self._stop(reason, now, measured, world)
                            elif "superseded" not in str(exc):
                                reason = "candidate_rejected"
                                self._stop(reason, now, measured, world)
                    if self._planning is not None and not self._planning.done() and now >= self._planning_deadline:
                        reason = "planner_deadline"
                        self._stop(reason, now, measured, world)
                    if self.controller.state == "running" and self._planning is None and self.controller.pending is None:
                        terminal = self.controller.active.trajectory.points[-1].state
                        nonstationary = any(abs(v) > 1e-8 for v in (*terminal.velocity, *terminal.acceleration))
                        remaining = self.controller.start_at+self.controller.active.trajectory.duration_s-now
                        horizon_due = nonstationary and remaining <= self.planning_lead_s+self.controller.stop_budget_s+2*self.tick_period_s
                        target_changed = not self._same_target(self._planned_target, self._desired_target)
                        if horizon_due or target_changed:
                            if not self._launch(now, world):
                                reason = "insufficient_committed_prefix"
                                self._stop(reason, now, measured, world)
                command = self.controller.tick(now=now, measured=measured, dependencies=world.dependencies,
                    continuation_valid=self.io.continuation_valid(now, measured, world) is True)
                if command is not None:
                    self._owns()
                    self.io.command(command, now)
                if self.controller.state == "fault":
                    reason = reason or "controller_fault"
                    self.io.supervisor_stop(reason, now)
                if self.controller.state not in {"held", "fault"}:
                    await asyncio.sleep(self.tick_period_s)
            reason = reason or self.controller.stop_reason or ""
        except Exception as exc:
            reason = str(exc)
            self.events.append({"event": "supervisor_required", "reason": reason, "time_s": self.clock()})
            self.controller.discard_pending("session ownership or callback failure")
            self.controller.state = "fault"
            self.io.supervisor_stop(reason, self.clock())
        finally:
            if self._planning is not None:
                self._planning.cancel()
                try:
                    await drain_nonpreemptible(self._planning)
                except (Exception, asyncio.CancelledError) as exc:
                    self.events.append({"event": "planner_drained", "detail": str(exc)})
                self._planning = None
            self.running = False
        held = self.io.quiescent() is True
        successful = (self.controller.state == "held" and not reason and held and measured is not None and world is not None
                      and self.io.goal_satisfied(self._desired_target, measured, world) is True)
        status = "succeeded" if successful else "cancelled" if cancel_event.is_set() and held else "failed"
        return SessionResult(status, reason or ("" if successful else "goal_or_hold_unverified"), self.controller.generation, held)
