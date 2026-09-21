"""Moving-state suffix replacement and independent timed-path checks.

This protocol is NEW project code, not a claim about the upstream ROS driver.
Trajectory interpolation is the exact quintic Hermite interpolation validated
here; adapters must not retime or interpolate it differently after validation.
No IK or arm path planner is implemented in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping


class MotionError(ValueError):
    """A candidate or continuation cannot be admitted."""


def _finite(values):
    return all(not isinstance(v, (bool, str)) and isinstance(v, (int, float)) and math.isfinite(v) for v in values)


@dataclass(frozen=True)
class JointState:
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]

    def __post_init__(self):
        for field in ("position", "velocity", "acceleration"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.position or not (len(self.position) == len(self.velocity) == len(self.acceleration)):
            raise MotionError("joint state dimensions disagree")
        if not all(_finite(v) for v in (self.position, self.velocity, self.acceleration)):
            raise MotionError("non-finite joint state")


@dataclass(frozen=True)
class TrajectoryPoint:
    time_s: float
    state: JointState

    def __post_init__(self):
        if not _finite((self.time_s,)) or self.time_s < 0 or not isinstance(self.state, JointState):
            raise MotionError("invalid trajectory point")


@dataclass(frozen=True)
class JointTrajectory:
    joint_names: tuple[str, ...]
    points: tuple[TrajectoryPoint, ...]
    provenance: str

    def __post_init__(self):
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "points", tuple(self.points))
        if not self.joint_names or any(not isinstance(name, str) or not name for name in self.joint_names):
            raise MotionError("trajectory joint names must be nonempty strings")
        if len(self.points) < 2 or self.points[0].time_s != 0:
            raise MotionError("trajectory requires at least two points beginning at zero")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise MotionError("duplicate joint names")
        previous = -1.0
        for point in self.points:
            if not math.isfinite(point.time_s) or point.time_s <= previous:
                raise MotionError("trajectory times must increase")
            if len(point.state.position) != len(self.joint_names):
                raise MotionError("trajectory joint dimensions disagree")
            previous = point.time_s
        if not self.provenance:
            raise MotionError("trajectory requires planner provenance")

    @property
    def duration_s(self):
        return self.points[-1].time_s

    @property
    def digest(self):
        payload = {"joints": self.joint_names, "provenance": self.provenance,
                   "points": [(p.time_s, p.state.position, p.state.velocity, p.state.acceleration) for p in self.points],
                   "interpolation": "quintic-hermite-v1"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def sample(self, time_s: float) -> JointState:
        if not math.isfinite(time_s) or time_s < 0 or time_s > self.duration_s + 1e-9:
            raise MotionError("sample outside validated trajectory")
        if time_s >= self.duration_s:
            return self.points[-1].state
        for first, last in zip(self.points, self.points[1:]):
            if first.time_s <= time_s <= last.time_s:
                dt = last.time_s - first.time_s
                u = (time_s - first.time_s) / dt
                positions, velocities, accelerations = [], [], []
                for q0, v0, a0, q1, v1, a1 in zip(
                    first.state.position, first.state.velocity, first.state.acceleration,
                    last.state.position, last.state.velocity, last.state.acceleration,
                ):
                    c0, c1, c2 = q0, v0 * dt, a0 * dt * dt / 2
                    delta, vel, acc = q1 - c0 - c1 - c2, v1 * dt - c1 - 2*c2, a1*dt*dt - 2*c2
                    c3, c4, c5 = 10*delta - 4*vel + acc/2, -15*delta + 7*vel - acc, 6*delta - 3*vel + acc/2
                    positions.append(c0 + c1*u + c2*u*u + c3*u**3 + c4*u**4 + c5*u**5)
                    velocities.append((c1 + 2*c2*u + 3*c3*u*u + 4*c4*u**3 + 5*c5*u**4)/dt)
                    accelerations.append((2*c2 + 6*c3*u + 12*c4*u*u + 20*c5*u**3)/(dt*dt))
                return JointState(tuple(positions), tuple(velocities), tuple(accelerations))
        raise MotionError("trajectory interval missing")


@dataclass(frozen=True)
class BoundaryTolerance:
    position_rad: float
    velocity_rad_s: float
    acceleration_rad_s2: float

    def __post_init__(self):
        if not _finite((self.position_rad, self.velocity_rad_s, self.acceleration_rad_s2)) or min(self.position_rad, self.velocity_rad_s, self.acceleration_rad_s2) < 0:
            raise MotionError("invalid boundary tolerance")

    def matches(self, expected: JointState, measured: JointState):
        if len(expected.position) != len(measured.position):
            return False
        return all(abs(a-b) <= bound for left, right, bound in (
            (expected.position, measured.position, self.position_rad),
            (expected.velocity, measured.velocity, self.velocity_rad_s),
            (expected.acceleration, measured.acceleration, self.acceleration_rad_s2),
        ) for a, b in zip(left, right))


@dataclass(frozen=True)
class JointLimits:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]

    def __post_init__(self):
        for field in ("lower", "upper", "velocity", "acceleration"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.lower or len({len(self.lower), len(self.upper), len(self.velocity), len(self.acceleration)}) != 1:
            raise MotionError("invalid limit dimensions")
        if not all(_finite(v) for v in (self.lower, self.upper, self.velocity, self.acceleration)):
            raise MotionError("non-finite limits")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)) or min(*self.velocity, *self.acceleration) <= 0:
            raise MotionError("invalid joint limits")

    def check(self, state: JointState):
        if len(state.position) != len(self.lower):
            raise MotionError("limits do not match robot")
        if any(not lo <= q <= hi for lo, q, hi in zip(self.lower, state.position, self.upper)):
            raise MotionError("joint position limit")
        if any(abs(v) > bound+1e-9 for v, bound in zip(state.velocity, self.velocity)):
            raise MotionError("joint velocity limit")
        if any(abs(a) > bound+1e-9 for a, bound in zip(state.acceleration, self.acceleration)):
            raise MotionError("joint acceleration limit")


@dataclass(frozen=True)
class ValidatedTrajectory:
    trajectory: JointTrajectory
    trajectory_digest: str
    dependencies: tuple[tuple[str, str], ...]
    expires_at: float
    validation_id: str
    validator_token: object


class TrajectoryValidator:
    """Trusted validation boundary, requiring an independent swept-path verifier.

    The geometric callback must cover the interpolated path AND stopping envelope,
    all links/payloads/contact allowances, not just pointwise optimizer success.
    The built-in sampled limit screen is insufficient alone for hardware admission.
    """
    def __init__(self, limits: JointLimits, swept_path_check: Callable[[JointTrajectory, Mapping[str, str]], bool], *, sample_dt_s=0.005):
        if not 0 < sample_dt_s <= 0.1:
            raise MotionError("invalid validation sample interval")
        self.limits, self.swept_path_check, self.sample_dt_s = limits, swept_path_check, sample_dt_s
        self._token = object()

    def validate(self, trajectory: JointTrajectory, dependencies: Mapping[str, str], *, now: float, expires_at: float):
        if not _finite((now, expires_at)) or expires_at <= now:
            raise MotionError("expired trajectory evidence")
        if not dependencies or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in dependencies.items()):
            raise MotionError("trajectory requires versioned dependency identities")
        samples = max(1, math.ceil(trajectory.duration_s/self.sample_dt_s))
        for i in range(samples+1):
            self.limits.check(trajectory.sample(i*trajectory.duration_s/samples))
        if not self.swept_path_check(trajectory, dependencies):
            raise MotionError("collision, constraint or stopping envelope rejected")
        digest = trajectory.digest
        return ValidatedTrajectory(trajectory, digest, tuple(sorted(dependencies.items())), expires_at, digest[:20], self._token)

    def check_certificate(self, certificate, dependencies, now):
        if not _finite((now,)):
            raise MotionError("invalid controller clock")
        if certificate.validator_token is not self._token or certificate.trajectory.digest != certificate.trajectory_digest:
            raise MotionError("untrusted or modified validation artifact")
        if now >= certificate.expires_at:
            raise MotionError("trajectory evidence expired")
        if any(dependencies.get(k) != v for k, v in certificate.dependencies):
            raise MotionError("trajectory dependencies changed")


@dataclass(frozen=True)
class MotionIdentity:
    task_id: str
    node_id: str
    attempt: int
    execution_epoch: int

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in (self.task_id, self.node_id)):
            raise MotionError("motion ownership identity is missing")
        if type(self.attempt) is not int or not 1 <= self.attempt <= 9007199254740991 or type(self.execution_epoch) is not int or not 0 <= self.execution_epoch <= 9007199254740991:
            raise MotionError("motion attempt/epoch is invalid")


@dataclass(frozen=True)
class Candidate:
    identity: MotionIdentity
    expected_generation: int
    next_generation: int
    switch_at: float
    expected_boundary: JointState
    certificate: ValidatedTrajectory

    def __post_init__(self):
        if not isinstance(self.identity, MotionIdentity) or not _finite((self.switch_at,)):
            raise MotionError("candidate identity or switch time is invalid")
        if any(type(v) is not int or not 0 <= v <= 9007199254740991 for v in (self.expected_generation, self.next_generation)):
            raise MotionError("candidate generation must be a bounded integer")


class RollingController:
    """Atomic acceptance/activation model for a single skill's trajectory buffer.

    The caller serializes install/tick/cancel on one event loop. This does not
    emulate undocumented upstream LATEST_WINS semantics. Every live stop is
    supplied and validated by the backend; this class never invents a stop path.
    stop_provider must return an already available stopping option within the
    local supervisor budget; it must not wait for a cloud/GPU planning round.
    """
    def __init__(self, identity, initial, validator, *, start_at, tolerance,
                 stop_provider, stop_budget_s, switch_lead_s=0.05, activation_lateness_s=0.01):
        if not _finite((start_at, stop_budget_s, switch_lead_s, activation_lateness_s)) or stop_budget_s <= 0 or switch_lead_s < 0 or activation_lateness_s < 0:
            raise MotionError("invalid timing budget")
        validator.check_certificate(initial, dict(initial.dependencies), start_at)
        self.identity, self.active, self.validator = identity, initial, validator
        self.start_at, self.tolerance, self.stop_provider = start_at, tolerance, stop_provider
        self.stop_budget_s, self.switch_lead_s = stop_budget_s, switch_lead_s
        self.activation_lateness_s = activation_lateness_s
        self.generation = 0
        self.pending = None
        self.state = "running"
        self.events = []
        self.stop_reason = None

    def expected(self, at):
        return self.active.trajectory.sample(at-self.start_at)

    def install(self, candidate, *, now, dependencies):
        if self.state != "running" or candidate.identity != self.identity:
            raise MotionError("motion ownership revoked")
        if candidate.expected_generation != self.generation or candidate.next_generation != self.generation+1:
            raise MotionError("stale or out-of-order trajectory generation")
        if self.pending is not None:
            raise MotionError("generation already pending; cancel it explicitly before replacing")
        if candidate.switch_at < now+self.switch_lead_s:
            raise MotionError("candidate missed committed-prefix deadline")
        self.validator.check_certificate(candidate.certificate, dependencies, now)
        expected = self.expected(candidate.switch_at)
        if not self.tolerance.matches(expected, candidate.expected_boundary) or not self.tolerance.matches(expected, candidate.certificate.trajectory.points[0].state):
            raise MotionError("discontinuous candidate boundary")
        self.pending = candidate
        ack = {"event": "installed", "generation": candidate.next_generation, "switch_at": candidate.switch_at,
               "trajectory_digest": candidate.certificate.trajectory_digest}
        self.events.append(ack)
        return ack

    def discard_pending(self, reason="superseded"):
        if self.pending:
            self.events.append({"event": "candidate_rejected", "generation": self.pending.next_generation, "reason": reason})
        self.pending = None

    def revalidate_active(self, certificate, *, now, dependencies):
        """Renew only independent validation of the exact currently active path."""
        if self.state != "running" or certificate.trajectory_digest != self.active.trajectory_digest:
            raise MotionError("revalidation must describe the exact active trajectory")
        self.validator.check_certificate(certificate, dependencies, now)
        self.active = certificate
        self.events.append({"event": "continuation_revalidated", "time_s": now})

    def request_stop(self, reason, *, now, measured, dependencies):
        if self.state in ("held", "fault", "stopping"):
            return
        self.discard_pending("motion revoked")
        self.stop_reason = reason
        try:
            stop = self.stop_provider(now, measured, dependencies)
            self.validator.check_certificate(stop, dependencies, now)
            if stop.trajectory.duration_s > self.stop_budget_s or not self.tolerance.matches(measured, stop.trajectory.points[0].state):
                raise MotionError("stop boundary or deadline invalid")
            terminal = stop.trajectory.points[-1].state
            if any(abs(v) > 1e-8 for v in (*terminal.velocity, *terminal.acceleration)):
                raise MotionError("stop does not end stationary")
            self.active, self.start_at, self.state = stop, now, "stopping"
            self.events.append({"event": "stop_started", "reason": reason, "time_s": now})
        except (MotionError, ValueError) as exc:
            self.state = "fault"
            self.events.append({"event": "supervisor_required", "reason": str(exc), "time_s": now})

    def tick(self, *, now, measured, dependencies, continuation_valid=True):
        if self.state in ("held", "fault"):
            return None
        if self.pending and now >= self.pending.switch_at:
            candidate = self.pending
            try:
                if now-candidate.switch_at > self.activation_lateness_s:
                    raise MotionError("activation deadline missed")
                self.validator.check_certificate(candidate.certificate, dependencies, now)
                expected_now = candidate.certificate.trajectory.sample(now-candidate.switch_at)
                if not self.tolerance.matches(expected_now, measured):
                    raise MotionError("measured activation state mismatch")
                self.active, self.start_at = candidate.certificate, candidate.switch_at
                self.generation, self.pending = candidate.next_generation, None
                self.events.append({"event": "activated", "generation": self.generation, "time_s": now})
            except MotionError as exc:
                self.discard_pending(str(exc))
        try:
            self.validator.check_certificate(self.active, dependencies, now)
            if not continuation_valid:
                raise MotionError("current stopping horizon invalid")
            elapsed = now-self.start_at
            if 0 <= elapsed <= self.active.trajectory.duration_s and not self.tolerance.matches(self.expected(now), measured):
                raise MotionError("active trajectory tracking mismatch")
        except MotionError as exc:
            if self.state == "stopping":
                self.state = "fault"
                self.events.append({"event": "supervisor_required", "reason": str(exc), "time_s": now})
                return None
            self.request_stop(str(exc), now=now, measured=measured, dependencies=dependencies)
        if self.state == "fault":
            return None
        remaining = self.start_at+self.active.trajectory.duration_s-now
        terminal = self.active.trajectory.points[-1].state
        stationary_end = all(abs(v) < 1e-8 for v in (*terminal.velocity, *terminal.acceleration))
        if self.state == "running" and self.active.expires_at-now <= self.stop_budget_s and remaining > self.active.expires_at-now:
            self.request_stop("evidence_horizon_depleted", now=now, measured=measured, dependencies=dependencies)
            remaining = self.start_at+self.active.trajectory.duration_s-now
            terminal = self.active.trajectory.points[-1].state
            stationary_end = all(abs(v) < 1e-8 for v in (*terminal.velocity, *terminal.acceleration))
        if self.state == "running" and remaining <= self.stop_budget_s and not stationary_end and self.pending is None:
            self.request_stop("buffer_underrun", now=now, measured=measured, dependencies=dependencies)
            remaining = self.start_at+self.active.trajectory.duration_s-now
            terminal = self.active.trajectory.points[-1].state
            stationary_end = all(abs(v) < 1e-8 for v in (*terminal.velocity, *terminal.acceleration))
        if self.state == "fault":
            return None
        if remaining <= 1e-9:
            if stationary_end and self.tolerance.matches(terminal, measured):
                self.state = "held"
                self.events.append({"event": "held", "time_s": now})
            else:
                self.state = "fault"
                self.events.append({"event": "supervisor_required", "reason": "terminal tracking mismatch", "time_s": now})
            return None
        return self.expected(now)
