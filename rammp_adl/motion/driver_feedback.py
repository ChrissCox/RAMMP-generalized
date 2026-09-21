"""Admission of NEW exchange-stamped driver feedback on the same Linux host.

Publication timestamps cannot establish acquisition freshness. The extension
brackets a successful transport exchange; a commissioned sensor-age bound is
subtracted from its start. No acceleration measurement is invented.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import numbers
import threading
import time

from .driver_state import ARM_JOINT_NAMES
from .driver_transport import DriverTransportError, VerifiedMotionState


@dataclass(frozen=True)
class FeedbackBounds:
    sensor_age_bound_s: float
    exchange_max_s: float
    receipt_max_age_s: float
    state_max_age_s: float
    velocity_error_rad_s: float
    maximum_jerk_rad_s3: float
    stationary_acceleration_rad_s2: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise DriverTransportError("Feedback bounds require explicit positive values: " + name)
        if self.sensor_age_bound_s + self.exchange_max_s >= self.state_max_age_s:
            raise DriverTransportError("Acquisition uncertainty consumes the entire state-age budget")


class DriverFeedbackBuffer:
    """Bounded cached evidence; repeated publication never refreshes a sample.

    A source reset, regression, invalid packet or fault latches rejection for
    this session. Start a new session after operator recovery; no automatic
    reset or resume. This class neither connects to nor commands a driver.
    """
    def __init__(self, *, bounds, simulation, extension_build_id, boot_id=None, clock=time.monotonic):
        if not isinstance(bounds, FeedbackBounds) or type(simulation) is not bool:
            raise DriverTransportError("Explicit feedback bounds and transport mode required")
        self.bounds, self.simulation, self.clock = bounds, simulation, clock
        if (not isinstance(extension_build_id, str) or len(extension_build_id) != 64
                or any(c not in "0123456789abcdef" for c in extension_build_id)):
            raise DriverTransportError("Explicit instrumented driver build identity required")
        self.extension_build_id = extension_build_id
        self.boot_id = boot_id or Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self._lock = threading.RLock()
        self._history = deque(maxlen=4096)
        self._last_identity = self._last_payload = None
        self._metadata = None
        self._fault = ""

    def ingest(self, message):
        with self._lock:
            if self._fault:
                return False
            try:
                now = self.clock()
                if (message.host_boot_id != self.boot_id or message.simulation is not self.simulation
                        or message.extension_build_id != self.extension_build_id
                        or tuple(message.joint_names) != ARM_JOINT_NAMES
                        or not isinstance(message.driver_session_id, str) or not message.driver_session_id
                        or len(message.driver_session_id) > 128):
                    raise DriverTransportError("Driver mode, host boot, session or joint identity mismatch")
                if (type(message.exchange_sequence) is not int or not 0 < message.exchange_sequence < 2**64
                        or any(type(v) is not int or v <= 0 for v in
                               (message.exchange_start_monotonic_ns, message.exchange_end_monotonic_ns))):
                    raise DriverTransportError("Invalid exchange sequence/timing")
                start, end = message.exchange_start_monotonic_ns / 1e9, message.exchange_end_monotonic_ns / 1e9
                if not 0 <= end-start <= self.bounds.exchange_max_s or end > now:
                    raise DriverTransportError("Successful exchange timing is invalid or exceeds its bound")
                if any(type(getattr(message, name)) is not bool for name in
                       ("fault", "heartbeat_required", "watchdog_latched", "owned")):
                    raise DriverTransportError("Malformed driver health metadata")
                if message.fault or message.watchdog_latched or not message.heartbeat_required:
                    raise DriverTransportError("Driver fault or independent watchdog unavailable/latched")
                timeout = message.heartbeat_timeout_s
                if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
                    raise DriverTransportError("Driver watchdog has no valid deadline")
                identity = (message.driver_session_id, message.exchange_sequence)
                # rosidl fixed float arrays are numpy.float64 on Humble.
                vectors = (message.position_rad, message.velocity_rad_s)
                if any(len(vector) != 7 or any(isinstance(v, (bool, str)) or not isinstance(v, numbers.Real)
                       or not math.isfinite(v) for v in vector) for vector in vectors):
                    raise DriverTransportError("Malformed acquired joint state")
                positions, velocities = (tuple(map(float, vector)) for vector in vectors)
                payload = (start, end, positions, velocities)
                metadata = (message.driver_session_id, message.owned, message.owner_id,
                            message.ownership_generation, timeout)
                if (type(message.ownership_generation) is not int or message.ownership_generation < 0
                        or not isinstance(message.owner_id, str)):
                    raise DriverTransportError("Malformed ownership in acquisition feedback")
                if self._last_identity is not None:
                    session, sequence = self._last_identity
                    if identity[0] != session or identity[1] < sequence:
                        raise DriverTransportError("Driver acquisition source reset or regressed")
                    if identity[1] == sequence:
                        if payload != self._last_payload:
                            raise DriverTransportError("Repeated exchange changed its acquired state")
                        self._metadata = metadata
                        return False
                    if start < self._history[-1][1]:
                        raise DriverTransportError("Driver successful exchanges overlap or regress")
                sample = VerifiedMotionState(positions, velocities,
                    start-self.bounds.sensor_age_bound_s, now, identity[1], identity[0])
                self._history.append((sample, end))
                self._last_identity, self._last_payload, self._metadata = identity, payload, metadata
                return True
            except (AttributeError, ValueError, TypeError, OverflowError) as exc:
                self._fault = str(exc)
                return False

    def read(self, now=None, *, ownership=None, watchdog_timeout_s=None):
        with self._lock:
            now = self.clock() if now is None else now
            if self._fault or not self._history:
                raise DriverTransportError(self._fault or "No successful driver exchange received")
            sample = self._history[-1][0]
            if (not 0 <= now-sample.acquired_at_monotonic_s <= self.bounds.state_max_age_s
                    or not 0 <= now-sample.received_at_monotonic_s <= self.bounds.receipt_max_age_s):
                raise DriverTransportError("Driver acquisition or receipt has expired")
            if ownership is not None and self._metadata[1:4] != (True, ownership.owner_id, ownership.generation):
                raise DriverTransportError("Acquisition feedback does not confirm this owner/generation")
            if watchdog_timeout_s is not None and abs(self._metadata[4]-watchdog_timeout_s) > 1e-9:
                raise DriverTransportError("Driver watchdog differs from commissioned timeout")
            return sample

    def require_unowned(self):
        with self._lock:
            sample = self.read()
            if self._metadata[1] is not False:
                raise DriverTransportError("Driver is already owned; this session will not seize control")
            return sample

    def stationary_start(self, *, duration_s, velocity_rad_s, position_span_rad, now=None):
        """Return nominal stationary planner q only after a measured dwell.

        Acceleration is bounded, not reported as measured: a secant of genuine
        velocity samples plus sensor uncertainty and a commissioned jerk bound
        encloses the current acceleration. Input timestamp intervals must be
        disjoint. A cached publication is never a second observation.
        """
        with self._lock:
            latest = self.read(now)
            selected = [item for item in self._history
                        if item[0].acquired_at_monotonic_s >= latest.acquired_at_monotonic_s-duration_s]
            if len(selected) < 2:
                raise DriverTransportError("Stationary dwell has insufficient independent exchanges")
            first = selected[0][0]
            # Sampling phase may shorten the inclusive window by one period.
            span = latest.acquired_at_monotonic_s-first.acquired_at_monotonic_s
            if span < duration_s-self.bounds.exchange_max_s:
                raise DriverTransportError("Stationary dwell is incomplete")
            for (older, _), (newer, _) in zip(selected, selected[1:]):
                if newer.acquired_at_monotonic_s-older.acquired_at_monotonic_s > self.bounds.state_max_age_s:
                    raise DriverTransportError("Acquisition gap interrupts stationary dwell")
            for joint in range(7):
                positions = [s.position_rad[joint] for s, _ in selected]
                if max(positions)-min(positions) > position_span_rad:
                    raise DriverTransportError("Arm position moved during stationary dwell")
                if any(abs(s.velocity_rad_s[joint])+self.bounds.velocity_error_rad_s > velocity_rad_s for s, _ in selected):
                    raise DriverTransportError("Arm velocity exceeds stationary boundary tolerance")
            earliest_latest = latest.acquired_at_monotonic_s
            end_latest = selected[-1][1]
            candidates = []
            for older, older_end in selected[:-1]:
                minimum_dt = earliest_latest-older_end
                maximum_dt = end_latest-older.acquired_at_monotonic_s
                if minimum_dt <= 0:
                    continue
                upper = max(abs(a-b) for a, b in zip(latest.velocity_rad_s, older.velocity_rad_s))
                upper = (upper+2*self.bounds.velocity_error_rad_s)/minimum_dt + self.bounds.maximum_jerk_rad_s3*maximum_dt
                candidates.append(upper)
            if not candidates or min(candidates) > self.bounds.stationary_acceleration_rad_s2:
                raise DriverTransportError("Stationary acceleration uncertainty exceeds commissioned boundary")
            return latest


class RosDriverFeedback:
    """Subscribe only; heartbeats are emitted explicitly by the monitor loop."""
    def __init__(self, node, buffer, *, feedback_topic="/rammp/driver_feedback", heartbeat_topic="/rammp/driver_heartbeat"):
        from rammp_adl_interfaces.msg import DriverFeedback, DriverHeartbeat
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.node, self.buffer, self._message = node, buffer, DriverHeartbeat
        self.subscription = node.create_subscription(DriverFeedback, feedback_topic, buffer.ingest, qos)
        self.publisher = node.create_publisher(DriverHeartbeat, heartbeat_topic, qos)
        self.sequence = 0

    def heartbeat(self, ownership, *, now=None):
        now = self.buffer.clock() if now is None else now
        sample = self.buffer.read(now, ownership=ownership)
        self.sequence += 1
        if self.sequence >= 2**64:
            raise DriverTransportError("Heartbeat sequence exhausted; new session required")
        msg = self._message()
        msg.driver_session_id, msg.host_boot_id = sample.source_id, self.buffer.boot_id
        msg.owner_id, msg.ownership_generation = ownership.owner_id, ownership.generation
        msg.token, msg.sequence = list(ownership.token), self.sequence
        msg.sent_monotonic_ns = int(now*1e9)
        self.publisher.publish(msg)

    def close(self):
        self.node.destroy_subscription(self.subscription)
        self.node.destroy_publisher(self.publisher)
