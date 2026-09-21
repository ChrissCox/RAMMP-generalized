"""Calibrated 2F-85 aperture transport; no grasp/retention inference.

Uses the pinned custom driver's GripperSetpoint (position, speed, current ceiling,
arm token) and NEW acquisition-stamped DriverFeedback. Ordinary GripperState's
publication time is not acquisition evidence. There are no default physical
limits/calibration and no ownership acquisition or robot launch in this module.

Cancellation uses the NEW driver transport halt latch. Its exchange receipt and
measured settling are independent conditions. It does not open the gripper, and
cannot be reset during a driver session. This is a local adapter, not an enabled
catalog skill. Calibrations/bounds/permits come only from trusted composition.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import struct
import threading
import time

from .driver_transport import DriverTransportError, TransferredOwnership


def _number(value, name, *, positive=False):
    if type(value) not in (float, int) or not math.isfinite(value) or (positive and value <= 0):
        raise DriverTransportError("Invalid gripper " + name)
    return float(value)


def _identity(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise DriverTransportError("Explicit gripper " + name + " required")
    return value


def _f32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


@dataclass(frozen=True)
class GripperCalibration:
    """Measured monotone aperture table; interpolation only, no extrapolation.

    normalized_position increases toward closed. aperture_m is the measured
    opposing finger gap, not knuckle joint angle. error_m bounds measurement,
    interpolation and hysteresis over the admitted table. It must be supplied
    from physical characterization for mode='commissioned'. 'simulation' is
    explicit and cannot be used by a physical transport.
    """
    gripper_id: str
    evidence_id: str
    mode: str
    normalized_position: tuple[float, ...]
    aperture_m: tuple[float, ...]
    error_m: float

    def __post_init__(self):
        _identity(self.gripper_id, "identity"); _identity(self.evidence_id, "calibration evidence")
        if self.mode not in {"simulation", "commissioned"}:
            raise DriverTransportError("Gripper calibration must have explicit simulation/commissioned mode")
        q = tuple(_number(v, "normalized sample") for v in self.normalized_position)
        aperture = tuple(_number(v, "aperture sample") for v in self.aperture_m)
        if (not 2 <= len(q) <= 4096 or len(q) != len(aperture) or not 0 <= q[0] < q[-1] <= 1
                or any(b <= a for a, b in zip(q, q[1:])) or aperture[-1] < 0
                or any(b >= a for a, b in zip(aperture, aperture[1:]))):
            raise DriverTransportError("Gripper aperture calibration must be strictly monotone and bounded")
        _number(self.error_m, "calibration error", positive=True)
        object.__setattr__(self, "normalized_position", q)
        object.__setattr__(self, "aperture_m", aperture)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _interpolate(value, xs, ys):
        if not xs[0] <= value <= xs[-1]:
            raise DriverTransportError("Gripper command/measurement lies outside the calibrated aperture range")
        for a, b, c, d in zip(xs, xs[1:], ys, ys[1:]):
            if a <= value <= b:
                return c + (d-c)*(value-a)/(b-a)
        raise DriverTransportError("Invalid aperture interpolation")

    def to_aperture(self, normalized):
        return self._interpolate(_number(normalized, "position"), self.normalized_position, self.aperture_m)

    def to_normalized(self, aperture_m):
        return self._interpolate(_number(aperture_m, "target aperture"),
                                 self.aperture_m[::-1], self.normalized_position[::-1])


@dataclass(frozen=True)
class GripperBounds:
    sensor_age_bound_s: float
    exchange_max_s: float
    state_max_age_s: float
    receipt_max_age_s: float
    target_tolerance_m: float
    stationary_span_m: float
    stationary_velocity_m_s: float
    maximum_acceleration_m_s2: float
    current_abort_a: float
    maximum_speed_fraction: float
    maximum_current_ceiling_fraction: float
    poll_period_s: float
    settle_duration_s: float
    command_timeout_s: float
    stop_timeout_s: float
    watchdog_timeout_s: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _number(getattr(self, name), name, positive=True)
        if (self.maximum_speed_fraction > 1 or self.maximum_current_ceiling_fraction > 1
                or self.sensor_age_bound_s+self.exchange_max_s >= self.state_max_age_s
                or self.poll_period_s >= min(self.state_max_age_s, self.receipt_max_age_s, self.settle_duration_s)
                or self.settle_duration_s >= min(self.command_timeout_s, self.stop_timeout_s)
                or max(self.command_timeout_s, self.stop_timeout_s) > 60):
            raise DriverTransportError("Inconsistent gripper commissioning timing or command bounds")


@dataclass(frozen=True)
class VerifiedGripperState:
    normalized_position: float
    aperture_m: float
    current_a: float
    acquired_lower_s: float
    acquired_upper_s: float
    received_at_s: float
    sequence: int
    source_id: str
    calibration_digest: str
    owned: bool
    owner_id: str
    ownership_generation: int
    watchdog_latched: bool
    halt_active: bool
    halt_generation: int
    halt_exchange_sequence: int
    halt_position_normalized: float


class GripperFeedbackBuffer:
    """Read-only admission of pinned exchange evidence, including halt evidence.

    A fault/source reset/mutated repeated acquisition latches rejection. A stop
    latch is distinct from a sensor fault: fresh measurements remain readable
    to determine physical settling after the driver revokes ownership.
    """
    def __init__(self, *, calibration, bounds, extension_build_id, simulation,
                 boot_id=None, clock=time.monotonic):
        if (not isinstance(calibration, GripperCalibration) or not isinstance(bounds, GripperBounds)
                or type(simulation) is not bool or (calibration.mode == "simulation") != simulation
                or not isinstance(extension_build_id, str) or len(extension_build_id) != 64
                or any(c not in '0123456789abcdef' for c in extension_build_id)):
            raise DriverTransportError("Explicit matching gripper calibration, bounds and driver build required")
        self.calibration, self.bounds, self.simulation = calibration, bounds, simulation
        self.extension_build_id, self.clock = extension_build_id, clock
        self.boot_id = boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self._lock = threading.RLock()
        self._last = self._payload = None
        self._fault = ''

    def ingest(self, msg):
        with self._lock:
            if self._fault:
                return False
            try:
                now = self.clock()
                if (msg.host_boot_id != self.boot_id or msg.extension_build_id != self.extension_build_id
                        or msg.simulation is not self.simulation):
                    raise DriverTransportError("Gripper driver host/build/mode mismatch")
                _identity(msg.driver_session_id, "driver session")
                for name in ('gripper_present', 'fault', 'heartbeat_required', 'watchdog_latched',
                             'owned', 'gripper_halt_supported', 'gripper_halt_active'):
                    if type(getattr(msg, name)) is not bool:
                        raise DriverTransportError("Malformed gripper health metadata")
                if msg.fault or not msg.gripper_present or not msg.heartbeat_required or not msg.gripper_halt_supported:
                    raise DriverTransportError("Gripper absent, driver fault, or guarded halt/watchdog unavailable")
                if abs(_number(msg.heartbeat_timeout_s, 'watchdog deadline')-self.bounds.watchdog_timeout_s) > 1e-9:
                    raise DriverTransportError("Gripper driver watchdog deadline mismatch")
                for name in ('exchange_sequence', 'exchange_start_monotonic_ns', 'exchange_end_monotonic_ns',
                             'ownership_generation', 'gripper_halt_generation', 'gripper_halt_exchange_sequence'):
                    value = getattr(msg, name)
                    if type(value) is not int or not 0 <= value < 2**64:
                        raise DriverTransportError("Malformed gripper exchange identity")
                start, end = msg.exchange_start_monotonic_ns/1e9, msg.exchange_end_monotonic_ns/1e9
                if (msg.exchange_sequence == 0 or start <= self.bounds.sensor_age_bound_s
                        or not 0 <= end-start <= self.bounds.exchange_max_s or end > now):
                    raise DriverTransportError("Invalid gripper acquisition clock interval")
                q = _number(msg.gripper_position_normalized, 'measured normalized position')
                aperture = self.calibration.to_aperture(q)
                current = _number(msg.gripper_current_a, 'measured current amps')
                hold = _number(msg.gripper_halt_position_normalized, 'halt position')
                if (not isinstance(msg.owner_id, str) or len(msg.owner_id) > 128
                        or (msg.owned and not msg.owner_id)
                        or (msg.gripper_halt_active and (msg.gripper_halt_generation != 1
                            or not 0 < msg.gripper_halt_exchange_sequence <= msg.exchange_sequence or not 0 <= hold <= 1))
                        or (not msg.gripper_halt_active and (msg.gripper_halt_generation or msg.gripper_halt_exchange_sequence))):
                    raise DriverTransportError("Malformed gripper ownership/halt acknowledgement")
                payload = (start, end, q, current, msg.gripper_halt_active, msg.gripper_halt_generation,
                           msg.gripper_halt_exchange_sequence, hold)
                if self._last:
                    old = self._last
                    if msg.driver_session_id != old.source_id or msg.exchange_sequence < old.sequence:
                        raise DriverTransportError("Gripper source reset or sequence regressed")
                    if msg.exchange_sequence == old.sequence:
                        if payload != self._payload:
                            raise DriverTransportError("Repeated gripper acquisition changed its payload")
                        self._last = replace(old, owned=msg.owned, owner_id=msg.owner_id,
                            ownership_generation=msg.ownership_generation, watchdog_latched=msg.watchdog_latched)
                        return False  # repeated publication never refreshes receipt/acquisition
                    if start < old.acquired_upper_s:
                        raise DriverTransportError("Gripper acquisitions overlap or regress")
                    if old.halt_active and not msg.gripper_halt_active:
                        raise DriverTransportError("Gripper halt cannot reset within the same driver session")
                self._last = VerifiedGripperState(q, aperture, current, start-self.bounds.sensor_age_bound_s,
                    end, now, msg.exchange_sequence, msg.driver_session_id, self.calibration.digest,
                    msg.owned, msg.owner_id, msg.ownership_generation, msg.watchdog_latched,
                    msg.gripper_halt_active, msg.gripper_halt_generation, msg.gripper_halt_exchange_sequence, hold)
                self._payload = payload
                return True
            except (AttributeError, ValueError, TypeError, OverflowError) as exc:
                self._fault = str(exc)
                return False

    def read(self, now=None, *, ownership=None):
        with self._lock:
            now = self.clock() if now is None else now
            state = self._last
            if self._fault or state is None:
                raise DriverTransportError(self._fault or 'No gripper exchange evidence')
            if (not 0 <= now-state.acquired_lower_s <= self.bounds.state_max_age_s
                    or not 0 <= now-state.received_at_s <= self.bounds.receipt_max_age_s):
                raise DriverTransportError('Gripper acquisition or receipt expired')
            if ownership is not None and (state.watchdog_latched or state.halt_active or not state.owned
                    or (state.owner_id, state.ownership_generation) != (ownership.owner_id, ownership.generation)):
                raise DriverTransportError('Gripper ownership lost or halt latched')
            return state


@dataclass(frozen=True)
class GripperCommand:
    aperture_m: float
    position_normalized: float
    speed_fraction: float
    current_ceiling_fraction: float
    calibration_digest: str


def calibrated_command(calibration, bounds, *, aperture_m, speed_fraction, current_ceiling_fraction):
    """Canonical float32 wire values are admitted before publishing; no defaults."""
    for value, maximum, name in ((speed_fraction, bounds.maximum_speed_fraction, 'speed'),
            (current_ceiling_fraction, bounds.maximum_current_ceiling_fraction, 'current ceiling')):
        if not 0 < _number(value, name) <= maximum or not 0 < _f32(value) <= maximum:
            raise DriverTransportError('Gripper command exceeds calibrated ' + name)
    q = _f32(calibration.to_normalized(aperture_m))
    realized = calibration.to_aperture(q)
    if abs(realized-aperture_m)+calibration.error_m > bounds.target_tolerance_m:
        raise DriverTransportError('Gripper quantization/calibration uncertainty exceeds target tolerance')
    return GripperCommand(float(aperture_m), q, _f32(speed_fraction), _f32(current_ceiling_fraction), calibration.digest)


def make_ros_gripper_setpoint(command, ownership):
    """Exact locally pinned rammp_arm_interfaces IDL, no legacy fallback."""
    if not isinstance(command, GripperCommand) or not isinstance(ownership, TransferredOwnership):
        raise DriverTransportError('Canonical calibrated command and transferred arm ownership required')
    from rammp_arm_interfaces.msg import GripperSetpoint
    msg = GripperSetpoint()
    msg.position, msg.speed, msg.force = command.position_normalized, command.speed_fraction, command.current_ceiling_fraction
    msg.token = list(ownership.token)
    return msg


@dataclass(frozen=True)
class GripperReceipt:
    status: str
    reason: str
    measured_aperture_m: float | None
    measured_quiescent: bool
    halt_acknowledged: bool
    ownership_retained: bool
    calibration_digest: str
    # No holding/released/retention assertion is manufactured from position/current.


class GripperTransport:
    """One active gripper command under the shared machine grant.

    The NEW local port has ready(), publish(command, ownership), and
    publish_stop(owner_id, reason). Stop acknowledgement comes from the actual
    exchange feedback, never the act of publishing. The permit callback must
    cover gripper emptiness/retention policy, aperture collision envelope,
    commissioned limits and task/node/attempt/epoch ownership.
    """
    def __init__(self, *, port, ownership, feedback, admission_check, owns_control, clock=time.monotonic):
        if (not isinstance(ownership, TransferredOwnership) or not isinstance(feedback, GripperFeedbackBuffer)
                or not callable(admission_check) or not callable(owns_control)):
            raise DriverTransportError('Trusted gripper permit, ownership and feedback required')
        self.port, self.ownership, self.feedback = port, ownership, feedback
        self.admission_check, self.owns_control, self.clock = admission_check, owns_control, clock
        self._lock = threading.Lock()
        self._active = self._faulted = False
        self.hardware_commands = getattr(port, 'hardware_commands', True) is not False
        if self.hardware_commands is feedback.simulation:
            raise DriverTransportError('Gripper command port and feedback simulation modes disagree')
        self._terminal_history = None
        self.last_receipt = None

    def measured_quiescent(self):
        """Extend the measured terminal dwell without waiting/reinitializing."""
        if self._active or not self.last_receipt or not self.last_receipt.measured_quiescent or not self._terminal_history:
            return False
        try:
            self._sample(self._terminal_history, after=0.)
            return self._settled(self._terminal_history)
        except DriverTransportError:
            return False

    def _settled(self, history):
        b, error = self.feedback.bounds, self.feedback.calibration.error_m
        if len(history) < 2 or history[-1].acquired_lower_s-history[0].acquired_upper_s < b.settle_duration_s:
            return False
        apertures = [state.aperture_m for state in history]
        if max(apertures)-min(apertures)+2*error > b.stationary_span_m:
            return False
        # An acceleration bound converts independent position secants to a
        # conservative instantaneous velocity upper bound. No velocity sensor
        # or force measurement is invented from GripperState.
        latest = history[-1]
        estimates = []
        for older in list(history)[:-1]:
            minimum_dt = latest.acquired_lower_s-older.acquired_upper_s
            maximum_dt = latest.acquired_upper_s-older.acquired_lower_s
            if minimum_dt > 0:
                estimates.append((abs(latest.aperture_m-older.aperture_m)+2*error)/minimum_dt
                                 +b.maximum_acceleration_m_s2*maximum_dt)
        since_capture = self.clock()-latest.acquired_lower_s
        return (bool(estimates) and since_capture >= 0
                and min(estimates)+b.maximum_acceleration_m_s2*since_capture <= b.stationary_velocity_m_s)

    def _sample(self, history, *, after, ownership=None):
        state = self.feedback.read(ownership=ownership)
        if state.acquired_lower_s <= after:
            return state
        if not history or state.sequence > history[-1].sequence:
            if history and state.acquired_lower_s-history[-1].acquired_upper_s > self.feedback.bounds.state_max_age_s:
                history.clear()
            history.append(state)
            # Retain a complete dwell plus one sample, bounded memory.
            while len(history) > 2 and state.acquired_lower_s-history[1].acquired_upper_s > self.feedback.bounds.settle_duration_s:
                history.popleft()
        return state

    async def _halt(self, reason):
        b = self.feedback.bounds
        started = self.clock()
        self.port.publish_stop(self.ownership.owner_id, reason)
        history = deque(maxlen=4096)
        acknowledged = False
        last = None
        until = started+b.stop_timeout_s
        while self.clock() < until:
            try:
                last = self._sample(history, after=started)
                acknowledged = (last.halt_active and last.halt_generation == 1
                    and last.halt_exchange_sequence == last.sequence and not last.owned
                    and last.acquired_lower_s > started)
                if acknowledged and self._settled(history):
                    self._terminal_history = history
                    return last, True, True
            except DriverTransportError:
                history.clear()
            await asyncio.sleep(b.poll_period_s)
        return last, False, acknowledged

    async def stop(self, reason='supervisor'):
        """Halt a previously commanded idle gateway; active work uses its cancel event."""
        if self._active:
            raise DriverTransportError('Cancel and drain active gripper execution before idle stop')
        if self.last_receipt is None or self._faulted:
            return self.last_receipt
        self._faulted = True
        worker = asyncio.create_task(self._halt(reason))
        while True:
            try:
                last, settled, acknowledged = await asyncio.shield(worker)
                break
            except asyncio.CancelledError:
                if worker.done(): raise
        self.last_receipt = GripperReceipt('cancelled',reason,last.aperture_m if last else None,
            settled,acknowledged,self.owns_control() is True,self.feedback.calibration.digest)
        return self.last_receipt

    async def execute(self, command, permit, *, cancel_event=None):
        with self._lock:
            if self._active or self._faulted:
                raise DriverTransportError('Gripper gateway busy or fault latched')
            self._active = True
        sent = False
        try:
            b, calibration = self.feedback.bounds, self.feedback.calibration
            expected = calibrated_command(calibration, b, aperture_m=command.aperture_m,
                speed_fraction=command.speed_fraction, current_ceiling_fraction=command.current_ceiling_fraction)
            if command != expected:
                raise DriverTransportError('Gripper command does not match its exact admitted calibration/wire values')
            if not self.port.ready():
                raise DriverTransportError('Gripper command or stop endpoint unavailable')
            history = deque(maxlen=4096)
            started = self.clock()
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise DriverTransportError('Gripper command cancelled')
                if self.owns_control() is not True or self.admission_check(permit, command) is not True:
                    raise DriverTransportError('Gripper permit or shared machine ownership revoked')
                state = self._sample(history, after=started, ownership=self.ownership)
                if abs(state.current_a) > b.current_abort_a:
                    raise DriverTransportError('Gripper measured current exceeds commissioned abort limit')
                if self.clock()-started >= b.command_timeout_s:
                    raise DriverTransportError('Gripper command deadline exceeded')
                if sent and self._settled(history) and all(
                        abs(s.aperture_m-command.aperture_m)+calibration.error_m <= b.target_tolerance_m for s in history):
                    self._terminal_history = history
                    self.last_receipt = GripperReceipt('succeeded', 'Measured aperture reached and settled', state.aperture_m,
                                                       True, False, True, calibration.digest)
                    return self.last_receipt
                # Best-effort/latest-wins upstream setpoints have no ACK. Repeat
                # the identical command while checking the permit every cycle.
                sent = True  # a publishing exception may still have transmitted
                self.port.publish(command, self.ownership)
                await asyncio.sleep(b.poll_period_s)
        except (Exception, asyncio.CancelledError) as exc:
            if not sent:
                raise
            self._faulted = True
            stop = asyncio.create_task(self._halt(str(exc) or 'Gripper task cancelled'))
            while True:
                try:
                    last, settled, acknowledged = await asyncio.shield(stop)
                    break
                except asyncio.CancelledError:
                    if stop.done():
                        raise
                    continue  # retain execution/ownership until bounded stop drains
            self.last_receipt = GripperReceipt('cancelled' if isinstance(exc, asyncio.CancelledError) or 'cancelled' in str(exc) else 'fault',
                str(exc) or 'Gripper task cancelled', last.aperture_m if last else None,
                settled, acknowledged, self.owns_control() is True, self.feedback.calibration.digest)
            return self.last_receipt
        finally:
            with self._lock:
                self._active = False


class RosGripperPort:
    """Clients only; no driver startup, ownership acquisition, or reset path."""
    hardware_commands = True
    def __init__(self, node, *, command_topic, stop_topic):
        from rammp_arm_interfaces.msg import GripperSetpoint
        from rammp_common_interfaces.msg import EStop
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        for name in (command_topic, stop_topic):
            if not isinstance(name, str) or not name.startswith('/') or len(name) > 256:
                raise DriverTransportError('Explicit absolute gripper endpoints required')
        self.node, self._stop_type = node, EStop
        self.publisher = node.create_publisher(GripperSetpoint, command_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE))
        self.stop_publisher = node.create_publisher(EStop, stop_topic,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE))

    def ready(self):
        return self.publisher.get_subscription_count() > 0 and self.stop_publisher.get_subscription_count() > 0

    def publish(self, command, ownership):
        self.publisher.publish(make_ros_gripper_setpoint(command, ownership))

    def publish_stop(self, owner_id, reason):
        msg = self._stop_type()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.engaged, msg.source, msg.reason = True, owner_id, reason[:1024]
        self.stop_publisher.publish(msg)

    def close(self):
        self.node.destroy_publisher(self.publisher)
        self.node.destroy_publisher(self.stop_publisher)
