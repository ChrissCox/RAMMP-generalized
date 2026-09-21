"""Bounded, token-preserving transport for the pinned Kinova trajectory action.

This sends a complete, locally admitted cuRobo trajectory; it does not plan,
acquire control, clear an e-stop, or implement rolling suffix replacement.
The caller supplies a trusted permit checker and acquisition-aware measured
state. Ordinary upstream JointState publications alone cannot provide that
evidence. A terminal action result alone never establishes quiescence.

The ROS action, arbitration status, release service and software-stop message
use the installed driver's rammp_arm_interfaces / rammp_common_interfaces split,
source-pinned with its guarded build in deployment/driver-hardware/prepare.py.
There is no automatic fallback to an older command interface. The small Python port
and admission/state callbacks below are NEW project-owned internal interfaces.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math
import threading
import time

from .driver_state import ARM_JOINT_NAMES
from .rolling import JointTrajectory, MotionError, TrajectoryPoint


class DriverTransportError(MotionError):
    pass


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise DriverTransportError(f"{label} must be finite and positive")


def _seven(values, label, *, positive=False):
    values = tuple(values)
    if (len(values) != 7 or any(type(v) not in (int, float) or not math.isfinite(v)
                              or (positive and v <= 0) for v in values)):
        raise DriverTransportError(f"{label} requires seven finite {'positive ' if positive else ''}values")
    return values


@dataclass(frozen=True)
class TransferredOwnership:
    """An explicit existing grant; construction never seizes another owner.

    The token is excluded from repr and every result/report. The composition
    must obtain this grant through an intentional operator/orchestrator transfer.
    The upstream AcquireControl service unconditionally seizes control, so this
    module deliberately contains no AcquireControl client or retry path.
    """
    owner_id: str
    token: bytes = field(repr=False)
    generation: int

    def __post_init__(self):
        if (not isinstance(self.owner_id, str) or not self.owner_id.strip()
                or len(self.owner_id) > 128 or type(self.token) is not bytes
                or len(self.token) != 16 or not any(self.token)
                or type(self.generation) is not int or not 1 <= self.generation < 2**64):
            raise DriverTransportError("explicit nonempty owner, nonzero 16-byte grant and generation required")


@dataclass(frozen=True)
class VerifiedMotionState:
    """Trusted composition output after source/clock/age verification.

    acquired_at is a conservative acquisition-time bound in the caller's local
    monotonic clock, NOT the ROS publication timestamp. This type intentionally
    has no acceleration field: the upstream feedback does not measure it.
    Admission must separately establish the planned stationary start boundary.
    """
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    acquired_at_monotonic_s: float
    received_at_monotonic_s: float
    sequence: int
    source_id: str

    def __post_init__(self):
        object.__setattr__(self, "position_rad", _seven(self.position_rad, "measured positions"))
        object.__setattr__(self, "velocity_rad_s", _seven(self.velocity_rad_s, "measured velocities"))
        times = (self.acquired_at_monotonic_s, self.received_at_monotonic_s)
        if (any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                or times[0] > times[1] or type(self.sequence) is not int or not 0 <= self.sequence < 2**64
                or not isinstance(self.source_id, str) or not self.source_id or len(self.source_id) > 256):
            raise DriverTransportError("invalid verified acquisition identity or clock bound")


@dataclass(frozen=True)
class TransportBounds:
    """Trusted commissioning bounds, all explicit and absent from model plans."""
    path_position_rad: tuple[float, ...]
    goal_position_rad: tuple[float, ...]
    start_position_rad: tuple[float, ...]
    stationary_velocity_rad_s: float
    stationary_position_span_rad: float
    state_max_age_s: float
    receipt_max_age_s: float
    poll_period_s: float
    send_timeout_s: float
    cancel_timeout_s: float
    stop_timeout_s: float
    settle_duration_s: float
    result_slack_s: float
    maximum_trajectory_s: float

    def __post_init__(self):
        for key in ("path_position_rad", "goal_position_rad", "start_position_rad"):
            object.__setattr__(self, key, _seven(getattr(self, key), key, positive=True))
        for key in self.__dataclass_fields__:
            if key not in ("path_position_rad", "goal_position_rad", "start_position_rad"):
                _positive(getattr(self, key), key)
        if (not self.poll_period_s < min(self.state_max_age_s, self.receipt_max_age_s,
                                        self.send_timeout_s, self.cancel_timeout_s, self.settle_duration_s)
                or self.cancel_timeout_s + self.settle_duration_s >= self.stop_timeout_s
                or self.maximum_trajectory_s > 3600 or self.stop_timeout_s > 60
                or self.send_timeout_s > 60 or self.result_slack_s > 60):
            raise DriverTransportError("transport timing bounds cannot sustain monitoring and bounded settling")


@dataclass(frozen=True)
class TransportReceipt:
    status: str
    reason: str
    trajectory_digest: str
    terminal_acknowledged: bool
    action_status: int | None
    driver_error_code: int | None
    measured_quiescent: bool
    ownership_retained: bool
    release_permitted: bool
    software_stop_published: bool


def canonical_ros_trajectory(trajectory):
    """Quantize times BEFORE validation so ROS encoding never silently retimes.

    Returns a new immutable trajectory and therefore a new digest when needed.
    No joint positions/derivatives or planner provenance are changed.
    """
    if not isinstance(trajectory, JointTrajectory) or trajectory.joint_names != ARM_JOINT_NAMES:
        raise DriverTransportError("driver requires exact joint_1 through joint_7 order")
    return JointTrajectory(trajectory.joint_names,
        tuple(TrajectoryPoint(round(p.time_s * 1e9) / 1e9, p.state) for p in trajectory.points),
        trajectory.provenance)


def _validate_wire_trajectory(trajectory, bounds):
    if not isinstance(trajectory, JointTrajectory) or trajectory.joint_names != ARM_JOINT_NAMES:
        raise DriverTransportError("driver requires exact joint_1 through joint_7 order")
    if len(trajectory.points) > 100000 or trajectory.duration_s > bounds.maximum_trajectory_s:
        raise DriverTransportError("trajectory exceeds the commissioned point/duration bound")
    for point in trajectory.points:
        if point.time_s != round(point.time_s * 1e9) / 1e9:
            raise DriverTransportError("canonicalize ROS nanosecond timing before trajectory validation")
    for point in (trajectory.points[0], trajectory.points[-1]):
        # Double-precision cuRobo quintic boundary evaluation has roundoff.
        # Preserve its derivatives exactly; this is a numerical zero screen,
        # not a commissioned moving-start tolerance or derivative rewriting.
        if (any(abs(value) > 1e-10 for value in point.state.velocity)
                or any(abs(value) > 1e-8 for value in point.state.acceleration)):
            raise DriverTransportError("this static transport requires planned stationary start and end boundaries")


class JointTrajectoryTransport:
    """Single command gateway using a trusted, nonblocking NEW internal port.

    Port methods are server_ready(), send_goal(trajectory, ownership, bounds),
    publish_stop(owner_id, reason), and release(token). Future/result/goal handles
    follow rclpy's source API. It must never start a driver or seize control.
    The caller keeps ROS spinning independently of this coroutine. The admission
    callback is a cached epoch/evidence gate invoked throughout execution; it is
    not a GPU/cloud operation. All geometric admission stays in the composition.
    """
    rolling_suffix_supported = False

    def __init__(self, *, port, ownership, admission_check, state_check, bounds, clock=time.monotonic):
        if (not isinstance(ownership, TransferredOwnership) or not isinstance(bounds, TransportBounds)
                or not all(callable(c) for c in (admission_check, state_check, clock))):
            raise DriverTransportError("trusted ownership, admission/state callbacks and bounds required")
        self.port, self.ownership, self.bounds = port, ownership, bounds
        self.admission_check, self.state_check, self.clock = admission_check, state_check, clock
        self._lock = threading.RLock()
        self._status, self._ownership_fault = None, ""
        self._ever_owned = self._running = self._released = self._faulted = False
        self._last_state = self._last_receipt = None
        self._software_stop_published = False

    def update_control_status(self, message):
        """The pinned reliable/transient-local status is ON CHANGE, not a heartbeat."""
        with self._lock:
            try:
                for key in ("arbitration_enabled", "estopped", "owned"):
                    if type(getattr(message, key)) is not bool:
                        raise DriverTransportError("malformed arbitration status")
                if (type(message.generation) is not int or not 0 <= message.generation < 2**64
                        or not isinstance(message.owner_id, str) or len(message.owner_id) > 128):
                    raise DriverTransportError("malformed arbitration identity")
                status = (message.arbitration_enabled, message.estopped, message.owned,
                          message.owner_id, message.generation)
                expected = (True, False, True, self.ownership.owner_id, self.ownership.generation)
                self._status = status
                if status == expected:
                    self._ever_owned = True
                elif self._ever_owned:
                    self._ownership_fault = "driver ownership revoked, replaced, disabled or stopped"
            except (AttributeError, DriverTransportError):
                self._status = None
                self._ownership_fault = "malformed driver ownership status"

    def owns_control(self):
        with self._lock:
            return (not self._released and not self._ownership_fault and self._status ==
                    (True, False, True, self.ownership.owner_id, self.ownership.generation))

    def _check_ownership(self):
        if not self.owns_control():
            raise DriverTransportError(self._ownership_fault or "transferred control grant not independently observed")

    def _state(self):
        now = self.clock()
        sample = self.state_check(now)
        if not isinstance(sample, VerifiedMotionState):
            raise DriverTransportError("trusted acquisition-aware measured state is unavailable")
        if (not 0 <= now - sample.acquired_at_monotonic_s <= self.bounds.state_max_age_s
                or not 0 <= now - sample.received_at_monotonic_s <= self.bounds.receipt_max_age_s):
            raise DriverTransportError("measured state acquisition or receipt is stale")
        if self._last_state is not None:
            old = self._last_state
            if (sample.source_id != old.source_id or sample.sequence < old.sequence
                    or sample.acquired_at_monotonic_s < old.acquired_at_monotonic_s
                    or sample.received_at_monotonic_s < old.received_at_monotonic_s
                    or (sample.sequence == old.sequence and sample != old)):
                raise DriverTransportError("measured state identity reset, regressed or refreshed without acquisition")
        self._last_state = sample
        return sample

    def _admitted(self, permit, trajectory):
        self._check_ownership()
        if self.admission_check(permit, trajectory) is not True:
            raise DriverTransportError("local trajectory permit was rejected or revoked")
        return self._state()

    async def _wait_future(self, future, deadline, *, checkpoint=None):
        # Do not cancel rclpy futures on timeout: late accepted goals need a
        # cancel, and a late release must not be mistaken for retained control.
        while not future.done():
            if checkpoint is not None:
                checkpoint()
            if self.clock() >= deadline:
                raise DriverTransportError("driver response deadline exceeded")
            await asyncio.sleep(self.bounds.poll_period_s)
        return future.result()

    def _engage_stop(self, reason):
        if not self._software_stop_published:
            self.port.publish_stop(self.ownership.owner_id, reason)
            self._software_stop_published = True
        self._faulted = True

    @staticmethod
    def _terminal(response):
        if (type(getattr(response, "status", None)) is not int or response.status not in (4, 5, 6)
                or type(getattr(getattr(response, "result", None), "error_code", None)) is not int):
            raise DriverTransportError("malformed or nonterminal driver result")
        return response

    async def _settled(self, after, deadline, *, target=None, require_ownership=False, checkpoint=None):
        anchor = None
        latest_sequence = None
        minimum = maximum = None
        while self.clock() < deadline:
            if require_ownership:
                self._check_ownership()
            if checkpoint is not None:
                checkpoint()
            try:
                sample = self._state()
            except Exception:
                anchor = None
                sample = None
            if sample is not None and sample.sequence != latest_sequence:
                latest_sequence = sample.sequence
                stopped = max(map(abs, sample.velocity_rad_s)) <= self.bounds.stationary_velocity_rad_s
                at_goal = target is None or all(abs(a-b) <= bound for a, b, bound in
                    zip(sample.position_rad, target, self.bounds.goal_position_rad))
                if sample.acquired_at_monotonic_s < after or not stopped or not at_goal:
                    anchor = None
                elif anchor is None:
                    anchor = sample.acquired_at_monotonic_s
                    minimum = maximum = sample.position_rad
                else:
                    minimum = tuple(min(a,b) for a,b in zip(minimum, sample.position_rad))
                    maximum = tuple(max(a,b) for a,b in zip(maximum, sample.position_rad))
                    if max(b-a for a,b in zip(minimum, maximum)) > self.bounds.stationary_position_span_rad:
                        anchor = sample.acquired_at_monotonic_s
                        minimum = maximum = sample.position_rad
                    elif sample.acquired_at_monotonic_s - anchor >= self.bounds.settle_duration_s:
                        return True
            await asyncio.sleep(self.bounds.poll_period_s)
        return False

    def _receipt(self, trajectory, *, status, reason, terminal=None, settled=False):
        acknowledged = terminal is not None
        owned = self.owns_control()
        receipt = TransportReceipt(status, reason, trajectory.digest, acknowledged,
            getattr(terminal, "status", None), getattr(getattr(terminal, "result", None), "error_code", None),
            settled, owned, acknowledged and settled and owned and not self._software_stop_published,
            self._software_stop_published)
        self._last_receipt = receipt
        return receipt

    async def execute(self, trajectory, *, permit, cancel_event=None):
        """No next goal is possible until this one has terminal AND stopped evidence."""
        with self._lock:
            if self._running or self._faulted or self._released:
                raise DriverTransportError("gateway busy, faulted or released; new motion refused")
            self._running = True
        cancel_event = cancel_event or asyncio.Event()
        work = asyncio.create_task(self._execute(trajectory, permit, cancel_event))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            cancel_event.set()
            # Keep the cleanup shielded even if callers repeatedly cancel.
            while not work.done():
                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError:
                    continue
            work.result()
            raise
        finally:
            self._running = False

    async def _execute(self, trajectory, permit, cancel_event):
        handle = send_future = terminal = None
        reason = ""
        def sending_checkpoint():
            if cancel_event.is_set():
                raise DriverTransportError("cancelled during trajectory admission")
            self._admitted(permit, trajectory)
        try:
            _validate_wire_trajectory(trajectory, self.bounds)
            sample = self._admitted(permit, trajectory)
            if (max(map(abs, sample.velocity_rad_s)) > self.bounds.stationary_velocity_rad_s
                    or any(abs(a-b) > tol for a,b,tol in zip(sample.position_rad,
                            trajectory.points[0].state.position, self.bounds.start_position_rad))):
                raise DriverTransportError("measured stationary start does not match admitted trajectory")
            if cancel_event.is_set():
                raise DriverTransportError("cancelled before dispatch")
            if not self.port.server_ready():
                raise DriverTransportError("trajectory action server unavailable")
            send_future = self.port.send_goal(trajectory, self.ownership, self.bounds)
            handle = await self._wait_future(send_future, self.clock()+self.bounds.send_timeout_s,
                checkpoint=sending_checkpoint)
            if not handle.accepted:
                return self._receipt(trajectory, status="rejected", reason="driver rejected trajectory")
            result_future = handle.get_result_async()
            deadline = self.clock()+trajectory.duration_s+self.bounds.result_slack_s
            while not result_future.done():
                self._admitted(permit, trajectory)
                if cancel_event.is_set():
                    raise DriverTransportError("cancelled")
                if self.clock() >= deadline:
                    raise DriverTransportError("trajectory terminal result deadline exceeded")
                await asyncio.sleep(self.bounds.poll_period_s)
            terminal = self._terminal(result_future.result())
            self._admitted(permit, trajectory)
            successful = terminal.status == 4 and terminal.result.error_code == 0
            if not successful:
                raise DriverTransportError("driver reported unsuccessful terminal result")
            settled = await self._settled(self.clock(), self.clock()+self.bounds.stop_timeout_s,
                target=trajectory.points[-1].state.position, require_ownership=True,
                checkpoint=lambda: self._admitted(permit, trajectory))
            self._admitted(permit, trajectory)
            if not settled:
                raise DriverTransportError("terminal result lacked fresh measured goal and quiescence")
            if cancel_event.is_set():
                return self._receipt(trajectory, status="cancelled", reason="cancelled during terminal verification",
                                     terminal=terminal, settled=True)
            return self._receipt(trajectory, status="succeeded", reason="", terminal=terminal, settled=True)
        except Exception as exc:
            reason = str(exc)
            if send_future is None:
                # No action request went on the wire; a rejected permit must
                # not stop somebody else's robot or release their resources.
                raise DriverTransportError(reason) from exc
            self._faulted = True
            return await self._stop_after_dispatch(trajectory, handle, send_future, terminal, reason, cancel_event)

    async def _stop_after_dispatch(self, trajectory, handle, send_future, terminal, reason, cancel_event):
        deadline = self.clock()+self.bounds.stop_timeout_s
        if not self.owns_control():
            self._engage_stop("ownership lost during trajectory: " + reason)
        if handle is None:
            # A send timeout is ambiguous. Prevent late acceptance from moving
            # the robot, and still cancel its exact handle if it later arrives.
            self._engage_stop("ambiguous trajectory admission: " + reason)
            def cancel_late(future):
                try:
                    late = future.result()
                    if late.accepted:
                        late.cancel_goal_async()
                except Exception:
                    pass
            send_future.add_done_callback(cancel_late)
        elif handle.accepted and terminal is None:
            try:
                cancel = await self._wait_future(handle.cancel_goal_async(),
                    min(deadline, self.clock()+self.bounds.cancel_timeout_s))
                wanted = bytes(handle.goal_id.uuid)
                if cancel.return_code != 0 or not any(bytes(g.goal_id.uuid) == wanted for g in cancel.goals_canceling):
                    raise DriverTransportError("exact goal cancellation was not acknowledged")
            except Exception:
                self._engage_stop("trajectory cancellation unresolved: " + reason)
            try:
                terminal = self._terminal(await self._wait_future(handle.get_result_async(),
                    deadline-self.bounds.settle_duration_s))
            except Exception:
                self._engage_stop("trajectory terminal acknowledgement unresolved: " + reason)
        settled = await self._settled(self.clock(), deadline)
        if not settled or terminal is None:
            self._engage_stop("measured stopping remains unresolved: " + reason)
        return self._receipt(trajectory,
            status="cancelled" if cancel_event.is_set() and settled and terminal is not None else "failed",
            reason=reason, terminal=terminal, settled=settled)

    async def release(self):
        """Explicit handback only after acknowledged terminal and verified stop."""
        if self._running or self._last_receipt is None or not self._last_receipt.release_permitted:
            raise DriverTransportError("ownership retained: terminal and measured quiescence are not both verified")
        self._check_ownership()
        sample = self._state()
        if max(map(abs, sample.velocity_rad_s)) > self.bounds.stationary_velocity_rad_s:
            raise DriverTransportError("ownership retained: latest measured state is moving")
        future = self.port.release(self.ownership.token)
        # Invalidate local authority before waiting: a timed-out service may
        # still release remotely and can never authorize a subsequent command.
        self._released = True
        response = await self._wait_future(future, self.clock()+self.bounds.send_timeout_s)
        if response.released is not True:
            raise DriverTransportError("driver refused explicit ownership release")
        return True


def make_ros_goal(trajectory, ownership, bounds):
    """Lossless q/dq/ddq mapping; upstream maps joints/tolerances by position."""
    _validate_wire_trajectory(trajectory, bounds)
    from control_msgs.msg import JointTolerance
    from trajectory_msgs.msg import JointTrajectoryPoint
    from rammp_arm_interfaces.action import ExecuteJointTrajectory
    goal = ExecuteJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINT_NAMES)
    for point in trajectory.points:
        message = JointTrajectoryPoint()
        message.positions = list(map(float, point.state.position))
        message.velocities = list(map(float, point.state.velocity))
        message.accelerations = list(map(float, point.state.acceleration))
        ns = round(point.time_s * 1e9)
        message.time_from_start.sec, message.time_from_start.nanosec = divmod(ns, 10**9)
        goal.trajectory.points.append(message)
    for field_name, values in (("path_tolerance", bounds.path_position_rad),
                               ("goal_tolerance", bounds.goal_position_rad)):
        for name, value in zip(ARM_JOINT_NAMES, values):
            tolerance = JointTolerance()
            tolerance.name, tolerance.position = name, float(value)
            getattr(goal, field_name).append(tolerance)
    ns = round(bounds.result_slack_s * 1e9)
    goal.goal_time_tolerance.sec, goal.goal_time_tolerance.nanosec = divmod(ns, 10**9)
    goal.control_mode = 0  # POSITION; no unverified impedance fallback.
    goal.preemption = 0    # QUEUE; never claim LATEST_WINS is a future suffix.
    goal.sender_id, goal.token = ownership.owner_id, list(ownership.token)
    return goal


class _RosPort:
    def __init__(self, node, *, action_name, release_service, stop_topic):
        from rclpy.action import ActionClient
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from rammp_arm_interfaces.action import ExecuteJointTrajectory
        from rammp_common_interfaces.srv import ReleaseControl
        from rammp_common_interfaces.msg import EStop
        self.node, self.release_type, self.stop_type = node, ReleaseControl, EStop
        self.action = ActionClient(node, ExecuteJointTrajectory, action_name)
        self.release_client = node.create_client(ReleaseControl, release_service)
        self.stop_pub = node.create_publisher(EStop, stop_topic, QoSProfile(depth=10,
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE))

    def server_ready(self):
        return self.action.server_is_ready() and self.stop_pub.get_subscription_count() > 0

    def send_goal(self, trajectory, ownership, bounds):
        return self.action.send_goal_async(make_ros_goal(trajectory, ownership, bounds))

    def publish_stop(self, owner_id, reason):
        message = self.stop_type()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.engaged, message.source, message.reason = True, owner_id, reason[:1024]
        self.stop_pub.publish(message)

    def release(self, token):
        request = self.release_type.Request()
        request.token = list(token)
        return self.release_client.call_async(request)

    def close(self):
        self.action.destroy()
        self.node.destroy_client(self.release_client)
        self.node.destroy_publisher(self.stop_pub)


class RosJointTrajectoryTransport(JointTrajectoryTransport):
    """Attach command clients to an already spun node; starts no driver process."""
    def __init__(self, node, *, ownership, admission_check, state_check, bounds,
                 action_name="/execute_joint_trajectory", control_topic="/control_status",
                 release_service="/release_control", stop_topic="/estop", clock=time.monotonic):
        for name in (action_name, control_topic, release_service, stop_topic):
            if not isinstance(name, str) or not name.startswith("/") or len(name) > 256:
                raise DriverTransportError("explicit absolute ROS endpoints required")
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from rammp_common_interfaces.msg import ControlStatus
        port = _RosPort(node, action_name=action_name, release_service=release_service, stop_topic=stop_topic)
        super().__init__(port=port, ownership=ownership, admission_check=admission_check,
                         state_check=state_check, bounds=bounds, clock=clock)
        self.node = node
        self.subscription = node.create_subscription(ControlStatus, control_topic, self.update_control_status,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def close(self):
        if self._running or (self._last_receipt is not None and not self._last_receipt.release_permitted
                             and not self._released):
            raise DriverTransportError("cannot discard transport while commanded stop or ownership is unresolved")
        self.node.destroy_subscription(self.subscription)
        self.port.close()
