"""Read-only, uncommissioned observations of the pinned Kinova ROS state API.

These are diagnostic records, not motion-state certificates. The upstream driver
stamps publications, exposes no hardware acquisition counter/time here, and does
not publish joint acceleration. Repeated or changing values cannot establish
acquisition freshness. No robot command, hardware capability or TF is created.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
from numbers import Real
import threading
import time

from .rolling import MotionError


DRIVER_SOURCE_COMMIT = "4aa7e5e1c2a649f522f1995edcc85b66041a6492"
ARM_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 8))
GRIPPER_JOINT_NAME = "robotiq_85_left_knuckle_joint"
EE_MESSAGE_TYPES = ("kinova_gen3_interfaces/msg/EeState", "rammp_arm_interfaces/msg/EeState")


class DriverStateError(MotionError):
    """Missing, malformed or stale diagnostic driver observations."""


def _finite(values, label):
    try:
        result = tuple(values)
    except TypeError as exc:
        raise DriverStateError(f"{label} must be a numeric sequence") from exc
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in result):
        raise DriverStateError(f"{label} must be finite numbers")
    return tuple(float(v) for v in result)


def _receipt(value):
    stamp = _finite((value,), "receipt time")[0]
    if stamp < 0:
        raise DriverStateError("receipt time must be nonnegative")
    return stamp


def _header(message):
    try:
        stamp, frame = message.header.stamp, message.header.frame_id
        sec, ns = stamp.sec, stamp.nanosec
    except AttributeError as exc:
        raise DriverStateError("driver state requires a ROS header") from exc
    if (type(sec) is not int or type(ns) is not int or not 0 <= sec < 2**31
            or not 0 <= ns < 10**9 or sec == ns == 0):
        raise DriverStateError("driver publication stamp must be positive ROS time")
    if not isinstance(frame, str) or len(frame) > 256:
        raise DriverStateError("invalid driver frame name")
    return sec * 10**9 + ns, frame


@dataclass(frozen=True)
class JointObservation:
    source_stamp_ns: int
    received_at_monotonic_s: float
    source_frame_id: str
    source_joint_names: tuple[str, ...]
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    effort_nm: tuple[float, ...] | None
    gripper_knuckle_position_rad: float | None

    @property
    def joint_names(self):
        return ARM_JOINT_NAMES


@dataclass(frozen=True)
class EeObservation:
    source_stamp_ns: int
    received_at_monotonic_s: float
    source_frame_id: str
    position_m: tuple[float, ...]
    quaternion_xyzw: tuple[float, ...]
    linear_velocity_m_s: tuple[float, ...]
    angular_velocity_rad_s: tuple[float, ...]


def joint_observation(message, *, received_at_monotonic_s):
    """Convert sensor_msgs/JointState using names, never positional guesses.

    The optional Robotiq knuckle is separate from the seven arm joints. Its NaN
    velocity/effort are unavailable, not zero, and its angle is not aperture.
    """
    stamp, frame = _header(message)
    try:
        names = tuple(message.name)
        positions, velocities, efforts = map(tuple, (message.position, message.velocity, message.effort))
    except (AttributeError, TypeError) as exc:
        raise DriverStateError("malformed JointState fields") from exc
    if (len(names) not in (7, 8) or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
            or set(names) not in (set(ARM_JOINT_NAMES), set(ARM_JOINT_NAMES) | {GRIPPER_JOINT_NAME})):
        raise DriverStateError("joint identity must be joint_1..joint_7 plus the optional pinned gripper joint")
    if len(positions) != len(names) or len(velocities) != len(names) or len(efforts) not in (0, len(names)):
        raise DriverStateError("JointState array dimensions disagree or arm velocity is missing")
    positions = _finite(positions, "joint positions")
    order = tuple(names.index(n) for n in ARM_JOINT_NAMES)
    arm_velocity = _finite((velocities[i] for i in order), "arm velocity")
    arm_effort = _finite((efforts[i] for i in order), "arm effort") if efforts else None
    gripper_position = None
    if GRIPPER_JOINT_NAME in names:
        i = names.index(GRIPPER_JOINT_NAME)
        gripper_position = positions[i]
        # NaN is the pinned publisher's documented unavailable value. Infinity
        # and strings are malformed, and finite values do not gain semantics.
        for value in (velocities[i], *((efforts[i],) if efforts else ())):
            if isinstance(value, bool) or not isinstance(value, Real) or math.isinf(value):
                raise DriverStateError("invalid optional gripper velocity/effort")
    return JointObservation(stamp, _receipt(received_at_monotonic_s), frame, names,
                            tuple(positions[i] for i in order), arm_velocity, arm_effort,
                            gripper_position)


def ee_observation(message, *, received_at_monotonic_s):
    """Convert kinova_gen3_interfaces/EeState without inventing its empty frame.

    Pose/twist are model-derived and use the driver's LOCAL_WORLD_ALIGNED
    convention. That convention does not establish model or TF agreement.
    """
    stamp, frame = _header(message)
    try:
        p, q, v, w = message.pose.position, message.pose.orientation, message.twist.linear, message.twist.angular
        position = _finite((p.x, p.y, p.z), "EE position")
        quaternion = _finite((q.x, q.y, q.z, q.w), "EE quaternion")
        linear = _finite((v.x, v.y, v.z), "EE linear velocity")
        angular = _finite((w.x, w.y, w.z), "EE angular velocity")
    except AttributeError as exc:
        raise DriverStateError("malformed EeState fields") from exc
    if abs(sum(v*v for v in quaternion) - 1.) > 1e-6:
        raise DriverStateError("EE quaternion is not unit length")
    return EeObservation(stamp, _receipt(received_at_monotonic_s), frame, position, quaternion, linear, angular)


@dataclass(frozen=True)
class DriverStateObservation:
    joints: JointObservation
    ee: EeObservation

    @property
    def motion_ready(self):
        return False

    def diagnostic(self):
        return {
            "scope": "uncommissioned_driver_publications",
            "expected_driver_source_commit": DRIVER_SOURCE_COMMIT,
            "installed_driver_source_verified": False,
            "joint_names": ARM_JOINT_NAMES,
            "joints": asdict(self.joints), "ee": asdict(self.ee),
            "source_stamp_semantics": "driver_publication_time; hardware_acquisition_time_unknown",
            "joint_acceleration_rad_s2": None,
            "gripper_aperture_m": None, "gripper_presence_verified": False,
            "effort_provenance_verified": False,
            "ee_pose_provenance": "upstream_model_fk; model_and_frame_agreement_unverified",
            "hardware_acquisition_freshness_verified": False,
            "clock_mapping_verified": False, "motion_ready": False,
        }

    def require_motion_state(self):
        # rolling.JointState requires q/dq/ddq. Do not invent ddq=0 or claim
        # measured hold just because this publication reports small velocity.
        raise DriverStateError("motion state unavailable: hardware acquisition age, acceleration and model agreement are unverified")


class DriverStateBuffer:
    """Bounded same-publication-tick pairing with local receipt-age checks.

    max_receipt_age_s is an explicit diagnostic transport timeout, never a
    commissioned source-age or stopping limit. Source clock resets require a new
    buffer/session; late or duplicate messages cannot refresh old observations.
    """

    def __init__(self, *, max_receipt_age_s, queue_size=8, clock=time.monotonic):
        age = _finite((max_receipt_age_s,), "receipt age limit")[0]
        if age <= 0 or type(queue_size) is not int or not 1 <= queue_size <= 128 or not callable(clock):
            raise DriverStateError("invalid driver state buffer configuration")
        self.max_receipt_age_s, self.queue_size, self.clock = age, queue_size, clock
        self._lock = threading.RLock()
        self._queues = {name: OrderedDict() for name in ("joint", "ee")}
        self._last_source = {name: -1 for name in self._queues}
        self._last_receipt = {name: -1. for name in self._queues}
        self._latest = None
        self.rejected = 0
        self.dropped = 0
        self.last_error = "no paired driver state"

    def _ingest(self, stream, message):
        with self._lock:
            try:
                receipt = _receipt(self.clock())
                convert = joint_observation if stream == "joint" else ee_observation
                observation = convert(message, received_at_monotonic_s=receipt)
                stamp = observation.source_stamp_ns
                if stamp <= self._last_source[stream] or receipt < self._last_receipt[stream]:
                    raise DriverStateError("duplicate/out-of-order driver publication or reversed receipt clock")
                self._last_source[stream], self._last_receipt[stream] = stamp, receipt
                queue = self._queues[stream]
                queue[stamp] = observation
                while len(queue) > self.queue_size:
                    queue.popitem(last=False)
                    self.dropped += 1
                if all(stamp in q for q in self._queues.values()):
                    paired = DriverStateObservation(self._queues["joint"][stamp], self._queues["ee"][stamp])
                    for q in self._queues.values():
                        for old in tuple(q):
                            if old <= stamp:
                                q.pop(old)
                    self._latest, self.last_error = paired, ""
                    return paired
                return None
            except DriverStateError as exc:
                self._latest = None
                for queue in self._queues.values():
                    queue.clear()
                self.rejected += 1
                self.last_error = str(exc)
                raise

    def ingest_joint(self, message):
        return self._ingest("joint", message)

    def ingest_ee(self, message):
        return self._ingest("ee", message)

    def snapshot(self):
        with self._lock:
            if self._latest is None:
                raise DriverStateError(self.last_error or "no paired driver state")
            now = _receipt(self.clock())
            for observation in (self._latest.joints, self._latest.ee):
                age = now - observation.received_at_monotonic_s
                if not 0 <= age <= self.max_receipt_age_s:
                    raise DriverStateError("driver publication receipt is stale or clock reversed")
            return self._latest


class RosDriverStateSource:
    """Attach two read-only subscriptions to a caller-owned, already spun node.

    Topic names are explicit deployment inputs. Constructing this object creates
    no publisher, client, timer, driver process or robot connection. The caller
    owns the node/executor; this adapter never starts a driver or acquires control.
    """

    def __init__(self, node, *, buffer, joint_topic, ee_topic,
                 ee_message_type=EE_MESSAGE_TYPES[0]):
        if not isinstance(buffer, DriverStateBuffer) or any(not isinstance(t, str) or not t for t in (joint_topic, ee_topic)) or joint_topic == ee_topic:
            raise DriverStateError("explicit distinct state topics and a DriverStateBuffer are required")
        if ee_message_type not in EE_MESSAGE_TYPES:
            raise DriverStateError("unsupported driver EeState interface; select an inspected package explicitly")
        try:
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState as RosJointState
            if ee_message_type == EE_MESSAGE_TYPES[0]:
                from kinova_gen3_interfaces.msg import EeState
            else:
                # The installed RAMMP split keeps Header/Pose/Twist semantics.
                # This only selects a subscription type, never a control API.
                from rammp_arm_interfaces.msg import EeState
        except ImportError as exc:
            raise DriverStateError(f"source ROS Humble and the selected {ee_message_type} build before subscribing") from exc
        self.node, self.buffer = node, buffer
        self._subscriptions = []
        try:
            for kind, topic, ingest in ((RosJointState, joint_topic, buffer.ingest_joint),
                                        (EeState, ee_topic, buffer.ingest_ee)):
                def callback(message, ingest=ingest):
                    try:
                        ingest(message)
                    except DriverStateError:
                        # Buffer rejects/invalidate atomically and retains a
                        # diagnostic reason; malformed data never kills spinning.
                        pass
                self._subscriptions.append(node.create_subscription(kind, topic, callback, qos_profile_sensor_data))
        except Exception:
            self.close()
            raise

    def close(self):
        for subscription in self._subscriptions:
            self.node.destroy_subscription(subscription)
        self._subscriptions.clear()

    def snapshot(self):
        return self.buffer.snapshot()
