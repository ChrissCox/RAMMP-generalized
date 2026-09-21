"""Conversion tests also use generated ROS messages when their packages exist.

The fixture fallback keeps the offline suite independent of an upstream ROS
build. Live DDS verification is separate from these transport-independent tests.
"""
import copy
import json
import math
from types import SimpleNamespace as NS
import unittest

from rammp_adl.motion.driver_state import (
    ARM_JOINT_NAMES, DRIVER_SOURCE_COMMIT, GRIPPER_JOINT_NAME,
    DriverStateBuffer, DriverStateError, ee_observation, joint_observation,
)


def joint_message(stamp=10, *, gripper=False):
    try:
        from sensor_msgs.msg import JointState
        message = JointState()
    except ImportError:
        message = NS(header=NS(stamp=NS(sec=0, nanosec=0), frame_id=""))
    message.header.stamp.sec = stamp
    message.header.stamp.nanosec = 23
    message.name = list(ARM_JOINT_NAMES)
    message.position = [i / 10 for i in range(7)]
    message.velocity = [i / 100 for i in range(7)]
    message.effort = [float(i) for i in range(7)]
    if gripper:
        message.name += [GRIPPER_JOINT_NAME]
        message.position = list(message.position) + [.4]
        message.velocity = list(message.velocity) + [math.nan]
        message.effort = list(message.effort) + [math.nan]
    return message


def ee_message(stamp=10):
    try:
        from kinova_gen3_interfaces.msg import EeState
        message = EeState()
    except ImportError:
        try:
            from std_msgs.msg import Header
            from geometry_msgs.msg import Pose, Twist
            message = NS(header=Header(), pose=Pose(), twist=Twist())
        except ImportError:
            vector = lambda: NS(x=0., y=0., z=0.)
            message = NS(header=NS(stamp=NS(sec=0, nanosec=0), frame_id=""),
                         pose=NS(position=vector(), orientation=NS(x=0., y=0., z=0., w=0.)),
                         twist=NS(linear=vector(), angular=vector()))
    message.header.stamp.sec = stamp
    message.header.stamp.nanosec = 23
    message.pose.position.x = .4
    message.pose.orientation.w = 1.
    return message


class DriverStateTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.buffer = DriverStateBuffer(max_receipt_age_s=.25, clock=lambda: self.now)

    def pair(self, stamp=10):
        self.buffer.ingest_joint(joint_message(stamp))
        return self.buffer.ingest_ee(ee_message(stamp))

    def test_joint_identity_reorders_without_gripper_nan_contamination(self):
        message = joint_message(gripper=True)
        order = [7, 5, 3, 1, 6, 4, 2, 0]
        for name in ("name", "position", "velocity", "effort"):
            values = getattr(message, name)
            setattr(message, name, [values[i] for i in order])
        result = joint_observation(message, received_at_monotonic_s=99.)
        self.assertEqual(result.joint_names, ARM_JOINT_NAMES)
        self.assertEqual(result.source_joint_names, tuple(message.name))
        self.assertEqual(result.position_rad, tuple(i / 10 for i in range(7)))
        self.assertEqual(result.velocity_rad_s, tuple(i / 100 for i in range(7)))
        self.assertEqual(result.effort_nm, tuple(float(i) for i in range(7)))
        self.assertEqual(result.gripper_knuckle_position_rad, .4)
        message.position[0] = .7
        self.assertEqual(result.gripper_knuckle_position_rad, .4)

    def test_missing_duplicate_unknown_and_wrong_length_joints_fail(self):
        for mutate in (
            lambda m: setattr(m, "name", list(m.name)[:-1]),
            lambda m: setattr(m, "name", ["joint_1"] * 7),
            lambda m: setattr(m, "name", ["other"] + list(m.name)[1:]),
            lambda m: setattr(m, "velocity", []),
            lambda m: setattr(m, "position", list(m.position)[:-1]),
            lambda m: setattr(m, "effort", [1.]),
        ):
            with self.subTest(mutate=mutate):
                message = joint_message()
                mutate(message)
                with self.assertRaises(DriverStateError):
                    joint_observation(message, received_at_monotonic_s=1.)

    def test_arm_nonfinite_is_not_treated_like_optional_gripper_nan(self):
        for field in ("position", "velocity", "effort"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, value=value):
                    message = joint_message(gripper=True)
                    getattr(message, field)[0] = value
                    with self.assertRaises(DriverStateError):
                        joint_observation(message, received_at_monotonic_s=1.)
        message = joint_message()
        message.effort = []
        self.assertIsNone(joint_observation(message, received_at_monotonic_s=1.).effort_nm)

    def test_ee_keeps_unset_frame_and_rejects_bad_quaternion_or_twist(self):
        result = ee_observation(ee_message(), received_at_monotonic_s=19.)
        self.assertEqual(result.source_frame_id, "")
        self.assertEqual(result.position_m, (.4, 0., 0.))
        self.assertEqual(result.quaternion_xyzw, (0., 0., 0., 1.))
        for field in ("quaternion", "twist"):
            message = ee_message()
            if field == "quaternion":
                message.pose.orientation.w = 0.
            else:
                message.twist.angular.x = math.nan
            with self.assertRaises(DriverStateError):
                ee_observation(message, received_at_monotonic_s=1.)

    def test_headers_must_be_stamped_and_clocks_remain_separate(self):
        message = joint_message()
        result = joint_observation(message, received_at_monotonic_s=99999.)
        self.assertEqual(result.source_stamp_ns, 10_000_000_023)
        self.assertEqual(result.received_at_monotonic_s, 99999.)
        for sec, ns in ((0, 0), (-1, 0), (2, 10**9)):
            message = copy.deepcopy(message)
            message.header.stamp.sec, message.header.stamp.nanosec = sec, ns
            with self.assertRaises(DriverStateError):
                joint_observation(message, received_at_monotonic_s=1.)
        with self.assertRaises(DriverStateError):
            joint_observation(joint_message(), received_at_monotonic_s=math.nan)

    def test_pair_requires_exact_publication_tick_and_tracks_both_receipts(self):
        self.assertIsNone(self.buffer.ingest_ee(ee_message(10)))
        self.now += .05
        self.assertIsNone(self.buffer.ingest_joint(joint_message(11)))
        with self.assertRaises(DriverStateError):
            self.buffer.snapshot()
        self.now += .05
        result = self.buffer.ingest_ee(ee_message(11))
        self.assertEqual(result.joints.source_stamp_ns, result.ee.source_stamp_ns)
        self.assertAlmostEqual(result.joints.received_at_monotonic_s, 100.05)
        self.assertAlmostEqual(result.ee.received_at_monotonic_s, 100.10)
        self.assertIs(self.buffer.snapshot(), result)

    def test_repeated_values_and_new_publications_never_claim_motion_freshness(self):
        first = self.pair(10)
        self.now += .1
        latest = self.pair(11)
        self.assertEqual(first.joints.position_rad, latest.joints.position_rad)
        self.assertNotEqual(first.joints.source_stamp_ns, latest.joints.source_stamp_ns)
        report = latest.diagnostic()
        self.assertEqual(report["expected_driver_source_commit"], DRIVER_SOURCE_COMMIT)
        for key in ("motion_ready", "hardware_acquisition_freshness_verified", "clock_mapping_verified", "installed_driver_source_verified", "effort_provenance_verified"):
            self.assertIs(report[key], False)
        self.assertIsNone(report["joint_acceleration_rad_s2"])
        self.assertFalse(latest.motion_ready)
        with self.assertRaisesRegex(DriverStateError, "hardware acquisition age"):
            latest.require_motion_state()
        json.dumps(report, allow_nan=False)

    def test_receipt_expiry_includes_oldest_half_and_cannot_be_refreshed_by_one_topic(self):
        self.buffer.ingest_joint(joint_message(10))
        self.now += .2
        self.buffer.ingest_ee(ee_message(10))
        self.assertIsNotNone(self.buffer.snapshot())
        self.now += .1
        self.buffer.ingest_ee(ee_message(11))
        with self.assertRaisesRegex(DriverStateError, "stale"):
            self.buffer.snapshot()

    def test_duplicate_reversed_source_and_receipt_clocks_invalidate_cache(self):
        for mode in ("duplicate", "source_backwards", "receipt_backwards"):
            with self.subTest(mode=mode):
                self.setUp()
                self.pair(10)
                if mode == "receipt_backwards":
                    self.now -= 1.
                message = joint_message(9 if mode == "source_backwards" else 11 if mode == "receipt_backwards" else 10)
                with self.assertRaises(DriverStateError):
                    self.buffer.ingest_joint(message)
                with self.assertRaises(DriverStateError):
                    self.buffer.snapshot()
                self.assertEqual(self.buffer.rejected, 1)

    def test_bad_message_invalidates_old_pair_until_new_valid_pair_arrives(self):
        self.pair(10)
        malformed = ee_message(11)
        malformed.pose.position.x = math.nan
        with self.assertRaises(DriverStateError):
            self.buffer.ingest_ee(malformed)
        with self.assertRaises(DriverStateError):
            self.buffer.snapshot()
        self.now += .1
        result = self.pair(12)
        self.assertIs(self.buffer.snapshot(), result)
        self.assertEqual(self.buffer.last_error, "")

    def test_queue_bound_drops_unmatched_publications_without_fabricating_a_pair(self):
        buffer = DriverStateBuffer(max_receipt_age_s=1., queue_size=2, clock=lambda: self.now)
        for stamp in range(1, 6):
            buffer.ingest_joint(joint_message(stamp))
        self.assertEqual(buffer.dropped, 3)
        self.assertIsNone(buffer.ingest_ee(ee_message(1)))
        with self.assertRaises(DriverStateError):
            buffer.snapshot()
        self.assertIsNotNone(buffer.ingest_ee(ee_message(5)))

    def test_configuration_and_reversed_snapshot_clock_fail_closed(self):
        for args in ({"max_receipt_age_s": 0.}, {"max_receipt_age_s": math.inf},
                     {"max_receipt_age_s": .1, "queue_size": 0},
                     {"max_receipt_age_s": .1, "queue_size": True}):
            with self.assertRaises(DriverStateError):
                DriverStateBuffer(**args)
        self.pair(10)
        self.now -= 1.
        with self.assertRaises(DriverStateError):
            self.buffer.snapshot()
