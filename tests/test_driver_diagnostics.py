import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rammp_adl.motion.driver_diagnostics import DriverStateRecording, capture_driver_state
from rammp_adl.motion.driver_state import DriverStateError, RosDriverStateSource
from test_driver_state import joint_message, ee_message


class DriverRecordingTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.buffer = DriverStateRecording(max_pairs=3, max_receipt_age_s=1., clock=lambda: self.now)

    def pair(self, stamp, position=0.):
        j = joint_message(stamp, gripper=True)
        j.position[0] = position
        self.buffer.ingest_joint(j)
        self.now += .002
        self.buffer.ingest_ee(ee_message(stamp))

    def test_observed_statistics_do_not_fill_commissioning_fields(self):
        self.pair(10, 1.)
        self.now = 101.
        self.pair(11, 1.002)
        r = self.buffer.summary()
        self.assertAlmostEqual(r["paired_delivery_hz"], 1.)
        self.assertAlmostEqual(r["publication_gap_s"]["mean"], 1.)
        self.assertAlmostEqual(r["pair_receipt_skew_s"]["max"], .002)
        self.assertAlmostEqual(r["joints"][0]["position_span_rad"], .002)
        self.assertAlmostEqual(r["joints"][0]["position_stddev_rad"], .001)
        self.assertEqual(r["gripper_knuckle_position_rad"]["count"], 2)
        for key in ("hardware_acquisition_age_s", "measured_stopping_latency_s",
                    "measured_stopping_distance_rad", "measured_motion_tracking_error_rad", "commissioned_limits"):
            self.assertIsNone(r[key])
        self.assertFalse(r["motion_ready"])
        self.assertFalse(r["motion_commanded_by_recorder"])
        json.dumps(r, allow_nan=False)

    def test_bounds_preserve_first_samples_without_silently_rolling_history(self):
        for stamp in range(10, 15):
            self.now += .1
            self.pair(stamp)
        self.assertTrue(self.buffer.full)
        r = self.buffer.summary()
        self.assertEqual(r["paired_samples"], 3)
        self.assertEqual(r["messages_received"], {"joint": 5, "ee": 5})
        self.assertEqual(r["pairs_not_retained"], 2)
        self.assertEqual(r["last_publication_stamp_ns"], 12_000_000_023)

    def test_malformed_and_duplicate_publications_do_not_become_measurements(self):
        self.pair(10)
        with self.assertRaises(DriverStateError):
            self.buffer.ingest_joint(joint_message(10))
        bad = joint_message(11)
        bad.position[0] = math.nan
        with self.assertRaises(DriverStateError):
            self.buffer.ingest_joint(bad)
        r = self.buffer.summary()
        self.assertEqual(r["paired_samples"], 1)
        self.assertEqual(r["rejected_messages"], 2)
        self.assertIsNone(r["paired_delivery_hz"])
        self.assertIsNone(r["publication_gap_s"])

    def test_missing_pairs_and_missing_effort_remain_unknown(self):
        r = self.buffer.summary()
        self.assertEqual(r["paired_samples"], 0)
        self.assertNotIn("joints", r)
        self.buffer.ingest_ee(ee_message(9))
        j = joint_message(10)
        j.effort = []
        self.buffer.ingest_joint(j)
        self.buffer.ingest_ee(ee_message(10))
        r = self.buffer.summary()
        self.assertEqual(r["paired_samples"], 1)
        self.assertIsNone(r["joints"][0]["reported_effort_nm"])
        self.assertIsNone(r["gripper_knuckle_position_rad"])

    def test_end_of_recording_dropout_does_not_look_like_fresh_feedback(self):
        self.pair(10)
        self.now += .1
        self.pair(11)
        self.assertTrue(self.buffer.summary()['latest_pair_receipt_fresh'])
        self.now += 2.
        r = self.buffer.summary()
        self.assertEqual(r['paired_samples'], 2)
        self.assertFalse(r['latest_pair_receipt_fresh'])
        self.assertIsNone(r['latest_pair_receipt_age_s'])
        self.assertIn('stale', r['receipt_check_error'])

    def test_bad_configuration_fails_before_ros_or_filesystem_access(self):
        for cap in (1, 100001, True, math.inf):
            with self.assertRaises(DriverStateError):
                DriverStateRecording(max_pairs=cap, max_receipt_age_s=1.)
        common = dict(output_dir="/must-not-be-created", joint_topic="/joint_states", ee_topic="/ee_state",
                      ee_message_type="rammp_arm_interfaces/msg/EeState")
        for extra in ({"duration_s": math.nan}, {"duration_s": 601}, {"max_pairs": 0},
                      {"joint_topic": "relative"}, {"ee_message_type": "unreviewed/msg/EeState"}):
            with self.assertRaises(DriverStateError):
                capture_driver_state(**(common | extra))

    def test_explicit_interface_types_only_construct_two_subscriptions(self):
        subscriptions = []
        # A node stub with no command, timer or client methods: attempting any
        # such operation would fail this test. No actual ROS participants.
        node = NS(create_subscription=lambda *args: subscriptions.append(args) or len(subscriptions),
                  destroy_subscription=lambda sub: None)
        for package in ("kinova_gen3_interfaces", "rammp_arm_interfaces"):
            with self.subTest(package=package):
                subscriptions.clear()
                fake_ee = type("EeState", (), {})
                fake_joint = type("JointState", (), {})
                with patch.dict("sys.modules", {
                    "rclpy.qos": NS(qos_profile_sensor_data=object()),
                    "sensor_msgs.msg": NS(JointState=fake_joint),
                    package+".msg": NS(EeState=fake_ee),
                }):
                    source = RosDriverStateSource(node, buffer=self.buffer, joint_topic="/j", ee_topic="/e",
                                                  ee_message_type=package+"/msg/EeState")
                    self.assertEqual(len(subscriptions), 2)
                    self.assertIs(subscriptions[0][0], fake_joint)
                    self.assertIs(subscriptions[1][0], fake_ee)
                    source.close()

    def test_capture_owns_its_executor_context_and_saves_bounded_local_records(self):
        class Context:
            active = False
            def ok(self): return self.active

        class Node:
            def __init__(node, name, *, context, enable_rosout, start_parameter_services, use_global_arguments):
                self.assertFalse(enable_rosout or start_parameter_services or use_global_arguments)
                node.context, node.callbacks = context, []
            def create_subscription(node, kind, topic, callback, qos):
                node.callbacks.append(callback)
                return callback
            def get_publishers_info_by_topic(node, topic):
                return [NS(node_name="observed_driver", node_namespace="/", endpoint_gid=[1],
                           topic_type="sensor_msgs/msg/JointState" if topic == "/joint_states" else "rammp_arm_interfaces/msg/EeState")]
            def destroy_subscription(node, sub): pass
            def destroy_node(node): pass

        class Executor:
            def __init__(executor, *, context):
                executor.context, executor.tick = context, 10
            def add_node(executor, node):
                self.assertIs(executor.context, node.context)
                executor.node = node
            def spin_once(executor, *, timeout_sec):
                executor.node.callbacks[0](joint_message(executor.tick))
                executor.node.callbacks[1](ee_message(executor.tick))
                executor.tick += 1
            def shutdown(executor): pass

        ros = NS(init=lambda *, args, context: setattr(context, 'active', True),
                 shutdown=lambda *, context: setattr(context, 'active', False),
                 get_rmw_implementation_identifier=lambda: 'test_only')
        # Keep message helpers' normal fixture fallback by using import failures
        # for constructors, while supplying subscription type placeholders.
        no_message = lambda: (_ for _ in ()).throw(ImportError())
        modules = {'rclpy': ros, 'rclpy.context': NS(Context=Context), 'rclpy.node': NS(Node=Node),
                   'rclpy.executors': NS(SingleThreadedExecutor=Executor, ExternalShutdownException=type('Shutdown', (Exception,), {})),
                   'rclpy.qos': NS(qos_profile_sensor_data=object()),
                   'sensor_msgs.msg': NS(JointState=no_message),
                   'rammp_arm_interfaces.msg': NS(EeState=object)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', modules):
            output = Path(tmp)/'capture'
            report = capture_driver_state(output_dir=output, joint_topic='/joint_states', ee_topic='/ee_state',
                                          ee_message_type='rammp_arm_interfaces/msg/EeState', max_pairs=2)
            self.assertEqual(report['termination'], 'sample_cap_reached')
            self.assertEqual(report['status'], 'captured')
            self.assertFalse(report['motion_commanded_by_recorder'])
            self.assertEqual(len((output/'samples.jsonl').read_text().splitlines()), 2)
            self.assertEqual(json.loads((output/'report.json').read_text())['paired_samples'], 2)
            with self.assertRaises(FileExistsError):
                capture_driver_state(output_dir=output, joint_topic='/joint_states', ee_topic='/ee_state',
                                     ee_message_type='rammp_arm_interfaces/msg/EeState')


if __name__ == "__main__":
    unittest.main()
