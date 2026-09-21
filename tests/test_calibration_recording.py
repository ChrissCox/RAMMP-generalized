"""Raw calibration association checks and an isolated ROS sensor-only rehearsal."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from rammp_adl.contracts import digest
from rammp_adl.motion.driver_state import DriverStateObservation, ee_observation, joint_observation
from rammp_adl.perception.calibration_recording import (
    associate_receipts, camera_integrity, capture_calibration, read_recording_config, recording_config)
from rammp_adl.perception.fiducial import SingleMarkerObserver
from rammp_adl.perception.geometry import PerceptionError
from rammp_adl.perception.ros_locality import (
    CYCLONEDDS_RMW, local_imagery_config_path, render_cyclonedds_config)
from rammp_adl.perception.ros_rgbd import RosRgbdSource
from test_driver_state import joint_message, ee_message
from test_ros_rgbd import pair


ROOT = Path(__file__).resolve().parents[1]


def robot(stamp, receipt):
    return DriverStateObservation(joint_observation(joint_message(stamp), received_at_monotonic_s=receipt-.001),
                                  ee_observation(ee_message(stamp), received_at_monotonic_s=receipt))


def camera_record(receipt, capture_id="camera-1"):
    meta = pair(10.).metadata
    meta["received_at_monotonic_s"] = dict.fromkeys(meta["source_stamps_ns"], receipt)
    spec = SingleMarkerObserver().spec
    return {"metadata": meta, "observation": {
        "capture_id": capture_id, "status": "provisional_pose_candidates",
        "source_stamps_ns": copy.deepcopy(meta["source_stamps_ns"]),
        "marker_spec_digest": digest(spec), "rgb_info_digest": digest(meta["rgb_info"]),
        "camera_frame": meta["rgb_info"]["frame_id"],
        "pose_candidates": [{"branch": 0}, {"branch": 1}]}}


class AssociationTests(unittest.TestCase):
    def test_near_receipts_preserve_unknown_cross_clock_relation_and_both_branches(self):
        driver = [robot(800, 100.), robot(801, 100.02)]
        camera = camera_record(100.009)
        before = copy.deepcopy(camera)
        row, = associate_receipts(driver, [camera], max_receipt_gap_s=.03)
        self.assertEqual(row["status"], "receipt_bracketed")
        self.assertEqual(row["driver_indices"], [0, 1])
        self.assertEqual(row["camera_minus_driver_publication_ns"], -790_000_000_023)
        self.assertFalse(row["calibration_sample_admitted"])
        for key in ("base_from_wrist", "timestamp_uncertainty_s", "branch_resolution_evidence_id"):
            self.assertIsNone(row[key])
        self.assertEqual(camera, before)
        self.assertEqual(driver[0].ee.source_frame_id, "")

    def test_no_extrapolation_and_large_gaps_are_not_silent_nearest_matches(self):
        drivers = [robot(10, 1.), robot(11, 1.5)]
        rows = associate_receipts(drivers, [camera_record(t) for t in (.9, 1.01, 1.6)], max_receipt_gap_s=.1)
        self.assertEqual([r["status"] for r in rows], ["no_receipt_bracket", "receipt_gap_exceeded", "no_receipt_bracket"])
        self.assertEqual(associate_receipts([], [camera_record(1)], max_receipt_gap_s=.1)[0]["driver_indices"], [])

    def test_both_robot_receipts_must_be_close_and_marker_must_be_observed(self):
        drivers = [robot(10, 1.), robot(11, 1.02)]
        drivers[0] = replace(drivers[0], joints=replace(drivers[0].joints, received_at_monotonic_s=.5))
        self.assertEqual(associate_receipts(drivers, [camera_record(1.01)], max_receipt_gap_s=.1)[0]["status"], "receipt_gap_exceeded")
        rec = camera_record(1.01)
        rec["observation"]["status"] = "marker_not_observed"
        self.assertEqual(associate_receipts([robot(10, 1.), robot(11, 1.02)], [rec], max_receipt_gap_s=.1)[0]["status"], "marker_pose_unavailable")

    def test_source_integrity_failure_invalidates_associations_without_erasing_records(self):
        for drivers, integrity in (([robot(10, 1.), robot(11, 1.02)], False),
                                   ([robot(10, 1.02), robot(11, 1.)], True)):
            result = associate_receipts(drivers, [camera_record(1.01)], max_receipt_gap_s=.1, source_integrity_ok=integrity)
            self.assertEqual(result[0]["status"], "source_integrity_failed")

    def test_changed_intrinsics_duplicate_capture_and_clock_resets_latch_failure(self):
        first, second = camera_record(1.), camera_record(1.1, "camera-2")
        second["metadata"]["source_stamps_ns"] = {k: v+1 for k, v in second["metadata"]["source_stamps_ns"].items()}
        second["observation"]["source_stamps_ns"] = copy.deepcopy(second["metadata"]["source_stamps_ns"])
        self.assertIsNone(camera_integrity([first, second]))
        for mutation in (lambda r: r["metadata"]["rgb_info"]["k"].__setitem__(0, 9.),
                         lambda r: r["observation"].update(capture_id=first["observation"]["capture_id"]),
                         lambda r: r["metadata"]["received_at_monotonic_s"].update(rgb=.9),
                         lambda r: r["observation"].update(marker_spec_digest="changed"),
                         lambda r: r["observation"]["source_stamps_ns"].update(rgb=1)):
            changed = copy.deepcopy(second)
            mutation(changed)
            self.assertIsNotNone(camera_integrity([first, changed]))

    def test_invalid_config_and_bounds_do_not_create_outputs_or_ros_contexts(self):
        config = read_recording_config(ROOT/"config/calibration-recording.json")
        for mutation in (lambda c: c.update(schema_version=True), lambda c: c["driver"].update(domain_id=True),
                         lambda c: c["driver"].update(ee_message_type="unknown/msg/EeState"),
                         lambda c: c["cameras"].update(bad={}),
                         lambda c: c["cameras"]["d405"].update(command_topic="/command"),
                         lambda c: c["cameras"]["d405"].update(rgb_topic="../relative"),
                         lambda c: c["cameras"].update(driver=c["cameras"]["d405"]),
                         lambda c: c["cameras"]["orbbec"].update(c["cameras"]["d405"])):
            bad = copy.deepcopy(config)
            mutation(bad)
            with self.assertRaises((PerceptionError, ValueError)):
                recording_config(bad)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"not-created"
            for extra in ({"duration_s": float("nan")}, {"duration_s": 61}, {"max_camera_samples": True},
                          {"max_driver_pairs": 1}, {"max_receipt_gap_s": 0}, {"pose_label": ""}):
                with self.assertRaises(ValueError):
                    capture_calibration(**(dict(config=config, output_dir=output, pose_label="test") | extra))
                self.assertFalse(output.exists())
        with patch.dict(os.environ, {"ROS_LOCALHOST_ONLY": "1", "CYCLONEDDS_URI": ""}):
            for domain in (True, -1, 233):
                with self.assertRaises(PerceptionError):
                    RosRgbdSource(rgb_topic="/r", depth_topic="/d", rgb_info_topic="/ri", depth_info_topic="/di", domain_id=domain)

    def test_missing_sources_save_incomplete_evidence_and_refuse_overwrite(self):
        config = read_recording_config(ROOT/"config/calibration-recording.json")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ROS_LOCALHOST_ONLY": "1", "CYCLONEDDS_URI": ""}), \
                patch.dict("sys.modules", {"rclpy": None}):
            output = Path(tmp)/"recording"
            report = capture_calibration(config=config, output_dir=output, pose_label="no-ros", duration_s=.1)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["termination"], "source_error")
            self.assertEqual(report["calibration_samples_admitted"], 0)
            self.assertEqual((output/"driver.jsonl").read_text(), "")
            self.assertFalse(report["motion_commanded_by_recorder"])
            with self.assertRaises(FileExistsError):
                capture_calibration(config=config, output_dir=output, pose_label="repeat")


class TransportLocalityTests(unittest.TestCase):
    """Camera domains must stay on loopback while the driver domain may not be."""

    def setUp(self):
        self.config = read_recording_config(ROOT/"config/calibration-recording.json")
        self.camera_domains = sorted({s["domain_id"] for s in self.config["cameras"].values()})
        self.environ = {"ROS_LOCALHOST_ONLY": "0", "RMW_IMPLEMENTATION": CYCLONEDDS_RMW,
                        "CYCLONEDDS_URI": "file://"+str(local_imagery_config_path())}

    def test_unconfined_camera_domain_refuses_before_creating_any_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"ROS_LOCALHOST_ONLY": "0", "CYCLONEDDS_URI": ""}):
            output = Path(tmp)/"refused"
            with self.assertRaisesRegex(PerceptionError, "ROS_LOCALHOST_ONLY"):
                capture_calibration(config=self.config, output_dir=output, pose_label="unconfined")
            self.assertFalse(output.exists())

    def test_repository_document_admits_the_driver_domain_without_confining_it(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self.environ), \
                patch.dict("sys.modules", {"rclpy": None}):
            report = capture_calibration(config=self.config, output_dir=Path(tmp)/"recording",
                                         pose_label="per-domain", duration_s=.1)
        locality = report["transport_locality"]
        self.assertTrue(locality["ok"])
        self.assertTrue(locality["config_dependent"])
        self.assertEqual(locality["camera_domains"], self.camera_domains)
        self.assertEqual(locality["shared_domains"], [])
        driver_domain = str(self.config["driver"]["domain_id"])
        self.assertFalse(locality["domains"][driver_domain]["confined_to_loopback"])
        for domain in self.camera_domains:
            evidence = locality["domains"][str(domain)]
            self.assertTrue(evidence["confined_to_loopback"])
            self.assertEqual(evidence["interfaces"], ["address=127.0.0.1"])
            self.assertEqual(len(evidence["config_sources"][0]["sha256"]), 64)
        self.assertEqual(report["frames_uploaded"], 0)
        self.assertFalse(report["motion_commanded_by_recorder"])
        self.assertTrue(any("configuration claim" in line for line in report["limitations"]))
        json.dumps(report, allow_nan=False)


@unittest.skipUnless(importlib.util.find_spec("rclpy"), "ROS Humble is a separate integration dependency")
class CalibrationRosTests(unittest.TestCase):
    def test_a_cyclone_document_is_refused_under_another_middleware(self):
        config = read_recording_config(ROOT/"config/calibration-recording.json")
        config["driver"]["domain_id"] = 89
        for settings in config["cameras"].values():
            settings["domain_id"] = 90
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp)/"local.xml"
            document.write_text(render_cyclonedds_config(loopback_domains=[90]), encoding="utf-8")
            # ROS_LOCALHOST_ONLY still confines every context this test creates;
            # only the recorder's own middleware agreement check is exercised.
            with patch.dict(os.environ, {"ROS_LOCALHOST_ONLY": "1", "CYCLONEDDS_URI": "file://"+str(document),
                                         "RMW_IMPLEMENTATION": CYCLONEDDS_RMW}), \
                    patch("rclpy.get_rmw_implementation_identifier", return_value="rmw_fastrtps_cpp"):
                report = capture_calibration(config=config, output_dir=Path(tmp)/"mismatch",
                                             pose_label="middleware mismatch", duration_s=.1)
        self.assertEqual(report["termination"], "source_error")
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(any("rmw_fastrtps_cpp" in message for message in report["errors"]), report["errors"])
        self.assertEqual(report["calibration_samples_admitted"], 0)

    def test_interrupt_saves_partial_report_and_closes_owned_contexts(self):
        from rclpy.executors import SingleThreadedExecutor
        config = read_recording_config(ROOT/"config/calibration-recording.json")
        config["driver"]["domain_id"] = 89
        for settings in config["cameras"].values():
            settings["domain_id"] = 90
        sources = []
        def camera_source(**kwargs):
            source = RosRgbdSource(**kwargs)
            sources.append(source)
            return source
        class InterruptExecutor(SingleThreadedExecutor):
            def spin_once(self, timeout_sec=None):
                if threading.current_thread() is threading.main_thread():
                    raise KeyboardInterrupt()
                return super().spin_once(timeout_sec=timeout_sec)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ROS_LOCALHOST_ONLY": "1", "CYCLONEDDS_URI": ""}), \
                patch("rclpy.executors.SingleThreadedExecutor", InterruptExecutor), \
                patch("rammp_adl.perception.calibration_recording.RosRgbdSource", side_effect=camera_source):
            report = capture_calibration(config=config, output_dir=Path(tmp)/"partial", pose_label="interrupted")
            self.assertEqual(report["termination"], "interrupted")
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(len(sources), 2)
            self.assertTrue(all(not s.thread.is_alive() and not s.context.ok() for s in sources))
            self.assertFalse(report["errors"])

    def test_separate_domains_actual_detection_raw_archive_and_cleanup(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo, JointState
        from rammp_arm_interfaces.msg import EeState

        config = read_recording_config(ROOT/"config/calibration-recording.json")
        prefix = "/test_calibration_"+uuid.uuid4().hex
        config["driver"]["domain_id"] = 89
        for name, settings in [("driver", config["driver"]), *config["cameras"].items()]:
            if name != "driver":
                settings.update(domain_id=90, reliability="best_effort")
            for key in settings:
                if key.endswith("_topic"):
                    settings[key] = prefix+"/"+name+"/"+key
        observer = SingleMarkerObserver()
        gray = observer.cv2.aruco.generateImageMarker(observer.cv2.aruco.getPredefinedDictionary(observer.cv2.aruco.DICT_4X4_50), 0, 150)
        rgb = np.full((480, 640, 3), 255, dtype=np.uint8)
        rgb[165:315, 245:395] = gray[..., None]
        stop = threading.Event()
        contexts, nodes, errors, worker = [], [], [], None
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ROS_LOCALHOST_ONLY": "1", "CYCLONEDDS_URI": "", "ROS_DOMAIN_ID": "88"}):
            try:
                for domain in (89, 90):
                    context = Context()
                    rclpy.init(args=[], context=context, domain_id=domain)
                    contexts.append(context)
                    nodes.append(Node("calibration_fixture_"+str(domain), context=context,
                                      enable_rosout=False, start_parameter_services=False))
                jp = nodes[0].create_publisher(JointState, config["driver"]["joint_topic"], qos_profile_sensor_data)
                ep = nodes[0].create_publisher(EeState, config["driver"]["ee_topic"], qos_profile_sensor_data)
                cp = {name: {key: nodes[1].create_publisher(CameraInfo if "info" in key else Image, value, qos_profile_sensor_data)
                             for key, value in settings.items() if key.endswith("_topic")}
                      for name, settings in config["cameras"].items()}

                camera_messages = []
                # Construct large ROS byte sequences once; Python message setters
                # can otherwise stall the synthetic joint publisher under the GIL.
                for name, pubs in cp.items():
                    for key, pub in pubs.items():
                        info = "info" in key
                        msg = CameraInfo() if info else Image()
                        msg.header.frame_id = name+"_optical"
                        msg.width, msg.height = 640, 480
                        if info:
                            msg.k = [600., 0., 320., 0., 600., 240., 0., 0., 1.]
                            msg.r = np.eye(3).ravel().tolist()
                            msg.p = [600., 0., 320., 0., 0., 600., 240., 0., 0., 0., 1., 0.]
                            msg.d, msg.distortion_model = [0.]*5, "plumb_bob"
                        else:
                            depth = key.startswith("depth")
                            msg.encoding, msg.step = ("16UC1", 1280) if depth else ("rgb8", 1920)
                            msg.data = (np.full((480, 640), 500, dtype="<u2") if depth else rgb).tobytes()
                        camera_messages.append((name, pub, msg))

                def publish():
                    tick = 0
                    try:
                        while not stop.wait(.01):
                            tick += 1
                            j, original = joint_message(100), ee_message(100)
                            e = EeState()
                            e.header, e.pose, e.twist = original.header, original.pose, original.twist
                            j.header.stamp.nanosec = e.header.stamp.nanosec = tick*1000000
                            jp.publish(j)
                            ep.publish(e)
                            if tick % 10:
                                continue
                            for name, pub, msg in camera_messages:
                                # The earlier camera reaches its explicit sample
                                # cap while the second camera is still starting.
                                if name == "orbbec" and tick < 150:
                                    continue
                                msg.header.stamp.sec, msg.header.stamp.nanosec = 10, tick*1000000
                                pub.publish(msg)
                    except Exception as exc:
                        errors.append(exc)
                worker = threading.Thread(target=publish, daemon=True)
                worker.start()
                output = Path(tmp)/"recording"
                sources = []
                def camera_source(**kwargs):
                    source = RosRgbdSource(**kwargs)
                    sources.append(source)
                    return source
                with patch("rammp_adl.perception.calibration_recording.RosRgbdSource", side_effect=camera_source):
                    report = capture_calibration(config=config, output_dir=output, pose_label="synthetic stationary fixture",
                                                 duration_s=4., max_camera_samples=5, max_receipt_gap_s=.25)
                self.assertFalse(errors)
                self.assertEqual(report["status"], "recorded_unapproved", report)
                self.assertEqual(os.environ["ROS_DOMAIN_ID"], "88")
                self.assertTrue(all(not source.thread.is_alive() and not source.context.ok() for source in sources))
                self.assertGreater(report["driver"]["paired_samples"], 2)
                self.assertEqual(report["termination"], "camera_sample_caps_reached")
                self.assertGreater(report["cameras"]["d405"]["last_rgb_receipt_age_s"], 1.)
                self.assertLess(report["cameras"]["d405"]["last_rgb_receipt_age_s_at_camera_cutoff"], 1.)
                for name in cp:
                    observations = [json.loads(line) for line in (output/name/"observations.jsonl").read_text().splitlines()]
                    self.assertGreaterEqual(len(observations), 2)
                    self.assertTrue(all(len(o["observation"]["pose_candidates"]) == 2 for o in observations))
                    self.assertGreaterEqual(report["cameras"][name]["association_counts"].get("receipt_bracketed", 0), 2)
                    with np.load(output/name/"last-capture.npz", allow_pickle=False) as archive:
                        self.assertEqual(json.loads(str(archive["metadata_json"]))["capture_id"], observations[-1]["observation"]["capture_id"])
                for name, expected in report["files_sha256"].items():
                    self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), expected)
            finally:
                stop.set()
                if worker is not None:
                    worker.join(2)
                for node in nodes:
                    node.destroy_node()
                for context in contexts:
                    context.try_shutdown()


if __name__ == "__main__":
    unittest.main()
