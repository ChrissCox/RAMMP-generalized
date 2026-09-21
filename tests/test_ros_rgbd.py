import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from rammp_adl.contracts import digest
from rammp_adl.perception import CaptureRegistry, PerceptionError, deproject
from rammp_adl.perception.ros_rgbd import (decode_image, camera_info, RgbdSynchronizer,
    RgbdCalibration, MetricRgbdAdapter, RosRgbdSource, capture_ros_rgbd, rectified_intrinsics)


def header(stamp=10., frame="optical"):
    ns = round(stamp*10**9)
    return NS(stamp=NS(sec=ns//10**9, nanosec=ns % 10**9), frame_id=frame)


def message(stream, stamp=10., frame="optical"):
    if stream.endswith("info"):
        return NS(header=header(stamp, frame), width=4, height=3, distortion_model="plumb_bob",
                  k=[4., 0., 2., 0., 4., 1., 0., 0., 1.], r=np.eye(3).ravel().tolist(),
                  p=[4., 0., 2., 0., 0., 4., 1., 0., 0., 0., 1., 0.], d=[0.]*5,
                  binning_x=0, binning_y=0, roi=NS(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))
    depth = stream == "depth"
    data = np.full((3, 4), 250, dtype="<u2") if depth else np.full((3, 4, 3), 100, np.uint8)
    return NS(header=header(stamp, frame), width=4, height=3, encoding="16UC1" if depth else "rgb8",
              is_bigendian=0, step=8 if depth else 12, data=data.tobytes())


def pair(stamp=10., depth_stamp=None):
    sync = RgbdSynchronizer(clock=lambda: stamp)
    result = None
    for stream in sync.streams:
        result = sync.add(stream, message(stream, depth_stamp if stream.startswith("depth") and depth_stamp is not None else stamp))
    return result


def calibration(source):
    return RgbdCalibration("test-camera", "optical", "test-cal", source.metadata["camera_info_digests"]["rgb"],
                           source.metadata["camera_info_digests"]["depth"], "test-alignment", "test-rectification", "test-clock")


class ImageDecodingTests(unittest.TestCase):
    def test_big_endian_padded_depth_and_invalid_pixels(self):
        msg = message("depth")
        msg.is_bigendian, msg.step = 1, 10
        row = np.array([0, 1000, 250, 65535], dtype=">u2").tobytes()+b"xx"
        msg.data = row*3
        result = decode_image(msg, depth=True)
        self.assertTrue(np.isnan(result[0, 0]))
        np.testing.assert_allclose(result[0, 1:], [1., .25, 65.535])
        self.assertFalse(result.flags.writeable)

    def test_float_depth_is_metres_with_invalid_values_preserved_as_unknown(self):
        msg = message("depth")
        msg.encoding, msg.step = "32FC1", 16
        msg.data = np.array([[.4, 0., -1., np.inf]]*3, dtype="<f4").tobytes()
        result = decode_image(msg, depth=True)
        self.assertAlmostEqual(result[0, 0], .4)
        self.assertTrue(np.isnan(result[:, 1:]).all())

    def test_bgr_color_conversion_and_padding(self):
        msg = message("rgb")
        msg.encoding, msg.step = "bgr8", 14
        msg.data = (bytes([1, 2, 3])*4+b"xx")*3
        result = decode_image(msg)
        np.testing.assert_array_equal(result[0, 0], [3, 2, 1])
        self.assertFalse(result.flags.writeable)

    def test_malformed_encoding_stride_length_dimensions_and_budget_rejected(self):
        for field, value in (("encoding", "mono16"), ("step", 1), ("data", b"x"), ("height", 0), ("is_bigendian", 2), ("height", 10**9)):
            with self.subTest(field=field, value=value):
                msg = message("depth")
                setattr(msg, field, value)
                with self.assertRaises(PerceptionError):
                    decode_image(msg, depth=True)

    def test_camera_info_does_not_claim_raw_distortion_is_rectified(self):
        msg = message("rgb_info")
        msg.d[0] = .1
        info = camera_info(msg)
        self.assertEqual(info["d"][0], .1)
        self.assertNotIn("rectified", info)
        msg.k[0] = float("nan")
        with self.assertRaises(PerceptionError):
            camera_info(msg)


class PairingTests(unittest.TestCase):
    def test_arbitrary_arrival_order_pairs_once_and_keeps_source_time(self):
        sync = RgbdSynchronizer(clock=lambda: 999.)
        for key in ("depth_info", "rgb_info", "depth"):
            self.assertIsNone(sync.add(key, message(key)))
        result = sync.add("rgb", message("rgb"))
        self.assertEqual(result.metadata["source_stamps_ns"]["rgb"], 10**10)
        self.assertEqual(result.metadata["received_at_monotonic_s"]["rgb"], 999.)
        self.assertFalse(result.metadata["metric_geometry_validated"])
        self.assertTrue(all(not queue for queue in sync.queues.values()))
        with self.assertRaises(PerceptionError):
            sync.add("rgb", message("rgb"))

    def test_unsynchronized_and_missing_info_never_pair(self):
        sync = RgbdSynchronizer(clock=lambda: 10.)
        for key in sync.streams:
            result = sync.add(key, message(key, 10.1 if key.startswith("depth") else 10.))
            self.assertIsNone(result)
        self.assertEqual(sync.pairs, 0)

    def test_frame_or_resolution_conflict_rejected(self):
        for mismatch in ("frame", "width"):
            sync = RgbdSynchronizer(clock=lambda: 10.)
            for key in ("rgb", "depth", "rgb_info"):
                sync.add(key, message(key))
            msg = message("depth_info")
            if mismatch == "frame":
                msg.header.frame_id = "other_optical"
            else:
                msg.width = 5
            with self.assertRaises(PerceptionError):
                sync.add("depth_info", msg)
            self.assertEqual(sync.rejected, 1)

    def test_zero_or_reset_clock_is_rejected_without_receipt_substitution(self):
        sync = RgbdSynchronizer(clock=lambda: 10.)
        sync.add("rgb", message("rgb"))
        for stamp in (0., 9., 10.):
            with self.assertRaises(PerceptionError):
                sync.add("rgb", message("rgb", stamp))
        self.assertEqual(sync.last_stamp["rgb"], 10**10)

    def test_queues_are_bounded_and_expire_disconnected_streams(self):
        now = [10.]
        sync = RgbdSynchronizer(queue_size=2, max_residence_s=.2, clock=lambda: now[0])
        for stamp in (10., 10.01, 10.02):
            sync.add("rgb", message("rgb", stamp))
        self.assertEqual(len(sync.queues["rgb"]), 2)
        now[0] = 11.
        sync.add("depth", message("depth", 10.02))
        self.assertEqual(len(sync.queues["rgb"]), 0)
        self.assertEqual(sync.dropped, 3)

    def test_camera_info_changes_are_visible_in_signature(self):
        a = camera_info(message("rgb_info"))
        b = camera_info(message("rgb_info", 11.))
        self.assertEqual(digest(a), digest(b))
        b["k"][0] = 8.
        self.assertNotEqual(digest(a), digest(b))

    def test_nonfinite_or_unbounded_queue_settings_rejected(self):
        for kwargs in ({"max_skew_s": float("nan")}, {"queue_size": 1000}, {"max_residence_s": 0}):
            with self.assertRaises(PerceptionError):
                RgbdSynchronizer(**kwargs)


class MetricAdmissionTests(unittest.TestCase):
    def adapter(self, sample=None, **kwargs):
        sample = sample or pair()
        return MetricRgbdAdapter(None, calibration=calibration(sample), clock_mapper=kwargs.pop("clock_mapper", lambda t: (t, .001)),
                                 max_age_s=.2, clock=kwargs.pop("clock", lambda: 10.01), **kwargs)

    def test_calibrated_pair_feeds_existing_registry_and_depth_geometry(self):
        sample = pair()
        adapter = self.adapter(sample)
        frame = adapter.admit(sample)
        registry = CaptureRegistry(clock=lambda: 10.01)
        registry.register(frame)
        self.assertEqual(frame.image_id, sample.capture_id)
        self.assertEqual(frame.calibration_id, "test-cal")
        np.testing.assert_allclose(deproject(frame, (2, 1), min_depth_m=.1, max_depth_m=.5), [0, 0, .25])
        self.assertFalse(hasattr(frame, "orientation_xyzw"))

    def test_diagnostic_cannot_be_registered_or_admitted_without_calibration(self):
        with self.assertRaises(AttributeError):
            CaptureRegistry(clock=lambda: 10.).register(pair())
        with self.assertRaises(PerceptionError):
            MetricRgbdAdapter(None, calibration=None, clock_mapper=lambda t: (t, 0), max_age_s=.2)
        with self.assertRaises(PerceptionError):
            replace(calibration(pair()), alignment_evidence_id="")

    def test_changed_intrinsics_or_frame_do_not_refresh_metric_buffer(self):
        sample = pair()
        adapter = self.adapter(sample)
        for key, value in (("frame_id", "other"), ("width", 5), ("d", [.1]*5)):
            meta = sample.metadata
            meta["rgb_info"][key] = value
            changed = replace(sample, metadata_json=json.dumps(meta))
            with self.assertRaises(PerceptionError):
                adapter.admit(changed)
            self.assertIsNone(adapter.buffer.latest)

    def test_clock_uncertainty_staleness_future_and_mapped_skew_rejected(self):
        for mapper in (lambda t: (9., .001), lambda t: (11., .001), lambda t: (t, .5), lambda t: (t, float("nan")), lambda t: (t, -.1)):
            adapter = self.adapter(clock_mapper=mapper)
            with self.assertRaises(PerceptionError):
                adapter.admit(pair())
            self.assertIsNone(adapter.buffer.latest)

    def test_both_exposures_contribute_to_uncertainty(self):
        sample = pair(10., depth_stamp=10.01)
        adapter = self.adapter(sample, clock=lambda: 10.02, max_timestamp_uncertainty_s=.005)
        with self.assertRaises(PerceptionError):
            adapter.admit(sample)
        self.assertIsNone(adapter.buffer.latest)
        accepted = self.adapter(sample, clock=lambda: 10.02).admit(sample)
        self.assertAlmostEqual(accepted.timestamp_uncertainty_s, .011)

    def test_repeat_capture_rejected(self):
        adapter = self.adapter()
        adapter.admit(pair())
        with self.assertRaises(PerceptionError):
            adapter.admit(pair())

    def test_binning_roi_and_stereo_projection_require_explicit_extension(self):
        original = camera_info(message("rgb_info"))
        for change in (lambda x: x.update(binning_x=2), lambda x: x["roi"].update(width=2),
                       lambda x: x["p"].__setitem__(3, .1), lambda x: x["k"].__setitem__(0, 0.)):
            info = copy.deepcopy(original)
            change(info)
            with self.assertRaises(PerceptionError):
                rectified_intrinsics(info)


class DiagnosticTests(unittest.TestCase):
    def test_failure_writes_report_and_never_mints_calibration(self):
        with tempfile.TemporaryDirectory() as temp, patch("rammp_adl.perception.ros_rgbd.RosRgbdSource", side_effect=PerceptionError("no camera")):
            result = capture_ros_rgbd(output_dir=Path(temp)/"capture")
            self.assertEqual(result["samples_received"], 0)
            self.assertEqual(result["status"], "incomplete_or_unavailable")
            self.assertFalse(result["metric_geometry_validated"])
            self.assertTrue((Path(temp)/"capture/report.json").exists())

    def test_local_npz_has_arrays_and_original_provenance_without_egress(self):
        sample = pair()
        class Source:
            closed = False
            def __init__(self, **kwargs): pass
            def capture(self, timeout_ms): return sample
            def close(self): Source.closed = True
            def statistics(self): return {"paired": 1}
        with tempfile.TemporaryDirectory() as temp, patch("rammp_adl.perception.ros_rgbd.RosRgbdSource", Source):
            result = capture_ros_rgbd(output_dir=Path(temp)/"capture", samples=1)
            self.assertEqual(result["status"], "captured")
            self.assertEqual(result["frames_transmitted"], 0)
            with np.load(result["local_capture_file"], allow_pickle=False) as data:
                np.testing.assert_array_equal(data["rgb"], sample.rgb)
                self.assertEqual(json.loads(str(data["metadata_json"]))["source_stamps_ns"]["rgb"], 10**10)
            self.assertTrue(Source.closed)
            with self.assertRaises(FileExistsError):
                capture_ros_rgbd(output_dir=Path(temp)/"capture", samples=1)

    def test_local_only_guard_precedes_ros_import_and_construction(self):
        with patch.dict(os.environ, {"ROS_LOCALHOST_ONLY": "0", "CYCLONEDDS_URI": ""}):
            with self.assertRaisesRegex(PerceptionError, "ROS_LOCALHOST_ONLY"):
                RosRgbdSource(rgb_topic="/r", depth_topic="/d", rgb_info_topic="/ri", depth_info_topic="/di")


@unittest.skipUnless(importlib.util.find_spec("rclpy"), "ROS Humble is a separate integration dependency")
class RosSubscriberTests(unittest.TestCase):
    def test_live_ros_synthetic_messages_qos_warm_capture_and_close(self):
        # Local DDS transport with fabricated images; this is NOT camera evidence.
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image, CameraInfo
        from uuid import uuid4
        topics = {key+"_topic": "/test_rgbd_"+uuid4().hex+"/"+key for key in RgbdSynchronizer.streams}
        with patch.dict(os.environ, {"ROS_LOCALHOST_ONLY": "1", "CYCLONEDDS_URI": "", "ROS_DOMAIN_ID": "88"}):
            source = RosRgbdSource(**topics)
            context = Context()
            rclpy.init(args=[], context=context)
            node = Node("test_rgbd_publisher", context=context, enable_rosout=False, start_parameter_services=False)
            publishers = {}
            try:
                for key in RgbdSynchronizer.streams:
                    publishers[key] = node.create_publisher(CameraInfo if key.endswith("info") else Image, topics[key+"_topic"],
                                                           QoSProfile(depth=6, reliability=ReliabilityPolicy.BEST_EFFORT if key == "depth" else ReliabilityPolicy.RELIABLE))
                deadline = time.monotonic()+5
                while time.monotonic() < deadline and any(p.get_subscription_count() == 0 for p in publishers.values()):
                    time.sleep(.02)
                self.assertTrue(all(p.get_subscription_count() for p in publishers.values()))
                def publish(stamp):
                    for key, publisher in publishers.items():
                        original = message(key, stamp)
                        msg = CameraInfo() if key.endswith("info") else Image()
                        msg.header.stamp.sec, msg.header.stamp.nanosec = original.header.stamp.sec, original.header.stamp.nanosec
                        msg.header.frame_id = original.header.frame_id
                        for name, value in vars(original).items():
                            if name not in ("header", "roi"):
                                setattr(msg, name, value)
                        publisher.publish(msg)
                publish(10.)
                first = source.capture(2000)
                self.assertEqual(first.metadata["source_stamps_ns"]["rgb"], 10**10)
                publish(11.)
                deadline = time.monotonic()+2
                while time.monotonic() < deadline and source.statistics()["paired"] < 2:
                    time.sleep(.01)
                self.assertEqual(source.statistics()["paired"], 2)  # Worker stays warm without capture().
                second = source.capture(1000)
                self.assertNotEqual(first.capture_id, second.capture_id)
                outcome = []
                def blocked_capture():
                    try:
                        source.capture(2000)
                    except PerceptionError as exc:
                        outcome.append(str(exc))
                waiter = threading.Thread(target=blocked_capture)
                waiter.start()
                source.close()
                waiter.join(1)
                self.assertFalse(waiter.is_alive())
                self.assertEqual(outcome, ["RGB-D source closed"])
                self.assertFalse(source.thread.is_alive())
            finally:
                source.close()
                node.destroy_node()
                context.try_shutdown()


if __name__ == "__main__":
    unittest.main()
