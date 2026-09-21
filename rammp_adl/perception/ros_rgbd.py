"""NEW local ROS RGB-D ingestion, separate from metric calibration admission.

Only Image/CameraInfo subscriptions are constructed. No driver is launched, no
robot transport or provider is imported, and receipt time never becomes capture
time. The diagnostic pair cannot be registered as a calibrated CameraFrame.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from ..contracts import digest
from .cameras import RosCameraBuffer
from .geometry import Intrinsics, PerceptionError, positive_bounds, require_ids
from .ros_locality import require_local_imagery


def stamp_ns(message):
    stamp = message.header.stamp
    if type(stamp.sec) is not int or type(stamp.nanosec) is not int or stamp.sec < 0 or not 0 <= stamp.nanosec < 10**9:
        raise PerceptionError("Invalid ROS acquisition timestamp")
    value = stamp.sec * 10**9 + stamp.nanosec
    if value == 0:
        raise PerceptionError("Zero acquisition timestamp is unknown")
    require_ids(message.header.frame_id)
    return value


def decode_image(message, *, depth=False, max_bytes=32*1024*1024):
    """Honor Image row stride/endianness; REP-118 depth units, no hole filling."""
    formats = {"16UC1": ("u2", 1), "32FC1": ("f4", 1)} if depth else {"rgb8": ("u1", 3), "bgr8": ("u1", 3)}
    if message.encoding not in formats:
        raise PerceptionError("Unsupported depth or RGB Image encoding")
    kind, channels = formats[message.encoding]
    dtype = np.dtype((">" if message.is_bigendian else "<") + kind)
    width, height, step = message.width, message.height, message.step
    if any(type(v) is not int for v in (width, height, step)) or min(width, height) <= 0 or message.is_bigendian not in (0, 1):
        raise PerceptionError("Invalid Image dimensions or byte order")
    if step < width*channels*dtype.itemsize or not 0 < step*height <= max_bytes:
        raise PerceptionError("Image stride or byte budget invalid")
    raw = memoryview(message.data)
    if raw.nbytes != step*height:
        raise PerceptionError("Image byte length does not match its stride")
    shape = (height, width) if depth else (height, width, channels)
    strides = (step, dtype.itemsize) if depth else (step, channels*dtype.itemsize, dtype.itemsize)
    array = np.ndarray(shape, dtype=dtype, buffer=raw, strides=strides)
    if depth:
        array = array.astype(np.float64)
        if message.encoding == "16UC1":
            array *= .001
        array[~np.isfinite(array) | (array <= 0)] = np.nan
    else:
        array = array[..., ::-1].copy() if message.encoding == "bgr8" else array.copy()
    array.setflags(write=False)
    return array


def camera_info(message):
    """Preserve reported calibration; do not infer rectification or alignment."""
    stamp_ns(message)
    width, height = message.width, message.height
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise PerceptionError("Invalid CameraInfo dimensions")
    matrices = {name: list(map(float, getattr(message, name))) for name in ("k", "r", "p", "d")}
    if any(len(matrices[k]) != size for k, size in (("k", 9), ("r", 9), ("p", 12))) or len(matrices["d"]) > 16:
        raise PerceptionError("Invalid CameraInfo matrix sizes")
    if not all(math.isfinite(v) for values in matrices.values() for v in values):
        raise PerceptionError("Nonfinite CameraInfo")
    roi = {name: getattr(message.roi, name) for name in ("x_offset", "y_offset", "height", "width", "do_rectify")}
    return {"frame_id": message.header.frame_id, "width": width, "height": height,
            "distortion_model": message.distortion_model, **matrices,
            "binning_x": message.binning_x, "binning_y": message.binning_y, "roi": roi}


@dataclass(frozen=True)
class RgbdPair:
    """Locally paired observations, with no claim of calibrated metric validity."""
    capture_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    metadata_json: str

    @property
    def metadata(self):
        return json.loads(self.metadata_json)


class RgbdSynchronizer:
    """Bounded queues, one-to-one matching, strictly increasing source stamps.

    Called serially by the subscriber thread. CameraInfo must accompany images;
    static/unstamped calibration is deliberately not silently reused.
    """
    streams = ("rgb", "depth", "rgb_info", "depth_info")

    def __init__(self, *, max_skew_s=.02, queue_size=6, max_residence_s=.5, clock=time.monotonic):
        positive_bounds(max_skew_s, max_residence_s)
        if type(queue_size) is not int or not 1 <= queue_size <= 16:
            raise PerceptionError("RGB-D queue size must be in 1..16")
        self.max_skew_ns = round(max_skew_s*10**9)
        self.max_residence_s, self.clock = max_residence_s, clock
        self.queues = {key: deque(maxlen=queue_size) for key in self.streams}
        self.last_stamp = dict.fromkeys(self.streams, -1)
        self.received = dict.fromkeys(self.streams, 0)
        self.pairs = self.dropped = self.rejected = 0
        self.last_error = None

    def add(self, stream, message):
        if stream not in self.queues:
            raise PerceptionError("Unknown RGB-D stream")
        self.received[stream] += 1
        try:
            stamp = stamp_ns(message)
            if stamp <= self.last_stamp[stream]:
                raise PerceptionError("Duplicate, out-of-order or reset capture clock; recreate source after clock reset")
            payload = camera_info(message) if stream.endswith("info") else decode_image(message, depth=stream == "depth")
            now = self.clock()
            self.last_stamp[stream] = stamp
            for queue in self.queues.values():
                while queue and now-queue[0][1] > self.max_residence_s:
                    queue.popleft()
                    self.dropped += 1
            queue = self.queues[stream]
            if len(queue) == queue.maxlen:
                self.dropped += 1
            image_format = None if stream.endswith("info") else {"encoding": message.encoding,
                "step": message.step, "is_bigendian": message.is_bigendian}
            queue.append((stamp, now, message.header.frame_id, payload, image_format))
            return self._match()
        except PerceptionError as exc:
            self.rejected += 1
            self.last_error = str(exc)
            raise

    def _match(self):
        # Prefer newest usable observation instead of accumulating processing lag.
        for rgb in reversed(self.queues["rgb"]):
            selected = {"rgb": rgb}
            for key in ("depth", "rgb_info", "depth_info"):
                reference = selected["depth"] if key == "depth_info" else rgb
                candidates = [v for v in self.queues[key] if abs(v[0]-reference[0]) <= self.max_skew_ns]
                if not candidates:
                    break
                selected[key] = min(candidates, key=lambda v: abs(v[0]-reference[0]))
            if len(selected) != 4:
                continue
            # Consume every selected message once, including rejected pairs.
            for key, value in selected.items():
                queue = self.queues[key]
                while queue and queue[0][0] <= value[0]:
                    previous = queue.popleft()
                    if previous[0] != value[0]:
                        self.dropped += 1
            for key in ("rgb", "depth"):
                image, info = selected[key], selected[key+"_info"]
                if image[2] != info[2] or image[3].shape[:2] != (info[3]["height"], info[3]["width"]):
                    raise PerceptionError("Image/CameraInfo dimensions or optical frame disagree")
            meta = {"source_stamps_ns": {k: v[0] for k, v in selected.items()},
                    "received_at_monotonic_s": {k: v[1] for k, v in selected.items()},
                    "rgb_depth_skew_s": abs(rgb[0]-selected["depth"][0])/10**9,
                    "image_formats": {k: selected[k][4] for k in ("rgb", "depth")},
                    "rgb_info": selected["rgb_info"][3], "depth_info": selected["depth_info"][3],
                    "metric_geometry_validated": False, "capture_clock_validated": False,
                    "frames_transmitted": 0}
            meta["camera_info_digests"] = {k: digest(meta[k+"_info"]) for k in ("rgb", "depth")}
            self.pairs += 1
            return RgbdPair("ros-rgbd-"+uuid.uuid4().hex, rgb[3], selected["depth"][3], json.dumps(meta, allow_nan=False))
        return None

    def statistics(self):
        return {"received": dict(self.received), "paired": self.pairs,
                "dropped": self.dropped, "rejected": self.rejected, "last_error": self.last_error}


@dataclass(frozen=True)
class RgbdCalibration:
    """Trusted composition input, never filled from a diagnostic or model reply.

    Evidence IDs refer to externally verified registration/rectification and
    clock characterization. Intrinsic hashes detect changes, not accuracy.
    Camera-to-base/tool TF remains a separate CalibratedTransform requirement.
    """
    camera_id: str
    frame_id: str
    calibration_id: str
    rgb_info_digest: str
    depth_info_digest: str
    alignment_evidence_id: str
    rectification_evidence_id: str
    clock_evidence_id: str

    def __post_init__(self):
        require_ids(*self.__dict__.values())


def rectified_intrinsics(info):
    """Current seam supports full-size rectified, co-located pinhole streams."""
    if info["k"][0] <= 0 or info["binning_x"] not in (0, 1) or info["binning_y"] not in (0, 1):
        raise PerceptionError("Uncalibrated or binned CameraInfo requires a verified adapter")
    if any(info["roi"].values()):
        raise PerceptionError("ROI CameraInfo requires a verified adapter")
    p = np.asarray(info["p"]).reshape(3, 4)
    expected = np.array([[p[0, 0], 0., p[0, 2], 0.], [0., p[1, 1], p[1, 2], 0.], [0., 0., 1., 0.]])
    if not np.allclose(p, expected, rtol=0, atol=1e-9) or not np.allclose(np.asarray(info["r"]).reshape(3, 3), np.eye(3), rtol=0, atol=1e-9):
        raise PerceptionError("Translated/stereo/rotated projection requires a verified adapter")
    return Intrinsics(info["width"], info["height"], p[0, 0], p[1, 1], p[0, 2], p[1, 2])


class MetricRgbdAdapter:
    """Admit a pair into the existing perception loop only with explicit evidence."""
    def __init__(self, source, *, calibration, clock_mapper, max_age_s, max_skew_s=.02,
                 max_timestamp_uncertainty_s=.02, clock=time.monotonic):
        if not isinstance(calibration, RgbdCalibration):
            raise PerceptionError("Verified RGB-D calibration is required")
        positive_bounds(max_age_s, max_timestamp_uncertainty_s)
        self.source, self.calibration, self.clock = source, calibration, clock
        self.max_age_s, self.max_uncertainty = max_age_s, max_timestamp_uncertainty_s
        self.buffer = RosCameraBuffer(camera_id=calibration.camera_id, frame_id=calibration.frame_id,
                                      calibration_id=calibration.calibration_id, clock_mapper=clock_mapper,
                                      max_skew_s=max_skew_s)

    def admit(self, pair):
        meta, cal = pair.metadata, self.calibration
        expected = {"rgb": cal.rgb_info_digest, "depth": cal.depth_info_digest}
        for key in ("rgb", "depth"):
            info = meta[key+"_info"]
            if digest(info) != expected[key] or info["frame_id"] != cal.frame_id:
                raise PerceptionError("RGB-D calibration identity changed or frame mismatch")
        rgb_intrinsics, depth_intrinsics = (rectified_intrinsics(meta[k+"_info"]) for k in ("rgb", "depth"))
        if rgb_intrinsics != depth_intrinsics:
            raise PerceptionError("Registered depth and RGB projection disagree")
        stamps = meta["source_stamps_ns"]
        # Check freshness before publishing to the existing buffer. Mapping is
        # supplied by calibration; host receipt time is not used as a substitute.
        now = self.clock()
        for value in stamps.values():
            stamp, uncertainty = self.buffer.clock_mapper(value/10**9)
            if not all(math.isfinite(v) for v in (stamp, uncertainty)) or not 0 <= now-stamp <= self.max_age_s or not 0 <= uncertainty <= self.max_uncertainty:
                raise PerceptionError("RGB-D capture lacks fresh bounded clock mapping")
        previous = self.buffer.latest
        try:
            frame = self.buffer.ingest(rgb=pair.rgb, depth_m=pair.depth_m, intrinsics=rgb_intrinsics,
                                       rgb_stamp_s=stamps["rgb"]/10**9, depth_stamp_s=stamps["depth"]/10**9, aligned=True)
            frame.require_fresh(now, self.max_age_s, max_timestamp_uncertainty_s=self.max_uncertainty)
        except PerceptionError:
            self.buffer.latest = previous
            raise
        # Retain the source capture identity for registry/crop provenance.
        from dataclasses import replace
        frame = replace(frame, image_id=pair.capture_id)
        self.buffer.latest = frame
        return frame

    def capture(self, timeout_ms=1000):
        return self.admit(self.source.capture(timeout_ms))

    def close(self):
        self.source.close()


class RosRgbdSource:
    """Warm local subscriptions with a latest-pair mailbox, owned ROS context.

    Sensor-data QoS receives both best-effort and reliable publishers. capture()
    blocks only the perception worker; subscription callbacks keep running while
    consumers process a frame. close() wakes consumers and joins before teardown.
    """
    def __init__(self, *, rgb_topic, depth_topic, rgb_info_topic, depth_info_topic,
                 max_skew_s=.02, queue_size=6, max_residence_s=.5, reliability="best_effort",
                 domain_id=None, locality_policy=None):
        if domain_id is not None and (type(domain_id) is not int or not 0 <= domain_id <= 232):
            raise PerceptionError("Explicit ROS domain must be an integer in 0..232")
        # Confinement is resolved for this subscription's own domain, so a
        # read-only subscription elsewhere never relaxes imagery locality. A
        # deployment policy is the only alternative, and it is recorded.
        self.locality = require_local_imagery(domain_id, policy=locality_policy)
        topics = dict(rgb=rgb_topic, depth=depth_topic, rgb_info=rgb_info_topic, depth_info=depth_info_topic)
        require_ids(*topics.values())
        if any(not value.startswith("/") for value in topics.values()):
            raise PerceptionError("Explicit absolute camera topic names are required")
        if reliability not in ("best_effort", "reliable"):
            raise PerceptionError("Explicit best_effort or reliable subscription QoS is required")
        self.synchronizer = RgbdSynchronizer(max_skew_s=max_skew_s, queue_size=queue_size,
                                             max_residence_s=max_residence_s)
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from sensor_msgs.msg import Image, CameraInfo
        self.context = Context()
        rclpy.init(args=[], context=self.context, domain_id=domain_id)
        self.node = self.executor = None
        self.condition = threading.Condition()
        self.closed, self.latest, self.failure = False, None, None
        self.max_residence_s = max_residence_s
        self.overwritten = 0
        self.topics = topics
        try:
            self.node = Node("adl_rgbd_"+uuid.uuid4().hex[:8], context=self.context,
                             enable_rosout=False, start_parameter_services=False,
                             use_global_arguments=False)
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=queue_size,
                             reliability=ReliabilityPolicy.BEST_EFFORT if reliability == "best_effort" else ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            for key, topic in topics.items():
                self.node.create_subscription(CameraInfo if key.endswith("info") else Image,
                                              topic, lambda msg, key=key: self._receive(key, msg), qos)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self._spin, name="adl-rgbd", daemon=True)
            self.thread.start()
        except BaseException:
            if self.executor is not None:
                self.executor.shutdown()
            if self.node is not None:
                self.node.destroy_node()
            self.context.try_shutdown()
            raise

    def _receive(self, key, message):
        with self.condition:
            try:
                pair = self.synchronizer.add(key, message)
            except PerceptionError:
                return  # Rejection is recorded and cannot produce a metric frame.
            if pair is not None:
                self.overwritten += self.latest is not None
                self.latest = (time.monotonic(), pair)
                self.condition.notify_all()

    def _spin(self):
        try:
            while not self.closed:
                self.executor.spin_once(timeout_sec=.05)
        except Exception as exc:
            with self.condition:
                self.failure = str(exc)
                self.condition.notify_all()

    def capture(self, timeout_ms=1000):
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= 2000:
            raise PerceptionError("Capture timeout must be in 1..2000 ms")
        deadline = time.monotonic()+timeout_ms/1000
        with self.condition:
            while True:
                if self.closed or self.failure:
                    raise PerceptionError(self.failure or "RGB-D source closed")
                if self.latest is not None:
                    received, pair = self.latest
                    self.latest = None
                    if 0 <= time.monotonic()-received <= self.max_residence_s:
                        return pair
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise PerceptionError("Timed out waiting for a new synchronized RGB-D/CameraInfo pair")
                self.condition.wait(remaining)

    def statistics(self):
        with self.condition:
            return {**self.synchronizer.statistics(), "overwritten_pairs": self.overwritten,
                    "worker_failure": self.failure}

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
            self.latest = None
            self.condition.notify_all()
        try:
            self.thread.join(timeout=2.)
        except KeyboardInterrupt:
            pass                                    # a second interrupt must not abort teardown
        if self.thread.is_alive():
            raise PerceptionError("RGB-D worker did not stop; refusing concurrent ROS teardown")
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.try_shutdown()


def capture_ros_rgbd(*, output_dir, samples=15, timeout_s=20., pair_observer=None, **source_kwargs):
    """Bounded diagnostic; save one local RGB/depth pair, never upload it."""
    if type(samples) is not int or not 1 <= samples <= 300:
        raise PerceptionError("Diagnostic sample count must be in 1..300")
    positive_bounds(timeout_s)
    if timeout_s > 60:
        raise PerceptionError("Diagnostic deadline must be at most 60 seconds")
    if pair_observer is not None and not callable(pair_observer):
        raise PerceptionError("Pair observer must be a trusted local callable")
    destination = Path(output_dir)
    # Refuse overwrite so a repeated probe preserves its original evidence.
    destination.mkdir(parents=True, exist_ok=False)
    started, first_at, pair, count = time.monotonic(), None, None, 0
    skews, capture_times, last_error = [], [], None
    observations = []
    source = None
    try:
        source = RosRgbdSource(**source_kwargs)
        deadline = started+timeout_s
        while count < samples and time.monotonic() < deadline:
            try:
                pair = source.capture(max(1, min(1000, math.ceil((deadline-time.monotonic())*1000))))
            except PerceptionError as exc:
                last_error = str(exc)
                if source.failure:
                    break
                continue
            if first_at is None:
                first_at = time.monotonic()
            count += 1
            skews.append(pair.metadata["rgb_depth_skew_s"])
            capture_times.append(pair.metadata["source_stamps_ns"]["rgb"]/10**9)
            if pair_observer is not None:
                try:
                    observation = pair_observer(pair)
                    if not isinstance(observation, dict) or not isinstance(observation.get("status"), str) or observation.get("capture_id") != pair.capture_id:
                        raise PerceptionError("Observer must return a status and the current capture identity")
                    json.dumps(observation, allow_nan=False)
                except (ValueError, RuntimeError, TypeError) as exc:
                    observation = {"status": "observation_rejected", "capture_id": pair.capture_id, "detail": str(exc)}
                observations.append(observation)
    except (ImportError, RuntimeError, ValueError) as exc:
        last_error = str(exc)
    finally:
        if source is not None:
            source.close()
    report = {"status": "captured" if count == samples else "incomplete_or_unavailable",
              "samples_requested": samples, "samples_received": count,
              "elapsed_s": time.monotonic()-started,
              "first_pair_wait_s": None if first_at is None else first_at-started,
              "source_stamp_pair_rate_hz": (count-1)/(capture_times[-1]-capture_times[0]) if count > 1 else None,
              "maximum_rgb_depth_skew_s": max(skews) if skews else None,
              "topics": source_kwargs, "statistics": source.statistics() if source else None,
              "last_wait_error": last_error, "frames_transmitted": 0,
              "metric_geometry_validated": False, "capture_clock_validated": False,
              "physical_robot_commands_enabled": False,
              "validation_scope": "Local ROS RGB-D receipt/encoding/pairing only; no calibrated base geometry, task grounding, GPU planning or robot validation"}
    if pair_observer is not None:
        from collections import Counter
        report["observations"] = observations
        report["observation_summary"] = dict(Counter(item["status"] for item in observations))
        if observations:
            (destination/"observation.json").write_text(json.dumps(observations[-1], indent=2, allow_nan=False)+"\n")
    if pair is not None:
        valid = pair.depth_m[np.isfinite(pair.depth_m)]
        report.update(capture_id=pair.capture_id, last_pair=pair.metadata,
                      rgb_shape=list(pair.rgb.shape), depth_shape=list(pair.depth_m.shape),
                      positive_finite_depth_fraction=float(valid.size/pair.depth_m.size),
                      depth_quantiles_m=np.quantile(valid, [.05, .5, .95]).tolist() if valid.size else None,
                      local_capture_file=str(destination/"capture.npz"))
        np.savez(destination/"capture.npz", rgb=pair.rgb, depth_m=pair.depth_m,
                 metadata_json=json.dumps({"capture_id": pair.capture_id, **pair.metadata}, allow_nan=False))
    (destination/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report
