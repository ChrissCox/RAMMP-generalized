"""Read-only camera adapters with explicit calibration and capture-clock mapping."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import subprocess
import sys
import uuid
import numpy as np

from .geometry import CameraFrame, Intrinsics, PerceptionError, positive_bounds, require_ids


def _probe_worker():
    result = {"realsense": [], "opencv_rgb_indices": [], "errors": []}
    if importlib.util.find_spec("pyrealsense2"):
        try:
            import pyrealsense2 as rs
            for device in rs.context().query_devices():
                result["realsense"].append({"name": device.get_info(rs.camera_info.name),
                                           "serial": device.get_info(rs.camera_info.serial_number)})
        except Exception as exc:
            result["errors"].append("RealSense: " + str(exc))
    if importlib.util.find_spec("cv2"):
        import cv2
        for index in range(2):
            camera = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            try:
                if camera.isOpened():
                    result["opencv_rgb_indices"].append(index)
            finally:
                camera.release()
    return result


def probe_cameras(*, timeout_s=8.0):
    """Bounded discovery in a child process; a hung vendor SDK cannot block control."""
    result = {"read_only": True, "frames_transmitted": 0,
              "sdk_available": {name: importlib.util.find_spec(name) is not None for name in ("pyrealsense2", "pyorbbecsdk", "cv2")}}
    command = [sys.executable, "-c", "import json; from rammp_adl.perception.cameras import _probe_worker; print(json.dumps(_probe_worker()))"]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=True)
        result.update(json.loads(process.stdout.strip().splitlines()[-1]))
        result["status"] = "devices_found" if result["realsense"] or result["opencv_rgb_indices"] else "no_local_camera_found"
    except subprocess.TimeoutExpired:
        result["status"] = "camera_probe_timeout"
    except (subprocess.CalledProcessError, ValueError, IndexError) as exc:
        result["status"], result["detail"] = "camera_probe_failed", str(exc)
    result["note"] = "No network camera endpoints are guessed. Configure camera roles and calibration explicitly."
    return result


def capture_preview(index, output_path, *, timeout_s=8.0):
    """Read a local RGB preview only; OpenCV receipt time is not metric capture time."""
    if type(index) is not int or not 0 <= index <= 16:
        raise PerceptionError("Preview requires an explicitly selected local camera index")
    script = """import cv2,json,sys,time
from pathlib import Path
index=int(sys.argv[1]); destination=Path(sys.argv[2])
camera=cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform=='win32' else cv2.CAP_ANY)
try:
    ok,frame=camera.read()
    if not ok or frame is None: raise RuntimeError('No RGB frame received')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(destination),frame): raise RuntimeError('Preview could not be saved')
    print(json.dumps({'status':'captured','width':frame.shape[1],'height':frame.shape[0], 'path':str(destination),'received_at_monotonic_s':time.monotonic(),'metric_geometry_validated':False,'frames_transmitted':0}))
finally:
    camera.release()
"""
    try:
        process = subprocess.run([sys.executable, "-c", script, str(index), str(output_path)],
                                 capture_output=True, text=True, timeout=timeout_s, check=True)
        return json.loads(process.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"status": "preview_timeout", "frames_transmitted": 0}
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return {"status": "preview_unavailable", "frames_transmitted": 0}


class RealSenseSource:
    """Aligned RGB-D using documented librealsense Python APIs.

    A calibrated mapper converts SDK capture timestamps into the runtime clock.
    No node publication/receipt timestamp is silently substituted for capture.
    """
    def __init__(self, *, serial, frame_id, calibration_id, clock_mapper, width=640, height=480, fps=30, max_rgb_depth_skew_s=.02):
        require_ids(serial, frame_id, calibration_id)
        positive_bounds(max_rgb_depth_skew_s)
        if not callable(clock_mapper):
            raise PerceptionError("RealSense requires explicit device, frame, calibration and timestamp mapper")
        import pyrealsense2 as rs
        self.rs, self.serial, self.frame_id, self.calibration_id = rs, serial, frame_id, calibration_id
        self.clock_mapper, self.max_skew_s = clock_mapper, max_rgb_depth_skew_s
        self.pipeline, config = rs.pipeline(), rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.profile = self.pipeline.start(config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.closed = False

    def capture(self, timeout_ms=1000):
        if self.closed or not 1 <= timeout_ms <= 2000:
            raise PerceptionError("Source closed or capture timeout outside bounds")
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise PerceptionError("Aligned RGB-D pair is missing")
        if color.get_frame_timestamp_domain() != depth.get_frame_timestamp_domain() or abs(color.get_timestamp()-depth.get_timestamp())/1000 > self.max_skew_s:
            raise PerceptionError("RGB/depth capture timestamps are incompatible")
        stamp, uncertainty = self.clock_mapper(color.get_timestamp()/1000, str(color.get_frame_timestamp_domain()))
        intrinsics = color.profile.as_video_stream_profile().intrinsics
        if intrinsics.model != self.rs.distortion.none:
            raise PerceptionError("Configured color stream is not rectified; use a verified rectification adapter")
        return CameraFrame(self.serial, "capture-"+uuid.uuid4().hex, np.asanyarray(color.get_data()),
            np.asanyarray(depth.get_data())*self.depth_scale,
            Intrinsics(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy),
            stamp, self.frame_id, self.calibration_id, uncertainty)

    def close(self):
        if not self.closed:
            self.pipeline.stop()
            self.closed = True


class RosCameraBuffer:
    """Driver-independent ingestion for calibrated Orbbec or RealSense ROS adapters.

    Topic names and clock conversion are supplied by deployment. Input arrays
    must already be rectified/aligned; a matching shape cannot establish that.
    """
    def __init__(self, *, camera_id, frame_id, calibration_id, clock_mapper, max_skew_s=.02):
        require_ids(camera_id, frame_id, calibration_id)
        positive_bounds(max_skew_s)
        if not callable(clock_mapper):
            raise PerceptionError("ROS camera requires a capture-clock mapper")
        self.camera_id, self.frame_id, self.calibration_id = camera_id, frame_id, calibration_id
        self.clock_mapper, self.max_skew_s = clock_mapper, max_skew_s
        self.latest = None

    def ingest(self, *, rgb, depth_m, intrinsics, rgb_stamp_s, depth_stamp_s, aligned):
        if aligned is not True or not all(math.isfinite(v) for v in (rgb_stamp_s, depth_stamp_s)) or abs(rgb_stamp_s-depth_stamp_s) > self.max_skew_s:
            raise PerceptionError("ROS color/depth pair is unaligned or unsynchronized")
        stamp, uncertainty = self.clock_mapper(rgb_stamp_s)
        depth_stamp, depth_uncertainty = self.clock_mapper(depth_stamp_s)
        if not all(math.isfinite(v) for v in (stamp, uncertainty, depth_stamp, depth_uncertainty)) or min(uncertainty, depth_uncertainty) < 0:
            raise PerceptionError("RGB/depth capture-clock mapping is invalid")
        if abs(stamp-depth_stamp) > self.max_skew_s:
            raise PerceptionError("Mapped RGB/depth capture timestamps are incompatible")
        # A frame's timestamp bound covers both exposures, not just RGB.
        uncertainty = max(uncertainty, abs(stamp-depth_stamp)+depth_uncertainty)
        frame = CameraFrame(self.camera_id, "capture-"+uuid.uuid4().hex, rgb, depth_m, intrinsics,
                            stamp, self.frame_id, self.calibration_id, uncertainty, aligned)
        if self.latest and frame.captured_at <= self.latest.captured_at:
            raise PerceptionError("Out-of-order camera capture")
        self.latest = frame
        return frame


class PerceptionLoop:
    def __init__(self, source, registry, on_frame, on_failure):
        self.source, self.registry = source, registry
        self.on_frame, self.on_failure = on_frame, on_failure
        self.stop_requested = asyncio.Event()

    async def run(self):
        from ..motion.leasing import drain_nonpreemptible
        capture = None
        try:
            while not self.stop_requested.is_set():
                try:
                    capture = asyncio.create_task(asyncio.to_thread(self.source.capture, 1000))
                    frame = await asyncio.shield(capture)
                    if self.stop_requested.is_set():
                        break
                    self.registry.register(frame)
                    result = self.on_frame(frame)
                    if asyncio.iscoroutine(result):
                        await result
                except (PerceptionError, RuntimeError) as exc:
                    result = self.on_failure(str(exc))
                    if asyncio.iscoroutine(result):
                        await result
                    # Failure remains visible to supervision; prevent a tight loop
                    # if a disconnected SDK throws without waiting for its timeout.
                    try:
                        await asyncio.wait_for(self.stop_requested.wait(), timeout=.1)
                    except asyncio.TimeoutError:
                        pass
        finally:
            # Vendor capture is non-preemptible: do not race SDK teardown with
            # an in-flight read. The source must honor its bounded timeout.
            if capture is not None and not capture.done():
                try:
                    await drain_nonpreemptible(capture)
                except Exception:
                    pass
            await asyncio.to_thread(self.source.close)

    def stop(self):
        self.stop_requested.set()
