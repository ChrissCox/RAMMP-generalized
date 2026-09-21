"""Keyframes: still-arm wrist captures selected for cloud grounding.

A keyframe is one RGB-D capture taken while the arm is verifiably still and
placed in the base frame by forward kinematics through the measured mount.
Captures are considered continuously; a keyframe is selected at most once per
``min_interval_s`` and only when the camera moved or the scene changed since
the previous one, or when a task asks for one. Every frame that leaves the
machine is downscaled, re-encoded and passed through the local face screen
first; without a screen result no frame is sent.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
import io
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..motion.kinematics import rotation_angle
from .geometry import PerceptionError, require_ids
from .images import ImageCrop



# The camera container stamps frames with the shared host system clock, and a
# frame cannot be received before it was exposed: the per-frame receipt latency
# therefore bounds the exposure instant to [stamp, receipt].


class StampReceiptClock:
    """Map source stamps on the host system clock to the world monotonic clock.

    ``note`` records each frame's receipt; the mapper then returns the stamp
    on the monotonic clock with that frame's receipt latency as its error and
    refuses stamps it never saw, stamps from another clock (negative latency)
    and frames that arrived later than ``max_latency_s``.
    """

    def __init__(self, *, max_latency_s=.15, floor_s=.002, capacity=256, wall=None, monotonic=None):
        import time
        self.max_latency_s, self.floor_s, self.capacity = float(max_latency_s), float(floor_s), int(capacity)
        self.wall = wall or time.time
        self.monotonic = monotonic or time.monotonic
        self._frames = OrderedDict()
        self._lock = threading.Lock()

    def note(self, stamp_ns, received_at_monotonic_s):
        if type(stamp_ns) is not int or stamp_ns <= 0 or not math.isfinite(received_at_monotonic_s):
            return
        offset = self.monotonic()-self.wall()
        mapped = stamp_ns/1e9+offset
        with self._lock:
            self._frames[stamp_ns] = (mapped, float(received_at_monotonic_s)-mapped)
            while len(self._frames) > self.capacity:
                self._frames.popitem(last=False)

    def __call__(self, stamp_s):
        if type(stamp_s) not in (int, float) or not math.isfinite(stamp_s):
            return math.nan, math.inf
        wanted = stamp_s*1e9
        with self._lock:
            nearest = min(self._frames, key=lambda key: abs(key-wanted), default=None)
            match = self._frames[nearest] if nearest is not None and abs(nearest-wanted) <= 2000 else None
        if match is None:
            return math.nan, math.inf
        mapped, latency = match
        if not -.005 <= latency <= self.max_latency_s:
            return math.nan, math.inf
        return mapped, max(latency, self.floor_s)


@dataclass(frozen=True)
class Keyframe:
    capture_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics_k: tuple
    frame_id: str
    stamp_ns: int
    received_at_s: float
    captured_at_s: float          # mapped onto the world clock
    uncertainty_s: float          # receipt-latency bound on that mapping
    base_from_camera: np.ndarray
    joints_rad: tuple
    reason: str
    selected_at_s: float

    @property
    def width(self):
        return int(self.rgb.shape[1])

    @property
    def height(self):
        return int(self.rgb.shape[0])

    @property
    def oldest_capture_s(self):
        return self.captured_at_s-self.uncertainty_s

    @property
    def camera_position_base(self):
        return np.asarray(self.base_from_camera[:3, 3], dtype=float)

    def project(self, point_base):
        """(u, v, depth) of a base-frame point in this keyframe, or None."""
        k = np.asarray(self.intrinsics_k, dtype=float).reshape(3, 3)
        camera = np.linalg.inv(self.base_from_camera) @ np.append(np.asarray(point_base, dtype=float), 1.)
        if camera[2] <= 0:
            return None
        u = k[0, 0]*camera[0]/camera[2]+k[0, 2]
        v = k[1, 1]*camera[1]/camera[2]+k[1, 2]
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        return float(u), float(v), float(camera[2])


def camera_moved(a, b, *, tolerance_m, tolerance_rad):
    translation = float(np.linalg.norm(np.asarray(a[:3, 3])-np.asarray(b[:3, 3])))
    return translation > tolerance_m or rotation_angle(a[:3, :3], b[:3, :3]) > tolerance_rad


def _block_mean(depth, block):
    """Mean depth per block x block cell, ignoring invalid pixels; NaN where a cell has none."""
    array = np.asarray(depth, dtype=float)
    height, width = (array.shape[0]//block)*block, (array.shape[1]//block)*block
    cells = array[:height, :width].reshape(height//block, block, width//block, block)
    valid = np.isfinite(cells) & (cells > 0)
    counts = valid.sum(axis=(1, 3))
    sums = np.where(valid, cells, 0.).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts >= block*block//2, sums/np.maximum(counts, 1), np.nan)
    return means


def depth_change_mask(previous, current, *, block=8, threshold_m=.03):
    """Per-cell mask of commonly valid depth cells whose mean moved more than threshold_m, or None."""
    a, b = _block_mean(previous, block), _block_mean(current, block)
    if a.shape != b.shape:
        return None
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 20:
        return None
    return valid & (np.abs(np.where(valid, a-b, 0.)) > threshold_m)


def depth_change(previous, current, *, block=8, threshold_m=.03):
    """(fraction, count) of commonly valid depth cells whose mean moved more than threshold_m.

    Cells average out sensor noise and edge flicker; a change has to be a
    region, not scattered pixels.
    """
    mask = depth_change_mask(previous, current, block=block, threshold_m=threshold_m)
    if mask is None:
        return 1., 0
    a = _block_mean(previous, block)
    valid = np.isfinite(a) & np.isfinite(_block_mean(current, block))
    return float(mask.sum()/max(1, valid.sum())), int(mask.sum())



class KeyframeSelector:
    """Decide which still captures become keyframes; never during motion."""

    def __init__(self, *, min_interval_s=3., move_tolerance_m=.02, move_tolerance_rad=.05,
                 change_fraction=.03, change_min_samples=12, change_depth_m=.03, max_capture_age_s=1.):
        for value in (min_interval_s, move_tolerance_m, move_tolerance_rad, change_fraction, change_min_samples,
                      change_depth_m, max_capture_age_s):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise PerceptionError("keyframe selector bounds must be positive numbers")
        self.min_interval_s, self.move_tolerance_m, self.move_tolerance_rad = min_interval_s, move_tolerance_m, move_tolerance_rad
        self.change_fraction, self.change_min_samples = change_fraction, int(change_min_samples)
        self.change_depth_m, self.max_capture_age_s = change_depth_m, max_capture_age_s
        self.last = None
        self._pending_change = None

    def scene_changed(self, previous_depth, depth):
        """A change is enough 8x8 cells, and the same cells in two consecutive captures.

        Depth flicker is uncorrelated between frames; a thing that moved or
        appeared stays changed. Only cells changed in both this capture and
        the previous one, against the same keyframe, count.
        """
        mask = depth_change_mask(previous_depth, depth, threshold_m=self.change_depth_m)
        if mask is None:
            self._pending_change = None
            return True
        persistent = mask if self._pending_change is None or self._pending_change.shape != mask.shape else (mask & self._pending_change)
        self._pending_change = mask
        count = int(persistent.sum())
        fraction = count/max(1, mask.size)
        return count >= self.change_min_samples and (fraction >= self.change_fraction or count >= 2*self.change_min_samples)

    def consider(self, pair, *, base_from_camera, joints, still_since, clock_mapper, now, force=False):
        metadata = pair.metadata
        stamp_ns = metadata["source_stamps_ns"]["rgb"]
        mapped, uncertainty = clock_mapper(stamp_ns/10**9)
        if not all(math.isfinite(v) for v in (mapped, uncertainty)):
            return None
        oldest = mapped-uncertainty
        if still_since is None or still_since > oldest or now-oldest > self.max_capture_age_s:
            return None                                  # captured while moving, or too old to place
        depth = np.asarray(pair.depth_m, dtype=float)
        rgb = np.asarray(pair.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            return None
        reason = None
        if self.last is None:
            reason = "initial"
        elif camera_moved(self.last.base_from_camera, base_from_camera,
                          tolerance_m=self.move_tolerance_m, tolerance_rad=self.move_tolerance_rad):
            reason = "camera_moved"
        elif self.scene_changed(self.last.depth_m, depth):
            reason = "scene_changed"
        elif force:
            reason = "requested"
        if reason is None or (not force and self.last is not None and now-self.last.selected_at_s < self.min_interval_s):
            return None
        self._pending_change = None
        keyframe = Keyframe(pair.capture_id, rgb, depth, tuple(float(v) for v in metadata["rgb_info"]["k"]),
                            metadata["rgb_info"]["frame_id"], int(stamp_ns),
                            float((metadata.get("received_at_monotonic_s") or {}).get("rgb", now)),
                            float(mapped), float(uncertainty), np.array(base_from_camera, dtype=float),
                            tuple(float(q) for q in joints), reason, float(now))
        self.last = keyframe
        return keyframe


class FaceScreen:
    """YuNet face detector; any detection withholds the frame.

    The model file's sha256 is pinned in a sidecar and verified on load. A
    detector can miss a face, so a clear screen is evidence, not proof.
    """

    def __init__(self, model_path, *, score_threshold=.5):
        import cv2
        path = Path(model_path)
        sidecar = Path(str(path)+".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise PerceptionError(f"face screen model or its sha256 sidecar is missing: {path}")
        expected = sidecar.read_text().split()[0]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise PerceptionError("face screen model does not match its pinned sha256")
        self.model_sha256 = actual
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold=float(score_threshold))
        self._size = None

    def screen(self, rgb):
        image = np.ascontiguousarray(np.asarray(rgb)[..., ::-1])
        size = (int(image.shape[1]), int(image.shape[0]))
        if size != self._size:
            self.detector.setInputSize(size)
            self._size = size
        _, faces = self.detector.detect(image)
        count = 0 if faces is None else int(len(faces))
        return {"contains_face": count > 0, "faces": count, "model_sha256": self.model_sha256}


def encode_keyframe(keyframe, *, camera_id, calibration_id, screen, max_long_edge=640, max_bytes=200000):
    """Downscale, re-encode and screen a keyframe into an egress-checked crop."""
    require_ids(camera_id, calibration_id)
    if screen is None:
        raise PerceptionError("no face screen is configured; keyframes cannot leave the machine")
    verdict = screen.screen(keyframe.rgb)
    if verdict["contains_face"]:
        raise PerceptionError(f"face screen found {verdict['faces']} face(s); the keyframe is withheld")
    from PIL import Image
    picture = Image.fromarray(np.ascontiguousarray(keyframe.rgb))
    picture.thumbnail((max_long_edge, max_long_edge))
    payload = b""
    for quality in (85, 70, 55, 40):
        output = io.BytesIO()
        picture.save(output, format="JPEG", quality=quality)
        payload = output.getvalue()
        if len(payload) <= max_bytes:
            break
    if len(payload) > max_bytes:
        raise PerceptionError("keyframe cannot fit the configured byte budget")
    crop = ImageCrop("keyframe-"+keyframe.capture_id, keyframe.capture_id, camera_id, keyframe.frame_id,
                     calibration_id, keyframe.captured_at_s, (0, 0, keyframe.width, keyframe.height),
                     *picture.size, payload, False, False)
    crop.validate_for_egress(max_bytes=max_bytes, max_long_edge=max_long_edge, allow_face=False)
    return crop
