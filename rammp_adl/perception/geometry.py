"""Timestamped local geometry. No semantic label or depth pixel implies a 6-DoF pose."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np


class PerceptionError(ValueError):
    pass


def positive_bounds(*values):
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
        raise PerceptionError("Calibrated bounds must be finite and positive")


def require_ids(*values):
    if any(not isinstance(v, str) or not 0 < len(v) <= 256 for v in values):
        raise PerceptionError("Nonempty bounded provenance IDs are required")


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rectified: bool = True

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) <= 0:
            raise PerceptionError("Invalid image dimensions")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)) or min(self.fx, self.fy) <= 0:
            raise PerceptionError("Invalid camera intrinsics")


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    image_id: str
    rgb: np.ndarray
    depth_m: np.ndarray | None
    intrinsics: Intrinsics | None
    captured_at: float
    frame_id: str
    calibration_id: str
    timestamp_uncertainty_s: float = 0.0
    depth_aligned: bool = True

    def __post_init__(self):
        rgb = np.array(self.rgb, copy=True)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise PerceptionError("RGB must be a nonempty uint8 HxWx3 array")
        require_ids(self.camera_id, self.image_id, self.frame_id, self.calibration_id)
        if not math.isfinite(self.captured_at) or not math.isfinite(self.timestamp_uncertainty_s) or self.timestamp_uncertainty_s < 0:
            raise PerceptionError("Capture clock or uncertainty is invalid")
        rgb.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        if self.depth_m is not None:
            depth = np.array(self.depth_m, dtype=float, copy=True)
            if depth.shape != rgb.shape[:2] or not self.depth_aligned or self.intrinsics is None:
                raise PerceptionError("Depth must be aligned to RGB with matching intrinsics")
            if (self.intrinsics.height, self.intrinsics.width) != depth.shape:
                raise PerceptionError("CameraInfo dimensions do not match the capture")
            depth.setflags(write=False)
            object.__setattr__(self, "depth_m", depth)

    def require_fresh(self, now, max_age_s, *, max_timestamp_uncertainty_s=.02):
        positive_bounds(max_age_s, max_timestamp_uncertainty_s)
        if not 0 <= now - self.captured_at <= max_age_s or self.timestamp_uncertainty_s > max_timestamp_uncertainty_s:
            raise PerceptionError("Capture is stale, future dated or lacks bounded timestamp alignment")


def deproject(frame: CameraFrame, pixel_xy, *, min_depth_m, max_depth_m, patch_radius=1):
    positive_bounds(min_depth_m, max_depth_m)
    if frame.depth_m is None or frame.intrinsics is None or not frame.intrinsics.rectified:
        raise PerceptionError("Metric deprojection needs aligned depth and rectified intrinsics")
    if not 0 < min_depth_m < max_depth_m or type(patch_radius) is not int or not 0 <= patch_radius <= 8:
        raise PerceptionError("Invalid calibrated depth bounds or patch radius")
    x, y = map(float, pixel_xy)
    intrinsics = frame.intrinsics
    if not all(math.isfinite(v) for v in (x, y)) or not 0 <= x < intrinsics.width or not 0 <= y < intrinsics.height:
        raise PerceptionError("Pixel lies outside its original capture")
    ix, iy = int(x), int(y)
    patch = frame.depth_m[max(0, iy-patch_radius):iy+patch_radius+1, max(0, ix-patch_radius):ix+patch_radius+1]
    valid = patch[np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)]
    if not valid.size:
        raise PerceptionError("Depth missing or outside calibrated range")
    depth = float(np.median(valid))
    point = np.array([(x-intrinsics.cx)*depth/intrinsics.fx, (y-intrinsics.cy)*depth/intrinsics.fy, depth])
    if not np.isfinite(point).all():
        raise PerceptionError("Deprojection is numerically invalid")
    return point  # A point, not an orientation or a grasp pose.


@dataclass(frozen=True)
class CalibratedTransform:
    source_frame: str
    target_frame: str
    rotation: np.ndarray
    translation_m: np.ndarray
    calibration_id: str
    base_epoch: str
    captured_at: float
    valid_for_s: float
    rigid_static: bool = False

    def __post_init__(self):
        rotation, translation = np.array(self.rotation, dtype=float, copy=True), np.array(self.translation_m, dtype=float, copy=True)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise PerceptionError("Invalid transform dimensions")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
            raise PerceptionError("Transform rotation is not a proper rotation")
        require_ids(self.source_frame, self.target_frame, self.calibration_id, self.base_epoch)
        positive_bounds(self.valid_for_s)
        if not math.isfinite(self.captured_at) or type(self.rigid_static) is not bool:
            raise PerceptionError("Transform capture time/static flag is invalid")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)

    def apply(self, point, *, capture_time, source_frame, calibration_id, base_epoch):
        if not math.isfinite(capture_time):
            raise PerceptionError("Transform requires a finite capture time")
        if (source_frame, calibration_id, base_epoch) != (self.source_frame, self.calibration_id, self.base_epoch):
            raise PerceptionError("Transform provenance mismatch")
        if not self.rigid_static and not self.captured_at <= capture_time <= self.captured_at + self.valid_for_s:
            raise PerceptionError("No valid transform at capture time")
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise PerceptionError("Invalid metric point")
        transformed = self.rotation @ point + self.translation_m
        if not np.isfinite(transformed).all():
            raise PerceptionError("Transformed point is numerically invalid")
        return transformed


def fuse_points(points, covariances):
    """Fuse already synchronized/transformed independent estimates in ONE frame."""
    if len(points) != len(covariances) or not len(points):
        raise PerceptionError("Fusion requires matching nonempty measurements")
    information = np.zeros((3, 3))
    weighted = np.zeros(3)
    for point, covariance in zip(points, covariances):
        point, covariance = np.asarray(point, dtype=float), np.asarray(covariance, dtype=float)
        if point.shape != (3,) or covariance.shape != (3, 3) or not np.isfinite(point).all() or not np.isfinite(covariance).all():
            raise PerceptionError("Invalid point covariance")
        if not np.allclose(covariance, covariance.T) or np.min(np.linalg.eigvalsh(covariance)) <= 0:
            raise PerceptionError("Covariance must be positive definite")
        precision = np.linalg.inv(covariance)
        if not np.isfinite(precision).all():
            raise PerceptionError("Covariance inversion is numerically invalid")
        information += precision
        weighted += precision @ point
    if not np.isfinite(information).all() or not np.isfinite(weighted).all():
        raise PerceptionError("Fusion accumulation is numerically invalid")
    covariance = np.linalg.inv(information)
    estimate = covariance @ weighted
    if not np.isfinite(covariance).all() or not np.isfinite(estimate).all():
        raise PerceptionError("Fused point is numerically invalid")
    return estimate, covariance


def fit_plane(points, *, max_rms_m):
    positive_bounds(max_rms_m)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise PerceptionError("Plane fit needs at least three finite 3D points")
    center = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points-center, full_matrices=False)
    if singular[1] < 1e-6:
        raise PerceptionError("Collinear points do not identify a plane")
    normal = axes[-1]
    rms = float(np.sqrt(np.mean(((points-center) @ normal)**2)))
    if not math.isfinite(rms) or rms > max_rms_m:
        raise PerceptionError("Plane residual exceeds local profile")
    return {"origin_m": center, "normal": normal, "rms_m": rms, "normal_sign_ambiguous": True}


def fit_prismatic(points, *, max_rms_m, min_stroke_m):
    positive_bounds(max_rms_m, min_stroke_m)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise PerceptionError("Translation fit needs finite motion samples")
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points-center, full_matrices=False)
    direction = axes[0]
    if np.dot(points[-1]-points[0], direction) < 0:
        direction = -direction
    along = (points-center) @ direction
    residual = points-center-along[:, None]*direction
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    if not math.isfinite(rms) or np.ptp(along) < min_stroke_m or rms > max_rms_m:
        raise PerceptionError("Insufficient stroke or inconsistent translation fit")
    return {"kind": "prismatic", "origin_m": points[0], "axis": direction, "rms_m": rms, "unit": "m"}


def fit_revolute(points, *, max_rms_m, min_span_rad, min_radius_m, max_radius_m):
    positive_bounds(max_rms_m, min_span_rad, min_radius_m, max_radius_m)
    if min_radius_m >= max_radius_m or min_span_rad > 2*math.pi:
        raise PerceptionError("Invalid hinge observability envelope")
    points = np.asarray(points, dtype=float)
    if len(points) < 5:
        raise PerceptionError("Hinge fit needs at least five observed positions")
    plane = fit_plane(points, max_rms_m=max_rms_m)
    center, normal = plane["origin_m"], plane["normal"]
    basis_x = points[0]-center
    basis_x -= np.dot(basis_x, normal)*normal
    norm = np.linalg.norm(basis_x)
    if not math.isfinite(norm) or norm < 1e-9:
        raise PerceptionError("Hinge samples do not identify an in-plane basis")
    basis_x /= norm
    basis_y = np.cross(normal, basis_x)
    local = np.column_stack(((points-center) @ basis_x, (points-center) @ basis_y))
    matrix = np.column_stack((2*local[:, 0], 2*local[:, 1], np.ones(len(points))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(local**2, axis=1), rcond=None)
    if rank < 3:
        raise PerceptionError("Hinge samples do not identify a circle")
    circle_center = solution[:2]
    radii = np.linalg.norm(local-circle_center, axis=1)
    radius = float(np.mean(radii))
    angles = np.unwrap(np.arctan2(local[:, 1]-circle_center[1], local[:, 0]-circle_center[0]))
    rms = float(np.sqrt(np.mean((radii-radius)**2)))
    if not math.isfinite(rms) or not min_radius_m <= radius <= max_radius_m or np.ptp(angles) < min_span_rad or rms > max_rms_m:
        raise PerceptionError("Hinge fit is under-observed or outside its calibrated envelope")
    if angles[-1] < angles[0]:
        normal = -normal
    return {"kind": "revolute", "origin_m": center+circle_center[0]*basis_x+circle_center[1]*basis_y,
            "axis": normal, "radius_m": radius, "rms_m": max(rms, plane["rms_m"]), "unit": "rad"}


class TrackStore:
    """Stable-ID local tracking; stale tracks remain present but invalid."""
    def __init__(self, *, max_age_s, max_displacement_m, clock=time.monotonic):
        positive_bounds(max_age_s, max_displacement_m)
        self.max_age_s, self.max_displacement_m, self.clock = max_age_s, max_displacement_m, clock
        self._tracks = {}

    def update(self, entity_id, point, *, captured_at, evidence_id, calibration_id, base_epoch, ambiguous=False):
        require_ids(entity_id, evidence_id, calibration_id, base_epoch)
        point = np.asarray(point, dtype=float)
        if ambiguous or not entity_id or point.shape != (3,) or not np.isfinite(point).all() or not 0 <= self.clock()-captured_at <= self.max_age_s:
            raise PerceptionError("Tracking update lacks fresh unambiguous geometry")
        old = self._tracks.get(entity_id)
        if old:
            if (old["calibration_id"], old["base_epoch"]) != (calibration_id, base_epoch):
                raise PerceptionError("Tracking requires reinitialization after frame/calibration change")
            if captured_at <= old["captured_at"] or np.linalg.norm(point-old["origin_m"]) > self.max_displacement_m:
                raise PerceptionError("Out-of-order update or cumulative tracking envelope exceeded")
        self._tracks[entity_id] = {"point_m": point.copy(), "origin_m": old["origin_m"] if old else point.copy(),
            "captured_at": captured_at, "evidence_id": evidence_id, "calibration_id": calibration_id, "base_epoch": base_epoch}

    def snapshot(self):
        return {entity: {**record, "point_m": record["point_m"].copy(), "origin_m": record["origin_m"].copy(),
                         "valid": 0 <= self.clock()-record["captured_at"] <= self.max_age_s}
                for entity, record in self._tracks.items()}
