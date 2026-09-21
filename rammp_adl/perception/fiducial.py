"""NEW provisional single-fiducial observations using raw color CameraInfo.

Both IPPE square pose branches are retained. Printed scale, intrinsics accuracy,
capture clocks and robot references remain unverified. These reports never
construct CalibratedTransform, publish TF, or admit robot motion capabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..contracts import Catalog, digest, strict_loads
from .calibration_target import read_local_capture
from .geometry import PerceptionError, positive_bounds


def raw_camera_model(info):
    """Use K/D for explicit raw color input, never reinterpret P as raw K."""
    if info["distortion_model"] not in ("plumb_bob", "rational_polynomial"):
        raise PerceptionError("Raw color distortion model is unsupported")
    if any(info["roi"].values()) or info["binning_x"] not in (0, 1) or info["binning_y"] not in (0, 1):
        raise PerceptionError("Raw color ROI/binning needs an explicit adapter")
    k = np.asarray(info["k"], dtype=float)
    d = np.asarray(info["d"], dtype=float)
    if k.shape != (9,) or d.ndim != 1 or len(d) not in (4, 5, 8, 12, 14) or not np.isfinite(k).all() or not np.isfinite(d).all():
        raise PerceptionError("Invalid raw camera matrix/distortion")
    k = k.reshape(3, 3)
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k, [[k[0, 0], 0, k[0, 2]], [0, k[1, 1], k[1, 2]], [0, 0, 1]], atol=1e-9, rtol=0):
        raise PerceptionError("Uncalibrated or unsupported raw camera intrinsics")
    return k, d


class SingleMarkerObserver:
    def __init__(self, spec=None):
        import cv2
        if spec is None:
            spec = strict_loads((Catalog().root/"config/calibration-marker.json").read_bytes())
        if not isinstance(spec, dict) or set(spec) != {"marker_id", "dictionary", "nominal_black_edge_m", "size_source"}:
            raise PerceptionError("Expected the local single-marker specification")
        if spec["dictionary"] != "DICT_4X4_50" or type(spec["marker_id"]) is not int or not 0 <= spec["marker_id"] < 50:
            raise PerceptionError("Unsupported marker dictionary or ID")
        positive_bounds(spec["nominal_black_edge_m"])
        if not .005 <= spec["nominal_black_edge_m"] <= 1. or not isinstance(spec["size_source"], str) or not 0 < len(spec["size_source"]) <= 256:
            raise PerceptionError("Marker nominal size or provenance is invalid")
        self.spec_json = json.dumps(spec, sort_keys=True, allow_nan=False)
        self.cv2 = cv2
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)

    @property
    def spec(self):
        return json.loads(self.spec_json)

    def object_points(self):
        h = self.spec["nominal_black_edge_m"]/2
        # IPPE_SQUARE order matches ArUco's canonical TL, TR, BR, BL.
        # Marker frame: center origin, x right, y up, z out of printed face.
        return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

    def pose_candidates(self, corners, info):
        points = np.asarray(corners, dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise PerceptionError("Four finite marker corners are required")
        if not self.cv2.isContourConvex(points.astype(np.float32)) or abs(self.cv2.contourArea(points.astype(np.float32))) < 1.:
            raise PerceptionError("Marker corners are degenerate")
        k, d = raw_camera_model(info)
        objects = self.object_points()
        success, rotations, translations, _ = self.cv2.solvePnPGeneric(objects, np.ascontiguousarray(points), k, d,
                                                                       flags=self.cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = []
        if not success:
            return candidates
        for rvec, tvec in zip(rotations, translations):
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            rotation = self.cv2.Rodrigues(rvec)[0]
            translation = tvec.reshape(3)
            if np.min((objects @ rotation.T + translation)[:, 2]) <= 0:
                continue
            projected = self.cv2.projectPoints(objects, rvec, tvec, k, d)[0].reshape(4, 2)
            rms = float(np.sqrt(np.mean(np.sum((projected-points)**2, axis=1))))
            if not np.isfinite(rms):
                continue
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, translation
            candidates.append({"camera_from_marker": transform.tolist(), "reprojection_rms_px": rms})
        return sorted(candidates, key=lambda c: c["reprojection_rms_px"])

    def inspect(self, rgb, metadata, *, capture_id):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not 0 < rgb.shape[0]*rgb.shape[1] <= 16000000:
            raise PerceptionError("Marker observation requires bounded uint8 RGB input")
        info = metadata["rgb_info"]
        if rgb.shape[:2] != (info["height"], info["width"]):
            raise PerceptionError("Raw color dimensions and CameraInfo disagree")
        report = {"status": "marker_not_observed", "capture_id": capture_id,
                  "source_stamps_ns": metadata["source_stamps_ns"],
                  "received_at_monotonic_s": metadata.get("received_at_monotonic_s"),
                  "camera_frame": info["frame_id"], "rgb_info_digest": digest(info),
                  "marker_spec": self.spec, "marker_spec_digest": digest(self.spec), "image_domain": "raw_color",
                  "opencv_version": self.cv2.__version__, "pose_candidates": [],
                  "physical_marker_size_verified": False, "capture_clock_validated": False,
                  "metric_geometry_validated": False, "robot_reference_calibrated": False, "frames_uploaded": 0}
        corners, ids, _ = self.detector.detectMarkers(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY))
        all_ids = [] if ids is None else ids.reshape(-1).tolist()
        report["observed_marker_ids"] = all_ids
        matches = [i for i, value in enumerate(all_ids) if value == self.spec["marker_id"]]
        if len(matches) != 1:
            if matches:
                report["status"] = "duplicate_marker_id_ambiguous"
            return report
        points = np.asarray(corners[matches[0]], dtype=np.float64).reshape(4, 2)
        h, w = rgb.shape[:2]
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= (w, h)):
            raise PerceptionError("Detected marker corners lie outside the capture")
        report.update(corners_xy_px=points.tolist(), minimum_edge_px=float(np.min(np.linalg.norm(points-np.roll(points, -1, axis=0), axis=1))))
        try:
            candidates = self.pose_candidates(points, info)
        except (PerceptionError, self.cv2.error) as exc:
            report.update(status="marker_observed_pose_unavailable", detail=str(exc))
            return report
        report.update(status="provisional_pose_candidates" if candidates else "marker_observed_pose_unavailable", pose_candidates=candidates)
        report["planar_ambiguity_resolved"] = False
        if len(candidates) > 1:
            report["branch_reprojection_gap_px"] = candidates[1]["reprojection_rms_px"]-candidates[0]["reprojection_rms_px"]
        return report

    def observe(self, pair):
        return self.inspect(pair.rgb, pair.metadata, capture_id=pair.capture_id)


def _validated_pose_candidates(observation):
    if observation["marker_spec_digest"] != digest(observation["marker_spec"]):
        raise PerceptionError("Marker specification digest does not match the observation")
    if observation["status"] != "provisional_pose_candidates" or not 1 <= len(observation["pose_candidates"]) <= 2:
        raise PerceptionError("Expected one or two provisional square pose candidates per camera")
    transforms = []
    for candidate in observation["pose_candidates"]:
        error = candidate["reprojection_rms_px"]
        if type(error) not in (int, float) or not np.isfinite(error) or error < 0:
            raise PerceptionError("Pose reprojection error must be finite and nonnegative")
        transform = np.asarray(candidate["camera_from_marker"], dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(transform[:3, :3]), 1.):
            raise PerceptionError("Pose candidate is not a rigid transform")
        transforms.append(transform)
    return transforms


def fixed_marker_reference(observation, *, marker_fixed_confirmed, camera_fixed_confirmed):
    """Export a preliminary marker frame from an explicitly fixed scene camera.

    The same marker must remain fixed. Its frame supports local scene previews;
    it is not base_link. Never apply this fixed-camera assumption to the wrist.
    Historical evidence retains its original stamps and unresolved uncertainty.
    """
    if marker_fixed_confirmed is not True or camera_fixed_confirmed is not True:
        raise PerceptionError("Both marker and selected scene camera must be confirmed fixed")
    transforms = _validated_pose_candidates(observation)
    spec = observation["marker_spec"]
    return {
        "status": "provisional_fixed_marker_reference",
        "reference_frame": f"marker_{spec['dictionary']}_{spec['marker_id']}",
        "camera_frame": observation["camera_frame"],
        "source_observation_digest": digest(observation),
        "source_capture_id": observation["capture_id"],
        "source_stamps_ns": observation["source_stamps_ns"],
        "rgb_info_digest": observation["rgb_info_digest"],
        "marker_spec": spec, "marker_spec_digest": observation["marker_spec_digest"],
        "candidates": [{"branch": i, "marker_from_camera": np.linalg.inv(t).tolist(),
                        "reprojection_rms_px": observation["pose_candidates"][i]["reprojection_rms_px"]}
                       for i, t in enumerate(transforms)],
        "marker_fixed_operator_confirmed": True, "scene_camera_fixed_operator_confirmed": True,
        "planar_ambiguity_resolved": False, "metric_geometry_validated": False,
        "physical_marker_size_verified": False, "capture_clock_validated": False,
        "robot_reference_calibrated": False, "base_from_marker": None,
        "position_uncertainty_m": None, "orientation_uncertainty_rad": None,
        "hardware_motion_enabled": False, "frames_uploaded": 0,
        "validity_scope": "Preliminary fixed scene-camera relation to a fixed marker; invalidate after either moves or intrinsics change",
        "wrist_rule": "A wrist camera needs a new capture-time marker observation or verified robot kinematics; its historical camera relation is not static",
        "geometry_use": "Local marker-frame previews under nominal print scale; no robot TF or motion admission",
    }


def relative_pose_candidates(first, second, *, stationary_setup_confirmed):
    """Conditional camera-to-camera hypotheses from the same stationary marker.

    All planar branch combinations remain visible; no winner is commissioned.
    Confirmation is a local operator statement, not measured clock alignment.
    With a wrist camera this relation is specific to the current arm posture.
    """
    if stationary_setup_confirmed is not True:
        raise PerceptionError("Both cameras and the same marker must have remained stationary between the captures")
    lefts, rights = map(_validated_pose_candidates, (first, second))
    if first["marker_spec_digest"] != second["marker_spec_digest"] or first["capture_id"] == second["capture_id"] or first["camera_frame"] == second["camera_frame"]:
        raise PerceptionError("Distinct camera captures with the same marker definition are required")
    if not first["pose_candidates"] or not second["pose_candidates"]:
        raise PerceptionError("The marker needs pose candidates in both cameras")
    candidates = []
    for i, a in enumerate(first["pose_candidates"]):
        for j, b in enumerate(second["pose_candidates"]):
            composed = rights[j] @ np.linalg.inv(lefts[i])
            candidates.append({"first_branch": i, "second_branch": j, "second_camera_from_first_camera": composed.tolist(),
                               "sum_reprojection_rms_px": a["reprojection_rms_px"]+b["reprojection_rms_px"]})
    return {"status": "provisional_relative_pose_candidates", "source_frame": first["camera_frame"], "target_frame": second["camera_frame"],
            "capture_ids": [first["capture_id"], second["capture_id"]],
            "marker_spec": first["marker_spec"], "marker_spec_digest": first["marker_spec_digest"],
            "rgb_info_digests": [first["rgb_info_digest"], second["rgb_info_digest"]],
            "source_stamps_ns": [first["source_stamps_ns"], second["source_stamps_ns"]],
            "candidate_order": "ascending summed pixel reprojection error; not a calibration approval",
            "candidates": sorted(candidates, key=lambda c: c["sum_reprojection_rms_px"]),
            "stationary_setup_operator_confirmed": True, "planar_ambiguity_resolved": False,
            "physical_marker_size_verified": False, "capture_clock_validated": False, "metric_geometry_validated": False,
            "robot_reference_calibrated": False, "frames_uploaded": 0,
            "validity_scope": "Conditional camera-to-camera relation at this fixed arm posture; invalid after wrist/base/camera motion"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-capture")
    inspect.add_argument("--capture", required=True)
    inspect.add_argument("--output", required=True)
    relative = commands.add_parser("relative")
    relative.add_argument("--first-observation", required=True)
    relative.add_argument("--second-observation", required=True)
    relative.add_argument("--stationary-setup-confirmed", action="store_true")
    relative.add_argument("--output", required=True)
    fixed = commands.add_parser("fixed-reference", help="Export a preliminary marker frame for a fixed scene camera")
    fixed.add_argument("--observation", required=True)
    fixed.add_argument("--marker-fixed-confirmed", action="store_true")
    fixed.add_argument("--camera-fixed-confirmed", action="store_true")
    fixed.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-capture":
            rgb, meta = read_local_capture(args.capture)
            report = SingleMarkerObserver().inspect(rgb, meta, capture_id=meta["capture_id"])
            report["historical_capture"] = True
        elif args.command == "fixed-reference":
            report = fixed_marker_reference(strict_loads(Path(args.observation).read_bytes()),
                                            marker_fixed_confirmed=args.marker_fixed_confirmed,
                                            camera_fixed_confirmed=args.camera_fixed_confirmed)
        else:
            report = relative_pose_candidates(strict_loads(Path(args.first_observation).read_bytes()),
                                              strict_loads(Path(args.second_observation).read_bytes()),
                                              stationary_setup_confirmed=args.stationary_setup_confirmed)
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(json.dumps({"status": "unavailable_or_rejected", "detail": str(exc), "hardware_commands": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
