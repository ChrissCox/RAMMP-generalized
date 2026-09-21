"""Live guards for a trajectory the driver is already flying.

The driver executes a static trajectory and watches nothing. These guards run
in the client beside it: a collision guard lifts the wrist camera's depth into
the base frame through forward kinematics and the measured D405 mount, and
checks the arm's *upcoming* sphere positions along the remaining path against
what the camera sees; an effort guard watches joint torque deviation after
motion has started. A trip cancels the goal; the driver then stops and holds.

This is a guard that stops, not a planner that steers around. It sees only
what the wrist D405 sees (nominally 7-50 cm), it cannot see through the
gripper, and a point that lies on the robot's own current geometry is
filtered as self-observation, so an obstacle already touching the fingers is
the effort guard's to catch. Depth is paired with the latest joint sample, so
motion between the frame's exposure and that sample is absorbed by margin.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .kinematics import UrdfChain, quaternion_matrix
from .rolling import JointTrajectory, MotionError
from .sheppy_client import wrap_diff


SEMANTICS = ("client-side stop guard from the wrist camera's own view and joint efforts; "
             "not a certified collision system and not a planner")
#: The D405 mount as box-opening measured it on the bench (camera_d405_wrist.yaml,
#: 2026-08-25): ee_from_camera, lens axis along the tool, 180 degrees about it.
MEASURED_D405_MOUNT = {"parent_link": "end_effector_link", "xyz": (0., .075, -.005),
                       "quaternion_xyzw": (0., 0., 1., 0.),
                       "evidence": "RAMMP-box-opening config/camera_d405_wrist.yaml, measured 2026-08-25"}


class GuardError(MotionError):
    """Unusable sphere model, depth frame or trajectory."""


def _finite(values, label):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise GuardError(f"{label} must be finite")
    return array


def load_spheres(path):
    """Per-link collision spheres in link frames, as the assembly bundle writes them."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    spheres = document.get("collision_spheres")
    if not isinstance(spheres, dict) or not spheres:
        raise GuardError("Sphere document lacks collision_spheres")
    model = {}
    for link, rows in spheres.items():
        centres = _finite([row["center"] for row in rows], f"{link} sphere centres")
        radii = _finite([row["radius"] for row in rows], f"{link} sphere radii")
        if centres.ndim != 2 or centres.shape[1] != 3 or (radii <= 0).any():
            raise GuardError(f"{link} spheres are malformed")
        model[link] = (centres, radii)
    return model


def mount_transform(mount=MEASURED_D405_MOUNT):
    """ee_from_camera as a 4x4, exactly as box-opening composes it."""
    transform = np.eye(4)
    transform[:3, :3] = quaternion_matrix(mount["quaternion_xyzw"])
    transform[:3, 3] = _finite(mount["xyz"], "mount translation")
    return transform


def trajectory_positions_at(trajectory, time_s):
    """Joint positions on the path at time_s; linear between waypoints, clamped."""
    if not isinstance(trajectory, JointTrajectory):
        raise GuardError("A JointTrajectory is required")
    points = trajectory.points
    if time_s <= points[0].time_s:
        return tuple(points[0].state.position)
    if time_s >= points[-1].time_s:
        return tuple(points[-1].state.position)
    for before, after in zip(points, points[1:]):
        if before.time_s <= time_s <= after.time_s:
            span = after.time_s-before.time_s
            fraction = 0. if span <= 0 else (time_s-before.time_s)/span
            return tuple(a+wrap_diff(b, a)*fraction
                         for a, b in zip(before.state.position, after.state.position))
    return tuple(points[-1].state.position)


def depth_to_points(depth_m, intrinsics_k, *, stride=8, min_range_m=.07, max_range_m=.5):
    """Back-project a depth image to camera-frame points; invalid depth is dropped."""
    depth = np.asarray(depth_m, dtype=float)
    k = _finite(intrinsics_k, "intrinsics").reshape(3, 3)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    if depth.ndim != 2 or fx <= 0 or fy <= 0:
        raise GuardError("Depth image and pinhole intrinsics are required")
    if isinstance(stride, bool) or type(stride) is not int or stride < 1:
        raise GuardError("Stride must be a positive integer")
    rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    keep = np.isfinite(z) & (z >= min_range_m) & (z <= max_range_m)
    z, u, v = z[keep], cols[keep].astype(float), rows[keep].astype(float)
    return np.column_stack(((u-cx)*z/fx, (v-cy)*z/fy, z))


class SphereModel:
    """The assembly's spheres placed in the base frame for any configuration."""

    def __init__(self, chain, spheres, *, joint_names=None, extra=None):
        if not isinstance(chain, UrdfChain):
            raise GuardError("An inspected UrdfChain is required")
        missing = [link for link in spheres if link not in chain.link_names]
        if missing:
            raise GuardError("Sphere links absent from the model: "+", ".join(sorted(missing)))
        self.chain, self.spheres = chain, dict(spheres)
        self.joint_names = tuple(joint_names or chain.actuated_joints)
        # Extra spheres are given in a parent link's frame through a fixed transform,
        # which is how the D405 rides on end_effector_link with the measured mount.
        self.extra = list(extra or [])
        for parent, transform, centres, radii in self.extra:
            if parent not in chain.link_names:
                raise GuardError(f"Extra sphere parent {parent} is not in the model")

    def configuration(self, positions):
        if len(positions) != len(self.joint_names):
            raise GuardError("Joint count does not match the sphere model")
        return dict(zip(self.joint_names, (float(q) for q in positions)))

    def placed(self, positions):
        """(centres_base (S,3), radii (S,), labels) for one configuration."""
        configuration = self.configuration(positions)
        centres, radii, labels = [], [], []
        for link, (local, radius) in self.spheres.items():
            pose = self.chain.base_from_link(configuration, link)
            centres.append(local @ pose[:3, :3].T + pose[:3, 3])
            radii.append(radius)
            labels.extend([link]*len(radius))
        for parent, transform, local, radius in self.extra:
            pose = self.chain.base_from_link(configuration, parent) @ transform
            centres.append(np.asarray(local, dtype=float) @ pose[:3, :3].T + pose[:3, 3])
            radii.append(np.asarray(radius, dtype=float))
            labels.extend([parent+"+extra"]*len(radius))
        return np.vstack(centres), np.concatenate(radii), labels

    def base_from_camera(self, positions, *, camera_parent, ee_from_camera):
        return self.chain.base_from_link(self.configuration(positions), camera_parent) @ ee_from_camera


def nearest_intrusion(points_base, centres, radii, labels, *, margin_m):
    """The closest obstacle point to any sphere surface, if inside the margin."""
    if len(points_base) == 0:
        return None
    deltas = points_base[:, None, :]-centres[None, :, :]
    clearance = np.linalg.norm(deltas, axis=2)-radii[None, :]
    index = np.unravel_index(int(np.argmin(clearance)), clearance.shape)
    distance = float(clearance[index])
    if distance > margin_m:
        return None
    return {"distance_m": distance, "link": labels[index[1]],
            "point_base_m": [float(v) for v in points_base[index[0]]], "margin_m": margin_m}


class CollisionGuard:
    """Check the remaining path against what the wrist camera currently sees."""

    def __init__(self, model, *, camera_parent="end_effector_link", ee_from_camera=None,
                 margin_m=.03, self_margin_m=.015, lookahead_s=1.5, sample_dt_s=.1,
                 stride=8, min_range_m=.07, max_range_m=.5, max_frame_age_s=.3):
        if not isinstance(model, SphereModel):
            raise GuardError("A SphereModel is required")
        for value, label in ((margin_m, "margin"), (self_margin_m, "self margin"),
                             (lookahead_s, "lookahead"), (sample_dt_s, "sample interval"),
                             (max_frame_age_s, "frame age")):
            if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise GuardError(f"{label} must be a positive number")
        if camera_parent not in model.chain.link_names:
            raise GuardError(f"Camera parent {camera_parent} is not in the model")
        self.model, self.camera_parent = model, camera_parent
        self.ee_from_camera = mount_transform() if ee_from_camera is None else _finite(ee_from_camera, "mount")
        self.margin_m, self.self_margin_m = float(margin_m), float(self_margin_m)
        self.lookahead_s, self.sample_dt_s = float(lookahead_s), float(sample_dt_s)
        self.stride, self.min_range_m, self.max_range_m = stride, min_range_m, max_range_m
        self.max_frame_age_s = float(max_frame_age_s)
        self.checks, self.last = 0, None

    def points_in_base(self, depth_m, intrinsics_k, joints_at_frame):
        camera = depth_to_points(depth_m, intrinsics_k, stride=self.stride,
                                 min_range_m=self.min_range_m, max_range_m=self.max_range_m)
        transform = self.model.base_from_camera(joints_at_frame, camera_parent=self.camera_parent,
                                                ee_from_camera=self.ee_from_camera)
        return camera @ transform[:3, :3].T + transform[:3, 3]

    def self_filtered(self, points_base, joints_now):
        """Drop points lying on the robot's own current geometry."""
        if len(points_base) == 0:
            return points_base
        centres, radii, _ = self.model.placed(joints_now)
        clearance = np.linalg.norm(points_base[:, None, :]-centres[None, :, :], axis=2)-radii[None, :]
        return points_base[(clearance > self.self_margin_m).all(axis=1)]

    def sweep(self, points_base, trajectory, elapsed_s):
        """Nearest intrusion along the upcoming path window, or None."""
        if len(points_base) == 0:
            return None
        worst = None
        end = min(trajectory.duration_s, elapsed_s+self.lookahead_s)
        time_s = max(0., elapsed_s)
        while True:
            centres, radii, labels = self.model.placed(trajectory_positions_at(trajectory, time_s))
            found = nearest_intrusion(points_base, centres, radii, labels, margin_m=self.margin_m)
            if found is not None and (worst is None or found["distance_m"] < worst["distance_m"]):
                worst = {**found, "time_s": time_s}
            if time_s >= end:
                break
            time_s = min(end, time_s+self.sample_dt_s)
        return worst

    def tool_position(self, joints_now, *, tool_from_ee_z_m=.12):
        """The planner's tool_frame in base for a configuration."""
        ee = self.model.chain.base_from_link(self.model.configuration(joints_now), "end_effector_link")
        return ee[:3, :3] @ np.array([0., 0., tool_from_ee_z_m])+ee[:3, 3]

    @staticmethod
    def excluded(points_base, exclusions):
        """Drop points inside the balls where contact is intended."""
        for centre, radius in exclusions:
            if len(points_base) == 0:
                break
            keep = np.linalg.norm(points_base-np.asarray(centre, dtype=float), axis=1) > float(radius)
            points_base = points_base[keep]
        return points_base

    def check(self, *, depth_m, intrinsics_k, frame_age_s, joints_now, trajectory, elapsed_s, exclusions=()):
        """One guard evaluation; a stale frame is reported, never treated as clear."""
        self.checks += 1
        if frame_age_s > self.max_frame_age_s:
            return {"kind": "depth_stale", "frame_age_s": frame_age_s, "max_frame_age_s": self.max_frame_age_s}
        points = self.self_filtered(self.points_in_base(depth_m, intrinsics_k, joints_now), joints_now)
        points = self.excluded(points, exclusions)
        intrusion = self.sweep(points, trajectory, elapsed_s)
        self.last = None if intrusion is None else {"kind": "collision", **intrusion,
                                                     "obstacle_points": int(len(points))}
        return self.last


class EffortGuard:
    """Joint-torque deviation after motion has started, as box-opening's torque guard."""

    def __init__(self, touch_nm, *, joints=(4, 5, 6), arm_after_progress=0.):
        if isinstance(touch_nm, bool) or type(touch_nm) not in (int, float) or not touch_nm > 0:
            raise GuardError("touch threshold must be positive")
        if not 0. <= float(arm_after_progress) < 1.:
            raise GuardError("arm_after_progress must be in [0, 1)")
        self.touch_nm, self.joints = float(touch_nm), tuple(joints)
        self.arm_after_progress = float(arm_after_progress)
        self.armed, self.baseline, self.peak_nm, self.progress = False, None, 0., 0.

    def on_progress(self, progress):
        self.progress = float(progress)
        if self.progress > 0.:
            self.armed = True

    def on_efforts(self, effort_nm):
        if not self.armed or effort_nm is None:
            return None
        selected = [float(effort_nm[i]) for i in self.joints]
        if self.baseline is None:
            self.baseline = selected
            return None
        deviation = max(abs(a-b) for a, b in zip(selected, self.baseline))
        self.peak_nm = max(self.peak_nm, deviation)
        if self.progress < self.arm_after_progress or deviation <= self.touch_nm:
            return None
        return {"kind": "contact", "deviation_nm": deviation, "touch_nm": self.touch_nm,
                "joints": list(self.joints)}


class GuardSet:
    """What the client consults each tick while a trajectory flies.

    depth_reader() returns (depth_m, intrinsics_k, received_at_monotonic_s) or
    None. A collision guard with no readable frame reports the outage rather
    than passing: a blind guard is not a guard.
    """

    def __init__(self, *, effort=None, collision=None, depth_reader=None, blind_after_s=1.,
                 exclusions=(), tool_exclusion_m=0.):
        if collision is not None and not callable(depth_reader):
            raise GuardError("A collision guard needs a depth reader")
        self.effort, self.collision, self.depth_reader = effort, collision, depth_reader
        self.blind_after_s = float(blind_after_s)
        # Intended contact: balls around a grasp target, or around the tool
        # while it holds a part, inside which depth points are not obstacles.
        self.exclusions = [(tuple(float(v) for v in centre), float(radius)) for centre, radius in exclusions]
        if any(r <= 0 for _, r in self.exclusions) or float(tool_exclusion_m) < 0:
            raise GuardError("exclusion radii must be positive")
        self.tool_exclusion_m = float(tool_exclusion_m)
        self.last_frame_seen_at = None
        self.trips = []

    def on_progress(self, progress):
        if self.effort is not None:
            self.effort.on_progress(progress)

    def check(self, *, live, trajectory, elapsed_s, now):
        if live is None:
            return {"kind": "state_stale", "message": "no fresh joint state during motion"}
        if self.effort is not None:
            trip = self.effort.on_efforts(live.get("effort_nm"))
            if trip is not None:
                self.trips.append(trip)
                return trip
        if self.collision is not None:
            frame = self.depth_reader()
            if frame is None:
                if self.last_frame_seen_at is not None and now-self.last_frame_seen_at > self.blind_after_s:
                    return {"kind": "depth_blind", "message": "no depth frame for longer than the blind limit"}
                if self.last_frame_seen_at is None:
                    self.last_frame_seen_at = now
                return None
            depth_m, intrinsics_k, received_at = frame
            self.last_frame_seen_at = received_at
            exclusions = list(self.exclusions)
            if self.tool_exclusion_m > 0:
                exclusions.append((self.collision.tool_position(live["position_rad"]), self.tool_exclusion_m))
            trip = self.collision.check(depth_m=depth_m, intrinsics_k=intrinsics_k,
                                        frame_age_s=now-received_at, joints_now=live["position_rad"],
                                        trajectory=trajectory, elapsed_s=elapsed_s, exclusions=exclusions)
            if trip is not None:
                self.trips.append(trip)
                return trip
        return None
