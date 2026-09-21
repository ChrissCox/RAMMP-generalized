"""Metric object geometry from the depth inside a grounded box.

Astra says what is where in the image; this module says where it is in the
world. Points inside the box are placed in the base frame through the
keyframe's still-arm camera pose, the support plane is removed, and the
remaining cluster yields a centroid, horizontal axes, extents and height.
Pose roles for the planner's tool_frame follow from that geometry alone.
"""
from __future__ import annotations

import math

import numpy as np

from ..motion.kinematics import quaternion_xyzw_from_matrix
from .geometry import PerceptionError

GRIPPER_OPEN_M = .085          # nominal Robotiq 2F-85 stroke, as the aperture map assumes
UP = np.array([0., 0., 1.])


def box_to_pixels(box_normalized, width, height):
    x0, y0, x1, y1 = box_normalized
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x0, y0, x1, y1)) or x0 >= x1 or y0 >= y1:
        raise PerceptionError("grounding box is not a normalized, positive-area box")
    return (int(math.floor(x0*width)), int(math.floor(y0*height)),
            max(int(math.ceil(x1*width)), int(math.floor(x0*width))+1), max(int(math.ceil(y1*height)), int(math.floor(y0*height))+1))


def points_in_region(keyframe, region_px, *, stride=2, min_range_m=.07, max_range_m=1.5):
    """Camera-frame points for valid depth pixels inside a pixel region."""
    x0, y0, x1, y1 = region_px
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(keyframe.width, x1), min(keyframe.height, y1)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((0, 3))
    k = np.asarray(keyframe.intrinsics_k, dtype=float).reshape(3, 3)
    depth = np.asarray(keyframe.depth_m, dtype=float)[y0:y1:stride, x0:x1:stride]
    v, u = np.mgrid[y0:y1:stride, x0:x1:stride]
    valid = np.isfinite(depth) & (depth >= min_range_m) & (depth <= max_range_m)
    z = depth[valid]
    x = (u[valid]-k[0, 2])/k[0, 0]*z
    y = (v[valid]-k[1, 2])/k[1, 1]*z
    return np.column_stack([x, y, z])


def to_base(points_camera, base_from_camera):
    points = np.asarray(points_camera, dtype=float)
    if not len(points):
        return points.reshape(0, 3)
    transform = np.asarray(base_from_camera, dtype=float)
    return points @ transform[:3, :3].T + transform[:3, 3]


def support_plane(points_base, *, camera_position=None, iterations=3, inlier_m=.02, min_points=200,
                  max_rms_m=.015, min_vertical=None, samples=150, seed=0):
    """The dominant plane in base frame with its normal toward the camera, or None.

    A table seen from above and a cabinet door seen from the front are the
    same thing here: the surface an object rests on or protrudes from. Its
    normal is the approach direction reversed.
    """
    points = np.asarray(points_base, dtype=float)
    if len(points) < min_points:
        return None
    # The largest plane, not the least-squares compromise: a door in front of
    # a far wall is two depths, and one fit through both is neither of them.
    rng = np.random.default_rng(seed)
    voters = points if len(points) <= 6000 else points[rng.choice(len(points), 6000, replace=False)]
    best, best_count = None, 0
    for a, b, c in voters[rng.integers(0, len(voters), size=(samples, 3))]:
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal/length
        count = int((np.abs((voters-a) @ normal) <= inlier_m).sum())
        if count > best_count:
            best, best_count = (a, normal), count
    if best is None:
        return None
    inliers = points[np.abs((points-best[0]) @ best[1]) <= inlier_m]
    if len(inliers) < min_points:
        return None
    for _ in range(iterations):
        centre = inliers.mean(axis=0)
        _, _, axes = np.linalg.svd(inliers-centre, full_matrices=False)
        normal = axes[-1]
        distance = (points-centre) @ normal
        selected = points[np.abs(distance) <= inlier_m]
        if len(selected) < min_points:
            return None
        inliers = selected
    centre = inliers.mean(axis=0)
    _, _, axes = np.linalg.svd(inliers-centre, full_matrices=False)
    normal = axes[-1]
    facing = (np.asarray(camera_position, dtype=float)-centre) if camera_position is not None else UP
    if normal @ facing < 0:
        normal = -normal
    if min_vertical is not None and normal @ UP < min_vertical:
        return None
    rms = float(np.sqrt(np.mean(((inliers-centre) @ normal)**2)))
    if not math.isfinite(rms) or rms > max_rms_m:
        return None
    return {"origin_m": centre.tolist(), "normal": normal.tolist(), "rms_m": rms, "inliers": int(len(inliers)),
            "vertical": bool(abs(normal @ UP) < .5)}


def object_geometry(points_base, plane, *, min_points=30, above_m=.01, band_m=.06):
    """Centroid, horizontal axes, extents and height of the object cluster."""
    points = np.asarray(points_base, dtype=float)
    if plane is not None:
        origin, normal = np.asarray(plane["origin_m"]), np.asarray(plane["normal"])
        elevation = (points-origin) @ normal
        points = points[elevation > above_m]
        up = normal
    else:
        # No support plane in view: keep the band around the nearest surface.
        if len(points) < min_points:
            raise PerceptionError("too few depth points inside the box")
        ranges = np.linalg.norm(points-points.mean(axis=0), axis=1)
        points = points[ranges <= np.median(ranges)+band_m]
        up = UP
    if len(points) < min_points:
        raise PerceptionError(f"{len(points)} object points above the support; {min_points} required")
    median = np.median(points, axis=0)
    spread = np.median(np.abs(points-median), axis=0)*1.4826+.02
    keep = np.all(np.abs(points-median) <= 3.*spread, axis=1)
    points = points[keep]
    if len(points) < min_points:
        raise PerceptionError("object points are too scattered to be one object")
    mean = points.mean(axis=0)
    # One viewpoint sees only some faces, so the visible surface is a partial
    # shell: its minimum-area footprint rectangle gives the object's axes,
    # extents and centre far better than the surface points' moments do.
    e1 = np.cross(up, [0., 0., 1.]) if abs(up @ [0., 0., 1.]) < .9 else np.cross(up, [1., 0., 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    horizontal = points-np.outer((points-mean) @ up, up)-mean
    planar = np.column_stack([horizontal @ e1, horizontal @ e2]).astype(np.float32)
    import cv2
    (cx, cy), _, angle = cv2.minAreaRect(planar)
    theta = math.radians(angle)
    a1 = math.cos(theta)*e1+math.sin(theta)*e2
    a2 = np.cross(up, a1)
    extent_1 = float((horizontal @ a1).max()-(horizontal @ a1).min())
    extent_2 = float((horizontal @ a2).max()-(horizontal @ a2).min())
    if extent_1 >= extent_2:
        major, minor, extent_major, extent_minor = a1, a2, extent_1, extent_2
    else:
        major, minor, extent_major, extent_minor = a2, -a1, extent_2, extent_1
    centre = mean+float(cx)*e1+float(cy)*e2
    origin = np.asarray(plane["origin_m"]) if plane is not None else mean
    elevation = (points-origin) @ up
    height = float(elevation.max()-(0. if plane is not None else elevation.min()))
    centroid = centre-up*((centre-origin) @ up)+up*(float(elevation.max())/2. if plane is not None else float((mean-origin) @ up))
    top = centroid-up*((centroid-origin) @ up)+up*float(elevation.max())
    return {"centroid_m": centroid.tolist(), "up": up.tolist(), "major_axis": major.tolist(), "minor_axis": minor.tolist(),
            "extent_major_m": extent_major, "extent_minor_m": extent_minor,
            "height_m": height, "top_m": top.tolist(), "points": int(len(points)),
            "support": None if plane is None else {"normal": plane["normal"], "rms_m": plane["rms_m"], "inliers": plane["inliers"]}}


def _frame(x_axis, z_axis):
    x = np.asarray(x_axis, dtype=float)
    z = np.asarray(z_axis, dtype=float)
    z /= np.linalg.norm(z)
    x -= (x @ z)*z
    if np.linalg.norm(x) < 1e-6:
        raise PerceptionError("degenerate tool frame")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.column_stack([x, y, z])
    return tuple(float(v) for v in quaternion_xyzw_from_matrix(rotation))


def lift_point(keyframe, xy_normalized, *, patch=3, min_range_m=.07, max_range_m=1.5):
    """A base-frame point under an image location, from the median depth of a small patch, or None."""
    u = int(round(float(xy_normalized[0])*(keyframe.width-1)))
    v = int(round(float(xy_normalized[1])*(keyframe.height-1)))
    window = np.asarray(keyframe.depth_m, dtype=float)[max(0, v-patch):v+patch+1, max(0, u-patch):u+patch+1]
    valid = window[np.isfinite(window) & (window >= min_range_m) & (window <= max_range_m)]
    if not valid.size:
        return None
    z = float(np.median(valid))
    k = np.asarray(keyframe.intrinsics_k, dtype=float).reshape(3, 3)
    camera = np.array([(u-k[0, 2])/k[0, 0]*z, (v-k[1, 2])/k[1, 1]*z, z])
    return to_base(camera[None, :], keyframe.base_from_camera)[0]


def pose_roles(geometry, *, camera_position_base, gripper_open_m=GRIPPER_OPEN_M, clearance_m=.01,
               top_grasp_max_height_m=.15, finger_depth_m=.03, pregrasp_m=.10, staging_m=.15,
               position_sigma_m=.015, rotation_sigma_rad=.1, grasp_point=None, tool_x_axis=None):
    """tool_frame poses for pregrasp, grasp, retract and staging from the object's geometry.

    Along the surface normal (top-down on a table, straight in on a door)
    when the object protrudes little and its minor width fits between the
    fingers; otherwise a side grasp along a footprint axis when the width
    across it fits; otherwise no grasp roles at all. retract equals pregrasp
    until a skill replaces it.
    """
    up = np.asarray(geometry["up"], dtype=float)
    centroid = np.asarray(geometry["centroid_m"], dtype=float)
    top = np.asarray(geometry["top_m"], dtype=float)
    minor = np.asarray(geometry["minor_axis"], dtype=float)
    fits_minor = geometry["extent_minor_m"]+clearance_m <= gripper_open_m
    strategy, position, x_axis, z_axis = "none", None, None, None
    if fits_minor and geometry["height_m"] <= top_grasp_max_height_m:
        strategy = "top_down"
        z_axis = -up
        x_axis = minor
        position = top-up*min(finger_depth_m, geometry["height_m"]/2.)
    else:
        # Approach along one measured horizontal axis so the fingers close
        # across the other, from whichever end faces the camera.
        toward = centroid-np.asarray(camera_position_base, dtype=float)
        toward -= (toward @ up)*up
        if np.linalg.norm(toward) < .05:
            toward = centroid-(centroid @ up)*up            # from the robot base outward
        major = np.asarray(geometry["major_axis"], dtype=float)
        options = [(major, geometry["extent_minor_m"]), (minor, geometry["extent_major_m"])]
        for axis, width_across in options:
            if width_across+clearance_m > gripper_open_m:
                continue
            approach = axis if axis @ toward >= 0 else -axis
            strategy = "side"
            z_axis = approach
            x_axis = np.cross(up, approach)
            position = centroid
            break
    if strategy == "none":
        return {"strategy": "none", "roles": {}, "reason": "object width exceeds the gripper opening"}
    grasp_point_used = False
    if grasp_point is not None:
        # The model's grasp point moves the closing point across the object;
        # depth along the approach still comes from the measured geometry.
        point = np.asarray(grasp_point, dtype=float)
        if np.isfinite(point).all() and np.linalg.norm(point-centroid) <= max(geometry["extent_major_m"], geometry["height_m"])+.03:
            offset = point-position
            position = position+offset-(offset @ z_axis)*z_axis
            grasp_point_used = True
    # A parallel jaw closes the same either way round, and a measured axis has no sign of its own:
    # take the way that turns the wrist least from where the tool is, else a fixed convention,
    # so two measurements of one object never ask for opposite wrists.
    reference = np.asarray(tool_x_axis, dtype=float) if tool_x_axis is not None else None
    if reference is None or abs(x_axis @ reference) < 1e-3:
        reference = np.array([0., 1., 0.]) if abs(x_axis[1]) > 1e-3 else np.array([1., 0., 0.]) if abs(x_axis[0]) > 1e-3 \
            else np.array([0., 0., 1.])
    if x_axis @ reference < 0:
        x_axis = -x_axis
    orientation = _frame(x_axis, z_axis)
    roles = {"grasp": (position, orientation),
             "pregrasp": (position-z_axis*pregrasp_m, orientation),
             "retract": (position-z_axis*pregrasp_m, orientation),
             "staging": (position+up*staging_m, orientation)}
    covariance = np.zeros((6, 6))
    covariance[:3, :3] = np.eye(3)*position_sigma_m**2
    covariance[3:, 3:] = np.eye(3)*rotation_sigma_rad**2
    return {"strategy": strategy,
            "roles": {name: {"position_m": [float(v) for v in pos], "orientation_xyzw": list(quat)} for name, (pos, quat) in roles.items()},
            "covariance": [float(v) for v in covariance.ravel()],
            "reason": "", "gripper_open_m": gripper_open_m, "grasp_point_used": grasp_point_used}
