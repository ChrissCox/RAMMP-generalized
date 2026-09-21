"""A tiny depth renderer: a table plane at z=0 plus axis-aligned boxes, seen from a posed pinhole camera."""
import json

import numpy as np

from rammp_adl.perception.ros_rgbd import RgbdPair

from test_ros_rgbd import pair as base_pair

K = (300., 0., 160., 0., 300., 120., 0., 0., 1.)
SIZE = (240, 320)


def looking_down(position):
    """Camera z down, camera x along base x, at `position` (base frame)."""
    transform = np.eye(4)
    transform[:3, :3] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    transform[:3, 3] = position
    return transform


def looking_at(position, target):
    """Camera z toward target, camera y as close to down as possible."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    z = target-position
    z /= np.linalg.norm(z)
    down = np.array([0., 0., -1.])
    y = down-(down @ z)*z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x, y, z])
    transform[:3, 3] = position
    return transform


def render(base_from_camera, boxes, *, size=SIZE, k=K, plane=("z", 0.)):
    """Depth (camera z) per pixel for one infinite plane and axis-aligned boxes.

    plane is (axis, value), e.g. ("z", 0.) for a floor or ("x", .6) for a
    wall facing the robot. A box is (x0, x1, y0, y1, height) standing on the
    floor, or (x0, x1, y0, y1, z0, z1) in full.
    """
    height, width = size
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    v, u = np.mgrid[0:height, 0:width]
    directions = np.stack([(u-cx)/fx, (v-cy)/fy, np.ones_like(u, dtype=float)], axis=-1)
    rotation, origin = base_from_camera[:3, :3], base_from_camera[:3, 3]
    world = directions @ rotation.T
    depth = np.full(size, np.nan)
    axis = "xyz".index(plane[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = (plane[1]-origin[axis])/world[..., axis]
    hit = (t_plane > 0) & np.isfinite(t_plane)
    depth[hit] = t_plane[hit]
    for box in boxes:
        if len(box) == 5:
            x0, x1, y0, y1, h = box
            z0, z1 = 0., h
        else:
            x0, x1, y0, y1, z0, z1 = box
        lower, upper = np.array([x0, y0, z0]), np.array([x1, y1, z1])
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (lower-origin)/world
            t1 = (upper-origin)/world
        near = np.max(np.minimum(t0, t1), axis=-1)
        far = np.min(np.maximum(t0, t1), axis=-1)
        inside = (near <= far) & (far > 0)
        t_box = np.where(near > 0, near, far)
        take = inside & (np.isnan(depth) | (t_box < depth))
        depth[take] = t_box[take]
    return depth


def normalized_box(base_from_camera, box, *, size=SIZE, k=K, margin_px=2):
    """Projected bounding box of one object's corners, normalized to the image."""
    if len(box) == 5:
        x0, x1, y0, y1, h = box
        z0, z1 = 0., h
    else:
        x0, x1, y0, y1, z0, z1 = box
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    camera = (corners-base_from_camera[:3, 3]) @ base_from_camera[:3, :3]
    u = k[0]*camera[:, 0]/camera[:, 2]+k[2]
    v = k[4]*camera[:, 1]/camera[:, 2]+k[5]
    height, width = size
    return [float(max(0., (u.min()-margin_px)/width)), float(max(0., (v.min()-margin_px)/height)),
            float(min(1., (u.max()+margin_px)/width)), float(min(1., (v.max()+margin_px)/height))]


def synthetic_pair(depth, *, capture_id="cap-1", stamp_ns=1_000_000_000_000, receipt=100.02, k=K):
    meta = base_pair().metadata
    height, width = depth.shape
    for key in ("rgb_info", "depth_info"):
        meta[key].update(width=width, height=height, k=list(k), d=[0.]*5, frame_id="d405_color_optical_frame")
    meta["source_stamps_ns"] = {"rgb": stamp_ns, "depth": stamp_ns}
    meta["received_at_monotonic_s"] = {"rgb": receipt, "depth": receipt}
    rgb = np.full((height, width, 3), 200, np.uint8)
    return RgbdPair(capture_id, rgb, depth, json.dumps(meta))
