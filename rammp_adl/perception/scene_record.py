"""Recorded wrist scenes and guard trips: what the offline bench replays without the arm.

A scene is one still keyframe as the runtime saw it: colour, depth in
millimetres, intrinsics, joints and the camera pose placed through the
measured mount. A guard trip is the depth frame, joints, trajectory and
exclusions a collision guard stopped on. Both stay on this machine under
artifacts/bench; nothing here uploads anything.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .keyframes import Keyframe

LAYOUT_FILE = "LAYOUT"
KEEP_SCENES = 600
KEEP_TRIPS = 200


def _stamp():
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())+f"{time.time() % 1:.3f}"[1:]


def _write_depth(path, depth_m):
    depth = np.asarray(depth_m, dtype=float)
    millimetres = np.where(np.isfinite(depth) & (depth > 0), np.clip(np.round(depth*1000.), 0, 65535), 0).astype(np.uint16)
    np.savez_compressed(path, depth_mm=millimetres)


def _read_depth(path):
    millimetres = np.load(path)["depth_mm"].astype(float)
    return np.where(millimetres > 0, millimetres/1000., np.nan)


def current_layout(root):
    """The operator's name for the current arrangement of the bench; scenes are compared within one layout."""
    try:
        return (Path(root)/"scenes"/LAYOUT_FILE).read_text().strip() or "default"
    except OSError:
        return "default"


def _prune(folder, keep):
    entries = sorted(p for p in folder.iterdir() if p.is_dir())
    for old in entries[:-keep]:
        for item in old.iterdir():
            item.unlink()
        old.rmdir()


def save_scene(root, keyframe, *, source, task_text="", extra=None):
    """Write one keyframe as a scene directory; returns its path. Never raises into the caller's thread."""
    try:
        folder = Path(root)/"scenes"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/f"{_stamp()}-{keyframe.capture_id[-12:]}"
        path.mkdir()
        cv2.imwrite(str(path/"rgb.jpg"), cv2.cvtColor(np.asarray(keyframe.rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        _write_depth(path/"depth.npz", keyframe.depth_m)
        meta = {"capture_id": keyframe.capture_id, "frame_id": keyframe.frame_id, "stamp_ns": int(keyframe.stamp_ns),
                "intrinsics_k": [float(v) for v in keyframe.intrinsics_k], "joints_rad": [float(v) for v in keyframe.joints_rad],
                "base_from_camera": np.asarray(keyframe.base_from_camera, dtype=float).tolist(), "reason": keyframe.reason,
                "source": source, "task_text": task_text, "layout": current_layout(root),
                "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})}
        (path/"meta.json").write_text(json.dumps(meta, indent=1)+"\n")
        _prune(folder, KEEP_SCENES)
        return path
    except Exception:                                   # noqa: BLE001 - recording is evidence, never a failure
        return None


def load_scene(path):
    """(Keyframe, meta) for a recorded scene; the keyframe is stamped as captured now, for replay."""
    path = Path(path)
    meta = json.loads((path/"meta.json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(path/"rgb.jpg")), cv2.COLOR_BGR2RGB)
    depth = _read_depth(path/"depth.npz")
    now = time.monotonic()
    keyframe = Keyframe(meta["capture_id"], rgb, depth, tuple(meta["intrinsics_k"]), meta["frame_id"], meta["stamp_ns"],
                        now, now, 0., np.asarray(meta["base_from_camera"], dtype=float), tuple(meta["joints_rad"]),
                        "replayed", now)
    return keyframe, meta


def list_scenes(root):
    folder = Path(root)/"scenes"
    return sorted(p for p in folder.iterdir() if (p/"meta.json").is_file()) if folder.is_dir() else []


def save_guard_trip(root, *, depth_frame, joints_rad, trajectory, elapsed_s, exclusions, tool_exclusion_m, trip, context):
    """Write what a collision guard stopped on; returns the path or None."""
    try:
        folder = Path(root)/"trips"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/_stamp()
        path.mkdir()
        meta = {"joints_rad": [float(v) for v in joints_rad], "elapsed_s": float(elapsed_s),
                "exclusions": [[list(map(float, centre)), float(radius)] for centre, radius in exclusions],
                "tool_exclusion_m": float(tool_exclusion_m), "trip": trip, "context": context,
                "trajectory": {"joint_names": list(trajectory.joint_names), "provenance": trajectory.provenance,
                               "points": [[p.time_s, list(p.state.position), list(p.state.velocity), list(p.state.acceleration)]
                                          for p in trajectory.points]},
                "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if depth_frame is not None:
            depth_m, intrinsics_k, _ = depth_frame
            _write_depth(path/"depth.npz", depth_m)
            meta["intrinsics_k"] = [float(v) for v in intrinsics_k]
        (path/"meta.json").write_text(json.dumps(meta, indent=1, default=str)+"\n")
        _prune(folder, KEEP_TRIPS)
        return path
    except Exception:                                   # noqa: BLE001 - recording is evidence, never a failure
        return None


def default_root():
    """artifacts/bench of whichever checkout this process runs from (a research worktree links the main one)."""
    return Path(os.getcwd())/"artifacts/bench"
