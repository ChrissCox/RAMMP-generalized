"""Constraints the AI proposes, the arm follows, and outcomes refine.

A constraint is a metric motion model for one grasped part: a hinge axis and
pivot for a door, a slide axis for a drawer. Astra proposes it from what it
sees and from the recorded history of earlier attempts on the same kind of
part; local geometry turns the proposal into base-frame numbers; the backend
follows it as cuRobo-planned waypoints under an effort budget; and every
attempt's outcome goes back into the store so the next proposal starts from
evidence, not from scratch.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .contracts import ContractError, checked_copy, digest
from .motion.kinematics import quaternion_matrix, quaternion_xyzw_from_matrix

STORE_DIR = "artifacts/constraints"
KINDS = ("revolute", "prismatic")
HINGE_SIDES = ("left", "right", "top", "bottom", "none")
OPENINGS = ("pull", "push", "slide_left", "slide_right", "slide_up", "slide_down")
DEFAULT_STEP = {"revolute": math.radians(5.), "prismatic": .02}
HANDLE_EDGE_OFFSET_M = .05      # handles sit near the free edge; the pivot is a width away, less this
WIDTH_BOUNDS_M = (.10, 1.20)


def slug(label):
    text = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return text[:40] or "part"


def turn_between(normal_before, normal_after, axis):
    """The unsigned angle a surface turned about the axis, from its normals before and after."""
    axis = _unit(axis, "constraint axis")
    def flatten(normal):
        vector = np.asarray(normal, dtype=float)
        vector = vector-(vector @ axis)*axis
        norm = np.linalg.norm(vector)
        return None if norm < 1e-6 else vector/norm
    a, b = flatten(normal_before), flatten(normal_after)
    if a is None or b is None:
        return None
    return float(math.atan2(np.linalg.norm(np.cross(a, b)), float(np.clip(a @ b, -1., 1.))))


def rotation_about(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    k = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3)+math.sin(angle)*k+(1.-math.cos(angle))*(k @ k)


def _unit(vector, label):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm < 1e-9:
        raise ContractError(f"{label} is not a usable direction")
    return vector/norm


def image_axes(geometry, base_from_camera):
    """In-plane 'right' and 'up' as the camera saw the part, plus the normal toward the camera."""
    normal = _unit(geometry["up"], "surface normal")
    transform = np.asarray(base_from_camera, dtype=float)
    right = transform[:3, 0]-(transform[:3, 0] @ normal)*normal
    up = -transform[:3, 1]-(-transform[:3, 1] @ normal)*normal
    if np.linalg.norm(right) < 1e-6 or np.linalg.norm(up) < 1e-6:
        raise ContractError("the camera looked along the surface; its image axes do not lie in the plane")
    return {"normal": normal, "right": right/np.linalg.norm(right), "up": up/np.linalg.norm(up)}


def metric_constraint(proposal, geometry, base_from_camera, *, constraint_id, entity_id, label,
                      surface_entity_id, handle_edge_offset_m=HANDLE_EDGE_OFFSET_M, door=None):
    """Turn Astra's proposal into a base-frame record the backend can follow.

    With a measured door, the pivot sits on the hinge-side edge exactly as
    far from the handle as the depth measured, scaled by the proposal's
    door_width_m relative to the measured width so refinement still bites.
    Without one, the pivot is a proposed width from the handle, less the
    usual edge offset.
    """
    kind = proposal["kind"]
    if kind not in KINDS:
        raise ContractError("unsupported constraint kind")
    axes = image_axes(geometry, base_from_camera)
    centroid = np.asarray(geometry["centroid_m"], dtype=float)
    width = float(proposal.get("door_width_m", .4))
    width = min(max(width, WIDTH_BOUNDS_M[0]), WIDTH_BOUNDS_M[1])
    opening = proposal["opening"]
    record = {"constraint_id": constraint_id, "entity_id": entity_id, "label": label,
              "surface_entity_id": surface_entity_id, "kind": kind,
              "unit": "rad" if kind == "revolute" else "m",
              "minimum": 0., "maximum": float(proposal["range"]),
              "step": DEFAULT_STEP[kind], "contact_effort_nm": float(proposal["contact_effort_nm"]),
              "hinge_side": proposal.get("hinge_side", "none"), "opening": opening, "door_width_m": width,
              "handle_position_m": centroid.tolist(), "surface_normal": axes["normal"].tolist(),
              "image_right": axes["right"].tolist(), "image_up": axes["up"].tolist(),
              "proposal": checked_copy(proposal), "parameters_version": 1, "attempts": [],
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if kind == "revolute":
        side = proposal.get("hinge_side", "none")
        toward = {"left": -axes["right"], "right": axes["right"], "top": axes["up"], "bottom": -axes["up"]}.get(side)
        if toward is None:
            raise ContractError("a hinge needs a side")
        axis = axes["up"] if side in ("left", "right") else axes["right"]
        if door is not None and side in door.get("handle_offsets_m", {}):
            scale = width/float(door["width_m"]) if door.get("width_m") else 1.
            reach = float(door["handle_offsets_m"][side])*scale
            record["measured_door"] = {"entity_id": door.get("entity_id"), "width_m": door.get("width_m"),
                                       "height_m": door.get("height_m"), "handle_offsets_m": door.get("handle_offsets_m")}
        else:
            reach = width-handle_edge_offset_m
        pivot = centroid+toward*max(reach, .02)
        # The sign that moves the handle toward the camera is a pull.
        lever = centroid-pivot
        moved = rotation_about(axis, .05) @ lever-lever
        direction = 1. if (moved @ axes["normal"] > 0) == (opening == "pull") else -1.
        record.update(axis_base=axis.tolist(), pivot_base=pivot.tolist(), direction=direction)
    else:
        axis = {"pull": axes["normal"], "push": -axes["normal"], "slide_left": -axes["right"],
                "slide_right": axes["right"], "slide_up": axes["up"], "slide_down": -axes["up"]}[opening]
        record.update(axis_base=axis.tolist(), pivot_base=None, direction=1.)
    record["digest"] = digest({k: v for k, v in record.items() if k not in ("attempts", "created_utc", "digest")})
    return record


def context_constraint(record, *, confidence=.6):
    """The symbolic constraint the world context and the plan schema use."""
    return {"constraint_id": record["constraint_id"], "entity_id": record["entity_id"], "kind": record["kind"],
            "unit": record["unit"], "minimum": record["minimum"], "maximum": record["maximum"],
            "validity": "true", "confidence": float(confidence)}


def waypoints(record, tool_position, tool_orientation_xyzw, target_value):
    """Tool poses along the constraint from the current tool pose up to the target."""
    step = float(record["step"])
    if not 0 < step or not math.isfinite(target_value) or target_value < 0:
        raise ContractError("constraint step and target must be positive")
    axis = _unit(record["axis_base"], "constraint axis")
    direction = float(record["direction"])
    position0 = np.asarray(tool_position, dtype=float)
    rotation0 = quaternion_matrix(tuple(float(v) for v in tool_orientation_xyzw))
    count = max(1, int(math.ceil(target_value/step-1e-9)))
    poses = []
    for index in range(1, count+1):
        value = min(target_value, index*step)
        if record["kind"] == "revolute":
            pivot = np.asarray(record["pivot_base"], dtype=float)
            rotation = rotation_about(axis, direction*value)
            position = pivot+rotation @ (position0-pivot)
            orientation = quaternion_xyzw_from_matrix(rotation @ rotation0)
        else:
            position = position0+axis*direction*value
            orientation = quaternion_xyzw_from_matrix(rotation0)
        poses.append((float(value), tuple(float(v) for v in position), tuple(float(v) for v in orientation)))
    return poses


PROGRESS_RUBRIC = ("0: the part has not been touched or moved", "1: the part is grasped but has not moved",
                   "2: the part moved less than half of the way", "3: the part moved at least half of the way",
                   "4: the part reached the target and the motion is verified")


def progress_score(*, achieved, target, grasped, verified):
    """The local 0 to 4 score behind the rubric; 4 needs the target and no contradicting measurement."""
    if not grasped:
        return 0
    if achieved <= 0.:
        return 1
    if achieved >= target-1e-9 and verified is not False:
        return 4
    return 3 if achieved >= .5*target else 2


def record_attempt(record, *, task_id, target, achieved, status, detail="", peak_effort_nm=None, trip=None, progress=None):
    attempt = {"task_id": task_id, "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "target": float(target), "achieved": float(achieved), "status": status, "detail": detail[:300],
               "peak_effort_nm": peak_effort_nm, "trip": checked_copy(trip) if trip is not None else None,
               "progress": checked_copy(progress) if progress is not None else None,
               "parameters": {"door_width_m": record.get("door_width_m"), "hinge_side": record.get("hinge_side"),
                              "opening": record.get("opening"), "step": record.get("step"),
                              "contact_effort_nm": record.get("contact_effort_nm"),
                              "parameters_version": record.get("parameters_version")}}
    record.setdefault("attempts", []).append(attempt)
    record["updated_utc"] = attempt["at_utc"]
    return attempt


def history_summary(record, *, limit=8):
    """What Astra and the local rule see: parameters and how far each attempt got."""
    attempts = record.get("attempts", [])[-limit:]
    return [{"parameters": a["parameters"], "target": a["target"], "achieved": a["achieved"], "status": a["status"],
             "peak_effort_nm": a.get("peak_effort_nm"), "detail": a.get("detail", ""), "progress": a.get("progress")}
            for a in attempts]


def next_parameters(record):
    """A local refinement when Astra repeats parameters that already failed.

    The pivot distance is the one number a hinge outcome is most sensitive to:
    a wrong radius drags the handle across the arc until the effort budget
    trips. Move the width toward whichever neighbouring attempt got further,
    or step away from a failed value when there is nothing better yet.
    """
    attempts = [a for a in record.get("attempts", []) if a["parameters"].get("door_width_m") is not None]
    if not attempts or record["kind"] != "revolute":
        return None
    last = attempts[-1]
    if last["status"] == "succeeded" and last["achieved"] >= last["target"]-1e-9:
        return None
    by_width = {}
    for attempt in attempts:
        width = float(attempt["parameters"]["door_width_m"])
        by_width[width] = max(by_width.get(width, -1.), float(attempt["achieved"]))
    widths = sorted(by_width)
    current = float(record["door_width_m"])
    best = max(widths, key=lambda w: (by_width[w], -abs(w-current)))
    untried_up, untried_down = best*1.15, best*.85
    if by_width.get(best, 0.) <= 0. and len(widths) >= 2:
        # Nothing has moved the part yet: alternate outward from the best guess.
        candidates = [w for w in (untried_up, untried_down) if all(abs(w-x) > 1e-3 for x in widths)]
        proposal = candidates[0] if candidates else best
    else:
        # Prefer the side of the best width that has not been explored.
        larger = [w for w in widths if w > best]
        smaller = [w for w in widths if w < best]
        proposal = untried_up if not larger else untried_down if not smaller else best
    proposal = min(max(proposal, WIDTH_BOUNDS_M[0]), WIDTH_BOUNDS_M[1])
    if abs(proposal-current) < 1e-3:
        return None
    return {"door_width_m": proposal}


def apply_parameters(record, geometry, base_from_camera, parameters, *, door=None):
    """Re-derive the metric record with new parameters; the history stays."""
    proposal = {**record["proposal"], **{k: v for k, v in parameters.items() if k in ("door_width_m", "hinge_side", "opening", "range", "contact_effort_nm")}}
    if door is None and record.get("measured_door"):
        door = record["measured_door"]
    updated = metric_constraint(proposal, geometry, base_from_camera, constraint_id=record["constraint_id"],
                                entity_id=record["entity_id"], label=record["label"],
                                surface_entity_id=record["surface_entity_id"], door=door)
    updated["attempts"] = record.get("attempts", [])
    updated["parameters_version"] = record.get("parameters_version", 1)+1
    updated["created_utc"] = record.get("created_utc", updated["created_utc"])
    return updated


class ConstraintStore:
    """One JSON record per part label, kept across runs."""

    def __init__(self, root):
        self.directory = Path(root)/STORE_DIR

    def path(self, label):
        return self.directory/f"{slug(label)}.json"

    def load(self, label):
        path = self.path(label)
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def save(self, record):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(record["label"])
        path.write_text(json.dumps(checked_copy(record), indent=2, allow_nan=False)+"\n")
        return path

    def labels(self):
        return sorted(p.stem for p in self.directory.glob("*.json")) if self.directory.is_dir() else []


def save_demonstration(store, record, crops, *, attempt_index=None):
    """Keep the screened keyframes of a successful attempt next to the record.

    Later proposals for the same kind of part see the first and last frame of
    the last success. The frames were screened for faces before they were
    ever encoded, and they are stored exactly as sent.
    """
    if not crops:
        return None
    index = len(record.get("attempts", [])) if attempt_index is None else int(attempt_index)
    folder = store.directory/slug(record["label"])/f"demo-{index:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    for order, crop in enumerate(crops):
        name = f"{order:02d}-{crop.image_id}.jpg"
        (folder/name).write_bytes(crop.jpeg_bytes)
        manifest.append({"file": name, "image_id": crop.image_id, "original_image_id": crop.original_image_id,
                         "camera_id": crop.camera_id, "source_frame": crop.source_frame, "calibration_id": crop.calibration_id,
                         "captured_at": crop.captured_at, "crop_xyxy": list(crop.crop_xyxy), "width": crop.width, "height": crop.height})
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    record["demonstration"] = {"folder": str(folder.relative_to(store.directory)), "frames": len(manifest),
                               "attempt_index": index}
    return folder


def load_demonstration(store, record, *, limit=2):
    """The stored frames of the record's demonstration as egress-ready crops (first and last)."""
    from .perception.images import ImageCrop
    demo = record.get("demonstration")
    if not demo:
        return []
    folder = store.directory/demo["folder"]
    manifest_path = folder/"manifest.json"
    if not manifest_path.is_file():
        return []
    entries = json.loads(manifest_path.read_text())
    if not entries:
        return []
    chosen = entries if len(entries) <= limit else [entries[0], entries[-1]][:limit]
    crops = []
    for entry in chosen:
        path = folder/entry["file"]
        if not path.is_file():
            continue
        crops.append(ImageCrop("demo-"+entry["image_id"], entry["original_image_id"], entry["camera_id"], entry["source_frame"],
                               entry["calibration_id"], float(entry["captured_at"]), tuple(entry["crop_xyxy"]), int(entry["width"]),
                               int(entry["height"]), path.read_bytes(), False, False))
    return crops
