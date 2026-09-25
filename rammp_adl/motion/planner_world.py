"""The planner's collision world for moves made after a part has been moved: the part where it now is.

cuRobo plans against the world the planner container holds (world_real_bench.yaml: the pedestal and the
table). A door the arm has swung open is not in it, so the way home can run through the door's edge, and
only the depth guard stops it. Here a swung door becomes an oriented box at its angle, added to the bench's
base obstacles in a scene YAML the planner loads by path (its world directory is mounted into the
container). The panel spans from the hinge line through the handle to the free edge, the height the depth
measured; the angle is the one the wrist camera measured when it could, else the commanded one widened by
a second box at the angle a released door springs back to.
"""
from __future__ import annotations

import math

import numpy as np

#: The planner's own base world, restored after a move made with moved parts in it.
BASE_WORLD = "world_real_bench.yaml"
#: The same obstacles as that world (rammp_curobo/configs/world_real_bench.yaml, read 2026-09-25): a scene
#: installed by path replaces the base one, so it must carry them too.
BASE_OBSTACLES = ({"name": "pedestal", "position": [0., 0., -.05], "dims": [.14, .14, .04]},
                  {"name": "table", "position": [.15, 0., -.10], "dims": [1.3, 1.4, .06]})
FREE_EDGE = {"left": "right", "right": "left"}


def rotation_about(axis, angle):
    axis = np.asarray(axis, dtype=float)/np.linalg.norm(axis)
    k = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3)+math.sin(angle)*k+(1.-math.cos(angle))*(k @ k)


def door_panel(record, arc, angle, *, name, thickness=.03, margin=.02):
    """The door a revolute record describes, turned by angle about the arc's hinge: one obstacle dict, or None."""
    door = record.get("measured_door") or {}
    offsets = door.get("handle_offsets_m") or {}
    side = record.get("hinge_side")
    if record.get("kind") != "revolute" or side not in FREE_EDGE or not {"top", "bottom", FREE_EDGE[side]} <= set(offsets):
        return None
    axis = np.asarray(arc["axis_base"], dtype=float)
    axis /= np.linalg.norm(axis)
    pivot = np.asarray(arc["pivot_base"], dtype=float)
    handle = np.asarray(record["handle_position_m"], dtype=float)
    radial = handle-pivot
    radial -= (radial @ axis)*axis
    reach = float(np.linalg.norm(radial))
    if reach < .03:
        return None
    along = rotation_about(axis, float(arc.get("direction", 1.))*angle) @ (radial/reach)
    length = reach+float(offsets[FREE_EDGE[side]])
    bottom, top = handle[2]-float(offsets["bottom"]), handle[2]+float(offsets["top"])
    centre = pivot+along*length/2.
    centre[2] = (bottom+top)/2.
    return {"name": name, "position": [round(float(v), 4) for v in centre],
            "rpy_deg": [0., 0., round(math.degrees(math.atan2(along[1], along[0])), 3)],
            "dims": [round(length+margin, 4), round(thickness+margin, 4), round(top-bottom+margin, 4)]}


def scene_yaml(obstacles):
    """A scene the planner loads: the base obstacles and these, as YAML (JSON is YAML)."""
    import json
    lines = ["# Written by the RAMMP-generalized runtime: the bench's base obstacles and the parts it has moved.",
             "base_frame: base_link", "obstacles:"]
    for obstacle in list(BASE_OBSTACLES)+list(obstacles):
        lines.append("  - "+json.dumps(obstacle))
    lines += ["objects: []", "targets: []", ""]
    return "\n".join(lines)
