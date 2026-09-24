"""A synthetic cabinet door recorded from several wrist views, and an Astra stand-in that names what the renderer drew."""
import numpy as np

from rammp_adl.perception.keyframes import Keyframe
from rammp_adl.perception.scene_record import save_scene
from rammp_adl.reasoning import ReasoningResult
from synthetic_scene import K, looking_at, normalized_box, render

DOOR_X = .6
HANDLE = (.565, .6, .08, .12, .25, .35)                 # a vertical bar 3.5 cm proud of the door face
DOOR = (.6, .601, -.05, .30, .12, .48)
HANDLE_CENTRE = np.array([(.565+.6)/2, .10, .30])
# Views of the door: most see the handle, the last looks away along the door.
VIEWS = [([.10, .10, .35], [.6, .10, .30]), ([.12, .02, .38], [.6, .08, .30]), ([.10, .18, .33], [.6, .12, .30]),
         ([.20, .10, .40], [.6, .10, .28]), ([.15, .05, .30], [.6, .10, .32]), ([.10, .10, .35], [.6, -.45, .30])]


def colour(depth, pose):
    """A picture to look at: wood where the door is, black where the handle stands proud, grey beyond."""
    rgb = np.full(depth.shape+(3,), (120, 124, 122), np.uint8)
    camera = np.asarray(pose)[:3, 3]
    distance = DOOR_X-camera[0]
    rgb[np.isfinite(depth) & (depth > 0)] = (176, 132, 88)
    rgb[np.isfinite(depth) & (depth < distance*.97)] = (28, 28, 30)
    return rgb


def keyframe(index, pose):
    depth = render(pose, [HANDLE], plane=("x", DOOR_X))
    return Keyframe(f"synthetic-{index:02d}", colour(depth, pose), depth, K, "d405_color_optical_frame", 1_000_000_000+index,
                    100., 100., .01, np.asarray(pose, dtype=float), (0., .26, -3.14, -2.27, 0., .96, 1.57), "synthetic", 100.)


def record(bench, views=VIEWS):
    poses = {}
    for index, (position, target) in enumerate(views):
        pose = looking_at(position, target)
        save_scene(bench, keyframe(index, pose), source="synthetic")
        poses["keyframe-"+f"synthetic-{index:02d}"] = pose
    return poses


def in_view(pose, point):
    camera = np.linalg.inv(pose) @ np.append(point, 1.)
    if camera[2] <= 0:
        return False
    u, v = K[0]*camera[0]/camera[2]+K[2], K[4]*camera[1]/camera[2]+K[5]
    return 0 <= u < 320 and 0 <= v < 240


class SyntheticAstra:
    """Names the door and the handle from the renderer's geometry; can be told to miss or invent one."""

    def __init__(self, poses, *, miss=(), invent=()):
        self.poses, self.miss, self.invent, self.calls = poses, set(miss), set(invent), []

    async def discover_scene(self, context, images, *, task_text=""):
        image = images[0]
        pose = self.poses[image.image_id]
        self.calls.append(image.image_id)
        visible = in_view(pose, HANDLE_CENTRE)
        candidates = [{"image_id": image.image_id, "label": "wooden cabinet door", "kind": "surface", "attached_to": "",
                       "box_xyxy_normalized": normalized_box(pose, DOOR), "confidence": .9}] if visible else []
        if (visible and image.image_id not in self.miss) or image.image_id in self.invent:
            box = normalized_box(pose, HANDLE) if visible else [.45, .40, .55, .70]
            candidates.append({"image_id": image.image_id, "label": "black bar handle", "kind": "handle",
                               "attached_to": "wooden cabinet door", "box_xyxy_normalized": box, "confidence": .8})
        seen = any(c["kind"] == "handle" for c in candidates)
        return ReasoningResult("OK" if candidates else "NO_DETECTION", candidates=tuple(candidates),
                               proposal={"target_visible": seen, "search_hint": "none" if seen else "left", "search_note": ""})
