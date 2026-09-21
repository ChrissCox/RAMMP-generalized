"""A cabinet door seen from the front: handle grasp, measured door, hinge at the far edge."""
import asyncio
import unittest
from pathlib import Path

import numpy as np

from rammp_adl.constraints import waypoints
from rammp_adl.intake import draft_context
from rammp_adl.motion.kinematics import UrdfChain, quaternion_matrix
from rammp_adl.perception.keyframes import KeyframeSelector, StampReceiptClock
from rammp_adl.reasoning import ReasoningResult
from rammp_adl.constraints import ConstraintStore
import tempfile

from synthetic_scene import looking_at, normalized_box, render, synthetic_pair
from test_grounded_scene import BENCH, FixedPoseScene, NoFace, StillClient, MONO, STAMP, WALL

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2"
DOOR_X = .6
HANDLE = (.565, .6, .08, .12, .25, .35)                 # a bar protruding 3.5 cm toward the robot
DOOR = (.6, .601, .0, .40, .15, .45)                     # the door face within the view; the handle sits 10 cm from its y=0 edge
CAMERA = looking_at([.1, .1, .35], [.6, .1, .3])         # looking along +x at the door


class DeclinesUnlessDoor:
    """Astra stand-in: names the door and its handle; proposes a hinge only when the door is measured."""
    def __init__(self, pose):
        self.pose, self.calls = pose, []

    async def discover_scene(self, context, images, *, task_text=""):
        self.calls.append("discover")
        return ReasoningResult("OK", candidates=(
            {"image_id": images[0].image_id, "label": "wooden cabinet door", "kind": "surface", "attached_to": "",
             "box_xyxy_normalized": normalized_box(self.pose, DOOR), "confidence": .9},
            {"image_id": images[0].image_id, "label": "black door handle", "kind": "handle", "attached_to": "wooden cabinet door",
             "box_xyxy_normalized": normalized_box(self.pose, HANDLE), "confidence": .8}))

    async def propose_constraint(self, context, entity_id, *, label, geometry, door=None, history=(), images=()):
        self.calls.append(("propose", door is not None, len(images)))
        if door is None:
            return ReasoningResult("AMBIGUOUS", detail="only the pull is described")
        return ReasoningResult("AMBIGUOUS", detail="still unsure")   # exercise the local default


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class DoorSceneTests(unittest.TestCase):
    def setUp(self):
        self.now = MONO+.05
        self.chain = UrdfChain.from_path(BUNDLE/"arm-gripper-locked.urdf")
        self.clock = StampReceiptClock(wall=lambda: WALL, monotonic=lambda: MONO)
        self.scene = FixedPoseScene(client=StillClient(), chain=self.chain, calibration_id=BENCH["calibration_id"],
                                    clock_mapper=self.clock, face_screen=NoFace(), selector=KeyframeSelector(min_interval_s=1.))
        self.scene.pose = CAMERA
        self.scene.clock = lambda: self.now
        self.reasoner = DeclinesUnlessDoor(CAMERA)
        depth = render(CAMERA, [HANDLE], plane=("x", DOOR_X))
        self.scene.on_pair(synthetic_pair(depth, capture_id="cap-door", stamp_ns=STAMP, receipt=MONO+.02))
        self.folder = tempfile.TemporaryDirectory()
        self.store = ConstraintStore(self.folder.name)
        self.context = draft_context(BENCH, [], task_id="startup", camera_id="wrist_d405")

    def tearDown(self):
        self.folder.cleanup()

    def test_handle_gets_a_straight_in_grasp_and_the_door_is_measured(self):
        found = asyncio.run(self.scene.discover(self.reasoner, self.context, "open the cabinet"))
        self.assertEqual(sorted(found["entities"]), ["black_door_handle_1", "wooden_cabinet_door_1"])
        handle = self.scene.entities["black_door_handle_1"]
        self.assertEqual(handle["kind"], "handle")
        self.assertEqual(handle["grasp"]["strategy"], "top_down")                  # straight in along the door normal
        rotation = quaternion_matrix(tuple(handle["grasp"]["roles"]["grasp"]["orientation_xyzw"]))
        np.testing.assert_allclose(rotation[:, 2], [1., 0., 0.], atol=.02)          # tool z into the door
        np.testing.assert_allclose(handle["position_m"], [.5825, .10, .30], atol=.012)
        door = self.scene.entities["wooden_cabinet_door_1"]
        self.assertIsNone(door["geometry"])
        surface = door["surface_geometry"]
        np.testing.assert_allclose(surface["normal"], [-1., 0., 0.], atol=1e-3)
        self.assertAlmostEqual(surface["width_m"], .40, delta=.06)     # the oblique box over-covers a little
        self.assertAlmostEqual(surface["height_m"], .30, delta=.06)
        measured = self.scene.door_for(handle)
        self.assertEqual(measured["entity_id"], "wooden_cabinet_door_1")
        offsets = measured["handle_offsets_m"]
        self.assertAlmostEqual(offsets["left"]+offsets["right"], measured["width_m"], places=6)
        self.assertLess(offsets["right"], offsets["left"])                         # looking along +x, the y=0 edge is image-right

    def test_articulation_uses_the_measured_door_and_falls_back_locally(self):
        asyncio.run(self.scene.discover(self.reasoner, self.context, "open the cabinet"))
        result = asyncio.run(self.scene.articulate(self.reasoner, self.context, "black_door_handle_1", store=self.store))
        self.assertEqual(self.reasoner.calls[-1], ("propose", True, 1))            # the door and one keyframe went along
        record = result["record"]
        self.assertEqual(record["hinge_side"], "left")                             # far from the handle
        self.assertEqual(record["opening"], "pull")
        self.assertIn("local default", result["proposal"]["rationale"])
        self.assertEqual(result["surface"]["entity_id"], "wooden_cabinet_door_1")
        self.assertEqual(record["surface_entity_id"], "wooden_cabinet_door_1")
        door = result["door"]
        pivot = np.asarray(record["pivot_base"])
        self.assertAlmostEqual(np.linalg.norm(pivot-np.asarray(record["handle_position_m"])), door["handle_offsets_m"]["left"], places=6)
        self.assertAlmostEqual(abs(pivot[0]-DOOR_X), .02, delta=.03)                # on the door plane, near its far edge
        np.testing.assert_allclose(record["axis_base"], [0., 0., 1.], atol=1e-3)
        first = waypoints(record, record["handle_position_m"], (0., 0., 0., 1.), .3)[-1]
        self.assertLess(first[1][0], record["handle_position_m"][0])               # pulling brings the handle toward the robot
        self.assertIsNotNone(self.store.load("wooden cabinet door"))


if __name__ == "__main__":
    unittest.main()
