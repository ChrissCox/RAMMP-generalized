"""Untagged objects end to end: discovery, carried positions, re-grounding, world commits, refresh."""
import asyncio
import json
import unittest
from pathlib import Path

import numpy as np

from rammp_adl.contracts import Catalog
from rammp_adl.handlers import BackendFailure, ExecutionContext
from rammp_adl.hardware_backend import HardwareObservationBackend
from rammp_adl.intake import draft_context, seed_visibility
from rammp_adl.motion.kinematics import UrdfChain
from rammp_adl.perception.grounded_scene import GroundedScene, SceneError, slug
from rammp_adl.perception.keyframes import KeyframeSelector, StampReceiptClock
from rammp_adl.reasoning import AstraReasoner, ReasoningResult
from rammp_adl.world import WorldModel

from synthetic_scene import looking_down, normalized_box, render, synthetic_pair
from test_intake import ScriptedTransport, completed

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2"
BENCH = json.loads((ROOT/"config/sheppy-bench.context.json").read_text())
WALL, MONO = 1000., 100.
STAMP = 1_000_000_000_000
BOX = (.27, .33, .10, .14, .08)


class NoFace:
    def screen(self, rgb):
        return {"contains_face": False, "faces": 0, "model_sha256": "fake"}


class StillClient:
    def __init__(self):
        self.still_since = MONO-10.

    def live_joints(self, *, max_age_s=None):
        return {"position_rad": (0.,)*7, "knuckle_rad": .01, "velocity_rad_s": (0.,)*7, "effort_nm": None, "received_at_monotonic_s": MONO}

    def still_since_s(self):
        return self.still_since


class FakeReasoner:
    """Astra stand-in: boxes come from the renderer's projection, never invented."""
    def __init__(self, scene_pose_getter, boxes):
        self.pose_getter, self.boxes, self.calls = scene_pose_getter, boxes, []

    async def discover_scene(self, context, images, *, task_text=""):
        self.calls.append(("discover", task_text, images[0].image_id))
        pose = self.pose_getter()
        candidates = []
        for label, box in self.boxes:
            rect = normalized_box(pose, box)
            candidates.append({"image_id": images[0].image_id, "label": label, "box_xyxy_normalized": rect, "confidence": .9,
                               "grasp_point_given": True, "grasp_point_xy_normalized": [(rect[0]+rect[2])/2., (rect[1]+rect[3])/2.]})
        return ReasoningResult("OK", candidates=tuple(candidates))

    async def ground_target(self, context, entity_id, images, *, query=""):
        self.calls.append(("ground", entity_id, images[0].image_id))
        pose = self.pose_getter()
        return ReasoningResult("OK", candidates=tuple({"image_id": images[0].image_id, "label": label, "entity_id": entity_id,
                                                       "box_xyxy_normalized": normalized_box(pose, box), "confidence": .8}
                                                      for label, box in self.boxes if slug(label) in entity_id))


class FixedPoseScene(GroundedScene):
    pose = looking_down([.3, 0., .6])

    def camera_pose(self):
        return self.pose, (0.,)*7


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class GroundedSceneTests(unittest.TestCase):
    def setUp(self):
        self.now = MONO+.05
        self.catalog = Catalog(ROOT)
        self.chain = UrdfChain.from_path(BUNDLE/"arm-gripper-locked.urdf")
        self.clock = StampReceiptClock(wall=lambda: WALL, monotonic=lambda: MONO)
        self.scene = FixedPoseScene(client=StillClient(), chain=self.chain, calibration_id=BENCH["calibration_id"],
                                    clock_mapper=self.clock, face_screen=NoFace(), selector=KeyframeSelector(min_interval_s=1.))
        self.scene.clock = lambda: self.now
        self.reasoner = FakeReasoner(lambda: self.scene.pose, [("small box", BOX)])
        self.scene.reasoner = self.reasoner
        self.base_context = draft_context(BENCH, [], task_id="startup", camera_id="wrist_d405")

    def feed(self, boxes=(BOX,), capture_id="cap-1", stamp_ns=STAMP, receipt=MONO+.02):
        sample = synthetic_pair(render(self.scene.pose, list(boxes)), capture_id=capture_id, stamp_ns=stamp_ns, receipt=receipt)
        return self.scene.on_pair(sample)

    def world(self):
        context = draft_context(BENCH, self.scene.descriptors(), task_id="task-1", camera_id=self.scene.camera_id)
        world = WorldModel(context, self.catalog, clock=lambda: self.now, trust_initial=False, max_evidence_age_s=120.)
        self.scene.bind_world(world)
        return world

    def execution_context(self, world):
        snapshot = world.snapshot()
        return ExecutionContext(snapshot.context["task_id"], "observe-1", snapshot=snapshot, execution_epoch=snapshot.execution_epoch)

    def test_discovery_names_and_places_the_object(self):
        self.assertEqual(self.feed().reason, "initial")
        found = asyncio.run(self.scene.discover(self.reasoner, self.base_context, "pick up the small box"))
        self.assertEqual(found["entities"], ["small_box_1"])
        self.assertEqual(found["descriptors"], [{"entity_id": "small_box_1", "label": "small box", "pose_roles": ["grasp", "pregrasp", "retract", "staging"]}])
        record = self.scene.entities["small_box_1"]
        np.testing.assert_allclose(record["position_m"][:2], [.30, .12], atol=.01)
        self.assertEqual(record["grasp"]["strategy"], "top_down")
        self.assertTrue(record["grasp"]["grasp_point_used"])                    # the model's point was lifted and used
        self.assertTrue(self.scene.visible_entities(self.now)["small_box_1"]["visible"])
        self.assertIsNone(self.scene.pending_refresh())                         # the keyframe was sent
        self.assertEqual(self.reasoner.calls[0][:2], ("discover", "pick up the small box"))

    def test_observation_reuses_the_discovery_keyframe_and_commits_three_poses(self):
        self.feed()
        asyncio.run(self.scene.discover(self.reasoner, self.base_context, "pick up the small box"))
        world = self.world()
        self.assertEqual(seed_visibility(world, self.scene, now=self.now)["visible"], ["small_box_1"])
        args = {"entity_id": "small_box_1", "camera": "wrist", "purpose": "pose"}
        measurement = asyncio.run(self.scene(args, self.execution_context(world)))
        self.assertEqual(sorted(p.pose_role for p in measurement.metric_poses), ["grasp", "pregrasp", "retract", "staging"])
        self.assertTrue(measurement.data["reused_keyframe"])
        self.assertEqual(measurement.data["frames_uploaded"], 0)
        self.assertEqual([c[0] for c in self.reasoner.calls], ["discover"])
        backend = HardwareObservationBackend(self.catalog, world, observer=self.scene, held_check=lambda: True)
        outcome = asyncio.run(backend.observe(args, self.execution_context(world)))
        self.assertEqual(len(outcome.metric_poses), 4)
        self.assertEqual(outcome.metric_poses[0].frame_id, "base_link")
        state = asyncio.run(self.scene({"entity_id": "small_box_1", "camera": "wrist", "purpose": "state"}, self.execution_context(world)))
        self.assertTrue(state.data["carried"]["consistent"])
        self.assertEqual(state.metric_poses, ())

    def test_a_later_keyframe_from_the_same_pose_keeps_the_measurement_without_a_cloud_call(self):
        # Intake's observation comes seconds after discovery: a new capture, the same still camera, the same object.
        self.feed()
        asyncio.run(self.scene.discover(self.reasoner, self.base_context, "pick up the small box"))
        world = self.world()
        self.now = MONO+12.05
        self.clock.note(STAMP+12_000_000_000, MONO+12.02)
        forced = self.scene.on_pair(synthetic_pair(render(self.scene.pose, [BOX]), capture_id="cap-2",
                                                   stamp_ns=STAMP+12_000_000_000, receipt=MONO+12.02), force=True)
        self.assertEqual(forced.capture_id, "cap-2")
        measurement = asyncio.run(self.scene({"entity_id": "small_box_1", "camera": "wrist", "purpose": "pose"}, self.execution_context(world)))
        self.assertTrue(measurement.data["reused_keyframe"])
        self.assertEqual(measurement.data["frames_uploaded"], 0)
        self.assertEqual([c[0] for c in self.reasoner.calls], ["discover"])
        # The object gone from that spot is not carried: it is grounded again.
        self.now = MONO+24.05
        self.clock.note(STAMP+24_000_000_000, MONO+24.02)
        self.scene.on_pair(synthetic_pair(render(self.scene.pose, []), capture_id="cap-3",
                                          stamp_ns=STAMP+24_000_000_000, receipt=MONO+24.02), force=True)
        self.reasoner.boxes = []
        with self.assertRaises(BackendFailure):
            asyncio.run(self.scene({"entity_id": "small_box_1", "camera": "wrist", "purpose": "pose"}, self.execution_context(world)))
        self.assertEqual([c[0] for c in self.reasoner.calls], ["discover", "ground"])

    def test_after_the_camera_moves_the_entity_is_re_grounded_once(self):
        self.feed()
        asyncio.run(self.scene.discover(self.reasoner, self.base_context, "pick up the small box"))
        world = self.world()
        self.scene.pose = looking_down([.36, .02, .55])
        with self.assertRaisesRegex(BackendFailure, "no still, fresh"):    # the old keyframe is not current any more
            asyncio.run(self.scene({"entity_id": "small_box_1", "camera": "wrist", "purpose": "pose"}, self.execution_context(world)))
        self.now = MONO+5.05
        self.clock.note(STAMP+5_000_000_000, MONO+5.02)
        moved = self.feed(capture_id="cap-2", stamp_ns=STAMP+5_000_000_000, receipt=MONO+5.02)
        self.assertEqual(moved.reason, "camera_moved")
        self.assertIsNotNone(self.scene.pending_refresh())
        measurement = asyncio.run(self.scene({"entity_id": "small_box_1", "camera": "wrist", "purpose": "pose"}, self.execution_context(world)))
        self.assertFalse(measurement.data["reused_keyframe"])
        self.assertEqual(measurement.data["frames_uploaded"], 1)
        self.assertEqual([c[0] for c in self.reasoner.calls], ["discover", "ground"])
        np.testing.assert_allclose(self.scene.entities["small_box_1"]["position_m"][:2], [.30, .12], atol=.015)
        self.assertIsNone(self.scene.pending_refresh())
        # A second pose observation from the same keyframe reuses it.
        asyncio.run(self.scene({"entity_id": "small_box_1", "camera": "wrist", "purpose": "pose"}, self.execution_context(world)))
        self.assertEqual(len(self.reasoner.calls), 2)

    def test_refresh_sends_only_unsent_selected_keyframes_and_carries_entities(self):
        self.feed()
        asyncio.run(self.scene.discover(self.reasoner, self.base_context, "look"))
        self.assertIsNone(asyncio.run(self.scene.refresh(self.reasoner, self.base_context)))
        self.now = MONO+5.05
        self.clock.note(STAMP+5_000_000_000, MONO+5.02)
        self.reasoner.boxes.append(("mug", (.40, .46, -.04, .02, .09)))
        self.feed(boxes=(BOX, (.40, .46, -.04, .02, .09)), capture_id="cap-2", stamp_ns=STAMP+5_000_000_000, receipt=MONO+5.02)
        found = asyncio.run(self.scene.refresh(self.reasoner, self.base_context))
        self.assertEqual(found["reason"], "scene_changed")
        self.assertEqual(sorted(self.scene.entities), ["mug_1", "small_box_1"])      # the box kept its id
        self.assertEqual(self.reasoner.calls[-1][:2], ("discover", ""))

    def test_egress_needs_a_screen_and_unknown_entities_are_refused(self):
        self.scene.face_screen = None
        self.feed()
        with self.assertRaises(SceneError) as caught:
            asyncio.run(self.scene.discover(self.reasoner, self.base_context, "look"))
        self.assertEqual(caught.exception.status, "EGRESS_REFUSED")
        world = self.world()
        with self.assertRaises(BackendFailure):
            asyncio.run(self.scene({"entity_id": "ghost_1", "camera": "wrist", "purpose": "pose"}, self.execution_context(world)))
        with self.assertRaises(SceneError):
            self.scene.latest = None
            self.scene._last_pair = None
            self.scene.keyframe_for_request()


class DiscoveryRequestTests(unittest.TestCase):
    """The provider exchange for discovery, with a scripted transport."""
    class Crop:
        image_id = "keyframe-1"

        def validate_for_egress(self, *, max_bytes, max_long_edge, allow_face):
            pass

        def cloud_metadata(self):
            return {"image_id": self.image_id}

        def as_openai_input(self):
            return {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/2Q==", "detail": "low"}

    def reasoner(self, *outcomes):
        return AstraReasoner(Catalog(ROOT), transport=ScriptedTransport(*outcomes), hold_assertion=lambda: True, epoch_getter=lambda _t: 1)

    def test_entities_are_returned_and_bad_boxes_regenerate(self):
        point = {"grasp_point_given": False, "grasp_point_xy_normalized": [0., 0.]}
        seen = {"target_visible": True, "search_hint": "none", "search_note": ""}
        good = {"entities": [{"image_id": "keyframe-1", "label": "  water   bottle ", "kind": "free_object", "attached_to": "",
                              "box_xyxy_normalized": [.1, .1, .3, .5], "confidence": .8,
                              "grasp_point_given": True, "grasp_point_xy_normalized": [.2, .3]},
                             {"image_id": "keyframe-1", "label": "handle", "kind": "handle", "attached_to": " cabinet  door ",
                              "box_xyxy_normalized": [.4, .4, .5, .6], "confidence": .7, **point}], **seen}
        bad = {"entities": [{"image_id": "keyframe-1", "label": "x", "kind": "free_object", "attached_to": "",
                             "box_xyxy_normalized": [.5, .5, .2, .2], "confidence": .8, **point}], **seen}
        outside = {"entities": [{"image_id": "keyframe-1", "label": "x", "kind": "free_object", "attached_to": "",
                                 "box_xyxy_normalized": [.1, .1, .2, .2], "confidence": .8,
                                 "grasp_point_given": True, "grasp_point_xy_normalized": [.9, .9]}], **seen}
        reasoner = self.reasoner(completed(bad), completed(good))
        result = asyncio.run(reasoner.discover_scene(BENCH, [self.Crop()], task_text="find the bottle"))
        self.assertEqual(result.status, "OK")
        self.assertEqual(result.candidates[0]["label"], "water bottle")
        self.assertEqual(result.candidates[1]["attached_to"], "cabinet door")
        unattached = completed({"entities": [{"image_id": "keyframe-1", "label": "knob", "kind": "handle", "attached_to": "",
                                              "box_xyxy_normalized": [.1, .1, .2, .2], "confidence": .5, **point}], **seen})
        orphan = self.reasoner(unattached, unattached)                      # regeneration gets one more chance
        astray = self.reasoner(completed(outside), completed(outside))
        self.assertEqual(asyncio.run(astray.discover_scene(BENCH, [self.Crop()])).status, "INVALID_OUTPUT")
        self.assertEqual(asyncio.run(orphan.discover_scene(BENCH, [self.Crop()])).status, "INVALID_OUTPUT")
        self.assertEqual(reasoner._budgets[BENCH["task_id"]].requests, 2)
        request = reasoner.transport.requests[0]
        self.assertEqual(request["text"]["format"]["name"], "adl_discovery")
        self.assertEqual(request["input"][0]["content"][1]["type"], "input_image")
        self.assertIn("Skip the robot", request["instructions"])
        empty = self.reasoner(completed({"entities": [], "target_visible": False, "search_hint": "left",
                                         "search_note": "  the wall continues to the left  "}))
        result = asyncio.run(empty.discover_scene(BENCH, [self.Crop()], task_text="open the cabinet"))
        self.assertEqual(result.status, "NO_DETECTION")
        self.assertEqual(result.proposal, {"target_visible": False, "search_hint": "left", "search_note": "the wall continues to the left"})
        self.assertEqual(asyncio.run(self.reasoner().discover_scene(BENCH, [])).status, "INVALID_OUTPUT")


if __name__ == "__main__":
    unittest.main()
