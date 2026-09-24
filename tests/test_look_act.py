"""Look-act alignment before the grasp approach, and verified, scored constraint following."""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rammp_adl.constraints import ConstraintStore, context_constraint, rotation_about, turn_between
from rammp_adl.contracts import Catalog
from rammp_adl.handlers import BackendFailure
from rammp_adl.intake import draft_context
from rammp_adl.motion.collision_guard import EffortGuard, GuardSet
from rammp_adl.motion.kinematics import UrdfChain
from rammp_adl.perception.images import ImageCrop
from rammp_adl.perception.keyframes import Keyframe
from rammp_adl.reasoning import ReasoningResult
from rammp_adl.sheppy_backend import SheppyArmBackend
from rammp_adl.world import MetricPose, WorldModel

from test_follow_constraint import PROFILES, TrackingClient, door_record
from test_sheppy_backend import execution_context

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2"
BENCH = json.loads((ROOT/"config/sheppy-bench.context.json").read_text())


def jpeg_crop(image_id):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (80, 80, 80)).save(buffer, format="JPEG")
    return ImageCrop(image_id, "cap-"+image_id, "wrist_d405", "d405_color_optical_frame", "cal", 100., (0, 0, 848, 480),
                     64, 48, buffer.getvalue(), False, False)


class ScriptedReasoner:
    def __init__(self, corrections=(), score=4):
        self.corrections, self.score, self.calls = list(corrections), score, []

    async def correct_pose(self, context, entity_id, *, label, state, images, step_limit_m, yaw_limit_deg):
        self.calls.append(("correct", entity_id, dict(state)))
        if not self.corrections:
            return ReasoningResult("OK", detail="centred", proposal={"status": "DONE"})
        step = self.corrections.pop(0)
        if step == "ABORT":
            return ReasoningResult("ABORT", detail="that is not the handle")
        if step == "DONE":
            return ReasoningResult("OK", detail="centred", proposal={"status": "DONE"})
        return ReasoningResult("OK", detail="shift", proposal={"status": "MOVE", "delta_camera_m": list(step[0]), "yaw_deg": step[1]})

    async def verify_progress(self, context, *, task_text, rubric, images, question=""):
        self.calls.append(("verify", task_text, len(images)))
        return ReasoningResult("OK", detail="the door stands open", proposal={"score": self.score, "confidence": .8})


class FakeScene:
    """Keyframes at a fixed camera pose; surface normals scripted per call."""
    def __init__(self, base_from_camera, normals=None):
        self.base_from_camera, self.normals, self.reasoner = np.asarray(base_from_camera, dtype=float), list(normals or []), None
        self.count = 0

    async def __call__(self, args, context):
        raise AssertionError("not observed in these tests")

    async def wait_for_keyframe(self, *, timeout_s=2., poll_s=.2, max_age_s=10.):
        self.count += 1
        rgb = np.zeros((48, 64, 3), np.uint8)
        return Keyframe(f"kf-{self.count}", rgb, np.full((48, 64), .3), (60., 0., 32., 0., 60., 24., 0., 0., 1.),
                        "d405_color_optical_frame", 1, 100., 100., .02, self.base_from_camera, (0.,)*7, "requested", 100.)

    def crop_for(self, keyframe):
        return jpeg_crop(keyframe.capture_id)

    async def surface_normal(self, **kwargs):
        keyframe = await self.wait_for_keyframe()
        if not self.normals:
            return None, keyframe
        normal = self.normals.pop(0) if len(self.normals) > 1 else self.normals[0]
        return (None if normal is None else list(normal)), keyframe


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class LookActTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(ROOT)
        self.chain = UrdfChain.from_path(BUNDLE/"arm-gripper-locked.urdf")
        self.folder = tempfile.TemporaryDirectory()
        self.store = ConstraintStore(self.folder.name)
        self.record = door_record()
        self.client = TrackingClient(knuckle=.01)
        camera = np.eye(4)                                  # camera axes aligned with base: image right is +x
        self.scene = FakeScene(camera)

    def tearDown(self):
        self.folder.cleanup()

    def backend(self, reasoner, **options):
        self.scene.reasoner = reasoner
        descriptors = [{"entity_id": "handle_1", "label": "handle", "pose_roles": ["grasp", "pregrasp", "retract", "staging"]},
                       {"entity_id": "cabinet_door_surface", "label": "cabinet door", "pose_roles": []}]
        context = draft_context(BENCH, descriptors, task_id="task-look", camera_id="wrist_d405",
                                constraints=[context_constraint(self.record)])
        world = WorldModel(context, self.catalog, trust_initial=False, max_evidence_age_s=600.)
        backend = SheppyArmBackend(self.catalog, world, client=self.client, profiles=PROFILES, chain=self.chain,
                                   observer=self.scene, constraints={self.record["constraint_id"]: self.record},
                                   constraint_store=self.store, speed_scales={"transit": 2., "contact": 4.},
                                   guard_factory=lambda touch_nm=3., exclusions=(), tool_exclusion_m=0.: GuardSet(
                                       effort=EffortGuard(touch_nm), exclusions=exclusions, tool_exclusion_m=tool_exclusion_m), **options)
        identities = world.snapshot().identities()
        authority = world.authorize_source("test_observation", self.catalog.predicates)
        world.register_evidence("pose-1", source=authority, ttl_s=120., observed_at=world.clock(),
                                predicates=[{"predicate": "pose_valid", "validity": "true", "args": {"entity_id": "handle_1", "pose_role": "grasp"}}])
        world.update_metric_pose(MetricPose("handle_1", "grasp", (.58, .1, .3), (0., 0., 0., 1.), tuple([0.]*36), world.clock(), "base_link",
                                            identities["entity:handle_1"]+1, identities["calibration_id"], identities["base_epoch"], "pose-1", 120.),
                                 source=authority)
        return backend, world

    def grasp_move(self, backend, world):
        return asyncio.run(backend.move_to_pose({"target": {"entity_id": "handle_1", "pose_role": "grasp"}, "profile_id": "bench_transit"},
                                                execution_context(world)))

    def test_a_shift_then_done_moves_the_standoff_and_the_target_together(self):
        reasoner = ScriptedReasoner(corrections=[((.02, 0., 0.), 0.), "DONE"])
        backend, world = self.backend(reasoner)
        outcome = self.grasp_move(backend, world)
        self.assertEqual(outcome.status, "succeeded")
        data = outcome.evidence[0]["data"]
        self.assertEqual(data["alignment"]["calls"], 2)
        self.assertTrue(data["alignment"]["converged"])
        np.testing.assert_allclose(data["alignment"]["shift_m"], [.02, 0., 0.])
        np.testing.assert_allclose(data["commanded"]["position_m"], [.60, .1, .3])
        self.assertEqual(len(self.client.sent), 2)                                          # the standoff shift, then the approach
        self.assertTrue(self.client.sent[0].provenance.endswith("x4") and self.client.sent[1].provenance.endswith("x4"))  # both at contact speed
        self.assertEqual(reasoner.calls[0][2]["calls_remaining"], 5)
        self.assertEqual(reasoner.calls[1][2]["shift_so_far_m"], [.02, 0., 0.])

    def test_abort_and_exhaustion_end_the_approach_and_shifts_are_bounded(self):
        backend, world = self.backend(ScriptedReasoner(corrections=["ABORT"]))
        with self.assertRaises(BackendFailure) as caught:
            self.grasp_move(backend, world)
        self.assertEqual(caught.exception.code, "target_changed")
        self.client = TrackingClient(knuckle=.01)
        reasoner = ScriptedReasoner(corrections=[((.03, .03, 0.), 20.)]*6)
        backend, world = self.backend(reasoner)
        with self.assertRaises(BackendFailure) as caught:
            self.grasp_move(backend, world)
        self.assertEqual(caught.exception.code, "target_changed")
        self.assertEqual(len(reasoner.calls), 5)
        self.assertLessEqual(np.linalg.norm(reasoner.calls[-1][2]["shift_so_far_m"]), .08+1e-9)
        self.assertLessEqual(abs(reasoner.calls[-1][2]["yaw_so_far_deg"]), 15.+1e-9)

    def test_a_sub_centimetre_nudge_counts_as_centred(self):
        reasoner = ScriptedReasoner(corrections=[((.004, .003, 0.), 1.)])
        backend, world = self.backend(reasoner)
        outcome = self.grasp_move(backend, world)
        alignment = outcome.evidence[0]["data"]["alignment"]
        self.assertTrue(alignment["converged"])
        self.assertEqual(alignment["calls"], 1)
        self.assertEqual(len(self.client.sent), 1)                                          # straight to the approach, no shift
        np.testing.assert_allclose(alignment["shift_m"], [0., 0., 0.])

    def test_the_final_approach_ignores_the_measured_surface_under_the_gripper_and_nothing_proud_of_it(self):
        from rammp_adl.motion.collision_guard import CollisionGuard
        guards = []
        backend, world = self.backend(None)
        factory = backend.guard_factory
        backend.guard_factory = lambda **options: guards.append(options) or factory(**options)
        self.scene.entities = {"handle_1": {"grasp": {"support": {"point_m": [.6, .1, .3], "normal": [-1., 0., 0.]}}}}
        self.grasp_move(backend, world)
        target_ball, surface_ball = guards[-1]["exclusions"]
        self.assertEqual(target_ball, ((.58, .1, .3), .10))
        foot = np.array([.6, .1, .3])
        on_face_far = foot+np.array([0., .105, 0.])
        proud_far = on_face_far+np.array([-.03, 0., 0.])                                      # 3 cm in front of the door face
        self.assertEqual(len(CollisionGuard.excluded(np.array([on_face_far, proud_far]), [surface_ball])), 1)
        centre, radius = surface_ball
        self.assertAlmostEqual(radius-np.linalg.norm(np.asarray(centre)-foot), backend.surface_protrusion_m, places=9)
        # Leaving the handle keeps the same surface out of the guard, placed from where the tool is.
        asyncio.run(backend.look("back", execution_context(world)))
        departing = guards[-1]["exclusions"]
        self.assertEqual(len(departing), 2)
        self.assertAlmostEqual(departing[1][1], radius, places=9)

    def test_the_standoff_view_re_measures_the_door_and_places_the_grasp_from_it(self):
        # Discovery put the door face 1.5 cm deep and 6 degrees off; the close view at the standoff sees it true.
        from synthetic_scene import K, looking_at, render
        from rammp_adl.motion.kinematics import quaternion_matrix, quaternion_xyzw_from_matrix
        from rammp_adl.perception.object_geometry import FINGERTIP_REACH_M, SURFACE_CLEARANCE_M
        door_x, handle = .6, (.565, .6, .08, .12, .25, .35)            # a bar 3.5 cm proud of the face at x = .6
        camera = looking_at([.27, .1, .3], [.6, .1, .3])
        depth = render(camera, [handle], plane=("x", door_x))
        scene = self.scene
        scene.base_from_camera = camera

        async def standoff_view(**_):
            return Keyframe("kf-close", np.zeros(depth.shape+(3,), np.uint8), depth, K, "d405_color_optical_frame", 1, 100., 100., .02,
                            camera, (0.,)*7, "requested", 100.)
        scene.wait_for_keyframe = standoff_view
        wrong = rotation_about([0., 0., 1.], np.radians(6.)) @ np.array([-1., 0., 0.])
        scene.entities = {"handle_1": {"grasp": {"support": {"point_m": [.615, .1, .3], "normal": wrong.tolist()}}}}
        guards = []
        backend, world = self.backend(None)
        factory = backend.guard_factory
        backend.guard_factory = lambda **options: guards.append(options) or factory(**options)
        approach = np.column_stack([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])     # tool z along +x, into the door
        start = np.array([.615, .1, .3])+wrong*(FINGERTIP_REACH_M+SURFACE_CLEARANCE_M)  # where discovery put the grasp

        def place(world, evidence_id):
            identities = world.snapshot().identities()
            authority = world.authorize_source("test_observation", self.catalog.predicates)
            world.register_evidence(evidence_id, source=authority, ttl_s=120., observed_at=world.clock(), predicates=[
                {"predicate": "pose_valid", "validity": "true", "args": {"entity_id": "handle_1", "pose_role": "grasp"}}])
            world.update_metric_pose(MetricPose("handle_1", "grasp", tuple(float(v) for v in start),
                                                tuple(float(v) for v in quaternion_xyzw_from_matrix(approach)), tuple([0.]*36),
                                                world.clock(), "base_link", identities["entity:handle_1"]+1, identities["calibration_id"],
                                                identities["base_epoch"], evidence_id, 120.), source=authority)
        place(world, "pose-close")
        outcome = self.grasp_move(backend, world)
        data = outcome.evidence[0]["data"]
        refined = data["alignment"]["refine"]
        self.assertTrue(refined["refined"])
        self.assertAlmostEqual(refined["tilt_deg"], 6., delta=1.)
        self.assertAlmostEqual(refined["part_height_m"], .035, delta=.006)
        commanded = np.asarray(data["commanded"]["position_m"])
        self.assertAlmostEqual(door_x-commanded[0], FINGERTIP_REACH_M+SURFACE_CLEARANCE_M, delta=.004)   # fingertips 1 cm off the real face
        rotation = quaternion_matrix(tuple(data["commanded"]["orientation_xyzw"]))
        self.assertGreater(float(rotation[:, 2] @ np.array([1., 0., 0.])), np.cos(np.radians(1.5)))         # squared to the real face
        surface = guards[-1]["exclusions"][1]
        self.assertAlmostEqual(surface[0][0]-door_x, backend.surface_disk_m**2/(2*backend.surface_protrusion_m)
                               -backend.surface_protrusion_m/2, delta=.01)                                 # centred behind the real face
        # A close view that disagrees with discovery beyond the bounds is refused, not approached.
        self.client = TrackingClient(knuckle=.01)
        scene.entities = {"handle_1": {"grasp": {"support": {"point_m": [.615, .1, .3],
                                                             "normal": (rotation_about([0., 0., 1.], np.radians(25.)) @ np.array([-1., 0., 0.])).tolist()}}}}
        backend, world = self.backend(None)
        place(world, "pose-far")
        with self.assertRaises(BackendFailure) as caught:
            self.grasp_move(backend, world)
        self.assertEqual(caught.exception.code, "target_changed")

    def test_without_a_reasoner_or_look_act_the_geometric_target_is_used(self):
        backend, world = self.backend(None)
        outcome = self.grasp_move(backend, world)
        self.assertEqual(outcome.evidence[0]["data"]["alignment"]["skipped"], "no scene or reasoner bound")
        self.assertEqual(len(self.client.sent), 1)
        self.client = TrackingClient(knuckle=.01)
        backend, world = self.backend(ScriptedReasoner(corrections=["ABORT"]), look_act=False)
        self.assertEqual(self.grasp_move(backend, world).status, "succeeded")

    def follow(self, backend, world, target=.3):
        backend.holding_id, backend.grasp_knuckle, backend.current_pose = "handle_1", .45, ("handle_1", "grasp")
        self.client.knuckle = .45
        return asyncio.run(backend.follow_constraint({"entity_id": "handle_1", "constraint_id": self.record["constraint_id"],
                                                      "target_value": target, "target_unit": "rad", "profile_id": "bench_contact"},
                                                     execution_context(world)))

    def test_a_door_face_that_turns_with_the_tool_verifies_scores_and_leaves_a_demonstration(self):
        from rammp_adl.constraints import waypoints
        axis = np.asarray(self.record["axis_base"])
        initial = np.array([-1., 0., 0.])
        values = [value for value, _, _ in waypoints(self.record, (.58, .1, .3), (0., 0., 0., 1.), .3)]
        steps = [rotation_about(axis, self.record["direction"]*value) @ initial for value in values]
        self.scene.normals = [initial.tolist()]+[n.tolist() for n in steps]
        reasoner = ScriptedReasoner(score=4)
        backend, world = self.backend(reasoner)
        outcome = self.follow(backend, world, .3)
        data = outcome.evidence[0]["data"]
        self.assertTrue(data["verified_locally"])
        self.assertEqual(data["progress"]["local"], 4)
        self.assertEqual(data["progress"]["model"], 4)
        self.assertIn("turned with it", data["evidence_basis"])
        self.assertAlmostEqual(data["measured"][-1]["turned_rad"], .3, places=6)
        self.assertEqual(reasoner.calls[-1][:2], ("verify", "pull the cabinet door"))
        saved = self.store.load("cabinet door")
        self.assertEqual(saved["attempts"][-1]["progress"]["local"], 4)
        self.assertEqual(saved["demonstration"]["frames"], 3)                                 # start, middle, end
        self.assertTrue((self.store.directory/saved["demonstration"]["folder"]/"manifest.json").is_file())

    def test_a_face_lost_from_view_late_in_the_arc_is_unobserved_not_contradicted(self):
        from rammp_adl.constraints import waypoints
        axis = np.asarray(self.record["axis_base"])
        initial = np.array([-1., 0., 0.])
        values = [value for value, _, _ in waypoints(self.record, (.58, .1, .3), (0., 0., 0., 1.), .3)]
        first = rotation_about(axis, self.record["direction"]*values[0]) @ initial
        self.scene.normals = [initial.tolist(), first.tolist()]+[None]*len(values)
        backend, world = self.backend(ScriptedReasoner(score=3))
        data = self.follow(backend, world, .3).evidence[0]["data"]
        self.assertIsNone(data["verified_locally"])                                          # agreed where seen; the end was not seen
        self.assertIn("not measured", data["evidence_basis"])
        self.assertEqual(data["progress"]["local"], 4)

    def test_a_door_face_that_does_not_turn_is_not_verified(self):
        self.scene.normals = [[-1., 0., 0.]]
        backend, world = self.backend(ScriptedReasoner())
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .3)
        self.assertEqual(caught.exception.code, "goal_unobserved")
        attempt = self.store.load("cabinet door")["attempts"][-1]
        self.assertEqual(attempt["status"], "unverified")
        self.assertFalse(attempt["progress"]["verified_locally"])
        self.assertEqual(attempt["progress"]["local"], 3)
        self.assertIsNone(self.store.load("cabinet door").get("demonstration"))

    def test_turn_between_measures_rotation_about_the_axis_only(self):
        self.assertAlmostEqual(turn_between([-1., 0., 0.], rotation_about([0., 0., 1.], .4) @ np.array([-1., 0., 0.]), [0., 0., 1.]), .4, places=9)
        self.assertAlmostEqual(turn_between([-1., 0., 0.], [-1., 0., .2], [0., 0., 1.]), 0., places=9)
        self.assertIsNone(turn_between([0., 0., 1.], [0., 0., 1.], [0., 0., 1.]))


if __name__ == "__main__":
    unittest.main()
