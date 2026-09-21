"""Constraint records: proposal to metric, waypoints, refinement from outcomes, the store."""
import math
import tempfile
import unittest

import numpy as np

from rammp_adl.constraints import (ConstraintStore, apply_parameters, context_constraint, history_summary,
                                   image_axes, metric_constraint, next_parameters, record_attempt, rotation_about,
                                   waypoints)
from rammp_adl.contracts import ContractError
from rammp_adl.motion.kinematics import quaternion_matrix

from synthetic_scene import looking_at

# A cabinet door in front of the robot: its face normal points back at the camera (-x), the handle at x=0.6.
GEOMETRY = {"centroid_m": [.6, .1, .3], "up": [-1., 0., 0.], "major_axis": [0., 0., 1.], "minor_axis": [0., 1., 0.],
            "extent_major_m": .12, "extent_minor_m": .02, "height_m": .035, "top_m": [.565, .1, .3], "points": 300,
            "support": {"normal": [-1., 0., 0.], "rms_m": .003, "inliers": 900}}
CAMERA = looking_at([.1, .1, .35], [.6, .1, .3])          # looking along +x at the door
PROPOSAL = {"status": "OK", "kind": "revolute", "hinge_side": "left", "opening": "pull", "door_width_m": .45,
            "range": 1.2, "contact_effort_nm": 8., "rationale": "hinge on the left edge"}


def record(**overrides):
    return metric_constraint({**PROPOSAL, **overrides}, GEOMETRY, CAMERA, constraint_id="cabinet_door_constraint",
                             entity_id="handle_1", label="cabinet door", surface_entity_id="cabinet_door_surface")


class MetricConstraintTests(unittest.TestCase):
    def test_image_axes_lie_in_the_door_plane(self):
        axes = image_axes(GEOMETRY, CAMERA)
        for name in ("right", "up"):
            self.assertAlmostEqual(axes[name] @ axes["normal"], 0., places=9)
            self.assertAlmostEqual(np.linalg.norm(axes[name]), 1., places=9)
        self.assertGreater(axes["up"] @ np.array([0., 0., 1.]), .99)             # image up is world up on a vertical door
        self.assertGreater(axes["right"] @ np.array([0., -1., 0.]), .99)         # looking along +x, image right is -y

    def test_hinge_on_the_left_puts_the_pivot_a_width_away_and_a_pull_moves_the_handle_toward_the_camera(self):
        rec = record()
        self.assertEqual(rec["kind"], "revolute")
        np.testing.assert_allclose(rec["axis_base"], [0., 0., 1.], atol=1e-9)
        pivot = np.asarray(rec["pivot_base"])
        self.assertAlmostEqual(np.linalg.norm(pivot-np.array(GEOMETRY["centroid_m"])), .40, places=6)   # width minus the edge offset
        self.assertGreater(pivot[1], .1)                                                                 # left in the image is +y
        lever = np.array(GEOMETRY["centroid_m"])-pivot
        moved = rotation_about(rec["axis_base"], rec["direction"]*.2) @ lever-lever
        self.assertLess(moved[0], 0.)                                                                    # toward the camera at -x
        pushed = record(opening="push")
        self.assertEqual(pushed["direction"], -rec["direction"])
        self.assertEqual(context_constraint(rec)["maximum"], 1.2)
        self.assertEqual(context_constraint(rec)["unit"], "rad")

    def test_prismatic_axes_follow_the_opening_direction(self):
        drawer = record(kind="prismatic", hinge_side="none", opening="pull", range=.2)
        np.testing.assert_allclose(drawer["axis_base"], [-1., 0., 0.], atol=1e-9)
        self.assertEqual(drawer["unit"], "m")
        self.assertIsNone(drawer["pivot_base"])
        with self.assertRaises(ContractError):
            record(kind="revolute", hinge_side="none")

    def test_waypoints_stay_on_the_arc_and_reach_the_target(self):
        rec = record()
        tool = np.array([.58, .1, .3])
        orientation = (0., 0., 0., 1.)
        steps = waypoints(rec, tool, orientation, .5)
        self.assertEqual(len(steps), 6)                                       # 5 degree steps to 0.5 rad
        self.assertAlmostEqual(steps[-1][0], .5)
        pivot = np.asarray(rec["pivot_base"])
        for value, position, quaternion in steps:
            self.assertAlmostEqual(np.linalg.norm(np.array(position[:2])-pivot[:2]), np.linalg.norm(tool[:2]-pivot[:2]), places=9)
            self.assertAlmostEqual(position[2], .3)
            rotation = quaternion_matrix(quaternion)
            self.assertAlmostEqual(abs(np.linalg.det(rotation)), 1., places=9)
        angle = math.acos(np.clip(np.trace(quaternion_matrix(steps[-1][2]))/2.-.5, -1., 1.))
        self.assertAlmostEqual(angle, .5, places=6)                           # the tool turns with the door
        slide = waypoints(record(kind="prismatic", hinge_side="none", opening="pull", range=.2), tool, orientation, .05)
        self.assertEqual(len(slide), 3)
        np.testing.assert_allclose(np.subtract(slide[-1][1], tool), [-.05, 0., 0.], atol=1e-9)

    def test_outcomes_refine_the_width_and_the_store_keeps_them(self):
        rec = record()
        record_attempt(rec, task_id="t1", target=1., achieved=.15, status="tripped", detail="contact", peak_effort_nm=8.2)
        self.assertEqual(history_summary(rec)[0]["parameters"]["door_width_m"], .45)
        first = next_parameters(rec)
        self.assertIsNotNone(first)
        self.assertNotAlmostEqual(first["door_width_m"], .45)
        refined = apply_parameters(rec, GEOMETRY, CAMERA, first)
        self.assertEqual(refined["parameters_version"], 2)
        self.assertEqual(len(refined["attempts"]), 1)
        self.assertNotAlmostEqual(np.linalg.norm(np.subtract(refined["pivot_base"], rec["pivot_base"])), 0.)
        record_attempt(refined, task_id="t2", target=1., achieved=.6, status="tripped", peak_effort_nm=8.)
        second = next_parameters(refined)
        self.assertIsNotNone(second)
        self.assertNotIn(round(second["door_width_m"], 3), {round(.45, 3), round(refined["door_width_m"], 3)})
        record_attempt(refined, task_id="t3", target=1., achieved=1., status="succeeded")
        self.assertIsNone(next_parameters(refined))
        with tempfile.TemporaryDirectory() as folder:
            store = ConstraintStore(folder)
            path = store.save(refined)
            self.assertTrue(path.name.startswith("cabinet_door"))
            loaded = store.load("cabinet door")
            self.assertEqual(loaded["attempts"][-1]["status"], "succeeded")
            self.assertEqual(store.labels(), ["cabinet_door"])
            self.assertIsNone(store.load("nothing"))


if __name__ == "__main__":
    unittest.main()


class DemonstrationTests(unittest.TestCase):
    def crop(self, image_id):
        import io
        from PIL import Image
        from rammp_adl.perception.images import ImageCrop
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), (90, 90, 90)).save(buffer, format="JPEG")
        return ImageCrop(image_id, "cap-"+image_id, "wrist_d405", "d405_color_optical_frame", "cal", 100., (0, 0, 848, 480),
                         64, 48, buffer.getvalue(), False, False)

    def test_successful_frames_are_stored_and_the_first_and_last_come_back(self):
        from rammp_adl.constraints import load_demonstration, save_demonstration
        rec = record()
        record_attempt(rec, task_id="t1", target=1., achieved=1., status="succeeded")
        with tempfile.TemporaryDirectory() as folder:
            store = ConstraintStore(folder)
            self.assertIsNone(save_demonstration(store, rec, []))
            saved = save_demonstration(store, rec, [self.crop("a"), self.crop("b"), self.crop("c")])
            self.assertTrue((saved/"manifest.json").is_file())
            self.assertEqual(rec["demonstration"]["frames"], 3)
            store.save(rec)
            loaded = load_demonstration(store, store.load("cabinet door"))
            self.assertEqual([c.image_id for c in loaded], ["demo-a", "demo-c"])
            loaded[0].validate_for_egress(max_bytes=200000, max_long_edge=640, allow_face=False)
            self.assertEqual(load_demonstration(store, {"label": "x"}), [])
