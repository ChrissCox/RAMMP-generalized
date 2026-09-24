"""Object geometry and tool_frame pose roles from a grounded box over synthetic depth."""
import unittest

import numpy as np

from rammp_adl.motion.kinematics import quaternion_matrix
from rammp_adl.perception.geometry import PerceptionError
from rammp_adl.perception.keyframes import Keyframe
from rammp_adl.perception.object_geometry import (FINGERTIP_REACH_M, SURFACE_CLEARANCE_M, box_to_pixels, object_geometry,
                                                   points_in_region, pose_roles, support_plane, to_base)

from synthetic_scene import K, looking_at, looking_down, normalized_box, render

MONO = 100.


def keyframe(pose, boxes):
    depth = render(pose, boxes)
    rgb = np.full(depth.shape+(3,), 200, np.uint8)
    return Keyframe("cap-1", rgb, depth, K, "d405_color_optical_frame", 1, MONO, MONO, .02, pose, (0.,)*7, "initial", MONO)


def measure(frame, box):
    whole = to_base(points_in_region(frame, (0, 0, frame.width, frame.height), stride=8), frame.base_from_camera)
    plane = support_plane(whole)
    region = box_to_pixels(normalized_box(frame.base_from_camera, box), frame.width, frame.height)
    inside = to_base(points_in_region(frame, region), frame.base_from_camera)
    return plane, object_geometry(inside, plane)


class GeometryTests(unittest.TestCase):
    def test_short_box_on_the_table_gets_a_top_down_grasp(self):
        pose = looking_down([.3, 0., .6])
        box = (.27, .33, .10, .14, .08)
        frame = keyframe(pose, [box])
        plane, geometry = measure(frame, box)
        np.testing.assert_allclose(plane["normal"], [0., 0., 1.], atol=1e-6)
        self.assertLess(abs(plane["origin_m"][2]), 1e-6)
        self.assertAlmostEqual(geometry["height_m"], .08, delta=.005)
        self.assertAlmostEqual(geometry["extent_minor_m"], .04, delta=.01)
        self.assertAlmostEqual(geometry["extent_major_m"], .06, delta=.01)
        np.testing.assert_allclose(geometry["centroid_m"][:2], [.30, .12], atol=.01)
        roles = pose_roles(geometry, camera_position_base=frame.camera_position_base)
        self.assertEqual(roles["strategy"], "top_down")
        grasp = roles["roles"]["grasp"]
        np.testing.assert_allclose(grasp["position_m"], [.30, .12, max(.05, FINGERTIP_REACH_M+SURFACE_CLEARANCE_M)], atol=.012)
        rotation = quaternion_matrix(tuple(grasp["orientation_xyzw"]))
        np.testing.assert_allclose(rotation[:, 2], [0., 0., -1.], atol=1e-6)         # tool points down
        self.assertAlmostEqual(abs(rotation[:, 0] @ np.array([0., 1., 0.])), 1., delta=.05)  # fingers close across the minor axis
        np.testing.assert_allclose(np.subtract(roles["roles"]["pregrasp"]["position_m"], grasp["position_m"]), [0., 0., .10], atol=1e-6)
        np.testing.assert_allclose(np.subtract(roles["roles"]["staging"]["position_m"], grasp["position_m"]), [0., 0., .15], atol=1e-6)
        covariance = np.asarray(roles["covariance"]).reshape(6, 6)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))

    def test_tall_narrow_object_gets_a_side_grasp_from_the_camera_side(self):
        pose = looking_at([.0, -.4, .5], [.4, .1, .1])
        box = (.37, .43, .07, .13, .25)
        frame = keyframe(pose, [box])
        plane, geometry = measure(frame, box)
        self.assertIsNotNone(plane)
        self.assertAlmostEqual(geometry["height_m"], .25, delta=.02)
        roles = pose_roles(geometry, camera_position_base=frame.camera_position_base)
        self.assertEqual(roles["strategy"], "side")
        rotation = quaternion_matrix(tuple(roles["roles"]["grasp"]["orientation_xyzw"]))
        approach = rotation[:, 2]
        self.assertLess(abs(approach[2]), 1e-3)                                       # horizontal to the fitted support
        self.assertGreater(approach @ np.array([.4, .5, 0.]), 0.)                         # away from the camera, toward the object
        self.assertAlmostEqual(abs(approach[0]) if abs(approach[0]) > .5 else abs(approach[1]), 1., delta=.05)  # along a footprint axis
        self.assertLess(abs(rotation[:, 0] @ approach), 1e-6)                        # closing axis across the approach
        np.testing.assert_allclose(np.subtract(roles["roles"]["grasp"]["position_m"], roles["roles"]["pregrasp"]["position_m"]), approach*.10, atol=1e-6)

    def test_too_wide_objects_get_no_grasp_roles(self):
        pose = looking_down([.3, 0., .6])
        box = (.24, .36, .05, .17, .06)
        frame = keyframe(pose, [box])
        plane, geometry = measure(frame, box)
        roles = pose_roles(geometry, camera_position_base=frame.camera_position_base)
        self.assertEqual(roles["strategy"], "none")
        self.assertEqual(roles["roles"], {})

    def test_without_a_support_plane_the_nearest_band_is_the_object(self):
        pose = looking_down([.3, 0., .6])
        box = (.27, .33, .10, .14, .08)
        frame = keyframe(pose, [box])
        region = box_to_pixels(normalized_box(pose, box), frame.width, frame.height)
        inside = to_base(points_in_region(frame, region), pose)
        geometry = object_geometry(inside, None)
        self.assertIsNone(geometry["support"])
        self.assertGreater(geometry["points"], 30)
        with self.assertRaises(PerceptionError):
            object_geometry(inside[:5], None)
        with self.assertRaises(PerceptionError):
            box_to_pixels([.5, .5, .2, .2], 320, 240)


if __name__ == "__main__":
    unittest.main()


class FingertipClearanceTests(unittest.TestCase):
    """The 2F-85's fingertips reach 4.6 cm past the planner's tool frame: a grasp keeps them off the surface the object stands on."""

    def handle(self, height):
        # A vertical bar on a door face at x = .6 whose normal faces the robot (-x).
        return {"up": [-1., 0., 0.], "top_m": [.6-height, .1, .3], "centroid_m": [.6-height/2, .1, .3], "height_m": height,
                "minor_axis": [0., 1., 0.], "major_axis": [0., 0., 1.], "extent_minor_m": .015, "extent_major_m": .11}

    def test_a_shallow_handle_is_pinched_by_the_fingertips_without_them_touching_the_door(self):
        roles = pose_roles(self.handle(.032), camera_position_base=[.2, .1, .3])
        self.assertEqual(roles["strategy"], "top_down")
        tool = np.asarray(roles["roles"]["grasp"]["position_m"])
        elevation = .6-tool[0]
        self.assertGreaterEqual(elevation-FINGERTIP_REACH_M, SURFACE_CLEARANCE_M-1e-9)   # fingertips a centimetre off the door
        self.assertLess(elevation-FINGERTIP_REACH_M, .032)                                 # and still beside the bar, not in front of it
        self.assertAlmostEqual(roles["fingertip_clearance_m"], elevation-FINGERTIP_REACH_M, places=6)
        np.testing.assert_allclose(roles["support"]["point_m"], [.6, .1, .3], atol=1e-9)    # the door face under the grasp
        np.testing.assert_allclose(roles["support"]["normal"], [-1., 0., 0.], atol=1e-9)

    def test_a_tall_object_keeps_its_grasp_depth(self):
        tall = dict(self.handle(.12), minor_axis=[0., 1., 0.])
        roles = pose_roles(tall, camera_position_base=[.2, .1, .3])
        np.testing.assert_allclose(roles["roles"]["grasp"]["position_m"], [.6-.12+.03, .1, .3], atol=1e-9)
        self.assertGreater(roles["fingertip_clearance_m"], SURFACE_CLEARANCE_M)


class WristChoiceTests(unittest.TestCase):
    """A measured axis has no sign; the jaw closes the same either way round. The choice must not flip between looks."""

    def geometry(self, minor):
        return {"up": [0., 0., 1.], "centroid_m": [.4, 0., .02], "top_m": [.4, 0., .04], "minor_axis": list(minor),
                "major_axis": [minor[1], -minor[0], 0.], "extent_minor_m": .03, "extent_major_m": .10, "height_m": .04}

    def closing_axis(self, roles):
        return quaternion_matrix(tuple(roles["roles"]["grasp"]["orientation_xyzw"]))[:3, 0]

    def test_either_sign_of_the_measured_axis_gives_the_same_wrist(self):
        one, other = (pose_roles(self.geometry(minor), camera_position_base=[.4, 0., .5]) for minor in ([0., 1., 0.], [0., -1., 0.]))
        np.testing.assert_allclose(self.closing_axis(one), self.closing_axis(other), atol=1e-9)

    def test_the_wrist_nearest_the_tool_is_taken(self):
        for tool_x in ([0., 1., 0.], [0., -1., 0.]):
            roles = pose_roles(self.geometry([0., 1., 0.]), camera_position_base=[.4, 0., .5], tool_x_axis=tool_x)
            self.assertGreater(float(self.closing_axis(roles) @ np.asarray(tool_x)), .99)


class TwoDepthTests(unittest.TestCase):
    """A near door beside a far wall, as the wrist camera first saw the cabinet: the plane is the door, not a fit through both."""

    def points(self):
        rng = np.random.default_rng(1)
        door = np.column_stack([rng.uniform(-.20, .06, 4000), rng.uniform(-.15, .15, 4000), .29+rng.normal(0., .002, 4000)])
        wall = np.column_stack([rng.uniform(.10, .90, 3000), rng.uniform(-.6, .6, 3000), 1.5+rng.normal(0., .01, 3000)])
        return door, wall

    def test_the_largest_plane_wins_over_a_compromise_between_two_depths(self):
        door, wall = self.points()
        plane = support_plane(np.vstack([door, wall]), camera_position=np.zeros(3))
        self.assertIsNotNone(plane)
        self.assertGreater(abs(plane["normal"][2]), .99)
        self.assertAlmostEqual(plane["origin_m"][2], .29, delta=.01)
        on = door[np.abs((door-np.asarray(plane["origin_m"])) @ np.asarray(plane["normal"])) <= .015]
        self.assertGreater(float(on[:, 0].max()-on[:, 0].min()), .25)                         # the whole door, not a 7 cm strip

    def test_the_fit_is_repeatable(self):
        door, wall = self.points()
        first, second = (support_plane(np.vstack([door, wall]), camera_position=np.zeros(3)) for _ in range(2))
        self.assertEqual(first, second)


class GraspPointTests(unittest.TestCase):
    def test_a_marked_point_moves_the_closing_point_across_the_object_only(self):
        from rammp_adl.perception.object_geometry import lift_point
        pose = looking_down([.3, 0., .6])
        box = (.24, .36, .10, .14, .08)                      # a 12 cm bar: fingers should close where the model points
        frame = keyframe(pose, [box])
        plane, geometry = measure(frame, box)
        plain = pose_roles(geometry, camera_position_base=frame.camera_position_base)
        # Project the bar's left end onto the image and lift it back.
        end = np.array([.26, .12, .08])
        camera = (end-pose[:3, 3]) @ pose[:3, :3]
        xy = ((300.*camera[0]/camera[2]+160.)/320., (300.*camera[1]/camera[2]+120.)/240.)
        lifted = lift_point(frame, xy)
        np.testing.assert_allclose(lifted, end, atol=.01)
        marked = pose_roles(geometry, camera_position_base=frame.camera_position_base, grasp_point=lifted)
        self.assertTrue(marked["grasp_point_used"])
        self.assertAlmostEqual(marked["roles"]["grasp"]["position_m"][0], .26, delta=.012)          # moved along the bar
        self.assertAlmostEqual(marked["roles"]["grasp"]["position_m"][2], plain["roles"]["grasp"]["position_m"][2], places=9)  # same depth
        far = pose_roles(geometry, camera_position_base=frame.camera_position_base, grasp_point=[.9, .9, .3])
        self.assertFalse(far["grasp_point_used"])
        self.assertIsNone(lift_point(frame, (.5, .5)) if np.isnan(frame.depth_m[120, 160]) else None)
