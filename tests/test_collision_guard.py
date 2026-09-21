"""Live guards checked against the real sphere bundle and synthetic depth."""
from pathlib import Path
import unittest

import numpy as np

from rammp_adl.motion.collision_guard import (
    MEASURED_D405_MOUNT, CollisionGuard, EffortGuard, GuardError, GuardSet, SphereModel,
    depth_to_points, load_spheres, mount_transform, nearest_intrusion, trajectory_positions_at)
from rammp_adl.motion.kinematics import UrdfChain
from rammp_adl.motion.rolling import JointState, JointTrajectory, TrajectoryPoint
from rammp_adl.motion.sheppy_client import JOINTS


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2"
HELD = (-1.3e-05, .262109, -3.141206, -2.269117, -5e-05, .959964, 1.570978)
K = (600., 0., 320., 0., 600., 240., 0., 0., 1.)


def path(start, end, duration=2.):
    return JointTrajectory(JOINTS, (
        TrajectoryPoint(0., JointState(tuple(start), (0.,)*7, (0.,)*7)),
        TrajectoryPoint(duration, JointState(tuple(end), (0.,)*7, (0.,)*7))), "rammp_curobo:test")


class PureHelperTests(unittest.TestCase):
    def test_back_projection_puts_the_centre_pixel_on_the_optical_axis(self):
        depth = np.full((480, 640), np.nan)
        depth[240, 320] = .2
        points = depth_to_points(depth, K, stride=1)
        self.assertEqual(points.shape, (1, 3))
        np.testing.assert_allclose(points[0], [0., 0., .2], atol=1e-12)

    def test_back_projection_drops_invalid_and_out_of_range_depth(self):
        depth = np.full((16, 16), .2)
        depth[0, 0], depth[0, 1], depth[0, 2] = np.nan, .01, 5.
        points = depth_to_points(depth, K, stride=1)
        self.assertEqual(len(points), 16*16-3)
        with self.assertRaises(GuardError):
            depth_to_points(depth, K, stride=0)
        with self.assertRaises(GuardError):
            depth_to_points(np.zeros(4), K)

    def test_path_interpolation_is_clamped_and_wrap_aware(self):
        route = path((0.,)*7, (1.,)*7, duration=2.)
        self.assertEqual(trajectory_positions_at(route, -1.), (0.,)*7)
        self.assertEqual(trajectory_positions_at(route, 9.), (1.,)*7)
        self.assertAlmostEqual(trajectory_positions_at(route, 1.)[0], .5, places=12)
        wrapping = path((np.pi-.1,)*7, (-np.pi+.1,)*7, duration=1.)
        self.assertAlmostEqual(abs(trajectory_positions_at(wrapping, .5)[0]), np.pi, places=9)

    def test_nearest_intrusion_reports_only_inside_the_margin(self):
        centres, radii, labels = np.array([[0., 0., 0.]]), np.array([.05]), ["link"]
        self.assertIsNone(nearest_intrusion(np.array([[0., 0., .2]]), centres, radii, labels, margin_m=.03))
        hit = nearest_intrusion(np.array([[0., 0., .07]]), centres, radii, labels, margin_m=.03)
        self.assertAlmostEqual(hit["distance_m"], .02, places=12)
        self.assertEqual(hit["link"], "link")
        self.assertIsNone(nearest_intrusion(np.zeros((0, 3)), centres, radii, labels, margin_m=.03))

    def test_mount_transform_is_the_measured_composition(self):
        transform = mount_transform()
        np.testing.assert_allclose(transform[:3, 3], MEASURED_D405_MOUNT["xyz"], atol=1e-12)
        # 180 degrees about the lens axis flips x and y, keeps z.
        np.testing.assert_allclose(transform[:3, :3] @ [0., 0., 1.], [0., 0., 1.], atol=1e-12)
        np.testing.assert_allclose(transform[:3, :3] @ [1., 0., 0.], [-1., 0., 0.], atol=1e-12)


class EffortGuardTests(unittest.TestCase):
    def test_baseline_is_taken_after_arming_and_deviation_trips(self):
        guard = EffortGuard(3.)
        self.assertIsNone(guard.on_efforts([0.]*7))     # not armed yet
        guard.on_progress(.1)
        self.assertIsNone(guard.on_efforts([1., 1., 1., 1., 2., 2., 2.]))  # baseline
        self.assertIsNone(guard.on_efforts([1., 1., 1., 1., 4., 2., 2.]))
        trip = guard.on_efforts([1., 1., 1., 1., 6., 2., 2.])
        self.assertEqual(trip["kind"], "contact")
        self.assertAlmostEqual(guard.peak_nm, 4., places=12)

    def test_arm_after_progress_holds_the_trip_until_the_fraction(self):
        guard = EffortGuard(1., arm_after_progress=.5)
        guard.on_progress(.1)
        guard.on_efforts([0.]*7)
        self.assertIsNone(guard.on_efforts([0., 0., 0., 0., 5., 0., 0.]))
        guard.on_progress(.6)
        self.assertIsNotNone(guard.on_efforts([0., 0., 0., 0., 5., 0., 0.]))

    def test_invalid_configuration_is_refused(self):
        for kwargs in ({"touch_nm": 0.}, {"touch_nm": True}, {"touch_nm": 1., "arm_after_progress": 1.}):
            with self.assertRaises(GuardError):
                EffortGuard(**kwargs)


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class SphereModelTests(unittest.TestCase):
    def setUp(self):
        self.chain = UrdfChain.from_path(BUNDLE/"arm-gripper-locked.urdf")
        self.spheres = load_spheres(BUNDLE/"collision-spheres.json")
        d405 = load_spheres(BUNDLE/"d405-collision-spheres.json")["wrist_d405_link"]
        self.model = SphereModel(self.chain, self.spheres,
                                 extra=[("end_effector_link", mount_transform(), d405[0], d405[1])])

    def test_the_bundle_places_all_spheres_plus_the_camera(self):
        centres, radii, labels = self.model.placed(HELD)
        self.assertEqual(len(centres), 132+1)
        self.assertEqual(len(radii), len(labels))
        self.assertTrue((radii > 0).all())
        self.assertIn("end_effector_link+extra", labels)
        with self.assertRaises(GuardError):
            self.model.placed(HELD[:6])
        with self.assertRaises(GuardError):
            SphereModel(self.chain, {"absent_link": self.spheres["bracelet_link"]})

    def test_camera_transform_composes_fk_with_the_mount(self):
        transform = self.model.base_from_camera(HELD, camera_parent="end_effector_link",
                                                ee_from_camera=mount_transform())
        ee = self.chain.base_from_link(self.model.configuration(HELD), "end_effector_link")
        np.testing.assert_allclose(transform, ee @ mount_transform(), atol=1e-12)


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class CollisionGuardTests(unittest.TestCase):
    def setUp(self):
        self.chain = UrdfChain.from_path(BUNDLE/"arm-gripper-locked.urdf")
        self.model = SphereModel(self.chain, load_spheres(BUNDLE/"collision-spheres.json"))
        self.guard = CollisionGuard(self.model, margin_m=.03, lookahead_s=2., sample_dt_s=.25)
        # Drive the elbow: the tool translates by decimetres, so a point at its
        # future position is genuinely clear of the arm at the start.
        self.route = path(HELD, tuple(q+d for q, d in zip(HELD, (0., 0., 0., .3, 0., 0., 0.))))

    def future_tool_point(self, time_s, offset):
        positions = trajectory_positions_at(self.route, time_s)
        pose = self.chain.base_from_link(self.model.configuration(positions), "end_effector_link")
        return pose[:3, 3]+np.asarray(offset)

    def test_a_point_on_the_future_path_is_an_intrusion_and_a_distant_one_is_not(self):
        obstacle = self.future_tool_point(2., (0., 0., 0.))
        centres, radii, labels = self.model.placed(HELD)
        self.assertIsNone(nearest_intrusion(np.array([obstacle]), centres, radii, labels, margin_m=.03),
                          "fixture: the future tool position must be clear of the arm now")
        hit = self.guard.sweep(np.array([obstacle]), self.route, elapsed_s=0.)
        self.assertIsNotNone(hit)
        self.assertLess(hit["distance_m"], .03)
        self.assertGreater(hit["time_s"], 0.)
        far = self.future_tool_point(2., (0., 0., 1.5))
        self.assertIsNone(self.guard.sweep(np.array([far]), self.route, elapsed_s=0.))

    def test_the_sweep_only_looks_ahead_of_the_elapsed_time(self):
        early = self.future_tool_point(0., (0., 0., 0.))
        self.assertIsNotNone(self.guard.sweep(np.array([early]), self.route, elapsed_s=0.))
        # Once the arm is past that point, a point at its old position is not ahead.
        self.assertIsNone(CollisionGuard(self.model, margin_m=.001, lookahead_s=.2, sample_dt_s=.1)
                          .sweep(np.array([early+np.array([0., 0., .2])]), self.route, elapsed_s=1.9))

    def test_points_on_the_robot_itself_are_filtered(self):
        centres, radii, _ = self.model.placed(HELD)
        on_finger = centres[-1]+np.array([radii[-1], 0., 0.])*.5
        self.assertEqual(len(self.guard.self_filtered(np.array([on_finger]), HELD)), 0)
        far = centres[-1]+np.array([1., 0., 0.])
        self.assertEqual(len(self.guard.self_filtered(np.array([far]), HELD)), 1)

    def test_check_lifts_depth_through_the_mount_and_refuses_stale_frames(self):
        depth = np.full((480, 640), np.nan)
        depth[240, 320] = .2  # a point 20 cm along the camera axis
        trip = self.guard.check(depth_m=depth, intrinsics_k=K, frame_age_s=.05, joints_now=HELD,
                                trajectory=self.route, elapsed_s=0.)
        transform = self.model.base_from_camera(HELD, camera_parent="end_effector_link",
                                                ee_from_camera=mount_transform())
        expected = transform[:3, :3] @ [0., 0., .2]+transform[:3, 3]
        if trip is not None:
            np.testing.assert_allclose(trip["point_base_m"], expected, atol=1e-9)
        stale = self.guard.check(depth_m=depth, intrinsics_k=K, frame_age_s=1., joints_now=HELD,
                                 trajectory=self.route, elapsed_s=0.)
        self.assertEqual(stale["kind"], "depth_stale")

    def test_guard_configuration_is_validated(self):
        for kwargs in ({"margin_m": 0.}, {"lookahead_s": -1.}, {"camera_parent": "absent"},
                       {"sample_dt_s": True}):
            with self.assertRaises(GuardError):
                CollisionGuard(self.model, **kwargs)
        with self.assertRaises(GuardError):
            CollisionGuard("not a model")


class GuardSetTests(unittest.TestCase):
    def test_effort_trip_is_reported_first_and_recorded(self):
        guard = GuardSet(effort=EffortGuard(1.))
        route = path((0.,)*7, (.1,)*7)
        live = {"position_rad": (0.,)*7, "effort_nm": [0.]*7}
        guard.on_progress(.2)
        self.assertIsNone(guard.check(live=live, trajectory=route, elapsed_s=0., now=0.))
        trip = guard.check(live={**live, "effort_nm": [0., 0., 0., 0., 5., 0., 0.]},
                           trajectory=route, elapsed_s=.1, now=.1)
        self.assertEqual(trip["kind"], "contact")
        self.assertEqual(guard.trips, [trip])

    def test_missing_joint_state_and_a_blind_camera_are_trips_not_passes(self):
        route = path((0.,)*7, (.1,)*7)
        stale = GuardSet(effort=EffortGuard(1.)).check(live=None, trajectory=route, elapsed_s=0., now=0.)
        self.assertEqual(stale["kind"], "state_stale")
        with self.assertRaises(GuardError):
            GuardSet(collision=object())

    def test_a_collision_guard_needs_frames_within_the_blind_limit(self):
        class Never:
            checks = 0

            def check(self, **kwargs):
                self.checks += 1
                return None

        frames = []
        guard = GuardSet(collision=Never(), depth_reader=lambda: frames.pop() if frames else None, blind_after_s=.5)
        route = path((0.,)*7, (.1,)*7)
        live = {"position_rad": (0.,)*7, "effort_nm": [0.]*7}
        self.assertIsNone(guard.check(live=live, trajectory=route, elapsed_s=0., now=0.))
        self.assertIsNone(guard.check(live=live, trajectory=route, elapsed_s=.1, now=.4))
        self.assertEqual(guard.check(live=live, trajectory=route, elapsed_s=.2, now=.9)["kind"], "depth_blind")
        frames.append((np.zeros((2, 2)), K, .95))
        self.assertIsNone(guard.check(live=live, trajectory=route, elapsed_s=.3, now=1.))


if __name__ == "__main__":
    unittest.main()
