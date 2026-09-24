"""The planner's waypoint solutions flown as one trajectory: through every waypoint, at rest at both ends, under every limit."""
import unittest

import numpy as np

from rammp_adl.motion.constraint_path import PathError, constraint_trajectory, deviation, knot_spline
from rammp_adl.motion.sheppy_client import DIFFERENCED, JOINT_VMAX, START_PREPENDED, executor_problems

LIMITS = [v/4. for v in JOINT_VMAX]


def door_knots(count=18, step=np.radians(5.)):
    """Joint solutions that bend like the planner's along a door arc: smooth, some joints speeding up near reach."""
    values = np.arange(count+1)*step
    start = np.array([-.53, .66, -2.57, -1.61, -.91, .62, 1.97])
    shape = np.stack([-.1*values, .3*values**2, -.2*values, .8*values**3, -1.5*values+.4*values**2, .9*values, 1.2*values], axis=1)
    return values, start+shape


class KnotSplineTests(unittest.TestCase):
    def test_a_branch_change_between_waypoints_is_refused(self):
        values, knots = door_knots(4)
        knots[3, 6] += 2.*np.pi-.3
        with self.assertRaises(PathError) as caught:
            knot_spline(values, knots)
        self.assertIn("joint 7", str(caught.exception))

    def test_values_must_start_at_zero_and_increase_and_shapes_agree(self):
        values, knots = door_knots(3)
        for bad_values in (values+.1, values[::-1], np.array([0., .1, .1, .2])):
            with self.assertRaises(PathError):
                knot_spline(bad_values, knots)
        with self.assertRaises(PathError):
            knot_spline(values, knots[:, :6])
        with self.assertRaises(PathError):
            knot_spline(values[:1], knots[:1])


class ConstraintTrajectoryTests(unittest.TestCase):
    def build(self, **options):
        values, knots = door_knots()
        settings = {"rate_limits": LIMITS, "value_rate": .3, "value_accel": .6, "provenance": "rammp_curobo:test", **options}
        return (values, knots)+constraint_trajectory(values, knots, **settings)

    def test_one_pass_through_every_waypoint_from_rest_to_rest_within_the_gates(self):
        values, knots, trajectory, timing, spline = self.build()
        points = trajectory.points
        np.testing.assert_allclose(points[0].state.position, knots[0])
        np.testing.assert_allclose(points[-1].state.position, knots[-1])
        self.assertEqual(points[0].state.position, points[1].state.position)       # the start held one sample, as the planner's
        self.assertTrue(all(v == 0. for v in points[0].state.velocity+points[1].state.velocity+points[-1].state.velocity))
        for value, knot in zip(values, knots):
            np.testing.assert_allclose(spline(value), knot, atol=1e-12)
        velocities = np.array([p.state.velocity for p in points])
        self.assertTrue((np.abs(velocities) <= np.asarray(LIMITS)*(1+1e-9)).all())
        self.assertGreater(np.max(np.abs(velocities)/LIMITS), .9)                    # and it uses them
        self.assertEqual(executor_problems(trajectory), [])
        self.assertIn(START_PREPENDED, trajectory.provenance)                       # the wire has the planner's shape
        self.assertIn(DIFFERENCED, trajectory.provenance)
        times = np.array([p.time_s for p in points])
        self.assertTrue(np.all(np.diff(times) > 0.))
        self.assertLess(trajectory.duration_s, 20.)

    def test_the_value_keeps_its_pace_and_the_timing_maps_both_ways(self):
        values, knots, trajectory, timing, spline = self.build(rate_limits=[10.]*7)   # joints unbound: the door's pace binds
        times, travelled = np.asarray(timing.times_s), np.asarray(timing.values)
        rates = np.diff(travelled)/np.diff(times)
        self.assertLessEqual(rates.max(), .3*(1+1e-4))
        self.assertTrue(np.all(np.diff(travelled) >= 0.))
        self.assertAlmostEqual(travelled[-1], values[-1])
        accelerations = np.diff(rates)/np.diff(times)[1:]
        self.assertLess(np.abs(accelerations).max(), .6*1.1)
        middle = values[9]
        self.assertAlmostEqual(timing.value_at(timing.time_at(middle)), middle, delta=.3*.04)
        self.assertEqual(timing.value_at(0.), 0.)
        self.assertAlmostEqual(timing.value_at(trajectory.duration_s), values[-1])
        # 90 degrees at 0.3 rad/s with gentle ends: a few seconds, not a minute.
        self.assertLess(trajectory.duration_s, values[-1]/.3+2.)

    def test_a_hold_keeps_the_start_before_moving_and_delays_everything_after(self):
        values, knots, plain, plain_timing, _ = self.build()
        _, _, held, timing, _ = self.build(hold_s=.5)
        self.assertAlmostEqual(held.duration_s, plain.duration_s+.5, places=9)
        still = [p for p in held.points if p.time_s <= .5+held.points[1].time_s+1e-9]
        self.assertTrue(all(p.state.position == held.points[0].state.position and not any(p.state.velocity) for p in still))
        self.assertEqual(executor_problems(held), [])
        self.assertEqual(timing.value_at(.5), 0.)
        self.assertAlmostEqual(timing.time_at(values[5]), plain_timing.time_at(values[5])+.5, places=9)

    def test_tighter_joint_limits_slow_the_whole_pass(self):
        fast = self.build()[2]
        slow = self.build(rate_limits=[v/2. for v in LIMITS])[2]
        self.assertGreater(slow.duration_s, fast.duration_s*1.4)
        velocities = np.array([p.state.velocity for p in slow.points])
        self.assertTrue((np.abs(velocities) <= np.asarray(LIMITS)/2.*(1+1e-9)).all())


class DeviationTests(unittest.TestCase):
    def test_the_worst_distance_and_turn_along_the_path_are_found(self):
        values, knots = door_knots(4)
        spline = knot_spline(values, knots)
        identity = np.eye(3)

        def tool_pose(joints):
            return np.asarray(joints[:3]), identity

        def exact(value):
            return spline(value)[:3], identity

        worst = deviation(spline, values[-1], tool_pose, exact)
        self.assertAlmostEqual(worst["distance_m"], 0.)

        def bent(value):
            angle = .1 if abs(value-values[2]) < 1e-9 else 0.
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
            return spline(value)[:3]+np.array([0., 0., .01*np.sin(np.pi*value/values[-1])]), rotation

        worst = deviation(spline, values[-1], tool_pose, bent, spacing=values[-1]/40.)
        self.assertAlmostEqual(worst["distance_m"], .01, places=4)
        self.assertAlmostEqual(worst["turn_rad"], .1, places=6)
        self.assertAlmostEqual(worst["turn_at"], values[2])


if __name__ == "__main__":
    unittest.main()
