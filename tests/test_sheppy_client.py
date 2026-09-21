"""Client-side gates in front of a driver that applies none of them."""
import math
import unittest

from rammp_adl.motion.rolling import JointState, JointTrajectory, TrajectoryPoint
from rammp_adl.motion.sheppy_client import (
    JOINTS, JOINT_POSITION_LIMITS, JOINT_VMAX, KNUCKLE_CLOSED_RAD, REQUIRED_RMW, RESULT_NAMES, START_GATE_RAD,
    SheppyClientError, describe_surface, executor_problems, knuckle_from_setpoint,
    refusal, result_message, reversed_trajectory, rmw_refusal, scale_trajectory_time, setpoint_from_knuckle,
    start_gap_rad, trajectory_from_planner, wrap_diff, DIFFERENCED, START_PREPENDED)


PROVENANCE = "unit-test fixture; not a planner output"


def trajectory(points, *, names=JOINTS):
    return JointTrajectory(tuple(names), tuple(
        TrajectoryPoint(time_s, JointState(tuple(position), None if velocity is None else tuple(velocity),
                                           tuple(acceleration)))
        for time_s, position, velocity, acceleration in points), PROVENANCE)


def simple(*, start=0., end=.1, duration=2., velocity=.05):
    """A zero-based path from `start` to `end`, as the canonical type requires."""
    return trajectory([(0., (start,)*7, (0.,)*7, (0.,)*7),
                       (duration/2., (start+(end-start)/2.,)*7, (velocity,)*7, (0.,)*7),
                       (duration, (end,)*7, (0.,)*7, (0.,)*7)])


class WrapTests(unittest.TestCase):
    def test_a_wrap_is_a_small_difference(self):
        self.assertAlmostEqual(wrap_diff(math.pi-.01, -math.pi+.01), -.02, places=9)
        self.assertAlmostEqual(wrap_diff(.2, .1), .1, places=12)

    def test_non_finite_angles_are_refused(self):
        for bad in (float("nan"), float("inf"), True, "x"):
            with self.assertRaises(SheppyClientError):
                wrap_diff(bad, 0.)

    def test_numpy_floats_from_joint_states_are_accepted(self):
        import numpy as np
        self.assertAlmostEqual(wrap_diff(np.float64(.2), np.float32(.1)), .1, places=6)


class MiddlewareTests(unittest.TestCase):
    def test_only_cyclone_may_talk_to_the_containers(self):
        self.assertIsNone(rmw_refusal({"RMW_IMPLEMENTATION": REQUIRED_RMW}))
        for environ in ({}, {"RMW_IMPLEMENTATION": ""}, {"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"}):
            reason = rmw_refusal(environ)
            self.assertIsNotNone(reason)
            self.assertIn(REQUIRED_RMW, reason)


class StartGateTests(unittest.TestCase):
    def test_a_trajectory_starting_where_the_arm_is_passes(self):
        path = simple(start=0., end=.1)
        live = path.points[0].state.position
        self.assertIsNone(refusal(path, live))
        self.assertAlmostEqual(start_gap_rad(live, path), 0., places=12)

    def test_a_distant_start_is_refused_because_the_driver_would_jump(self):
        path = simple(start=0., end=.1)
        live = tuple(q+START_GATE_RAD*2 for q in path.points[0].state.position)
        reason = refusal(path, live)
        self.assertIn("would jump to its first waypoint", reason)

    def test_the_gap_is_wrap_aware(self):
        # Only the continuous joints may sit near pi; the bounded ones stay at zero.
        near = tuple(math.pi-.01 if bounds is None else 0. for bounds in JOINT_POSITION_LIMITS)
        end = tuple(math.pi-.005 if bounds is None else 0. for bounds in JOINT_POSITION_LIMITS)
        path = trajectory([(0., near, (0.,)*7, (0.,)*7), (2., end, (0.,)*7, (0.,)*7)])
        live = tuple(-math.pi+.01 if bounds is None else 0. for bounds in JOINT_POSITION_LIMITS)
        self.assertLess(start_gap_rad(live, path), START_GATE_RAD)
        self.assertIsNone(refusal(path, live))

    def test_mismatched_live_joint_count_is_refused(self):
        with self.assertRaises(SheppyClientError):
            start_gap_rad((0., 0.), simple())


class ExecutorGateTests(unittest.TestCase):
    def live(self, path):
        return path.points[0].state.position

    def test_dimension_mismatch_against_the_limits_is_refused(self):
        path = simple()
        self.assertIn("dimensions disagree", "; ".join(executor_problems(path, vmax=(1.,)*6,
                                                                          position_limits=(None,)*6)))
        with self.assertRaises(SheppyClientError):
            executor_problems(path, vmax=(1.,)*6)

    def test_velocity_beyond_the_urdf_limit_is_refused(self):
        fast = tuple(v*1.5 for v in JOINT_VMAX)
        path = trajectory([(0., (0.,)*7, (0.,)*7, (0.,)*7), (1., (.05,)*7, fast, (0.,)*7),
                           (2., (.1,)*7, (0.,)*7, (0.,)*7)])
        self.assertIn("velocity exceeds limit", refusal(path, (0.,)*7))

    def test_a_small_slack_above_the_limit_is_tolerated(self):
        edge = tuple(v*1.005 for v in JOINT_VMAX)
        path = trajectory([(0., (0.,)*7, (0.,)*7, (0.,)*7), (1., (.05,)*7, edge, (0.,)*7),
                           (2., (.1,)*7, (0.,)*7, (0.,)*7)])
        self.assertNotIn("velocity exceeds limit", executor_problems(path))

    def test_monotonic_zero_based_timing_is_a_type_invariant_not_a_gate(self):
        from rammp_adl.motion.rolling import MotionError
        with self.assertRaises(MotionError):
            trajectory([(1., (0.,)*7, (0.,)*7, (0.,)*7), (2., (.05,)*7, (0.,)*7, (0.,)*7)])
        with self.assertRaises(MotionError):
            trajectory([(0., (0.,)*7, (0.,)*7, (0.,)*7), (0., (.05,)*7, (0.,)*7, (0.,)*7)])
        # The open wire question is recorded where a reader will meet it.
        self.assertIn("TODO: confirm against driver", executor_problems.__doc__)

    def test_a_waypoint_outside_a_declared_range_is_refused(self):
        beyond = [0.]*7
        beyond[1] = 2.5  # joint_2 declares +/-2.41
        path = trajectory([(0., (0.,)*7, (0.,)*7, (0.,)*7), (4., tuple(beyond), (0.,)*7, (0.,)*7)])
        self.assertIn("declared joint range", refusal(path, (0.,)*7))
        # Continuous joints carry no declared bound and are not invented one.
        spun = [0.]*7
        spun[0] = 4.
        path = trajectory([(0., (0.,)*7, (0.,)*7, (0.,)*7), (4., tuple(spun), (0.,)*7, (0.,)*7)])
        self.assertNotIn("declared joint range", refusal(path, (0.,)*7) or "")
        self.assertEqual(sum(bounds is None for bounds in JOINT_POSITION_LIMITS), 4)

    def test_a_position_jump_between_waypoints_is_a_discontinuity(self):
        path = trajectory([(0., (0.,)*7, (0.,)*7, (0.,)*7), (.01, (3.,)*7, (0.,)*7, (0.,)*7)])
        self.assertIn("discontinuity", refusal(path, (0.,)*7))

    def test_an_empty_trajectory_is_refused(self):
        from rammp_adl.motion.rolling import MotionError
        with self.assertRaises(MotionError):
            JointTrajectory(JOINTS, (), PROVENANCE)

    def test_a_reordered_joint_list_is_refused_because_the_driver_maps_by_index(self):
        shuffled = (JOINTS[1], JOINTS[0], *JOINTS[2:])
        path = simple()
        path = JointTrajectory(shuffled, path.points, PROVENANCE)
        self.assertIn("is not the driver's", refusal(path, path.points[0].state.position))

    def test_a_non_trajectory_is_refused(self):
        with self.assertRaises(SheppyClientError):
            refusal("not a trajectory", (0.,)*7)


class GripperUnitTests(unittest.TestCase):
    def test_knuckle_radians_normalize_onto_the_wire_setpoint(self):
        self.assertAlmostEqual(setpoint_from_knuckle(KNUCKLE_CLOSED_RAD), 1., places=12)
        self.assertAlmostEqual(setpoint_from_knuckle(0.), 0., places=12)
        self.assertAlmostEqual(setpoint_from_knuckle(.4), .5, places=12)
        self.assertAlmostEqual(knuckle_from_setpoint(.5), .4, places=12)
        self.assertAlmostEqual(knuckle_from_setpoint(setpoint_from_knuckle(.23)), .23, places=12)

    def test_targets_outside_the_knuckle_range_are_refused(self):
        for bad in (-.01, KNUCKLE_CLOSED_RAD+.01, float("nan"), True):
            with self.assertRaises(SheppyClientError):
                setpoint_from_knuckle(bad)
        for bad in (-.01, 1.01, float("inf"), True):
            with self.assertRaises(SheppyClientError):
                knuckle_from_setpoint(bad)


class ResultTests(unittest.TestCase):
    def test_driver_codes_are_named_with_their_own_reason(self):
        self.assertEqual(result_message(0), "SUCCESSFUL")
        self.assertEqual(result_message(-8, "another owner"), "NOT_AUTHORIZED: another owner")
        self.assertIn("UNKNOWN(-99)", result_message(-99))
        # Exactly the codes the pinned action declares; nothing invented.
        self.assertEqual(set(RESULT_NAMES), {0, -1, -4, -5, -6, -8, -9})

    def test_the_surface_description_states_what_this_runtime_does_not_own(self):
        surface = describe_surface()
        self.assertFalse(surface["owns_containers"])
        self.assertFalse(surface["starts_driver"])
        self.assertEqual(surface["joint_order"], list(JOINTS))
        self.assertIn("not a commissioning profile", surface["semantics"])


class PlannerIngestionTests(unittest.TestCase):
    """The wire boundary, where a missing velocity profile can actually appear."""

    def rows(self, *, velocity=(0.,)*7, acceleration=(0.,)*7, start=0.):
        return [(start, (0.,)*7, velocity, acceleration),
                (1., (.05,)*7, (.05,)*7, (0.,)*7),
                (2., (.1,)*7, (0.,)*7, (0.,)*7)]

    def test_a_well_formed_plan_becomes_the_canonical_trajectory(self):
        built = trajectory_from_planner(JOINTS, self.rows(), provenance="rammp_curobo:test")
        self.assertEqual(built.joint_names, JOINTS)
        self.assertEqual(len(built.points), 3)
        self.assertEqual(built.points[0].time_s, 0.)
        self.assertIsNone(refusal(built, (0.,)*7))

    def test_a_missing_velocity_profile_is_refused_rather_than_zero_filled(self):
        with self.assertRaisesRegex(SheppyClientError, "linearly and unchecked"):
            trajectory_from_planner(JOINTS, self.rows(velocity=None), provenance="p")
        with self.assertRaisesRegex(SheppyClientError, "linearly and unchecked"):
            trajectory_from_planner(JOINTS, self.rows(velocity=()), provenance="p")

    def test_the_planners_velocity_only_plan_is_accepted_with_differenced_accelerations(self):
        # RAMMP-CuRobo v1.0.0 fills positions and velocities only; that is every real plan.
        rows = [(time_s, position, velocity, None) for time_s, position, velocity, _ in self.rows()]
        built = trajectory_from_planner(JOINTS, rows, provenance="rammp_curobo:test")
        self.assertIn(DIFFERENCED, built.provenance)
        self.assertEqual(built.points[0].state.acceleration, (.05,)*7)        # (0.05-0)/1 s, one-sided at the ends
        self.assertEqual(built.points[1].state.acceleration, (0.,)*7)         # (0-0)/2 s, central
        self.assertEqual(built.points[2].state.acceleration, (-.05,)*7)
        self.assertIsNone(refusal(built, (0.,)*7))
        slowed = scale_trajectory_time(built, 2.5)
        self.assertIn(DIFFERENCED, slowed.provenance)                         # the wire still omits them after scaling

    def test_accelerations_on_some_waypoints_only_are_refused(self):
        with self.assertRaisesRegex(SheppyClientError, "some waypoints only"):
            trajectory_from_planner(JOINTS, self.rows(acceleration=None), provenance="p")
        with self.assertRaisesRegex(SheppyClientError, "malformed acceleration"):
            trajectory_from_planner(JOINTS, self.rows(acceleration=(0.,)*3), provenance="p")

    def test_a_reversed_path_is_the_same_waypoints_flown_backwards(self):
        built = trajectory_from_planner(JOINTS, self.rows(), provenance="rammp_curobo:test")
        back = reversed_trajectory(built)
        self.assertEqual([p.state.position for p in back.points], [p.state.position for p in reversed(built.points)])
        self.assertEqual(back.points[0].time_s, 0.)
        self.assertEqual(back.duration_s, built.duration_s)
        self.assertEqual(back.points[1].state.velocity, (-.05,)*7)
        self.assertIsNone(refusal(back, (.1,)*7))                              # admitted from where the way in ended
        self.assertIsNotNone(refusal(back, (0.,)*7))                           # and from nowhere else

    def test_the_planners_one_step_ahead_stamping_gets_the_start_state_at_zero(self):
        # RAMMP-CuRobo stamps waypoint k at (k+1)*dt and waypoint 0 is the start state.
        rows = [(.04*(k+1), position, velocity, None) for k, (_, position, velocity, _) in enumerate(self.rows())]
        built = trajectory_from_planner(JOINTS, rows, provenance="rammp_curobo:test")
        self.assertIn(START_PREPENDED, built.provenance)
        self.assertEqual(len(built.points), 4)
        self.assertEqual((built.points[0].time_s, built.points[0].state.position, built.points[0].state.velocity),
                         (0., rows[0][1], (0.,)*7))
        self.assertEqual([p.time_s for p in built.points[1:]], [row[0] for row in rows])
        self.assertIsNone(refusal(built, (0.,)*7))

    def test_order_count_and_provenance_are_required(self):
        with self.assertRaises(SheppyClientError):
            trajectory_from_planner((JOINTS[1], JOINTS[0], *JOINTS[2:]), self.rows(), provenance="p")
        with self.assertRaises(SheppyClientError):
            trajectory_from_planner(JOINTS, self.rows()[:1], provenance="p")
        for bad in ("", "  ", None):
            with self.assertRaises(SheppyClientError):
                trajectory_from_planner(JOINTS, self.rows(), provenance=bad)
        with self.assertRaises(SheppyClientError):
            trajectory_from_planner(JOINTS, [(0., (0.,)*7), (1., (0.,)*7)], provenance="p")


if __name__ == "__main__":
    unittest.main()
