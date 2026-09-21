"""Numeric interpolation checks; GPU evidence is recorded separately."""
import unittest

import numpy as np

from rammp_adl.motion.curobo_stationary import interpolate_stationary_knots
from rammp_adl.motion.rolling import (
    JointLimits, JointState, JointTrajectory, MotionError, TrajectoryPoint, TrajectoryValidator,
)


def as_trajectory(result):
    return JointTrajectory(tuple("j" + str(i) for i in range(result.positions.shape[1])), tuple(
        TrajectoryPoint(i * result.dt, JointState(tuple(q), tuple(v), tuple(a)))
        for i, (q, v, a) in enumerate(zip(result.positions, result.velocities, result.accelerations))
    ), "explicit-interpolation-unit-fixture-not-curobo")


class StationaryInterpolationTests(unittest.TestCase):
    def test_exact_quintic_and_analytic_derivatives_survive_receiver(self):
        times = np.linspace(0., 1., 11)
        q = 10*times**3 - 15*times**4 + 6*times**5
        source = np.column_stack((q, 2.*q - .3))
        before = source.copy()
        result = interpolate_stationary_knots(source, knot_dt_s=.1, maximum_sample_dt_s=.03)
        path = as_trajectory(result)
        self.assertEqual(result.subdivisions, 4)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_allclose(result.positions[::4], source, atol=1e-13)
        for t in np.linspace(0., 1., 97):
            state = path.sample(float(t))
            position = 10*t**3 - 15*t**4 + 6*t**5
            velocity = 30*t**2 - 60*t**3 + 30*t**4
            acceleration = 60*t - 180*t**2 + 120*t**3
            np.testing.assert_allclose(state.position, (position, 2*position-.3), atol=2e-12)
            np.testing.assert_allclose(state.velocity, (velocity, 2*velocity), atol=2e-10)
            np.testing.assert_allclose(state.acceleration, (acceleration, 2*acceleration), atol=2e-8)

    def test_static_duplicated_endpoint_knots_are_preserved_with_true_boundaries(self):
        q = np.array([[0., 1.], [0., 1.], [.1, 1.05], [.18, .98], [.2, .9], [.2, .9], [.2, .9]])
        result = interpolate_stationary_knots(q, knot_dt_s=.07, maximum_sample_dt_s=.02)
        np.testing.assert_allclose(result.positions[::result.subdivisions], q, atol=1e-12)
        for array in (result.velocities, result.accelerations):
            np.testing.assert_allclose(array[[0, -1]], 0., atol=1e-9)
        self.assertAlmostEqual(as_trajectory(result).duration_s, .42)

    def test_changed_curve_must_pass_independent_dynamic_validation(self):
        q = np.array([[0.], [0.], [.5], [0.], [0.], [0.]])
        result = interpolate_stationary_knots(q, knot_dt_s=.02, maximum_sample_dt_s=.01)
        path = as_trajectory(result)
        validator = TrajectoryValidator(JointLimits((-1.,), (1.,), (1.,), (4.,)),
                                        lambda *_: True, sample_dt_s=.001)
        with self.assertRaises(MotionError):
            validator.validate(path, {"world": "fixture"}, now=0., expires_at=1.)

    def test_rejects_malformed_nonfinite_and_unbounded_input(self):
        q = np.zeros((6, 2))
        for changed in (np.zeros((5, 2)), np.zeros((4097, 2)), np.zeros((6, 0)),
                        np.zeros((6, 33)), np.full((6, 2), np.nan), np.zeros(6)):
            with self.subTest(shape=changed.shape), self.assertRaises(MotionError):
                interpolate_stationary_knots(changed, knot_dt_s=.02, maximum_sample_dt_s=.02)
        for dt, maximum in ((0., .02), (True, .02), (.02, float("nan")),
                            (.02, .2), (25., .02), (.02, 1e-9)):
            with self.subTest(dt=dt, maximum=maximum), self.assertRaises(MotionError):
                interpolate_stationary_knots(q, knot_dt_s=dt, maximum_sample_dt_s=maximum)


if __name__ == "__main__":
    unittest.main()
