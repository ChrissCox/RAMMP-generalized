import asyncio
from dataclasses import replace
import math
import unittest

from rammp_adl.motion import (
    BoundaryTolerance, Candidate, JointLimits, JointState, JointTrajectory,
    MotionError, MotionIdentity, RollingController, TrajectoryPoint, TrajectoryValidator,
)
from rammp_adl.motion.guards import ConstraintGeometry, TargetObservation, TargetUpdateGuard
from rammp_adl.motion.leasing import PlannerLeaseBroker
from rammp_adl.motion.curobo import RammpCuroboAdapter, CuroboUnavailable, RAMMP_COMMIT
from rammp_adl.simulation import fixture_joint_trajectory, fixture_stop_provider, rolling_fixture_trace


class RollingTests(unittest.TestCase):
    def setUp(self):
        self.zero = JointState((0.,)*7, (0.,)*7, (0.,)*7)
        self.dependencies = {"scene": "clear", "calibration": "v1"}
        self.validator = TrajectoryValidator(JointLimits((-6.,)*7, (6.,)*7, (3.,)*7, (20.,)*7), lambda trajectory, deps: deps.get("scene") != "unsafe")
        self.initial = fixture_joint_trajectory(self.zero, (0.2,)*7, duration_s=2.)
        initial = self.certify(self.initial)
        self.controller = RollingController(MotionIdentity("task", "move", 1, 9), initial, self.validator, start_at=0.,
                                            tolerance=BoundaryTolerance(0.001, 0.001, 0.005),
                                            stop_provider=fixture_stop_provider(self.validator), stop_budget_s=0.5)

    def certify(self, trajectory, *, expires_at=10., dependencies=None):
        return self.validator.validate(trajectory, dependencies or self.dependencies, now=0., expires_at=expires_at)

    def candidate(self, switch_at=0.8):
        boundary = self.initial.sample(switch_at)
        suffix = fixture_joint_trajectory(boundary, (0.3,)*7, duration_s=1.5)
        return Candidate(self.controller.identity, 0, 1, switch_at, boundary, self.certify(suffix))

    def test_moving_target_suffix_continuity_and_epoch_preserved(self):
        candidate = self.candidate()
        self.assertGreater(abs(candidate.expected_boundary.velocity[0]), 0.01)
        self.controller.install(candidate, now=0.5, dependencies=self.dependencies)
        self.controller.tick(now=0.8, measured=candidate.expected_boundary, dependencies=self.dependencies)
        self.assertEqual(self.controller.generation, 1)
        self.assertEqual(self.controller.identity.execution_epoch, 9)
        self.assertEqual(self.controller.active.trajectory.points[0].state, candidate.expected_boundary)

    def test_state_and_path_copy_mutable_inputs_before_validation(self):
        q = [0.]*7
        state = JointState(q, [0.]*7, [0.]*7)
        q[0] = 2.
        self.assertEqual(state.position, (0.,)*7)
        points = list(self.initial.points)
        names = list(self.initial.joint_names)
        trajectory = JointTrajectory(names, points, "test")
        before = trajectory.digest
        points.clear()
        names[0] = "different"
        self.assertEqual(before, trajectory.digest)

    def test_invalid_clocks_and_numeric_strings_cannot_bypass_admission(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(MotionError):
                self.validator.validate(self.initial, self.dependencies, now=value, expires_at=10.)
            with self.assertRaises(MotionError):
                self.validator.check_certificate(self.certify(self.initial), self.dependencies, value)
            with self.assertRaises(MotionError):
                replace(self.candidate(), switch_at=value)
            with self.assertRaises(MotionError):
                PlannerLeaseBroker(maximum_call_s=value)
        with self.assertRaises(MotionError):
            JointState(("0",)*7, (0.,)*7, (0.,)*7)

    def test_late_candidate_rejected(self):
        with self.assertRaisesRegex(MotionError, "deadline"):
            self.controller.install(self.candidate(), now=0.79, dependencies=self.dependencies)

    def test_out_of_order_generation_rejected(self):
        with self.assertRaisesRegex(MotionError, "generation"):
            self.controller.install(replace(self.candidate(), next_generation=3), now=0.5, dependencies=self.dependencies)

    def test_second_pending_cannot_overwrite_accepted_generation(self):
        self.controller.install(self.candidate(), now=0.5, dependencies=self.dependencies)
        with self.assertRaisesRegex(MotionError, "already pending"):
            self.controller.install(self.candidate(), now=0.5, dependencies=self.dependencies)

    def test_wrong_epoch_rejected(self):
        with self.assertRaisesRegex(MotionError, "ownership"):
            self.controller.install(replace(self.candidate(), identity=MotionIdentity("task", "move", 1, 8)), now=0.5, dependencies=self.dependencies)

    def test_discontinuous_derivatives_rejected(self):
        candidate = self.candidate()
        wrong = replace(candidate.expected_boundary, velocity=(0.,)*7)
        trajectory = fixture_joint_trajectory(wrong, (0.3,)*7, duration_s=1.5)
        with self.assertRaisesRegex(MotionError, "discontinuous"):
            self.controller.install(replace(candidate, certificate=self.certify(trajectory)), now=0.5, dependencies=self.dependencies)

    def test_switch_time_rechecks_actual_state_and_stops(self):
        candidate = self.candidate()
        self.controller.install(candidate, now=0.5, dependencies=self.dependencies)
        wrong = replace(candidate.expected_boundary, position=(0.01,)*7)
        self.controller.tick(now=0.8, measured=wrong, dependencies=self.dependencies)
        self.assertEqual(self.controller.generation, 0)
        self.assertEqual(self.controller.state, "stopping")
        self.assertTrue(any("activation state" in event.get("reason", "") for event in self.controller.events))

    def test_switch_expired_candidate_leaves_valid_continuation(self):
        candidate = self.candidate()
        candidate = replace(candidate, certificate=self.certify(candidate.certificate.trajectory, expires_at=0.7))
        self.controller.install(candidate, now=0.5, dependencies=self.dependencies)
        self.controller.tick(now=0.8, measured=candidate.expected_boundary, dependencies=self.dependencies)
        self.assertEqual(self.controller.state, "running")
        self.assertEqual(self.controller.generation, 0)

    def test_changed_dependencies_rejected_at_activation(self):
        candidate = self.candidate()
        self.controller.install(candidate, now=0.5, dependencies=self.dependencies)
        dependencies = {**self.dependencies, "calibration": "v2"}
        self.controller.tick(now=0.8, measured=candidate.expected_boundary, dependencies=dependencies)
        self.assertEqual(self.controller.generation, 0)
        self.assertEqual(self.controller.state, "stopping")

    def test_unrelated_entity_update_preserves_validation(self):
        self.controller.tick(now=0.2, measured=self.initial.sample(0.2), dependencies={**self.dependencies, "unrelated_cup": "new"})
        self.assertEqual(self.controller.state, "running")

    def test_revalidation_requires_same_exact_trajectory(self):
        with self.assertRaisesRegex(MotionError, "exact active"):
            self.controller.revalidate_active(self.candidate().certificate, now=0.2, dependencies=self.dependencies)

    def test_new_obstacle_invalidates_stop_horizon_without_waiting(self):
        self.controller.tick(now=0.2, measured=self.initial.sample(0.2), dependencies={"scene": "unsafe", "calibration": "v1"}, continuation_valid=False)
        self.assertEqual(self.controller.state, "fault")
        self.assertEqual(self.controller.events[-1]["event"], "supervisor_required")

    def test_cancel_revokes_candidate_and_bounded_stop(self):
        self.controller.install(self.candidate(), now=0.5, dependencies=self.dependencies)
        self.controller.request_stop("cancelled", now=0.6, measured=self.initial.sample(0.6), dependencies=self.dependencies)
        self.assertIsNone(self.controller.pending)
        with self.assertRaises(MotionError):
            self.controller.install(self.candidate(), now=0.6, dependencies=self.dependencies)
        stop = self.controller.active.trajectory
        self.controller.tick(now=1.1, measured=stop.points[-1].state, dependencies=self.dependencies)
        self.assertEqual(self.controller.state, "held")
        self.assertLessEqual(stop.duration_s, self.controller.stop_budget_s)

    def test_buffer_underrun_stops_before_nonstationary_tail(self):
        moving = fixture_joint_trajectory(self.zero, (0.2,)*7, duration_s=2., terminal_velocity=(0.1,)*7)
        self.controller.active = self.certify(moving)
        self.controller.tick(now=1.5, measured=moving.sample(1.5), dependencies=self.dependencies)
        self.assertEqual(self.controller.stop_reason, "buffer_underrun")
        self.assertEqual(self.controller.state, "stopping")

    def test_evidence_horizon_depletion_stops_early(self):
        self.controller.active = self.certify(self.initial, expires_at=1.)
        self.controller.tick(now=0.5, measured=self.initial.sample(0.5), dependencies=self.dependencies)
        self.assertEqual(self.controller.stop_reason, "evidence_horizon_depleted")

    def test_invalid_limits_and_nonfinite_rejected(self):
        with self.assertRaises(MotionError):
            JointState((float("nan"),), (0.,), (0.,))
        with self.assertRaises(MotionError):
            self.certify(fixture_joint_trajectory(self.zero, (20.,)*7, duration_s=1.))

    def test_forged_certificate_rejected(self):
        certificate = replace(self.candidate().certificate, validator_token=object())
        with self.assertRaisesRegex(MotionError, "untrusted"):
            self.controller.install(replace(self.candidate(), certificate=certificate), now=0.5, dependencies=self.dependencies)

    def test_quintic_interpolation_derivatives_agree_with_finite_difference(self):
        t, epsilon = 0.6, 1e-5
        center, before, after = self.initial.sample(t), self.initial.sample(t-epsilon), self.initial.sample(t+epsilon)
        self.assertAlmostEqual(center.velocity[0], (after.position[0]-before.position[0])/(2*epsilon), places=6)
        self.assertAlmostEqual(center.acceleration[0], (after.velocity[0]-before.velocity[0])/(2*epsilon), places=6)

    def test_trace_is_explicitly_fixture_only(self):
        trace = rolling_fixture_trace()
        self.assertEqual(trace["state"], "held")
        self.assertFalse(trace["curobo_planned"])
        self.assertFalse(trace["physics_validated"])


class GeometryGuardTests(unittest.TestCase):
    def test_target_inputs_are_immutable_and_capture_identity_cannot_be_reused(self):
        position = [0., 0., 0.]
        initial = TargetObservation("handle", "grasp", position, 0., .001)
        position[0] = 1.
        self.assertEqual(initial.position_m, (0., 0., 0.))
        guard = TargetUpdateGuard(initial, max_correction_m=.03, max_uncertainty_m=.005, max_age_s=.2, meaningful_change_m=.002)
        with self.assertRaisesRegex(MotionError, "same capture"):
            guard.accept(replace(initial, position_m=(.001, 0., 0.)), now=0.)

    def test_nonfinite_target_or_constraint_limits_cannot_disable_envelopes(self):
        initial = TargetObservation("handle", "grasp", (0., 0., 0.), 0., .001)
        geometry = ConstraintGeometry("hinge", "handle", "revolute", (0., 0., 0.), (0., 0., 1.), (.5, 0., 0.), 0., 1.5, .002, frozenset({("gripper", "handle")}))
        for value in (float("nan"), float("inf")):
            with self.assertRaises(MotionError):
                TargetUpdateGuard(initial, max_correction_m=value, max_uncertainty_m=.005, max_age_s=.2, meaningful_change_m=.002)
            with self.assertRaises(MotionError):
                replace(geometry, path_tolerance_m=value)
            with self.assertRaises(MotionError):
                geometry.check_refinement(geometry, max_origin_shift_m=value, max_axis_angle_rad=.01)

    def test_pose_update_bounds_are_cumulative(self):
        initial = TargetObservation("handle", "grasp", (0., 0., 0.), 0., 0.001)
        guard = TargetUpdateGuard(initial, max_correction_m=0.03, max_uncertainty_m=0.005, max_age_s=0.2, meaningful_change_m=0.002)
        self.assertTrue(guard.accept(replace(initial, position_m=(0.02, 0., 0.), captured_at=0.1), now=0.1))
        with self.assertRaisesRegex(MotionError, "cumulative"):
            guard.accept(replace(initial, position_m=(0.04, 0., 0.), captured_at=0.2), now=0.2)
        with self.assertRaisesRegex(MotionError, "identity"):
            guard.accept(replace(initial, entity_id="other", captured_at=0.2), now=0.2)

    def test_pose_noise_and_stale_evidence(self):
        initial = TargetObservation("handle", "grasp", (0., 0., 0.), 0., 0.001)
        guard = TargetUpdateGuard(initial, max_correction_m=0.03, max_uncertainty_m=0.005, max_age_s=0.2, meaningful_change_m=0.002)
        self.assertFalse(guard.accept(replace(initial, position_m=(0.001, 0., 0.), captured_at=0.1), now=0.1))
        self.assertTrue(guard.accept(replace(initial, position_m=(0.0021, 0., 0.), captured_at=0.2), now=0.2))
        with self.assertRaises(MotionError):
            guard.accept(initial, now=1.)

    def test_target_rotation_is_bounded_and_quaternion_sign_equivalent(self):
        initial = TargetObservation("handle", "grasp", (0., 0., 0.), 0., 0.001)
        guard = TargetUpdateGuard(initial, max_correction_m=0.03, max_uncertainty_m=0.005, max_age_s=0.2, meaningful_change_m=0.002)
        self.assertFalse(guard.accept(replace(initial, quaternion_xyzw=(0., 0., 0., -1.), captured_at=0.1), now=0.1))
        with self.assertRaisesRegex(MotionError, "rotation"):
            guard.accept(replace(initial, quaternion_xyzw=(0., 0., math.sin(.2), math.cos(.2)), captured_at=0.2), now=0.2)

    def test_constraint_motion_checks_contact_axis_units_and_refinement(self):
        geometry = ConstraintGeometry("hinge", "handle", "revolute", (0., 0., 0.), (0., 0., 1.), (0.5, 0., 0.), 0., 1.5, 0.002, frozenset({("gripper", "handle")}))
        point = geometry.point_at(1.)
        geometry.check(1., point, {("gripper", "handle")}, target_unit="rad")
        with self.assertRaisesRegex(MotionError, "unmodeled"):
            geometry.check(1., point, {("wrist", "door")}, target_unit="rad")
        with self.assertRaisesRegex(MotionError, "model_mismatch"):
            geometry.check(1., (0.5, 0., 0.), {("gripper", "handle")}, target_unit="rad")
        with self.assertRaisesRegex(MotionError, "units"):
            geometry.check(1., point, {("gripper", "handle")}, target_unit="m")
        geometry.check_refinement(replace(geometry, origin_m=(0.001, 0., 0.)), max_origin_shift_m=0.003, max_axis_angle_rad=0.01)
        with self.assertRaisesRegex(MotionError, "envelope"):
            geometry.check_refinement(replace(geometry, origin_m=(0.1, 0., 0.)), max_origin_shift_m=0.003, max_axis_angle_rad=0.01)


class LeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_work_has_priority_over_queued_speculation(self):
        broker = PlannerLeaseBroker(maximum_call_s=1.)
        acquired = []
        async def work(name, active):
            async with broker.lease(name, active_motion=active, phase_authorized=True):
                acquired.append(name)
        async with broker.lease("initial", active_motion=True):
            speculative = asyncio.create_task(work("speculative", False))
            active = asyncio.create_task(work("active", True))
            await asyncio.sleep(0)
        await asyncio.gather(speculative, active)
        self.assertEqual(acquired, ["active", "speculative"])

    async def test_speculation_without_phase_rejected(self):
        broker = PlannerLeaseBroker(maximum_call_s=1.)
        with self.assertRaisesRegex(MotionError, "phase"):
            async with broker.lease("speculative", active_motion=False):
                self.fail("lease incorrectly acquired")

    async def test_cancellation_waits_for_nonpreemptible_solver(self):
        broker = PlannerLeaseBroker(maximum_call_s=1.)
        started, release = asyncio.Event(), asyncio.Event()
        async def operation():
            started.set()
            await release.wait()
        task = asyncio.create_task(broker.run("active", operation, active_motion=True))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertEqual(broker.owner, "active")
        task.cancel()
        await asyncio.sleep(0)
        self.assertEqual(broker.owner, "active")
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(broker.owner)


class CuroboBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_static_adapter_refuses_nonzero_velocity(self):
        adapter = RammpCuroboAdapter(object(), source_commit=RAMMP_COMMIT, installed_module_path=__file__, source_root=__file__)
        with self.assertRaisesRegex(CuroboUnavailable, "moving"):
            await adapter.plan_pose(position_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.),
                                    start=JointState((0.,)*7, (0.1,)*7, (0.,)*7), world=[], world_identity="v1")


if __name__ == "__main__":
    unittest.main()
