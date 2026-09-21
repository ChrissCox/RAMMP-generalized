"""Scheduling fixtures only: no physical transport and no cuRobo/GPU claims."""
import asyncio
import time
import unittest

from rammp_adl.motion.guards import TargetObservation, TargetUpdateGuard
from rammp_adl.motion.leasing import PlannerLeaseBroker
from rammp_adl.motion.rolling import (
    BoundaryTolerance, JointLimits, JointState, MotionIdentity, RollingController, TrajectoryValidator,
)
from rammp_adl.motion.session import RollingMotionSession, SessionWorld
from rammp_adl.simulation import fixture_joint_trajectory, fixture_stop_provider


class JointFixturePlanner:
    """Async delay plus explicit joint fixture, never a Cartesian planner."""
    def __init__(self, *, delay=.015):
        self.delay, self.started, self.finished = delay, asyncio.Event(), asyncio.Event()
        self.calls = []

    async def candidate(self, *, moving_boundary, goal_pose, world_config, world_identity):
        self.calls.append((time.monotonic(), moving_boundary, goal_pose, world_identity))
        self.started.set()
        try:
            await asyncio.sleep(self.delay)
            endpoint = list(moving_boundary.position)
            endpoint[0] = goal_pose[0][0]  # Test-only coordinate-to-joint fixture knob.
            return fixture_joint_trajectory(moving_boundary, endpoint, duration_s=.3)
        finally:
            self.finished.set()


class PerfectTrackingFixtureIO:
    hardware_commands = False

    def __init__(self, controller, *, changes=((.02, .12),), ownership_until=None, world_change=None):
        self.controller, self.changes = controller, changes
        self.started = controller.start_at
        self.ownership_until, self.world_change = ownership_until, world_change
        self.commands, self.ownership_checks = [], []
        self.supervisor_stopped = False

    def owns(self, identity, resources):
        self.ownership_checks.append((time.monotonic(), identity, resources))
        return (identity == self.controller.identity and resources == frozenset({"ARM", "PLANNER"})
                and (self.ownership_until is None or time.monotonic()-self.started < self.ownership_until))

    def read_state(self, now):
        # This is an ideal joint tracking fixture, not physical validation.
        trajectory = self.controller.active.trajectory
        return trajectory.sample(max(0., min(trajectory.duration_s, now-self.controller.start_at)))

    def read_target(self, now):
        position = .1
        for at, value in self.changes:
            if now-self.started >= at:
                position = value
        return TargetObservation("same_object", "approach", (position, 0., .4), now, .001)

    def read_world(self, now):
        revision = "new" if self.world_change is not None and now-self.started >= self.world_change else "fixture"
        return SessionWorld({"scene": revision, "world": revision}, {"cuboid": {}}, revision)

    def continuation_valid(self, now, measured, world):
        return True

    def command(self, state, now):
        self.commands.append((now, state, self.controller.generation))

    def supervisor_stop(self, reason, now):
        self.supervisor_stopped = True

    def quiescent(self):
        return self.supervisor_stopped or self.controller.state == "held"

    def goal_satisfied(self, target, measured, world):
        return abs(measured.position[0]-target.position_m[0]) < .001


class RollingSessionTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self, *, delay=.015, budget=.08, changes=((.02, .12),), duration=.5,
                     ownership_until=None, world_change=None, terminal_velocity=None, required_resources=frozenset({"ARM", "PLANNER"})):
        initial_state = JointState((0.,)*7, (0.,)*7, (0.,)*7)
        validator = TrajectoryValidator(JointLimits((-7.,)*7, (7.,)*7, (10.,)*7, (1000.,)*7),
            lambda trajectory, dependencies: bool(dependencies.get("scene")), sample_dt_s=.005)
        initial = fixture_joint_trajectory(initial_state, (.1, 0., 0., 0., 0., 0., 0.), duration_s=duration,
                                           terminal_velocity=terminal_velocity)
        now = time.monotonic()
        certificate = validator.validate(initial, {"scene": "fixture", "world": "fixture"}, now=now, expires_at=now+5.)
        controller = RollingController(MotionIdentity("fixture_task", "move", 1, 1), certificate, validator,
            start_at=now, tolerance=BoundaryTolerance(.02, 1., 100.), stop_provider=fixture_stop_provider(validator, duration_s=.06),
            stop_budget_s=.06, switch_lead_s=.005, activation_lateness_s=.03)
        io = PerfectTrackingFixtureIO(controller, changes=changes, ownership_until=ownership_until, world_change=world_change)
        initial_target = TargetObservation("same_object", "approach", (.1, 0., .4), now, .001)
        guard = TargetUpdateGuard(initial_target, max_correction_m=.05, max_uncertainty_m=.005,
                                  max_age_s=.1, meaningful_change_m=.002)
        planner = JointFixturePlanner(delay=delay)
        broker = PlannerLeaseBroker(maximum_call_s=budget)
        session = RollingMotionSession(controller=controller, planner=planner, broker=broker, guard=guard, io=io,
            required_resources=required_resources,
            planning_lead_s=.18, tick_period_s=.002, maximum_tick_gap_s=.12, maximum_duration_s=2., evidence_lifetime_s=2.)
        return session, controller, planner, broker, io

    async def test_candidate_plans_while_commands_continue_then_future_switches(self):
        session, controller, planner, broker, io = self.make_session(delay=.03)
        task = asyncio.create_task(session.run())
        await planner.started.wait()
        commands_at_start = len(io.commands)
        self.assertFalse(session.quiescent())
        await planner.finished.wait()
        self.assertGreater(len(io.commands), commands_at_start)
        result = await task
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.generation, 1)
        self.assertTrue(result.backend_quiescent)
        self.assertTrue(session.quiescent())
        self.assertFalse(result.hardware_validated)
        self.assertTrue(any(event["event"] == "activated" for event in controller.events))
        self.assertTrue(all(event["active_motion"] for event in broker.events))
        self.assertGreater(abs(planner.calls[0][1].velocity[0]), .01)
        self.assertGreater(len(io.ownership_checks), len(io.commands))

    async def test_target_changes_during_work_discard_old_candidate_then_replan(self):
        session, controller, planner, broker, io = self.make_session(delay=.05, changes=((.02, .12), (.035, .13)))
        result = await session.run()
        self.assertEqual(result.status, "succeeded")
        self.assertGreaterEqual(len(planner.calls), 2)
        self.assertTrue(any(event["event"] == "planning_rejected" and "superseded" in event["reason"] for event in session.events))
        self.assertEqual(result.generation, 1)

    async def test_target_change_after_install_before_switch_discards_pending_generation(self):
        session, controller, planner, broker, io = self.make_session(delay=.01, changes=((.02, .12), (.11, .13)))
        result = await session.run()
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.generation, 1)
        self.assertEqual(len(planner.calls), 2)
        self.assertTrue(any(event["event"] == "candidate_rejected" and "before activation" in event["reason"] for event in controller.events))

    async def test_late_planner_stops_motion_before_worker_drains(self):
        session, controller, planner, broker, io = self.make_session(delay=.18, budget=.04)
        task = asyncio.create_task(session.run())
        await planner.started.wait()
        await asyncio.sleep(.12)
        self.assertEqual(controller.state, "held")
        self.assertFalse(planner.finished.is_set())
        self.assertIsNotNone(broker.owner)
        self.assertFalse(session.quiescent())
        self.assertFalse(task.done())
        result = await task
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "planner_deadline")
        self.assertTrue(result.backend_quiescent)
        self.assertIsNone(broker.owner)

    async def test_cancel_retains_lease_until_uncancellable_work_and_hold_finish(self):
        session, controller, planner, broker, io = self.make_session(delay=.12, budget=.13)
        cancel = asyncio.Event()
        task = asyncio.create_task(session.run(cancel))
        await planner.started.wait()
        cancel.set()
        await asyncio.sleep(.08)
        self.assertEqual(controller.state, "held")
        self.assertFalse(task.done())
        self.assertIsNotNone(broker.owner)
        result = await task
        self.assertEqual(result.status, "cancelled")
        self.assertTrue(planner.finished.is_set())
        self.assertTrue(session.quiescent())

    async def test_repeated_python_task_cancellation_still_drains_solver_and_holds(self):
        session, controller, planner, broker, io = self.make_session(delay=.12, budget=.13)
        task = asyncio.create_task(session.run())
        await planner.started.wait()
        task.cancel()
        await asyncio.sleep(.02)
        task.cancel()
        self.assertFalse(session.quiescent())
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(controller.state, "held")
        self.assertTrue(planner.finished.is_set())
        self.assertIsNone(broker.owner)
        self.assertTrue(session.quiescent())

    async def test_ownership_revocation_never_sends_more_trajectory_commands(self):
        session, controller, planner, broker, io = self.make_session(delay=.09, budget=.1, ownership_until=.05)
        result = await session.run()
        self.assertEqual(result.status, "failed")
        self.assertTrue(io.supervisor_stopped)
        self.assertTrue(all(at-io.started < .05 for at, _, _ in io.commands))
        self.assertEqual(controller.generation, 0)
        self.assertTrue(planner.finished.is_set())
        self.assertIsNone(broker.owner)

    async def test_world_change_revokes_old_certificate_and_stops_without_switch(self):
        session, controller, planner, broker, io = self.make_session(delay=.09, budget=.1, world_change=.05)
        result = await session.run()
        self.assertEqual(result.status, "failed")
        self.assertEqual(controller.generation, 0)
        self.assertTrue(result.backend_quiescent)
        self.assertTrue(any(event["event"] == "stop_started" for event in controller.events))

    async def test_gripper_ownership_is_required_when_catalog_claims_it(self):
        # Trusted construction mirrors a carried-object skill's complete claims.
        session, controller, planner, broker, io = self.make_session(required_resources={"ARM", "PLANNER", "GRIPPER"})
        seen = []
        def owns(identity, resources):
            seen.append(resources)
            return resources <= {"ARM", "PLANNER"}  # Gripper lease was revoked.
        io.owns = owns
        result = await session.run()
        self.assertEqual(result.status, "failed")
        self.assertFalse(io.commands)
        self.assertFalse(planner.calls)
        self.assertTrue(seen and all("GRIPPER" in resources for resources in seen))


if __name__ == "__main__":
    unittest.main()
