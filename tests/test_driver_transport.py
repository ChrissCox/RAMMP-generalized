"""Fault injection for the command gateway, not physical stopping validation."""
import asyncio
from dataclasses import replace
import sys
from types import SimpleNamespace as NS
import time
import unittest
from unittest.mock import patch

from rammp_adl.motion.driver_state import ARM_JOINT_NAMES
from rammp_adl.motion.driver_transport import (
    DriverTransportError, JointTrajectoryTransport, TransferredOwnership,
    TransportBounds, VerifiedMotionState, canonical_ros_trajectory, make_ros_goal,
)
from rammp_adl.motion.rolling import JointState, JointTrajectory, TrajectoryPoint


def fixture_trajectory():
    zero = JointState((0.,)*7, (0.,)*7, (0.,)*7)
    end = JointState((.01,)*7, (0.,)*7, (0.,)*7)
    return JointTrajectory(ARM_JOINT_NAMES, (TrajectoryPoint(0., zero), TrajectoryPoint(.02, end)),
                           "test-only synthetic transport fixture; not cuRobo")


def bounds():
    return TransportBounds(path_position_rad=(.1,)*7, goal_position_rad=(.002,)*7,
        start_position_rad=(.002,)*7, stationary_velocity_rad_s=.001,
        stationary_position_span_rad=.001, state_max_age_s=.1, receipt_max_age_s=.1,
        poll_period_s=.001, send_timeout_s=.015, cancel_timeout_s=.01,
        stop_timeout_s=.045, settle_duration_s=.005, result_slack_s=.02,
        maximum_trajectory_s=1.)


def terminal(code=0, status=4):
    return NS(status=status, result=NS(error_code=code, error_string=""))


def future(value=None, *, pending=False):
    result = asyncio.get_running_loop().create_future()
    if not pending:
        result.set_result(value)
    return result


class FakeHandle:
    def __init__(self, port):
        self.port, self.accepted = port, True
        self.goal_id = NS(uuid=bytes(range(16)))
        self.result_future = future(pending=True)

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.port.cancelled += 1
        if self.port.cancel_hangs:
            return future(pending=True)
        if self.port.cancel_settles and not self.result_future.done():
            self.port.velocity = 0.
            self.result_future.set_result(terminal(-9, 5))
        wanted = self.goal_id if not self.port.wrong_cancel_id else NS(uuid=b"x"*16)
        return future(NS(return_code=0, goals_canceling=[NS(goal_id=wanted)]))


class FakePort:
    def __init__(self):
        self.sent, self.cancelled, self.released, self.stops = [], 0, 0, []
        self.position, self.velocity = 0., 0.
        self.sequence = 0
        self.source_id, self.stale, self.frozen = "sim-driver-session", False, None
        self.ready = True
        self.send_hangs = self.cancel_hangs = self.wrong_cancel_id = False
        self.cancel_settles = True
        self.auto_finish = True
        self.finish_velocity = 0.
        self.handle = self.send_future = None

    def server_ready(self):
        return self.ready

    def send_goal(self, trajectory, ownership, configured_bounds):
        self.sent.append(trajectory)
        self.handle = FakeHandle(self)
        self.send_future = future(self.handle, pending=self.send_hangs)
        if self.auto_finish and not self.send_hangs:
            asyncio.get_running_loop().call_later(.003, self.finish)
        return self.send_future

    def finish(self):
        if not self.handle.result_future.done():
            self.position = .01
            self.velocity = self.finish_velocity
            self.handle.result_future.set_result(terminal())

    def publish_stop(self, owner, reason):
        self.stops.append((owner, reason))

    def release(self, token):
        self.released += 1
        return future(NS(released=True))

    def state(self, now):
        if self.frozen is not None:
            return self.frozen
        self.sequence += 1
        return VerifiedMotionState((self.position,)*7, (self.velocity,)*7,
            now-(1. if self.stale else .000001), now, self.sequence, self.source_id)


def make_transport(port=None, check=None):
    port = port or FakePort()
    owner = TransferredOwnership("fixture-orchestrator", bytes(range(1,17)), 7)
    transport = JointTrajectoryTransport(port=port, ownership=owner,
        admission_check=check or (lambda permit, trajectory: permit == "trusted-fixture-permit"),
        state_check=port.state, bounds=bounds())
    transport.update_control_status(NS(arbitration_enabled=True, estopped=False, owned=True,
                                      owner_id=owner.owner_id, generation=owner.generation))
    return transport, port


class DriverTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_action_requires_post_terminal_measured_settling_before_release(self):
        transport, port = make_transport()
        receipt = await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit")
        self.assertEqual(receipt.status, "succeeded")
        self.assertTrue(receipt.terminal_acknowledged)
        self.assertTrue(receipt.measured_quiescent)
        self.assertTrue(receipt.release_permitted)
        self.assertEqual(port.released, 0)
        self.assertGreater(port.sequence, 2)
        await transport.release()
        self.assertEqual(port.released, 1)
        with self.assertRaises(DriverTransportError):
            await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit")

    async def test_rejected_permit_sends_no_motion_or_stop(self):
        transport, port = make_transport()
        with self.assertRaisesRegex(DriverTransportError, "permit"):
            await transport.execute(fixture_trajectory(), permit="untrusted")
        self.assertEqual(port.sent, [])
        self.assertEqual(port.stops, [])

    async def test_stale_or_moving_start_and_wrong_joint_order_are_not_dispatched(self):
        for bad in ("stale", "moving", "joint_order", "nonstationary_boundary", "timing"):
            transport, port = make_transport()
            trajectory = fixture_trajectory()
            if bad == "stale":
                port.stale = True
            elif bad == "moving":
                port.velocity = .2
            elif bad == "joint_order":
                trajectory = replace(trajectory, joint_names=tuple(reversed(ARM_JOINT_NAMES)))
            elif bad == "nonstationary_boundary":
                last = trajectory.points[-1]
                state = replace(last.state, acceleration=(.001,)*7)
                trajectory = replace(trajectory, points=(trajectory.points[0], replace(last, state=state)))
            else:
                trajectory = replace(trajectory, points=(trajectory.points[0],
                    replace(trajectory.points[-1], time_s=.0200000001)))
            with self.subTest(bad=bad), self.assertRaises(DriverTransportError):
                await transport.execute(trajectory, permit="trusted-fixture-permit")
            self.assertEqual(port.sent, [])

    async def test_completion_message_does_not_claim_hold_while_measured_robot_moves(self):
        transport, port = make_transport()
        port.finish_velocity = .2
        receipt = await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit")
        self.assertEqual(receipt.status, "failed")
        self.assertTrue(receipt.terminal_acknowledged)
        self.assertFalse(receipt.measured_quiescent)
        self.assertTrue(receipt.software_stop_published)
        self.assertEqual(port.released, 0)
        with self.assertRaises(DriverTransportError):
            await transport.release()

    async def test_cancel_requires_exact_action_ack_and_independent_hold(self):
        transport, port = make_transport()
        port.auto_finish = False
        event = asyncio.Event()
        asyncio.get_running_loop().call_later(.004, event.set)
        receipt = await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit", cancel_event=event)
        self.assertEqual(receipt.status, "cancelled")
        self.assertEqual(port.cancelled, 1)
        self.assertTrue(receipt.measured_quiescent)
        self.assertFalse(receipt.software_stop_published)
        self.assertTrue(receipt.release_permitted)

    async def test_unresolved_cancel_escalates_and_retains_local_claims(self):
        transport, port = make_transport()
        port.auto_finish, port.cancel_settles, port.cancel_hangs = False, False, True
        event = asyncio.Event()
        asyncio.get_running_loop().call_later(.003, event.set)
        receipt = await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit", cancel_event=event)
        self.assertEqual(receipt.status, "failed")
        self.assertFalse(receipt.terminal_acknowledged)
        self.assertFalse(receipt.release_permitted)
        self.assertTrue(receipt.software_stop_published)
        self.assertEqual(len(port.stops), 1)
        with self.assertRaises(DriverTransportError):
            await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit")

    async def test_wrong_cancel_uuid_is_not_acknowledgement(self):
        transport, port = make_transport()
        port.auto_finish, port.wrong_cancel_id = False, True
        event = asyncio.Event()
        asyncio.get_running_loop().call_later(.003, event.set)
        receipt = await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit", cancel_event=event)
        self.assertTrue(receipt.software_stop_published)
        self.assertFalse(receipt.release_permitted)

    async def test_send_timeout_prevents_late_acceptance_and_cancels_exact_late_handle(self):
        transport, port = make_transport()
        port.send_hangs = True
        receipt = await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit")
        self.assertFalse(receipt.terminal_acknowledged)
        self.assertTrue(receipt.software_stop_published)
        self.assertFalse(receipt.release_permitted)
        port.send_future.set_result(port.handle)
        await asyncio.sleep(.001)
        self.assertEqual(port.cancelled, 1)

    async def test_ownership_loss_never_reacquires_or_releases_replacement_owner(self):
        transport, port = make_transport()
        port.auto_finish = False
        def takeover():
            transport.update_control_status(NS(arbitration_enabled=True, estopped=False, owned=True,
                                                owner_id="operator", generation=8))
        asyncio.get_running_loop().call_later(.003, takeover)
        receipt = await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit")
        self.assertFalse(receipt.ownership_retained)
        self.assertFalse(receipt.release_permitted)
        self.assertTrue(receipt.software_stop_published)
        self.assertEqual(port.released, 0)
        transport.update_control_status(NS(arbitration_enabled=True, estopped=False, owned=True,
            owner_id=transport.ownership.owner_id, generation=7))
        self.assertFalse(transport.owns_control())

    async def test_frozen_or_relabelled_measurement_cannot_establish_post_terminal_hold(self):
        for relabel in (False, True):
            transport, port = make_transport()
            sample = port.state(time.monotonic())
            if relabel:
                transport.state_check = lambda now: replace(sample, received_at_monotonic_s=now)
            else:
                port.frozen = sample
            receipt = await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit")
            self.assertFalse(receipt.measured_quiescent)
            self.assertFalse(receipt.release_permitted)
            self.assertTrue(receipt.software_stop_published)

    async def test_revoked_permit_stops_inflight_goal(self):
        valid = [True]
        transport, port = make_transport(check=lambda p,t: valid[0])
        port.auto_finish = False
        asyncio.get_running_loop().call_later(.003, lambda: valid.__setitem__(0, False))
        receipt = await transport.execute(fixture_trajectory(), permit=object())
        self.assertEqual(receipt.status, "failed")
        self.assertEqual(port.cancelled, 1)
        self.assertTrue(receipt.measured_quiescent)

    async def test_caller_coroutine_cancellation_drains_stop_before_propagating(self):
        transport, port = make_transport()
        port.auto_finish = False
        work = asyncio.create_task(transport.execute(fixture_trajectory(), permit="trusted-fixture-permit"))
        await asyncio.sleep(.003)
        work.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await work
        self.assertTrue(transport._last_receipt.measured_quiescent)
        self.assertTrue(transport._last_receipt.terminal_acknowledged)
        self.assertFalse(transport._running)

    async def test_second_goal_cannot_overlap_first(self):
        transport, port = make_transport()
        work = asyncio.create_task(transport.execute(fixture_trajectory(), permit="trusted-fixture-permit"))
        await asyncio.sleep(.001)
        with self.assertRaisesRegex(DriverTransportError, "busy"):
            await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit")
        await work
        self.assertEqual(len(port.sent), 1)

    async def test_malformed_terminal_is_not_acknowledged(self):
        transport, port = make_transport()
        port.auto_finish = False
        asyncio.get_running_loop().call_later(.003, lambda: port.handle.result_future.set_result(NS(status=2)))
        receipt = await transport.execute(fixture_trajectory(), permit="trusted-fixture-permit")
        self.assertFalse(receipt.terminal_acknowledged)
        self.assertFalse(receipt.release_permitted)

    def test_quantization_precedes_admission_and_preserves_joint_derivatives(self):
        trajectory = fixture_trajectory()
        trajectory = replace(trajectory, points=(trajectory.points[0],
            replace(trajectory.points[-1], time_s=.0200000001)))
        canonical = canonical_ros_trajectory(trajectory)
        self.assertNotEqual(canonical.digest, trajectory.digest)
        self.assertEqual(canonical.points[-1].time_s, .02)
        self.assertEqual(canonical.points[-1].state, trajectory.points[-1].state)

    def test_split_interface_goal_preserves_all_derivatives_and_authority(self):
        # ROS packages remain optional for the pure runtime test suite. This
        # checks wire field mapping; actual generated types are exercised by the
        # isolated DDS reproducer, not claimed by these stand-ins.
        class Goal:
            def __init__(self):
                self.trajectory = NS(joint_names=[], points=[])
                self.path_tolerance, self.goal_tolerance = [], []
                self.goal_time_tolerance = NS(sec=0, nanosec=0)
        class Point:
            def __init__(self):
                self.time_from_start = NS(sec=0, nanosec=0)
        modules = {
            "control_msgs.msg": NS(JointTolerance=NS),
            "trajectory_msgs.msg": NS(JointTrajectoryPoint=Point),
            "rammp_arm_interfaces.action": NS(ExecuteJointTrajectory=NS(Goal=Goal)),
        }
        owner = TransferredOwnership("fixture-owner", bytes(range(1,17)), 2)
        path = fixture_trajectory()
        middle = TrajectoryPoint(.012345678, JointState((.004,)*7, (.02,)*7, (-.03,)*7))
        path = replace(path, points=(path.points[0], middle, path.points[-1]))
        configured = bounds()
        with patch.dict(sys.modules, modules):
            goal = make_ros_goal(path, owner, configured)
        self.assertEqual(goal.trajectory.joint_names, list(ARM_JOINT_NAMES))
        for actual, expected in zip(goal.trajectory.points, path.points):
            self.assertEqual(actual.positions, list(expected.state.position))
            self.assertEqual(actual.velocities, list(expected.state.velocity))
            self.assertEqual(actual.accelerations, list(expected.state.acceleration))
            self.assertEqual(actual.time_from_start.sec * 10**9 + actual.time_from_start.nanosec,
                             round(expected.time_s * 10**9))
        self.assertEqual((goal.control_mode, goal.preemption), (0, 0))
        self.assertEqual((goal.sender_id, goal.token), (owner.owner_id, list(owner.token)))
        self.assertEqual([value.name for value in goal.path_tolerance], list(ARM_JOINT_NAMES))
        self.assertEqual([value.position for value in goal.goal_tolerance], list(configured.goal_position_rad))

    def test_token_redaction_and_bounds_validation(self):
        owner = TransferredOwnership("fixture", b"sensitive-token!", 1)
        self.assertNotIn("sensitive", repr(owner))
        with self.assertRaises(DriverTransportError):
            replace(bounds(), stop_timeout_s=.001)
        with self.assertRaises(DriverTransportError):
            TransferredOwnership("fixture", bytes(16), 1)


if __name__ == "__main__":
    unittest.main()
