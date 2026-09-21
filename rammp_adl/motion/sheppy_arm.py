"""One node's clients onto sheppy's arm module; nothing is sent unless armed.

Constructing SheppyArmClient creates subscriptions, action/service clients and
one publisher. It never starts a container, claims arbitration or commands
anything until ``arm()`` was called deliberately, and every send passes the
gates in sheppy_client first. Methods are coroutines that poll rclpy futures,
so the owning node must be spun by its own executor while they run.

TODO: confirm against driver - the sheppy arm node runs
arbitration_mode=disabled, so the goal/setpoint token stays zero. Under an
enforced mode a token from /acquire_control must be threaded through here.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time

from .sheppy_client import (
    DIFFERENCED, EXECUTE_ACTION, GRIPPER_SETPOINT_TOPIC, JOINT_STATE_TOPIC, JOINTS, PLANNER_NAMESPACE,
    START_PREPENDED, SUCCESSFUL, SheppyClientError, describe_surface, refusal, result_message, rmw_refusal,
    setpoint_from_knuckle, trajectory_from_planner, wrap_diff)


class SheppyArmClient:
    JOINT_STATE_STALE_S = .25
    STILL_VELOCITY_RAD_S = 2e-3
    GRIPPER_AT_TARGET_TOL = .02
    GRIPPER_MOVED_MIN = .01
    GRIPPER_SETTLE_TOL = .003
    GRIPPER_SETTLE_S = .15
    GRIPPER_NO_MOTION_S = 1.
    GRIPPER_RESEND_S = .1

    def __init__(self, node, *, environ=None, sender_id="rammp_adl", gripper_speed=1., gripper_force=.5,
                 planner_namespace=PLANNER_NAMESPACE, execute_action=EXECUTE_ACTION,
                 gripper_topic=GRIPPER_SETPOINT_TOPIC, joint_state_topic=JOINT_STATE_TOPIC):
        reason = rmw_refusal(os.environ if environ is None else environ)
        if reason:
            raise SheppyClientError(reason)
        from rclpy.action import ActionClient
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from rammp_arm_interfaces.action import ExecuteJointTrajectory
        from rammp_arm_interfaces.msg import GripperSetpoint
        from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
        from rammp_curobo_interfaces.srv import SetWorld
        if not isinstance(sender_id, str) or not sender_id:
            raise SheppyClientError("A sender identity is required")
        for value, label in ((gripper_speed, "gripper speed"), (gripper_force, "gripper force")):
            if isinstance(value, bool) or type(value) not in (int, float) or not 0. < value <= 1.:
                raise SheppyClientError(f"{label} must be in (0, 1]")
        self.node, self.sender_id = node, sender_id
        self.gripper_speed, self.gripper_force = float(gripper_speed), float(gripper_force)
        self._armed = False
        self._lock = threading.Lock()
        self._joints = None
        self._still_since = None
        self._active_goal = None
        self._gripper_target = None
        self.world_held = None
        self._Execute, self._Gripper, self._SetWorld = ExecuteJointTrajectory, GripperSetpoint, SetWorld
        self._PlanToPose, self._PlanToJoints = PlanToPose, PlanToJoints
        self._subscription = node.create_subscription(JointState, joint_state_topic, self._on_joint_state,
                                                      qos_profile_sensor_data)
        self._execute = ActionClient(node, ExecuteJointTrajectory, execute_action)
        self._plan_pose = ActionClient(node, PlanToPose, planner_namespace+"/plan_to_pose")
        self._plan_joints = ActionClient(node, PlanToJoints, planner_namespace+"/plan_to_joints")
        self._set_world = node.create_client(SetWorld, planner_namespace+"/set_world")
        self._gripper_pub = node.create_publisher(
            GripperSetpoint, gripper_topic, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self._gripper_timer = node.create_timer(self.GRIPPER_RESEND_S, self._publish_gripper)

    # -- state ---------------------------------------------------------------
    def arm(self):
        """Permit sends. A deliberate act, never a constructor default."""
        self._armed = True

    def disarm(self):
        """Withdraw permission to send; planning and state stay available."""
        self._armed = False

    @property
    def motion_enabled(self):
        return self._armed

    @property
    def in_flight(self):
        """A trajectory goal has been sent and its result has not come back."""
        return self._active_goal is not None

    def _on_joint_state(self, message):
        names = list(message.name)
        try:
            order = [names.index(name) for name in JOINTS]
        except ValueError:
            return
        knuckle = None
        for index, name in enumerate(names):
            if "knuckle" in name and index < len(message.position):
                knuckle = float(message.position[index])
                break
        q = tuple(float(message.position[i]) for i in order)
        effort = tuple(float(message.effort[i]) for i in order) if len(message.effort) >= 7 else None
        velocity = tuple(float(message.velocity[i]) for i in order) if len(message.velocity) >= 7 else None
        now = time.monotonic()
        moving = velocity is None or any(abs(v) > self.STILL_VELOCITY_RAD_S for v in velocity)
        with self._lock:
            previous = self._joints
            if moving:
                self._still_since = None
            elif self._still_since is None or previous is None or now-previous[4] > self.JOINT_STATE_STALE_S:
                self._still_since = now
            self._joints = (q, knuckle, effort, velocity, now)

    def still_since_s(self):
        """Monotonic time since which every fresh sample reported all joints still, else None.

        Unknown velocity is not stillness. A gap in samples restarts the window.
        """
        with self._lock:
            sample, since = self._joints, self._still_since
        if sample is None or since is None or time.monotonic()-sample[4] > self.JOINT_STATE_STALE_S:
            return None
        return since

    def live_joints(self, *, max_age_s=None):
        """The latest joint state if it is fresh, else None; never a frozen sample."""
        limit = self.JOINT_STATE_STALE_S if max_age_s is None else max_age_s
        with self._lock:
            sample = self._joints
        if sample is None or time.monotonic()-sample[4] > limit:
            return None
        return {"position_rad": sample[0], "knuckle_rad": sample[1], "effort_nm": sample[2],
                "velocity_rad_s": sample[3], "received_at_monotonic_s": sample[4]}

    async def stationary(self, *, duration_s=.5, velocity_threshold_rad_s=2e-3, position_span_rad=2e-3):
        """Fresh samples for duration_s with every joint still; False otherwise."""
        first, deadline = None, time.monotonic()+duration_s
        while time.monotonic() < deadline:
            live = self.live_joints()
            if live is None:
                return False
            if live["velocity_rad_s"] is not None and any(abs(v) > velocity_threshold_rad_s
                                                          for v in live["velocity_rad_s"]):
                return False
            first = live if first is None else first
            if any(abs(wrap_diff(a, b)) > position_span_rad
                   for a, b in zip(live["position_rad"], first["position_rad"])):
                return False
            await asyncio.sleep(.02)
        return first is not None

    async def _await(self, future, timeout_s, *, cancel_event=None):
        deadline = time.monotonic()+timeout_s
        while not future.done():
            if cancel_event is not None and cancel_event.is_set():
                return None
            if time.monotonic() > deadline:
                return None
            await asyncio.sleep(.02)
        return future.result()

    async def surface(self, *, timeout_s=5.):
        """Which of the expected servers answer; a readiness report, not a claim."""
        found = {}
        for name, client in (("execute_joint_trajectory", self._execute), ("plan_to_pose", self._plan_pose),
                             ("plan_to_joints", self._plan_joints)):
            found[name] = await asyncio.to_thread(client.wait_for_server, timeout_s)
        found["set_world"] = await asyncio.to_thread(self._set_world.wait_for_service, timeout_s)
        found["joint_states"] = self.live_joints() is not None
        return {**found, "armed": self._armed, **describe_surface()}

    # -- planner -------------------------------------------------------------
    async def set_world(self, path_or_name, *, timeout_s=10.):
        key = str(path_or_name)
        if key == self.world_held:
            return True, "held"
        if not self._set_world.wait_for_service(timeout_sec=2.):
            return False, "set_world service unavailable"
        response = await self._await(self._set_world.call_async(self._SetWorld.Request(world=key)), timeout_s)
        if response is None or not response.success:
            self.world_held = None
            return False, "set_world timed out" if response is None else response.message
        self.world_held = key
        return True, response.message

    async def _plan(self, client, goal, *, timeout_s, cancel_event=None):
        if not client.wait_for_server(timeout_sec=2.):
            raise SheppyClientError("planner not reachable; is sheppy's planner node up?")
        handle = await self._await(client.send_goal_async(goal), 10., cancel_event=cancel_event)
        if handle is None or not handle.accepted:
            raise SheppyClientError("planner did not accept the goal")
        wrapped = await self._await(handle.get_result_async(), timeout_s, cancel_event=cancel_event)
        if wrapped is None:
            raise SheppyClientError("planner result timed out or was cancelled")
        result = wrapped.result
        if not result.success:
            raise SheppyClientError("planner refused: "+(result.message or "no reason given"))
        points = [(point.time_from_start.sec+point.time_from_start.nanosec*1e-9,
                   tuple(point.positions), tuple(point.velocities) or None,
                   tuple(point.accelerations) or None) for point in result.trajectory.points]
        trajectory = trajectory_from_planner(
            tuple(result.trajectory.joint_names), points,
            provenance=f"rammp_curobo:{PLANNER_NAMESPACE}:planning_time={float(result.planning_time):.4f}")
        return trajectory, {"message": result.message, "planning_time_s": float(result.planning_time)}

    async def plan_to_pose(self, position_m, quaternion_xyzw, *, timeout_s=120., cancel_event=None):
        """A collision-free path from the live joints to a base-frame tool pose."""
        live = self.live_joints()
        if live is None:
            raise SheppyClientError("no fresh /joint_states; the planner needs the measured start")
        goal = self._PlanToPose.Goal()
        goal.target.position.x, goal.target.position.y, goal.target.position.z = (float(v) for v in position_m)
        (goal.target.orientation.x, goal.target.orientation.y, goal.target.orientation.z,
         goal.target.orientation.w) = (float(v) for v in quaternion_xyzw)
        goal.start_joints = [float(v) for v in live["position_rad"]]
        return await self._plan(self._plan_pose, goal, timeout_s=timeout_s, cancel_event=cancel_event)

    async def plan_to_joints(self, target_joints, *, timeout_s=120., cancel_event=None):
        live = self.live_joints()
        if live is None:
            raise SheppyClientError("no fresh /joint_states; the planner needs the measured start")
        goal = self._PlanToJoints.Goal(target_joints=[float(v) for v in target_joints])
        goal.start_joints = [float(v) for v in live["position_rad"]]
        return await self._plan(self._plan_joints, goal, timeout_s=timeout_s, cancel_event=cancel_event)

    # -- execution -----------------------------------------------------------
    def _ros_trajectory(self, trajectory):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory as RosTrajectory, JointTrajectoryPoint
        message = RosTrajectory()
        message.joint_names = list(trajectory.joint_names)
        # What goes to the driver is what the planner stamped: its first waypoint one
        # step ahead, no accelerations it never gave. The driver samples from t=0.
        points = trajectory.points[1:] if START_PREPENDED in trajectory.provenance else trajectory.points
        for point in points:
            row = JointTrajectoryPoint()
            row.positions = [float(v) for v in point.state.position]
            row.velocities = [float(v) for v in point.state.velocity]
            if DIFFERENCED not in trajectory.provenance:
                row.accelerations = [float(v) for v in point.state.acceleration]
            row.time_from_start = Duration(sec=int(point.time_s), nanosec=int(round((point.time_s % 1.)*1e9)))
            message.points.append(row)
        return message

    async def settle(self, *, timeout_s=6., window_s=.5):
        """Wait for the arm to come to rest after a motion; False if it never does.

        The driver reports success when its reference ends, a moment before
        the joints stop ringing. One look at that moment reads "moving" and
        fails a good motion; waiting, and failing closed on the timeout, does not.
        """
        deadline = time.monotonic()+timeout_s
        while time.monotonic() < deadline:
            if await self.stationary(duration_s=window_s):
                return True
            await asyncio.sleep(.05)
        return False

    async def execute(self, trajectory, *, cancel_event=None, timeout_s=240., goal_tolerance_rad=.02,
                      guard=None):
        """Send one gated trajectory; a receipt says what the driver and the arm did.

        guard, when given, is consulted every tick with the live state and the
        elapsed path time (driver progress times the path duration). A trip
        cancels the goal; the driver stops and holds, and the receipt carries
        the trip so the caller can name the failure correctly.
        """
        receipt = {"status": "refused", "message": "", "progress": 0., "error_code": None,
                   "sent": False, "cancel_requested": False, "final_position_rad": None,
                   "goal_gap_rad": None, "trip": None}
        if not self._armed:
            receipt["message"] = "motion disabled: the client was not armed"
            return receipt
        if self._active_goal is not None:
            receipt["message"] = "a trajectory is already in flight"
            return receipt
        live = self.live_joints()
        if live is None:
            receipt["message"] = "no fresh /joint_states; refusing to send a trajectory"
            return receipt
        why = refusal(trajectory, live["position_rad"])
        if why:
            receipt["message"] = "refused before sending: "+why
            return receipt
        if not self._execute.wait_for_server(timeout_sec=2.):
            receipt["message"] = "execute_joint_trajectory not available; is the arm node up?"
            return receipt
        goal = self._Execute.Goal()
        goal.trajectory = self._ros_trajectory(trajectory)
        goal.control_mode, goal.preemption = 0, 0
        goal.sender_id = self.sender_id

        def feedback(message):
            receipt["progress"] = float(message.feedback.fraction_complete)

        receipt["sent"] = True
        handle = await self._await(self._execute.send_goal_async(goal, feedback_callback=feedback), 10.)
        if handle is None or not handle.accepted:
            receipt.update(status="rejected", message="goal rejected by the driver")
            return receipt
        self._active_goal = handle
        wrapped = None
        try:
            result_future = handle.get_result_async()
            started = time.monotonic()
            while not result_future.done():
                if cancel_event is not None and cancel_event.is_set():
                    receipt["cancel_requested"] = True
                    await self._await(handle.cancel_goal_async(), 3.)
                    await self._await(result_future, 10.)
                    break
                if guard is not None:
                    guard.on_progress(receipt["progress"])
                    trip = guard.check(live=self.live_joints(), trajectory=trajectory,
                                       elapsed_s=receipt["progress"]*trajectory.duration_s,
                                       now=time.monotonic())
                    if trip is not None:
                        receipt["trip"] = trip
                        await self._await(handle.cancel_goal_async(), 3.)
                        await self._await(result_future, 10.)
                        receipt.update(status="guard_trip",
                                       message=f"guard tripped ({trip.get('kind')}); goal cancelled, driver holds")
                        return receipt
                if time.monotonic()-started > timeout_s:
                    await self._await(handle.cancel_goal_async(), 3.)
                    receipt.update(status="timeout", message=f"execution watchdog ({timeout_s} s)")
                    return receipt
                await asyncio.sleep(.05)
            wrapped = result_future.result() if result_future.done() else None
        finally:
            self._active_goal = None
        if wrapped is None:
            receipt.update(status="cancelled" if receipt["cancel_requested"] else "no_result",
                           message="no result from the driver")
            return receipt
        result = wrapped.result
        receipt["error_code"] = int(result.error_code)
        receipt["message"] = result_message(result.error_code, result.error_string)
        if receipt["cancel_requested"]:
            receipt["status"] = "cancelled"
            return receipt
        if result.error_code != SUCCESSFUL:
            receipt["status"] = "failed"
            return receipt
        settled = await self.settle()
        live = self.live_joints()
        receipt["final_position_rad"] = None if live is None else list(live["position_rad"])
        gap = None if live is None else max(abs(wrap_diff(a, b)) for a, b in
                                            zip(live["position_rad"], trajectory.points[-1].state.position))
        receipt["goal_gap_rad"] = gap
        if not settled or gap is None or gap > goal_tolerance_rad:
            detail = "not still" if not settled else ("unread" if gap is None else f"{gap:.4f} rad from the goal")
            receipt.update(status="goal_not_reached", message="driver reported success but the arm is "+detail)
            return receipt
        receipt["status"] = "succeeded"
        return receipt

    async def cancel(self):
        """Cancel an in-flight goal; the driver stops and holds its last reference."""
        handle = self._active_goal
        if handle is None:
            return False
        await self._await(handle.cancel_goal_async(), 3.)
        return True

    # -- gripper --------------------------------------------------------------
    def _publish_gripper(self):
        if self._gripper_target is None:
            return
        message = self._Gripper()
        message.position, message.speed, message.force = self._gripper_target, self.gripper_speed, self.gripper_force
        self._gripper_pub.publish(message)

    async def gripper(self, knuckle_rad, *, timeout_s=10.):
        """Command a knuckle angle and read completion off /joint_states.

        The driver's gripper reports no result, so completion is: at target;
        moved then still (a stall, closed on something); or never moved, which
        is a failure.
        """
        outcome = {"ok": False, "knuckle_rad": None, "stalled": False, "sent": False, "message": ""}
        if not self._armed:
            outcome["message"] = "gripper disabled: the client was not armed"
            return outcome
        setpoint = setpoint_from_knuckle(knuckle_rad)
        live = self.live_joints()
        if live is None or live["knuckle_rad"] is None:
            outcome["message"] = "no fresh knuckle reading; nothing could confirm the command"
            return outcome
        start, target = float(live["knuckle_rad"]), float(knuckle_rad)
        self._gripper_target = setpoint
        self._publish_gripper()
        outcome["sent"] = True
        t0 = time.monotonic()
        last_pos, still_since = start, None
        try:
            while time.monotonic()-t0 < timeout_s:
                await asyncio.sleep(.02)
                live = self.live_joints()
                now = time.monotonic()
                if live is None or live["knuckle_rad"] is None:
                    still_since = None
                    continue
                pos = float(live["knuckle_rad"])
                outcome["knuckle_rad"] = pos
                if abs(pos-target) <= self.GRIPPER_AT_TARGET_TOL:
                    outcome.update(ok=True, message="at target")
                    return outcome
                moved = abs(pos-start) >= self.GRIPPER_MOVED_MIN
                if abs(pos-last_pos) > self.GRIPPER_SETTLE_TOL:
                    still_since = None
                elif moved and still_since is None:
                    still_since = now
                last_pos = pos
                if moved and still_since is not None and now-still_since >= self.GRIPPER_SETTLE_S:
                    outcome.update(ok=True, stalled=True, message="moved then settled short of the target")
                    return outcome
                if not moved and now-t0 >= self.GRIPPER_NO_MOTION_S:
                    outcome["message"] = "the gripper never moved"
                    return outcome
            outcome["message"] = "gripper wait timed out"
            return outcome
        finally:
            self._gripper_target = None

    def close(self):
        for resource, action in ((self._gripper_timer, self.node.destroy_timer),
                                 (self._subscription, self.node.destroy_subscription),
                                 (self._gripper_pub, self.node.destroy_publisher),
                                 (self._set_world, self.node.destroy_client)):
            try:
                action(resource)
            except Exception:                            # noqa: BLE001 - teardown must finish
                pass
        for client in (self._execute, self._plan_pose, self._plan_joints):
            try:
                client.destroy()
            except Exception:                            # noqa: BLE001
                pass
