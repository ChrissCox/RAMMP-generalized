#!/usr/bin/env python3
"""Exercise actual ROS DDS against an already-running instrumented SIM driver.

This intentionally rejects physical feedback before acquiring control. The arm
paths below are synthetic joint fixtures solely for transport conformance, not
cuRobo plans or hardware qualification. The driver process is started separately
with a source-reviewed Kortex-disabled build and --sim.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rammp_adl.motion.driver_state import ARM_JOINT_NAMES
from rammp_adl.motion.driver_transport import (
    RosJointTrajectoryTransport, TransferredOwnership, TransportBounds,
    VerifiedMotionState, canonical_ros_trajectory, make_ros_goal,
)
from rammp_adl.motion.rolling import JointState, JointTrajectory, TrajectoryPoint


async def bounded(future, seconds=3.):
    end = time.monotonic()+seconds
    while not future.done():
        if time.monotonic() > end:
            raise RuntimeError("ROS test response deadline exceeded")
        await asyncio.sleep(.005)
    return future.result()


async def check(args):
    if os.environ.get("ROS_LOCALHOST_ONLY") != "1" or os.environ.get("ROS_DOMAIN_ID") != "91":
        raise RuntimeError("this SIM-only check requires localhost and isolated ROS_DOMAIN_ID=91")
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.serialization import serialize_message, deserialize_message
    from kinova_gen3_interfaces.action import ExecuteJointTrajectory
    from kinova_gen3_interfaces.srv import AcquireControl
    from rammp_adl_interfaces.msg import DriverFeedback, DriverHeartbeat

    manifest = json.loads(Path(args.source_manifest).read_text())
    expected_build = manifest["extension_build_id"]
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    context = rclpy.context.Context()
    rclpy.init(context=context)
    node = rclpy.create_node("rammp_transport_sim_check", context=context)
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    latest, lock = {}, threading.Lock()
    def feedback(message):
        with lock:
            if not latest or latest["msg"].exchange_sequence != message.exchange_sequence:
                latest.update(msg=message, received=time.monotonic())
    subscription = node.create_subscription(DriverFeedback, "/rammp/driver_feedback", feedback,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
    heartbeat_pub = node.create_publisher(DriverHeartbeat, "/rammp/driver_heartbeat",
        QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
    acquire = node.create_client(AcquireControl, "/acquire_control")
    thread.start()
    transports, receipts, mapped = [], [], False
    active_grant = None
    heartbeat_sequence = 0

    def snapshot():
        with lock:
            if not latest:
                raise RuntimeError("no instrumented simulation feedback")
            message, received = latest["msg"], latest["received"]
        now = time.monotonic()
        if (message.simulation is not True or message.host_boot_id != boot_id
                or message.extension_build_id != expected_build or message.heartbeat_required is not True
                or message.watchdog_latched or message.fault
                or tuple(message.joint_names) != ARM_JOINT_NAMES
                or not 0 <= now-received < .2
                or not 0 <= now-message.exchange_start_monotonic_ns/1e9 < .2):
            raise RuntimeError("SIM-only source, build, watchdog or freshness check rejected feedback")
        return message, received

    def state(now):
        nonlocal heartbeat_sequence
        message, received = snapshot()
        if active_grant is not None:
            heartbeat_sequence += 1
            beat = DriverHeartbeat()
            beat.driver_session_id, beat.host_boot_id = message.driver_session_id, boot_id
            beat.owner_id, beat.ownership_generation = active_grant.owner_id, active_grant.generation
            beat.token, beat.sequence = list(active_grant.token), heartbeat_sequence
            beat.sent_monotonic_ns = time.monotonic_ns()
            heartbeat_pub.publish(beat)
        return VerifiedMotionState(tuple(message.position_rad), tuple(message.velocity_rad_s),
            message.exchange_start_monotonic_ns/1e9, received, message.exchange_sequence,
            message.driver_session_id)

    configured = TransportBounds(path_position_rad=(.15,)*7, goal_position_rad=(.008,)*7,
        start_position_rad=(.005,)*7, stationary_velocity_rad_s=.01,
        stationary_position_span_rad=.001, state_max_age_s=.2, receipt_max_age_s=.2,
        poll_period_s=.005, send_timeout_s=2., cancel_timeout_s=.3, stop_timeout_s=1.,
        settle_duration_s=.1, result_slack_s=.5, maximum_trajectory_s=5.)
    try:
        deadline = time.monotonic()+8.
        while time.monotonic() < deadline:
            try:
                snapshot()
                if acquire.service_is_ready():
                    break
            except RuntimeError:
                pass
            await asyncio.sleep(.05)
        snapshot()
        for cancel in (False, True):
            message, _ = snapshot()
            if message.owned:
                raise RuntimeError("simulation already has an owner; test refuses to seize it")
            request = AcquireControl.Request()
            request.owner_id = "rammp-sim-check-"+uuid.uuid4().hex[:8]
            response = await bounded(acquire.call_async(request))
            if not response.accepted:
                raise RuntimeError("simulation refused explicit control grant")
            active_grant = TransferredOwnership(request.owner_id, bytes(response.token), response.generation)
            permit = object()
            admitted = []
            transport = RosJointTrajectoryTransport(node, ownership=active_grant,
                admission_check=lambda p,t: p is permit and t.digest == admitted[0], state_check=state,
                bounds=configured)
            transports.append(transport)
            deadline = time.monotonic()+2.
            while not transport.owns_control():
                state(time.monotonic())
                if time.monotonic() > deadline:
                    raise RuntimeError("transferred grant not independently observed")
                await asyncio.sleep(.005)
            sample = state(time.monotonic())
            goal = list(sample.position_rad)
            goal[5] += .03
            trajectory = canonical_ros_trajectory(JointTrajectory(ARM_JOINT_NAMES, (
                TrajectoryPoint(0., JointState(sample.position_rad, (0.,)*7, (0.,)*7)),
                TrajectoryPoint(.6, JointState(tuple(goal), (0.,)*7, (0.,)*7)),
                TrajectoryPoint(1.2, JointState(sample.position_rad, (0.,)*7, (0.,)*7))),
                "synthetic joint fixture for isolated SimTransport DDS conformance; no cuRobo"))
            admitted.append(trajectory.digest)
            wire = make_ros_goal(trajectory, active_grant, configured)
            restored = deserialize_message(serialize_message(wire), ExecuteJointTrajectory.Goal)
            for actual, expected in zip(restored.trajectory.points, trajectory.points):
                assert tuple(actual.positions) == expected.state.position
                assert tuple(actual.velocities) == expected.state.velocity
                assert tuple(actual.accelerations) == expected.state.acceleration
            mapped = True
            cancel_event = asyncio.Event()
            if cancel:
                asyncio.get_running_loop().call_later(.2, cancel_event.set)
            receipt = await transport.execute(trajectory, permit=permit, cancel_event=cancel_event)
            receipts.append(asdict(receipt))
            expected = "cancelled" if cancel else "succeeded"
            if receipt.status != expected or not receipt.release_permitted:
                raise RuntimeError("unexpected transport receipt: "+json.dumps(asdict(receipt)))
            await transport.release()
            active_grant = None
            transport.close()
            transports.remove(transport)
            deadline = time.monotonic()+2.
            while snapshot()[0].owned:
                if time.monotonic() > deadline:
                    raise RuntimeError("simulation ownership release was not observed")
                await asyncio.sleep(.01)
        report = {"scope": "actual ROS DDS to instrumented upstream SimTransport; synthetic joint fixtures",
                  "hardware_connected": False, "curobo_planned": False, "extension_build_id": expected_build,
                  "actual_generated_idl_roundtrip": mapped, "receipts": receipts,
                  "limitations": ["Upstream SimTransport holds arm q/dq fixed; no arm dynamics or tracking",
                                  "Roundtrip fixture ends at initial pose; success is transport conformance only"]}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
    finally:
        for transport in transports:
            transport._engage_stop("SIM-only test cleanup after failure")
        executor.shutdown(timeout_sec=2.)
        thread.join(timeout=2.)
        node.destroy_node()
        context.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--report", required=True)
    asyncio.run(check(parser.parse_args()))
