"""Actual ROS DDS checks against a Kortex-disabled, explicit SimTransport build.

Never accepts a hardware-capable build. Joint stimuli check transport only:
upstream SimTransport keeps arm q/dq fixed and is not a physics simulator.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rammp_adl.motion.driver_feedback import DriverFeedbackBuffer, FeedbackBounds, RosDriverFeedback
from rammp_adl.motion.driver_transport import (
    RosJointTrajectoryTransport, TransferredOwnership, TransportBounds, make_ros_goal,
)
from rammp_adl.motion.driver_state import ARM_JOINT_NAMES
from rammp_adl.motion.rolling import JointState, JointTrajectory, TrajectoryPoint
from rammp_adl.motion.hardware_test import _wait


async def exercise(workspace, namespace):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rammp_common_interfaces.srv import AcquireControl
    from rammp_adl_interfaces.msg import DriverFeedback
    rclpy.init(args=[])
    node = rclpy.create_node("rammp_transport_conformance")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    manifest = json.loads((workspace/"source-manifest.json").read_text())
    buffer = DriverFeedbackBuffer(bounds=FeedbackBounds(.0001, .1, .25, .25, .00001, .1, .1),
        simulation=True, extension_build_id=manifest["extension_build_id"])
    feedback = RosDriverFeedback(node, buffer, feedback_topic=namespace+"/driver_feedback",
                                 heartbeat_topic=namespace+"/driver_heartbeat")
    raw = []
    subscription = node.create_subscription(DriverFeedback, namespace+"/driver_feedback", lambda msg: raw.append(msg), 1)
    claim = node.create_client(AcquireControl, namespace+"/acquire_control")
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    reports = {}
    reports["command_interfaces"] = {
        "action": "rammp_arm_interfaces/action/ExecuteJointTrajectory",
        "ownership": "rammp_common_interfaces/srv/AcquireControl",
        "pins": manifest["pins"],
        "upstream_interfaces_sha256": manifest["upstream_interfaces_sha256"],
        "extension_build_id": manifest["extension_build_id"],
    }
    configured = TransportBounds((.05,)*7, (.0005,)*7, (.001,)*7, .001, .001,
        .25, .25, .01, 2., .2, 1., .1, .5, 5.)
    owned = []
    async def acquire(name):
        await _wait(buffer.require_unowned, timeout_s=5.)
        await _wait(claim.service_is_ready, timeout_s=5.)
        request = AcquireControl.Request()
        request.owner_id = name
        pending = claim.call_async(request)
        await _wait(pending.done, timeout_s=2.)
        result = pending.result()
        if not result.accepted:
            raise RuntimeError("SimTransport ownership refused")
        owner = TransferredOwnership(name, bytes(result.token), result.generation)
        owned.append(owner)
        return owner
    def candidate(end=0., duration=.3):
        zero = JointState((0.,)*7, (0.,)*7, (0.,)*7)
        middle = JointState((.001,)*7, (0.,)*7, (0.,)*7)
        final = JointState((end,)*7, (0.,)*7, (0.,)*7)
        return JointTrajectory(ARM_JOINT_NAMES, (TrajectoryPoint(0.,zero), TrajectoryPoint(duration/2,middle),
            TrajectoryPoint(duration,final)), "EXPLICIT_SIM_TRANSPORT_FIXTURE_NOT_CUROBO")
    def gateway_for(owner):
        def measured(now):
            sample = buffer.read(now, ownership=owner, watchdog_timeout_s=2.)
            feedback.heartbeat(owner, now=now)
            return sample
        gateway = RosJointTrajectoryTransport(node, ownership=owner,
            admission_check=lambda permit,path: permit is path, state_check=measured, bounds=configured,
            action_name=namespace+"/execute_joint_trajectory", control_topic=namespace+"/control_status",
            release_service=namespace+"/release_control", stop_topic=namespace+"/estop")
        return gateway, measured
    try:
        first = await _wait(buffer.read, timeout_s=10.)
        await asyncio.sleep(.05)
        newer = buffer.read()
        assert newer.sequence > first.sequence
        reports["successful_exchange_feedback"] = True
        owner = await acquire("rammp-transport-test-success")
        gateway, measured = gateway_for(owner)
        await _wait(lambda: (measured(time.monotonic()),gateway.owns_control() and gateway.port.server_ready())[1], timeout_s=1.)
        # A second claim MUST be refused without invalidating the first token.
        stealing = AcquireControl.Request()
        stealing.owner_id = "must-not-steal"
        denied = claim.call_async(stealing)
        await _wait(denied.done, timeout_s=.5)
        assert denied.result().accepted is False
        reports["nonstealing_claim"] = True
        path = candidate()
        unauthorized = make_ros_goal(path, owner, configured)
        unauthorized.token = [0] * 16
        denied_goal = gateway.port.action.send_goal_async(unauthorized)
        await _wait(denied_goal.done, timeout_s=.5)
        assert denied_goal.result().accepted is False
        reports["unauthorized_trajectory_rejected"] = True
        receipt = await gateway.execute(path, permit=path)
        assert receipt.status == "succeeded" and receipt.measured_quiescent, asdict(receipt)
        reports["successful_roundtrip"] = asdict(receipt)
        await gateway.release()
        gateway.close()
        owner = await acquire("rammp-transport-test-cancel")
        gateway, measured = gateway_for(owner)
        await _wait(lambda: (measured(time.monotonic()),gateway.owns_control() and gateway.port.server_ready())[1], timeout_s=1.)
        path, cancel = candidate(duration=1.), asyncio.Event()
        asyncio.get_running_loop().call_later(.08, cancel.set)
        receipt = await gateway.execute(path, permit=path, cancel_event=cancel)
        assert receipt.status == "cancelled" and receipt.measured_quiescent, asdict(receipt)
        reports["cancel_and_measured_stop"] = asdict(receipt)
        await gateway.release()
        gateway.close()
        owner = await acquire("rammp-transport-test-false-success")
        gateway, measured = gateway_for(owner)
        await _wait(lambda: (measured(time.monotonic()),gateway.owns_control() and gateway.port.server_ready())[1], timeout_s=1.)
        path = candidate(end=.01)
        receipt = await gateway.execute(path, permit=path)
        assert receipt.status == "failed", asdict(receipt)
        reports["fixed_feedback_cannot_fake_goal"] = asdict(receipt)
        if receipt.release_permitted:
            await gateway.release()
            gateway.close()
        else:
            raise RuntimeError("Test could not establish measured stopped handback")
        owner = await acquire("rammp-transport-test-watchdog")
        await _wait(lambda: buffer.read(ownership=owner), timeout_s=1.)
        # No heartbeat on purpose. The C++ driver must stop without any Python
        # task issuing a cancellation or /estop command.
        await asyncio.sleep(2.3)
        assert raw and raw[-1].watchdog_latched and not raw[-1].owned
        reports["independent_watchdog_latched"] = True
        reports["hardware_started"] = False
        reports["validation_scope"] = "Actual generated ROS DDS and SimTransport only; arm q/dq fixed, no physics or physical stopping validation"
        return reports
    finally:
        executor.shutdown(timeout_sec=2.)
        spin.join(timeout=2.)
        feedback.close()
        node.destroy_subscription(subscription)
        node.destroy_client(claim)
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if os.environ.get("ROS_LOCALHOST_ONLY") != "1" or os.environ.get("ROS_DOMAIN_ID") != "91":
        raise RuntimeError("Use explicit isolated ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=91")
    cache = workspace/"build/kinova_lowlevel/CMakeCache.txt"
    if "KINOVA_ENABLE_KORTEX:BOOL=OFF" not in cache.read_text():
        raise RuntimeError("This reproducer refuses a hardware-capable build")
    executable = workspace/"install/kinova_gen3_ros2/lib/kinova_gen3_ros2/kinova_gen3_node"
    manifest = json.loads((workspace/"build-manifest.json").read_text())
    if (manifest.get("kortex_compiled") is not False or manifest.get("build_returncode") != 0
            or manifest.get("executable_sha256") != hashlib.sha256(executable.read_bytes()).hexdigest()):
        raise RuntimeError("Simulation executable must match its successful Kortex-disabled build manifest")
    model = workspace/"src/kinova-gen3-driver/models/gen3_7dof_2f85.urdf"
    output.mkdir(parents=True, exist_ok=False)
    namespace = "/rammp_transport_test_"+uuid.uuid4().hex
    command = [str(executable), "--sim", "--urdf", str(model), "--rt-priority", "0", "--ros-args",
               "-p", "arbitration_mode:=enforced", "-p", "heartbeat_timeout_s:=2.0",
               "-r", "__ns:="+namespace]
    for endpoint in ("execute_joint_trajectory", "control_status", "acquire_control", "release_control", "estop"):
        command.extend(["-r", "/"+endpoint+":="+namespace+"/"+endpoint])
    for endpoint in ("driver_feedback", "driver_heartbeat"):
        command.extend(["-r", "/rammp/"+endpoint+":="+namespace+"/"+endpoint])
    (output/"invocation.json").write_text(json.dumps({"command":command,"hardware_started":False}, indent=2)+"\n")
    with (output/"driver.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            report = asyncio.run(exercise(workspace, namespace))
            report["exclusive_test_namespace"] = namespace
            report["simulation_executable_sha256"] = manifest["executable_sha256"]
            report["reproducer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            (output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(report, indent=2))
        finally:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2.)


if __name__ == "__main__":
    main()
