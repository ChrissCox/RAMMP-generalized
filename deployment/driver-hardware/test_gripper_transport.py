#!/usr/bin/env python3
"""Real DDS + pinned SimTransport gripper checks; refuses Kortex-capable builds."""
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
from rammp_adl.motion.driver_transport import DriverTransportError, TransferredOwnership
from rammp_adl.motion.gripper import (GripperCalibration, GripperBounds, GripperFeedbackBuffer,
    GripperTransport, RosGripperPort, calibrated_command)


class IsolatedSimGripperPort(RosGripperPort):
    """Test-only port created after main verifies the Kortex-disabled manifest,
    localhost domain 93 and fresh namespace. Never a production port fallback.
    """
    hardware_commands = False


async def wait_value(read, timeout=5.):
    until = time.monotonic()+timeout
    while time.monotonic() < until:
        try:
            value = read()
            if value: return value
        except DriverTransportError: pass
        await asyncio.sleep(.005)
    raise RuntimeError('Simulation endpoint/evidence deadline exceeded')


async def exercise(workspace, namespace):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rammp_adl_interfaces.msg import DriverFeedback
    from rammp_common_interfaces.srv import AcquireControl
    from rammp_common_interfaces.msg import ControlStatus
    manifest = json.loads((workspace/'build-manifest.json').read_text())
    build = manifest['extension_build_id']
    calibration = GripperCalibration('simulated-2f85', 'synthetic-aperture-table', 'simulation',
                                      (0., .5, 1.), (.08, .04, 0.), .000001)
    bounds = GripperBounds(.0001, .02, .2, .2, .0001, .00001, .002, .001,
                           .8, .5, .5, .005, .06, 2., 1., 2.)
    grip = GripperFeedbackBuffer(calibration=calibration, bounds=bounds, extension_build_id=build, simulation=True)
    arm = DriverFeedbackBuffer(bounds=FeedbackBounds(.0001,.02,.2,.2,.000001,.01,.01),
        simulation=True, extension_build_id=build)
    rclpy.init()
    node = rclpy.create_node('rammp_gripper_dds_test_'+uuid.uuid4().hex[:8])
    feedback = RosDriverFeedback(node, arm, feedback_topic=namespace+'/driver_feedback',
                                 heartbeat_topic=namespace+'/driver_heartbeat')
    sub = node.create_subscription(DriverFeedback, namespace+'/driver_feedback', grip.ingest,
        QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE))
    status = []
    status_sub = node.create_subscription(ControlStatus, namespace+'/control_status', lambda msg:status.append(msg),
        QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    claim = node.create_client(AcquireControl, namespace+'/acquire_control')
    port = IsolatedSimGripperPort(node, command_topic=namespace+'/setpoint/gripper', stop_topic=namespace+'/estop')
    executor = MultiThreadedExecutor(num_threads=2); executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True); spinner.start()
    pulse = None
    try:
        await wait_value(grip.read)
        await wait_value(claim.service_is_ready)
        request = AcquireControl.Request(); request.owner_id = 'simulation-gripper-commissioning'
        pending = claim.call_async(request); await wait_value(pending.done)
        granted = pending.result(); assert granted.accepted
        owner = TransferredOwnership(request.owner_id, bytes(granted.token), granted.generation)
        await wait_value(lambda:grip.read(ownership=owner))
        def owns():
            return bool(status and status[-1].owned and not status[-1].estopped and status[-1].arbitration_enabled
                        and status[-1].owner_id == owner.owner_id and status[-1].generation == owner.generation)
        async def heartbeat():
            while True:
                try: feedback.heartbeat(owner)
                except DriverTransportError: pass
                await asyncio.sleep(.01)
        pulse = asyncio.create_task(heartbeat())
        gateway = GripperTransport(port=port, ownership=owner, feedback=grip,
                                  admission_check=lambda permit, cmd:permit == 'explicit-simulation-permit', owns_control=owns)
        await wait_value(lambda:owns() and port.ready())
        command = calibrated_command(calibration,bounds,aperture_m=.04,speed_fraction=.125,current_ceiling_fraction=.25)
        first = await gateway.execute(command,'explicit-simulation-permit')
        assert first.status == 'succeeded' and first.measured_quiescent, asdict(first)
        cancel = asyncio.Event()
        next_command = calibrated_command(calibration,bounds,aperture_m=.064,speed_fraction=.125,current_ceiling_fraction=.25)
        asyncio.get_running_loop().call_later(.03,cancel.set)
        stopped = await gateway.execute(next_command,'explicit-simulation-permit',cancel_event=cancel)
        assert stopped.status == 'cancelled' and stopped.halt_acknowledged and stopped.measured_quiescent, asdict(stopped)
        last = grip.read()
        await asyncio.sleep(.07)
        later = grip.read()
        assert later.sequence > last.sequence and later.halt_active and not later.owned
        assert abs(later.normalized_position-last.halt_position_normalized) < 1e-6
        # A stopped commanded gripper cannot silently restore the old target in a new owner session.
        denied = claim.call_async(request); await wait_value(denied.done)
        assert not denied.result().accepted
        return {'success':asdict(first),'cancel':asdict(stopped),'fresh_halt_exchange_confirmed':True,
                'stale_target_not_resumed':True,'new_grant_rejected_after_gripper_halt':True,
                'hardware_started':False,'scope':'Actual ROS DDS + pinned synthetic SimTransport gripper model; no physical stopping/contact/retention validation'}
    finally:
        if pulse:
            pulse.cancel()
            try: await pulse
            except asyncio.CancelledError: pass
        executor.shutdown(timeout_sec=2.); spinner.join(timeout=2.)
        port.close(); feedback.close()
        node.destroy_subscription(sub); node.destroy_subscription(status_sub); node.destroy_client(claim)
        node.destroy_node(); rclpy.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); workspace=args.workspace.resolve(); output=args.output.resolve()
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '93':
        raise RuntimeError('Requires isolated ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=93')
    if 'KINOVA_ENABLE_KORTEX:BOOL=OFF' not in (workspace/'build/kinova_lowlevel/CMakeCache.txt').read_text():
        raise RuntimeError('Refusing hardware-capable executable')
    binary=workspace/'install/kinova_gen3_ros2/lib/kinova_gen3_ros2/kinova_gen3_node'
    manifest=json.loads((workspace/'build-manifest.json').read_text())
    if (manifest.get('kortex_compiled') is not False or manifest.get('build_returncode') != 0
            or manifest.get('executable_sha256') != hashlib.sha256(binary.read_bytes()).hexdigest()):
        raise RuntimeError('Simulation executable must match its successful Kortex-disabled build manifest')
    output.mkdir(parents=True,exist_ok=False)
    namespace='/rammp_gripper_test_'+uuid.uuid4().hex
    command=[str(binary),'--sim','--urdf',str(workspace/'src/kinova-gen3-driver/models/gen3_7dof_2f85.urdf'),
        '--rt-priority','0','--ros-args','-p','arbitration_mode:=enforced','-p','heartbeat_timeout_s:=2.0','-r','__ns:='+namespace]
    for endpoint in ('execute_joint_trajectory','control_status','acquire_control','release_control','estop','setpoint/gripper'):
        command.extend(['-r','/'+endpoint+':='+namespace+'/'+endpoint])
    for endpoint in ('driver_feedback','driver_heartbeat'):
        command.extend(['-r','/rammp/'+endpoint+':='+namespace+'/'+endpoint])
    (output/'invocation.json').write_text(json.dumps({'command':command,'hardware_started':False},indent=2)+'\n')
    with (output/'driver.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            report=asyncio.run(exercise(workspace,namespace))
            report['namespace']=namespace
            report['extension_build_id']=manifest['extension_build_id']
            (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
        finally:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2.)


if __name__ == '__main__':main()
