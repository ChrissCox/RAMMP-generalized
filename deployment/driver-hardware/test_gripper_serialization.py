#!/usr/bin/env python3
"""Generated ROS serialization only: no ROS node/context or hardware endpoint."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rammp_adl.motion.gripper import GripperCalibration, GripperBounds, calibrated_command, make_ros_gripper_setpoint
from rammp_adl.motion.driver_transport import TransferredOwnership
from rclpy.serialization import serialize_message, deserialize_message
from rammp_arm_interfaces.msg import GripperSetpoint
from rammp_adl_interfaces.msg import DriverFeedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Output must be new')
    calibration = GripperCalibration('simulation-2f85', 'synthetic-gap-table', 'simulation', (0., .5, 1.), (.08, .04, 0.), 1e-6)
    bounds = GripperBounds(1e-6, .001, .03, .03, .0001, .00001, .002, .001, .4, .5, .5, .002, .008, .1, .06, .25)
    ownership = TransferredOwnership('simulation-only-serialization', bytes(range(1, 17)), 2)
    command = calibrated_command(calibration, bounds, aperture_m=.04, speed_fraction=.125, current_ceiling_fraction=.25)
    wire = deserialize_message(serialize_message(make_ros_gripper_setpoint(command, ownership)), GripperSetpoint)
    assert (wire.position, wire.speed, wire.force) == (.5, .125, .25)
    assert bytes(wire.token) == ownership.token
    evidence = DriverFeedback()
    evidence.gripper_halt_supported = evidence.gripper_halt_active = True
    evidence.exchange_sequence = evidence.gripper_halt_exchange_sequence = 314
    evidence.gripper_halt_generation = 1
    evidence.gripper_halt_position_normalized = .5
    evidence.gripper_current_a = .012
    decoded = deserialize_message(serialize_message(evidence), DriverFeedback)
    assert decoded.gripper_halt_active and decoded.gripper_halt_exchange_sequence == 314
    assert decoded.gripper_halt_generation == 1 and decoded.gripper_halt_position_normalized == .5
    assert decoded.gripper_current_a == .012
    report = {'passed': True, 'command': asdict(command), 'token_roundtrip_exact': True,
        'halt_ack_roundtrip_exact': True, 'hardware_started': False, 'ros_nodes_started': False,
        'scope': 'Actual generated Humble IDL serialization; no DDS or physical gripper execution'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
