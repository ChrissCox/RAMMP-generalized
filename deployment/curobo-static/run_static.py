"""Generate explicit bare-arm fixture configuration and run actual cuRobo.

Only simulation model data is read. No ROS imports, camera images, robot state,
execution client, or controller fallback exists in this commissioning harness.
The production adapter and independent MuJoCo verifier retain their gates.
"""
import argparse
import asyncio
import copy
import json
import logging
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from rammp_adl.contracts import Catalog, digest
from rammp_adl.motion.curobo import RAMMP_COMMIT, RammpCuroboAdapter
from rammp_adl.motion.integration import (
    MujocoCandidateVerifier, _plan_validate_replay, collision_world_from_mujoco,
    file_digest, load_contract, model_assets_digest,
)
from rammp_adl.motion.rolling import JointState
from rammp_adl.simulation import MujocoReplay

if not __debug__:
    raise RuntimeError("Model verification requires assertions enabled; Python -O is unsupported")


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rejection_report(output, start, error, *, physics_time_s=0.):
    """Record the failed gate; never turn a generated candidate into execution."""
    candidate_path = output / "curobo-trajectory.json"
    report = {"mode": "curobo_static_candidate_rejected", "status": "rejected", "reason": str(error),
              "curobo_planned": candidate_path.is_file(), "simulation_only": True,
              "independent_validation_passed": False, "physics_replayed": physics_time_s > 0,
              "physics_time_s": physics_time_s,
              "hardware_commands": False, "hardware_validated": False,
              "gripper_simulated": False, "exit_code": 2}
    if candidate_path.is_file():
        path = json.loads(candidate_path.read_text())
        first, last = path["points"][0], path["points"][-1]
        report.update(trajectory_digest=path["digest"], point_count=len(path["points"]),
                      trajectory_duration_s=last["time_s"], planning_wall_s=path["planning_wall_s"])
        report["boundary_diagnostics"] = {
            "start_position_error_rad": float(np.max(np.abs(np.asarray(first["position"]) - start.position))),
            "start_maximum_velocity_rad_s": max(map(abs, first["velocity"])),
            "start_maximum_acceleration_rad_s2": max(map(abs, first["acceleration"])),
            "end_maximum_velocity_rad_s": max(map(abs, last["velocity"])),
            "end_maximum_acceleration_rad_s2": max(map(abs, last["acceleration"])),
            "required_start_position_rad": 1e-5, "required_start_velocity_rad_s": 1e-5,
            "required_start_acceleration_rad_s2": 1e-4,
            "required_end_velocity_rad_s": 1e-5, "required_end_acceleration_rad_s2": 1e-5,
        }
    return report


def prepare(output, wrapper):
    import curobo
    content = Path(curobo.__file__).resolve().parent / "content"
    stock_path = content / "configs/robot/kinova_gen3.yml"
    spheres_path = content / "configs/robot/spheres/kinova_gen3.yml"
    urdf = content / "assets/robot/kinova/kinova_gen3_7dof.urdf"
    stock = yaml.safe_load(stock_path.read_text())
    robot = copy.deepcopy(stock)
    kin = robot["robot_cfg"]["kinematics"]
    # These are the eight bodies in the existing MuJoCo bare-arm model, already
    # listed in the source config's mesh_link_names. No gripper is simulated.
    links = list(kin["mesh_link_names"])
    assert links == ["base_link", "shoulder_link", "half_arm_1_link", "half_arm_2_link", "forearm_link",
                     "spherical_wrist_1_link", "spherical_wrist_2_link", "bracelet_link"]
    kin["collision_link_names"] = links
    spheres = yaml.safe_load(spheres_path.read_text())["collision_spheres"]
    kin["collision_spheres"] = {name: copy.deepcopy(spheres[name]) for name in links}
    kin["self_collision_buffer"] = {name: kin["self_collision_buffer"][name] for name in links}
    kin["self_collision_ignore"] = {name: [other for other in kin["self_collision_ignore"].get(name, []) if other in links]
                                    for name in links}
    kin["ee_link"] = "end_effector_link"
    kin["urdf_path"] = str(urdf)
    kin["asset_root_path"] = str(content / "assets/robot/kinova")
    # Verify the selected frame is the *existing* fixture site, not a guessed
    # physical gripper/tool calibration. Rotation Rx(pi) is expressed in URDF.
    joint = next(node for node in ET.parse(urdf).getroot().findall("joint") if node.attrib["name"] == "end_effector")
    assert joint.find("parent").attrib["link"] == "bracelet_link"
    assert joint.find("child").attrib["link"] == "end_effector_link"
    assert np.allclose([float(v) for v in joint.find("origin").attrib["xyz"].split()], [0., 0., -.061525], atol=1e-12)
    assert np.allclose([float(v) for v in joint.find("origin").attrib["rpy"].split()], [np.pi, 0., 0.], atol=1e-12)
    simulation_root = Catalog().root / "simulation"
    sim = MujocoReplay(simulation_root / "rolling_scene.xml")
    start = sim.home()
    kin["cspace"]["retract_config"] = [float(value) for value in start.position]
    # Local static simulation settings, not commissioned hardware profiles.
    kin["cspace"]["max_acceleration"] = 4.0
    kin["cspace"]["max_jerk"] = 100.0
    robot_file = output / "bare-gen3.yaml"
    robot_file.write_text(yaml.safe_dump(robot, sort_keys=False), encoding="utf-8")
    contract = {
        "simulation_only": True, "contract_id": "gen3-bare-static-v078-2026-09-11",
        "planner_source_commit": RAMMP_COMMIT, "planner_config_digest": "",
        "mujoco_assets_digest": model_assets_digest(simulation_root), "planner_urdf_digest": file_digest(urdf),
        "planner_robot_config_digest": digest(robot["robot_cfg"]), "base_frame": "base_link", "tool_site": "pinch_site",
        "joint_names": list(sim.joint_names), "workspace_lower_m": [-1.5, -1.5, 0.], "workspace_upper_m": [1.5, 1.5, 1.5],
        "fk_position_tolerance_m": .001, "fk_rotation_tolerance_rad": .001,
        "joint_velocity_rad_s": [1.2] * 7, "joint_acceleration_rad_s2": [4.] * 7,
        "validation_sample_dt_s": .005, "maximum_tracking_error_rad": .02,
        "goal_position_tolerance_m": .01, "goal_rotation_tolerance_rad": .02,
    }
    world = collision_world_from_mujoco(sim, contract)
    world_file = output / "world.yaml"
    world_file.write_text(yaml.safe_dump({"base_frame": "base_link", "obstacles": world, "objects": [], "targets": []}), encoding="utf-8")
    planner = yaml.safe_load((wrapper / "core/rammp_curobo/configs/gen3.yaml").read_text())
    planner.update(robot=str(robot_file), world=str(world_file))
    planner["planner"].update(world_padding=0.0, no_pad_names=[], limit_clamp_rad=0.0,
                              joint_space_method="js", warmup=True)
    planner_file = output / "planner.yaml"
    planner_file.write_text(yaml.safe_dump(planner, sort_keys=False), encoding="utf-8")
    contract["planner_config_digest"] = file_digest(planner_file)
    contract_file = output / "model-contract.json"
    save(contract_file, contract)
    load_contract(contract_file)
    save(output / "model-provenance.json", {
        "simulation_only": True, "hardware_commands": False, "gripper_simulated": False,
        "source_stock_robot": str(stock_path), "source_stock_robot_digest": file_digest(stock_path),
        "source_stock_spheres": str(spheres_path), "source_stock_spheres_digest": file_digest(spheres_path),
        "source_urdf": str(urdf), "source_urdf_digest": file_digest(urdf),
        "selected_ee_link": "end_effector_link", "mujoco_tool_site": "pinch_site",
        "frame_derivation": "Both frames use bracelet xyz=(0,0,-0.061525) and Rx(pi); no physical tool offset.",
        "collision_model": "Unchanged upstream v0.7.8 arm spheres, eight arm links only; separate MuJoCo mesh checks.",
        "limits_scope": "Local static simulation settings; no swept/stopping or hardware acceptance.",
    })
    return sim, start, planner_file, contract


class RecordingAdapter:
    def __init__(self, adapter, output):
        self.adapter, self.output, self.planner = adapter, output, adapter.planner

    async def plan_pose(self, **request):
        began = time.monotonic()
        path = await self.adapter.plan_pose(**request)
        if hasattr(self.planner, "boundary_debug"):
            save(self.output / "curobo-optimized.json", self.planner.boundary_debug)
        save(self.output / "curobo-trajectory.json", {
            "joint_names": list(path.joint_names), "provenance": path.provenance, "digest": path.digest,
            "planning_wall_s": time.monotonic() - began,
            "points": [{"time_s": p.time_s, "position": p.state.position,
                        "velocity": p.state.velocity, "acceleration": p.state.acceleration} for p in path.points],
        })
        return path


async def main(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    sim, start, planner_file, contract = prepare(output, args.wrapper_source.resolve())
    adapter = await RammpCuroboAdapter.load(source_root=args.wrapper_source, planner_config=planner_file)
    load_s = time.monotonic() - began
    verifier = MujocoCandidateVerifier(sim, adapter.planner, contract)
    # FK comparisons are independent static model checks; these are not paths
    # or IK solutions and are never executed as trajectories.
    verifier.check_state(start)
    for joint in range(7):
        for offset in (-.04, .04):
            position = list(start.position)
            position[joint] += offset
            verifier.check_state(JointState(tuple(position), (0.,) * 7, (0.,) * 7))
    save(output / "fk-probe.json", {"states_checked": verifier.states_checked,
                                  "maximum_position_error_m": verifier.maximum_fk_position_error_m,
                                  "maximum_rotation_error_rad": verifier.maximum_fk_rotation_error_rad})
    position, quaternion = verifier.pose(start)
    position[0] += .03  # Explicit 3 cm free-space simulation target, no arm IK here.
    goal = {"position_m": position.tolist(), "quaternion_xyzw": quaternion.tolist()}
    save(output / "goal.json", goal)
    try:
        result = await _plan_validate_replay(RecordingAdapter(adapter, output), sim, contract, goal, start)
    except Exception as error:
        report = rejection_report(output, start, error, physics_time_s=float(sim.data.time))
        report.update(planner_initialization_wall_s=load_s, total_wall_s=time.monotonic() - began)
        save(output / "result.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return 2
    result.update(planner_initialization_wall_s=load_s, total_wall_s=time.monotonic() - began)
    save(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(main(args)))
