"""Explicit cuRobo -> independently checked MuJoCo static simulation runner.

This is a commissioning seam, not the task executor's reactive capability. Every
arm path comes from the pinned cuRobo adapter. A local compatibility contract and
actual FK comparisons are mandatory; no guessed gripper/tool transform is used.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..contracts import Catalog, ContractError, digest, strict_loads
from ..simulation import MujocoReplay
from .curobo import RAMMP_COMMIT, RammpCuroboAdapter
from .guards import quaternion_distance
from .rolling import BoundaryTolerance, JointLimits, JointState, MotionError, TrajectoryValidator


def file_digest(path):
    return "sha256:"+hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_assets_digest(simulation_root):
    root = Path(simulation_root).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file() and
                   (path.suffix.lower() in {".xml", ".stl", ".obj", ".dae"} or path.name == "LICENSE"))
    if not files:
        raise ContractError("No simulation model assets found")
    return digest({path.relative_to(root).as_posix(): file_digest(path) for path in files})


def load_contract(path):
    contract = strict_loads(Path(path).read_bytes())
    required = {"simulation_only", "contract_id", "planner_source_commit", "planner_config_digest", "mujoco_assets_digest", "planner_urdf_digest",
                "planner_robot_config_digest", "base_frame", "tool_site", "joint_names", "workspace_lower_m", "workspace_upper_m",
                "fk_position_tolerance_m", "fk_rotation_tolerance_rad", "joint_velocity_rad_s", "joint_acceleration_rad_s2",
                "validation_sample_dt_s", "maximum_tracking_error_rad", "goal_position_tolerance_m", "goal_rotation_tolerance_rad"}
    if not isinstance(contract, dict) or set(contract) != required or contract["simulation_only"] is not True:
        raise ContractError("Static simulation requires the closed simulation-only model contract")
    if contract["planner_source_commit"] != RAMMP_COMMIT or contract["base_frame"] != "base_link" or contract["tool_site"] != "pinch_site":
        raise ContractError("Contract does not identify the supported pinned bare-Gen3 planning frames")
    if not isinstance(contract["contract_id"], str) or not contract["contract_id"]:
        raise ContractError("A local model compatibility contract ID is required")
    for field in ("planner_config_digest", "mujoco_assets_digest", "planner_robot_config_digest", "planner_urdf_digest"):
        value = contract[field]
        if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:") or any(c not in "0123456789abcdef" for c in value[7:]):
            raise ContractError("Invalid compatibility digest: "+field)
    for field, length in (("workspace_lower_m", 3), ("workspace_upper_m", 3), ("joint_velocity_rad_s", 7), ("joint_acceleration_rad_s2", 7)):
        value = contract[field]
        if not isinstance(value, list) or len(value) != length or not all(type(v) in (int, float) and math.isfinite(v) for v in value):
            raise ContractError("Invalid model-contract numeric vector: "+field)
    if any(lo >= hi for lo, hi in zip(contract["workspace_lower_m"], contract["workspace_upper_m"])):
        raise ContractError("Invalid bounded simulation workspace")
    for field in ("fk_position_tolerance_m", "fk_rotation_tolerance_rad", "validation_sample_dt_s", "maximum_tracking_error_rad", "goal_position_tolerance_m", "goal_rotation_tolerance_rad"):
        if type(contract[field]) not in (int, float) or not math.isfinite(contract[field]) or contract[field] <= 0:
            raise ContractError("Invalid model-contract tolerance: "+field)
    if contract["validation_sample_dt_s"] > .02 or contract["fk_position_tolerance_m"] > .005 or contract["fk_rotation_tolerance_rad"] > .02:
        raise ContractError("Model compatibility checks cannot be relaxed beyond this simulation runner's bounds")
    if min(*contract["joint_velocity_rad_s"], *contract["joint_acceleration_rad_s2"]) <= 0:
        raise ContractError("Joint test limits must be positive")
    if contract["joint_names"] != [f"joint_{i}" for i in range(1, 8)]:
        raise ContractError("Joint names/order differ from the supported model")
    return contract


class MujocoCandidateVerifier:
    """Independent sampled kinematic checks of the exact final interpolation.

    This is explicitly not a proof of continuous swept/stopping safety. It never
    enables runtime hardware, contact or reactive-motion capabilities.
    """
    def __init__(self, simulation, planner, contract):
        self.simulation, self.planner, self.contract = simulation, planner, contract
        self.mujoco, self.model = simulation.mujoco, simulation.model
        self.data = self.mujoco.MjData(self.model)
        self.tool_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, contract["tool_site"])
        base_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, contract["base_frame"])
        if self.tool_id < 0 or base_id < 0 or self.model.body_parentid[base_id] != 0 or self.model.body_jntnum[base_id] != 0:
            raise ContractError("Simulation model is not the admitted fixed-base/tool configuration")
        if not np.allclose(self.model.body_pos[base_id], 0., atol=1e-9) or not np.allclose(self.model.body_quat[base_id], (1., 0., 0., 0.), atol=1e-9):
            raise ContractError("A non-identity base transform requires a verified adapter")
        if tuple(planner.joint_names) != simulation.joint_names or tuple(contract["joint_names"]) != simulation.joint_names:
            raise ContractError("cuRobo and MuJoCo joint order/model identity differ")
        # This private metadata is read only at the pinned source revision, where
        # _robot_cfg is the dictionary passed to MotionGenConfig. No API guess.
        if not hasattr(planner, "_robot_cfg") or digest(planner._robot_cfg) != contract["planner_robot_config_digest"]:
            raise ContractError("Loaded cuRobo robot configuration differs from the reviewed model contract")
        kinematics = planner._robot_cfg.get("kinematics", {})
        urdf = Path(kinematics.get("urdf_path", ""))
        if (not urdf.is_absolute() or not urdf.is_file() or file_digest(urdf) != contract["planner_urdf_digest"]
                or kinematics.get("use_usd_kinematics", False) or kinematics.get("base_link") != "base_link"
                or not isinstance(kinematics.get("collision_spheres"), dict) or not kinematics["collision_spheres"]):
            raise ContractError("Explicit hashed URDF, base frame and inlined collision spheres are required; implicit model asset resolution is unavailable")
        self.maximum_fk_position_error_m, self.maximum_fk_rotation_error_rad = 0., 0.
        self.states_checked = 0

    def pose(self, joint_state):
        self.data.qpos[:] = joint_state.position
        self.data.qvel[:] = joint_state.velocity
        self.mujoco.mj_forward(self.model, self.data)
        wxyz = np.empty(4)
        self.mujoco.mju_mat2Quat(wxyz, self.data.site_xmat[self.tool_id])
        return self.data.site_xpos[self.tool_id].copy(), np.asarray([*wxyz[1:], wxyz[0]])

    def check_state(self, state):
        position, quaternion = self.pose(state)
        planner_position, planner_quaternion = self.planner.fk(list(state.position), quat_order="xyzw")
        planner_position, planner_quaternion = np.asarray(planner_position), np.asarray(planner_quaternion)
        if planner_position.shape != (3,) or planner_quaternion.shape != (4,) or not np.isfinite(planner_position).all() or not np.isfinite(planner_quaternion).all() or abs(np.dot(planner_quaternion, planner_quaternion)-1.) > 1e-5:
            raise MotionError("cuRobo FK returned invalid pose metadata")
        position_error = float(np.linalg.norm(position-planner_position))
        rotation_error = quaternion_distance(tuple(quaternion), tuple(planner_quaternion))
        if position_error > self.contract["fk_position_tolerance_m"] or rotation_error > self.contract["fk_rotation_tolerance_rad"]:
            raise MotionError("cuRobo/MuJoCo tool or kinematic model mismatch; no guessed tool correction applied")
        self.maximum_fk_position_error_m = max(self.maximum_fk_position_error_m, position_error)
        self.maximum_fk_rotation_error_rad = max(self.maximum_fk_rotation_error_rad, rotation_error)
        if np.any(position < self.contract["workspace_lower_m"]) or np.any(position > self.contract["workspace_upper_m"]):
            raise MotionError("Tool leaves the admitted simulation workspace")
        for joint in range(self.model.njnt):
            if self.model.jnt_limited[joint] and not self.model.jnt_range[joint, 0] <= state.position[joint] <= self.model.jnt_range[joint, 1]:
                raise MotionError("Trajectory exceeds MuJoCo joint limits")
        for contact in self.data.contact:
            if contact.dist < -1e-7:
                raise MotionError("MuJoCo candidate has an unpermitted geometric penetration")
        self.states_checked += 1

    def check_trajectory(self, trajectory, dependencies):
        if not trajectory.provenance.startswith("rammp_curobo:"+RAMMP_COMMIT+":"):
            raise MotionError("Simulation integration accepts only the pinned cuRobo adapter's trajectory")
        steps = max(1, math.ceil(trajectory.duration_s/self.contract["validation_sample_dt_s"]))
        if steps > 100000 or trajectory.duration_s > 120.:
            raise MotionError("Static simulation trajectory exceeds bounded verification work")
        for step in range(steps+1):
            self.check_state(trajectory.sample(trajectory.duration_s*step/steps))
        return True


def collision_world_from_mujoco(simulation, contract):
    """Represent supported static environment geometry in the verified wrapper API.

    The plane is bounded conservatively beyond the complete simulation workspace.
    Non-axis-aligned/unsupported environments are rejected instead of omitted.
    Robot geometry is independently checked by MuJoCo, not exported as obstacles.
    """
    model, data, mj = simulation.model, simulation.data, simulation.mujoco
    mj.mj_forward(model, data)
    lower, upper = np.asarray(contract["workspace_lower_m"]), np.asarray(contract["workspace_upper_m"])
    obstacles = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != 0 or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        if not np.allclose(data.geom_xmat[geom].reshape(3, 3), np.eye(3), atol=1e-9):
            raise ContractError("Static world rotation is not supported by this verified wrapper mapping")
        position = data.geom_xpos[geom].copy()
        kind = model.geom_type[geom]
        if kind == mj.mjtGeom.mjGEOM_PLANE:
            depth = max(2., float(upper[2]-lower[2])+2.)
            position = np.asarray([(lower[0]+upper[0])/2, (lower[1]+upper[1])/2, position[2]-depth/2])
            dims = [float(upper[0]-lower[0]+4.), float(upper[1]-lower[1]+4.), depth]
        elif kind == mj.mjtGeom.mjGEOM_BOX:
            dims = (2*model.geom_size[geom]).tolist()
        else:
            raise ContractError("Unsupported static collision geometry; an explicit representation adapter is required")
        obstacles.append({"name": "simulation_world_"+str(geom), "position": position.tolist(), "dims": dims})
    if not obstacles:
        raise ContractError("Empty planner collision world is not accepted")
    return obstacles


async def run_static_simulation(*, source_root, planner_config, contract_path, goal, image_path=None):
    contract = load_contract(contract_path)
    if not isinstance(goal, dict) or set(goal) != {"position_m", "quaternion_xyzw"}:
        raise ContractError("Goal must contain only metric position and xyzw quaternion")
    for key, size in (("position_m", 3), ("quaternion_xyzw", 4)):
        if not isinstance(goal[key], list) or len(goal[key]) != size or not all(type(v) in (int, float) and math.isfinite(v) for v in goal[key]):
            raise ContractError("Invalid explicit simulation goal")
    if abs(sum(v*v for v in goal["quaternion_xyzw"])-1) > 1e-6:
        raise ContractError("Goal quaternion is not normalized")
    root = Catalog().root/"simulation"
    if model_assets_digest(root) != contract["mujoco_assets_digest"] or file_digest(planner_config) != contract["planner_config_digest"]:
        raise ContractError("Model assets or planner configuration changed since compatibility review")
    simulation = MujocoReplay(root/"rolling_scene.xml")
    start = simulation.home()
    adapter = await RammpCuroboAdapter.load(source_root=source_root, planner_config=planner_config)
    return await _plan_validate_replay(adapter, simulation, contract, goal, start, image_path=image_path)


async def _plan_validate_replay(adapter, simulation, contract, goal, start, *, image_path=None, api_test_double=False):
    """Injectable seam for API wiring tests; test-double reports stay explicit."""
    verifier = MujocoCandidateVerifier(simulation, adapter.planner, contract)
    verifier.check_state(start)
    world = collision_world_from_mujoco(simulation, contract)
    world_identity = digest(world)
    trajectory = await adapter.plan_pose(position_m=goal["position_m"], quaternion_xyzw=goal["quaternion_xyzw"],
                                         start=start, world=world, world_identity=world_identity)
    if trajectory.duration_s > 120. or trajectory.duration_s/contract["validation_sample_dt_s"] > 100000:
        raise MotionError("Static simulation trajectory exceeds bounded verification work")
    if trajectory.joint_names != simulation.joint_names or not BoundaryTolerance(1e-5, 1e-5, 1e-4).matches(start, trajectory.points[0].state):
        raise MotionError("Returned trajectory does not start at the verified simulation joint boundary")
    terminal = trajectory.points[-1].state
    if any(abs(value) > 1e-5 for value in (*terminal.velocity, *terminal.acceleration)):
        raise MotionError("Static commissioning trajectory must finish stationary")
    limits = adapter.planner.joint_limits()
    bounds = np.asarray(limits["position"])
    velocity_bounds = np.asarray(limits["velocity"])
    if bounds.shape != (2, 7) or not np.isfinite(bounds).all() or velocity_bounds.shape != (7,) or not np.isfinite(velocity_bounds).all() or np.any(velocity_bounds <= 0):
        raise MotionError("Planner joint limit metadata invalid")
    validator = TrajectoryValidator(JointLimits(tuple(map(float, bounds[0])), tuple(map(float, bounds[1])), tuple(map(float, np.minimum(velocity_bounds, contract["joint_velocity_rad_s"]))),
                                               tuple(contract["joint_acceleration_rad_s2"])), verifier.check_trajectory,
                                    sample_dt_s=contract["validation_sample_dt_s"])
    certificate = validator.validate(trajectory, {"world": world_identity, "model": contract["mujoco_assets_digest"]}, now=0., expires_at=trajectory.duration_s+1.)
    reached_position, reached_quaternion = verifier.pose(terminal)
    if np.linalg.norm(reached_position-goal["position_m"]) > contract["goal_position_tolerance_m"] or quaternion_distance(tuple(reached_quaternion), tuple(goal["quaternion_xyzw"])) > contract["goal_rotation_tolerance_rad"]:
        raise MotionError("cuRobo candidate misses the explicit goal in the independent MuJoCo model")
    report = simulation.replay(certificate.trajectory, render_path=image_path)
    if report["maximum_tracking_error_rad"] > contract["maximum_tracking_error_rad"]:
        raise MotionError("Actual MuJoCo replay exceeds the local simulation tracking criterion")
    if report["maximum_contact_count"]:
        raise MotionError("Unexpected contact during physical replay; this seam has no intended-contact policy")
    actual_position, actual_quaternion = verifier.pose(JointState(tuple(map(float, simulation.data.qpos)), tuple(map(float, simulation.data.qvel)), tuple(map(float, simulation.data.qacc))))
    actual_position_error = float(np.linalg.norm(actual_position-goal["position_m"]))
    actual_rotation_error = quaternion_distance(tuple(actual_quaternion), tuple(goal["quaternion_xyzw"]))
    if actual_position_error > contract["goal_position_tolerance_m"] or actual_rotation_error > contract["goal_rotation_tolerance_rad"]:
        raise MotionError("Actual MuJoCo final tool pose misses the admitted simulation goal tolerance")
    report.update({"mode": "curobo_mujoco_static_simulation" if not api_test_double else "mock_curobo_api_mujoco_wiring_test",
                   "curobo_planned": not api_test_double, "planner_gpu_tested": not api_test_double,
                   "trajectory_digest": certificate.trajectory_digest, "contract_id": contract["contract_id"],
                   "world_identity": world_identity, "states_independently_checked": verifier.states_checked,
                   "maximum_fk_position_error_m": verifier.maximum_fk_position_error_m,
                   "maximum_fk_rotation_error_rad": verifier.maximum_fk_rotation_error_rad,
                   "actual_goal_position_error_m": actual_position_error, "actual_goal_rotation_error_rad": actual_rotation_error,
                   "reactive_motion_validated": False, "validation_scope": "Static simulation with sampled joint/FK/collision checks; no continuous swept/stopping proof, hardware or contact validation"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pinned cuRobo to MuJoCo static simulation; no physical robot transport")
    parser.add_argument("--describe-model", action="store_true", help="Print local MuJoCo asset identity; does not create a compatibility approval")
    parser.add_argument("--source-root")
    parser.add_argument("--planner-config")
    parser.add_argument("--contract")
    parser.add_argument("--goal", help="JSON file with position_m and quaternion_xyzw")
    parser.add_argument("--output", default="artifacts/curobo-mujoco.json")
    parser.add_argument("--image", default="artifacts/curobo-mujoco.png")
    args = parser.parse_args(argv)
    try:
        if args.describe_model:
            print(json.dumps({"mujoco_assets_digest": model_assets_digest(Catalog().root/"simulation"),
                              "planner_source_commit": RAMMP_COMMIT, "hardware_commands": False,
                              "detail": "A reviewed compatible planner robot/tool configuration and explicit local simulation criteria are still required"}, indent=2))
            return 0
        if not all((args.source_root, args.planner_config, args.contract, args.goal)):
            parser.error("--source-root, --planner-config, --contract and --goal are required")
        report = asyncio.run(run_static_simulation(source_root=args.source_root, planner_config=args.planner_config,
            contract_path=args.contract, goal=strict_loads(Path(args.goal).read_bytes()), image_path=args.image))
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"report": str(destination), "mode": report["mode"], "hardware_commands": False}))
        return 0
    except (ContractError, MotionError, RuntimeError, OSError, ImportError) as exc:
        print(json.dumps({"status": "unavailable_or_rejected", "detail": str(exc), "hardware_commands": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
