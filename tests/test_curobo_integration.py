"""Real MuJoCo, explicitly mocked planner API; these are not GPU tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rammp_adl.contracts import ContractError, digest
from rammp_adl.motion.curobo import RAMMP_COMMIT, RammpCuroboAdapter, CuroboMpcAdapter
from rammp_adl.motion.integration import (
    _plan_validate_replay, file_digest, load_contract, model_assets_digest,
    MujocoCandidateVerifier, run_static_simulation,
)
from rammp_adl.motion.rolling import MotionError
from rammp_adl.simulation import MujocoReplay, fixture_joint_trajectory

ROOT = Path(__file__).resolve().parents[1]


class CuroboBoundaryTests(unittest.TestCase):
    def test_mpc_constructor_rejects_unbounded_or_nonnumeric_timing(self):
        valid = {"step_dt_s": .01, "maximum_solve_s": .1, "horizon_steps": 10, "joint_names": ["joint_1"]}
        for key, value in (("step_dt_s", True), ("maximum_solve_s", float("nan")), ("horizon_steps", 2.5), ("joint_names", ["j", "j"])):
            with self.subTest(key=key), self.assertRaises(MotionError):
                CuroboMpcAdapter(None, None, **dict(valid, **{key: value}))


class ExplicitPlannerApiDouble:
    """Verified wrapper API shape, with a JOINT FIXTURE instead of cuRobo."""
    def __init__(self, simulation, urdf, target, *, fk_offset=0.):
        self.simulation = simulation
        self.data = simulation.mujoco.MjData(simulation.model)
        self.joint_names = list(simulation.joint_names)
        self._robot_cfg = {"kinematics": {"urdf_path": str(urdf.resolve()), "base_link": "base_link",
                                         "collision_spheres": {"fixture": [{"center": [0., 0., 0.], "radius": .01}]}}}
        self.target, self.fk_offset = target, fk_offset
        self.calls = []

    def update_world(self, world):
        self.calls.append(("update_world", world))

    def plan_to_pose(self, position, quaternion, start, *, quat_order, apply_tool_correction):
        self.calls.append(("plan_to_pose", list(position), list(quaternion), start, quat_order, apply_tool_correction))
        return SimpleNamespace(success=True, status="EXPLICIT_API_TEST_DOUBLE", joint_traj=self.target)

    def fk(self, q, *, quat_order):
        if quat_order != "xyzw":
            raise AssertionError("Integration changed the verified quaternion convention")
        mj, model = self.simulation.mujoco, self.simulation.model
        self.data.qpos[:] = q
        mj.mj_forward(model, self.data)
        tool = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pinch_site")
        wxyz = np.empty(4)
        mj.mju_mat2Quat(wxyz, self.data.site_xmat[tool])
        position = self.data.site_xpos[tool].copy()
        position[0] += self.fk_offset
        return position.tolist(), [*map(float, wxyz[1:]), float(wxyz[0])]

    def joint_limits(self):
        return {"position": np.asarray([[-7.]*7, [7.]*7]), "velocity": np.asarray([2.]*7)}


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "optional MuJoCo unavailable")
class StaticIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.urdf = Path(self.temp.name)/"fixture.urdf"
        self.urdf.write_text("Explicit API test double, not a planner robot description", encoding="utf-8")
        self.simulation = MujocoReplay(ROOT/"simulation/rolling_scene.xml")
        self.start = self.simulation.home()
        end = list(self.start.position)
        end[0] += .04
        fixture = fixture_joint_trajectory(self.start, end, duration_s=2.)
        self.target = SimpleNamespace(joint_names=list(fixture.joint_names), dt=2.,
            positions=[p.state.position for p in fixture.points], velocities=[p.state.velocity for p in fixture.points],
            accelerations=[p.state.acceleration for p in fixture.points])
        self.planner = ExplicitPlannerApiDouble(self.simulation, self.urdf, self.target)
        self.adapter = RammpCuroboAdapter(self.planner, source_commit=RAMMP_COMMIT,
                                        installed_module_path=ROOT/"fixture_planner.py", source_root=ROOT)
        self.contract = {"simulation_only": True, "contract_id": "API_TEST_DOUBLE_NOT_GPU",
            "planner_source_commit": RAMMP_COMMIT, "planner_config_digest": file_digest(self.urdf),
            "mujoco_assets_digest": model_assets_digest(ROOT/"simulation"), "planner_urdf_digest": file_digest(self.urdf),
            "planner_robot_config_digest": digest(self.planner._robot_cfg), "base_frame": "base_link", "tool_site": "pinch_site",
            "joint_names": list(self.simulation.joint_names), "workspace_lower_m": [-2., -2., -.1], "workspace_upper_m": [2., 2., 2.],
            "fk_position_tolerance_m": .001, "fk_rotation_tolerance_rad": .001,
            "joint_velocity_rad_s": [2.]*7, "joint_acceleration_rad_s2": [10.]*7, "validation_sample_dt_s": .02,
            "maximum_tracking_error_rad": .02, "goal_position_tolerance_m": .01, "goal_rotation_tolerance_rad": .02}
        position, quaternion = self.planner.fk(end, quat_order="xyzw")
        self.goal = {"position_m": position, "quaternion_xyzw": quaternion}

    async def run_seam(self):
        return await _plan_validate_replay(self.adapter, self.simulation, self.contract, self.goal, self.start, api_test_double=True)

    async def test_verified_adapter_wiring_replays_exact_returned_path_in_mujoco(self):
        report = await self.run_seam()
        self.assertEqual(report["mode"], "mock_curobo_api_mujoco_wiring_test")
        self.assertFalse(report["curobo_planned"])
        self.assertFalse(report["planner_gpu_tested"])
        self.assertFalse(report["hardware_validated"])
        self.assertEqual(report["maximum_contact_count"], 0)
        self.assertLess(report["actual_goal_position_error_m"], .01)
        self.assertGreater(report["states_independently_checked"], 100)
        self.assertEqual([call[0] for call in self.planner.calls], ["update_world", "plan_to_pose"])
        self.assertEqual(self.planner.calls[1][1:3], (self.goal["position_m"], self.goal["quaternion_xyzw"]))
        self.assertEqual(self.planner.calls[1][-2:], ("xyzw", False))
        self.assertEqual(report["world_identity"], digest(self.planner.calls[0][1]))
        expected = await self.adapter.plan_pose(position_m=self.goal["position_m"], quaternion_xyzw=self.goal["quaternion_xyzw"],
            start=self.start, world=self.planner.calls[0][1], world_identity=report["world_identity"])
        self.assertEqual(report["trajectory_digest"], expected.digest)

    async def test_fk_tool_mismatch_rejected_before_planning(self):
        self.planner.fk_offset = .02
        with self.assertRaisesRegex(MotionError, "model mismatch"):
            await self.run_seam()
        self.assertEqual(self.planner.calls, [])

    async def test_altered_urdf_rejected_before_planning(self):
        self.urdf.write_text("mutated after review", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "hashed URDF"):
            await self.run_seam()
        self.assertEqual(self.planner.calls, [])

    async def test_returned_out_of_bounds_trajectory_never_replayed(self):
        changed = list(self.target.positions[-1])
        changed[1] = 3.
        self.target.positions[-1] = changed
        with patch.object(self.simulation, "replay") as replay:
            with self.assertRaises(MotionError):
                await self.run_seam()
            replay.assert_not_called()

    async def test_returned_start_mismatch_never_replayed(self):
        changed = list(self.target.positions[0])
        changed[0] += .001
        self.target.positions[0] = changed
        with patch.object(self.simulation, "replay") as replay:
            with self.assertRaisesRegex(MotionError, "joint boundary"):
                await self.run_seam()
            replay.assert_not_called()

    async def test_model_hash_rejection_precedes_gpu_load(self):
        self.contract["mujoco_assets_digest"] = "sha256:"+"0"*64
        contract_path = Path(self.temp.name)/"contract.json"
        contract_path.write_text(json.dumps(self.contract), encoding="utf-8")
        with patch.object(RammpCuroboAdapter, "load") as load:
            with self.assertRaisesRegex(ContractError, "changed since"):
                await run_static_simulation(source_root=ROOT, planner_config=self.urdf, contract_path=contract_path, goal=self.goal)
            load.assert_not_called()

    def test_closed_contract_rejects_nonfinite_unknown_fields_and_implicit_assets(self):
        contract_path = Path(self.temp.name)/"contract.json"
        contract_path.write_text(json.dumps(self.contract), encoding="utf-8")
        self.assertEqual(load_contract(contract_path), self.contract)
        for field, value in (("validation_sample_dt_s", float("nan")), ("unexpected", True)):
            changed = dict(self.contract)
            changed[field] = value
            contract_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_contract(contract_path)
        self.planner._robot_cfg["kinematics"]["urdf_path"] = "implicit/robot.urdf"
        self.contract["planner_robot_config_digest"] = digest(self.planner._robot_cfg)
        with self.assertRaisesRegex(ContractError, "implicit model asset"):
            MujocoCandidateVerifier(self.simulation, self.planner, self.contract)


if __name__ == "__main__":
    unittest.main()
