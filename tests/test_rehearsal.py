"""Real MuJoCo plus synthetic planner fixtures; actual GPU is a separate check."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from rammp_adl.contracts import Catalog, digest
from rammp_adl.motion.curobo import CUROBO_VERSION, RAMMP_COMMIT
from rammp_adl.motion.curobo_worker import serialize_trajectory
from rammp_adl.motion.integration import model_assets_digest
from rammp_adl.motion.rehearsal import (MujocoDriverPort, SimulationAdmission,
    checked_simulation_candidate, run_case, simulation_bounds, simulation_world)
from rammp_adl.motion.rolling import JointState, MotionError
from rammp_adl.simulation import MujocoReplay, fixture_joint_trajectory


def model_contract():
    return {"joint_names": [f"joint_{i}" for i in range(1,8)], "base_frame": "base_link", "tool_site": "pinch_site",
        "planner_config_digest": "simulation-config", "planner_urdf_digest": "simulation-urdf",
        "planner_robot_config_digest": digest({"test_only": True}),
        "mujoco_assets_digest": model_assets_digest(Catalog().root/"simulation"),
        "workspace_lower_m": [-1.5,-1.5,0.], "workspace_upper_m": [1.5,1.5,1.5],
        "fk_position_tolerance_m": .001, "fk_rotation_tolerance_rad": .001,
        "goal_position_tolerance_m": .01, "goal_rotation_tolerance_rad": .02,
        "maximum_tracking_error_rad": .02,
        "joint_velocity_rad_s": [1.2]*7, "joint_acceleration_rad_s2": [4.]*7,
        "validation_sample_dt_s": .005}


def simulation():
    return MujocoReplay(Catalog().root/"simulation/rolling_scene.xml")


class PlannerFixture:
    """Analytic joint stimulus for wiring tests only; never used by the CLI."""
    def __init__(self, contract):
        self.contract = contract
        self.model = simulation()
        self.admission = SimulationAdmission(self.model, self, contract, simulation_bounds(), [{"fixture": True}])
        home = self.model.home()
        self.goal_position = tuple(q + (.012 if i == 0 else 0.) for i,q in enumerate(home.position))
        endpoint, quat = self.admission.pose(JointState(self.goal_position, (0.,)*7, (0.,)*7))
        self.goal = {"position_m": endpoint.tolist(), "quaternion_xyzw": quat.tolist()}
        self.last_response = self.last_request = None

    async def plan_pose(self, *, position_m, quaternion_xyzw, start, world, world_identity):
        path = fixture_joint_trajectory(start, self.goal_position, duration_s=.5)
        endpoint, quat = self.admission.pose(path.points[-1].state)
        self.last_request = {"goal": {"position_m": position_m, "quaternion_xyzw": quaternion_xyzw}}
        self.last_response = {"endpoint_fk": {"position_m": endpoint.tolist(), "quaternion_xyzw": quat.tolist()},
            "joint_limits": {"position": [[-6.]*7, [6.]*7], "velocity": [1.2]*7}, "planning_wall_s": 0.}
        return path


class RehearsalPhysicsTests(unittest.IsolatedAsyncioTestCase):
    async def test_world_binding_gateway_completion_cancel_and_invalidation_with_actual_physics(self):
        contract = model_contract()
        planner = PlannerFixture(contract)
        with tempfile.TemporaryDirectory() as directory:
            for case in ("complete", "cancel", "dependency_change", "driver_fault", "stale_feedback"):
                with self.subTest(case=case):
                    result = await run_case(planner=planner, contract=contract, goal=planner.goal,
                                           case=case, output=Path(directory)/(case+".json"))
                    self.assertTrue(result["passed"])
                    self.assertGreater(result["physics_steps"], 20)
                    self.assertEqual(result["simulated_action_count"], 1)
                    self.assertEqual(result["maximum_contact_count"], 0)
                    self.assertFalse(result["motion_sent_to_physical_robot"])
                    recorded = json.loads((Path(directory)/(case+"-trial")/"report.json").read_text())
                    self.assertEqual(recorded["scope"], "admitted_simulation_trial")
                    self.assertGreater(recorded["sample_count"], 0)
                    self.assertFalse(recorded["hardware_limits_established"])
                    if case == "stale_feedback":
                        self.assertFalse(result["receipt"]["measured_quiescent"])
                        self.assertEqual(result["release_count"], 0)
                        self.assertEqual(set(result["resources_retained"]), {"ARM", "PLANNER"})
                    else:
                        self.assertTrue(result["receipt"]["measured_quiescent"])
                    if case == "complete":
                        self.assertEqual(result["release_count"], 1)
                        self.assertEqual(result["resources_retained"], {})
                        self.assertLess(result["actual_goal_position_error_m"], .01)

    async def test_frozen_feedback_never_refreshes_acquisition_identity(self):
        port = MujocoDriverPort(simulation())
        port.start()
        try:
            await port.stationary(simulation_bounds())
            previous = port.latest
            port.feedback_frozen = True
            await asyncio.sleep(.02)
            self.assertIs(port.latest, previous)
            self.assertGreater(port.sequence, previous.sequence)
        finally:
            await port.close()

    async def test_terminal_ack_does_not_overwrite_measured_joint_state(self):
        port = MujocoDriverPort(simulation())
        port.start()
        try:
            sample = await port.stationary(simulation_bounds())
            start = JointState(sample.position_rad, (0.,)*7, (0.,)*7)
            path = fixture_joint_trajectory(start, tuple(q+.01 for q in start.position), duration_s=.5)
            handle = port.send_goal(path, None, simulation_bounds()).result()
            await asyncio.sleep(.08)
            before_q, before_dq = port.data.qpos.copy(), port.data.qvel.copy()
            handle.cancel_goal_async()
            self.assertTrue((before_q == port.data.qpos).all())
            self.assertTrue((before_dq == port.data.qvel).all())
            self.assertGreater(max(map(abs, before_dq)), 0.)
            self.assertEqual(handle.result.result().status, 5)
        finally:
            await port.close()


class RehearsalProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.contract = model_contract()
        self.request = {"world_identity": "world-fixture"}
        start = JointState((0.,)*7, (0.,)*7, (0.,)*7)
        path = fixture_joint_trajectory(start, (.01,)*7, duration_s=.5)
        from dataclasses import replace
        self.path = replace(path, provenance="rammp_curobo:"+RAMMP_COMMIT+":test-double-only")
        self.response = {"status": "planned", "hardware_commands": False, "curobo_planned": True,
            "request_digest": digest(self.request), "world_identity": self.request["world_identity"],
            "planner_source_commit": RAMMP_COMMIT, "curobo_version": CUROBO_VERSION,
            "base_frame": "base_link", "ee_link": "end_effector_link", "planner_robot_config": {"test_only": True},
            **{k:self.contract[k] for k in ("planner_config_digest", "planner_urdf_digest", "planner_robot_config_digest")},
            "trajectory": serialize_trajectory(self.path)}

    def test_wrong_world_model_source_digest_and_interpolation_are_rejected(self):
        self.assertEqual(checked_simulation_candidate(self.response, self.request, self.contract), self.path)
        for key in ("world_identity", "request_digest", "planner_source_commit", "planner_urdf_digest", "base_frame"):
            response = deepcopy(self.response)
            response[key] = "incorrect"
            with self.subTest(key=key), self.assertRaises(MotionError):
                checked_simulation_candidate(response, self.request, self.contract)
        for key in ("digest", "interpolation"):
            response = deepcopy(self.response)
            response["trajectory"][key] = "incorrect"
            with self.subTest(key=key), self.assertRaises(MotionError):
                checked_simulation_candidate(response, self.request, self.contract)

    def test_target_fixture_has_no_hardware_capabilities_or_predicted_success(self):
        planner = PlannerFixture(self.contract)
        world = simulation_world(planner.goal, self.contract)
        snapshot = world.snapshot()
        self.assertEqual(snapshot.context["available_skills"], [])
        self.assertNotEqual(snapshot.fact("at_pose", {"entity_id": "simulation_target", "pose_role": "staging"}), "true")
        pose = snapshot.metric_poses[("simulation_target", "staging")]
        self.assertEqual(pose.evidence_id, "simulation-input")
        self.assertIn("simulation", pose.calibration_id)


if __name__ == "__main__":
    unittest.main()
