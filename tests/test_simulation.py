import asyncio
import importlib.util
import json
from pathlib import Path
import unittest

from rammp_adl.handlers import ExecutionContext, build_handlers
from rammp_adl.simulation import FixtureBackend, MujocoReplay, fixture_joint_trajectory

ROOT = Path(__file__).resolve().parents[1]


class FixtureSkillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = json.loads((ROOT/"examples/cabinet.context.json").read_text())
        self.backend = FixtureBackend(self.context)
        self.handlers = build_handlers(self.backend)

    async def call(self, skill, args, node=None):
        return await self.handlers[skill].execute(args, ExecutionContext("task", node or skill))

    async def aligned_grasp(self):
        move = await self.call("move_to_pose", {"target": {"entity_id": "cabinet_handle_1", "pose_role": "grasp"}, "profile_id": "sim_transit"})
        self.assertEqual(move.status, "succeeded")
        grasp = await self.call("grasp", {"entity_id": "cabinet_handle_1", "profile_id": "sim_gripper"})
        self.assertEqual(grasp.status, "succeeded")

    def test_exactly_six_explicit_handlers_no_hardware_capabilities(self):
        self.assertEqual(set(self.handlers), {"observe", "move_to_pose", "set_gripper", "grasp", "release", "follow_constraint"})
        self.assertFalse(self.backend.hardware_commands)
        self.assertFalse(self.backend.physical_capabilities)

    async def test_preshape_overlaps_move_with_per_skill_quiescence(self):
        self.backend.time_scale = 0.01
        move = asyncio.create_task(self.call("move_to_pose", {"target": {"entity_id": "cabinet_handle_1", "pose_role": "pregrasp"}, "profile_id": "sim_transit"}))
        gripper = await self.call("set_gripper", {"aperture_m": 0.06, "profile_id": "sim_gripper"})
        self.assertTrue(gripper.backend_quiescent)
        self.assertFalse(await self.backend.quiescent())
        self.assertEqual((await move).status, "succeeded")

    async def test_fixed_handle_grasp_retains_supported_relation(self):
        await self.aligned_grasp()
        entity = self.backend.entities["cabinet_handle_1"]
        self.assertEqual(entity.fixed_support, "cabinet_door_1")
        self.assertTrue(entity.retained)

    async def test_held_gripper_cannot_preshape(self):
        await self.aligned_grasp()
        result = await self.call("set_gripper", {"aperture_m": 0.08, "profile_id": "sim_gripper"})
        self.assertEqual(result.failure_code, "safety_fault")
        self.assertEqual(self.backend.holding_id, "cabinet_handle_1")

    async def test_constraint_partial_failure_never_claims_goal_success(self):
        await self.aligned_grasp()
        self.backend.failures["follow_constraint"] = ["model_mismatch"]
        result = await self.call("follow_constraint", {"entity_id": "cabinet_handle_1", "constraint_id": "hinge_1", "target_value": 1., "target_unit": "rad", "profile_id": "sim_cabinet_contact"})
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.backend_quiescent)
        self.assertAlmostEqual(self.backend.constraints["hinge_1"]["coordinate"], 0.35)
        self.assertEqual(result.proposed_effects[0]["validity"], "false")
        self.assertTrue(result.evidence[0]["data"]["partial"])

    async def test_constraint_absolute_goal_and_release(self):
        await self.aligned_grasp()
        args = {"entity_id": "cabinet_handle_1", "constraint_id": "hinge_1", "target_value": 1., "target_unit": "rad", "profile_id": "sim_cabinet_contact"}
        for _ in range(2):
            self.assertEqual((await self.call("follow_constraint", args)).status, "succeeded")
        self.assertEqual(self.backend.constraints["hinge_1"]["coordinate"], 1.)
        released = await self.call("release", {"entity_id": "cabinet_handle_1", "support_id": "cabinet_door_1", "profile_id": "sim_gripper"})
        self.assertEqual(released.status, "succeeded")
        self.assertIsNone(self.backend.holding_id)

    async def test_support_loss_preserves_grip(self):
        await self.aligned_grasp()
        self.backend.entities["cabinet_handle_1"].support_id = None
        released = await self.call("release", {"entity_id": "cabinet_handle_1", "support_id": "cabinet_door_1", "profile_id": "sim_gripper"})
        self.assertEqual(released.failure_code, "support_lost")
        self.assertEqual(self.backend.holding_id, "cabinet_handle_1")

    async def test_cancel_during_skill_produces_no_success_facts(self):
        self.backend.time_scale = 0.1
        context = ExecutionContext("task", "move")
        task = asyncio.create_task(self.handlers["move_to_pose"].execute({"target": {"entity_id": "cabinet_handle_1", "pose_role": "grasp"}, "profile_id": "sim_transit"}, context))
        await asyncio.sleep(0)
        context.cancel_event.set()
        result = await task
        self.assertEqual(result.status, "cancelled")
        self.assertFalse(result.proposed_effects)
        self.assertTrue(result.backend_quiescent)

    async def test_observation_requires_actual_visibility(self):
        self.backend.entities["cabinet_handle_1"].visible = False
        result = await self.call("observe", {"entity_id": "cabinet_handle_1", "camera": "wrist", "purpose": "grasp"})
        self.assertEqual(result.failure_code, "no_detection")


@unittest.skipUnless(importlib.util.find_spec("mujoco") and importlib.util.find_spec("numpy"), "optional MuJoCo/numpy physics dependencies unavailable")
class MujocoPhysicsTests(unittest.TestCase):
    def test_moving_generation_switch_against_actual_physics_state(self):
        report = MujocoReplay().replay_rolling_fixture()
        self.assertEqual(report["state"], "held")
        self.assertEqual(report["generation"], 1)
        self.assertLess(report["actual_activation_error"]["position_rad"], 0.015)
        self.assertLess(report["final_tracking_error_rad"], 0.01)
        self.assertFalse(report["curobo_planned"])
        self.assertFalse(report["hardware_validated"])

    def test_pinned_gen3_dynamic_joint_replay(self):
        simulation = MujocoReplay()
        home = simulation.home()
        target = list(home.position)
        target[0] += 0.1
        trajectory = fixture_joint_trajectory(home, target, duration_s=2.)
        report = simulation.replay(trajectory)
        self.assertEqual(report["mode"], "mujoco_physics_replay")
        self.assertFalse(report["curobo_planned"])
        self.assertFalse(report["hardware_validated"])
        self.assertFalse(report["gripper_simulated"])
        # Loose numerical regression criteria for THIS pinned bare-arm fixture,
        # not commissioned robot safety/tracking thresholds.
        self.assertLess(report["maximum_tracking_error_rad"], 0.05)
        self.assertLess(report["final_tracking_error_rad"], 0.01)
        self.assertGreater(report["maximum_joint_velocity_rad_s"], 0.01)
        self.assertGreater(len(report["samples"]), 20)


if __name__ == "__main__":
    unittest.main()
