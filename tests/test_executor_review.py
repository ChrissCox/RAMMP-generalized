"""Adversarial executor/bridge race tests using explicit synthetic backends."""
from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import unittest

from rammp_adl.app import fixture_runtime
from rammp_adl.contracts import ContractError, strict_loads
from rammp_adl.handlers import SkillOutcome
from rammp_adl.ros_bridge import RuntimeBridge

ROOT = Path(__file__).resolve().parents[1]


class ExecutorReviewTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self, **kwargs):
        context = strict_loads((ROOT / "examples/cabinet.context.json").read_bytes())
        return fixture_runtime(context, root=ROOT, **kwargs)

    def plan(self, runtime, nodes=None):
        plan = strict_loads((ROOT / "examples/cabinet.plan.json").read_bytes())
        snapshot = runtime.world.snapshot()
        for name in ("task_id", "snapshot_id", "execution_epoch"):
            plan[name] = snapshot.context[name]
        if nodes is not None:
            plan["nodes"], plan["edges"] = nodes, []
        return plan

    def observe(self, node_id="o", purpose="pose"):
        return {"id": node_id, "skill": "observe", "args": {
            "entity_id": "cabinet_handle_1", "camera": "scene", "purpose": purpose}}

    def gripper(self, node_id="g"):
        return {"id": node_id, "skill": "set_gripper", "args": {"aperture_m": 0.06, "profile_id": "sim_gripper"}}

    async def wait_until(self, predicate, timeout=2.0):
        async def poll():
            while not predicate():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(poll(), timeout)

    def replace_handler(self, runtime, skill, execute):
        handlers = dict(runtime.registry.handlers)
        handlers[skill] = SimpleNamespace(execute=execute)
        runtime.registry.handlers = MappingProxyType(handlers)

    async def validated_request(self, bridge, plan, node_id):
        snapshot = bridge.runtime.world.snapshot()
        await bridge.validate(plan, snapshot_id=snapshot.snapshot_id, epoch=snapshot.execution_epoch)
        receipt = await bridge.validate(plan, snapshot_id=snapshot.snapshot_id, epoch=snapshot.execution_epoch, node_id=node_id)
        node = next(n for n in plan["nodes"] if n["id"] == node_id)
        return {"validation_id": receipt.receipt_id, "task_id": plan["task_id"],
                "execution_epoch": plan["execution_epoch"], "node_id": node_id,
                "skill_id": node["skill"], "args": copy.deepcopy(node["args"]), "attempt": 1}

    async def test_held_flag_clears_during_motion_and_refreshes_at_global_completion(self):
        runtime = self.runtime(time_scale=0.2)
        task = asyncio.create_task(runtime.executor.run_plan(self.plan(runtime)))
        await self.wait_until(lambda: "move_to_pose" in runtime.backend.active)
        self.assertFalse(runtime.executor.safety.held_verified)
        await self.wait_until(lambda: any(e["event"] == "finished" and e.get("node_id") == "c3" for e in runtime.backend.events))
        if "move_to_pose" in runtime.backend.active:
            self.assertFalse(runtime.executor.safety.held_verified)
        result = await task
        self.assertEqual(result.status, "succeeded")
        self.assertTrue(runtime.executor.safety.held_verified)

    async def test_unknown_handler_exception_stops_and_releases_only_verified_resources(self):
        runtime = self.runtime()
        async def broken(args, context):
            raise RuntimeError("injected handler exception")
        self.replace_handler(runtime, "set_gripper", broken)
        result = await runtime.executor.run_plan(self.plan(runtime, [self.gripper()]))
        self.assertNotEqual(result.status, "succeeded")
        self.assertEqual(result.nodes[0].status, "failed")
        self.assertTrue(runtime.executor.safety.held_verified)
        self.assertEqual(runtime.executor.resources.owners, {})

    async def test_handler_quiescent_label_is_checked_against_backend(self):
        runtime = self.runtime()
        async def dishonest(args, context):
            runtime.backend.active.add("set_gripper")
            return SkillOutcome("succeeded", backend_quiescent=True)
        self.replace_handler(runtime, "set_gripper", dishonest)
        result = await runtime.executor.run_plan(self.plan(runtime, [self.gripper()]))
        self.assertEqual(result.status, "safety_fault")
        self.assertNotIn("g", runtime.world.snapshot().context["completed_nodes"])

    async def test_noncooperative_handler_is_bounded_and_keeps_ownership(self):
        runtime = self.runtime()
        runtime.executor.safety.stop_timeout_s = 0.02
        runtime.catalog.skills["set_gripper"]["timing"]["timeout_s"] = 0.02
        release = asyncio.Event()
        async def stuck(args, context):
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            return SkillOutcome("cancelled", backend_quiescent=True)
        self.replace_handler(runtime, "set_gripper", stuck)
        try:
            result = await asyncio.wait_for(runtime.executor.run_plan(self.plan(runtime, [self.gripper()])), 1.0)
            self.assertEqual(result.status, "safety_fault")
            self.assertFalse(result.nodes[0].backend_quiescent)
            self.assertEqual(runtime.executor.resources.owners, {"GRIPPER": "g"})
        finally:
            release.set()
            await asyncio.sleep(0)

    async def test_user_cancellation_never_turns_into_task_replanning(self):
        runtime = self.runtime(time_scale=0.3)
        calls = []
        class Reasoner:
            async def generate_plan(self, *args, **kwargs):
                calls.append("unexpected replan")
                return SimpleNamespace(status="NEED_OBSERVATION", detail="must not be called")
        task = asyncio.create_task(runtime.executor.run_task("open", Reasoner(), initial_plan=self.plan(runtime)))
        await self.wait_until(lambda: "move_to_pose" in runtime.backend.active)
        await runtime.executor.cancel("USER_CANCELLED")
        result = await task
        self.assertEqual(result.status, "cancelled", result.to_dict())
        self.assertEqual(calls, [])

    async def test_task_ownership_covers_cloud_wait_and_cancel_is_prompt(self):
        runtime = self.runtime()
        entered = asyncio.Event()
        class Reasoner:
            async def generate_plan(self, *args, **kwargs):
                entered.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(runtime.executor.run_task("open", Reasoner()))
        await entered.wait()
        with self.assertRaises(RuntimeError):
            await runtime.executor.run_plan(self.plan(runtime))
        with self.assertRaises(RuntimeError):
            await runtime.executor.run_task("second", Reasoner())
        await runtime.executor.cancel()
        result = await asyncio.wait_for(task, 1.0)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(runtime.backend.events, [])

    async def test_outer_reasoning_deadline_is_enforced_independently_of_provider(self):
        runtime = self.runtime()
        runtime.executor._reasoning_deadline_s = 0.02
        class Reasoner:
            async def generate_plan(self, *args, **kwargs):
                await asyncio.Event().wait()
        result = await asyncio.wait_for(runtime.executor.run_task("open", Reasoner()), 1.0)
        self.assertEqual(result.status, "incomplete")
        self.assertIn("deadline", result.reason)
        self.assertTrue(runtime.executor.safety.held_verified)
        self.assertEqual(runtime.backend.events, [])

    async def test_late_success_after_cancel_commits_measurement_without_node_success(self):
        runtime = self.runtime()
        entered = asyncio.Event()
        async def late(args, context):
            entered.set()
            await context.cancel_event.wait()
            return runtime.backend._outcome([{"predicate": "aperture_reached", "args": {"aperture_m": 0.06}, "validity": "true"}])
        self.replace_handler(runtime, "set_gripper", late)
        task = asyncio.create_task(runtime.executor.run_plan(self.plan(runtime, [self.gripper()])))
        await entered.wait()
        await runtime.executor.cancel()
        result = await task
        self.assertEqual(result.status, "cancelled")
        self.assertNotIn("g", runtime.world.snapshot().context["completed_nodes"])
        self.assertEqual(runtime.world.snapshot().fact("aperture_reached", {"aperture_m": 0.06}), "true")

    async def test_replan_budget_cannot_be_overridden_above_catalog_policy(self):
        runtime = self.runtime()
        with self.assertRaises(ContractError):
            await runtime.executor.run_task("open", object(), max_replans=4)

    async def test_catalog_handback_is_not_automatically_replanned(self):
        runtime = self.runtime(failures={"c1": ["no_detection", "no_detection"]})
        calls = []
        class Reasoner:
            async def generate_plan(self, *args, **kwargs):
                calls.append("unexpected replan")
                return SimpleNamespace(status="REFUSED", detail="not called")
        result = await runtime.executor.run_task("open", Reasoner(), initial_plan=self.plan(runtime))
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.task_replans, 0)
        self.assertEqual(calls, [])

    async def test_external_grounding_reserves_held_state_and_blocks_execution(self):
        runtime = self.runtime()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        class Reasoner:
            async def ground_target(self, context, entity_id, images, *, query, request_id):
                calls.append((entity_id, query, request_id))
                entered.set()
                await release.wait()
                return SimpleNamespace(status="OK", candidates=[])
        bridge = RuntimeBridge(runtime, Reasoner())
        plan = self.plan(runtime, [self.observe()])
        action_request = await self.validated_request(bridge, plan, "o")
        grounding = asyncio.create_task(bridge.ground_target({"request_id": "g1", "epoch": 1,
                                                             "entity_id": "cabinet_handle_1", "image_ids": [],
                                                             "query": "find this exact handle"}))
        await entered.wait()
        self.assertEqual(set(runtime.executor.resources.owners), {"ARM", "GRIPPER", "CLOUD_RPC"})
        with self.assertRaises(RuntimeError):
            await runtime.executor.run_plan(plan)
        with self.assertRaises(ContractError):
            await bridge.execute_skill(action_request)
        release.set()
        self.assertEqual((await grounding).status, "OK")
        self.assertEqual(calls, [("cabinet_handle_1", "find this exact handle", "g1")])
        self.assertEqual(runtime.executor.resources.owners, {})

    async def test_manual_failure_reconciles_epoch_before_next_task(self):
        runtime = self.runtime(failures={"o": ["no_detection"]})
        bridge = RuntimeBridge(runtime)
        plan = self.plan(runtime, [self.observe()])
        request = await self.validated_request(bridge, plan, "o")
        result = await bridge.execute_skill(request)
        self.assertEqual(result.status, "failed")
        self.assertTrue(runtime.executor.safety.held_verified)
        self.assertFalse(runtime.executor.safety.fault_latched)
        self.assertEqual(runtime.executor.resources.owners, {})
        self.assertIn(1, runtime.world._sealed)

    async def test_manual_replay_uses_node_attempt_guard_not_only_receipt_id(self):
        runtime = self.runtime()
        bridge = RuntimeBridge(runtime)
        plan = self.plan(runtime, [self.observe()])
        first = await self.validated_request(bridge, plan, "o")
        second = await self.validated_request(bridge, plan, "o")
        result = await bridge.execute_skill(first)
        self.assertEqual(result.status, "succeeded")
        with self.assertRaises(ContractError):
            await bridge.execute_skill(second)
        self.assertEqual(sum(e["event"] == "started" for e in runtime.backend.events), 1)

    async def test_manual_plans_cannot_interleave_under_one_epoch(self):
        runtime = self.runtime()
        bridge = RuntimeBridge(runtime)
        first_plan = self.plan(runtime, [self.observe("a", "pose")])
        second_plan = self.plan(runtime, [self.observe("b", "articulation")])
        first = await self.validated_request(bridge, first_plan, "a")
        second = await self.validated_request(bridge, second_plan, "b")
        self.assertEqual((await bridge.execute_skill(first)).status, "succeeded")
        with self.assertRaises(ContractError):
            await bridge.execute_skill(second)
        with self.assertRaises(RuntimeError):
            await runtime.executor.run_plan(self.plan(runtime))

    async def test_atomic_success_check_rejects_contradictory_terminal_effects(self):
        runtime = self.runtime()
        world = runtime.world
        authority = world.authorize_source("review", ["at_pose"])
        effects = [{"predicate": "at_pose", "args": {"entity_id": "cabinet_handle_1", "pose_role": role},
                    "validity": "true", "evidence_id": "contradiction"} for role in ("pregrasp", "grasp")]
        world.register_evidence("contradiction", source=authority, predicates=effects, ttl_s=10)
        world.register_operation("op", 1)
        with self.assertRaises(ContractError):
            world.commit_effects("op", effects, 1, 1, completed_node="move",
                                 expected_postconditions=[{"predicate": "at_pose", "args": effects[0]["args"]}])
        self.assertEqual(world.snapshot().revision, 1)
        self.assertEqual(world.snapshot().context["completed_nodes"], [])

    async def test_external_commit_lookup_cannot_take_over_inflight_operation(self):
        runtime = self.runtime()
        world = runtime.world
        world.register_operation("op", 1)
        with self.assertRaises(ContractError):
            world.committed_receipt("op", [], 1)
        committed = world.commit_effects("op", [], 1, 1)
        self.assertEqual(world.committed_receipt("op", [], 1), committed)
        with self.assertRaises(ContractError):
            world.committed_receipt("op", [], 2)


if __name__ == "__main__":
    unittest.main()
