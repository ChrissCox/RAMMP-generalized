import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from rammp_adl.app import fixture_runtime
from rammp_adl.contracts import ContractError, strict_loads
from rammp_adl.ros_bridge import RuntimeBridge
from rammp_adl.ros_node import AsyncWorker


ROOT = Path(__file__).resolve().parents[1]


def load(name, kind="plan"):
    return strict_loads((ROOT / "examples" / f"{name}.{kind}.json").read_bytes())


class ExecutorIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self, **kwargs):
        return fixture_runtime(load("cabinet", "context"), root=ROOT, **kwargs)

    async def test_cabinet_complete_and_preshape_overlaps_transit(self):
        runtime = self.runtime(time_scale=.05)
        result = await runtime.executor.run_plan(load("cabinet"))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertTrue(runtime.world.goal_satisfied())
        self.assertEqual(len(result.nodes), 8)
        self.assertEqual(runtime.executor.resources.owners, {})
        events = runtime.backend.events
        order = {(e["event"], e.get("node_id")): i for i, e in enumerate(events)}
        self.assertLess(order["started", "c2"], order["finished", "c3"])
        self.assertLess(order["started", "c3"], order["finished", "c2"])
        commits = [e for e in runtime.trace.events if e["event"] == "effects_committed"]
        self.assertEqual(len(commits), 8)

    async def test_invalid_whole_plan_executes_no_prefix(self):
        runtime = self.runtime()
        plan = load("cabinet")
        plan["nodes"][-1]["args"]["target"]["entity_id"] = "invented-object"
        result = await runtime.executor.run_plan(plan)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(runtime.backend.events, [])

    async def test_model_mismatch_stops_before_release_and_records_partial_state(self):
        runtime = self.runtime(failures={"c6": ["model_mismatch"]})
        result = await runtime.executor.run_plan(load("cabinet"))
        self.assertEqual(result.status, "incomplete", result.to_dict())
        self.assertNotIn("c7", [e.get("node_id") for e in runtime.backend.events])
        self.assertFalse(runtime.world.goal_satisfied())
        self.assertTrue(runtime.executor.safety.held_verified)
        self.assertEqual(runtime.backend.holding_id, "cabinet_handle_1")
        self.assertEqual(sum(e["event"] == "started" and e.get("node_id") == "c6" for e in runtime.backend.events), 1)

    async def test_catalog_allows_one_stationary_observation_retry(self):
        runtime = self.runtime(failures={"c1": ["no_detection"]})
        result = await runtime.executor.run_plan(load("cabinet"))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual(sum(e["event"] == "started" and e.get("node_id") == "c1" for e in runtime.backend.events), 2)

    async def test_retry_cap_exhaustion_executes_no_motion(self):
        runtime = self.runtime(failures={"c1": ["no_detection", "no_detection", "no_detection"]})
        result = await runtime.executor.run_plan(load("cabinet"))
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(sum(e["event"] == "started" and e.get("node_id") == "c1" for e in runtime.backend.events), 2)
        self.assertFalse(any(e.get("skill") == "move_to_pose" for e in runtime.backend.events))

    async def test_user_cancel_preserves_hold_and_blocks_dependents(self):
        runtime = self.runtime(time_scale=.5)
        work = asyncio.create_task(runtime.executor.run_plan(load("cabinet")))
        async def wait_for_motion():
            while "move_to_pose" not in runtime.backend.active:
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait_for_motion(), 2.)
        await runtime.executor.cancel()
        result = await asyncio.wait_for(work, 3.)
        self.assertNotEqual(result.status, "succeeded")
        self.assertTrue(runtime.executor.safety.held_verified)
        self.assertEqual(runtime.executor.resources.owners, {})
        self.assertFalse(any(e.get("node_id") == "c5" for e in runtime.backend.events))

    async def test_bottle_fixture_runs_to_its_staging_goal(self):
        runtime = fixture_runtime(load("bottle", "context"), root=ROOT)
        result = await runtime.executor.run_plan(load("bottle"))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertTrue(runtime.world.goal_satisfied())

    async def test_the_same_failure_twice_ends_the_task_instead_of_spending_the_replan_budget(self):
        # The door model is wrong every time it is followed: c6 in the first plan, r2 in each resume.
        runtime = self.runtime(failures={"c6": ["model_mismatch"], "r2": ["model_mismatch"]*3})
        plans = [load("cabinet-reobserve"), load("cabinet-resume"), load("cabinet-resume")]
        class FixtureReasoner:
            async def generate_plan(self, context, **kwargs):
                plan = copy.deepcopy(plans.pop(0))
                for key in ("task_id", "snapshot_id", "execution_epoch"):
                    plan[key] = context[key]
                return SimpleNamespace(status="OK", plan=plan, detail="explicit test response")
        result = await runtime.executor.run_task("Open the cabinet", FixtureReasoner(), initial_plan=load("cabinet"))
        self.assertEqual(result.status, "incomplete", result.to_dict())
        self.assertIn("model_mismatch repeated after a replan", result.reason)
        self.assertEqual(len(plans), 1)                                          # the third resume was never asked for

    async def test_recovery_keeps_original_goal_across_observation_phase(self):
        runtime = self.runtime(failures={"c6": ["model_mismatch"]})
        plans = [load("cabinet-reobserve"), load("cabinet-resume")]
        class FixtureReasoner:
            async def generate_plan(self, context, **kwargs):
                plan = copy.deepcopy(plans.pop(0))
                for key in ("task_id", "snapshot_id", "execution_epoch"):
                    plan[key] = context[key]
                return SimpleNamespace(status="OK", plan=plan, detail="explicit test response")
        result = await runtime.executor.run_task("Open the cabinet", FixtureReasoner(), initial_plan=load("cabinet"))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual(result.task_replans, 2)
        self.assertTrue(runtime.world.goal_satisfied())
        self.assertFalse(plans)


class BridgeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_forged_skill_args_cannot_use_valid_receipt(self):
        runtime = fixture_runtime(load("cabinet", "context"), root=ROOT)
        bridge = RuntimeBridge(runtime)
        plan = load("cabinet")
        snapshot = runtime.world.snapshot()
        await bridge.validate(plan, snapshot_id=snapshot.snapshot_id, epoch=snapshot.execution_epoch)
        dispatch = await bridge.validate(plan, snapshot_id=snapshot.snapshot_id, epoch=snapshot.execution_epoch, node_id="c1")
        request = {"task_id": plan["task_id"], "execution_epoch": plan["execution_epoch"], "node_id": "c1",
                   "attempt": 1, "skill_id": "observe", "args": {**plan["nodes"][0]["args"], "entity_id": "robot"},
                   "validation_id": dispatch.receipt_id}
        with self.assertRaises(ContractError):
            await bridge.execute_skill(request)
        self.assertEqual(runtime.backend.events, [])

    async def test_direct_skill_requires_predecessor_effects(self):
        runtime = fixture_runtime(load("cabinet", "context"), root=ROOT)
        bridge = RuntimeBridge(runtime)
        plan = load("cabinet")
        snapshot = runtime.world.snapshot()
        await bridge.validate(plan, snapshot_id=snapshot.snapshot_id, epoch=snapshot.execution_epoch)
        with self.assertRaises(ContractError):
            await bridge.validate(plan, snapshot_id=snapshot.snapshot_id, epoch=snapshot.execution_epoch, node_id="c6")


class AsyncWorkerTests(unittest.TestCase):
    def test_ros_async_worker_runs_and_shuts_down_without_ros(self):
        worker = AsyncWorker()
        async def value():
            return 42
        try:
            self.assertEqual(worker.call(value()), 42)
        finally:
            worker.close()
        self.assertFalse(worker.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
