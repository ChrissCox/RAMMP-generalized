"""Astra request/response contract through the actual executor; provider is scripted."""
import copy
import json
from pathlib import Path
import unittest

from rammp_adl.app import astra_for, fixture_runtime
from rammp_adl.contracts import strict_loads

ROOT = Path(__file__).resolve().parents[1]


class CloudExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_astra_gateway_executes_and_replans_with_current_world_identities(self):
        runtime = fixture_runtime(ROOT / "examples/cabinet.context.json", root=ROOT,
                                  failures={"c6": ["model_mismatch"]})
        plans = [strict_loads((ROOT / f"examples/{name}.plan.json").read_bytes())
                 for name in ("cabinet", "cabinet-reobserve", "cabinet-resume")]
        requests = []
        class Transport:
            async def create(self, **request):
                requests.append(copy.deepcopy(request))
                context = json.loads(request["input"][0]["content"][0]["text"])["world_context"]
                plan = plans.pop(0)
                for key in ("task_id", "snapshot_id", "execution_epoch"):
                    plan[key] = context[key]
                return {"status": "completed", "model": "gpt-6-astra", "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": json.dumps({
                        "result": {"status": "OK", "plan": plan}})}]}]}
        reasoner = astra_for(runtime, transport=Transport())
        result = await runtime.executor.run_task("Open the cabinet door", reasoner)
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual(result.task_replans, 2)
        self.assertEqual(len(requests), 3)
        self.assertTrue(runtime.world.goal_satisfied())
        self.assertTrue(runtime.executor.safety.held_verified)
        self.assertEqual(runtime.executor.resources.owners, {})
        self.assertEqual(plans, [])

    async def test_invalid_cloud_plan_never_executes_a_valid_prefix(self):
        runtime = fixture_runtime(ROOT / "examples/cabinet.context.json", root=ROOT)
        plan = strict_loads((ROOT / "examples/cabinet.plan.json").read_bytes())
        # Schema-valid unknown dependency: local whole-plan validation must reject it.
        plan["edges"][-1]["from"] = "missing-node"
        class Transport:
            async def create(self, **request):
                context = json.loads(request["input"][0]["content"][0]["text"])["world_context"]
                plan["snapshot_id"] = context["snapshot_id"]
                plan["execution_epoch"] = context["execution_epoch"]
                return {"status": "completed", "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": json.dumps({"result": {"status": "OK", "plan": plan}})}]}]}
        result = await runtime.executor.run_task("Open the cabinet", astra_for(runtime, transport=Transport()))
        self.assertNotEqual(result.status, "succeeded")
        self.assertEqual(runtime.backend.events, [])
        self.assertTrue(runtime.executor.safety.held_verified)
