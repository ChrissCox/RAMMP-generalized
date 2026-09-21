"""Context-bound cloud proposals and bounded feedback; no live provider or robot."""
import copy
import json
from pathlib import Path
import unittest

from rammp_adl.app import astra_for, fixture_runtime
from rammp_adl.contracts import ContractError, strict_loads, validate_schema
from rammp_adl.reasoning import AstraReasoner


ROOT = Path(__file__).resolve().parents[1]


def plan_named(name="cabinet"):
    return strict_loads((ROOT / "examples" / f"{name}.plan.json").read_bytes())


def provider_response(plan):
    return {"status": "completed", "model": "gpt-6-astra", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": json.dumps(
            {"result": {"status": "OK", "plan": plan}})}]}]}


def request_context(request):
    return json.loads(request["input"][0]["content"][0]["text"])


def bind_request(plan, context):
    plan = copy.deepcopy(plan)
    for key in ("task_id", "snapshot_id", "execution_epoch"):
        plan[key] = context[key]
    return plan


class PlanningFeedbackTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self, **kwargs):
        return fixture_runtime(ROOT / "examples/cabinet.context.json", root=ROOT, **kwargs)

    async def test_wrong_safety_profile_is_not_a_proposal_and_catalog_stays_unchanged(self):
        runtime = self.runtime()
        catalog = runtime.catalog
        context = runtime.world.snapshot().context
        original_library = copy.deepcopy(catalog.library)
        original_skills = copy.deepcopy(catalog.skills)
        original_schema = catalog.plan_schema()
        original_response_schema = catalog.response_schema()
        good = bind_request(plan_named(), context)
        wrong = copy.deepcopy(good)
        # A real but incompatible profile caused the observed repeated admission failures.
        wrong["nodes"][5]["args"]["profile_id"] = "sim_gripper"
        validate_schema(wrong, original_schema)
        bound_schema = catalog.plan_schema(context["available_skills"], context=context)
        validate_schema(good, bound_schema)
        with self.assertRaises(ContractError):
            validate_schema(wrong, bound_schema)

        requests = []
        class Transport:
            async def create(self, **request):
                requests.append(copy.deepcopy(request))
                return provider_response(wrong)

        reasoner = AstraReasoner(catalog, {"max_plan_regenerations": 0}, transport=Transport(),
                                hold_assertion=lambda: True,
                                epoch_getter=lambda task: context["execution_epoch"])
        result = await reasoner.generate_plan(context, task_text="Open the cabinet door")
        self.assertEqual(result.status, "INVALID_OUTPUT", result.detail)
        self.assertIsNone(result.plan)
        self.assertEqual(len(requests), 1)
        provider_schema = requests[0]["text"]["format"]["schema"]
        validate_schema({"result": {"status": "OK", "plan": good}}, provider_schema)
        with self.assertRaises(ContractError):
            validate_schema({"result": {"status": "OK", "plan": wrong}}, provider_schema)
        self.assertEqual(runtime.backend.events, [])
        self.assertEqual(catalog.library, original_library)
        self.assertEqual(catalog.skills, original_skills)
        self.assertEqual(catalog.plan_schema(), original_schema)
        self.assertEqual(catalog.response_schema(), original_response_schema)

    async def test_missing_profiles_keep_observation_but_shortcircuit_motion_only_context(self):
        runtime = self.runtime()
        context = copy.deepcopy(runtime.world.snapshot().context)
        context["profiles"] = []
        context["available_skills"] = ["observe", "move_to_pose"]
        original_context = copy.deepcopy(context)
        schema = runtime.catalog.plan_schema(context["available_skills"], context=context)
        observe_only = bind_request(plan_named(), context)
        observe_only["nodes"] = observe_only["nodes"][:1]
        observe_only["edges"] = []
        validate_schema(observe_only, schema)
        # Even a familiar arm skill is unavailable when it has no compatible local profile.
        motion_only = copy.deepcopy(observe_only)
        motion_only["nodes"] = [plan_named()["nodes"][1]]
        with self.assertRaises(ContractError):
            validate_schema(motion_only, schema)
        self.assertEqual(context, original_context)

        context["available_skills"] = ["move_to_pose"]
        requests = []
        class Transport:
            async def create(self, **request):
                requests.append(request)
                raise AssertionError("profile-less motion must not reach the provider")

        result = await astra_for(runtime, transport=Transport()).generate_plan(context)
        self.assertEqual(result.status, "NEED_CAPABILITY", result.detail)
        self.assertIsNone(result.plan)
        self.assertEqual(result.provider_requests, 0)
        self.assertEqual(requests, [])
        self.assertEqual(runtime.backend.events, [])

    async def test_rejected_plan_feedback_reaches_next_request_before_any_dispatch(self):
        runtime = self.runtime()
        requests = []
        test = self
        class Transport:
            async def create(self, **request):
                content = request_context(request)
                requests.append(content)
                test.assertEqual(runtime.backend.events, [])
                test.assertTrue(runtime.executor.safety.held_verified)
                plan = bind_request(plan_named(), content["world_context"])
                if len(requests) == 1:
                    test.assertIsNone(content["feedback"])
                    plan["edges"][-1]["from"] = "missing-node"
                else:
                    rejected = [event for event in runtime.trace.events
                                if event["event"] == "plan_rejected"]
                    test.assertEqual(len(rejected), 1)
                    previous = content["feedback"]["previous_attempt"]
                    test.assertEqual(previous["status"], "rejected")
                    test.assertEqual(previous["reason"], rejected[0]["detail"][:512])
                    test.assertIsNone(previous["last_failed_node"])
                    test.assertGreater(content["world_context"]["execution_epoch"],
                                       requests[0]["world_context"]["execution_epoch"])
                return provider_response(plan)

        result = await runtime.executor.run_task("Open the cabinet door", astra_for(runtime, transport=Transport()))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual(result.task_replans, 1)
        self.assertEqual(len(requests), 2)
        self.assertTrue(runtime.world.goal_satisfied())
        self.assertEqual(runtime.executor.resources.owners, {})

    async def test_replan_exhaustion_preserves_the_local_rejection_cause(self):
        runtime = self.runtime()
        requests = []
        class Transport:
            async def create(self, **request):
                content = request_context(request)
                requests.append(content)
                plan = bind_request(plan_named(), content["world_context"])
                plan["edges"][-1]["from"] = "missing-node"
                return provider_response(plan)

        result = await runtime.executor.run_task("Open the cabinet door", astra_for(runtime, transport=Transport()),
                                                 max_replans=1)
        rejections = [event["detail"] for event in runtime.trace.events if event["event"] == "plan_rejected"]
        self.assertEqual(result.status, "incomplete", result.to_dict())
        self.assertEqual(result.task_replans, 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(rejections), 2)
        self.assertIn("replan budget exhausted", result.reason.lower())
        self.assertIn(rejections[-1], result.reason)
        self.assertEqual(requests[-1]["feedback"]["previous_attempt"]["reason"], rejections[0][:512])
        self.assertEqual(runtime.backend.events, [])
        self.assertTrue(runtime.executor.safety.held_verified)

    async def test_partial_failure_feedback_clears_after_observation_recovery(self):
        runtime = self.runtime(failures={"c6": ["model_mismatch"]})
        plans = [plan_named(name) for name in ("cabinet", "cabinet-reobserve", "cabinet-resume")]
        requests = []
        class Transport:
            async def create(self, **request):
                content = request_context(request)
                requests.append(content)
                return provider_response(bind_request(plans.pop(0), content["world_context"]))

        result = await runtime.executor.run_task("Open the cabinet door", astra_for(runtime, transport=Transport()))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual(len(requests), 3)
        self.assertIsNone(requests[0]["feedback"])
        failed = requests[1]["feedback"]["previous_attempt"]
        self.assertEqual(failed["status"], "incomplete")
        self.assertEqual(failed["last_failed_node"]["node_id"], "c6")
        self.assertEqual(failed["last_failed_node"]["skill"], "follow_constraint")
        self.assertEqual(failed["last_failed_node"]["failure_code"], "model_mismatch")
        self.assertLessEqual(len(failed["reason"]), 512)
        self.assertLessEqual(len(failed["last_failed_node"]["detail"]), 512)
        observed = requests[2]["feedback"]["previous_attempt"]
        self.assertEqual(observed["status"], "incomplete")
        self.assertIsNone(observed["last_failed_node"])
        self.assertGreater(requests[2]["world_context"]["execution_epoch"],
                           requests[1]["world_context"]["execution_epoch"])
        self.assertTrue(runtime.world.goal_satisfied())


if __name__ == "__main__":
    unittest.main()
