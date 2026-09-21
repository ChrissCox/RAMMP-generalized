"""Typed-task intake: goal candidates, Astra goal normalization, world goal swap, visibility seeding."""
import asyncio
import copy
import json
import unittest
from pathlib import Path

from rammp_adl.contracts import Catalog, ContractError
from rammp_adl.intake import (IntakeError, draft_context, goal_candidates, new_task_id, normalize_task,
                              seed_visibility, validate_goal)
from rammp_adl.reasoning import AstraReasoner, _provider_schema
from rammp_adl.world import WorldModel

ROOT = Path(__file__).resolve().parents[1]
BENCH = json.loads((ROOT/"config/sheppy-bench.context.json").read_text())
DESCRIPTORS = [{"entity_id": "bottle_1", "label": "water bottle", "pose_roles": ["grasp", "pregrasp", "staging"]}]
AVAILABLE = ["observe", "move_to_pose", "set_gripper", "grasp"]


def completed(payload):
    return {"status": "completed", "model": "gpt-6-astra",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(payload)}]}]}


class ScriptedTransport:
    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    async def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        return copy.deepcopy(self.outcomes.pop(0))

    async def close(self):
        pass


class FakeScene:
    camera_id = "wrist_d405"

    def __init__(self, visible):
        self.visible = visible

    def visible_entities(self, now, *, max_age_s=2.):
        return {name: {"visible": name in self.visible, "capture_id": "cap", "report_digest": "d"} for name in ("bottle_1",)}


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(ROOT)
        self.context = draft_context(BENCH, DESCRIPTORS, task_id="task-test", camera_id="wrist_d405")
        self.world = WorldModel(self.context, self.catalog, trust_initial=False, max_evidence_age_s=120.)

    def reasoner(self, *outcomes):
        return AstraReasoner(self.catalog, transport=ScriptedTransport(*outcomes),
                             hold_assertion=lambda: True, epoch_getter=lambda _task: 1)

    def test_goal_candidates_are_postconditions_of_available_skills_in_catalog_order(self):
        self.assertEqual(goal_candidates(self.catalog, AVAILABLE), ("observation_valid", "at_pose", "aperture_reached", "holding"))
        self.assertEqual(goal_candidates(self.catalog, []), ())

    def test_draft_context_declares_scene_entities_without_trusted_facts(self):
        ids = [entity["entity_id"] for entity in self.context["entities"]]
        self.assertEqual(ids, ["robot", "bottle_1"])
        self.assertEqual(self.context["entities"][1]["pose_roles"], ["grasp", "pregrasp", "staging"])
        self.assertEqual(self.context["task_id"], "task-test")
        self.assertEqual(self.context["goal"], BENCH["goal"])
        snapshot = self.world.snapshot()
        self.assertEqual(snapshot.fact("held_state", {"robot_id": "robot"}), "unknown")
        self.assertEqual(snapshot.fact("entity_exists", {"entity_id": "bottle_1"}), "unknown")
        self.assertTrue(new_task_id().startswith("task-"))

    def test_goal_schema_binds_ids_to_the_scene_and_lowers_to_the_provider(self):
        reasoner = self.reasoner()
        schema = reasoner.goal_schema(("holding", "at_pose", "constraint_goal_verified", "aperture_reached"), context=self.context)
        variants = schema["properties"]["result"]["anyOf"][0]["properties"]["goal"]["anyOf"]
        self.assertEqual([v["properties"]["predicate"]["enum"][0] for v in variants], ["holding", "at_pose", "aperture_reached"])
        self.assertEqual(variants[0]["properties"]["args"]["properties"]["entity_id"]["enum"], ["robot", "bottle_1"])
        _provider_schema(schema)

    def test_normalize_task_accepts_a_bound_goal_and_records_one_request(self):
        reasoner = self.reasoner(completed({"result": {"status": "OK", "rationale": "pick up",
                                                       "goal": {"predicate": "holding", "args": {"entity_id": "bottle_1"}}}}))
        goal = asyncio.run(normalize_task(reasoner, self.catalog, self.context, "pick up the water bottle", available_skills=AVAILABLE))
        self.assertEqual(goal, {"predicate": "holding", "args": {"entity_id": "bottle_1"}})
        request = reasoner.transport.requests[0]
        self.assertEqual(request["text"]["format"]["name"], "adl_goal")
        self.assertIn("Turn the request into one goal", request["instructions"])
        self.assertEqual(request["reasoning"], {"effort": "low"})                 # binding a goal is a fast decision
        content = json.loads(request["input"][0]["content"][0]["text"])
        self.assertEqual(content["task_text"], "pick up the water bottle")
        self.assertEqual(content["offered_predicates"], ["observation_valid", "at_pose", "aperture_reached", "holding"])
        self.assertEqual(reasoner._budgets["task-test"].requests, 1)
        self.assertIsNone(reasoner._budgets["task-test"].goal)

    def test_declined_and_invalid_outcomes(self):
        reasoner = self.reasoner(completed({"result": {"status": "UNSUPPORTED", "detail": "no door here"}}))
        with self.assertRaises(IntakeError) as caught:
            asyncio.run(normalize_task(reasoner, self.catalog, self.context, "open the cabinet door", available_skills=AVAILABLE))
        self.assertEqual(caught.exception.status, "UNSUPPORTED")
        reasoner = self.reasoner(completed({"result": {"status": "OK", "rationale": "x",
                                                       "goal": {"predicate": "released", "args": {"entity_id": "bottle_1", "support_id": "robot"}}}}),
                                 completed({"result": {"status": "OK", "rationale": "x",
                                                       "goal": {"predicate": "at_pose", "args": {"entity_id": "bottle_1", "pose_role": "grasp"}}}}))
        goal = asyncio.run(normalize_task(reasoner, self.catalog, self.context, "go to the bottle", available_skills=AVAILABLE))
        self.assertEqual(goal["predicate"], "at_pose")
        self.assertEqual(reasoner._budgets["task-test"].requests, 2)
        with self.assertRaises(IntakeError) as caught:
            asyncio.run(normalize_task(self.reasoner(), self.catalog, self.context, "anything", available_skills=[]))
        self.assertEqual(caught.exception.status, "NEED_CAPABILITY")

    def test_local_goal_validation_rejects_unknown_ids_and_unoffered_predicates(self):
        candidates = goal_candidates(self.catalog, AVAILABLE)
        with self.assertRaises(IntakeError) as caught:
            validate_goal(self.catalog, {"predicate": "holding", "args": {"entity_id": "cup_9"}}, self.context, candidates)
        self.assertEqual(caught.exception.status, "INVALID_OUTPUT")
        with self.assertRaises(IntakeError) as caught:
            validate_goal(self.catalog, {"predicate": "released", "args": {"entity_id": "bottle_1", "support_id": "robot"}}, self.context, candidates)
        self.assertEqual(caught.exception.status, "UNSUPPORTED")
        with self.assertRaises(IntakeError):
            validate_goal(self.catalog, {"predicate": "holding"}, self.context, candidates)

    def test_world_replaces_the_goal_only_before_execution(self):
        before = self.world.snapshot()
        goal = self.world.replace_goal({"predicate": "holding", "args": {"entity_id": "bottle_1"}})
        after = self.world.snapshot()
        self.assertEqual(after.context["goal"], goal)
        self.assertEqual(after.revision, before.revision+1)
        with self.assertRaises(ContractError):
            self.world.replace_goal({"predicate": "no_such_predicate", "args": {}})
        seed_visibility(self.world, FakeScene({"bottle_1"}), now=self.world.clock())    # a committed operation is fine
        self.world.replace_goal({"predicate": "at_pose", "args": {"entity_id": "bottle_1", "pose_role": "grasp"}})

    def test_visibility_seeds_entity_existence_from_measurement_only(self):
        result = seed_visibility(self.world, FakeScene(set()), now=self.world.clock())
        self.assertEqual(result["visible"], [])
        self.assertEqual(self.world.snapshot().fact("entity_exists", {"entity_id": "bottle_1"}), "unknown")
        result = seed_visibility(self.world, FakeScene({"bottle_1"}), now=self.world.clock())
        self.assertEqual(result["visible"], ["bottle_1"])
        self.assertEqual(self.world.snapshot().fact("entity_exists", {"entity_id": "bottle_1"}), "true")


if __name__ == "__main__":
    unittest.main()
