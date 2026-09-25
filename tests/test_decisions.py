"""Jev's fast typed decisions: goals picked from what the scene allows, stop-or-replan after a failure, Astra on any doubt."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from rammp_adl.app import fixture_runtime
from rammp_adl.contracts import Catalog, strict_loads
from rammp_adl.decisions import NONE, JevDecider, decide_after_failure, goal_options, load_config
from rammp_adl.intake import goal_candidates, normalize_task

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ["observe", "move_to_pose", "set_gripper", "grasp", "release", "follow_constraint"]


def load(name, kind="plan"):
    return strict_loads((ROOT/"examples"/f"{name}.{kind}.json").read_bytes())


def scripted(answers):
    """A transport that answers each question from a script: {question: (choice, confidence)} or an exception."""
    seen = []

    async def transport(url, headers, body, timeout_s):
        seen.append(body)
        question = next(iter(body["questions"]))
        answer = answers[question]
        if isinstance(answer, Exception):
            raise answer
        choice, confidence = answer
        if callable(choice):
            choice = choice(body["questions"][question]["criteria"])
        return {"model": "jev-1.13.0", "answers": {question: {"type": "choice", "choice": choice, "confidence": confidence,
                                                              "probabilities": {choice: confidence}}},
                "usage": {"input_tokens": 120, "output_tokens": 10}}
    transport.seen = seen
    return transport


def decider(answers):
    transport = scripted(answers)
    return JevDecider(load_config(ROOT), environ={}, transport=transport), transport


class Astra:
    def __init__(self, goal=None):
        self.calls, self.goal = [], goal

    async def normalize_goal(self, context, task_text, *, predicates):
        self.calls.append(task_text)
        return SimpleNamespace(status="OK", goal=self.goal, detail="astra")


def door_goal(criteria):
    return next(key for key in criteria if key.startswith("move_") and key.endswith("_fully"))


class DeciderTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_pick_is_parsed_and_counted_and_a_failure_is_a_decision_with_no_choice(self):
        jev, transport = decider({"q": ("b", .91)})
        decision = await jev.choose("state", question="q", instructions="pick", options={"a": "first", "b": "second"})
        self.assertEqual((decision.choice, decision.confidence, jev.calls, jev.input_tokens), ("b", .91, 1, 120))
        self.assertEqual(transport.seen[0]["questions"]["q"]["criteria"], {"a": "first", "b": "second"})
        self.assertEqual(transport.seen[0]["model"], "jev-latest")
        broken, _ = decider({"q": TimeoutError("slow")})
        decision = await broken.choose("state", question="q", instructions="pick", options={"a": "first"})
        self.assertIsNone(decision.choice)
        self.assertIn("TimeoutError", decision.error)
        stray, _ = decider({"q": ("zzz", .99)})                                       # not an option: no decision
        self.assertIsNone((await stray.choose("s", question="q", instructions="pick", options={"a": "x"})).choice)

    def test_without_a_key_it_is_not_available(self):
        self.assertFalse(JevDecider(load_config(ROOT), environ={}).available)
        self.assertTrue(JevDecider(load_config(ROOT), environ={"TYPESAFE_API_KEY": "k"}).available)


class GoalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.catalog = Catalog(ROOT)
        self.context = load("cabinet", "context")
        self.candidates = goal_candidates(self.catalog, SKILLS)

    def test_every_option_is_a_goal_the_scene_can_bind_with_targets_from_the_scene(self):
        options = goal_options(self.context, self.candidates)
        self.assertIn(NONE, options)
        constraint = self.context["constraints"][0]
        fully = next(goal for key, (_, goal) in options.items() if key.endswith("_fully"))
        self.assertEqual(fully["args"]["target_value"], round(float(constraint["maximum"]), 3))
        self.assertEqual(fully["args"]["constraint_id"], constraint["constraint_id"])
        from rammp_adl.intake import validate_goal
        for key, (description, goal) in options.items():
            if goal is not None:
                validate_goal(self.catalog, goal, self.context, self.candidates)   # none of them is refused locally

    async def test_a_confident_pick_is_the_goal_and_astra_is_not_asked(self):
        jev, _ = decider({"goal": (door_goal, .93)})
        astra = Astra()
        goal = await normalize_task(astra, self.catalog, self.context, "open the cabinet", available_skills=SKILLS, decider=jev)
        self.assertEqual(goal["predicate"], "constraint_goal_verified")
        self.assertEqual(astra.calls, [])

    async def test_doubt_none_of_these_or_a_failure_asks_astra_as_before(self):
        astra_goal = {"predicate": "holding", "args": {"entity_id": "cabinet_handle_1"}}
        for answers in ({"goal": (door_goal, .6)}, {"goal": (NONE, .97)}, {"goal": ConnectionError("down")}):
            jev, _ = decider(answers)
            astra = Astra(astra_goal)
            goal = await normalize_task(astra, self.catalog, self.context, "grab the handle", available_skills=SKILLS,
                                        decider=jev)
            self.assertEqual(goal, astra_goal)
            self.assertEqual(len(astra.calls), 1)


class AfterFailureTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self, **kwargs):
        return fixture_runtime(load("cabinet", "context"), root=ROOT, **kwargs)

    async def test_the_judgment_is_stop_only_when_confident(self):
        for answer, stops in ((("stop", .9), True), (("stop", .7), False), (("replan", .95), False)):
            jev, _ = decider({"after_failure": answer})
            stop, _ = await decide_after_failure(jev, task_text="open", skill="follow_constraint", failure_code="model_mismatch",
                                                 detail="stopped at 0.3 of 1.57 rad", attempt=0, threshold=.85)
            self.assertEqual(stop, stops)

    def planner(self, plans):
        class Reasoner:
            calls = 0

            async def generate_plan(self, context, **kwargs):
                Reasoner.calls += 1
                plan = copy.deepcopy(plans.pop(0))
                for key in ("task_id", "snapshot_id", "execution_epoch"):
                    plan[key] = context[key]
                return SimpleNamespace(status="OK", plan=plan, detail="explicit test response")
        return Reasoner

    async def test_a_confident_stop_ends_the_task_without_paying_for_a_replan(self):
        runtime = self.runtime(failures={"c6": ["model_mismatch"]})
        runtime.executor.decider, _ = decider({"after_failure": ("stop", .95)})
        Reasoner = self.planner([load("cabinet-reobserve"), load("cabinet-resume")])
        result = await runtime.executor.run_task("Open the cabinet", Reasoner(), initial_plan=load("cabinet"))
        self.assertEqual(result.status, "incomplete")
        self.assertIn("stopped without a replan", result.reason)
        self.assertEqual(Reasoner.calls, 0)
        decisions = [e for e in runtime.trace.events if e["event"] == "fast_decision"]
        self.assertEqual((decisions[0]["choice"], decisions[0]["confidence"]), ("stop", .95))

    async def test_replan_or_doubt_recovers_exactly_as_without_it(self):
        for answer in (("replan", .9), ("stop", .6)):
            runtime = self.runtime(failures={"c6": ["model_mismatch"]})
            runtime.executor.decider, _ = decider({"after_failure": answer})
            Reasoner = self.planner([load("cabinet-reobserve"), load("cabinet-resume")])
            result = await runtime.executor.run_task("Open the cabinet", Reasoner(), initial_plan=load("cabinet"))
            self.assertEqual(result.status, "succeeded", result.to_dict())
            self.assertEqual(Reasoner.calls, 2)


if __name__ == "__main__":
    unittest.main()
