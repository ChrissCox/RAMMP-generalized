"""Jev's fast typed decisions: goals picked from what the scene allows, stop-or-replan after a failure, Astra on any doubt."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from rammp_adl.app import fixture_runtime
from rammp_adl.contracts import Catalog, strict_loads
from rammp_adl.decisions import (NONE, JevConstraintReasoner, JevDecider, decide_after_failure, decide_constraint,
                                 decide_plan_variants, goal_options, load_config)
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
        replies = {}
        for question, spec in body["questions"].items():
            answer = answers[question]
            if isinstance(answer, Exception):
                raise answer
            if spec["type"] == "noul":
                replies[question] = {"type": "noul", "noul": answer}
                continue
            choice, confidence = answer
            if callable(choice):
                choice = choice(spec["criteria"])
            replies[question] = {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": {choice: confidence}}
        return {"model": "jev-1.13.0", "answers": replies, "usage": {"input_tokens": 120, "output_tokens": 10}}
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
        description, shut = options[f"move_{constraint['constraint_id']}_shut"]          # a close task has its own outcome
        self.assertEqual(shut["args"]["target_value"], round(float(constraint["minimum"]), 3))
        self.assertTrue(description.startswith("close"))
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

    async def test_closing_is_an_outcome_jev_can_pick_without_astra(self):
        jev, _ = decider({"goal": (lambda criteria: next(k for k in criteria if k.endswith("_shut")), .92)})
        astra = Astra()
        goal = await normalize_task(astra, self.catalog, self.context, "close the cabinet", available_skills=SKILLS, decider=jev)
        self.assertEqual((goal["predicate"], goal["args"]["target_value"]),
                         ("constraint_goal_verified", round(float(self.context["constraints"][0]["minimum"]), 3)))
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

    async def test_stopping_is_not_on_the_menu_the_task_goes_on_while_a_way_on_is_left(self):
        runtime = self.runtime(failures={"c6": ["model_mismatch"]})
        runtime.executor.decider, transport = decider({"recovery": ("stop", .95)})
        Reasoner = self.planner([load("cabinet-reobserve"), load("cabinet-resume")])
        result = await runtime.executor.run_task("Open the cabinet", Reasoner(), initial_plan=load("cabinet"))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertNotIn("stop", transport.seen[0]["questions"]["recovery"]["criteria"])
        self.assertEqual(Reasoner.calls, 0)                                        # an answer off the menu: the cheapest way on
        recovered = [e for e in runtime.trace.events if e["event"] == "local_recovery"]
        self.assertEqual(recovered[0]["choice"], "retry_from_failed")

    async def test_a_confident_local_retry_recovers_without_the_planner(self):
        runtime = self.runtime(failures={"c6": ["model_mismatch"]})
        runtime.executor.decider, transport = decider({"recovery": ("retry_from_failed", .9)})
        Reasoner = self.planner([load("cabinet-reobserve"), load("cabinet-resume")])
        result = await runtime.executor.run_task("Open the cabinet", Reasoner(), initial_plan=load("cabinet"))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual(Reasoner.calls, 0)                                        # no planner request at all
        menu = transport.seen[0]["questions"]["recovery"]["criteria"]
        self.assertEqual(set(menu), {"retry_from_failed", "ask_planner"})        # a follow failure: no look-again offer
        recovered = [e for e in runtime.trace.events if e["event"] == "local_recovery"][0]
        self.assertEqual(recovered["nodes"], ["retry_c6", "retry_c7", "retry_c8"])

    async def test_asking_the_planner_takes_its_plans_and_doubt_takes_the_cheapest_way_on(self):
        for answer, calls in ((("ask_planner", .9), 2), (("retry_from_failed", .6), 0), (("ask_planner", .6), 0)):
            runtime = self.runtime(failures={"c6": ["model_mismatch"]})
            runtime.executor.decider, _ = decider({"recovery": answer})
            Reasoner = self.planner([load("cabinet-reobserve"), load("cabinet-resume")])
            result = await runtime.executor.run_task("Open the cabinet", Reasoner(), initial_plan=load("cabinet"))
            self.assertEqual(result.status, "succeeded", result.to_dict())
            self.assertEqual(Reasoner.calls, calls, answer)

    async def test_a_planner_that_declines_leaves_the_local_ways_on_to_try(self):
        runtime = self.runtime(failures={"c6": ["model_mismatch"]})
        runtime.executor.decider, transport = decider({"recovery": ("ask_planner", .95)})

        class Declining:
            calls = 0

            async def generate_plan(self, context, **kwargs):
                Declining.calls += 1
                return SimpleNamespace(status="NEED_CAPABILITY", plan=None, detail="available skills cannot revise the model")
        result = await runtime.executor.run_task("Open the cabinet", Declining(), initial_plan=load("cabinet"))
        self.assertEqual(result.status, "succeeded", result.to_dict())            # the retry after the decline opened it
        self.assertEqual(Declining.calls, 1)
        self.assertTrue([e for e in runtime.trace.events if e["event"] == "planner_declined"])
        self.assertEqual([e["choice"] for e in runtime.trace.events if e["event"] == "local_recovery"], ["retry_from_failed"])


class ConstraintAndPlanTests(unittest.IsolatedAsyncioTestCase):
    DOOR = {"width_m": .27, "height_m": .45, "handle_offsets_m": {"left": .249, "right": .022, "bottom": .14, "top": .31}}
    BAR = {"major_axis": [0., 0., 1.]}

    async def test_jev_says_what_it_is_the_depth_says_where_the_hinge_is(self):
        jev, transport = decider({"kind": ("revolute", .97), "opening": ("pull", .6), "swing": ("quarter_turn", .88),
                                  "slide": ("medium", .4)})
        proposal, answers = await decide_constraint(jev, label="cabinet door", geometry=self.BAR, door=self.DOOR, threshold=.8)
        self.assertEqual(len(transport.seen), 1)                                   # one call
        self.assertEqual({k: proposal[k] for k in ("kind", "hinge_side", "opening", "range", "contact_effort_nm")},
                         {"kind": "revolute", "hinge_side": "left", "opening": "pull", "range": 1.57, "contact_effort_nm": 5.})
        self.assertAlmostEqual(proposal["door_width_m"], .249)                     # measured, not proposed
        self.assertIn("vertical handle", transport.seen[0]["state"])
        oven = {"width_m": .6, "height_m": .45, "handle_offsets_m": {"left": .3, "right": .3, "bottom": .4, "top": .04}}
        jev, _ = decider({"kind": ("revolute", .96), "opening": ("pull", .96), "swing": ("part_way", .98), "slide": ("short", .3)})
        proposal, _ = await decide_constraint(jev, label="oven door", geometry={"major_axis": [0., 1., 0.]}, door=oven, threshold=.8)
        self.assertEqual((proposal["hinge_side"], proposal["range"]), ("bottom", .79))
        jev, _ = decider({"kind": ("revolute", .96), "opening": ("push", .95), "swing": ("quarter_turn", .9), "slide": ("short", .3)})
        proposal, _ = await decide_constraint(jev, label="cellar door", geometry=self.BAR, door=self.DOOR, threshold=.8)
        self.assertEqual(proposal["opening"], "push")                             # only when Jev is sure of it

    async def test_unsure_of_the_kind_no_door_measured_or_a_failure_leaves_the_proposal_to_astra(self):
        for answers, door in (({"kind": ("revolute", .7), "opening": ("pull", .95), "swing": ("quarter_turn", .9), "slide": ("short", .9)}, self.DOOR),
                              ({"kind": ("revolute", .97), "opening": ("pull", .95), "swing": ("quarter_turn", .9), "slide": ("short", .9)}, None),
                              ({"kind": ("prismatic", .97), "opening": ("slide_right", .3), "swing": ("quarter_turn", .9), "slide": ("long", .9)}, None),
                              ({"kind": ConnectionError("down"), "opening": ("pull", .9), "swing": ("quarter_turn", .9), "slide": ("short", .9)}, self.DOOR)):
            jev, _ = decider(answers)
            astra = SimpleNamespace(calls=[])

            async def propose_constraint(context, entity_id, **kwargs):
                astra.calls.append(entity_id)
                return SimpleNamespace(status="OK", proposal={"kind": "revolute"}, detail="astra")
            astra.propose_constraint = propose_constraint
            proxy = JevConstraintReasoner(astra, jev)
            result = await proxy.propose_constraint({}, "handle_1", label="cabinet door", geometry=self.BAR, door=door)
            self.assertEqual((result.detail, astra.calls), ("astra", ["handle_1"]))

    async def test_a_drawer_is_a_slide_and_the_plan_variants_come_from_one_call(self):
        jev, _ = decider({"kind": ("prismatic", .95), "opening": ("pull", .93), "swing": ("part_way", .3), "slide": ("medium", .9)})
        proposal, _ = await decide_constraint(jev, label="kitchen drawer", geometry={"major_axis": [0., 1., 0.]}, door=None, threshold=.8)
        self.assertEqual((proposal["kind"], proposal["opening"], proposal["range"]), ("prismatic", "pull", .30))
        jev, transport = decider({"release": .9, "retract": .2})
        variants, _ = await decide_plan_variants(jev, "open the drawer and keep holding it", "open the part", threshold=.8)
        self.assertEqual(variants, {"release": True, "retract": False})
        self.assertEqual(set(transport.seen[0]["questions"]), {"release", "retract"})
        unsure, _ = decider({"release": .55, "retract": .45})
        self.assertEqual((await decide_plan_variants(unsure, "open it", "open it", threshold=.8))[0], {"release": True, "retract": True})


class TemplateTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_door_template_is_admitted_and_opens_the_cabinet(self):
        from rammp_adl.plans import template_for_goal
        runtime = fixture_runtime(load("cabinet", "context"), root=ROOT, time_scale=.05)
        context = runtime.world.snapshot().context
        goal = next(g for g in (load("cabinet", "context").get("goal"),) if g) if load("cabinet", "context").get("goal") else None
        goal = goal or {"predicate": "constraint_goal_verified", "args": {"constraint_id": "hinge_1", "target_value": 1., "target_unit": "rad"}}
        plan = template_for_goal(goal, context, runtime.catalog, support_of={"cabinet_handle_1": "cabinet_door_1"})
        self.assertEqual([n["skill"] for n in plan["nodes"]],
                         ["move_to_pose", "set_gripper", "move_to_pose", "grasp", "follow_constraint", "release", "move_to_pose"])
        result = await runtime.executor.run_plan(plan)
        self.assertEqual(result.status, "succeeded", result.to_dict())
        no_release = template_for_goal(goal, context, runtime.catalog, support_of={"cabinet_handle_1": "cabinet_door_1"}, release=False)
        self.assertEqual(no_release["nodes"][-1]["skill"], "follow_constraint")
        self.assertIsNone(template_for_goal({"predicate": "released", "args": {"entity_id": "x", "support_id": "y"}},
                                            context, runtime.catalog))

if __name__ == "__main__":
    unittest.main()
