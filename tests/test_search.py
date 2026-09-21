"""Searching for the task's target: look moves from the tool pose and the intake search loop."""
import asyncio
import json
import unittest
from pathlib import Path

import numpy as np

from rammp_adl.contracts import Catalog
from rammp_adl.handlers import BackendFailure, ExecutionContext
from rammp_adl.intake import IntakeError, draft_context, search_until_visible
from rammp_adl.motion.collision_guard import EffortGuard, GuardSet
from rammp_adl.motion.kinematics import UrdfChain, quaternion_matrix
from rammp_adl.sheppy_backend import SheppyArmBackend
from rammp_adl.world import WorldModel

from test_follow_constraint import PROFILES, TrackingClient

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2"
BENCH = json.loads((ROOT/"config/sheppy-bench.context.json").read_text())


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class LookTargetTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(ROOT)
        self.chain = UrdfChain.from_path(BUNDLE/"arm-gripper-locked.urdf")
        self.client = TrackingClient(knuckle=.01)
        context = draft_context(BENCH, [], task_id="task-search", camera_id="wrist_d405")
        self.world = WorldModel(context, self.catalog, trust_initial=False, max_evidence_age_s=600.)
        self.backend = SheppyArmBackend(self.catalog, self.world, client=self.client, profiles=PROFILES, chain=self.chain,
                                        guard_factory=lambda **o: GuardSet(effort=EffortGuard(o.get("touch_nm", 3.))),
                                        speed_scales={"transit": 2.5, "contact": 4.})
        self.position, self.orientation = self.backend._tool_pose((0.,)*7)
        self.view = quaternion_matrix(self.orientation)[:, 2]

    def test_hints_turn_or_move_the_camera_as_named(self):
        left_p, left_q = self.backend.look_target("left", (0.,)*7)
        np.testing.assert_allclose(left_p, self.position)
        turned = quaternion_matrix(left_q)[:, 2]
        expected = np.array([[np.cos(.5), -np.sin(.5), 0.], [np.sin(.5), np.cos(.5), 0.], [0., 0., 1.]]) @ self.view
        np.testing.assert_allclose(turned, expected, atol=1e-9)                     # a pan about the base vertical
        right = quaternion_matrix(self.backend.look_target("right", (0.,)*7)[1])[:, 2]
        self.assertAlmostEqual(right[2], self.view[2], places=9)
        # Tilting needs a view that is not already vertical: find a configuration looking sideways.
        sideways = next(q for q in ((0., .8, 0., 1.2, 0., .6, 0.), (0., 1.2, 0., 1.0, 0., .3, 0.), (.3, .6, .2, 1.4, 0., .9, .1))
                        if abs(quaternion_matrix(self.backend._tool_pose(q)[1])[:, 2][2]) < .9)
        view = quaternion_matrix(self.backend._tool_pose(sideways)[1])[:, 2]
        up = quaternion_matrix(self.backend.look_target("up", sideways)[1])[:, 2]
        down = quaternion_matrix(self.backend.look_target("down", sideways)[1])[:, 2]
        self.assertGreater(up[2], view[2])
        self.assertLess(down[2], view[2])
        back_p, back_q = self.backend.look_target("back", (0.,)*7)
        np.testing.assert_allclose(np.subtract(back_p, self.position), -self.view*.10, atol=1e-9)
        np.testing.assert_allclose(back_q, self.orientation, atol=1e-9)
        closer_p, _ = self.backend.look_target("closer", (0.,)*7)
        np.testing.assert_allclose(np.subtract(closer_p, self.position), self.view*.10, atol=1e-9)
        with self.assertRaises(BackendFailure):
            self.backend.look_target("around", (0.,)*7)

    def test_a_look_is_one_guarded_transit(self):
        context = ExecutionContext(task_id="task-search", node_id="search-0", execution_epoch=1)
        move = asyncio.run(self.backend.look("left", context))
        self.assertEqual(move["receipt_status"], "succeeded")
        self.assertEqual(len(self.client.sent), 1)
        self.assertTrue(self.client.sent[0].provenance.endswith("x2.5"))
        self.assertEqual(self.backend.active, set())
        self.client.plan_error = "unreachable"
        with self.assertRaises(BackendFailure) as caught:
            asyncio.run(self.backend.look("up", context))
        self.assertEqual(caught.exception.code, "planning_failed")


class ScriptedScene:
    def __init__(self, verdicts):
        self.verdicts, self.calls = list(verdicts), 0

    async def discover(self, reasoner, context, task_text):
        self.calls += 1
        verdict = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        return {"keyframe": f"kf-{self.calls}", "reason": "requested", "entities": verdict.get("entities", []),
                "descriptors": [], "status": "OK", "search": {"target_visible": verdict["visible"], "search_hint": verdict["hint"],
                                                               "search_note": verdict.get("note", "")}}


class ScriptedBackend:
    def __init__(self, refuse=()):
        self.moves, self.refuse = [], set(refuse)
        self.client = type("C", (), {"stationary": staticmethod(lambda **k: _true())})()

    async def look(self, hint, context, **options):
        if hint in self.refuse:
            raise BackendFailure("planning_failed", "unreachable")
        self.moves.append(hint)
        return {"hint": hint, "receipt_status": "succeeded"}


async def _true():
    return True


class SearchLoopTests(unittest.TestCase):
    CONTEXT = {"task_id": "t", "execution_epoch": 1}

    def test_the_loop_turns_as_told_until_the_target_appears(self):
        scene = ScriptedScene([{"visible": False, "hint": "left", "note": "wall"}, {"visible": False, "hint": "left"},
                               {"visible": True, "hint": "none", "entities": ["door_1"]}])
        backend, notes = ScriptedBackend(), []
        found = asyncio.run(search_until_visible(scene, backend, None, self.CONTEXT, "open the door", log=notes.append))
        self.assertEqual(found["entities"], ["door_1"])
        self.assertEqual(backend.moves, ["left", "left"])
        self.assertEqual([m["hint"] for m in found["viewpoints"]], ["left", "left"])
        self.assertEqual(scene.calls, 3)
        self.assertIn("wall", notes[0])

    def test_no_target_after_the_budget_is_a_need_for_observation(self):
        scene = ScriptedScene([{"visible": False, "hint": "right", "note": "nothing here"}])
        backend = ScriptedBackend()
        with self.assertRaises(IntakeError) as caught:
            asyncio.run(search_until_visible(scene, backend, None, self.CONTEXT, "open the door", max_viewpoints=3))
        self.assertEqual(caught.exception.status, "NEED_OBSERVATION")
        self.assertIn("3 viewpoints", caught.exception.detail)
        self.assertEqual(backend.moves, ["right"]*3)

    def test_a_refused_look_tries_the_opposite_and_repeated_backing_turns_instead(self):
        scene = ScriptedScene([{"visible": False, "hint": "up"}, {"visible": False, "hint": "back"}, {"visible": False, "hint": "back"},
                               {"visible": True, "hint": "none"}])
        backend = ScriptedBackend(refuse={"up"})
        found = asyncio.run(search_until_visible(scene, backend, None, self.CONTEXT, "open the door"))
        self.assertEqual(backend.moves, ["down", "back", "left"])
        self.assertEqual([m["status"] for m in found["viewpoints"]], ["refused: planning_failed", "succeeded", "succeeded", "succeeded"])

    def test_a_visible_target_or_no_hint_needs_no_move(self):
        scene = ScriptedScene([{"visible": True, "hint": "left", "entities": ["cup_1"]}])
        backend = ScriptedBackend()
        found = asyncio.run(search_until_visible(scene, backend, None, self.CONTEXT, "pick up the cup"))
        self.assertEqual(found["entities"], ["cup_1"])
        self.assertEqual(backend.moves, [])
        scene = ScriptedScene([{"visible": False, "hint": "none"}])
        found = asyncio.run(search_until_visible(scene, backend, None, self.CONTEXT, "describe"))
        self.assertEqual(backend.moves, [])


if __name__ == "__main__":
    unittest.main()
