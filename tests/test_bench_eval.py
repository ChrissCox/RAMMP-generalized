"""The bench score is what an automatic research loop optimises; it must say what a run did and no more."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_eval", ROOT/"tools/bench_eval.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def node(skill, status="succeeded", **extra):
    return {"node_id": skill, "skill": skill, "status": status, **extra}


def grasp_move(status="succeeded", **extra):
    return node("move_to_pose", status, pose_role="grasp", **extra)


GOAL = {"predicate": "constraint_goal_verified", "args": {}}


class ScoreTests(unittest.TestCase):
    def test_a_declined_task_scores_nothing_and_a_full_run_scores_everything(self):
        self.assertEqual(bench.score_run({"status": "incomplete", "goal": None, "nodes": []})["score"], 0.)
        full = {"status": "succeeded", "goal": GOAL, "nodes": [node("observe"), node("move_to_pose"), node("set_gripper"),
                                                               grasp_move(), node("grasp"), node("follow_constraint"), node("release")]}
        self.assertEqual(bench.score_run(full)["score"], 100.)

    def test_progress_is_credited_by_stage_and_by_how_far_the_part_was_followed(self):
        standoff = {"status": "incomplete", "goal": GOAL, "nodes": [node("move_to_pose"), node("move_to_pose"), grasp_move("failed", failure_code="planning_failed", detail="x")]}
        scored = bench.score_run(standoff)
        self.assertEqual(scored["score"], 35.)
        self.assertEqual(scored["first_failure"]["failure_code"], "planning_failed")
        partway = {"status": "incomplete", "goal": GOAL, "nodes": [node("move_to_pose"), grasp_move(), node("grasp"),
                                                                   node("follow_constraint", "failed", failure_code="model_mismatch")]}
        scored = bench.score_run(partway, {"target": 1.5, "achieved": .5})
        self.assertEqual(scored["followed_fraction"], .333)
        self.assertEqual(scored["score"], 75.)

    def test_a_run_the_supervisor_ended_is_worth_half(self):
        faulted = {"status": "safety_fault", "goal": GOAL, "nodes": [node("move_to_pose"), node("move_to_pose", "cancelled")]}
        self.assertEqual(bench.score_run(faulted)["score"], 17.5)
        self.assertTrue(bench.score_run(faulted)["safety_fault"])


class GateTests(unittest.TestCase):
    def setUp(self):
        # The operator's real unattended window and halt must not leak into these tests.
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for name in ("UNATTENDED", "HALTED"):
            patcher = patch.object(bench, name, Path(folder.name)/name.lower())
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_changed_safety_file_or_a_missing_pin_is_named(self):
        with tempfile.TemporaryDirectory() as folder:
            pin = Path(folder)/"frozen.json"
            with patch.object(bench, "PIN", pin):
                self.assertIn("no operator pin", bench.frozen_problems()[0])
                pin.write_text(json.dumps(bench.digests()))
                self.assertEqual(bench.frozen_problems(), [])
                pin.write_text(json.dumps({**bench.digests(), "rammp_adl/safety.py": "0"*64}))
                self.assertEqual(bench.frozen_problems(), ["rammp_adl/safety.py differs from the operator's pin"])

    def test_no_go_means_no_motion_and_a_stale_go_is_not_a_go(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(bench, "BENCH", Path(folder)):
            (Path(folder)/"GO").write_text("1.0")                                  # written long before this run
            self.assertFalse(bench.wait_for_go(1.5))
            bench.go(None)
            self.assertTrue(bench.wait_for_go(3.))
            self.assertFalse((Path(folder)/"GO").exists())                         # used once

    def test_a_standing_go_expires_and_a_halt_holds_until_the_operator_returns(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        with patch.object(bench, "BENCH", root), patch.object(bench, "UNATTENDED", root/"unattended-until"), \
                patch.object(bench, "HALTED", root/"halted"):
            self.assertFalse(bench.unattended_active())
            bench.unattended(type("A", (), {"hours": 1.})())
            self.assertTrue(bench.wait_for_go(.1))                                  # no per-run GO needed
            bench.end_unattended("the door moved")
            self.assertFalse(bench.unattended_active())
            self.assertIn("the door moved", bench.halted())
            self.assertFalse(bench.wait_for_go(1.2))                                # a person is needed again
            bench.attended(None)
            self.assertIsNone(bench.halted())
            bench.end_unattended("nothing to end")                                  # attended runs never halt
            self.assertIsNone(bench.halted())
            (root/"unattended-until").write_text("1.0 long ago")
            self.assertFalse(bench.unattended_active())
            self.assertIn("expired", bench.halted())
            with self.assertRaises(SystemExit):
                bench.unattended(type("A", (), {"hours": 48.})())

    def test_the_node_roots_are_read_from_running_processes_only(self):
        roots = bench.node_roots()
        self.assertIsInstance(roots, set)
        self.assertTrue(all(isinstance(root, str) and root.startswith("/") for root in roots))


if __name__ == "__main__":
    unittest.main()
