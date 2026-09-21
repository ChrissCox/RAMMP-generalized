"""Task latency includes initial reasoning without requiring a live provider."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from rammp_adl.app import fixture_runtime
from rammp_adl.contracts import strict_loads
from rammp_adl.telemetry import TraceRecorder


ROOT = Path(__file__).resolve().parents[1]


class TaskTimingTests(unittest.TestCase):
    def test_first_motion_uses_task_request_or_legacy_cloud_wait(self):
        cases = {
            "task request includes hold and cloud": ([
                (90.0, "runtime_ready", {}),
                (100.0, "task_request_started", {}),
                (103.0, "wait_started", {"wait_id": "reasoning:0", "reason": "cloud_reasoning"}),
                (150.0, "wait_ended", {"wait_id": "reasoning:0"}),
                (150.0, "task_started", {}),
                (152.0, "motion_started", {}),
            ], 52.0, 47.0),
            "existing trace includes initial cloud": ([
                (100.0, "wait_started", {"wait_id": "reasoning:0", "reason": "cloud_reasoning"}),
                (150.0, "wait_ended", {"wait_id": "reasoning:0"}),
                (150.0, "task_started", {}),
                (152.0, "motion_started", {}),
            ], 52.0, 50.0),
            "direct plan retains task admission start": ([
                (90.0, "runtime_ready", {}),
                (100.0, "task_started", {}),
                (101.0, "wait_started", {"wait_id": "admission", "reason": "plan_validation"}),
                (102.0, "wait_ended", {"wait_id": "admission"}),
                (102.0, "motion_started", {}),
            ], 2.0, 0.0),
            "no executed motion has no latency": ([
                (100.0, "task_request_started", {}),
                (103.0, "wait_started", {"wait_id": "reasoning:0", "reason": "cloud_reasoning"}),
                (150.0, "wait_ended", {"wait_id": "reasoning:0"}),
                (150.0, "task_started", {}),
                (151.0, "plan_rejected", {}),
            ], None, 47.0),
        }
        for name, (events, expected_latency, expected_cloud_wait) in cases.items():
            with self.subTest(name=name):
                now = [0.0]
                trace = TraceRecorder(clock=lambda: now[0])
                for stamp, event, fields in events:
                    now[0] = stamp
                    trace.emit(event, **fields)
                summary = trace.summary()
                self.assertEqual(summary["first_motion_latency_s"], expected_latency)
                self.assertEqual(summary["wait_time_by_reason_s"].get("cloud_reasoning", 0.0),
                                 expected_cloud_wait)
                self.assertEqual(summary["open_waits"], [])


class ExecutorTaskTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_start_precedes_reasoning_and_motion_includes_its_wait(self):
        runtime = fixture_runtime(ROOT / "examples/cabinet.context.json", root=ROOT)
        now = [100.0]
        runtime.trace.clock = lambda: now[0]
        test = self
        class ScriptedReasoner:
            async def generate_plan(self, context, **kwargs):
                events = runtime.trace.events
                test.assertEqual(events[0]["event"], "task_request_started")
                test.assertTrue(any(event["event"] == "wait_started"
                                    and event["reason"] == "cloud_reasoning" for event in events))
                test.assertFalse(any(event["event"] == "motion_started" for event in events))
                test.assertTrue(runtime.executor.safety.held_verified)
                # The fake trace clock accounts for a provider wait without delaying the test.
                now[0] = 150.0
                plan = strict_loads((ROOT / "examples/cabinet.plan.json").read_bytes())
                for key in ("task_id", "snapshot_id", "execution_epoch"):
                    plan[key] = context[key]
                return SimpleNamespace(status="OK", plan=plan, detail="scripted test response")

        result = await runtime.executor.run_task("Open the cabinet door", ScriptedReasoner())
        self.assertEqual(result.status, "succeeded", result.to_dict())
        summary = runtime.trace.summary()
        self.assertEqual(summary["wait_time_by_reason_s"]["cloud_reasoning"], 50.0)
        self.assertEqual(summary["first_motion_latency_s"], 50.0)
        self.assertEqual(sum(event["event"] == "task_request_started" for event in runtime.trace.events), 1)


if __name__ == "__main__":
    unittest.main()
