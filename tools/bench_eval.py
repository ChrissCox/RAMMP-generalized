#!/usr/bin/env python3
"""One scored attempt at the bench task, for an automatic research loop to optimise.

    python tools/bench_eval.py offline            # no motion: tests, design check, live plan-only if the stack is up
    python tools/bench_eval.py hardware           # one attended run on the arm; prints one JSON line with "score"
    python tools/bench_eval.py go                 # operator: the door is closed, I am at the e-stop, run the next one
    python tools/bench_eval.py record-start       # operator: remember the arm's present joints as the start pose
    python tools/bench_eval.py freeze             # operator: pin the safety files an experiment may not change

The hardware tier never moves without a fresh operator GO, refuses to run when
a pinned safety file differs from the operator's pin, and sends only
cuRobo-planned, gated, slowed trajectories: the reset to the start pose goes
through the same client and gates as every skill. A score is evidence about
one run on this bench, not validation of the robot.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH = ROOT/"artifacts/bench"
PIN = Path(os.environ.get("RAMMP_BENCH_PIN", Path.home()/".config/rammp-bench/frozen.json"))
MANIFEST_DIR = Path(os.environ.get("RAMMP_SHEPPY_DIR", "/home/abra/rammp-deployments/december_2026"))
NODE_LOGS = Path.home()/".sheppy/logs/adl"
TASK = os.environ.get("RAMMP_BENCH_TASK", "open the cabinet door in front of you")

# What stands between an experiment and the arm. An experiment that needs one of
# these changed is a conversation with the operator, not a commit.
FROZEN = ("rammp_adl/safety.py", "rammp_adl/motion/collision_guard.py", "rammp_adl/motion/sheppy_client.py",
          "rammp_adl/motion/sheppy_arm.py", "config/sheppy-bench.context.json", "config/imagery-locality.json",
          "config/reasoning.json", "tools/bench_eval.py")
# The node's own limits, read back from the running process before every run: an experiment may slow the
# arm or tighten the free-space touch threshold, never the reverse.
CEILINGS = {"transit_speed_scale": .4, "contact_speed_scale": .25, "touch_nm": 3.}

STAGES = (("intake", 10), ("planned", 10), ("standoff", 15), ("at_handle", 15), ("grasped", 15), ("followed", 30), ("released", 5))


def digests():
    return {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in FROZEN}


def frozen_problems():
    if not PIN.is_file():
        return [f"no operator pin at {PIN}; run `python tools/bench_eval.py freeze` once"]
    pinned = json.loads(PIN.read_text())
    return [f"{path} differs from the operator's pin" for path, digest in digests().items() if pinned.get(path) != digest]


def score_run(result, attempt=None):
    """0 to 100 from what the run measurably did. Pure; the tests pin it.

    result is the task's result JSON; attempt is the constraint record's
    attempt for this task, which carries how far the part was followed.
    """
    nodes = result.get("nodes", [])
    done = [n for n in nodes if n.get("status") == "succeeded"]
    moves = [n for n in done if n.get("skill") == "move_to_pose"]
    reached = {"intake": result.get("goal") is not None, "planned": bool(nodes), "standoff": len(moves) >= 1,
               "at_handle": len(moves) >= 2, "grasped": any(n.get("skill") == "grasp" for n in done),
               "released": any(n.get("skill") == "release" for n in done)}
    fraction = 0.
    if attempt and attempt.get("target"):
        fraction = max(0., min(1., float(attempt.get("achieved", 0.))/float(attempt["target"])))
    if any(n.get("skill") == "follow_constraint" for n in done):
        fraction = 1.
    points = sum(weight for stage, weight in STAGES if reached.get(stage)) + dict(STAGES)["followed"]*fraction
    fault = result.get("status") == "safety_fault"
    if fault:
        points *= .5                                           # a run the supervisor had to end is worth half
    failed = next((n for n in nodes if n.get("status") in ("failed", "cancelled")), None)
    return {"score": round(points, 1), "status": result.get("status"), "reason": result.get("reason", "")[:300],
            "stages": {stage: bool(reached.get(stage)) for stage, _ in STAGES if stage != "followed"},
            "followed_fraction": round(fraction, 3), "safety_fault": fault, "task_replans": result.get("task_replans"),
            "first_failure": None if failed is None else {k: failed.get(k) for k in ("node_id", "skill", "failure_code", "detail")}}


def emit(payload):
    BENCH.mkdir(parents=True, exist_ok=True)
    payload = {"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}
    (BENCH/f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{payload.get('tier', 'run')}.json").write_text(json.dumps(payload, indent=1)+"\n")
    print(json.dumps(payload), flush=True)
    return payload


# -- operator commands ---------------------------------------------------------------
def freeze(_args):
    PIN.parent.mkdir(parents=True, exist_ok=True)
    PIN.write_text(json.dumps(digests(), indent=1)+"\n")
    print(f"pinned {len(FROZEN)} files at {PIN}")


def go(_args):
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH/"GO").write_text(str(time.time()))
    print("GO recorded: the next hardware run may move the arm. It is used once.")


def wait_for_go(timeout_s):
    """A GO written after this call began, consumed on use. No GO, no motion."""
    started, flag = time.time(), BENCH/"GO"
    print(f"waiting for the operator: close the door, stand at the e-stop, then `python tools/bench_eval.py go` "
          f"(up to {timeout_s:.0f} s)", file=sys.stderr, flush=True)
    while time.time()-started < timeout_s:
        try:
            if float(flag.read_text()) >= started-1.:
                flag.unlink()
                return True
        except (OSError, ValueError):
            pass
        time.sleep(1.)
    return False


# -- ROS side ------------------------------------------------------------------------
class Stack:
    """An unarmed-by-default client on its own node, spun in a thread."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from rammp_adl.motion.sheppy_arm import SheppyArmClient
        rclpy.init()
        self.rclpy, self.node = rclpy, Node("rammp_bench_eval")
        self.client = SheppyArmClient(self.node)
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        deadline = time.time()+10.
        while self.client.live_joints() is None and time.time() < deadline:
            time.sleep(.1)

    def joints(self):
        live = self.client.live_joints()
        return None if live is None else live

    def parameters(self, names):
        from rcl_interfaces.srv import GetParameters
        service = self.node.create_client(GetParameters, "/rammp_adl_runtime/get_parameters")
        if not service.wait_for_service(timeout_sec=10.):
            return None
        future = service.call_async(GetParameters.Request(names=list(names)))
        deadline = time.time()+5.
        while not future.done() and time.time() < deadline:
            time.sleep(.05)
        if not future.done():
            return None
        return {name: value.double_value for name, value in zip(names, future.result().values)}

    async def reset(self, start):
        """Open the hand and return to the recorded start joints: planned, gated, slowed, under an effort guard."""
        from rammp_adl.motion.collision_guard import EffortGuard, GuardSet
        from rammp_adl.motion.sheppy_client import scale_trajectory_time
        self.client.arm()
        try:
            opened = await self.client.gripper(0.)
            if not await self.client.settle(timeout_s=10.):
                return {"ok": False, "detail": "the arm is not still; reset refused"}
            live = self.client.live_joints()["position_rad"]
            if max(abs(a-b) for a, b in zip(live, start)) < .02:
                return {"ok": True, "detail": "already at the start pose", "gripper": opened["message"]}
            trajectory, planning = await self.client.plan_to_joints(start)
            receipt = await self.client.execute(scale_trajectory_time(trajectory, 1./CEILINGS["transit_speed_scale"]),
                                                guard=GuardSet(effort=EffortGuard(CEILINGS["touch_nm"])))
            return {"ok": receipt["status"] == "succeeded", "detail": receipt["message"] or receipt["status"],
                    "planning": planning["message"], "gripper": opened["message"]}
        finally:
            self.client.disarm()

    def send_task(self, text, timeout_s):
        from rclpy.action import ActionClient
        from rammp_adl_interfaces.action import ExecuteTask
        action = ActionClient(self.node, ExecuteTask, "/rammp/execute_task")
        if not action.wait_for_server(timeout_sec=30.):
            return None, ["action server /rammp/execute_task did not appear"]
        phases = []

        def feedback(message):
            if not phases or phases[-1] != message.feedback.phase:
                phases.append(message.feedback.phase)
        sent = action.send_goal_async(ExecuteTask.Goal(task_id="", task_text=text, plan_json=""), feedback_callback=feedback)
        deadline = time.time()+timeout_s
        while not sent.done() and time.time() < deadline:
            time.sleep(.1)
        if not sent.done() or not sent.result().accepted:
            return None, phases+["goal not accepted"]
        outcome = sent.result().get_result_async()
        while not outcome.done() and time.time() < deadline:
            time.sleep(.2)
        if not outcome.done():
            sent.result().cancel_goal_async()
            return None, phases+["timed out; cancel requested"]
        return json.loads(outcome.result().result.result_json), phases


def record_start(_args):
    stack = Stack()
    live = stack.joints()
    if live is None:
        raise SystemExit("no /joint_states; is the arm node up?")
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH/"start-joints.json").write_text(json.dumps({"position_rad": list(live["position_rad"]),
                                                       "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1)+"\n")
    print("start pose recorded:", [round(v, 3) for v in live["position_rad"]])
    os._exit(0)


def restart_node(timeout_s=90.):
    before = set(NODE_LOGS.glob("*.log"))
    done = subprocess.run(["sheppy", "restart", "adl"], cwd=MANIFEST_DIR, capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        return False, (done.stdout+done.stderr)[-400:], None
    deadline = time.time()+timeout_s
    while time.time() < deadline:
        fresh = sorted(set(NODE_LOGS.glob("*.log"))-before)
        if fresh:
            text = fresh[-1].read_text(errors="replace")
            if "Traceback" in text:
                return False, text[-800:], fresh[-1]
            if "robot facts bootstrapped" in text:
                return True, "", fresh[-1]
        time.sleep(1.)
    return False, "the node did not report ready", None


def attempt_for(task_id):
    for path in sorted((ROOT/"artifacts/constraints").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for attempt in reversed(record.get("attempts", [])):
            if attempt.get("task_id") == task_id:
                return attempt
    return None


def hardware(args):
    problems = frozen_problems()
    if problems:
        return emit({"tier": "hardware", "score": 0., "status": "refused", "reason": "; ".join(problems), "moved": False})
    start_file = BENCH/"start-joints.json"
    if not start_file.is_file():
        return emit({"tier": "hardware", "score": 0., "status": "refused", "moved": False,
                     "reason": "no start pose; the operator runs `python tools/bench_eval.py record-start` once"})
    if not wait_for_go(args.go_timeout_s):
        return emit({"tier": "hardware", "score": 0., "status": "operator_absent", "moved": False,
                     "reason": "no operator GO; nothing was sent to the arm"})
    ready, detail, log = restart_node()
    if not ready:
        return emit({"tier": "hardware", "score": 0., "status": "node_failed_to_start", "reason": detail, "moved": False})
    stack = Stack()
    limits = stack.parameters(list(CEILINGS))
    if limits is None or any(limits[name] > ceiling+1e-9 for name, ceiling in CEILINGS.items()):
        return emit({"tier": "hardware", "score": 0., "status": "refused", "moved": False,
                     "reason": f"the node's speed or effort limits are looser than the bench allows: {limits}"})
    reset = asyncio.run(stack.reset(json.loads(start_file.read_text())["position_rad"]))
    if not reset["ok"]:
        return emit({"tier": "hardware", "score": 0., "status": "reset_failed", "reason": reset["detail"], "reset": reset})
    time.sleep(args.settle_s)                                  # a still keyframe from the start pose
    began = time.time()
    result, phases = stack.send_task(TASK, args.task_timeout_s)
    if result is None:
        return emit({"tier": "hardware", "score": 0., "status": "no_result", "reason": phases[-1] if phases else "", "phases": phases})
    scored = score_run(result, attempt_for(result.get("task_id")))
    lines = [] if log is None else [line[line.find("]: ")+3:][:240] for line in log.read_text(errors="replace").splitlines()
                                    if "rammp_adl_runtime" in line and "known gap" not in line][-40:]
    return emit({"tier": "hardware", **scored, "duration_s": round(time.time()-began, 1), "phases": phases, "task": TASK,
                 "task_id": result.get("task_id"), "reset": reset, "node_log_tail": lines, "limits": limits})


def offline(args):
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    design = subprocess.run([sys.executable, "tools/check_design.py"], cwd=ROOT, capture_output=True, text=True, env=env)
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT, capture_output=True, text=True, env=env)
    tail = (tests.stderr or tests.stdout).strip().splitlines()[-3:]
    ran = next((int(line.split()[1]) for line in tail if line.startswith("Ran ")), 0)
    passed = tests.returncode == 0
    payload = {"tier": "offline", "design_check": design.returncode == 0, "tests_ran": ran, "tests_passed": passed,
               "frozen_problems": frozen_problems(), "score": (50. if passed else 0.)+(10. if design.returncode == 0 else 0.)}
    if not passed:
        payload["test_tail"] = (tests.stderr or tests.stdout)[-1500:]
    if args.plan:
        payload["plan_only"] = plan_only()
        payload["score"] += 40.*payload["plan_only"].get("fraction", 0.)
    return emit(payload)


def plan_only():
    """Ask the live planner for a small joint-space path and run it through every send gate. Nothing is armed."""
    from rammp_adl.motion.sheppy_client import SheppyClientError, refusal, scale_trajectory_time
    try:
        stack = Stack()
        live = stack.joints()
        if live is None:
            return {"fraction": 0., "detail": "no /joint_states"}
        target = [q+(.15 if index == 5 else 0.) for index, q in enumerate(live["position_rad"])]
        trajectory, planning = asyncio.run(stack.client.plan_to_joints(target, timeout_s=60.))
        why = refusal(scale_trajectory_time(trajectory, 2.5), stack.joints()["position_rad"])
        return {"fraction": 0. if why else 1., "detail": why or planning["message"], "armed": stack.client.motion_enabled}
    except (SheppyClientError, Exception) as exc:              # noqa: BLE001 - a measurement, reported
        return {"fraction": 0., "detail": f"{type(exc).__name__}: {exc}"[:300]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("hardware")
    run.add_argument("--go-timeout-s", type=float, default=900.)
    run.add_argument("--task-timeout-s", type=float, default=900.)
    run.add_argument("--settle-s", type=float, default=4.)
    run.set_defaults(function=hardware)
    off = commands.add_parser("offline")
    off.add_argument("--plan", action="store_true", help="also ask the live planner for a path (no motion)")
    off.set_defaults(function=offline)
    for name, function in (("go", go), ("freeze", freeze), ("record-start", record_start)):
        commands.add_parser(name).set_defaults(function=function)
    args = parser.parse_args()
    args.function(args)
    sys.stdout.flush()
    os._exit(0)                                                # the ROS spin thread is a daemon; do not wait on it


if __name__ == "__main__":
    main()
