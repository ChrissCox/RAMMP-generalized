#!/usr/bin/env python3
"""One scored attempt at the bench task, for an automatic research loop to optimise.

    python tools/bench_eval.py offline            # no motion: tests, design check, live plan-only if the stack is up
    python tools/bench_eval.py hardware           # one attended run on the arm; prints one JSON line with "score"
    python tools/bench_eval.py cycle              # open the door, then close it in a second task; one JSON line
    python tools/bench_eval.py go                 # operator: the door is closed, I am at the e-stop, run the next one
    python tools/bench_eval.py record-start       # operator: remember the arm's present joints as the start pose
    python tools/bench_eval.py freeze             # operator: pin the safety files an experiment may not change
    python tools/bench_eval.py unattended --hours 12   # operator: runs go without a GO until then
    python tools/bench_eval.py attended           # operator: back to one GO per run; clears a halt

Unattended mode is the operator's standing GO with an expiry, kept outside the
repository. It halts the bench, for good until the operator returns, the first
time a run faults, cannot be reset, returns no result or moves the door at all,
or when it expires: the arm cannot close the door, and a fault wants a person.
A halted bench answers every hardware run at once with status "halted" and
moves nothing.

The hardware tier never moves without a fresh operator GO, refuses to run when
a pinned safety file differs from the operator's pin, and sends only
cuRobo-planned, gated, slowed trajectories: the reset to the start pose goes
through the same client and gates as every skill. A score is evidence about
one run on this bench, not validation of the robot.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
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
# The sheppy manifest starts the node from the directory this file names, so a run in a
# research worktree tests that worktree's code. It names the main checkout between runs.
ACTIVE_ROOT = Path.home()/".config/rammp-bench/active-root"
MAIN_ROOT = Path(os.environ.get("RAMMP_MAIN_ROOT", "/home/abra/RAMMP-generalized"))
TASK = os.environ.get("RAMMP_BENCH_TASK", "open the cabinet door in front of you")

# What stands between an experiment and the arm. An experiment that needs one of
# these changed is a conversation with the operator, not a commit.
FROZEN = ("rammp_adl/safety.py", "rammp_adl/motion/collision_guard.py", "rammp_adl/motion/sheppy_client.py",
          "rammp_adl/motion/sheppy_arm.py", "config/sheppy-bench.context.json", "config/imagery-locality.json",
          "config/reasoning.json", "tools/bench_eval.py",
          # The learned-skill sandbox: what model-written code can reach. Never widened to make a skill work.
          "rammp_adl/learned/jail.py", "rammp_adl/learned/gate.py", "rammp_adl/learned/api.py")
# The node's own limits, read back from the running process before every run: an experiment may slow the
# arm or tighten the free-space touch threshold, never the reverse.
CEILINGS = {"transit_speed_scale": .4, "contact_speed_scale": .25, "touch_nm": 3.}

UNATTENDED = PIN.parent/"unattended-until"
HALTED = PIN.parent/"halted"
MAX_UNATTENDED_HOURS = 12.

STAGES = (("intake", 10), ("planned", 10), ("standoff", 15), ("at_handle", 15), ("grasped", 15), ("followed", 30), ("released", 5))

# The self-resetting cycle: one task opens the door, the next, in the same node, closes it again.
OPEN_TASK, CLOSE_TASK = "open the cabinet door", "close the cabinet door"
# The evaluator's own view of the door, independent of the runtime: the wrist camera's depth from the start
# pose, before the cycle and after each task. Only the near field counts (the door fills it from the start pose;
# past FAR_MM the D405's depth is too noisy): a pixel in it before the cycle has changed when it left the near
# field or moved by more than CHANGED_MM (or 3 % of its range). Two still views a few seconds apart differ in
# about 0.5 % of the pixels; the door opened changes most of them.
DEPTH_TOPIC = "/wrist_camera/aligned_depth_to_color/image_raw"
CHANGED_MM, NEAR_MM, FAR_MM = 30., 150., 800.
DOOR_MOVED, DOOR_BACK = .08, .03                               # fractions of the pixels with depth before the cycle


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
    # Arriving at the handle is a completed move to the grasp role, however many other moves a replan added.
    reached = {"intake": result.get("goal") is not None, "planned": bool(nodes), "standoff": len(moves) >= 1,
               "at_handle": any(n.get("pose_role") == "grasp" for n in moves),
               "grasped": any(n.get("skill") == "grasp" for n in done),
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


def changed_fraction(reference, now):
    """The fraction of the reference's near-field pixels whose depth changed.

    None when the reference has too little near field to judge (the door is not in view) or the camera gave no
    depth at all now; a door swung away leaves the near field, so the present view is not required to keep it.
    """
    import numpy as np
    before, after = np.asarray(reference, float), np.asarray(now, float)
    seen = (before > NEAR_MM) & (before < FAR_MM)
    if seen.sum() < .2*seen.size or (after > 0).sum() < .2*after.size:
        return None
    changed = seen & ((after <= NEAR_MM) | (after >= FAR_MM) | (np.abs(after-before) > np.maximum(CHANGED_MM, .03*before)))
    return round(float(changed.sum()/seen.sum()), 4)


def score_cycle(opened, closed, door):
    """0 to 100 for one open-then-close cycle. Pure; the tests pin it.

    opened and closed are score_run results (closed is None when the close task was not sent); door holds the
    evaluator's own changed fractions after each task. The open half keeps its followed points only when the
    camera saw the door move; the close half is 80 % the close task's stages and 20 % the camera seeing the
    scene back as it was before the cycle, and counts only after a door that moved.
    """
    moved = door.get("after_open") is not None and door["after_open"] >= DOOR_MOVED
    back = door.get("after_close") is not None and door["after_close"] <= DOOR_BACK
    open_points = opened["score"] if moved else min(opened["score"], 100.-dict(STAGES)["followed"])
    close_points = 0.
    if moved and closed is not None:
        close_points = .8*closed["score"]+(20. if back else 0.)
    return {"score": round(.5*open_points+.5*close_points, 1), "open_score": round(open_points, 1),
            "close_score": round(close_points, 1), "door_moved": moved, "door_back": back}


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


def unattended_active():
    """The operator's standing GO, if it is set and unexpired. An expired one halts the bench."""
    try:
        until = float(UNATTENDED.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return False
    if time.time() < until:
        return True
    end_unattended("the unattended window expired")
    return False


def halted():
    try:
        return HALTED.read_text().strip() or "halted"
    except OSError:
        return None


def end_unattended(reason):
    """The bench needs a person from here: halt it until the operator returns. Attended runs are unaffected."""
    if not UNATTENDED.exists():
        return
    UNATTENDED.unlink()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    HALTED.write_text(f"{stamp} {reason}\n")
    BENCH.mkdir(parents=True, exist_ok=True)
    with open(BENCH/"unattended-ended.txt", "a") as log:
        log.write(f"{stamp} {reason}\n")


def unattended(args):
    hours = float(args.hours)
    if not 0. < hours <= MAX_UNATTENDED_HOURS:
        raise SystemExit(f"between 0 and {MAX_UNATTENDED_HOURS:g} hours")
    UNATTENDED.parent.mkdir(parents=True, exist_ok=True)
    until = time.time()+hours*3600.
    UNATTENDED.write_text(f"{until} until {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(until))}\n")
    HALTED.unlink(missing_ok=True)
    print(f"unattended until {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(until))}: hardware runs go without a GO. "
          "The bench halts itself on a fault, a failed reset, a run with no result, any movement of the door, or expiry.")


def attended(_args):
    UNATTENDED.unlink(missing_ok=True)
    HALTED.unlink(missing_ok=True)
    print("attended: every hardware run waits for a GO")


def wait_for_go(timeout_s):
    """A GO written after this call began, consumed on use; or the operator's standing GO. Neither, no motion."""
    if unattended_active():
        return True
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
    """The evaluator's ROS side, on its own node spun in a thread.

    Its arm client lives on a second node that exists only while the evaluator itself moves or plans for the
    arm (a reset, the plan-only check, reading the joints): the runtime refuses a task while any other node holds
    a client of the driver's actions, as it should, so none is left behind when a task is sent.
    """
    ARM_NODE = "rammp_bench_eval_arm"

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        rclpy.init()
        self.rclpy, self.Node, self.node = rclpy, Node, Node("rammp_bench_eval")
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    @contextlib.contextmanager
    def arm_client(self):
        """An unarmed-by-default arm client for the length of the block; its node is gone from the graph after."""
        from rammp_adl.motion.sheppy_arm import SheppyArmClient
        node = self.Node(self.ARM_NODE)
        client = SheppyArmClient(node)
        self.executor.add_node(node)
        deadline = time.time()+10.
        while client.live_joints() is None and time.time() < deadline:
            time.sleep(.1)
        try:
            yield client
        finally:
            client.disarm()
            self.executor.remove_node(node)
            # Humble's destroy_node leaves action clients (waitables) alive, and they keep the node in the graph.
            for waitable in list(node.waitables):
                if hasattr(waitable, "destroy"):
                    waitable.destroy()
            node.destroy_node()
            deadline = time.time()+10.
            while time.time() < deadline and any(name == self.ARM_NODE for name, _ in self.node.get_node_names_and_namespaces()):
                time.sleep(.1)

    def joints(self):
        with self.arm_client() as client:
            return client.live_joints()

    def depth(self, frames=5, timeout_s=10.):
        """The per-pixel median of a few depth frames from the wrist camera, every 4th pixel, in mm; None without them."""
        import numpy as np
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        got = []

        def take(message):
            if len(got) < frames and message.encoding in ("16UC1", "mono16"):
                rows = np.frombuffer(message.data, np.uint16).reshape(message.height, message.step//2)
                got.append(rows[::4, :message.width:4].copy())
        subscription = self.node.create_subscription(Image, DEPTH_TOPIC, take, QoSProfile(
            depth=5, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT))
        deadline = time.time()+timeout_s
        while len(got) < frames and time.time() < deadline:
            time.sleep(.05)
        self.node.destroy_subscription(subscription)
        return np.median(np.stack(got[:frames]), axis=0) if len(got) >= frames else None

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
        from rammp_adl.motion.sheppy_client import CONTINUOUS, scale_trajectory_time, wrap_diff
        with self.arm_client() as client:
            client.arm()
            opened = await client.gripper(0.)
            if not await client.settle(timeout_s=10.):
                return {"ok": False, "detail": "the arm is not still; reset refused"}
            live = client.live_joints()["position_rad"]
            if max(abs(wrap_diff(a, b)) for a, b in zip(live, start)) < .02:
                return {"ok": True, "detail": "already at the start pose", "gripper": opened["message"]}
            # The short way round: a continuous joint read across +-pi from the recorded start would otherwise be
            # planned the long way, most of a turn.
            nearest = [q+wrap_diff(s, q) if i in CONTINUOUS else s for i, (q, s) in enumerate(zip(live, start))]
            trajectory, planning = await client.plan_to_joints(nearest)
            receipt = await client.execute(scale_trajectory_time(trajectory, 1./CEILINGS["transit_speed_scale"]),
                                           guard=GuardSet(effort=EffortGuard(CEILINGS["touch_nm"])))
            return {"ok": receipt["status"] == "succeeded", "detail": receipt["message"] or receipt["status"],
                    "planning": planning["message"], "gripper": opened["message"]}

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


def node_roots():
    """The working directory of every running adl node process: the checkout whose code it imported."""
    roots = set()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc/"cmdline").read_bytes().split(b"\0")[:3] == [b"python", b"-m", b"rammp_adl.ros_node"]:
                roots.add(os.path.realpath(proc/"cwd"))
        except OSError:
            continue
    return roots


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
                roots = node_roots()
                if roots != {os.path.realpath(ROOT)}:
                    # The deployment manifest no longer starts the node from active-root: this run would
                    # score another checkout's code. Not a measurement.
                    return False, f"the node runs {sorted(roots)}, not {ROOT}; the manifest must start it from active-root", fresh[-1]
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


def start_run(args, tier):
    """What every run that moves the arm does first: the pin, the start pose, the halt, the GO, the node restarted
    on this checkout, its limits read back, and the arm reset to the start pose. (stack, facts), or (None, the
    emitted refusal) when the run may not go on."""
    refuse = lambda status, reason, **extra: (None, emit({"tier": tier, "score": 0., "status": status, "reason": reason, **extra}))
    problems = frozen_problems()
    if problems:
        return refuse("refused", "; ".join(problems), moved=False)
    start_file = BENCH/"start-joints.json"
    if not start_file.is_file():
        return refuse("refused", "no start pose; the operator runs `python tools/bench_eval.py record-start` once", moved=False)
    stopped = halted()
    if stopped:
        return refuse("halted", f"the bench is halted until the operator returns ({stopped})", moved=False)
    standing = unattended_active()
    if not wait_for_go(args.go_timeout_s):
        return refuse("operator_absent", "no operator GO; nothing was sent to the arm", moved=False)
    ACTIVE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_ROOT.write_text(str(ROOT)+"\n")
    try:
        ready, detail, log = restart_node()
    finally:
        ACTIVE_ROOT.write_text(str(MAIN_ROOT)+"\n")            # the next plain restart is the main checkout again
    if not ready:
        return refuse("node_failed_to_start", detail, moved=False)
    stack = Stack()
    limits = stack.parameters(list(CEILINGS))
    if limits is None or any(limits[name] > ceiling+1e-9 for name, ceiling in CEILINGS.items()):
        return refuse("refused", f"the node's speed or effort limits are looser than the bench allows: {limits}", moved=False)
    start = json.loads(start_file.read_text())["position_rad"]
    reset = asyncio.run(stack.reset(start))
    if not reset["ok"]:
        end_unattended("reset failed: "+str(reset["detail"])[:200])
        return refuse("reset_failed", reset["detail"], reset=reset)
    time.sleep(args.settle_s)                                  # a still keyframe from the start pose
    return stack, {"start": start, "log": log, "limits": limits, "reset": reset,
                   "operator": "standing GO" if standing else "GO"}


def log_tail(log, count=40):
    return [] if log is None else [line[line.find("]: ")+3:][:240] for line in log.read_text(errors="replace").splitlines()
                                   if "rammp_adl_runtime" in line and "known gap" not in line][-count:]


def hardware(args):
    stack, facts = start_run(args, "hardware")
    if stack is None:
        return facts
    began = time.time()
    result, phases = stack.send_task(TASK, args.task_timeout_s)
    if result is None:
        end_unattended("a run returned no result")
        return emit({"tier": "hardware", "score": 0., "status": "no_result", "reason": phases[-1] if phases else "", "phases": phases})
    scored = score_run(result, attempt_for(result.get("task_id")))
    if scored["safety_fault"]:
        end_unattended("a run ended in a safety fault: "+scored["reason"][:200])
    elif scored["followed_fraction"] > 0. or scored["stages"]["grasped"]:
        end_unattended("the door was grasped or moved; it has to be closed by hand")
    return emit({"tier": "hardware", **scored, "operator": facts["operator"], "duration_s": round(time.time()-began, 1), "phases": phases, "task": TASK,
                 "task_id": result.get("task_id"), "reset": facts["reset"], "node_log_tail": log_tail(facts["log"]), "limits": facts["limits"]})


def cycle(args):
    """One open-then-close cycle in one node: the second task closes the door the first one opened.

    After each task the arm is back at the start pose (the runtime's own promise; the reset only checks it or
    finishes the way) and the evaluator compares the wrist camera's depth with its view before the cycle. A cycle
    that leaves the door as it found it needs no person, so it does not halt the bench."""
    stack, facts = start_run(args, "cycle")
    if stack is None:
        return facts
    reference = stack.depth()
    if reference is None or changed_fraction(reference, reference) is None:
        end_unattended("the door is not in the wrist camera's near field from the start pose")
        return emit({"tier": "cycle", "score": 0., "status": "no_view", "moved": True, "reset": facts["reset"],
                     "reason": "no depth frames, or too little near field to see the door, from the start pose"})
    began, door, halves, phases, durations, task_ids, homes = time.time(), {}, {}, {}, {}, {}, {}
    for name, text in (("open", OPEN_TASK), ("close", CLOSE_TASK)):
        if name == "close" and (halves["open"]["safety_fault"] or (door["after_open"] or 0.) < DOOR_MOVED):
            break                                              # nothing to close, or a fault a person must see
        sent = time.time()
        result, phases[name] = stack.send_task(text, args.task_timeout_s)
        durations[name] = round(time.time()-sent, 1)
        if result is None:
            end_unattended(f"the {name} task returned no result")
            return emit({"tier": "cycle", "score": 0., "status": "no_result", "reason": f"{name}: "+(phases[name][-1] if phases[name] else ""),
                         "phases": phases, "node_log_tail": log_tail(facts["log"], 60)})
        halves[name] = score_run(result, attempt_for(result.get("task_id")))
        halves[name]["task_replans"] = result.get("task_replans") or 0
        task_ids[name] = result.get("task_id")
        homes[name] = asyncio.run(stack.reset(facts["start"]))
        if not homes[name]["ok"]:
            door[f"after_{name}"] = None
            break
        time.sleep(args.settle_s)
        view = stack.depth()
        door[f"after_{name}"] = None if view is None else changed_fraction(reference, view)
    scored = score_cycle(halves["open"], halves.get("close"), door)
    fault = any(half["safety_fault"] for half in halves.values())
    last = door.get("after_close", door.get("after_open"))
    if fault:
        end_unattended("a run ended in a safety fault: "+next(h["reason"] for h in halves.values() if h["safety_fault"])[:200])
    elif not all(home["ok"] for home in homes.values()):
        end_unattended("the arm could not be returned to the start pose after a task")
    elif last is None or last > DOOR_BACK:
        end_unattended("the door was left open or the view from the start pose changed; it has to be put back by hand")
    failed = next((half["first_failure"] for half in halves.values() if half["first_failure"]), None)
    status = ("safety_fault" if fault else "succeeded" if scored["door_back"] and all(h["status"] == "succeeded" for h in halves.values())
              and len(halves) == 2 else "incomplete")
    return emit({"tier": "cycle", **scored, "status": status, "safety_fault": fault, "door": door, "first_failure": failed,
                 "open": halves["open"], "close": halves.get("close"), "duration_s": round(time.time()-began, 1),
                 "durations_s": durations, "task_replans": sum(h["task_replans"] for h in halves.values()),
                 "tasks": {"open": OPEN_TASK, "close": CLOSE_TASK}, "task_ids": task_ids, "phases": phases,
                 "homes": {name: home["detail"] for name, home in homes.items()}, "operator": facts["operator"],
                 "reset": facts["reset"], "node_log_tail": log_tail(facts["log"], 60), "limits": facts["limits"]})


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
        with stack.arm_client() as client:
            live = client.live_joints()
            if live is None:
                return {"fraction": 0., "detail": "no /joint_states"}
            target = [q+(.15 if index == 5 else 0.) for index, q in enumerate(live["position_rad"])]
            trajectory, planning = asyncio.run(client.plan_to_joints(target, timeout_s=60.))
            why = refusal(scale_trajectory_time(trajectory, 2.5), client.live_joints()["position_rad"])
            return {"fraction": 0. if why else 1., "detail": why or planning["message"], "armed": client.motion_enabled}
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
    both = commands.add_parser("cycle", help="open the door, then close it in a second task; scored by stages and the camera")
    both.add_argument("--go-timeout-s", type=float, default=900.)
    both.add_argument("--task-timeout-s", type=float, default=900.)
    both.add_argument("--settle-s", type=float, default=4.)
    both.set_defaults(function=cycle)
    off = commands.add_parser("offline")
    off.add_argument("--plan", action="store_true", help="also ask the live planner for a path (no motion)")
    off.set_defaults(function=offline)
    standing = commands.add_parser("unattended")
    standing.add_argument("--hours", type=float, required=True)
    standing.set_defaults(function=unattended)
    for name, function in (("go", go), ("freeze", freeze), ("record-start", record_start), ("attended", attended)):
        commands.add_parser(name).set_defaults(function=function)
    args = parser.parse_args()
    args.function(args)
    sys.stdout.flush()
    os._exit(0)                                                # the ROS spin thread is a daemon; do not wait on it


if __name__ == "__main__":
    main()
