#!/usr/bin/env python3
"""Record teleop demonstrations for a learned skill phase.

    python tools/demo_record.py record --task door_handle_grasp     record episodes while you drive the Xbox pad
    python tools/demo_record.py list                                 episodes per task

Bring the arm up on the demos profile (sheppy up demos): the Kinova driver, the
planner, the runtime's own D405 wrist camera and the Xbox teleop. This tool only
listens; it never commands the arm.

Controls while recording:
    A        start an episode / stop it and keep it
    LB       stop and throw the current episode away (or, between episodes, the last kept one)
    Enter    same as A, from the keyboard
    Ctrl+C   quit

Each episode is saved under artifacts/demos/<task>/episode_NNN/: the wrist colour
frames (half resolution JPEG), depth in millimetres, and one line per frame with
the time, joints, gripper knuckle and the operator's twist and gripper commands.
Episodes stay on this machine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEMOS = Path(os.environ.get("RAMMP_DEMOS_DIR", ROOT/"artifacts/demos"))
TOPICS = dict(rgb_topic="/wrist_camera/color/image_raw", depth_topic="/wrist_camera/aligned_depth_to_color/image_raw",
              rgb_info_topic="/wrist_camera/color/camera_info", depth_info_topic="/wrist_camera/aligned_depth_to_color/camera_info")
BUTTON_A, BUTTON_LB = 0, 4                  # rammp-teleop's Xbox layout: [A, B, X, Y, LB, RB, Back, Start, ...]; A and LB are unused by teleop
RATE_HZ = 15.
SCALE = .5                                  # frames are stored at half resolution; intrinsics are scaled with them


class Episode:
    """One demonstration, held in memory and written when it is kept."""

    def __init__(self, task, index, intrinsics):
        self.task, self.index, self.started = task, index, time.time()
        self.intrinsics = [v*SCALE if i in (0, 2, 4, 5) else v for i, v in enumerate(intrinsics)]
        self.frames, self.depths, self.steps = [], [], []

    def add(self, rgb, depth_m, step):
        import cv2
        import numpy as np
        size = (int(rgb.shape[1]*SCALE), int(rgb.shape[0]*SCALE))
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
        depth = cv2.resize(np.nan_to_num(np.asarray(depth_m, dtype=float), nan=0.), size, interpolation=cv2.INTER_NEAREST)
        self.frames.append(jpeg.tobytes())
        self.depths.append(np.clip(np.round(depth*1000.), 0, 65535).astype(np.uint16))
        self.steps.append(step)

    def save(self, folder):
        import numpy as np
        path = folder/self.task/f"episode_{self.index:03d}"
        (path/"frames").mkdir(parents=True, exist_ok=False)
        for i, jpeg in enumerate(self.frames):
            (path/"frames"/f"{i:05d}.jpg").write_bytes(jpeg)
        np.savez_compressed(path/"depth.npz", depth_mm=np.stack(self.depths))
        with open(path/"steps.jsonl", "w") as out:
            for step in self.steps:
                out.write(json.dumps(step)+"\n")
        from rammp_adl.perception.scene_record import current_layout
        (path/"meta.json").write_text(json.dumps({
            "task": self.task, "episode": self.index, "frames": len(self.frames), "rate_hz": RATE_HZ,
            "duration_s": round(self.steps[-1]["t"]-self.steps[0]["t"], 2) if self.steps else 0.,
            "intrinsics_k": self.intrinsics, "image_scale": SCALE, "camera": "wrist_d405",
            "layout": current_layout(ROOT/"artifacts/bench"), "outcome": "success",
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started))}, indent=1)+"\n")
        return path


def next_index(task):
    folder = DEMOS/task
    taken = [int(p.name.split("_")[1]) for p in folder.glob("episode_*") if p.name.split("_")[1].isdigit()] if folder.is_dir() else []
    return max(taken, default=-1)+1


def record(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Joy
    from rammp_adl.contracts import strict_loads
    from rammp_adl.motion.sheppy_arm import SheppyArmClient
    from rammp_adl.perception.ros_rgbd import RosRgbdSource

    task = args.task.strip().replace(" ", "_")
    rclpy.init()
    node = Node("rammp_demo_record")
    client = SheppyArmClient(node)                       # joints and knuckle only; never armed
    commands = {"twist": None, "gripper": None}
    presses = []
    held = {BUTTON_A: False, BUTTON_LB: False}

    def on_joy(message):
        for button in held:
            down = len(message.buttons) > button and bool(message.buttons[button])
            if down and not held[button]:
                presses.append(button)
            held[button] = down
    node.create_subscription(Joy, "/joy", on_joy, qos_profile_sensor_data)
    try:
        from rammp_arm_interfaces.msg import GripperSetpoint, TwistSetpoint
        node.create_subscription(TwistSetpoint, "/setpoint/twist", lambda m: commands.__setitem__("twist", _twist(m)), qos_profile_sensor_data)
        node.create_subscription(GripperSetpoint, "/setpoint/gripper",
                                 lambda m: commands.__setitem__("gripper", float(getattr(m, "position", 0.))), 10)
    except ImportError:
        pass
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    threading.Thread(target=lambda: [presses.append(BUTTON_A) for _ in iter(sys.stdin.readline, "")], daemon=True).start()
    policy = strict_loads((ROOT/"config/imagery-locality.json").read_bytes())
    source = RosRgbdSource(**TOPICS, reliability="reliable", domain_id=0, locality_policy=policy)
    episode, last_saved = None, None
    print(f"task {task}: A (or Enter) starts and stops an episode, LB throws one away, Ctrl+C quits. "
          f"{next_index(task)} episodes so far.", flush=True)
    period, next_tick = 1./RATE_HZ, time.monotonic()
    try:
        while True:
            while presses:
                button = presses.pop(0)
                if button == BUTTON_A:
                    if episode is None:
                        live = client.live_joints()
                        if live is None:
                            print("no /joint_states; is the arm up?", flush=True)
                            continue
                        episode = "pending"
                        print("recording… (A to stop and keep, LB to throw away)", flush=True)
                    else:
                        if isinstance(episode, Episode) and len(episode.frames) >= int(RATE_HZ):
                            last_saved = episode.save(DEMOS)
                            print(f"kept {last_saved.name}: {len(episode.frames)} frames, "
                                  f"{episode.steps[-1]['t']-episode.steps[0]['t']:.1f} s", flush=True)
                        else:
                            print("too short to keep (under a second)", flush=True)
                        episode = None
                elif button == BUTTON_LB:
                    if episode is not None:
                        episode = None
                        print("thrown away", flush=True)
                    elif last_saved is not None and last_saved.exists():
                        for item in sorted(last_saved.rglob("*"), reverse=True):
                            item.unlink() if item.is_file() else item.rmdir()
                        last_saved.rmdir()
                        print(f"deleted {last_saved.name}", flush=True)
                        last_saved = None
            try:
                pair = source.capture(500)
            except Exception:                           # noqa: BLE001 - a dropped frame; the next one comes
                continue
            if episode is None or time.monotonic() < next_tick:
                continue
            next_tick += period
            if next_tick < time.monotonic():
                next_tick = time.monotonic()+period
            live = client.live_joints()
            if live is None:
                continue
            if episode == "pending":
                episode = Episode(task, next_index(task), pair.metadata["rgb_info"]["k"])
            episode.add(pair.rgb, pair.depth_m, {
                "t": round(time.time(), 4), "joints_rad": [round(v, 5) for v in live["position_rad"]],
                "knuckle_rad": None if live["knuckle_rad"] is None else round(live["knuckle_rad"], 4),
                "velocity_rad_s": None if live["velocity_rad_s"] is None else [round(v, 4) for v in live["velocity_rad_s"]],
                "twist_cmd": commands["twist"], "gripper_cmd": commands["gripper"],
                "stamp_ns": int(pair.metadata["source_stamps_ns"]["rgb"])})
    except KeyboardInterrupt:
        pass
    finally:
        if isinstance(episode, Episode):
            print("an episode was still recording; it was not kept", flush=True)
        source.close()
        os._exit(0)


def _twist(message):
    try:
        twist = message.twist
        return [round(float(v), 4) for v in (twist.linear.x, twist.linear.y, twist.linear.z,
                                             twist.angular.x, twist.angular.y, twist.angular.z)]
    except AttributeError:
        return None


def listing(_args):
    if not DEMOS.is_dir():
        print("no demonstrations yet")
        return
    for task in sorted(p for p in DEMOS.iterdir() if p.is_dir()):
        episodes = sorted(task.glob("episode_*/meta.json"))
        seconds = sum(json.loads(p.read_text())["duration_s"] for p in episodes)
        print(f"{task.name:<28} {len(episodes):>4} episodes  {seconds/60:6.1f} min")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("record")
    run.add_argument("--task", required=True, help="one name per skill phase, e.g. door_handle_grasp")
    run.set_defaults(function=record)
    commands.add_parser("list").set_defaults(function=listing)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
