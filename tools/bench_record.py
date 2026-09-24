#!/usr/bin/env python3
"""Record wrist scenes for the offline bench (tools/bench_offline.py).

    python tools/bench_record.py snapshot          the current still view; nothing moves
    python tools/bench_record.py survey            operator at the arm: ten views around the start pose, then back
    python tools/bench_record.py layout NAME       name the bench's arrangement after moving the cabinet or camera
    python tools/bench_record.py list              what has been recorded

Robot runs record themselves: the adl node saves every keyframe it selects and
every collision-guard trip under artifacts/bench. A survey adds views on purpose.
It moves the arm only through cuRobo plans, slowed to the bench's transit speed,
under the effort guard, the same way the bench's reset does, and it refuses to
start while a bench hardware run is in progress. Needs the arm, planner and wrist
camera up (the adl profile); the adl node itself may be up or not.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH = Path(os.environ.get("RAMMP_BENCH_DIR", ROOT/"artifacts/bench"))
TOPICS = dict(rgb_topic="/wrist_camera/color/image_raw", depth_topic="/wrist_camera/aligned_depth_to_color/image_raw",
              rgb_info_topic="/wrist_camera/color/camera_info", depth_info_topic="/wrist_camera/aligned_depth_to_color/camera_info")
TRANSIT_SLOWDOWN = 2.5                   # the bench's transit speed scale, 0.4
TOUCH_NM = 3.
# Around the start pose: base yaw (joint 1) and wrist pitch (joint 6), in radians, and one view 8 cm closer.
YAWS, PITCHES, CLOSER_M = (-.2, 0., .2), (-.15, 0., .15), .08


class Rig:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from rammp_adl.contracts import strict_loads
        from rammp_adl.motion.kinematics import UrdfChain
        from rammp_adl.motion.sheppy_arm import SheppyArmClient
        from rammp_adl.perception.grounded_scene import GroundedScene
        from rammp_adl.perception.keyframes import KeyframeSelector
        from rammp_adl.perception.ros_rgbd import RosRgbdSource
        rclpy.init()
        self.node = Node("rammp_bench_record")
        self.client = SheppyArmClient(self.node)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)
        threading.Thread(target=executor.spin, daemon=True).start()
        deadline = time.time()+10.
        while self.client.live_joints() is None and time.time() < deadline:
            time.sleep(.1)
        if self.client.live_joints() is None:
            raise SystemExit("no /joint_states: bring up the arm (sheppy up adl)")
        self.chain = UrdfChain.from_path(ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2/arm-gripper-locked.urdf")
        base = strict_loads((ROOT/"config/sheppy-bench.context.json").read_bytes())
        self.scene = GroundedScene(client=self.client, chain=self.chain, calibration_id=base["calibration_id"],
                                   selector=KeyframeSelector(min_interval_s=.5))
        policy = strict_loads((ROOT/"config/imagery-locality.json").read_bytes())
        self.source = RosRgbdSource(**TOPICS, reliability="reliable", domain_id=0, locality_policy=policy)

    def capture(self, source_name, timeout_s=8.):
        """A still keyframe of the current view, saved as a scene; its path or None."""
        from rammp_adl.perception.scene_record import save_scene
        deadline = time.time()+timeout_s
        while time.time() < deadline:
            try:
                pair = self.source.capture(1000)
            except Exception:                           # noqa: BLE001 - a dropped frame; try the next
                continue
            keyframe = self.scene.on_pair(pair, force=True)
            if keyframe is not None:
                return save_scene(BENCH, keyframe, source=source_name)
        return None

    async def move(self, *, joints=None, pose=None):
        from rammp_adl.motion.collision_guard import EffortGuard, GuardSet
        from rammp_adl.motion.sheppy_client import scale_trajectory_time
        if not await self.client.settle(timeout_s=10.):
            return "the arm is not still"
        if joints is not None:
            trajectory, _ = await self.client.plan_to_joints(joints)
        else:
            trajectory, _ = await self.client.plan_to_pose(*pose)
        receipt = await self.client.execute(scale_trajectory_time(trajectory, TRANSIT_SLOWDOWN),
                                            guard=GuardSet(effort=EffortGuard(TOUCH_NM)))
        return None if receipt["status"] == "succeeded" else f"{receipt['status']}: {receipt['message']}"

    def closer_pose(self, joints, distance_m):
        import numpy as np
        from rammp_adl.motion.kinematics import quaternion_xyzw_from_matrix
        from rammp_adl.motion.sheppy_client import JOINTS, TOOL_FRAME_FROM_FLANGE_M
        tool = self.chain.base_from_link(dict(zip(JOINTS, joints)), "end_effector_link") @ np.array(
            [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., TOOL_FRAME_FROM_FLANGE_M], [0., 0., 0., 1.]])
        position = tool[:3, 3]+tool[:3, 2]*distance_m
        return tuple(float(v) for v in position), tuple(float(v) for v in quaternion_xyzw_from_matrix(tool[:3, :3]))

    def close(self):
        self.client.disarm()
        self.source.close()


def snapshot(_args):
    rig = Rig()
    try:
        path = rig.capture("snapshot")
        print(f"recorded {path}" if path else f"no still keyframe: {rig.scene.refusal_reason()}")
    finally:
        rig.close()
        os._exit(0)


def survey(_args):
    if subprocess.run(["pgrep", "-f", "bench_eval.py hardware"], capture_output=True).returncode == 0:
        raise SystemExit("a bench hardware run is in progress; the arm is not free")
    rig = Rig()
    start_file = BENCH/"start-joints.json"
    start = json.loads(start_file.read_text())["position_rad"] if start_file.is_file() else list(rig.client.live_joints()["position_rad"])
    views = [("start", {"joints": start})]
    for yaw in YAWS:
        for pitch in PITCHES:
            if yaw or pitch:
                joints = list(start)
                joints[0] += yaw
                joints[5] += pitch
                views.append((f"yaw {yaw:+.2f} pitch {pitch:+.2f}", {"joints": joints}))
    views.append((f"{CLOSER_M*100:.0f} cm closer", {"pose": rig.closer_pose(start, CLOSER_M)}))
    rig.client.arm()
    recorded = []

    async def run():
        for name, target in views:
            problem = await rig.move(**target)
            if problem:
                print(f"{name}: not reached ({problem}); returning to the start")
                break
            await rig.client.settle(timeout_s=10.)
            path = await asyncio.to_thread(rig.capture, "survey")
            print(f"{name}: {'recorded '+path.name if path else 'no still keyframe'}", flush=True)
            if path:
                recorded.append(path)
        problem = await rig.move(joints=start)
        print("back at the start" if problem is None else f"return to the start failed: {problem}")
    try:
        asyncio.run(run())
    finally:
        rig.close()
        print(f"{len(recorded)} scenes recorded under {BENCH/'scenes'}", flush=True)
        os._exit(0)


def layout(args):
    folder = BENCH/"scenes"
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"LAYOUT").write_text(args.name.strip()+"\n")
    print(f"scenes recorded from now on belong to layout {args.name.strip()!r}")


def listing(_args):
    from rammp_adl.perception.scene_record import current_layout, list_scenes
    scenes = [json.loads((p/"meta.json").read_text()) for p in list_scenes(BENCH)]
    counts = Counter((m.get("layout", "default"), m.get("source")) for m in scenes)
    trips = len(list((BENCH/"trips").glob("*/meta.json"))) if (BENCH/"trips").is_dir() else 0
    print(f"current layout: {current_layout(BENCH)}")
    for (name, source), count in sorted(counts.items()):
        print(f"  {name:<16} {source:<9} {count}")
    print(f"{len(scenes)} scenes, {trips} guard trips")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("snapshot").set_defaults(function=snapshot)
    commands.add_parser("survey").set_defaults(function=survey)
    named = commands.add_parser("layout")
    named.add_argument("name")
    named.set_defaults(function=layout)
    commands.add_parser("list").set_defaults(function=listing)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
