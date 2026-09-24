#!/usr/bin/env python3
"""Turn teleop demonstrations into a learned skill phase, on the Jetson.

Run with the policy environment, which pins LeRobot to the Jetson's CUDA PyTorch:

    .venv-policy/bin/python tools/policy_train.py convert --task door_handle_grasp
    .venv-policy/bin/python tools/policy_train.py train   --task door_handle_grasp [--steps 40000]
    .venv-policy/bin/python tools/policy_train.py status  --task door_handle_grasp

convert: every kept episode under artifacts/demos/<task>/ becomes one LeRobot
episode: the wrist image and the state (seven joints and the gripper knuckle) at
each frame, and as the action the state one frame later, so the policy learns
where the arm went next.

train: an ACT policy (one-second action chunks at 15 Hz) from that dataset into
artifacts/policies/<task>/train/. Nothing here moves the arm; a trained policy is
used by a skill only after it passes on the real bench.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMOS = Path(os.environ.get("RAMMP_DEMOS_DIR", ROOT/"artifacts/demos"))
POLICIES = Path(os.environ.get("RAMMP_POLICIES_DIR", ROOT/"artifacts/policies"))
FPS = 15
CHUNK = 15                   # one second of actions per prediction
EXECUTE = 8                  # of which the first half second is executed before predicting again
DESCRIPTIONS = {"door_handle_grasp": "reach in and grasp the door handle", "door_pull": "pull the grasped door open"}


def episodes(task):
    folder = DEMOS/task
    return sorted(p for p in folder.glob("episode_*") if (p/"meta.json").is_file()) if folder.is_dir() else []


def convert(args):
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    paths = [p for p in episodes(args.task) if json.loads((p/"meta.json").read_text()).get("outcome") == "success"]
    if not paths:
        raise SystemExit(f"no kept episodes under {DEMOS/args.task}")
    first = sorted((paths[0]/"frames").glob("*.jpg"))[0]
    width, height = Image.open(first).size
    root = POLICIES/args.task/"dataset"
    if root.exists():
        shutil.rmtree(root)
    features = {"observation.images.wrist": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channels"]},
                "observation.state": {"dtype": "float32", "shape": (8,), "names": [f"joint_{i+1}" for i in range(7)]+["knuckle"]},
                "action": {"dtype": "float32", "shape": (8,), "names": [f"joint_{i+1}" for i in range(7)]+["knuckle"]}}
    dataset = LeRobotDataset.create(repo_id=f"rammp/{args.task}", fps=FPS, features=features, root=root,
                                    robot_type="kinova_gen3_2f85", use_videos=False)
    description = DESCRIPTIONS.get(args.task, args.task.replace("_", " "))
    frames_total = 0
    for path in paths:
        steps = [json.loads(line) for line in (path/"steps.jsonl").read_text().splitlines()]
        images = sorted((path/"frames").glob("*.jpg"))
        states = [np.array(s["joints_rad"]+[s["knuckle_rad"] if s["knuckle_rad"] is not None else 0.], dtype=np.float32) for s in steps]
        for i, (image, state) in enumerate(zip(images, states)):
            dataset.add_frame({"observation.images.wrist": np.asarray(Image.open(image).convert("RGB")),
                               "observation.state": state, "action": states[min(i+1, len(states)-1)], "task": description})
        dataset.save_episode()
        frames_total += len(states)
    dataset.finalize()
    print(json.dumps({"task": args.task, "episodes": len(paths), "frames": frames_total, "dataset": str(root)}))


def train(args):
    dataset = POLICIES/args.task/"dataset"
    if not dataset.is_dir():
        raise SystemExit("convert the demonstrations first")
    output = POLICIES/args.task/"train"
    if output.exists() and not args.resume:
        raise SystemExit(f"{output} exists; pass --resume to continue it, or move it aside")
    command = [str(Path(sys.executable).parent/"lerobot-train"), f"--dataset.repo_id=rammp/{args.task}", f"--dataset.root={dataset}",
               "--policy.type=act", "--policy.device=cuda", "--policy.push_to_hub=false", f"--policy.chunk_size={CHUNK}",
               f"--policy.n_action_steps={EXECUTE}", f"--output_dir={output}", f"--steps={args.steps}", f"--batch_size={args.batch_size}",
               f"--save_freq={max(1, min(args.steps, 5000))}", "--num_workers=2", "--wandb.enable=false", f"--job_name={args.task}"]
    if args.resume:
        command.append("--resume=true")
    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, env={**os.environ, "HF_HUB_OFFLINE": "1"}))


def status(args):
    kept = episodes(args.task)
    checkpoints = sorted((POLICIES/args.task/"train"/"checkpoints").glob("*")) if (POLICIES/args.task/"train").is_dir() else []
    print(json.dumps({"task": args.task, "episodes": len(kept), "dataset": (POLICIES/args.task/"dataset").is_dir(),
                      "checkpoints": [p.name for p in checkpoints]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("convert", convert), ("train", train), ("status", status)):
        command = commands.add_parser(name)
        command.add_argument("--task", required=True)
        if name == "train":
            command.add_argument("--steps", type=int, default=40000)
            command.add_argument("--batch-size", type=int, default=8)
            command.add_argument("--resume", action="store_true")
        command.set_defaults(function=function)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
