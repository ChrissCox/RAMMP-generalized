"""Local command line for validation, simulation, camera probing and Astra."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from .contracts import Catalog, strict_loads


def _write_json(path, result):
    data = json.dumps(result, indent=2, allow_nan=False)
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data + "\n", encoding="utf-8")
    if path and isinstance(result, dict):
        preview = {key: value for key, value in result.items() if key not in {"samples", "final_world", "backend_events", "events"}}
        print(json.dumps({"report": str(path), **preview}, indent=2, allow_nan=False))
    else:
        print(data)


def doctor():
    modules = ("jsonschema", "numpy", "mujoco", "openai", "cv2", "rclpy", "curobo", "pyrealsense2", "pyorbbecsdk")
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "modules": {module: importlib.util.find_spec(module) is not None for module in modules},
        "commands": {command: shutil.which(command) is not None for command in ("ros2", "colcon", "docker", "nvidia-smi")},
        "astra_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "physical_robot_commands_enabled": False,
        "note": "Dependency presence is not provider access, driver commissioning or robot validation.",
    }


async def _run(args):
    from .app import astra_for, fixture_runtime
    catalog = Catalog(args.root)
    context_path = Path(args.context) if args.context else catalog.root / f"examples/{args.scenario}.context.json"
    plan_path = Path(args.plan) if args.plan else catalog.root / f"examples/{args.scenario}.plan.json"
    failures = {}
    for item in args.inject:
        node, separator, code = item.partition(":")
        if not separator or not node or not code:
            raise ValueError("--inject requires NODE:FAILURE_CODE")
        failures.setdefault(node, []).append(code)
    runtime = fixture_runtime(context_path, root=catalog.root, time_scale=args.time_scale, failures=failures)
    if args.reasoner == "astra":
        if not args.task:
            raise ValueError("--task is required when using Astra")
        reasoner = astra_for(runtime)
        try:
            result = await runtime.executor.run_task(args.task, reasoner)
        finally:
            await reasoner.close()
    else:
        plan = strict_loads(plan_path.read_bytes())
        result = await runtime.executor.run_plan(plan)
    report = {"result": result.to_dict(), "timing": runtime.trace.summary(),
              "validation_scope": "synthetic task-state simulation; no cuRobo planning or physical contact validation",
              "final_world": runtime.world.snapshot().context,
              "backend_events": runtime.backend.events}
    _write_json(args.output, report)
    if args.trace:
        runtime.trace.write(args.trace)
    return 0 if result.status == "succeeded" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAMMP ADL simulation, integration and commissioning tools")
    parser.add_argument("--root", help="Directory containing canonical skills/schemas/config/tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report dependencies without connecting to robot hardware")
    commands.add_parser("registry", help="List fixed catalog skills and deployment status")
    hardware = commands.add_parser("hardware-check", help="Read commissioning readiness without connecting to or starting a driver")
    hardware.add_argument("--commissioning")
    hardware.add_argument("--profile")
    template = commands.add_parser("hardware-template", help="Write an inadmissible commissioning template with unknown measurements left null")
    template.add_argument("--output", required=True)
    run = commands.add_parser("simulate", help="Run the real DAG executor against explicit synthetic task-state simulation")
    run.add_argument("--scenario", default="cabinet")
    run.add_argument("--context")
    run.add_argument("--plan")
    run.add_argument("--reasoner", choices=("fixture", "astra"), default="fixture")
    run.add_argument("--task")
    run.add_argument("--time-scale", type=float, default=0.02)
    run.add_argument("--inject", action="append", default=[], metavar="NODE:FAILURE_CODE")
    run.add_argument("--output", default="artifacts/simulation.json")
    run.add_argument("--trace", default="artifacts/trace.jsonl")
    physics = commands.add_parser("physics", help="Run actual Gen3 MuJoCo joint-dynamics replay, separate from task simulation")
    physics.add_argument("--output", default="artifacts/physics.json")
    physics.add_argument("--image", default="artifacts/gen3-simulation.png")
    physics.add_argument("--rolling", action="store_true", help="Test mid-motion suffix replacement against measured MuJoCo state")
    rolling = commands.add_parser("rolling", help="Exercise in-flight trajectory replacement using explicit joint test fixtures")
    rolling.add_argument("--output", default="artifacts/rolling.json")
    cameras = commands.add_parser("cameras", help="Read-only local camera discovery; never transmits frames")
    cameras.add_argument("--output", default="artifacts/cameras.json")
    cameras.add_argument("--capture-index", type=int, help="Read one local RGB preview from this explicit device index")
    cameras.add_argument("--image", default="artifacts/camera-preview.png")
    state = commands.add_parser("record-driver-state", help="Passively record existing ROS joint/EE publications; never commands or starts a driver")
    state.add_argument("--joint-topic", required=True)
    state.add_argument("--ee-topic", required=True)
    from .motion.driver_state import EE_MESSAGE_TYPES
    state.add_argument("--ee-message-type", required=True, choices=EE_MESSAGE_TYPES)
    state.add_argument("--duration-s", type=float, default=30.)
    state.add_argument("--max-pairs", type=int, default=10000)
    state.add_argument("--output-dir", required=True)
    calibration = commands.add_parser("record-calibration", help="Passively pair local marker observations with custom-driver publications; no motion or calibration approval")
    calibration.add_argument("--config", help="Explicit subscription mappings; defaults to config/calibration-recording.json")
    calibration.add_argument("--pose-label", required=True, help="Operator label for one recording window; does not certify stationarity")
    calibration.add_argument("--duration-s", type=float, default=15.)
    calibration.add_argument("--max-camera-samples", type=int, default=300)
    calibration.add_argument("--max-driver-pairs", type=int, default=10000)
    calibration.add_argument("--max-receipt-gap-s", type=float, default=.1, help="Diagnostic association bound, not acquisition-time uncertainty")
    calibration.add_argument("--output-dir", required=True)
    rgbd = commands.add_parser("capture-rgbd", help="Subscribe to explicit local ROS RGB-D topics; no drivers or robot transport are started")
    for name in ("rgb-topic", "depth-topic", "rgb-info-topic", "depth-info-topic"):
        rgbd.add_argument("--"+name, required=True)
    rgbd.add_argument("--output-dir", required=True, help="New local directory for one NPZ capture and metadata report")
    rgbd.add_argument("--samples", type=int, default=15)
    rgbd.add_argument("--timeout-s", type=float, default=20.)
    rgbd.add_argument("--max-skew-s", type=float, default=.02, help="Diagnostic pairing tolerance; does not establish capture clock calibration")
    rgbd.add_argument("--reliability", choices=("best_effort", "reliable"), default="best_effort", help="Use reliable only when the selected publishers offer reliable QoS")
    rgbd.add_argument("--inspect-marker", action="store_true", help="Observe the configured single marker in raw color; save provisional pose branches locally")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            _write_json(None, doctor())
        elif args.command == "hardware-check":
            from .motion.commissioning import commissioning_status
            _write_json(None, commissioning_status(args.commissioning or Catalog(args.root).root/"config/commissioning.json", args.profile))
        elif args.command == "hardware-template":
            from .motion.commissioning import profile_template
            target = Path(args.output)
            if target.exists():
                raise ValueError("Template output already exists")
            _write_json(target, profile_template(Catalog(args.root).root/"config/commissioning.json"))
        elif args.command == "registry":
            catalog = Catalog(args.root)
            _write_json(None, {"catalog_hash": catalog.hash,
                               "skills": [{"id": k, "hardware_status": v["implementation_status"], "claims": v["claims"],
                                           "capabilities_required": v["capabilities_required"]} for k, v in catalog.skills.items()]})
        elif args.command == "simulate":
            return asyncio.run(_run(args))
        elif args.command == "rolling":
            from .simulation import rolling_fixture_trace
            _write_json(args.output, rolling_fixture_trace())
        elif args.command == "physics":
            from .simulation import MujocoReplay, fixture_joint_trajectory
            simulator = MujocoReplay()
            if args.rolling:
                _write_json(args.output, simulator.replay_rolling_fixture())
            else:
                start = simulator.home()
                target = tuple(q + (0.08 if i == 1 else 0.0) for i, q in enumerate(start.position))
                stimulus = fixture_joint_trajectory(start, target, duration_s=2.0)
                _write_json(args.output, simulator.replay(stimulus, render_path=args.image))
        elif args.command == "cameras":
            from .perception import probe_cameras, capture_preview
            report = probe_cameras()
            if args.capture_index is not None:
                report["preview"] = capture_preview(args.capture_index, args.image)
            _write_json(args.output, report)
        elif args.command == "record-calibration":
            from .perception.calibration_recording import capture_calibration, read_recording_config
            root = Catalog(args.root).root
            report = capture_calibration(config=read_recording_config(args.config or root/"config/calibration-recording.json"),
                        marker_spec=strict_loads((root/"config/calibration-marker.json").read_bytes()),
                        output_dir=args.output_dir, pose_label=args.pose_label, duration_s=args.duration_s,
                        max_camera_samples=args.max_camera_samples, max_driver_pairs=args.max_driver_pairs,
                        max_receipt_gap_s=args.max_receipt_gap_s)
            _write_json(None, {"report": str(Path(args.output_dir)/"report.json"), "status": report["status"],
                        "driver_pairs": report["driver"]["paired_samples"],
                        "cameras": {name: {k: c[k] for k in ("samples_received", "observation_counts", "association_counts")}
                                    for name, c in report["cameras"].items()},
                        "calibration_samples_admitted": 0, "motion_commanded_by_recorder": False})
            return 0 if report["status"] == "recorded_unapproved" else 2
        elif args.command == "record-driver-state":
            from .motion.driver_diagnostics import capture_driver_state
            report = capture_driver_state(output_dir=args.output_dir, joint_topic=args.joint_topic,
                                          ee_topic=args.ee_topic, ee_message_type=args.ee_message_type,
                                          duration_s=args.duration_s, max_pairs=args.max_pairs)
            _write_json(None, report)
            return 0 if report["status"] == "captured" else 2
        elif args.command == "capture-rgbd":
            from .perception.ros_rgbd import capture_ros_rgbd
            observer = None
            if args.inspect_marker:
                from .perception.fiducial import SingleMarkerObserver
                observer = SingleMarkerObserver().observe
            report = capture_ros_rgbd(output_dir=args.output_dir, samples=args.samples, timeout_s=args.timeout_s,
                                      pair_observer=observer,
                                      max_skew_s=args.max_skew_s, reliability=args.reliability, rgb_topic=args.rgb_topic, depth_topic=args.depth_topic,
                                      rgb_info_topic=args.rgb_info_topic, depth_info_topic=args.depth_info_topic)
            _write_json(None, report)
            return 0 if report["status"] == "captured" else 2
        return 0
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(json.dumps({"status": "unavailable_or_rejected", "detail": str(exc), "physical_robot_commands_enabled": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
