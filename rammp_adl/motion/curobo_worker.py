"""Bounded one-shot cuRobo planning worker, with no execution transport.

The caller owns process isolation and the motion-state, world/model and full
timed-path admission checks. This worker never reads camera/robot devices,
imports ROS, opens a control connection, or claims hardware admission.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from rammp_adl.contracts import digest, strict_loads
from .curobo import CUROBO_VERSION, RAMMP_COMMIT, RammpCuroboAdapter
from .rolling import JointState, MotionError


def parse_request(value):
    if not isinstance(value, dict) or set(value) != {"start", "goal", "world", "world_identity"}:
        raise MotionError("Planning request must contain only start, goal, world, world_identity")
    start = value["start"]
    if not isinstance(start, dict) or set(start) != {"position", "velocity", "acceleration"}:
        raise MotionError("Planning start requires explicit position, velocity and acceleration")
    if any(not isinstance(start[key], list) or len(start[key]) != 7 for key in start):
        raise MotionError("Planning start must contain seven joints in configured controller order")
    state = JointState(**start)
    if any(abs(v) > 1e-8 for v in (*state.velocity, *state.acceleration)):
        raise MotionError("Static worker cannot plan from a moving start")
    goal = value["goal"]
    if not isinstance(goal, dict) or set(goal) != {"position_m", "quaternion_xyzw"}:
        raise MotionError("Planning goal must contain position_m and quaternion_xyzw")
    for key, size in (("position_m", 3), ("quaternion_xyzw", 4)):
        if (not isinstance(goal[key], list) or len(goal[key]) != size
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in goal[key])):
            raise MotionError("Planning goal requires finite metric coordinates")
    if abs(sum(v*v for v in goal["quaternion_xyzw"]) - 1.) > 1e-4:
        raise MotionError("Planning goal quaternion is not normalized")
    world = value["world"]
    if not isinstance(world, list) or not 1 <= len(world) <= 256 or not all(isinstance(v, dict) for v in world):
        raise MotionError("Planning world must be a bounded nonempty list of wrapper obstacle records")
    if value["world_identity"] != digest(world):
        raise MotionError("Planning world identity does not match the supplied world")
    return state


def file_digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialize_trajectory(path):
    return {
        "joint_names": list(path.joint_names), "provenance": path.provenance, "digest": path.digest,
        "interpolation": "quintic-hermite-v1",
        "points": [{"time_s": point.time_s, "position": list(point.state.position),
                    "velocity": list(point.state.velocity), "acceleration": list(point.state.acceleration)}
                   for point in path.points],
    }


async def plan_request(request, *, source_root, planner_config):
    """Return a candidate; callers must independently certify before dispatch."""
    parse_request(request)
    began = time.monotonic()
    adapter = await RammpCuroboAdapter.load(source_root=source_root, planner_config=planner_config)
    loaded = time.monotonic()
    return await plan_loaded(request, adapter=adapter, planner_config=planner_config,
                             initialization_wall_s=loaded-began)


async def plan_loaded(request, *, adapter, planner_config, initialization_wall_s=0.):
    """Caller retains exclusive ownership through solve and provenance capture."""
    state = parse_request(request)
    loaded = time.monotonic()
    path = await adapter.plan_pose(**request["goal"], start=state,
                                   world=request["world"], world_identity=request["world_identity"])
    planned = time.monotonic()
    planner = adapter.planner
    position, quaternion = planner.fk(path.points[-1].state.position, quat_order="xyzw")
    kin = planner._robot_cfg["kinematics"]
    urdf = Path(kin["urdf_path"])
    if not urdf.is_absolute() or not urdf.is_file():
        raise MotionError("Worker requires an explicit absolute planner URDF for provenance")
    limits = planner.joint_limits()
    return {
        "status": "planned", "hardware_commands": False, "hardware_validated": False,
        "curobo_planned": True, "independent_validation_required": True,
        "request_digest": digest(request), "world_identity": request["world_identity"],
        "trajectory": serialize_trajectory(path),
        "planner_source_commit": RAMMP_COMMIT, "curobo_version": CUROBO_VERSION,
        "planner_config_digest": file_digest(planner_config),
        "planner_robot_config": planner._robot_cfg,
        "planner_robot_config_digest": digest(planner._robot_cfg),
        "planner_urdf_digest": file_digest(urdf),
        "base_frame": kin["base_link"], "ee_link": kin["ee_link"],
        "joint_limits": {key: value.tolist() for key, value in limits.items()},
        "endpoint_fk": {"position_m": position, "quaternion_xyzw": quaternion},
        "initialization_wall_s": initialization_wall_s, "planning_wall_s": planned - loaded,
        "interpolation_evidence": planner.boundary_debug,
        "validation_scope": "Planner candidate and cuRobo sample checks; no independent swept/stopping proof or hardware admission",
    }


async def serve(*, source_root, planner_config, input_stream, output_stream):
    """One retained GPU planner, sequential bounded JSON-lines requests.

    stdout is reserved for protocol; library logs belong on stderr. No ROS,
    command transport, background speculative solves or parallel world changes.
    An invalid frame terminates this session so queued data cannot be retagged.
    """
    def send(value):
        output_stream.write(json.dumps(value, allow_nan=False)+"\n")
        output_stream.flush()
    with contextlib.redirect_stdout(sys.stderr):
        began = time.monotonic()
        adapter = await RammpCuroboAdapter.load(source_root=source_root, planner_config=planner_config)
        initialization = time.monotonic()-began
    send({"protocol": 1, "status": "ready", "hardware_commands": False,
          "planner_source_commit": RAMMP_COMMIT, "curobo_version": CUROBO_VERSION,
          "planner_config_digest": file_digest(planner_config),
          "initialization_wall_s": initialization})
    expected_id = 1
    while True:
        line = input_stream.readline(1048577)
        if not line:
            return
        try:
            frame = strict_loads(line, max_bytes=1048576)
            if (not isinstance(frame, dict) or set(frame) != {"request_id", "request"}
                    or type(frame["request_id"]) is not int or frame["request_id"] != expected_id):
                raise MotionError("Expected next sequential request_id and request only")
        except (ValueError, RuntimeError) as error:
            send({"protocol": 1, "status": "protocol_error", "reason": str(error), "hardware_commands": False})
            return
        expected_id += 1
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = await plan_loaded(frame["request"], adapter=adapter, planner_config=planner_config)
        except (ValueError, RuntimeError, OSError) as error:
            result = {"status": "rejected", "reason": str(error), "hardware_commands": False,
                      "hardware_validated": False, "independent_validation_required": True}
        send({"protocol": 1, "request_id": frame["request_id"], "result": result})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--planner-config", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serve", action="store_true", help="Retain a warm planner; bounded sequential requests on stdin/stdout")
    args = parser.parse_args(argv)
    if args.serve:
        if args.request is not None or args.output is not None:
            parser.error("--serve uses stdin/stdout, not --request/--output")
        # Preserve a protocol fd and send Python AND native CUDA/library stdout
        # to the log stream. Python redirect_stdout alone misses native writes.
        with os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1) as protocol:
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            asyncio.run(serve(source_root=args.wrapper_source, planner_config=args.planner_config,
                              input_stream=sys.stdin, output_stream=protocol))
        return 0
    if args.request is None or args.output is None:
        parser.error("one-shot mode requires --request and --output")
    if args.output.exists():
        raise MotionError("Worker output must be a new file")
    if args.request.stat().st_size > 1048576:
        raise MotionError("Planning request exceeds one MiB")
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        result = asyncio.run(plan_request(request, source_root=args.wrapper_source, planner_config=args.planner_config))
    except (ValueError, RuntimeError, OSError) as error:
        result = {"status": "rejected", "reason": str(error), "hardware_commands": False,
                  "hardware_validated": False, "independent_validation_required": True}
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "planned" else 2


if __name__ == "__main__":
    raise SystemExit(main())
