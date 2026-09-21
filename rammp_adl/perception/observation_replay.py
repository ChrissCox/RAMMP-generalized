"""Recorded local camera evidence through task reasoning and the real DAG runtime.

This is an observation-only simulation composition. The acquisition stamps are
unchanged and the world uses an explicitly frozen playback clock; historical
imagery never becomes fresh physical evidence. Hold and clock correspondence are
simulation fixtures. Robot geometry, motion and physical task state remain absent.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from ..app import astra_for
from ..contracts import Catalog, ContractError, checked_copy, digest, strict_loads
from ..hardware_backend import observation_runtime
from ..world import WorldModel
from .calibration_target import read_local_capture
from .fiducial import SingleMarkerObserver
from .geometry import PerceptionError
from .measured_marker import LocalMarkerObserver, MarkerEntityBinding
from .ros_rgbd import RgbdPair


class RecordingReasoner:
    """Retain numeric/text task inputs and proposals, using the existing gateway."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.records = []

    async def generate_plan(self, context, **kwargs):
        if kwargs.get("images"):
            raise ContractError("Recorded observation rehearsal never uploads imagery")
        record = {"world_context": checked_copy(context), "task_text": kwargs.get("task_text", ""),
                  "feedback": checked_copy(kwargs.get("feedback")), "images_uploaded": 0}
        self.records.append(record)
        response = await self.delegate.generate_plan(context, **kwargs)
        record.update(status=response.status, detail=response.detail, model_id=response.model_id,
                      provider_requests=response.provider_requests,
                      plan=checked_copy(response.plan) if response.plan is not None else None)
        return response


def marker_context(catalog, observation, *, camera_role, entity_id=None):
    """A canonical instance describing just the observed marker and simulated hold."""
    spec = observation["marker_spec"]
    entity_id = entity_id or f"marker_{spec['dictionary']}_{spec['marker_id']}"
    if not isinstance(entity_id, str) or not entity_id or entity_id == "robot":
        raise ContractError("Marker requires a stable non-robot entity identity")
    evidence_id = "recorded-marker-exists-" + digest(observation)
    def entity(identity, label, facts, sources):
        return {"entity_id": identity, "label": label, "entity_revision": 0, "pose_roles": [],
                "facts": facts, "confidence": 0., "age_s": 0., "position_validity": "unknown",
                "orientation_validity": "unknown", "source_ids": sources}
    # confidence=0 means no calibrated probabilistic confidence was supplied;
    # matching the configured marker ID supports the scoped existence assertion.
    marker_fact = {"predicate": "entity_exists", "args": {"entity_id": entity_id},
                   "validity": "true", "evidence_id": evidence_id, "age_s": 0.}
    held_fact = {"predicate": "held_state", "args": {"robot_id": "robot"}, "validity": "true",
                 "evidence_id": "SIMULATION-held-no-robot-transport", "age_s": 0.}
    context = {"schema_version": "1.0.0", "task_id": "recorded-marker-observation",
        "snapshot_id": "recorded-marker-initial-" + digest(observation)[:16],
        "execution_epoch": 1, "revision": 0, "collision_revision": 0,
        "base_epoch": "SIMULATION-no-robot-base-reference", "calibration_id": "SIMULATION-playback-no-extrinsics",
        "robot_config_id": "SIMULATION-observation-only", "attachment_id": "empty", "grasp_state_id": "empty",
        "entities": [entity(entity_id,
            f"Configured {spec['dictionary']} ID {spec['marker_id']} marker seen in recorded {camera_role} camera raw color; robot pose unknown",
            [marker_fact], [evidence_id]),
            entity("robot", "SIMULATION held-state fixture; no physical robot transport", [held_fact], [held_fact["evidence_id"]])],
        "constraints": [], "profiles": [], "available_skills": ["observe"], "completed_nodes": [],
        "hazards": ["Historical camera playback, not fresh physical observations",
                    "Robot calibration, full scene collision geometry and physical hold are unavailable"],
        "goal": {"predicate": "observation_valid", "args": {"entity_id": entity_id, "purpose": "state"}}}
    return context


def local_plan(catalog, snapshot, *, entity_id, camera_role):
    """Return a reviewable plan artifact; sequencing remains in the canonical DAG."""
    return {"schema_version": "1.0.0", "skill_library_hash": catalog.hash,
        "task_id": snapshot.context["task_id"], "snapshot_id": snapshot.snapshot_id,
        "execution_epoch": snapshot.execution_epoch,
        "nodes": [{"id": "observe-recorded-marker", "skill": "observe",
                   "args": {"entity_id": entity_id, "camera": camera_role, "purpose": "state"}}], "edges": []}


def build_replay(capture, *, camera_role="scene", root=None):
    if camera_role not in {"scene", "wrist"}:
        raise ContractError("Explicit scene or wrist camera role is required")
    capture = Path(capture)
    rgb, metadata = read_local_capture(capture)
    # Depth stays unused and local: color marker visibility does not establish
    # depth registration, object geometry, orientation or a collision model.
    pair = RgbdPair(metadata["capture_id"], rgb, np.empty((0, 0)), json.dumps(metadata))
    detector = SingleMarkerObserver()
    observation = detector.observe(pair)
    if observation["status"] != "provisional_pose_candidates":
        raise PerceptionError("Recorded capture has no unique configured marker; no known entity can be bootstrapped")
    catalog = Catalog(root)
    context = marker_context(catalog, observation, camera_role=camera_role)
    source_stamp = observation["source_stamps_ns"]["rgb"] / 10**9
    playback_time = source_stamp + .1
    world = WorldModel(context, catalog, clock=lambda: playback_time,
                       trust_initial=False, max_evidence_age_s=5.)
    entity_id = context["goal"]["args"]["entity_id"]
    # Initial visibility comes from local image analysis, not the context JSON.
    marker_source = world.authorize_source("local_recorded_marker_detector", ["entity_exists"])
    marker_fact = context["entities"][0]["facts"][0]
    world.register_evidence(marker_fact["evidence_id"], source=marker_source, predicates=[marker_fact],
        observed_at=source_stamp, ttl_s=4., data={"observation": observation, "historical_playback": True})
    held_source = world.authorize_source("SIMULATION-held-state-fixture", ["held_state"])
    held_fact = context["entities"][1]["facts"][0]
    world.register_evidence(held_fact["evidence_id"], source=held_source, predicates=[held_fact],
        observed_at=source_stamp, ttl_s=4., data={"simulation_only": True, "physical_hold_measured": False})
    binding = MarkerEntityBinding(entity_id, observation["marker_spec_digest"], "viewpoint",
        tuple(map(tuple, np.eye(4))), "marker-itself-identity-frame-no-object-attachment")
    observer = LocalMarkerObserver(world, camera_role=camera_role,
        camera_id="recorded-" + observation["camera_frame"], binding=binding,
        max_age_s=4., max_timestamp_uncertainty_s=.02,
        clock_mapper=lambda stamp: (stamp, 0.),
        clock_evidence_id="SIMULATION-frozen-source-clock-playback-NOT-live-clock-calibration",
        detector=detector, expected_frame_id=observation["camera_frame"],
        expected_rgb_info_digest=observation["rgb_info_digest"])
    observer.on_pair(pair)
    runtime = observation_runtime(world, observer=observer, held_check=lambda: True,
        capabilities={"local_geometry", "cloud_grounding"}, mode="simulation")
    executor = runtime.executor
    # Existing consent-policy control flow is exercised with explicit simulated
    # assent. This cannot approve real image egress; no image registry is present.
    executor.confirmation_callback = lambda plan, node, bound_digest: executor.confirmations.grant(
        task_id=plan["task_id"], epoch=plan["execution_epoch"], node_id=node["id"],
        digest=bound_digest, ttl_s=1., accepted=True).confirmation_id
    return runtime, observer, {"capture_file": str(capture.resolve()),
        "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(), "observation": observation,
        "playback_clock_s": playback_time, "playback_clock_frozen": True,
        "physical_capture_clock_calibrated": False, "physical_robot_hold_measured": False,
        "entity_id": entity_id, "camera_role": camera_role}


async def run_replay(capture, *, output_dir, camera_role="scene", reasoner="local", plan=None,
                     task_text=None, transport=None, root=None):
    if reasoner not in {"local", "astra"}:
        raise ContractError("Reasoner must be local or astra")
    if reasoner == "astra" and plan is not None:
        raise ContractError("Choose either a supplied local plan or Astra planning")
    if reasoner == "local" and transport is not None:
        raise ContractError("Local replay cannot construct a provider transport")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    runtime, observer, provenance = build_replay(capture, camera_role=camera_role, root=root)
    def save(name, value):
        (destination / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    initial = runtime.world.snapshot()
    save("initial-context.json", initial.context)
    save("capture-evidence.json", provenance)
    proposals, reasoning_records = [], []
    if reasoner == "astra":
        delegate = astra_for(runtime, transport=transport)
        recorded = RecordingReasoner(delegate)
        reasoning_records = recorded.records
        text = task_text or (f"Confirm visibility of marker {provenance['entity_id']} using the {camera_role} camera "
                             "in this recorded scene. Complete the normalized observation goal; this is simulation playback.")
        try:
            result = await runtime.executor.run_task(text, recorded)
        finally:
            await delegate.close()
            save("reasoning-records.json", recorded.records)
        proposals = [r["plan"] for r in recorded.records if r.get("plan") is not None]
    else:
        chosen = strict_loads(Path(plan).read_bytes()) if isinstance(plan, (str, Path)) else checked_copy(plan)
        if chosen is None:
            chosen = local_plan(runtime.catalog, initial, entity_id=provenance["entity_id"], camera_role=camera_role)
        proposals = [chosen]
        result = await runtime.executor.run_plan(chosen)
    save("plans.json", proposals)
    final = runtime.world.snapshot()
    report = {"result": result.to_dict(), "reasoner": reasoner,
        "provider_request_count": sum(r.get("provider_requests", 0) for r in reasoning_records),
        "provider_successful_proposal": (reasoner == "astra" and transport is None
                                          and any(r.get("status") == "OK" for r in reasoning_records)),
        "provider_transport_is_test_double": transport is not None,
        "model_id": "gpt-6-astra" if reasoner == "astra" else None,
        "timing": runtime.trace.summary(), "source": provenance,
        "final_world": final.context, "metric_pose_count": len(final.metric_poses),
        "frames_uploaded": 0, "hardware_commands": False, "simulation_only": True,
        "validation_scope": "Recorded camera marker detection through canonical task DAG validation and atomic world effects; frozen playback clock and simulated hold",
        "limitations": ["Historical image is not fresh physical evidence", "No robot reference or metric pose is admitted",
            "No arm/gripper command transport, cuRobo motion or contact physics is constructed",
            "Astra receives compact numeric/text world context only; marker pose branches and RGB/depth stay local"]}
    save("result.json", report)
    runtime.trace.write(destination / "trace.jsonl")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--camera-role", choices=("scene", "wrist"), default="scene")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reasoner", choices=("local", "astra"), default="local")
    parser.add_argument("--plan", help="Optional canonical local DAG JSON; no automatic identity retagging")
    parser.add_argument("--task")
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(run_replay(args.capture, output_dir=args.output_dir, camera_role=args.camera_role,
            reasoner=args.reasoner, plan=args.plan, task_text=args.task))
        print(json.dumps({"report": str(Path(args.output_dir) / "result.json"), "result": report["result"],
            "reasoner": report["reasoner"], "simulation_only": True, "frames_uploaded": 0}, indent=2))
        return 0 if report["result"]["status"] == "succeeded" else 2
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(json.dumps({"status": "rejected", "detail": str(exc), "hardware_commands": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
