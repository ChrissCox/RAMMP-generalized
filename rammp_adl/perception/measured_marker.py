"""Local single-marker observations connected to the canonical world writer.

Raw color detection does not require registered depth. Detection reports remain
provisional; robot-frame poses additionally require an explicit trusted pose
resolver, calibrated capture clock and known marker/entity attachment. No driver
is launched, image uploaded, world mutated or motion capability registered here.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading

import numpy as np

from ..contracts import ContractError, canonical_json, checked_copy, digest
from ..handlers import BackendFailure
from ..hardware_backend import ObservationMeasurement
from ..motion.leasing import drain_nonpreemptible
from ..world import MetricPose
from .calibration_target import read_local_capture
from .fiducial import SingleMarkerObserver, _validated_pose_candidates
from .geometry import CalibratedTransform, PerceptionError, positive_bounds, require_ids
from .ros_rgbd import RgbdPair


def _rigid(value):
    transform = np.array(value, dtype=float, copy=True)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0., 0., 0., 1.], rtol=0, atol=1e-9)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., rtol=0, atol=1e-6)):
        raise PerceptionError("Marker binding must be a finite rigid transform")
    return tuple(tuple(map(float, row)) for row in transform)


def _quaternion(rotation):
    """Stable matrix-to-xyzw conversion, including rotations near pi."""
    r = np.asarray(rotation)
    matrix = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[1, 0]+r[0, 1], r[2, 0]+r[0, 2], r[2, 1]-r[1, 2]],
        [r[1, 0]+r[0, 1], r[1, 1]-r[0, 0]-r[2, 2], r[2, 1]+r[1, 2], r[0, 2]-r[2, 0]],
        [r[2, 0]+r[0, 2], r[2, 1]+r[1, 2], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3.
    _, vectors = np.linalg.eigh(matrix)
    q = vectors[:, -1]
    if q[3] < 0:
        q = -q
    return tuple(map(float, q))


@dataclass(frozen=True)
class MarkerEntityBinding:
    """Trusted known marker attachment, never inferred from a detected ID.

    marker_from_entity maps the requested entity pose role into marker axes.
    A table marker therefore cannot identify a cabinet handle without a separate
    supported attachment/reference measurement. Identity is appropriate when the
    entity is the marker itself. The complete uncertainty belongs in the resolver.
    """
    entity_id: str
    marker_spec_digest: str
    pose_role: str
    marker_from_entity: tuple
    binding_evidence_id: str

    def __post_init__(self):
        require_ids(self.entity_id, self.marker_spec_digest, self.pose_role, self.binding_evidence_id)
        object.__setattr__(self, "marker_from_entity", _rigid(self.marker_from_entity))


@dataclass(frozen=True)
class MarkerPoseSolution:
    """Trusted resolver result for this exact capture, not a winning RMS guess.

    covariance_base is the final 6x6 covariance in base axes, including marker
    scale, corner/intrinsics, attachment, extrinsic and capture-time uncertainty.
    The resolver owns branch disambiguation and its evidence; lowest pixel error
    alone never resolves the single planar marker's ambiguity.
    """
    observation_digest: str
    branch_index: int
    branch_evidence_id: str
    camera_to_base: CalibratedTransform
    covariance_base: tuple
    uncertainty_evidence_id: str
    intrinsic_evidence_id: str
    scale_evidence_id: str

    def __post_init__(self):
        require_ids(self.observation_digest, self.branch_evidence_id, self.uncertainty_evidence_id,
                    self.intrinsic_evidence_id, self.scale_evidence_id)
        if type(self.branch_index) is not int or not 0 <= self.branch_index <= 1:
            raise PerceptionError("Resolver must identify an explicit square-pose branch")
        if not isinstance(self.camera_to_base, CalibratedTransform):
            raise PerceptionError("Resolver requires the canonical CalibratedTransform")
        values = tuple(self.covariance_base)
        if len(values) != 36 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise PerceptionError("Resolver requires a finite 6x6 base-frame covariance")
        object.__setattr__(self, "covariance_base", values)


class LocalMarkerObserver:
    """Warm raw-color detector and bounded latest-report runtime callback.

    Feed on_pair from a local RGB-D consumer or await run(source), where source
    is an already constructed RosRgbdSource. The optional clock mapper accepts
    the original RGB source stamp in seconds and returns (world_time, error_s).
    Without its explicit evidence ID the pipeline remains diagnostic only.
    pose_resolver(report, snapshot) is synchronous trusted local code, called only
    for purpose=pose. It may reject missing calibration or ambiguous branches.
    """
    def __init__(self, world, *, camera_role, camera_id, binding, max_age_s,
                 max_timestamp_uncertainty_s, clock_mapper=None, clock_evidence_id=None,
                 pose_resolver=None, detector=None, expected_frame_id=None,
                 expected_rgb_info_digest=None):
        if camera_role not in {"wrist", "scene"} or not isinstance(binding, MarkerEntityBinding):
            raise PerceptionError("A known camera role and marker/entity binding are required")
        require_ids(camera_id)
        positive_bounds(max_age_s, max_timestamp_uncertainty_s)
        if max_age_s > world.max_evidence_age_s:
            raise PerceptionError("Observation validity exceeds world policy")
        if (clock_mapper is None) != (clock_evidence_id is None):
            raise PerceptionError("Capture-clock mapper and evidence must be provided together")
        if clock_mapper is not None:
            if not callable(clock_mapper):
                raise PerceptionError("Capture-clock mapping must be callable")
            require_ids(clock_evidence_id)
        if pose_resolver is not None and not callable(pose_resolver):
            raise PerceptionError("Pose resolver must be trusted local callable code")
        if (expected_frame_id is None) != (expected_rgb_info_digest is None):
            raise PerceptionError("Expected camera frame and intrinsic digest must be supplied together")
        if expected_frame_id is not None:
            require_ids(expected_frame_id, expected_rgb_info_digest)
        self.world, self.camera_role, self.camera_id, self.binding = world, camera_role, camera_id, binding
        self.max_age_s, self.max_uncertainty_s = max_age_s, max_timestamp_uncertainty_s
        self.clock_mapper, self.clock_evidence_id = clock_mapper, clock_evidence_id
        self.pose_resolver = pose_resolver
        self.detector = detector if detector is not None else SingleMarkerObserver()
        if digest(self.detector.spec) != binding.marker_spec_digest:
            raise PerceptionError("Detector and entity marker definitions differ")
        self._lock, self._ingest_lock = threading.Lock(), threading.Lock()
        self._latest_json, self._failure = None, None
        self._last_stamp = -1
        self._last_capture = None
        # Pinning the first diagnostic signature detects later changes; it does
        # not verify intrinsic accuracy. A configured signature can reject the
        # wrong camera/calibration before its very first capture.
        self._camera_signature = ((expected_frame_id, expected_rgb_info_digest)
                                  if expected_frame_id is not None else None)
        self._source_invalidated = False
        self._running, self._stopping = False, asyncio.Event()

    def on_pair(self, pair):
        """Retain numeric evidence only; invalid/new missing observations replace old ones."""
        if self._stopping.is_set():
            raise PerceptionError("Marker observer is stopped; construct a new source session")
        if not isinstance(pair, RgbdPair):
            raise PerceptionError("Expected a local RgbdPair with original source metadata")
        if not self._ingest_lock.acquire(blocking=False):
            raise PerceptionError("Marker detector already has an active capture")
        try:
            metadata = pair.metadata
            stamp = metadata["source_stamps_ns"]["rgb"]
            if type(stamp) is not int or stamp <= 0:
                raise PerceptionError("Marker capture requires a nonzero acquisition stamp")
            require_ids(pair.capture_id)
            with self._lock:
                signature = (metadata["rgb_info"]["frame_id"], digest(metadata["rgb_info"]))
                if self._source_invalidated or (self._camera_signature is not None
                                               and signature != self._camera_signature):
                    self._source_invalidated = True
                    raise PerceptionError("Camera frame/intrinsics changed or capture clock invalidated; rebuild the source with current calibration")
                if stamp <= self._last_stamp or pair.capture_id == self._last_capture:
                    self._source_invalidated = True
                    raise PerceptionError("Marker capture clock reset, duplicate or out of order")
                self._camera_signature = signature
                self._last_stamp, self._last_capture = stamp, pair.capture_id
            report = self.detector.observe(pair)
            if (report["capture_id"] != pair.capture_id
                    or report["source_stamps_ns"] != metadata["source_stamps_ns"]
                    or report["rgb_info_digest"] != digest(metadata["rgb_info"])
                    or report["marker_spec_digest"] != self.binding.marker_spec_digest):
                raise PerceptionError("Detector result lost its exact capture provenance")
            report = checked_copy(report)
            encoded = canonical_json(report)
            if len(encoded.encode()) > 65536:
                raise PerceptionError("Marker report exceeds the bounded local evidence budget")
            with self._lock:
                if self._stopping.is_set():
                    raise PerceptionError("Marker observer stopped during local detection")
                self._latest_json, self._failure = encoded, None
            return checked_copy(report)
        except Exception as exc:
            self.on_failure(str(exc))
            raise
        finally:
            self._ingest_lock.release()

    def on_failure(self, detail):
        with self._lock:
            self._latest_json, self._failure = None, str(detail)[:1000]

    def latest_report(self):
        with self._lock:
            return None if self._latest_json is None else json.loads(self._latest_json)

    def _capture_time(self, report):
        if self.clock_mapper is None:
            raise BackendFailure("stale_state", "Camera capture clock has no verified world-clock mapping")
        stamp, uncertainty = self.clock_mapper(report["source_stamps_ns"]["rgb"] / 10**9)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (stamp, uncertainty))
                or not 0 <= uncertainty <= self.max_uncertainty_s):
            raise BackendFailure("stale_state", "Camera clock mapping lacks a bounded acquisition error")
        # The oldest plausible exposure controls validity; uncertainty cannot
        # silently lengthen freshness or place part of the exposure in the future.
        captured_at = stamp - uncertainty
        now = self.world.clock()
        if stamp + uncertainty > now or not 0 <= now-captured_at < self.max_age_s:
            raise BackendFailure("stale_state", "Camera observation is stale or future dated")
        return captured_at, stamp + uncertainty, stamp, uncertainty

    async def __call__(self, args, context):
        try:
            return self.measure(args, context)
        except (PerceptionError, ContractError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise BackendFailure("geometry_invalid", str(exc)) from exc

    def measure(self, args, context):
        if self._stopping.is_set():
            raise BackendFailure("cancelled", "Marker observation source is stopping or closed")
        binding = self.binding
        if (args.get("entity_id"), args.get("camera")) != (binding.entity_id, self.camera_role):
            raise BackendFailure("no_detection", "Requested entity/view lacks this exact marker attachment")
        if args.get("purpose") not in {"state", "pose"}:
            raise BackendFailure("geometry_invalid", "Marker visibility does not establish grasp, articulation or food state")
        snapshot = context.snapshot
        if snapshot is None or context.execution_epoch != snapshot.execution_epoch or context.cancel_event.is_set():
            raise BackendFailure("cancelled", "Marker observation has no active input snapshot")
        if snapshot.context["task_id"] != context.task_id:
            raise BackendFailure("stale_state", "Marker observation task identity differs")
        identities = snapshot.identities()
        keys = ("execution_epoch", "calibration_id", "base_epoch", "entity:" + binding.entity_id)
        dependencies = {name: identities[name] for name in keys}
        current = self.world.snapshot().identities()
        if any(current.get(k) != v for k, v in dependencies.items()):
            raise BackendFailure("stale_state", "Marker observation dependencies changed")
        with self._lock:
            report = None if self._latest_json is None else json.loads(self._latest_json)
            failure = self._failure
        if report is None:
            raise BackendFailure("no_detection", failure or "No local marker capture is available")
        if report["status"] != "provisional_pose_candidates":
            raise BackendFailure("no_detection", "Requested marker is absent, duplicated or lacks usable pose candidates")
        candidates = _validated_pose_candidates(report)
        captured_at, latest_time, estimate, uncertainty = self._capture_time(report)
        evidence_id = "marker-measurement-" + digest({"report": report, "camera_id": self.camera_id,
            "dependencies": dependencies, "node": context.node_id, "attempt": context.attempt})
        assertions = [
            {"predicate": "entity_exists", "args": {"entity_id": binding.entity_id}, "validity": "true"},
            {"predicate": "observation_valid", "args": {"entity_id": binding.entity_id, "purpose": args["purpose"]}, "validity": "true"},
        ]
        data = {"kind": "local_marker_visibility", "observation": report, "camera_id": self.camera_id,
                "binding_evidence_id": binding.binding_evidence_id, "clock_evidence_id": self.clock_evidence_id,
                "mapped_capture_estimate_s": estimate, "timestamp_uncertainty_s": uncertainty,
                "capture_validity_uses_oldest_bound": True, "frames_uploaded": 0,
                "collision_geometry_complete": False, "physical_contact_state_measured": False}
        poses = ()
        if args["purpose"] == "pose":
            if self.pose_resolver is None:
                raise BackendFailure("geometry_invalid", "Robot reference, uncertainty and planar-branch evidence are required")
            solution = self.pose_resolver(checked_copy(report), snapshot)
            if not isinstance(solution, MarkerPoseSolution) or solution.observation_digest != digest(report):
                raise PerceptionError("Pose resolver did not bind this exact observation")
            if solution.branch_index >= len(candidates):
                raise PerceptionError("Pose resolver selected an absent candidate")
            tf = solution.camera_to_base
            if self.camera_role == "wrist" and tf.rigid_static:
                raise PerceptionError("Wrist-to-base transform must be resolved at capture time")
            if tf.target_frame != self.world.catalog.library["frames"]["planning"]:
                raise PerceptionError("Marker pose is missing its robot planning-frame reference")
            camera_from_entity = candidates[solution.branch_index] @ np.asarray(binding.marker_from_entity)
            for at in (captured_at, latest_time):
                tf.apply(camera_from_entity[:3, 3], capture_time=at, source_frame=report["camera_frame"],
                         calibration_id=identities["calibration_id"], base_epoch=identities["base_epoch"])
            position = tf.rotation @ camera_from_entity[:3, 3] + tf.translation_m
            orientation = _quaternion(tf.rotation @ camera_from_entity[:3, :3])
            pose = MetricPose(binding.entity_id, binding.pose_role, tuple(map(float, position)), orientation,
                solution.covariance_base, captured_at, tf.target_frame, identities["entity:" + binding.entity_id] + 1,
                identities["calibration_id"], identities["base_epoch"], evidence_id, self.max_age_s)
            poses = (pose,)
            assertions.append({"predicate": "pose_valid", "args": {"entity_id": binding.entity_id,
                "pose_role": binding.pose_role}, "validity": "true"})
            data.update(kind="local_marker_pose", selected_branch=solution.branch_index,
                branch_evidence_id=solution.branch_evidence_id, uncertainty_evidence_id=solution.uncertainty_evidence_id,
                intrinsic_evidence_id=solution.intrinsic_evidence_id, scale_evidence_id=solution.scale_evidence_id)
        if self._stopping.is_set():
            raise BackendFailure("cancelled", "Marker observation source stopped during local evaluation")
        return ObservationMeasurement(binding.entity_id, self.camera_role, args["purpose"], evidence_id,
            captured_at, self.max_age_s, tuple(assertions), data, dependencies, metric_poses=poses)

    async def run(self, source):
        """Continuously capture and inspect locally with one retained detector.

        Source creation/launch is the caller's responsibility. Closing waits for
        bounded non-preemptible capture/inspection before source destruction.
        """
        if self._running or self._stopping.is_set():
            raise PerceptionError("Marker capture loop is already running or stopped; construct a new source session")
        self._running = True
        work = None
        try:
            while not self._stopping.is_set():
                try:
                    work = asyncio.create_task(asyncio.to_thread(source.capture, 1000))
                    pair = await asyncio.shield(work)
                    if self._stopping.is_set():
                        break
                    work = asyncio.create_task(asyncio.to_thread(self.on_pair, pair))
                    await asyncio.shield(work)
                except (PerceptionError, RuntimeError, ValueError, KeyError) as exc:
                    self.on_failure(str(exc))
                    try:
                        await asyncio.wait_for(self._stopping.wait(), timeout=.1)
                    except asyncio.TimeoutError:
                        pass
        finally:
            self._stopping.set()
            if work is not None and not work.done():
                try:
                    await drain_nonpreemptible(work)
                except Exception:
                    pass
            self.on_failure("Local marker source is stopping; observation is unavailable")
            try:
                close = asyncio.create_task(asyncio.to_thread(source.close))
                try:
                    await asyncio.shield(close)
                except asyncio.CancelledError:
                    try:
                        await drain_nonpreemptible(close)
                    finally:
                        raise
            finally:
                self._running = False

    def stop(self):
        self._stopping.set()
        self.on_failure("Local marker source was stopped; observation is unavailable")


def replay_capture(path, *, detector=None):
    """Reinspect archived raw color; retain all branches and original timestamps."""
    rgb, metadata = read_local_capture(path)
    observer = detector if detector is not None else SingleMarkerObserver()
    report = observer.inspect(rgb, metadata, capture_id=metadata["capture_id"])
    return {"status": "historical_local_marker_replay", "observation": report,
        "historical_capture": True, "world_model_committed": False,
        "robot_reference_available": False, "hardware_commands": False, "frames_uploaded": 0,
        "limitations": ["No capture-clock mapping or robot extrinsics supplied",
            "All planar pose branches retained; nominal marker scale", "No full-scene collision reconstruction"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = replay_capture(args.capture)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"report": str(destination), "status": report["status"],
                          "branches": len(report["observation"]["pose_candidates"]), "frames_uploaded": 0}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(json.dumps({"status": "rejected", "detail": str(exc), "hardware_commands": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
