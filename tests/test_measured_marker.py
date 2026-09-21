"""Actual OpenCV marker detection, synthetic calibration; no robot or camera I/O."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import threading
import unittest

import numpy as np

from rammp_adl.contracts import Catalog, digest
from rammp_adl.handlers import BackendFailure, ExecutionContext
from rammp_adl.hardware_backend import observation_runtime
from rammp_adl.perception.fiducial import SingleMarkerObserver
from rammp_adl.perception.geometry import CalibratedTransform, PerceptionError
from rammp_adl.perception.measured_marker import (
    LocalMarkerObserver, MarkerEntityBinding, MarkerPoseSolution, _quaternion,
)
from rammp_adl.perception.ros_rgbd import RgbdPair
from rammp_adl.world import WorldModel

from test_ros_rgbd import pair

ROOT = Path(__file__).resolve().parents[1]
ENTITY = "cabinet_handle_1"


class MeasuredMarkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.catalog = Catalog(ROOT)
        self.now = 100.
        context = json.loads((ROOT / "examples/cabinet.context.json").read_text())
        context["available_skills"] = ["observe"]
        self.world = WorldModel(context, self.catalog, clock=lambda: self.now,
                                trust_initial=True, max_evidence_age_s=5.)
        self.detector = SingleMarkerObserver()
        self.binding = MarkerEntityBinding(ENTITY, digest(self.detector.spec), "pregrasp",
            tuple(map(tuple, np.eye(4))), "SYNTHETIC-marker-attached-to-fixture-entity")
        meta = pair().metadata
        meta["rgb_info"].update(width=640, height=480,
            k=[600., 0., 320., 0., 600., 240., 0., 0., 1.], d=[0.] * 5)
        self.metadata = meta
        gray = self.detector.cv2.aruco.generateImageMarker(
            self.detector.cv2.aruco.getPredefinedDictionary(self.detector.cv2.aruco.DICT_4X4_50), 0, 150)
        image = np.full((480, 640, 3), 255, np.uint8)
        image[165:315, 245:395] = gray[..., None]
        self.sample = RgbdPair("synthetic-capture", image, np.full((480, 640), np.nan), json.dumps(meta))
        self.args = {"entity_id": ENTITY, "camera": "scene", "purpose": "pose"}

    def context(self):
        snap = self.world.snapshot()
        return ExecutionContext(snap.context["task_id"], "observe-marker", snapshot=snap,
                                execution_epoch=snap.execution_epoch)

    def solution(self, report, snapshot):
        # Fixture disambiguation is explicitly supplied; the production adapter
        # has no pixel-error winner rule. Exact camera matrix and marker pose are
        # synthetic, and the transform/covariance are synthetic test inputs.
        tf = CalibratedTransform(report["camera_frame"], "base_link", np.eye(3), np.array([.2, .3, .4]),
            snapshot.context["calibration_id"], snapshot.context["base_epoch"], 90., 20.)
        return MarkerPoseSolution(digest(report), 0, "SYNTHETIC-branch-proof", tf,
            tuple(map(float, (np.eye(6) * .0001).ravel())), "SYNTHETIC-uncertainty",
            "SYNTHETIC-intrinsics", "SYNTHETIC-scale")

    def observer(self, **kwargs):
        options = dict(camera_role="scene", camera_id="SYNTHETIC-camera", binding=self.binding,
            max_age_s=4., max_timestamp_uncertainty_s=.02, clock_mapper=lambda stamp: (stamp + 89.7, .005),
            clock_evidence_id="SYNTHETIC-clock", pose_resolver=self.solution, detector=self.detector)
        options.update(kwargs)
        return LocalMarkerObserver(self.world, **options)

    async def test_real_detection_proposes_geometry_without_mutating_world(self):
        observer = self.observer()
        report = observer.on_pair(self.sample)
        before = self.world.snapshot()
        measured = await observer(self.args, self.context())
        self.assertEqual(len(report["pose_candidates"]), 2)
        self.assertEqual(len(measured.metric_poses), 1)
        self.assertEqual(self.world.snapshot().snapshot_id, before.snapshot_id)
        self.assertEqual(dict(self.world.snapshot().metric_poses), {})
        pose = measured.metric_poses[0]
        np.testing.assert_allclose(pose.position_m,
            np.asarray(report["pose_candidates"][0]["camera_from_marker"])[:3, 3] + [.2, .3, .4])
        self.assertAlmostEqual(pose.captured_at, 99.695)
        self.assertEqual(pose.entity_revision, before.identities()["entity:" + ENTITY] + 1)
        self.assertEqual(measured.data["observation"]["source_stamps_ns"], self.metadata["source_stamps_ns"])
        self.assertFalse(measured.data["collision_geometry_complete"])
        self.assertEqual(measured.data["frames_uploaded"], 0)

    async def test_pose_commits_via_existing_dag_atomic_world_writer(self):
        observer = self.observer()
        observer.on_pair(self.sample)
        runtime = observation_runtime(self.world, observer=observer, held_check=lambda: True,
            capabilities={"local_geometry", "cloud_grounding"}, mode="simulation")
        executor = runtime.executor
        executor.confirmation_callback = lambda plan, node, bound_digest: executor.confirmations.grant(
            task_id=plan["task_id"], epoch=plan["execution_epoch"], node_id=node["id"],
            digest=bound_digest, ttl_s=1., accepted=True).confirmation_id
        snap = self.world.snapshot()
        plan = {"schema_version": "1.0.0", "skill_library_hash": self.catalog.hash,
            "task_id": snap.context["task_id"], "snapshot_id": snap.snapshot_id,
            "execution_epoch": snap.execution_epoch,
            "nodes": [{"id": "watch", "skill": "observe", "args": self.args}], "edges": []}
        result = await executor.run_plan(plan)
        self.assertEqual(result.nodes[0].status, "succeeded", result.to_dict())
        self.assertIsNotNone(result.nodes[0].commit_receipt)
        pose = self.world.snapshot().metric_poses[(ENTITY, "pregrasp")]
        self.assertAlmostEqual(pose.captured_at, 99.695)
        self.assertNotEqual(result.status, "succeeded")

    async def test_no_capture_clock_never_uses_receipt_time(self):
        observer = self.observer(clock_mapper=None, clock_evidence_id=None)
        observer.on_pair(self.sample)
        with self.assertRaisesRegex(BackendFailure, "capture clock"):
            await observer(self.args, self.context())
        self.assertIsNotNone(observer.latest_report())

    async def test_pose_requires_explicit_base_reference_and_branch_evidence(self):
        observer = self.observer(pose_resolver=None)
        observer.on_pair(self.sample)
        with self.assertRaisesRegex(BackendFailure, "Robot reference"):
            await observer(self.args, self.context())
        measured = await observer({**self.args, "purpose": "state"}, self.context())
        self.assertEqual(measured.metric_poses, ())
        self.assertTrue(all(a["predicate"] != "pose_valid" for a in measured.assertions))

    async def test_wrong_entity_camera_and_unmeasured_purpose_rejected(self):
        observer = self.observer()
        observer.on_pair(self.sample)
        for changed in ({"entity_id": "unbound-handle"}, {"camera": "wrist"}, {"purpose": "grasp"},
                        {"purpose": "food"}, {"purpose": "articulation"}):
            with self.subTest(changed=changed), self.assertRaises(BackendFailure):
                await observer({**self.args, **changed}, self.context())

    async def test_capture_interval_stale_future_or_uncertain_rejected(self):
        for mapped in ((95., .01), (100., .001), (99., .1), (float("nan"), .01), (99., -1)):
            with self.subTest(mapped=mapped):
                observer = self.observer(clock_mapper=lambda stamp: mapped)
                observer.on_pair(self.sample)
                with self.assertRaises(BackendFailure):
                    await observer(self.args, self.context())

    async def test_ambiguous_duplicate_id_loss_and_clock_reset_invalidate_latest(self):
        observer = self.observer()
        observer.on_pair(self.sample)
        with self.assertRaisesRegex(PerceptionError, "duplicate or out of order"):
            observer.on_pair(self.sample)
        self.assertIsNone(observer.latest_report())
        for duplicate in (False, True):
            observer = self.observer()
            rgb = self.sample.rgb.copy()
            if duplicate:
                rgb[165:315, 50:200] = rgb[165:315, 245:395]
            else:
                rgb[:] = 255
            observer.on_pair(replace(self.sample, rgb=rgb))
            with self.assertRaises(BackendFailure):
                await observer(self.args, self.context())

    async def test_new_blank_frame_cannot_reuse_prior_detection(self):
        observer = self.observer()
        observer.on_pair(self.sample)
        meta = self.sample.metadata
        meta["source_stamps_ns"] = {k: v + 100000000 for k, v in meta["source_stamps_ns"].items()}
        observer.on_pair(replace(self.sample, capture_id="new-blank", rgb=np.full_like(self.sample.rgb, 255),
                                 metadata_json=json.dumps(meta)))
        with self.assertRaises(BackendFailure):
            await observer(self.args, self.context())

    async def test_resolver_identity_transform_and_covariance_rejected(self):
        for corruption in ("digest", "frame", "calibration", "base", "time", "covariance"):
            def bad(report, snapshot):
                solution = self.solution(report, snapshot)
                tf = solution.camera_to_base
                if corruption == "digest": return replace(solution, observation_digest="other-capture")
                if corruption == "covariance": return replace(solution, covariance_base=tuple(map(float, (-np.eye(6)).ravel())))
                if corruption == "frame": tf = replace(tf, target_frame="marker_frame")
                if corruption == "calibration": tf = replace(tf, calibration_id="old-calibration")
                if corruption == "base": tf = replace(tf, base_epoch="old-base")
                if corruption == "time": tf = replace(tf, captured_at=99.7)  # Oldest exposure is earlier.
                return replace(solution, camera_to_base=tf)
            observer = self.observer(pose_resolver=bad)
            observer.on_pair(self.sample)
            with self.subTest(corruption=corruption), self.assertRaises(BackendFailure):
                await observer(self.args, self.context())

    async def test_wrist_cannot_use_static_base_transform_but_fixed_scene_can(self):
        def static(report, snapshot):
            solution = self.solution(report, snapshot)
            return replace(solution, camera_to_base=replace(solution.camera_to_base, rigid_static=True))
        for role in ("scene", "wrist"):
            observer = self.observer(camera_role=role, pose_resolver=static)
            observer.on_pair(self.sample)
            if role == "wrist":
                with self.assertRaisesRegex(BackendFailure, "capture time"):
                    await observer({**self.args, "camera": role}, self.context())
            else:
                self.assertEqual(len((await observer(self.args, self.context())).metric_poses), 1)

    def test_camera_signature_change_latches_rejection_even_if_old_signature_returns(self):
        observer = self.observer()
        observer.on_pair(self.sample)
        meta = self.sample.metadata
        meta["source_stamps_ns"]["rgb"] += 100000000
        meta["rgb_info"]["k"][0] += 1.
        with self.assertRaisesRegex(PerceptionError, "frame/intrinsics changed"):
            observer.on_pair(replace(self.sample, capture_id="changed-camera", metadata_json=json.dumps(meta)))
        self.assertIsNone(observer.latest_report())
        meta["rgb_info"]["k"][0] -= 1.
        meta["source_stamps_ns"]["rgb"] += 100000000
        with self.assertRaisesRegex(PerceptionError, "frame/intrinsics changed"):
            observer.on_pair(replace(self.sample, capture_id="old-signature", metadata_json=json.dumps(meta)))

    def test_configured_camera_signature_rejects_first_wrong_capture(self):
        observer = self.observer(expected_frame_id=self.metadata["rgb_info"]["frame_id"],
                                 expected_rgb_info_digest="different-intrinsic-record")
        with self.assertRaisesRegex(PerceptionError, "frame/intrinsics changed"):
            observer.on_pair(self.sample)

    async def test_epoch_change_cannot_retag_capture_as_current(self):
        observer = self.observer()
        observer.on_pair(self.sample)
        old = self.context()
        self.world.cancel_epoch()
        with self.assertRaises(BackendFailure):
            await observer(self.args, old)

    async def test_report_is_detached_from_mutable_callers(self):
        observer = self.observer()
        first = observer.on_pair(self.sample)
        first["pose_candidates"] = []
        second = observer.latest_report()
        second["pose_candidates"] = []
        self.assertEqual(len(observer.latest_report()["pose_candidates"]), 2)

    def test_rigid_binding_and_marker_mismatch_rejected(self):
        with self.assertRaises(PerceptionError):
            replace(self.binding, marker_from_entity=np.zeros((4, 4)))
        with self.assertRaises(PerceptionError):
            self.observer(binding=replace(self.binding, marker_spec_digest="wrong-marker"))

    def test_quaternion_handles_pi_rotations_and_composition(self):
        cv = self.detector.cv2
        for vector in ([0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [.6, 1.4, -.9]):
            rotation = cv.Rodrigues(np.asarray(vector, dtype=float))[0]
            x, y, z, w = _quaternion(rotation)
            reconstructed = np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            np.testing.assert_allclose(reconstructed, rotation, atol=1e-12)

    async def test_capture_cancel_drains_before_source_close(self):
        entered, release = threading.Event(), threading.Event()
        closed = []
        class Source:
            def capture(inner, timeout_ms):
                entered.set()
                if not release.wait(2.):
                    raise RuntimeError("test capture timeout")
                return self.sample
            def close(inner):
                closed.append(release.is_set())
        observer = self.observer()
        running = asyncio.create_task(observer.run(Source()))
        await asyncio.to_thread(entered.wait, 1.)
        running.cancel()
        await asyncio.sleep(.01)
        self.assertEqual(closed, [])
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertEqual(closed, [True])
        self.assertIsNone(observer.latest_report())

    async def test_stop_immediately_revokes_cached_observation_and_source_reuse(self):
        observer = self.observer()
        observer.on_pair(self.sample)
        observer.stop()
        self.assertIsNone(observer.latest_report())
        with self.assertRaisesRegex(BackendFailure, "stopping or closed"):
            await observer(self.args, self.context())
        with self.assertRaisesRegex(PerceptionError, "stopped"):
            observer.on_pair(self.sample)
        with self.assertRaisesRegex(PerceptionError, "stopped"):
            await observer.run(object())

    def test_capture_clock_regression_cannot_recover_by_catching_up(self):
        observer = self.observer()
        observer.on_pair(self.sample)
        meta = self.sample.metadata
        meta["source_stamps_ns"]["rgb"] -= 100000000
        with self.assertRaisesRegex(PerceptionError, "clock reset"):
            observer.on_pair(replace(self.sample, capture_id="clock-reset", metadata_json=json.dumps(meta)))
        meta["source_stamps_ns"]["rgb"] += 200000000
        with self.assertRaisesRegex(PerceptionError, "capture clock invalidated"):
            observer.on_pair(replace(self.sample, capture_id="caught-up", metadata_json=json.dumps(meta)))

    async def test_stop_during_inflight_detection_discards_finished_report(self):
        entered, release = threading.Event(), threading.Event()
        detector = self.detector
        class SlowDetector:
            spec = detector.spec
            def observe(inner, sample):
                entered.set()
                release.wait(2.)
                return detector.observe(sample)
        observer = self.observer(detector=SlowDetector())
        work = asyncio.create_task(asyncio.to_thread(observer.on_pair, self.sample))
        await asyncio.to_thread(entered.wait, 1.)
        observer.stop()
        release.set()
        with self.assertRaisesRegex(PerceptionError, "stopped during"):
            await work
        self.assertIsNone(observer.latest_report())

    async def test_repeated_cancellation_during_close_retains_lifecycle_ownership(self):
        entered, release = threading.Event(), threading.Event()
        observer = self.observer()
        class Source:
            def capture(inner, timeout_ms):
                raise PerceptionError("test disconnected")
            def close(inner):
                entered.set()
                release.wait(2.)
        running = asyncio.create_task(observer.run(Source()))
        await asyncio.sleep(.01)
        running.cancel()
        await asyncio.to_thread(entered.wait, 1.)
        running.cancel()
        await asyncio.sleep(.01)
        self.assertFalse(running.done())
        self.assertTrue(observer._running)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertFalse(observer._running)


if __name__ == "__main__":
    unittest.main()
