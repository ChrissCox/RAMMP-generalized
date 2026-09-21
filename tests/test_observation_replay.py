"""Recorded synthetic marker pixels through the real runtime/provider adapter."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rammp_adl.contracts import Catalog, ContractError
from rammp_adl.handlers import BackendFailure, ExecutionContext
from rammp_adl.perception.fiducial import SingleMarkerObserver
from rammp_adl.perception.geometry import PerceptionError
from rammp_adl.perception.observation_replay import RecordingReasoner, build_replay, local_plan, run_replay
from rammp_adl.world import WorldModel
from test_reasoning_runtime import response
from test_ros_rgbd import pair


class AutoPlanTransport:
    """Explicit fake Responses transport; validates actual emitted request shape."""
    def __init__(self, *, refuse=False):
        self.requests, self.closed, self.refuse = [], False, refuse

    async def create(self, **request):
        self.requests.append(request)
        if self.refuse:
            return {"status": "completed", "output": [{"type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "refusal", "refusal": "test refusal"}]}]}
        context = json.loads(request["input"][0]["content"][0]["text"])["world_context"]
        entity_id = context["goal"]["args"]["entity_id"]
        plan = {"schema_version": "1.0.0", "skill_library_hash": Catalog().hash,
            "task_id": context["task_id"], "snapshot_id": context["snapshot_id"],
            "execution_epoch": context["execution_epoch"], "nodes": [{"id": "provider-observe", "skill": "observe",
                "args": {"entity_id": entity_id, "camera": "scene", "purpose": "state"}}], "edges": []}
        return response({"result": {"status": "OK", "plan": plan}})

    async def close(self):
        self.closed = True


class ObservationReplayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.capture = self.root / "capture.npz"
        detector = SingleMarkerObserver()
        meta = pair().metadata
        meta["capture_id"] = "SYNTHETIC-recorded-marker"
        meta["rgb_info"].update(width=640, height=480,
            k=[600., 0., 320., 0., 600., 240., 0., 0., 1.], d=[0.] * 5)
        self.metadata = meta
        image = np.full((480, 640, 3), 255, np.uint8)
        marker = detector.cv2.aruco.generateImageMarker(detector.cv2.aruco.getPredefinedDictionary(
            detector.cv2.aruco.DICT_4X4_50), 0, 150)
        image[165:315, 245:395] = marker[..., None]
        self.image = image
        self.save_capture(image)

    def tearDown(self):
        self.temp.cleanup()

    def save_capture(self, image):
        np.savez(self.capture, rgb=image, depth_m=np.full((480, 640), np.nan),
                 metadata_json=json.dumps(self.metadata))

    async def test_recorded_pixels_complete_observation_goal_with_zero_metric_poses(self):
        report = await run_replay(self.capture, output_dir=self.root / "local")
        self.assertEqual(report["result"]["status"], "succeeded", report["result"])
        self.assertTrue(report["result"]["nodes"][0]["commit_receipt"])
        self.assertEqual(report["metric_pose_count"], 0)
        self.assertEqual(report["frames_uploaded"], 0)
        self.assertTrue(report["source"]["playback_clock_frozen"])
        self.assertEqual(report["source"]["observation"]["source_stamps_ns"], self.metadata["source_stamps_ns"])
        self.assertFalse(report["source"]["physical_robot_hold_measured"])
        self.assertEqual(report["provider_request_count"], 0)
        self.assertFalse(report["provider_successful_proposal"])
        self.assertTrue((self.root / "local" / "trace.jsonl").is_file())
        self.assertEqual(report["final_world"]["available_skills"], ["observe"])

    async def test_astra_gateway_uses_context_only_pinned_model_and_local_validation(self):
        transport = AutoPlanTransport()
        report = await run_replay(self.capture, output_dir=self.root / "astra",
                                  reasoner="astra", transport=transport)
        self.assertEqual(report["result"]["status"], "succeeded", report["result"])
        self.assertEqual(report["result"]["nodes"][0]["node_id"], "provider-observe")
        self.assertEqual(report["provider_request_count"], 1)
        self.assertFalse(report["provider_successful_proposal"])
        self.assertTrue(report["provider_transport_is_test_double"])
        self.assertTrue(transport.closed)
        request = transport.requests[0]
        self.assertEqual(request["model"], "gpt-6-astra")
        self.assertFalse(request["store"])
        self.assertNotIn("tools", request)
        content = request["input"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["input_text"])
        data = json.loads(content[0]["text"])
        self.assertEqual(data["image_metadata"], [])
        self.assertEqual(data["world_context"]["goal"]["predicate"], "observation_valid")
        self.assertEqual(data["world_context"]["entities"][0]["pose_roles"], [])
        self.assertNotIn("pose_candidates", content[0]["text"])

    async def test_provider_refusal_stays_held_with_no_observation_commit(self):
        transport = AutoPlanTransport(refuse=True)
        report = await run_replay(self.capture, output_dir=self.root / "refused", reasoner="astra", transport=transport)
        self.assertEqual(report["result"]["status"], "incomplete")
        self.assertIn("REFUSED", report["result"]["reason"])
        self.assertEqual(report["result"]["nodes"], [])
        self.assertEqual(report["metric_pose_count"], 0)
        self.assertTrue(transport.closed)

    async def test_supplied_plan_is_validated_without_identity_retagging(self):
        runtime, _, provenance = build_replay(self.capture)
        plan = local_plan(runtime.catalog, runtime.world.snapshot(), entity_id=provenance["entity_id"], camera_role="scene")
        plan["snapshot_id"] = "a-different-scene"
        report = await run_replay(self.capture, output_dir=self.root / "old-plan", plan=plan)
        self.assertNotEqual(report["result"]["status"], "succeeded")
        self.assertEqual(report["result"]["nodes"], [])

    async def test_motion_skill_in_supplied_plan_never_reaches_a_handler(self):
        runtime, _, provenance = build_replay(self.capture)
        plan = local_plan(runtime.catalog, runtime.world.snapshot(), entity_id=provenance["entity_id"], camera_role="scene")
        plan["nodes"][0].update(skill="move_to_pose", args={"entity_id": provenance["entity_id"],
            "pose_role": "viewpoint", "profile_id": "invented"})
        report = await run_replay(self.capture, output_dir=self.root / "motion-plan", plan=plan)
        self.assertNotEqual(report["result"]["status"], "succeeded")
        self.assertEqual(report["result"]["nodes"], [])
        self.assertFalse(report["hardware_commands"])

    def test_missing_marker_cannot_manufacture_initial_entity_evidence(self):
        self.save_capture(np.full_like(self.image, 255))
        with self.assertRaisesRegex(PerceptionError, "no unique"):
            build_replay(self.capture)

    async def test_output_overwrite_and_conflicting_reasoners_rejected(self):
        destination = self.root / "already-exists"
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            await run_replay(self.capture, output_dir=destination)
        with self.assertRaises(ContractError):
            await run_replay(self.capture, output_dir=self.root / "invalid", reasoner="astra", plan={})

    async def test_replay_observer_cannot_accept_an_execution_context_from_live_calibration(self):
        runtime, observer, provenance = build_replay(self.capture)
        self.assertEqual(runtime.registry.mode, "simulation")
        context = runtime.world.snapshot().context
        context["calibration_id"] = "different-live-calibration"
        context["base_epoch"] = "different-live-base"
        other = WorldModel(context, runtime.catalog, trust_initial=False)
        snapshot = other.snapshot()
        execution = ExecutionContext(snapshot.context["task_id"], "other-world",
            snapshot=snapshot, execution_epoch=snapshot.execution_epoch)
        with self.assertRaisesRegex(BackendFailure, "dependencies changed"):
            await observer({"entity_id": provenance["entity_id"], "camera": "scene", "purpose": "state"}, execution)
        self.assertEqual(dict(other.snapshot().metric_poses), {})

    async def test_recording_reasoner_rejects_image_arguments_before_delegate(self):
        class Delegate:
            async def generate_plan(inner, *args, **kwargs):
                raise AssertionError("Image argument reached provider delegate")
        with self.assertRaisesRegex(ContractError, "never uploads"):
            await RecordingReasoner(Delegate()).generate_plan({}, images=[object()])


if __name__ == "__main__":
    unittest.main()
