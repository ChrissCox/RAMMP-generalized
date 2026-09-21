"""Worker request framing only; no GPU or robot connection is made here."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rammp_adl.contracts import digest
from rammp_adl.motion.curobo_worker import main, parse_request, plan_request, serialize_trajectory
from rammp_adl.motion.rolling import JointState, JointTrajectory, MotionError, TrajectoryPoint


def request():
    world = [{"name": "fixture_table", "position": [0., 0., -.1], "dims": [1., 1., .1]}]
    return {"start": {"position": [0.] * 7, "velocity": [0.] * 7, "acceleration": [0.] * 7},
            "goal": {"position_m": [.3, 0., .4], "quaternion_xyzw": [0., 0., 0., 1.]},
            "world": world, "world_identity": digest(world)}


class WorkerRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_invalid_and_moving_requests_before_gpu_load(self):
        good = request()
        self.assertIsInstance(parse_request(good), JointState)
        bad = []
        value = copy.deepcopy(good); value["start"]["velocity"][0] = .1; bad.append(value)
        value = copy.deepcopy(good); value["start"]["position"][0] = True; bad.append(value)
        value = copy.deepcopy(good); value["world_identity"] = "unrelated-world"; bad.append(value)
        value = copy.deepcopy(good); value["world"] = []; bad.append(value)
        value = copy.deepcopy(good); value["goal"]["quaternion_xyzw"] = [0.] * 4; bad.append(value)
        value = copy.deepcopy(good); value["execute"] = True; bad.append(value)
        value = copy.deepcopy(good); del value["start"]["acceleration"]; bad.append(value)
        for value in bad:
            with self.subTest(value=value), patch("rammp_adl.motion.curobo_worker.RammpCuroboAdapter.load") as load:
                with self.assertRaises(MotionError):
                    await plan_request(value, source_root="unused", planner_config="unused")
                load.assert_not_called()

    def test_serialization_preserves_exact_timed_trajectory_identity(self):
        state = JointState((0.,) * 7, (0.,) * 7, (0.,) * 7)
        path = JointTrajectory(tuple("joint_" + str(i) for i in range(1, 8)),
                               (TrajectoryPoint(0., state), TrajectoryPoint(.1, state)), "explicit-test-double")
        record = serialize_trajectory(path)
        restored = JointTrajectory(tuple(record["joint_names"]), tuple(
            TrajectoryPoint(p["time_s"], JointState(p["position"], p["velocity"], p["acceleration"]))
            for p in json.loads(json.dumps(record))["points"]), record["provenance"])
        self.assertEqual(record["digest"], restored.digest)
        self.assertEqual(record["interpolation"], "quintic-hermite-v1")

    def test_rejection_is_structured_and_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request.json").write_text('{"execute":true}')
            argv = ["--wrapper-source", "unused", "--planner-config", "unused", "--request",
                    str(root / "request.json"), "--output", str(root / "response.json")]
            with patch("rammp_adl.motion.curobo_worker.RammpCuroboAdapter.load") as load:
                self.assertEqual(main(argv), 2)
                load.assert_not_called()
            response = json.loads((root / "response.json").read_text())
            self.assertEqual(response["status"], "rejected")
            self.assertFalse(response["hardware_commands"])
            with self.assertRaises(MotionError):
                main(argv)


if __name__ == "__main__":
    unittest.main()
