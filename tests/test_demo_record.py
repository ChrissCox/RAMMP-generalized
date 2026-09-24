"""Teleop demonstrations are written whole, at half resolution, with scaled intrinsics and one step per frame."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo_record", ROOT/"tools/demo_record.py")
demos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demos)


class EpisodeTests(unittest.TestCase):
    def test_an_episode_is_saved_with_frames_depth_steps_and_scaled_intrinsics(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(demos, "DEMOS", Path(folder)):
            self.assertEqual(demos.next_index("door_handle_grasp"), 0)
            episode = demos.Episode("door_handle_grasp", 0, [430., 0., 424., 0., 430., 240., 0., 0., 1.])
            rgb = np.full((480, 848, 3), 90, np.uint8)
            depth = np.full((480, 848), .31)
            depth[0, 0] = np.nan
            for i in range(20):
                episode.add(rgb, depth, {"t": 100.+i/15., "joints_rad": [0.]*7, "knuckle_rad": .01})
            path = episode.save(Path(folder))
            meta = json.loads((path/"meta.json").read_text())
            self.assertEqual((meta["frames"], meta["intrinsics_k"][0], meta["intrinsics_k"][2]), (20, 215., 212.))
            self.assertAlmostEqual(meta["duration_s"], 19/15., places=2)
            stack = np.load(path/"depth.npz")["depth_mm"]
            self.assertEqual(stack.shape, (20, 240, 424))
            self.assertEqual(int(stack[0, 5, 5]), 310)
            self.assertEqual(len(list((path/"frames").glob("*.jpg"))), 20)
            self.assertEqual(len((path/"steps.jsonl").read_text().splitlines()), 20)
            self.assertEqual(demos.next_index("door_handle_grasp"), 1)


if __name__ == "__main__":
    unittest.main()
