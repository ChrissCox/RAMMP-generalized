"""The offline bench replays recorded scenes through the runtime's own discovery and geometry, and scores them."""
import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rammp_adl.perception.scene_record import list_scenes, load_scene, save_guard_trip, save_scene
from synthetic_bench import HANDLE_CENTRE, SyntheticAstra, keyframe, record
from synthetic_scene import looking_at
from test_grounded_scene import NoFace

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2"
spec = importlib.util.spec_from_file_location("bench_offline", ROOT/"tools/bench_offline.py")
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


class RecordTests(unittest.TestCase):
    def test_a_scene_round_trips_with_depth_in_millimetres(self):
        with tempfile.TemporaryDirectory() as folder:
            original = keyframe(0, looking_at([.1, .1, .35], [.6, .1, .3]))
            path = save_scene(folder, original, source="test", task_text="open the door")
            loaded, meta = load_scene(path)
            self.assertEqual(meta["source"], "test")
            self.assertEqual(meta["layout"], "default")
            np.testing.assert_allclose(loaded.base_from_camera, original.base_from_camera)
            valid = np.isfinite(original.depth_m)
            np.testing.assert_allclose(loaded.depth_m[valid], original.depth_m[valid], atol=.0006)
            self.assertEqual(list_scenes(folder), [path])
            self.assertEqual(loaded.rgb.shape, original.rgb.shape)

    def test_a_guard_trip_is_recorded_without_raising(self):
        from test_sheppy_backend import trajectory
        with tempfile.TemporaryDirectory() as folder:
            path = save_guard_trip(folder, depth_frame=(np.full((4, 4), .5), (1., 0., 2., 0., 1., 2., 0., 0., 1.), 0.),
                                   joints_rad=(0.,)*7, trajectory=trajectory((0.,)*7, (.1,)*7), elapsed_s=.4,
                                   exclusions=[((.5, 0., .3), .1)], tool_exclusion_m=0., trip={"kind": "collision"}, context={})
            meta = json.loads((path/"meta.json").read_text())
            self.assertEqual(meta["trip"]["kind"], "collision")
            self.assertTrue((path/"depth.npz").is_file())
            self.assertIsNone(save_guard_trip("/proc/no-such-place", depth_frame=None, joints_rad=(), trajectory=None, elapsed_s=0.,
                                              exclusions=(), tool_exclusion_m=0., trip={}, context={}))


class RuntimeRecordingTests(unittest.TestCase):
    def test_the_scene_hands_every_selected_keyframe_to_its_recorder_and_nothing_else(self):
        from test_grounded_scene import GroundedSceneTests
        case = GroundedSceneTests("test_discovery_names_and_places_the_object")
        case.setUp()
        saved = []
        case.scene.recorder = saved.append
        self.assertIsNotNone(case.feed())
        self.assertEqual([k.capture_id for k in saved], ["cap-1"])
        case.feed(capture_id="cap-2")                                                 # the same still view: not selected again
        self.assertEqual(len(saved), 1)


class CacheTests(unittest.TestCase):
    def test_the_same_request_is_answered_once_and_a_miss_without_network_is_refused(self):
        class Inner:
            calls = 0

            async def create(self, **request):
                Inner.calls += 1
                return {"status": "completed", "usage": {"input_tokens": 10, "output_tokens": 2}, "output": []}

            async def close(self):
                pass
        with tempfile.TemporaryDirectory() as folder:
            transport = offline.CachingTransport(folder)
            transport.inner = Inner()
            for _ in range(2):
                asyncio.run(transport.create(model="m", input=[{"x": 1}]))
            self.assertEqual((Inner.calls, transport.stats["cached"], transport.stats["input_tokens"]), (1, 1, 20))
            offline_only = offline.CachingTransport(folder, network=False)
            asyncio.run(offline_only.create(model="m", input=[{"x": 1}]))
            with self.assertRaises(offline.CacheMiss):
                asyncio.run(offline_only.create(model="m", input=[{"x": 2}]))


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class ScoreTests(unittest.TestCase):
    def replay(self, astra):
        return asyncio.run(offline.replay_all(list_scenes(self.bench), astra, bench=self.bench, screen=NoFace()))

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.bench = Path(folder.name)
        self.poses = record(self.bench)

    def test_labels_come_from_the_consensus_and_a_perfect_run_scores_full_marks(self):
        results, keyframes = self.replay(SyntheticAstra(self.poses))
        labels = offline.make_labels(results, keyframes)
        visible = [s for s, l in labels.items() if l["handle_visible"]]
        self.assertEqual(len(visible), 5)
        self.assertEqual(sum(1 for l in labels.values() if l["handle_visible"] is False), 1)
        placed = np.asarray(next(iter(labels.values()))["position_m"])
        self.assertLess(np.linalg.norm(placed[1:]-HANDLE_CENTRE[1:]), .02)            # across the door face, where the bar is
        scored = offline.score(results, labels)
        self.assertEqual(scored["parts"]["handle_recall"], 1.)
        self.assertEqual(scored["parts"]["grasp_ok"], 1.)
        self.assertEqual(scored["parts"]["visibility_agreement"], 1.)
        self.assertEqual(scored["parts"]["no_false_handles"], 1.)
        self.assertGreater(scored["score"], 95.)

    def test_a_missed_handle_and_an_invented_one_cost_what_they_should(self):
        results, keyframes = self.replay(SyntheticAstra(self.poses))
        labels = offline.make_labels(results, keyframes)
        worse, _ = self.replay(SyntheticAstra(self.poses, miss={"keyframe-synthetic-01"}, invent={"keyframe-synthetic-05"}))
        scored = offline.score(worse, labels)
        self.assertEqual(scored["parts"]["handle_recall"], .8)
        self.assertEqual(scored["parts"]["no_false_handles"], 0.)
        self.assertEqual(scored["status"][next(r["scene"] for r in worse if r["scene"].endswith("synthetic-01"))], "missed")
        self.assertLess(scored["score"], offline.score(results, labels)["score"]-15.)

    def test_the_report_renders_every_scene_and_loads_nothing_but_fonts(self):
        results, keyframes = self.replay(SyntheticAstra(self.poses))
        labels = offline.make_labels(results, keyframes)
        path = self.bench/"report.html"
        offline.write_report(path, results, labels, offline.score(results, labels), keyframes, screen=NoFace())
        page = path.read_text()
        self.assertEqual(page.count('<article class="scene">'), 6)
        self.assertNotIn("<script src", page)
        self.assertEqual(offline.digest(self.bench), offline.digest(self.bench))


if __name__ == "__main__":
    unittest.main()
