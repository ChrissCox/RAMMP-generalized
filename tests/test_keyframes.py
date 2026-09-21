"""Keyframe selection, the face screen and egress encoding."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rammp_adl.perception.geometry import PerceptionError
from rammp_adl.motion.sheppy_client import TOOL_FRAME_FROM_FLANGE_M
from rammp_adl.perception.keyframes import (FaceScreen, Keyframe, KeyframeSelector, StampReceiptClock, depth_change,
                                             encode_keyframe)

from synthetic_scene import looking_down, render, synthetic_pair

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/"artifacts/models/face_detection_yunet_2023mar.onnx"
WALL, MONO = 1000., 100.
STAMP = 1_000_000_000_000


class NoFace:
    def screen(self, rgb):
        return {"contains_face": False, "faces": 0, "model_sha256": "fake"}


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.clock = StampReceiptClock(wall=lambda: WALL, monotonic=lambda: MONO)
        self.pose = looking_down([.3, 0., .6])
        self.depth = render(self.pose, [(.27, .33, .10, .14, .08)])

    def pair(self, depth=None, *, capture_id="cap-1", now=MONO+.05):
        """A capture exposed 50 ms before `now` and received 30 ms before it."""
        stamp_ns = int(round((WALL+(now-MONO)-.05)*1e9))
        sample = synthetic_pair(self.depth if depth is None else depth, capture_id=capture_id, stamp_ns=stamp_ns, receipt=now-.03)
        self.clock.note(stamp_ns, now-.03)
        return sample

    def consider(self, selector, sample, *, pose=None, still_since=MONO-5., now=MONO+.05, force=False):
        return selector.consider(sample, base_from_camera=self.pose if pose is None else pose, joints=(0.,)*7,
                                 still_since=still_since, clock_mapper=self.clock, now=now, force=force)

    def test_only_still_fresh_captures_become_keyframes(self):
        selector = KeyframeSelector()
        self.assertIsNone(self.consider(selector, self.pair(), still_since=None))
        self.assertIsNone(self.consider(selector, self.pair(), still_since=MONO+.01))       # moved during exposure
        self.assertIsNone(self.consider(selector, self.pair(), now=MONO+5.))                # too old to place
        keyframe = self.consider(selector, self.pair())
        self.assertEqual(keyframe.reason, "initial")
        self.assertAlmostEqual(keyframe.oldest_capture_s, MONO+.05-.07)
        self.assertEqual(keyframe.width, 320)

    def test_interval_motion_and_change_rules(self):
        selector = KeyframeSelector(min_interval_s=3.)
        first = self.consider(selector, self.pair())
        self.assertIsNotNone(first)
        # Same pose, same scene: nothing, even after the interval; a request still gets one.
        self.assertIsNone(self.consider(selector, self.pair(capture_id="cap-2", now=MONO+.5), now=MONO+.5))
        self.assertIsNone(self.consider(selector, self.pair(capture_id="cap-3", now=MONO+4.), now=MONO+4.))
        self.assertEqual(self.consider(selector, self.pair(capture_id="cap-4", now=MONO+4.), now=MONO+4., force=True).reason, "requested")
        # Camera moved: a keyframe, but only once the interval has elapsed.
        moved = looking_down([.35, 0., .6])
        self.assertIsNone(self.consider(selector, self.pair(capture_id="cap-5", now=MONO+4.5), pose=moved, now=MONO+4.5))
        self.assertEqual(self.consider(selector, self.pair(capture_id="cap-6", now=MONO+8.), pose=moved, now=MONO+8.).reason, "camera_moved")
        # Scene changed at the same pose: a new object appears.
        changed = render(moved, [(.27, .33, .10, .14, .08), (.40, .48, -.05, .05, .10)])
        self.assertEqual(self.consider(selector, self.pair(changed, capture_id="cap-7", now=MONO+12.), pose=moved, now=MONO+12.).reason, "scene_changed")
        self.assertGreater(depth_change(self.depth, changed)[1], 0)
        self.assertEqual(depth_change(self.depth, self.depth), (0., 0))

    def test_projection_round_trip(self):
        selector = KeyframeSelector()
        keyframe = self.consider(selector, self.pair())
        u, v, depth = keyframe.project([.30, .12, .08])
        self.assertAlmostEqual(depth, .52)
        self.assertIsNone(keyframe.project([.30, .12, .7]))          # behind the camera
        self.assertTrue(0 <= u < 320 and 0 <= v < 240)
        self.assertEqual(TOOL_FRAME_FROM_FLANGE_M, .12)


class StampClockTests(unittest.TestCase):
    def test_receipt_latency_bounds_the_mapping(self):
        import math
        clock = StampReceiptClock(max_latency_s=.15, wall=lambda: WALL, monotonic=lambda: MONO)
        clock.note(STAMP, MONO+.02)
        mapped, uncertainty = clock(STAMP/1e9)
        self.assertAlmostEqual(mapped, MONO)
        self.assertAlmostEqual(uncertainty, .02)
        self.assertTrue(math.isnan(clock(STAMP/1e9+5.)[0]))                 # never seen
        clock.note(STAMP+10_000_000, MONO-.5)                                # stamped after receipt: another clock
        self.assertTrue(math.isnan(clock((STAMP+10_000_000)/1e9)[0]))
        clock.note(STAMP+20_000_000, MONO+.4)                                # arrived too late to bound exposure
        self.assertTrue(math.isnan(clock((STAMP+20_000_000)/1e9)[0]))
        clock.note(STAMP+30_000_000, MONO+.03)                               # floor, not zero
        self.assertEqual(clock((STAMP+30_000_000)/1e9)[1], .002)


class EncodingTests(unittest.TestCase):
    def keyframe(self, width=848, height=480):
        rgb = np.full((height, width, 3), 120, np.uint8)
        return Keyframe("cap-9", rgb, np.full((height, width), .5), (600., 0., width/2, 0., 600., height/2, 0., 0., 1.),
                        "d405_color_optical_frame", STAMP, MONO+.02, MONO, .02, looking_down([0., 0., .5]), (0.,)*7, "initial", MONO)

    def test_a_screened_keyframe_becomes_a_bounded_full_frame_crop(self):
        crop = encode_keyframe(self.keyframe(), camera_id="wrist_d405", calibration_id="cal", screen=NoFace())
        self.assertEqual((crop.width, crop.height), (640, 362))
        self.assertLessEqual(len(crop.jpeg_bytes), 200000)
        self.assertEqual(crop.crop_xyxy, (0, 0, 848, 480))
        self.assertIs(crop.contains_face, False)
        self.assertEqual(crop.image_id, "keyframe-cap-9")
        self.assertEqual(crop.original_box([0., 0., .5, .5]), (0., 0., 424., 240.))

    def test_no_screen_or_a_face_withholds_the_frame(self):
        with self.assertRaisesRegex(PerceptionError, "no face screen"):
            encode_keyframe(self.keyframe(), camera_id="wrist_d405", calibration_id="cal", screen=None)
        class Face:
            def screen(self, rgb):
                return {"contains_face": True, "faces": 1, "model_sha256": "fake"}
        with self.assertRaisesRegex(PerceptionError, "withheld"):
            encode_keyframe(self.keyframe(), camera_id="wrist_d405", calibration_id="cal", screen=Face())


@unittest.skipUnless(MODEL.is_file() and Path(str(MODEL)+".sha256").is_file(), "The face model is separate evidence")
class FaceScreenTests(unittest.TestCase):
    def test_pinned_model_loads_and_clears_a_blank_frame(self):
        screen = FaceScreen(MODEL)
        verdict = screen.screen(np.zeros((480, 848, 3), np.uint8))
        self.assertFalse(verdict["contains_face"])
        self.assertEqual(len(verdict["model_sha256"]), 64)

    def test_a_model_that_does_not_match_its_pin_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            copy_path = Path(folder)/MODEL.name
            shutil.copy(MODEL, copy_path)
            Path(str(copy_path)+".sha256").write_text("0"*64+"  x\n")
            with self.assertRaisesRegex(PerceptionError, "sha256"):
                FaceScreen(copy_path)
            with self.assertRaisesRegex(PerceptionError, "missing"):
                FaceScreen(Path(folder)/"absent.onnx")


if __name__ == "__main__":
    unittest.main()
