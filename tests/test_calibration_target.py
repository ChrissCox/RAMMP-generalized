import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from rammp_adl.perception.calibration_target import CalibrationTarget, generate_printables, inspect_capture
from rammp_adl.perception.geometry import PerceptionError


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is an optional camera dependency")
class CalibrationTargetTests(unittest.TestCase):
    def setUp(self):
        self.target = CalibrationTarget()
        self.cv2 = self.target.cv2

    def test_perspective_projection_corner_ids_and_locations(self):
        # Independently project known board-grid coordinates into a synthetic
        # camera image. This does not claim measured camera calibration.
        image = self.target.image()
        h, w = image.shape
        source = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dest = np.float32([[80, 60], [950, 110], [1010, 760], [130, 800]])
        transform = self.cv2.getPerspectiveTransform(source, dest)
        warped = self.cv2.warpPerspective(image, transform, (1100, 900), borderValue=255)
        report = self.target.detect(np.repeat(warped[..., None], 3, axis=2))
        self.assertEqual(report["corner_count"], 24)
        self.assertEqual(set(report["corner_ids"]), set(range(24)))
        grid = np.float32([[(i%6+1)*280, (i//6+1)*280] for i in report["corner_ids"]])
        expected = self.cv2.perspectiveTransform(grid.reshape(-1, 1, 2), transform).reshape(-1, 2)
        errors = np.linalg.norm(expected-np.asarray(report["corners_xy_px"]), axis=1)
        self.assertLess(float(errors.max()), 1.)
        self.assertFalse(report["extrinsics_calibrated"])
        self.assertFalse(report["physical_target_dimensions_verified"])

    def test_blank_image_and_wrong_marker_dictionary_do_not_make_target(self):
        self.assertEqual(self.target.detect(np.full((480, 640, 3), 255, np.uint8))["status"], "target_not_observed")
        other = self.cv2.aruco.CharucoBoard((7, 5), .03, .0225,
            self.cv2.aruco.getPredefinedDictionary(self.cv2.aruco.DICT_4X4_50))
        picture = np.pad(other.generateImage((1400, 1000)), 50, constant_values=255)
        result = self.target.detect(np.repeat(picture[..., None], 3, axis=2))
        self.assertEqual(result["corner_count"], 0)

    def test_partial_visibility_is_reported_as_partial(self):
        picture = np.pad(self.target.image(), 60, constant_values=255)
        picture[:, :picture.shape[1]//2] = 255
        report = self.target.detect(np.repeat(picture[..., None], 3, axis=2))
        self.assertGreater(report["corner_count"], 0)
        self.assertLess(report["corner_count"], report["total_target_corners"])
        self.assertFalse(report["metric_geometry_validated"])

    def test_invalid_spec_and_non_image_input_rejected(self):
        for key, value in (("squares_x", True), ("square_length_m", float("nan")),
                           ("marker_length_m", .1), ("dictionary", "arbitrary"), ("legacy_pattern", 0)):
            with self.subTest(key=key), self.assertRaises(PerceptionError):
                CalibrationTarget({**self.target.spec, key: value})
        for rgb in (np.zeros((2, 2, 3)), np.zeros((2, 2), np.uint8), np.empty((0, 0, 3), np.uint8)):
            with self.assertRaises(PerceptionError):
                self.target.detect(rgb)

    @unittest.skipUnless(shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler PDF checks require local command-line tools")
    def test_printed_pdf_scale_and_rasterized_pattern(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"target"
            report = generate_printables(root)
            for name, info in report["outputs"].items():
                pdf = root/info["file"]
                metadata = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True, timeout=10).stdout
                page = next(line for line in metadata.splitlines() if line.startswith("Page size:"))
                values = page.split()
                np.testing.assert_allclose([float(values[2]), float(values[4])], np.asarray(info["page_mm"])*72/25.4, atol=.01)
                out = root/(name+"-rendered")
                subprocess.run(["pdftoppm", "-singlefile", "-r", "300", "-png", str(pdf), str(out)], capture_output=True, check=True, timeout=10)
                gray = self.cv2.imread(str(out)+".png", self.cv2.IMREAD_GRAYSCALE)
                detected = self.target.detect(np.repeat(gray[..., None], 3, axis=2))
                self.assertEqual(detected["corner_count"], 24)
                points = dict(zip(detected["corner_ids"], detected["corners_xy_px"]))
                # Corner 0 to corner 5 spans five squares: 150 mm at 300 dpi.
                # Poppler rounds the embedded-image raster bounds. Allow 1.5
                # output pixels (0.127 mm), not a percentage scale adjustment.
                measured_px = np.linalg.norm(np.asarray(points[5])-points[0])
                self.assertAlmostEqual(measured_px, 150*300/25.4, delta=1.5)
            with self.assertRaises(FileExistsError):
                generate_printables(root)

    def test_saved_capture_preserves_source_identity_and_never_refreshes_time(self):
        from test_ros_rgbd import pair
        sample = pair()
        picture = np.pad(self.target.image(), 60, constant_values=255)
        rgb = np.repeat(picture[..., None], 3, axis=2)
        metadata = sample.metadata
        metadata["capture_id"] = sample.capture_id
        metadata["rgb_info"].update(width=rgb.shape[1], height=rgb.shape[0])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"capture.npz"
            np.savez(path, rgb=rgb, depth_m=np.zeros(rgb.shape[:2]), metadata_json=json.dumps(metadata))
            report = inspect_capture(path)
            self.assertEqual(report["corner_count"], 24)
            self.assertEqual(report["capture_id"], sample.capture_id)
            self.assertEqual(report["source_stamps_ns"]["rgb"], 10**10)
            self.assertTrue(report["historical_capture"])
            self.assertFalse(report["metric_geometry_validated"])
            self.assertNotIn("captured_at", report)

    def test_unexpected_or_object_arrays_in_capture_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"capture.npz"
            np.savez(path, arbitrary=np.zeros(1))
            with self.assertRaises(PerceptionError):
                inspect_capture(path)
            np.savez(path, rgb=np.array([object()], dtype=object), depth_m=np.zeros(1), metadata_json=np.array([object()], dtype=object))
            with self.assertRaises(ValueError):
                inspect_capture(path)


if __name__ == "__main__":
    unittest.main()
