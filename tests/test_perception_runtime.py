import unittest
import asyncio
import threading
import numpy as np

from rammp_adl.perception import (CameraFrame, Intrinsics, CalibratedTransform,
    CaptureRegistry, PerceptionError, TrackStore, RosCameraBuffer,
    deproject, fit_plane, fit_prismatic, fit_revolute, fuse_points)


def frame(stamp=10.0):
    return CameraFrame("wrist", "image-1", np.zeros((80, 100, 3), np.uint8),
                       np.full((80, 100), .25), Intrinsics(100, 80, 100., 100., 50., 40.),
                       stamp, "wrist_optical", "calibration-1")


class PerceptionGeometryTests(unittest.TestCase):
    def test_deprojection_is_a_point_and_uses_original_intrinsics(self):
        point = deproject(frame(), (60, 40), min_depth_m=.07, max_depth_m=.5)
        np.testing.assert_allclose(point, (.025, 0., .25))
        self.assertEqual(point.shape, (3,))

    def test_invalid_depth_is_not_free_space(self):
        source = frame()
        invalid = CameraFrame("wrist", "invalid", source.rgb, np.zeros((80, 100)), source.intrinsics,
                              10., "wrist_optical", "calibration-1")
        with self.assertRaises(PerceptionError):
            deproject(invalid, (50, 40), min_depth_m=.07, max_depth_m=.5)

    def test_capture_arrays_cannot_be_mutated_after_registration(self):
        with self.assertRaises(ValueError):
            frame().rgb[0, 0, 0] = 255

    def test_freshness_and_clock_uncertainty_are_required(self):
        with self.assertRaises(PerceptionError):
            frame().require_fresh(20., .2)
        with self.assertRaises(PerceptionError):
            frame().require_fresh(9., .2)

    def test_transform_uses_capture_time_and_calibration_identity(self):
        transform = CalibratedTransform("wrist_optical", "base_link", np.eye(3), np.array([.1, .2, .3]),
                                         "calibration-1", "base-1", 10., .1)
        kwargs = dict(capture_time=10.05, source_frame="wrist_optical", calibration_id="calibration-1", base_epoch="base-1")
        np.testing.assert_allclose(transform.apply(np.zeros(3), **kwargs), (.1, .2, .3))
        with self.assertRaises(PerceptionError):
            transform.apply(np.zeros(3), **{**kwargs, "capture_time": 10.5})
        with self.assertRaises(PerceptionError):
            transform.apply(np.zeros(3), **{**kwargs, "base_epoch": "base-2"})

    def test_fusion_respects_uncertainty_and_rejects_invalid_covariance(self):
        point, covariance = fuse_points([np.zeros(3), np.ones(3)], [np.eye(3), 3*np.eye(3)])
        np.testing.assert_allclose(point, (.25, .25, .25))
        np.testing.assert_allclose(covariance, .75*np.eye(3))
        with self.assertRaises(PerceptionError):
            fuse_points([np.zeros(3)], [-np.eye(3)])

    def test_plane_does_not_claim_complete_object_orientation(self):
        points = [[x, y, .3] for x in (0., .1, .2) for y in (0., .1)]
        result = fit_plane(points, max_rms_m=.001)
        self.assertTrue(result["normal_sign_ambiguous"])
        self.assertAlmostEqual(abs(result["normal"][2]), 1.)
        with self.assertRaises(PerceptionError):
            fit_plane([[0, 0, 0], [1, 0, 0], [2, 0, 0]], max_rms_m=.001)

    def test_prismatic_fit_has_measured_stroke_gate(self):
        result = fit_prismatic([[0, 0, 0], [.05, 0, 0], [.1, 0, 0]], max_rms_m=.001, min_stroke_m=.05)
        np.testing.assert_allclose(result["axis"], (1., 0., 0.))
        with self.assertRaises(PerceptionError):
            fit_prismatic([[0, 0, 0], [.001, 0, 0], [.002, 0, 0]], max_rms_m=.001, min_stroke_m=.05)

    def test_revolute_fit_from_observed_arc_and_short_arc_rejected(self):
        angles = np.linspace(0., 1., 12)
        points = np.column_stack((.2*np.cos(angles)+.4, .2*np.sin(angles)+.3, np.full(12, .5)))
        kwargs = dict(max_rms_m=.001, min_span_rad=.3, min_radius_m=.1, max_radius_m=.4)
        result = fit_revolute(points, **kwargs)
        np.testing.assert_allclose(result["origin_m"], (.4, .3, .5), atol=1e-6)
        self.assertAlmostEqual(result["radius_m"], .2)
        with self.assertRaises(PerceptionError):
            fit_revolute(points, **{**kwargs, "min_span_rad": 1.2})

    def test_tracks_keep_stale_obstacles_and_bound_cumulative_corrections(self):
        now = [10.]
        tracks = TrackStore(max_age_s=.2, max_displacement_m=.1, clock=lambda: now[0])
        kwargs = dict(evidence_id="local", calibration_id="cal", base_epoch="base")
        tracks.update("object", (0., 0., 0.), captured_at=10., **kwargs)
        now[0] += .05
        tracks.update("object", (.06, 0., 0.), captured_at=now[0], **kwargs)
        now[0] += .05
        with self.assertRaises(PerceptionError):
            tracks.update("object", (.12, 0., 0.), captured_at=now[0], **kwargs)
        now[0] += 1.
        self.assertIn("object", tracks.snapshot())
        self.assertFalse(tracks.snapshot()["object"]["valid"])


class CropTests(unittest.TestCase):
    def test_crop_backprojection_survives_resize_and_egress_is_bounded(self):
        registry = CaptureRegistry(clock=lambda: 10.)
        registry.register(frame())
        crop = registry.create_crop("image-1", (10, 20, 90, 60), contains_face=False, max_long_edge=40)
        self.assertEqual((crop.width, crop.height), (40, 20))
        np.testing.assert_allclose(crop.original_box((.25, .25, .75, .75)), (30, 30, 70, 50))
        crop.validate_for_egress(max_bytes=200000, max_long_edge=640, allow_face=False)
        self.assertTrue(crop.as_openai_input()["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("depth", crop.cloud_metadata())

    def test_unknown_face_content_cannot_leave_device(self):
        registry = CaptureRegistry(clock=lambda: 10.)
        registry.register(frame())
        crop = registry.create_crop("image-1", (0, 0, 100, 80))
        with self.assertRaises(PerceptionError):
            crop.validate_for_egress(max_bytes=200000, max_long_edge=640, allow_face=False)

    def test_eviction_removes_crops_with_their_capture_provenance(self):
        registry = CaptureRegistry(max_frames=1, clock=lambda: 10.)
        registry.register(frame())
        crop = registry.create_crop("image-1", (0, 0, 100, 80), contains_face=False)
        second = CameraFrame("wrist", "image-2", frame().rgb, frame().depth_m, frame().intrinsics, 10., "wrist_optical", "calibration-1")
        registry.register(second)
        with self.assertRaises(KeyError):
            registry.resolve(crop.image_id)

    def test_ros_ingestion_rejects_unaligned_and_out_of_order_frames(self):
        buffer = RosCameraBuffer(camera_id="scene", frame_id="scene_optical", calibration_id="cal", clock_mapper=lambda t: (t, .001))
        kwargs = dict(rgb=frame().rgb, depth_m=frame().depth_m, intrinsics=frame().intrinsics,
                      rgb_stamp_s=10., depth_stamp_s=10., aligned=True)
        buffer.ingest(**kwargs)
        with self.assertRaises(PerceptionError):
            buffer.ingest(**kwargs)
        with self.assertRaises(PerceptionError):
            buffer.ingest(**{**kwargs, "rgb_stamp_s": 11., "aligned": False})


class PerceptionBoundaryTests(unittest.TestCase):
    def test_nonfinite_profile_bounds_cannot_disable_geometry_gates(self):
        points = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
        for value in (float("nan"), float("inf"), -1., 0.):
            with self.subTest(value=value):
                for operation in (
                    lambda: fit_plane(points, max_rms_m=value),
                    lambda: fit_prismatic(points, max_rms_m=.01, min_stroke_m=value),
                    lambda: fit_revolute(points, max_rms_m=.01, min_span_rad=value, min_radius_m=.1, max_radius_m=1.),
                    lambda: TrackStore(max_age_s=.1, max_displacement_m=value),
                    lambda: CaptureRegistry(max_age_s=value),
                    lambda: deproject(frame(), (50, 40), min_depth_m=.07, max_depth_m=value),
                    lambda: RosCameraBuffer(camera_id="a", frame_id="b", calibration_id="c", clock_mapper=lambda t: (t, 0.), max_skew_s=value),
                ):
                    with self.assertRaises(PerceptionError):
                        operation()

    def test_static_transform_still_requires_finite_time_and_provenance(self):
        args = ("camera", "base", np.eye(3), np.zeros(3), "cal", "base-1")
        with self.assertRaises(PerceptionError):
            CalibratedTransform(*args, float("nan"), .1, rigid_static=True)
        transform = CalibratedTransform(*args, 0., .1, rigid_static=True)
        with self.assertRaises(PerceptionError):
            transform.apply(np.zeros(3), capture_time=float("nan"), source_frame="camera", calibration_id="cal", base_epoch="base-1")

    def test_crops_are_bounded_even_without_new_frames(self):
        registry = CaptureRegistry(max_crops=1, clock=lambda: 10.)
        registry.register(frame())
        first = registry.create_crop("image-1", (0, 0, 20, 20), contains_face=False)
        second = registry.create_crop("image-1", (0, 0, 30, 30), contains_face=False)
        self.assertEqual(len(registry.crops), 1)
        with self.assertRaises(KeyError):
            registry.resolve(first.image_id)
        self.assertEqual(registry.resolve(second.image_id)[0], second)

    def test_bridge_resolves_only_fresh_locally_registered_crops(self):
        from rammp_adl.ros_bridge import RuntimeBridge
        from rammp_adl.contracts import ContractError
        now = [10.]
        registry = CaptureRegistry(clock=lambda: now[0])
        registry.register(frame())
        crop = registry.create_crop("image-1", (0, 0, 20, 20), contains_face=False)
        bridge = RuntimeBridge(None, capture_registry=registry)
        self.assertEqual(bridge.resolve_images([crop.image_id]), (crop,))
        now[0] = 20.
        with self.assertRaises(ContractError):
            bridge.resolve_images([crop.image_id])


class PerceptionCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_waits_for_capture_before_closing_sdk(self):
        from rammp_adl.perception import PerceptionLoop
        entered, release = threading.Event(), threading.Event()
        events = []
        class Source:
            def capture(self, timeout_ms):
                events.append("read")
                entered.set()
                release.wait(.5)
                events.append("read_done")
                return frame()
            def close(self):
                events.append("close")
        loop = PerceptionLoop(Source(), CaptureRegistry(clock=lambda: 10.),
                              lambda f: events.append("published"), lambda e: None)
        task = asyncio.create_task(loop.run())
        await asyncio.to_thread(entered.wait, .5)
        task.cancel()
        await asyncio.sleep(.01)
        self.assertNotIn("close", events)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, .5)
        self.assertEqual(events, ["read", "read_done", "close"])


if __name__ == "__main__":
    unittest.main()
