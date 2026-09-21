import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rammp_adl.contracts import digest
from rammp_adl.perception.fiducial import SingleMarkerObserver, raw_camera_model, relative_pose_candidates, fixed_marker_reference
from rammp_adl.perception.geometry import PerceptionError
from rammp_adl.perception.ros_rgbd import capture_ros_rgbd
from test_ros_rgbd import pair


class FiducialTests(unittest.TestCase):
    def setUp(self):
        self.observer = SingleMarkerObserver()
        self.cv = self.observer.cv2
        self.meta = pair().metadata
        self.meta['rgb_info'].update(width=640, height=480, k=[600., 0., 320., 0., 600., 240., 0., 0., 1.])
        self.meta['rgb_info']['d'] = [-.05, .02, .001, -.001, 0.]

    def image(self, marker_id=0):
        gray = self.cv.aruco.generateImageMarker(self.cv.aruco.getPredefinedDictionary(self.cv.aruco.DICT_4X4_50), marker_id, 150)
        rgb = np.full((480, 640, 3), 255, np.uint8)
        rgb[165:315, 245:395] = gray[..., None]
        return rgb

    def test_requested_marker_detection_retains_provenance_and_both_branches(self):
        result = self.observer.inspect(self.image(), self.meta, capture_id='test-marker')
        self.assertEqual(result['observed_marker_ids'], [0])
        self.assertEqual(result['status'], 'provisional_pose_candidates')
        self.assertEqual(result['capture_id'], 'test-marker')
        self.assertEqual(result['source_stamps_ns']['rgb'], 10**10)
        self.assertEqual(len(result['pose_candidates']), 2)
        self.assertFalse(result['planar_ambiguity_resolved'])
        self.assertFalse(result['physical_marker_size_verified'])
        self.assertFalse(result['metric_geometry_validated'])
        self.assertFalse(result['capture_clock_validated'])
        self.assertFalse(result['robot_reference_calibrated'])

    def test_wrong_id_and_duplicate_same_id_are_rejected(self):
        wrong = self.observer.inspect(self.image(1), self.meta, capture_id='wrong')
        self.assertEqual(wrong['status'], 'marker_not_observed')
        rgb = self.image()
        rgb[165:315, 50:200] = rgb[165:315, 245:395]
        duplicate = self.observer.inspect(rgb, self.meta, capture_id='duplicate')
        self.assertEqual(duplicate['status'], 'duplicate_marker_id_ambiguous')
        self.assertEqual(duplicate['pose_candidates'], [])

    def test_fixed_marker_frame_preserves_branches_scale_and_historical_evidence(self):
        observation = self.observer.inspect(self.image(), self.meta, capture_id='fixed-scene-capture')
        original = copy.deepcopy(observation)
        reference = fixed_marker_reference(observation, marker_fixed_confirmed=True, camera_fixed_confirmed=True)
        self.assertEqual(len(reference['candidates']), 2)
        for i, candidate in enumerate(reference['candidates']):
            np.testing.assert_allclose(np.asarray(candidate['marker_from_camera']) @
                                       np.asarray(observation['pose_candidates'][i]['camera_from_marker']), np.eye(4), atol=1e-12)
        self.assertEqual(reference['source_stamps_ns'], observation['source_stamps_ns'])
        self.assertEqual(reference['source_observation_digest'], digest(observation))
        self.assertEqual(reference['marker_spec'], observation['marker_spec'])
        self.assertIsNone(reference['base_from_marker'])
        self.assertIsNone(reference['position_uncertainty_m'])
        self.assertFalse(reference['robot_reference_calibrated'])
        self.assertFalse(reference['planar_ambiguity_resolved'])
        self.assertFalse(reference['hardware_motion_enabled'])
        self.assertEqual(observation, original)

    def test_fixed_reference_requires_fixed_scene_camera_and_valid_geometry(self):
        obs = self.observer.inspect(self.image(), self.meta, capture_id='fixed-scene')
        for marker, camera in ((False, True), (True, False)):
            with self.assertRaises(PerceptionError):
                fixed_marker_reference(obs, marker_fixed_confirmed=marker, camera_fixed_confirmed=camera)
        for bad in ('digest', 'rotation', 'empty', 'nonfinite'):
            changed = copy.deepcopy(obs)
            if bad == 'digest': changed['marker_spec']['nominal_black_edge_m'] *= 2
            elif bad == 'rotation': changed['pose_candidates'][0]['camera_from_marker'][0][0] = 2.
            elif bad == 'empty': changed['pose_candidates'] = []
            else: changed['pose_candidates'][0]['reprojection_rms_px'] = float('nan')
            with self.subTest(bad=bad), self.assertRaises(PerceptionError):
                fixed_marker_reference(changed, marker_fixed_confirmed=True, camera_fixed_confirmed=True)

    def test_ippe_recovers_synthetic_distorted_projection_in_correct_frame(self):
        rvec, translation = np.array([2.7, .15, -.1]), np.array([.03, -.02, .6])
        k, d = raw_camera_model(self.meta['rgb_info'])
        corners = self.cv.projectPoints(self.observer.object_points(), rvec, translation, k, d)[0].reshape(4, 2)
        result = self.observer.pose_candidates(corners, self.meta['rgb_info'])
        self.assertEqual(len(result), 2)
        best = np.asarray(result[0]['camera_from_marker'])
        np.testing.assert_allclose(best[:3, 3], translation, atol=1e-7)
        np.testing.assert_allclose(best[:3, :3], self.cv.Rodrigues(rvec)[0], atol=1e-6)
        self.assertLess(result[0]['reprojection_rms_px'], 1e-6)

    def test_nominal_size_controls_scale_but_never_becomes_verified(self):
        corners = np.array([[270., 190.], [370., 190.], [370., 290.], [270., 290.]])
        first = self.observer.pose_candidates(corners, self.meta['rgb_info'])
        larger = SingleMarkerObserver({**self.observer.spec, 'nominal_black_edge_m': .1})
        second = larger.pose_candidates(corners, self.meta['rgb_info'])
        np.testing.assert_allclose(np.asarray(second[0]['camera_from_marker'])[:3, 3], 2*np.asarray(first[0]['camera_from_marker'])[:3, 3], atol=1e-8)

    def test_bad_intrinsics_keep_detection_but_block_pose(self):
        meta = copy.deepcopy(self.meta)
        meta['rgb_info']['k'][0] = 0.
        result = self.observer.inspect(self.image(), meta, capture_id='uncalibrated')
        self.assertEqual(result['status'], 'marker_observed_pose_unavailable')
        self.assertEqual(result['pose_candidates'], [])
        for change in (lambda x: x.update(distortion_model='equidistant'), lambda x: x.update(binning_x=2),
                       lambda x: x['roi'].update(width=100), lambda x: x['d'].__setitem__(0, float('nan'))):
            info = copy.deepcopy(self.meta['rgb_info'])
            change(info)
            with self.assertRaises(PerceptionError):
                raw_camera_model(info)

    def test_degenerate_corners_rejected(self):
        for corners in (np.zeros((4, 2)), np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), np.full((4, 2), np.nan)):
            with self.assertRaises(PerceptionError):
                self.observer.pose_candidates(corners, self.meta['rgb_info'])

    def test_relative_transform_direction_and_all_planar_combinations(self):
        observations, truth = [], []
        k, d = raw_camera_model(self.meta['rgb_info'])
        for i, (rvec, position) in enumerate(((np.array([2.7, .15, -.1]), np.array([.03, -.02, .6])),
                                              (np.array([2.45, -.25, .1]), np.array([.25, -.1, .75])))):
            projected = self.cv.projectPoints(self.observer.object_points(), rvec, position, k, d)[0].reshape(4, 2)
            obs = self.observer.inspect(self.image(), self.meta, capture_id=f'camera-{i}')
            obs['camera_frame'] = f'optical-{i}'
            obs['pose_candidates'] = self.observer.pose_candidates(projected, self.meta['rgb_info'])
            observations.append(obs)
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = self.cv.Rodrigues(rvec)[0], position
            truth.append(transform)
        result = relative_pose_candidates(*observations, stationary_setup_confirmed=True)
        self.assertEqual(len(result['candidates']), 4)
        np.testing.assert_allclose(result['candidates'][0]['second_camera_from_first_camera'], truth[1] @ np.linalg.inv(truth[0]), atol=1e-6)
        self.assertEqual(result['source_frame'], 'optical-0')
        self.assertEqual(result['target_frame'], 'optical-1')
        self.assertFalse(result['metric_geometry_validated'])
        self.assertFalse(result['capture_clock_validated'])
        self.assertEqual(result['source_stamps_ns'][0], observations[0]['source_stamps_ns'])
        with self.assertRaises(PerceptionError):
            relative_pose_candidates(*observations, stationary_setup_confirmed=False)
        with self.assertRaises(PerceptionError):
            relative_pose_candidates(observations[0], observations[0], stationary_setup_confirmed=True)
        for mutation in (lambda x: x['marker_spec'].update(nominal_black_edge_m=.1),
                         lambda x: x['pose_candidates'][0].update(reprojection_rms_px=float('nan')),
                         lambda x: x.update(pose_candidates=x['pose_candidates']*2)):
            altered = copy.deepcopy(observations[0])
            mutation(altered)
            with self.assertRaises(PerceptionError):
                relative_pose_candidates(altered, observations[1], stationary_setup_confirmed=True)

    def test_capture_observer_is_reused_and_failure_does_not_claim_observation(self):
        pairs = iter([pair(10.), pair(11.)])
        class Source:
            failure = None
            def __init__(self, **kwargs): pass
            def capture(self, timeout_ms): return next(pairs)
            def close(self): pass
            def statistics(self): return {'paired': 2}
        calls = []
        def observe(p):
            calls.append(p.capture_id)
            if len(calls) == 1:
                raise PerceptionError('no usable camera model')
            return {'status': 'marker_not_observed', 'capture_id': p.capture_id}
        with tempfile.TemporaryDirectory() as temp, patch('rammp_adl.perception.ros_rgbd.RosRgbdSource', Source):
            destination = Path(temp)/'capture'
            report = capture_ros_rgbd(output_dir=destination, samples=2, pair_observer=observe)
            self.assertEqual(report['status'], 'captured')
            self.assertEqual(report['observation_summary'], {'observation_rejected': 1, 'marker_not_observed': 1})
            self.assertEqual(json.loads((destination/'observation.json').read_text())['capture_id'], report['capture_id'])


if __name__ == '__main__':
    unittest.main()
