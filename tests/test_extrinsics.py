"""Synthetic calibration datasets; no sensor or physical calibration claims."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from rammp_adl.contracts import digest
from rammp_adl.perception.extrinsics import fit_hand_eye, fixed_camera_candidate, main
from rammp_adl.perception.geometry import PerceptionError


def transform(rotvec, position):
    value = np.eye(4)
    value[:3,:3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3,3] = position
    return value


def dataset(*, noise=0., parallel=False):
    rng = np.random.default_rng(741)
    x = transform([.2,-.1,.3], [.04,-.03,.09])
    y = transform([-.3,.2,.1], [.6,.1,.2])
    samples = []
    for i in range(12):
        axis = [0.,0.,.1*i] if parallel else rng.uniform(-.7,.7,3)
        b = transform(axis, rng.uniform(-.2,.2,3))
        c = np.linalg.inv(x) @ np.linalg.inv(b) @ y
        c[:3,3] += rng.normal(0.,noise,3)
        samples.append({'sample_id':f'sample-{i}','robot_evidence_id':f'robot-{i}',
            'camera_evidence_id':f'camera-{i}','branch_resolution_evidence_id':f'synthetic-branch-{i}',
            'robot_captured_at_s':10.+i,'camera_captured_at_s':10.001+i,'timestamp_uncertainty_s':.001,
            'base_from_wrist':b.tolist(),'camera_from_marker':c.tolist()})
    return {'schema_version':1,'frames':{'base':'base_link','wrist':'wrist','camera':'wrist_optical','marker':'marker'},
            'identities':{'robot_model_digest':'synthetic-model','intrinsics_digest':'synthetic-intrinsics',
                          'marker_spec_digest':'synthetic-marker','base_epoch':'base-fixture',
                          'clock_evidence_id':'synthetic-clock','marker_placement_id':'fixed-fixture'},
            'criteria':{'max_pair_skew_s':.01,'max_timestamp_uncertainty_s':.005,
                        'min_rotation_span_rad':.05,'min_axis_ratio':.1,'max_translation_condition':100.,
                        'max_translation_residual_m':.001,'max_rotation_residual_rad':.005},
            'samples':samples}, x, y


class ExtrinsicsTests(unittest.TestCase):
    def test_recovers_transform_directions_and_preserves_unapproved_status(self):
        data,x,y = dataset()
        report = fit_hand_eye(data)
        np.testing.assert_allclose(report['wrist_from_camera'],x,atol=1e-10)
        np.testing.assert_allclose(report['base_from_marker'],y,atol=1e-10)
        self.assertEqual(report['source_digest'],digest(data))
        self.assertFalse(report['robot_reference_calibrated'])
        self.assertFalse(report['hardware_motion_enabled'])
        self.assertIsNone(report['position_uncertainty_m'])

    def test_small_noise_is_reported_as_fit_error_not_physical_uncertainty(self):
        data,x,_ = dataset(noise=1e-5)
        report = fit_hand_eye(data)
        np.testing.assert_allclose(report['wrist_from_camera'],x,atol=2e-5)
        self.assertGreater(max(report['diagnostics']['translation_residuals_m']),1e-6)
        self.assertIsNone(report['orientation_uncertainty_rad'])

    def test_one_pose_repeats_and_parallel_motion_are_unobservable(self):
        data,_,_ = dataset()
        data['samples'] = data['samples'][:1]
        with self.assertRaisesRegex(PerceptionError,'diverse'):
            fit_hand_eye(data)
        data,_,_ = dataset(parallel=True)
        with self.assertRaisesRegex(PerceptionError,'parallel'):
            fit_hand_eye(data)
        data,_,_ = dataset()
        for sample in data['samples'][1:]:
            sample['base_from_wrist'] = data['samples'][0]['base_from_wrist']
            sample['camera_from_marker'] = data['samples'][0]['camera_from_marker']
        with self.assertRaisesRegex(PerceptionError,'rotational span'):
            fit_hand_eye(data)

    def test_bad_branch_or_changed_marker_is_not_silently_fitted(self):
        data,_,_ = dataset()
        data['samples'][5]['camera_from_marker'][0][3] += .1
        with self.assertRaisesRegex(PerceptionError,'residual'):
            fit_hand_eye(data)

    def test_capture_pairing_identity_and_numeric_inputs_are_checked(self):
        mutations = [lambda d:d['samples'][1].update(camera_evidence_id='camera-0'),
                     lambda d:d['samples'][1].update(camera_captured_at_s=50.),
                     lambda d:d['samples'][1].update(timestamp_uncertainty_s=.02),
                     lambda d:d['samples'][1].update(branch_resolution_evidence_id=''),
                     lambda d:d['samples'][1]['base_from_wrist'][0].__setitem__(0,True),
                     lambda d:d['samples'][1]['base_from_wrist'][3].__setitem__(3,True),
                     lambda d:d.update(frames=[]),
                     lambda d:d['samples'][1]['base_from_wrist'][3].__setitem__(0,.1)]
        for mutation in mutations:
            data,_,_ = dataset()
            mutation(data)
            with self.subTest(mutation=mutation), self.assertRaises(PerceptionError):
                fit_hand_eye(data)

    def test_fixed_camera_requires_exact_marker_and_explicit_branch(self):
        data,_,y = dataset()
        spec = {'dictionary':'DICT_4X4_50','marker_id':0,'nominal_black_edge_m':.05,'size_source':'synthetic'}
        data['identities']['marker_spec_digest'] = digest(spec)
        hand_eye = fit_hand_eye(data)
        c = transform([.1,.2,-.1],[.1,.2,.8])
        observation = {'status':'provisional_pose_candidates','marker_spec':spec,'marker_spec_digest':digest(spec),
            'camera_frame':'scene_optical','source_stamps_ns':{'rgb':100},
            'rgb_info_digest':'synthetic-scene-intrinsics',
            'pose_candidates':[{'camera_from_marker':c.tolist(),'reprojection_rms_px':.2}]}
        kw = {'branch':0,'branch_resolution_evidence_id':'synthetic-resolution','marker_placement_id':'fixed-fixture',
              'fixed_mount':True,'mount_evidence_id':'synthetic-fixed-mount'}
        candidate = fixed_camera_candidate(hand_eye,observation,**kw)
        np.testing.assert_allclose(candidate['base_from_camera'],y @ np.linalg.inv(c),atol=1e-10)
        self.assertFalse(candidate['robot_reference_calibrated'])
        self.assertEqual(candidate['base_epoch'],'base-fixture')
        self.assertEqual(candidate['marker_placement_id'],'fixed-fixture')
        self.assertEqual(candidate['intrinsics_digest'],'synthetic-scene-intrinsics')
        for changed in ({'branch':True},{'marker_placement_id':'moved'},{'branch_resolution_evidence_id':''},
                        {'fixed_mount':False},{'fixed_mount':1},{'mount_evidence_id':''}):
            with self.subTest(changed=changed), self.assertRaises(PerceptionError):
                fixed_camera_candidate(hand_eye,observation,**{**kw,**changed})
        observation['camera_frame'] = hand_eye['frames']['camera']
        with self.assertRaisesRegex(PerceptionError,'wrist camera'):
            fixed_camera_candidate(hand_eye,observation,**kw)

    def test_cli_writes_candidate_without_overwriting_or_admitting_it(self):
        data,_,_ = dataset()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source,target = folder/'input.json',folder/'candidate.json'
            source.write_text(json.dumps(data))
            self.assertEqual(main(['--input',str(source),'--output',str(target)]),0)
            self.assertFalse(json.loads(target.read_text())['hardware_motion_enabled'])
            self.assertEqual(main(['--input',str(source),'--output',str(target)]),2)


if __name__ == '__main__':
    unittest.main()
