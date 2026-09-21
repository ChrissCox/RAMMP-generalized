"""Offline wrist-camera calibration candidates from paired measured poses.

No ROS, robot command, TF publication or calibration admission occurs here.
This solves B_i X C_i = Y: base_from_wrist * wrist_from_camera *
camera_from_marker = base_from_fixed_marker. Transform directions follow the
OpenCV hand-eye convention; the bounded SVD implementation is local project code.
Fit residuals are consistency diagnostics, not physical uncertainty estimates.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ..contracts import checked_copy, digest, strict_loads
from .geometry import PerceptionError, positive_bounds, require_ids


MAX_SAMPLES = 128
SOURCE_CONVENTION = 'https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html'


def _rigid(value, name):
    try:
        # Check scalar types before NumPy can promote a mixed bool/float row.
        objects = np.asarray(value, dtype=object)
        if any(isinstance(v, (bool, np.bool_)) for v in objects.flat):
            raise ValueError('boolean transform element')
        if np.asarray(value).dtype.kind not in 'iuf':
            raise ValueError('non-numeric transform')
        matrix = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise PerceptionError(name + ' must be a rigid 4x4 matrix') from exc
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-9, rtol=0)
            or not np.allclose(matrix[:3,:3].T @ matrix[:3,:3], np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3,:3]), 1., atol=1e-7, rtol=0)):
        raise PerceptionError(name + ' must be a finite proper rigid transform')
    return matrix


def _inverse(matrix):
    inverse = np.eye(4)
    inverse[:3,:3] = matrix[:3,:3].T
    inverse[:3,3] = -inverse[:3,:3] @ matrix[:3,3]
    return inverse


def _mean_transform(transforms):
    result = np.eye(4)
    u, _, vt = np.linalg.svd(sum(t[:3,:3] for t in transforms))
    result[:3,:3] = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    result[:3,3] = np.mean([t[:3,3] for t in transforms], axis=0)
    return result


def fit_hand_eye(document):
    """Return an unapproved candidate or reject an unobservable/inconsistent set.

    Numeric criteria are explicit local fit criteria, never motion limits.
    Inputs must already resolve square-marker branches using external evidence.
    No lowest-reprojection branch is selected automatically.
    timestamp_uncertainty_s is the combined bound for both capture clocks,
    including their mapping error; it is added once to the capture-time gap.
    """
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise PerceptionError('Offline calibration requires the calibration extra (scipy)') from exc
    data = checked_copy(document, max_bytes=2097152)
    if (not isinstance(data, dict) or set(data) != {'schema_version','frames','identities','criteria','samples'}
            or type(data['schema_version']) is not int or data['schema_version'] != 1):
        raise PerceptionError('Expected the version 1 offline hand-eye sample contract')
    frames, identities, criteria, samples = (data[k] for k in ('frames','identities','criteria','samples'))
    if (not isinstance(frames, dict) or not isinstance(identities, dict)
            or set(frames) != {'base','wrist','camera','marker'}
            or any(not isinstance(v, str) for v in frames.values()) or len(set(frames.values())) != 4
            or set(identities) != {'robot_model_digest','intrinsics_digest','marker_spec_digest',
                                  'base_epoch','clock_evidence_id','marker_placement_id'}):
        raise PerceptionError('Distinct named frames and complete fixed session identities are required')
    require_ids(*frames.values(), *identities.values())
    keys = {'max_pair_skew_s','max_timestamp_uncertainty_s','min_rotation_span_rad',
            'min_axis_ratio','max_translation_condition','max_translation_residual_m',
            'max_rotation_residual_rad'}
    if not isinstance(criteria, dict) or set(criteria) != keys:
        raise PerceptionError('Every local fit criterion must be explicit')
    positive_bounds(*criteria.values())
    if (criteria['min_axis_ratio'] >= 1. or criteria['max_translation_condition'] <= 1.
            or criteria['min_rotation_span_rad'] >= np.pi):
        raise PerceptionError('Fit observability criteria are invalid')
    if not isinstance(samples, list) or not 5 <= len(samples) <= MAX_SAMPLES:
        raise PerceptionError('Hand-eye calibration needs 5 to 128 diverse paired poses; one stationary view is insufficient')
    robot, camera, ids, robot_ids, camera_ids = [], [], set(), set(), set()
    previous_robot = previous_camera = -1.
    sample_keys = {'sample_id','robot_captured_at_s','camera_captured_at_s','timestamp_uncertainty_s',
                   'robot_evidence_id','camera_evidence_id','branch_resolution_evidence_id',
                   'base_from_wrist','camera_from_marker'}
    maximum_pair_skew = 0.
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != sample_keys:
            raise PerceptionError('A sample lacks exact paired pose, capture or branch evidence')
        require_ids(*(sample[k] for k in ('sample_id','robot_evidence_id','camera_evidence_id',
                                         'branch_resolution_evidence_id')))
        if (sample['sample_id'] in ids or sample['robot_evidence_id'] in robot_ids
                or sample['camera_evidence_id'] in camera_ids):
            raise PerceptionError('Repeated captures cannot increase calibration observability')
        ids.add(sample['sample_id'])
        robot_ids.add(sample['robot_evidence_id'])
        camera_ids.add(sample['camera_evidence_id'])
        rtime, ctime, uncertainty = (sample[k] for k in
            ('robot_captured_at_s','camera_captured_at_s','timestamp_uncertainty_s'))
        if (any(type(v) not in (int,float) or not np.isfinite(v) or v < 0
                for v in (rtime,ctime,uncertainty)) or rtime <= previous_robot or ctime <= previous_camera):
            raise PerceptionError('Capture times must be finite, ordered and in one documented clock domain')
        skew = abs(rtime-ctime) + uncertainty
        if skew > criteria['max_pair_skew_s'] or uncertainty > criteria['max_timestamp_uncertainty_s']:
            raise PerceptionError('Paired pose clock uncertainty or skew exceeds the fit criteria')
        maximum_pair_skew = max(maximum_pair_skew, skew)
        previous_robot, previous_camera = rtime, ctime
        robot.append(_rigid(sample['base_from_wrist'], 'base_from_wrist'))
        camera.append(_rigid(sample['camera_from_marker'], 'camera_from_marker'))

    pairs, alpha, beta = [], [], []
    for i, j in itertools.combinations(range(len(samples)), 2):
        a, b = _inverse(robot[j]) @ robot[i], camera[j] @ _inverse(camera[i])
        av, bv = Rotation.from_matrix(a[:3,:3]).as_rotvec(), Rotation.from_matrix(b[:3,:3]).as_rotvec()
        # Near-pi logarithms have an ambiguous axis sign; discard these pairs.
        if (min(np.linalg.norm(av), np.linalg.norm(bv)) < criteria['min_rotation_span_rad']
                or max(np.linalg.norm(av), np.linalg.norm(bv)) >= np.pi-1e-4):
            continue
        pairs.append((a,b)); alpha.append(av); beta.append(bv)
    if len(pairs) < 2:
        raise PerceptionError('Robot motion has insufficient observable rotational span')
    alpha, beta = np.asarray(alpha).T, np.asarray(beta).T
    singular_a, singular_b = np.linalg.svd(alpha, compute_uv=False), np.linalg.svd(beta, compute_uv=False)
    axis_ratio = min(singular_a[1]/singular_a[0], singular_b[1]/singular_b[0])
    if axis_ratio < criteria['min_axis_ratio']:
        raise PerceptionError('Parallel or nearly parallel rotation axes leave calibration unobservable')
    u, _, vt = np.linalg.svd(alpha @ beta.T)
    x = np.eye(4)
    x[:3,:3] = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    lhs = np.vstack([a[:3,:3]-np.eye(3) for a,b in pairs])
    rhs = np.concatenate([x[:3,:3] @ b[:3,3]-a[:3,3] for a,b in pairs])
    translation, _, rank, singular = np.linalg.lstsq(lhs, rhs, rcond=None)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf')
    if rank != 3 or condition > criteria['max_translation_condition']:
        raise PerceptionError('Robot motion does not constrain camera translation sufficiently')
    x[:3,3] = translation
    closures = [b @ x @ c for b,c in zip(robot,camera)]
    y = _mean_transform(closures)
    translation_errors = [float(np.linalg.norm(t[:3,3]-y[:3,3])) for t in closures]
    rotation_errors = [float(Rotation.from_matrix(y[:3,:3].T @ t[:3,:3]).magnitude()) for t in closures]
    if (max(translation_errors) > criteria['max_translation_residual_m']
            or max(rotation_errors) > criteria['max_rotation_residual_rad']):
        raise PerceptionError('Fixed-marker closure residual exceeds the fit criteria; inspect branches, pairing, scale and kinematic model')
    return {'schema_version':1,'status':'unapproved_hand_eye_candidate','method':'rotation_log_svd_then_linear_translation',
            'transform_convention_source':SOURCE_CONVENTION,'frames':frames,'identities':identities,
            'source_digest':digest(data),'sample_count':len(samples),'relative_motion_pairs':len(pairs),
            'criteria':criteria,'wrist_from_camera':x.tolist(),'base_from_marker':y.tolist(),
            'diagnostics':{'axis_ratio':float(axis_ratio),'translation_condition':condition,
                           'maximum_pair_skew_s':maximum_pair_skew,
                           'translation_residuals_m':translation_errors,'rotation_residuals_rad':rotation_errors},
            'position_uncertainty_m':None,'orientation_uncertainty_rad':None,
            'robot_reference_calibrated':False,'hardware_motion_enabled':False,'frames_uploaded':0,
            'limitations':['Input provenance and physical scale require independent verification',
                           'Fit residuals do not measure kinematic, marker, mount or timing uncertainty',
                           'No TF or admitted CalibratedTransform is created; no robot motion is requested']}


def fixed_camera_candidate(hand_eye, observation, *, branch, branch_resolution_evidence_id,
                           marker_placement_id, fixed_mount, mount_evidence_id):
    """Compose the fixed scene camera only with an explicitly resolved branch."""
    from .fiducial import _validated_pose_candidates
    intrinsics_digest = observation.get('rgb_info_digest')
    require_ids(branch_resolution_evidence_id, marker_placement_id, mount_evidence_id, intrinsics_digest)
    if fixed_mount is not True:
        raise PerceptionError('Fixed scene-camera mounting must be explicitly established')
    if hand_eye.get('status') != 'unapproved_hand_eye_candidate':
        raise PerceptionError('An exact local hand-eye candidate is required')
    if observation.get('camera_frame') == hand_eye['frames']['camera']:
        raise PerceptionError('A wrist camera cannot be admitted as a fixed scene camera')
    transforms = _validated_pose_candidates(observation)
    if type(branch) is not int or not 0 <= branch < len(transforms):
        raise PerceptionError('An explicitly resolved marker pose branch is required')
    if (observation['marker_spec_digest'] != hand_eye['identities']['marker_spec_digest']
            or marker_placement_id != hand_eye['identities']['marker_placement_id']):
        raise PerceptionError('Camera references must share the fixed marker definition')
    base_from_camera = _rigid(hand_eye['base_from_marker'], 'base_from_marker') @ _inverse(transforms[branch])
    return {'status':'unapproved_fixed_camera_candidate','base_from_camera':base_from_camera.tolist(),
            'source_frame':observation['camera_frame'],'target_frame':hand_eye['frames']['base'],
            'hand_eye_digest':digest(hand_eye),'observation_digest':digest(observation),
            'base_epoch':hand_eye['identities']['base_epoch'],
            'marker_placement_id':marker_placement_id,'marker_spec_digest':observation['marker_spec_digest'],
            'intrinsics_digest':intrinsics_digest,'fixed_mount':True,'mount_evidence_id':mount_evidence_id,
            'branch':branch,'branch_resolution_evidence_id':branch_resolution_evidence_id,
            'source_stamps_ns':observation['source_stamps_ns'],'position_uncertainty_m':None,
            'orientation_uncertainty_rad':None,'robot_reference_calibrated':False,
            'hardware_motion_enabled':False,'frames_uploaded':0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Local paired-pose JSON; no live robot access')
    parser.add_argument('--output', required=True, help='New local candidate JSON')
    args = parser.parse_args(argv)
    try:
        candidate = fit_hand_eye(strict_loads(Path(args.input).read_bytes(), max_bytes=2097152))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(candidate, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'status':candidate['status'],'report':str(output),'hardware_motion_enabled':False}))
        return 0
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({'status':'calibration_rejected','reason':str(exc),'hardware_motion_enabled':False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
