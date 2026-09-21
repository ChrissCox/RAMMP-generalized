"""Bounded local commissioning evidence; no robot or ROS transport."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from rammp_adl.motion.driver_transport import VerifiedMotionState
from rammp_adl.motion.rolling import MotionError
from rammp_adl.motion.trial_recording import TrialRecording


def state(sequence=1, acquired=10., received=10.01):
    return VerifiedMotionState((0.,)*7, (.002,)*7, acquired, received, sequence, 'test-acquisition')


class TrialRecordingTests(unittest.TestCase):
    def test_preserves_acquisition_identity_and_does_not_make_profile(self):
        recorder = TrialRecording(simulation=True, clock=lambda: 10.)
        first = state()
        self.assertIs(recorder.sample(first), first)
        recorder.sample(first)  # Monitor polling does not manufacture acquisitions.
        recorder.sample(state(2, 10.02, 10.03))
        payload = {'trajectory_digest': 'local-test'}
        recorder.event('dispatch_attempted', payload)
        payload['trajectory_digest'] = 'changed-after-recording'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'evidence'
            report = recorder.write(path)
            self.assertEqual(report['sample_count'], 2)
            self.assertEqual(report['scope'], 'admitted_simulation_trial')
            self.assertFalse(report['hardware_limits_established'])
            self.assertIsNone(report['commissioned_limits'])
            samples = [json.loads(line) for line in (path/'states.jsonl').read_text().splitlines()]
            self.assertEqual(samples[0]['acquired_at_monotonic_s'], 10.)
            self.assertEqual(samples[1]['sequence'], 2)
            event = json.loads((path/'events.jsonl').read_text())
            self.assertEqual(event['data']['trajectory_digest'], 'local-test')
            with self.assertRaises(FileExistsError):
                recorder.write(path)

    def test_rejects_state_reset_mutation_and_clock_regression(self):
        first = state()
        invalid = [replace(first, source_id='new-source'), replace(first, position_rad=(.1,)*7),
                   state(0), state(2, 9.99), state(2, 10., 10.005)]
        for sample in invalid:
            with self.subTest(sample=sample):
                recorder = TrialRecording(simulation=False)
                recorder.sample(first)
                with self.assertRaises(MotionError):
                    recorder.sample(sample)

    def test_capacity_exhaustion_is_explicit_and_preserves_existing_samples(self):
        recorder = TrialRecording(simulation=False, max_samples=2)
        recorder.sample(state())
        recorder.sample(state(2, 10.02, 10.03))
        with self.assertRaisesRegex(MotionError, 'capacity'):
            recorder.sample(state(3, 10.04, 10.05))
        with tempfile.TemporaryDirectory() as temporary:
            report = recorder.write(Path(temporary)/'record')
            self.assertEqual(report['sample_count'], 2)
            self.assertFalse(report['hardware_limits_established'])

    def test_invalid_or_unbounded_event_and_mode_are_rejected(self):
        with self.assertRaises(MotionError):
            TrialRecording(simulation='physical')
        recorder = TrialRecording(simulation=True)
        with self.assertRaises(MotionError):
            recorder.sample({'position': [0]*7})
        with self.assertRaises(MotionError):
            recorder.event('enable_hardware')
        with self.assertRaises(ValueError):
            recorder.event('fault', {'reason': 'x'*9000})
        for clock in (lambda: float('nan'), lambda: -1., lambda: True):
            with self.subTest(clock=clock), self.assertRaises(MotionError):
                TrialRecording(simulation=True, clock=clock).event('fault')
        times = iter([2., 1.])
        recorder = TrialRecording(simulation=True, clock=lambda: next(times))
        recorder.event('planning_started')
        with self.assertRaises(MotionError):
            recorder.event('candidate_prepared')


if __name__ == '__main__':
    unittest.main()
