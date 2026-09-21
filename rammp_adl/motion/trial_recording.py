"""Local evidence from an admitted commissioning trial; never safety profiles.

Recording publication/command times does not establish device acquisition error,
continuous tracking accuracy or a worst-case physical stopping envelope.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import threading
import time

from ..contracts import checked_copy
from .driver_transport import VerifiedMotionState
from .rolling import MotionError


class TrialRecording:
    """Bounded in-memory collection, with no disk writes on the monitor path."""
    EVENTS = frozenset({'planning_started','candidate_prepared','control_claim_attempted',
                        'control_claimed','dispatch_attempted','cancel_requested',
                        'transport_receipt','ownership_released','fault','software_stop'})

    def __init__(self, *, simulation, max_samples=100000, clock=time.monotonic):
        if type(simulation) is not bool or type(max_samples) is not int or not 2 <= max_samples <= 100000:
            raise MotionError('Trial recorder requires explicit mode and bounded sample count')
        self.simulation, self.max_samples, self.clock = simulation, max_samples, clock
        self._lock = threading.RLock()
        self._states, self._events = [], []

    def sample(self, state):
        if not isinstance(state, VerifiedMotionState):
            raise MotionError('Trial recording requires acquisition-aware admitted state')
        with self._lock:
            if self._states:
                previous = self._states[-1]
                if state == previous:
                    return state
                if (state.source_id != previous.source_id or state.sequence <= previous.sequence
                        or state.acquired_at_monotonic_s < previous.acquired_at_monotonic_s
                        or state.received_at_monotonic_s < previous.received_at_monotonic_s):
                    raise MotionError('Trial feedback identity reset, regressed or changed without a new acquisition')
            if len(self._states) >= self.max_samples:
                raise MotionError('Trial evidence capacity exhausted; stop the bounded trial')
            self._states.append(state)
            return state

    def event(self, name, data=None):
        if name not in self.EVENTS:
            raise MotionError('Unknown local commissioning event')
        record = {'event': name, 'monotonic_s': self.clock(),
                  'data': checked_copy({} if data is None else data, max_bytes=8192)}
        if (type(record['monotonic_s']) not in (int, float)
                or not math.isfinite(record['monotonic_s']) or record['monotonic_s'] < 0):
            raise MotionError('Trial event clock must be finite and monotonic')
        with self._lock:
            if self._events and record['monotonic_s'] < self._events[-1]['monotonic_s']:
                raise MotionError('Trial event clock regressed')
            if len(self._events) >= 1000:
                raise MotionError('Trial event capacity exhausted')
            self._events.append(record)

    def write(self, output):
        """Write after monitoring ends. Raw q/dq never leave the device."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        with self._lock:
            states, events = tuple(self._states), tuple(self._events)
        with (output/'states.jsonl').open('x') as stream:
            for sample in states:
                stream.write(json.dumps(asdict(sample),allow_nan=False)+'\n')
        with (output/'events.jsonl').open('x') as stream:
            for event in events: stream.write(json.dumps(event,allow_nan=False)+'\n')
        report = {
            'scope': 'admitted_simulation_trial' if self.simulation else 'admitted_physical_commissioning_trial',
            'sample_count': len(states), 'event_count': len(events),
            'hardware_limits_established': False, 'commissioned_limits': None,
            'limits': [
                'Acquisition time is the conservative bound supplied by the configured feedback adapter',
                'Command/receipt event times are local process times, not controller activation timestamps',
                'Samples cannot prove an unsampled peak, worst-case stopping envelope or independent sensor accuracy',
                'No profile is generated or enabled by this recorder',
            ],
        }
        if states:
            report['source_id'] = states[0].source_id
            report['first_sequence'], report['last_sequence'] = states[0].sequence, states[-1].sequence
            report['max_abs_observed_velocity_rad_s'] = [max(abs(s.velocity_rad_s[i]) for s in states) for i in range(7)]
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
