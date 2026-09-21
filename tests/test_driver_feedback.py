"""Acquisition/clock/ownership regressions with explicit synthetic packets."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

from rammp_adl.motion.driver_feedback import DriverFeedbackBuffer, FeedbackBounds
from rammp_adl.motion.driver_transport import DriverTransportError, TransferredOwnership
from rammp_adl.motion.driver_state import ARM_JOINT_NAMES


BUILD = "a"*64


def bounds():
    return FeedbackBounds(.0001, .015, .1, .1, .00001, .1, .1)


def packet(at=10., sequence=1, **changes):
    fields = dict(host_boot_id="fixture-boot", driver_session_id="fixture-session", extension_build_id=BUILD,
        simulation=True, joint_names=ARM_JOINT_NAMES, exchange_sequence=sequence,
        exchange_start_monotonic_ns=round((at-.002)*1e9), exchange_end_monotonic_ns=round((at-.001)*1e9),
        position_rad=[0.]*7, velocity_rad_s=[0.]*7, fault=False, heartbeat_required=True,
        watchdog_latched=False, heartbeat_timeout_s=.5, owned=False, owner_id="", ownership_generation=0)
    fields.update(changes)
    return NS(**fields)


class DriverFeedbackTests(unittest.TestCase):
    def make_buffer(self):
        now = [10.]
        buffer = DriverFeedbackBuffer(bounds=bounds(), simulation=True, extension_build_id=BUILD,
                                      boot_id="fixture-boot", clock=lambda: now[0])
        return buffer, now

    def test_republication_never_refreshes_acquisition_or_receipt(self):
        buffer, now = self.make_buffer()
        msg = packet()
        self.assertTrue(buffer.ingest(msg))
        first = buffer.read()
        now[0] += .05
        self.assertFalse(buffer.ingest(msg))
        self.assertEqual(buffer.read(), first)
        now[0] += .06
        with self.assertRaisesRegex(DriverTransportError, "expired"):
            buffer.read()

    def test_ros_fixed_float_array_representation(self):
        import numpy as np
        buffer, _ = self.make_buffer()
        self.assertTrue(buffer.ingest(packet(position_rad=np.zeros(7), velocity_rad_s=np.zeros(7))))
        self.assertEqual(buffer.read().position_rad, (0.,)*7)

    def test_wrong_boot_build_mode_and_future_exchange_are_rejected(self):
        for change in ({"host_boot_id":"other"}, {"extension_build_id":"b"*64},
                       {"simulation":False}, {"exchange_end_monotonic_ns":11_000_000_000},
                       {"joint_names":tuple(reversed(ARM_JOINT_NAMES))}):
            with self.subTest(change=change):
                buffer, _ = self.make_buffer()
                self.assertFalse(buffer.ingest(packet(**change)))
                with self.assertRaises(DriverTransportError):
                    buffer.read()

    def test_changed_same_sequence_and_source_restart_latch_rejection(self):
        for change in ({"position_rad":[.01]*7}, {"driver_session_id":"new-session", "exchange_sequence":2}):
            buffer, _ = self.make_buffer()
            buffer.ingest(packet())
            self.assertFalse(buffer.ingest(packet(**change)))
            self.assertFalse(buffer.ingest(packet(sequence=3)))
            with self.assertRaises(DriverTransportError):
                buffer.read()

    def test_fault_and_missing_watchdog_cannot_authorize_motion(self):
        for change in ({"fault":True}, {"watchdog_latched":True}, {"heartbeat_required":False}):
            buffer, _ = self.make_buffer()
            self.assertFalse(buffer.ingest(packet(**change)))
            with self.assertRaises(DriverTransportError):
                buffer.read()

    def test_feedback_must_independently_confirm_owner_and_timeout(self):
        buffer, _ = self.make_buffer()
        owner = TransferredOwnership("fixture-owner", b"x"*16, 2)
        buffer.ingest(packet(owned=True, owner_id=owner.owner_id, ownership_generation=owner.generation))
        self.assertEqual(buffer.read(ownership=owner, watchdog_timeout_s=.5).sequence, 1)
        with self.assertRaisesRegex(DriverTransportError, "already owned"):
            buffer.require_unowned()
        with self.assertRaisesRegex(DriverTransportError, "watchdog"):
            buffer.read(ownership=owner, watchdog_timeout_s=.25)
        with self.assertRaisesRegex(DriverTransportError, "owner"):
            buffer.read(ownership=replace(owner, generation=3))

    def test_stationary_dwell_requires_independent_exchanges_and_acceleration_bound(self):
        buffer, now = self.make_buffer()
        buffer.ingest(packet())
        with self.assertRaisesRegex(DriverTransportError, "insufficient"):
            buffer.stationary_start(duration_s=.1, velocity_rad_s=.001, position_span_rad=.001)
        for i in range(1, 12):
            now[0] = 10.+i*.01
            buffer.ingest(packet(at=now[0], sequence=i+1))
        state = buffer.stationary_start(duration_s=.1, velocity_rad_s=.001, position_span_rad=.001)
        self.assertEqual(state.sequence, 12)
        buffer.bounds = replace(buffer.bounds, stationary_acceleration_rad_s2=.000001)
        with self.assertRaisesRegex(DriverTransportError, "acceleration uncertainty"):
            buffer.stationary_start(duration_s=.1, velocity_rad_s=.001, position_span_rad=.001)

    def test_motion_during_dwell_and_acquisition_gaps_are_rejected(self):
        buffer, now = self.make_buffer()
        for i in range(12):
            now[0] = 10.+i*.01
            buffer.ingest(packet(at=now[0], sequence=i+1, velocity_rad_s=[.1 if i == 8 else 0.]*7))
        with self.assertRaisesRegex(DriverTransportError, "velocity"):
            buffer.stationary_start(duration_s=.1, velocity_rad_s=.001, position_span_rad=.001)


if __name__ == "__main__":
    unittest.main()
