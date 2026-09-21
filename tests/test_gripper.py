"""Explicit synthetic gripper samples/commands; never connects to hardware."""
import asyncio
from dataclasses import replace
import time
from types import SimpleNamespace as NS
import unittest

from rammp_adl.motion.gripper import (GripperCalibration, GripperBounds, GripperFeedbackBuffer,
    GripperTransport, calibrated_command)
from rammp_adl.motion.driver_transport import DriverTransportError, TransferredOwnership


BUILD = 'a'*64
OWNER = TransferredOwnership('simulation-gripper-owner', b'x'*16, 3)


def calibration():
    return GripperCalibration('simulation-2f85', 'synthetic-gap-table', 'simulation',
                              (0., .5, 1.), (.08, .04, 0.), .000001)


def bounds():
    return GripperBounds(.000001, .001, .03, .03, .0001, .00001, .002,
                         .001, .4, .5, .5, .002, .008, .1, .06, .25)


def packet(now=10., sequence=1, **change):
    values = dict(host_boot_id='simulation-boot', extension_build_id=BUILD, simulation=True,
        driver_session_id='simulation-session', gripper_present=True, fault=False,
        heartbeat_required=True, watchdog_latched=False, owned=True,
        gripper_halt_supported=True, gripper_halt_active=False, heartbeat_timeout_s=.25,
        exchange_sequence=sequence, exchange_start_monotonic_ns=int((now-.0001)*1e9),
        exchange_end_monotonic_ns=int((now-.00001)*1e9), ownership_generation=OWNER.generation,
        owner_id=OWNER.owner_id, gripper_position_normalized=.5, gripper_current_a=.01,
        gripper_halt_generation=0, gripper_halt_exchange_sequence=0, gripper_halt_position_normalized=0.)
    values.update(change)
    return NS(**values)


class GripperCalibrationTests(unittest.TestCase):
    def test_nonlinear_measured_table_roundtrip(self):
        cal = replace(calibration(), normalized_position=[0., .4, 1.], aperture_m=[.08, .035, 0.])
        for q in (0., .1, .4, .7, 1.):
            self.assertAlmostEqual(cal.to_normalized(cal.to_aperture(q)), q)
        self.assertIsInstance(cal.normalized_position, tuple)
        self.assertAlmostEqual(cal.to_aperture(.4), .035)

    def test_invalid_table_and_unmeasured_extrapolation_rejected(self):
        for change in ({'normalized_position':(0., .5, .4)}, {'aperture_m':(.08, .09, 0.)},
                       {'error_m':0.}, {'mode':'assumed'}, {'evidence_id':''}):
            with self.subTest(change=change), self.assertRaises(DriverTransportError):
                replace(calibration(), **change)
        for gap in (-.001, .081, float('nan'), True):
            with self.assertRaises(DriverTransportError):
                calibration().to_normalized(gap)

    def test_all_wire_fields_explicit_and_canonical(self):
        command = calibrated_command(calibration(), bounds(), aperture_m=.04,
            speed_fraction=.125, current_ceiling_fraction=.25)
        self.assertEqual((command.position_normalized, command.speed_fraction, command.current_ceiling_fraction),
                         (.5, .125, .25))
        self.assertEqual(command.calibration_digest, calibration().digest)
        with self.assertRaises(DriverTransportError):
            calibrated_command(calibration(), bounds(), aperture_m=.04, speed_fraction=1., current_ceiling_fraction=.25)
        with self.assertRaises(DriverTransportError):
            calibrated_command(calibration(), bounds(), aperture_m=.04, speed_fraction=1e-60, current_ceiling_fraction=.25)
        with self.assertRaises(DriverTransportError):
            calibrated_command(replace(calibration(), error_m=.001), bounds(), aperture_m=.04,
                               speed_fraction=.125, current_ceiling_fraction=.25)


class GripperFeedbackTests(unittest.TestCase):
    def make(self):
        now = [10.]
        buffer = GripperFeedbackBuffer(calibration=calibration(), bounds=bounds(), extension_build_id=BUILD,
            simulation=True, boot_id='simulation-boot', clock=lambda:now[0])
        return buffer, now

    def test_physical_mode_cannot_reuse_simulation_calibration(self):
        with self.assertRaises(DriverTransportError):
            GripperFeedbackBuffer(calibration=calibration(), bounds=bounds(), extension_build_id=BUILD, simulation=False)

    def test_republication_never_refreshes_age(self):
        buffer, now = self.make()
        self.assertTrue(buffer.ingest(packet()))
        original = buffer.read(ownership=OWNER)
        now[0] += .02
        self.assertFalse(buffer.ingest(packet()))
        self.assertEqual(buffer.read(), original)
        now[0] += .02
        with self.assertRaisesRegex(DriverTransportError, 'expired'):
            buffer.read()

    def test_mutated_duplicate_reset_wrong_mode_and_fault_latch(self):
        for changes in ({'gripper_position_normalized':.6}, {'driver_session_id':'restart'},
                        {'simulation':False}, {'fault':True}, {'gripper_current_a':float('nan')},
                        {'gripper_present':False}, {'gripper_halt_supported':False}):
            with self.subTest(changes=changes):
                buffer, _ = self.make()
                buffer.ingest(packet())
                self.assertFalse(buffer.ingest(packet(**changes)))
                self.assertFalse(buffer.ingest(packet(sequence=2)))
                with self.assertRaises(DriverTransportError): buffer.read()

    def test_publication_only_state_cannot_be_admitted(self):
        buffer, _ = self.make()
        self.assertFalse(buffer.ingest(NS(position=.5, current=.01, present=True, header=NS(stamp=10.))))
        with self.assertRaises(DriverTransportError): buffer.read()

    def test_halt_ack_is_readable_but_never_authorizes_new_command(self):
        buffer, _ = self.make()
        self.assertTrue(buffer.ingest(packet(watchdog_latched=True, owned=False, owner_id='',
            gripper_halt_active=True, gripper_halt_generation=1,
            gripper_halt_exchange_sequence=1, gripper_halt_position_normalized=.5)))
        self.assertTrue(buffer.read().halt_active)
        with self.assertRaisesRegex(DriverTransportError, 'ownership'):
            buffer.read(ownership=OWNER)

    def test_duplicate_acquisition_still_updates_ownership_loss(self):
        buffer, _ = self.make()
        buffer.ingest(packet())
        before = buffer.read()
        self.assertFalse(buffer.ingest(packet(owned=False, owner_id='', watchdog_latched=True)))
        self.assertEqual(buffer.read().received_at_s, before.received_at_s)
        with self.assertRaises(DriverTransportError):
            buffer.read(ownership=OWNER)

    def test_halt_receipt_cannot_precede_exchange_or_reset_in_session(self):
        buffer, now = self.make()
        self.assertFalse(buffer.ingest(packet(gripper_halt_active=True, gripper_halt_generation=1,
            gripper_halt_exchange_sequence=2, gripper_halt_position_normalized=.5)))
        buffer, now = self.make()
        buffer.ingest(packet(gripper_halt_active=True, gripper_halt_generation=1,
            gripper_halt_exchange_sequence=1, gripper_halt_position_normalized=.5))
        now[0] += .002
        self.assertFalse(buffer.ingest(packet(now=now[0], sequence=2)))


class SimulatedGripper:
    """Synthetic actuator state, deliberately no physical-dynamics claim."""
    hardware_commands = False
    def __init__(self):
        self.buffer = GripperFeedbackBuffer(calibration=calibration(), bounds=bounds(), extension_build_id=BUILD,
            simulation=True, boot_id='simulation-boot')
        self.position = .25
        self.target = None
        self.command_count = self.sequence = 0
        self.owned = self.admitted = self.ticking = True
        self.stopped = False
        self.ack = True
        self.current = .01
        self.worker = None

    def ready(self): return True

    def publish(self, command, owner):
        if owner != OWNER: raise AssertionError('wrong synthetic grant')
        self.command_count += 1
        self.target = command.position_normalized

    def publish_stop(self, owner_id, reason):
        self.owned = False
        self.stopped = True
        self.target = self.position

    def update(self):
        if not self.ticking: return
        self.sequence += 1
        if self.target is not None and not self.stopped:
            self.position += max(-.1, min(.1, self.target-self.position))
        active = self.stopped and self.ack
        self.buffer.ingest(packet(now=time.monotonic(), sequence=self.sequence,
            gripper_position_normalized=self.position, gripper_current_a=self.current,
            owned=self.owned, owner_id=OWNER.owner_id if self.owned else '', watchdog_latched=self.stopped,
            gripper_halt_active=active, gripper_halt_generation=1 if active else 0,
            gripper_halt_exchange_sequence=self.sequence if active else 0,
            gripper_halt_position_normalized=self.position if active else 0.))

    async def start(self):
        self.update()
        async def tick():
            while True:
                await asyncio.sleep(.001)
                self.update()
        self.worker = asyncio.create_task(tick())
        return GripperTransport(port=self, ownership=OWNER, feedback=self.buffer,
            admission_check=lambda permit, cmd:permit == 'synthetic-permit' and self.admitted,
            owns_control=lambda:self.owned)

    async def close(self):
        self.worker.cancel()
        try: await self.worker
        except asyncio.CancelledError: pass


class GripperExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sim = SimulatedGripper()
        self.transport = await self.sim.start()
        self.command = calibrated_command(calibration(), bounds(), aperture_m=.04,
            speed_fraction=.125, current_ceiling_fraction=.25)

    async def asyncTearDown(self): await self.sim.close()

    async def wait_sent(self):
        while self.sim.command_count < 2: await asyncio.sleep(.001)

    async def test_measured_aperture_success_does_not_claim_retention(self):
        receipt = await self.transport.execute(self.command, 'synthetic-permit')
        self.assertEqual(receipt.status, 'succeeded')
        self.assertTrue(receipt.measured_quiescent)
        self.assertFalse(receipt.halt_acknowledged)
        self.assertFalse(hasattr(receipt, 'holding'))
        self.assertFalse(self.sim.stopped)

    async def test_simulation_feedback_cannot_authorize_physical_port(self):
        self.sim.hardware_commands = True
        with self.assertRaisesRegex(DriverTransportError,'simulation modes'):
            GripperTransport(port=self.sim,ownership=OWNER,feedback=self.sim.buffer,
                admission_check=lambda *a:True,owns_control=lambda:True)
        self.sim.hardware_commands = False
        self.sim.buffer.simulation = False
        with self.assertRaisesRegex(DriverTransportError,'simulation modes'):
            GripperTransport(port=self.sim,ownership=OWNER,feedback=self.sim.buffer,
                admission_check=lambda *a:True,owns_control=lambda:True)

    async def test_still_fresh_old_sample_does_not_certify_current_zero_velocity(self):
        state=self.sim.buffer.read()
        older=replace(state,acquired_lower_s=state.acquired_lower_s-.02,
                      acquired_upper_s=state.acquired_upper_s-.02,sequence=state.sequence-1)
        self.sim.buffer.bounds=replace(self.sim.buffer.bounds,maximum_acceleration_m_s2=.02,state_max_age_s=.2)
        self.transport.clock=lambda:state.acquired_upper_s
        self.assertTrue(self.transport._settled([older,state]))
        self.transport.clock=lambda:state.acquired_upper_s+.1
        self.assertFalse(self.transport._settled([older,state]))

    async def test_cancel_requires_exchange_ack_and_measured_settling(self):
        cancel = asyncio.Event()
        task = asyncio.create_task(self.transport.execute(self.command, 'synthetic-permit', cancel_event=cancel))
        await self.wait_sent(); cancel.set()
        receipt = await task
        self.assertEqual(receipt.status, 'cancelled')
        self.assertTrue(receipt.halt_acknowledged and receipt.measured_quiescent)
        self.assertFalse(receipt.ownership_retained)
        with self.assertRaisesRegex(DriverTransportError, 'fault latched'):
            await self.transport.execute(self.command, 'synthetic-permit')

    async def test_python_task_cancel_drains_stop_before_return(self):
        task = asyncio.create_task(self.transport.execute(self.command, 'synthetic-permit'))
        await self.wait_sent(); task.cancel()
        receipt = await task
        self.assertEqual(receipt.status, 'cancelled')
        self.assertTrue(receipt.measured_quiescent)

    async def test_revoked_permit_or_current_fault_stops(self):
        for failure in ('permit', 'current'):
            if failure == 'current':
                await self.sim.close()
                self.sim = SimulatedGripper(); self.transport = await self.sim.start()
            task = asyncio.create_task(self.transport.execute(self.command, 'synthetic-permit'))
            await self.wait_sent()
            if failure == 'permit': self.sim.admitted = False
            else: self.sim.current = .8
            receipt = await task
            self.assertEqual(receipt.status, 'fault')
            self.assertTrue(receipt.halt_acknowledged and receipt.measured_quiescent)

    async def test_stop_publish_without_exchange_ack_is_not_quiescence(self):
        self.sim.ack = False
        cancel = asyncio.Event()
        task = asyncio.create_task(self.transport.execute(self.command, 'synthetic-permit', cancel_event=cancel))
        await self.wait_sent(); cancel.set()
        receipt = await task
        self.assertFalse(receipt.halt_acknowledged or receipt.measured_quiescent)

    async def test_stale_feedback_prevents_success_and_handback(self):
        task = asyncio.create_task(self.transport.execute(self.command, 'synthetic-permit'))
        await self.wait_sent(); self.sim.ticking = False
        receipt = await task
        self.assertEqual(receipt.status, 'fault')
        self.assertFalse(receipt.measured_quiescent)

    async def test_concurrent_and_mutated_dispatch_rejected(self):
        with self.assertRaises(DriverTransportError):
            await self.transport.execute(replace(self.command, position_normalized=.6), 'synthetic-permit')
        self.assertEqual(self.sim.command_count, 0)
        task = asyncio.create_task(self.transport.execute(self.command, 'synthetic-permit'))
        await self.wait_sent()
        with self.assertRaisesRegex(DriverTransportError, 'busy'):
            await self.transport.execute(self.command, 'synthetic-permit')
        await task


if __name__ == '__main__': unittest.main()
