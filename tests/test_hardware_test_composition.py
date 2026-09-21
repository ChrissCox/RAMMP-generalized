"""Commissioning composition boundaries; all planners and ports are test doubles."""
import asyncio
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rammp_adl.contracts import Catalog, digest
from rammp_adl.motion.commissioning import sha256
from rammp_adl.motion.curobo import CUROBO_VERSION, RAMMP_COMMIT
from rammp_adl.motion.curobo_process import WarmGpuPlanner
from rammp_adl.motion.hardware_test import CheckedGpuPlanner, checked_candidate, run_test
from rammp_adl.motion import hardware_test
from rammp_adl.motion.rolling import JointState, MotionError
from rammp_adl.motion.trial_recording import TrialRecording
from test_commissioning import settings, trajectory
from test_driver_transport import FakePort, fixture_trajectory, make_transport


class CandidateTests(unittest.TestCase):
    def test_exact_worker_trajectory_and_model_identity_required(self):
        candidate = trajectory()
        request = {'fixture': 'request'}
        robot = {'fixture': 'model'}
        settings = SimpleNamespace(world_digest='test-world', planner_model_digest=digest(robot))
        with tempfile.TemporaryDirectory() as temporary:
            config, urdf = Path(temporary)/'config', Path(temporary)/'model'
            config.write_text('planner configuration')
            urdf.write_text('unit model')
            encoded = {'joint_names': list(candidate.joint_names), 'interpolation':'quintic-hermite-v1',
                       'provenance': candidate.provenance, 'digest':candidate.digest,
                       'points':[{'time_s':p.time_s, 'position':list(p.state.position),
                                  'velocity':list(p.state.velocity), 'acceleration':list(p.state.acceleration)}
                                 for p in candidate.points]}
            response = {'status':'planned', 'request_digest':digest(request), 'world_identity':'test-world',
                        'planner_source_commit':RAMMP_COMMIT, 'curobo_version':CUROBO_VERSION,
                        'planner_config_digest':sha256(config), 'planner_urdf_digest':sha256(urdf),
                        'planner_robot_config_digest':digest(robot), 'planner_robot_config':robot,
                        'trajectory':encoded}
            self.assertEqual(checked_candidate(response, request, settings, config, urdf), candidate)
            for field in ('request_digest', 'world_identity', 'planner_source_commit', 'curobo_version',
                          'planner_config_digest', 'planner_urdf_digest', 'planner_robot_config_digest'):
                with self.subTest(field=field), self.assertRaises(MotionError):
                    checked_candidate({**response, field:'changed'}, request, settings, config, urdf)
            encoded['points'][1]['position'][0] += .001
            with self.assertRaisesRegex(MotionError, 'digest'):
                checked_candidate(response, request, settings, config, urdf)


class CompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_absent_physical_profile_rejects_before_ros_or_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'must-not-exist'
            args = SimpleNamespace(commissioning=Catalog().root/'config/commissioning.json',
                profile='free_space', simulation=False, execute=True, output=output)
            with patch('rammp_adl.motion.hardware_test.RosJointTrajectoryTransport') as transport, \
                 patch('rammp_adl.motion.hardware_test.run_gpu_worker') as planner:
                with self.assertRaisesRegex(MotionError, 'disabled'):
                    await run_test(args)
                transport.assert_not_called()
                planner.assert_not_called()
                self.assertFalse(output.exists())

    async def test_cancellation_drains_solver_and_retains_serialization(self):
        entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
        calls = []
        worker = object.__new__(WarmGpuPlanner)
        worker.planner_config = Path('/unit-planner')
        def plan(request):
            calls.append(request)
            if len(calls) == 1:
                entered.set()
                if not release.wait(2.):
                    raise RuntimeError('Unit test failed to release solver')
            else:
                second_entered.set()
            return {'status':'planned'}
        worker.plan = plan
        settings = SimpleNamespace(unchanged=lambda: None)
        facade = CheckedGpuPlanner(worker, settings=settings, robot_urdf='/unit-model')
        arguments = dict(position_m=(.1,.2,.3), quaternion_xyzw=(0.,0.,0.,1.),
                         start=JointState((0.,)*7,(0.,)*7,(0.,)*7), world=[], world_identity='unit')
        with patch('rammp_adl.motion.hardware_test.checked_candidate', return_value='checked'):
            first = asyncio.create_task(facade.plan_pose(**arguments))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1.))
            first.cancel()
            second = asyncio.create_task(facade.plan_pose(**arguments))
            try:
                await asyncio.sleep(.02)
                self.assertFalse(first.done())
                self.assertFalse(second_entered.is_set())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual(await second, 'checked')
            self.assertTrue(second_entered.is_set())

    async def test_configuration_revocation_after_solver_discards_candidate(self):
        worker = object.__new__(WarmGpuPlanner)
        worker.planner_config = Path('/unit-planner')
        current = [True]
        def unchanged():
            if not current[0]:
                raise MotionError('configuration changed')
        def plan(request):
            current[0] = False
            return {'status':'planned'}
        worker.plan = plan
        facade = CheckedGpuPlanner(worker, settings=SimpleNamespace(unchanged=unchanged), robot_urdf='/unit-model')
        with patch('rammp_adl.motion.hardware_test.checked_candidate') as check:
            with self.assertRaisesRegex(MotionError, 'configuration changed'):
                await facade.plan_pose(position_m=(0.,)*3, quaternion_xyzw=(0.,0.,0.,1.),
                    start=JointState((0.,)*7,(0.,)*7,(0.,)*7), world=[], world_identity='unit')
            check.assert_not_called()

    async def test_recording_fault_after_successful_or_ambiguous_release_never_stops_new_owner(self):
        for timeout in (False, True):
            with self.subTest(release_timeout=timeout):
                gateway, port = make_transport()
                receipt = await gateway.execute(fixture_trajectory(), permit='trusted-fixture-permit')
                self.assertTrue(receipt.release_permitted)
                if timeout:
                    port.release = lambda token: asyncio.get_running_loop().create_future()
                    with self.assertRaisesRegex(MotionError, 'deadline'):
                        await gateway.release()
                else:
                    await gateway.release()
                self.assertFalse(gateway.owns_control())
                report = {'motion_sent': 'dispatch_attempted', 'ownership_release_attempted': True}
                if not timeout:
                    report['ownership_released'] = True
                hardware_test._stop_after_fault(gateway, True, report)
                self.assertEqual(port.stops, [])
                self.assertFalse(report.get('software_stop_published', False))

    async def test_composition_fault_stops_an_owned_dispatched_operation(self):
        gateway, port = make_transport()
        report = {'motion_sent': 'dispatch_attempted'}
        hardware_test._stop_after_fault(gateway, True, report)
        self.assertEqual(len(port.stops), 1)
        self.assertTrue(report['software_stop_published'])

    async def test_cancelled_readiness_cannot_invoke_or_accept_ready_check(self):
        cancelled = asyncio.Event()
        cancelled.set()
        check = Mock(return_value=True)
        with self.assertRaisesRegex(MotionError, '[Cc]ancel'):
            await hardware_test._wait(check, timeout_s=.1, cancel_event=cancelled)
        check.assert_not_called()
        cancelled.clear()
        calls = []
        def waiting():
            calls.append(True)
            cancelled.set()
            return False
        with self.assertRaisesRegex(MotionError, '[Cc]ancel'):
            await hardware_test._wait(waiting, timeout_s=.1, period_s=.001, cancel_event=cancelled)
        self.assertEqual(len(calls), 1)
        cancelled.clear()
        def ready_while_cancelled():
            cancelled.set()
            return True
        with self.assertRaisesRegex(MotionError, '[Cc]ancel'):
            await hardware_test._wait(ready_while_cancelled, timeout_s=.1, cancel_event=cancelled)

    async def test_operator_cancel_during_validation_never_reports_prepared_or_claims_control(self):
        # Exercise the actual orchestration around paused validation. Every ROS
        # object and provider is an inert local double; no ROS graph is opened.
        for execute in (False, True):
            with self.subTest(execute=execute), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                collision_world = [{'name': 'unit-only-world'}]
                config_path, request_path = root/'commissioning.json', root/'request.json'
                config_path.write_text(json.dumps({'profiles': {'fixture': {'free_space_test': {'assets': {
                    'planner_config': {'path': 'planner.yaml'}, 'robot_urdf': {'path': 'robot.urdf'}}}}}}))
                request_path.write_text(json.dumps({'goal': {'position_m': [0.,0.,0.],
                    'quaternion_xyzw': [0.,0.,0.,1.]}, 'world': collision_world}))
                configured = replace(settings(), world_digest=digest(collision_world))
                args = SimpleNamespace(commissioning=config_path, profile='fixture', simulation=True,
                    execute=execute, output=root/'output', namespace='/fixture', request=request_path,
                    model_dir=root, wrapper_checkout=root, gpu_cache=root)
                port = FakePort()
                buffer = SimpleNamespace(require_unowned=lambda: None,
                    stationary_start=lambda **kw: port.state(time.monotonic()))
                node = Mock()
                ros = SimpleNamespace(init=Mock(), create_node=Mock(return_value=node), shutdown=Mock())
                executor = Mock()
                feedback = Mock()
                modules = {'rclpy': ros,
                    'rclpy.executors': SimpleNamespace(SingleThreadedExecutor=lambda: executor),
                    'rammp_common_interfaces.srv': SimpleNamespace(AcquireControl=SimpleNamespace(Request=SimpleNamespace))}
                entered, resume = threading.Event(), threading.Event()
                path = trajectory()
                def validate(candidate):
                    entered.set()
                    if not resume.wait(2.):
                        raise RuntimeError('Unit test failed to resume validation')
                    return SimpleNamespace(trajectory=path)
                admission = SimpleNamespace(validate=validate, check=Mock(return_value=True))
                callbacks = {}
                loop = asyncio.get_running_loop()
                with ExitStack() as stack:
                    stack.enter_context(patch.dict(sys.modules, modules))
                    stack.enter_context(patch.object(loop, 'add_signal_handler',
                        side_effect=lambda sig, callback: callbacks.__setitem__(sig, callback)))
                    stack.enter_context(patch.object(loop, 'remove_signal_handler'))
                    for name,value in (('load_settings', Mock(return_value=configured)),
                        ('DriverFeedbackBuffer', Mock(return_value=buffer)),
                        ('RosDriverFeedback', Mock(return_value=feedback)),
                        ('run_gpu_worker', Mock(return_value={})),
                        ('checked_candidate', Mock(return_value=path)),
                        ('CommissioningAdmission', Mock(return_value=admission))):
                        stack.enter_context(patch.object(hardware_test, name, value))
                    gateway = stack.enter_context(patch.object(hardware_test, 'RosJointTrajectoryTransport'))
                    work = asyncio.create_task(run_test(args))
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 1.))
                        callbacks[signal.SIGINT]()
                    finally:
                        resume.set()
                    with self.assertRaisesRegex(MotionError, '[Cc]ancel'):
                        await work
                    gateway.assert_not_called()
                    node.create_client.assert_not_called()
                result = json.loads((root/'output/result.json').read_text())
                self.assertNotIn(result['status'], ('succeeded', 'prepared'))
                self.assertFalse(result['motion_sent'])


class CompositionEvidenceTests(unittest.TestCase):
    def test_measurement_write_failure_is_persisted_without_losing_motion_outcome(self):
        recording = TrialRecording(simulation=True)
        for status in ('prepared', 'succeeded'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                report = {'status': status, 'ownership_released': status == 'succeeded'}
                with patch.object(recording, 'write', side_effect=OSError('unit disk full')):
                    with self.assertRaisesRegex(OSError, 'disk full'):
                        hardware_test._write_run_evidence(recording, output, report)
                saved = json.loads((output/'result.json').read_text())
                self.assertEqual(saved['status'], 'recording_failed')
                self.assertEqual(saved['motion_outcome'], status)
                self.assertIn('disk full', saved['recording_error'])

    def test_successful_evidence_write_preserves_motion_outcome_and_scope(self):
        recording = TrialRecording(simulation=True)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = {'status': 'succeeded', 'ownership_released': True}
            hardware_test._write_run_evidence(recording, output, report)
            saved = json.loads((output/'result.json').read_text())
            self.assertEqual(saved, report)
            evidence = json.loads((output/'measurements/report.json').read_text())
            self.assertEqual(evidence['scope'], 'admitted_simulation_trial')
            self.assertFalse(evidence['hardware_limits_established'])


if __name__ == '__main__':
    unittest.main()
