"""Durable cache exclusion only; subprocesses never run Docker, ROS or GPU work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rammp_adl.motion.gpu_lease import GpuLeaseJournal, JOURNAL_NAME
from rammp_adl.motion.rolling import MotionError


NAME = 'rammp-reactive-probe-0123456789ab'


class GpuLeaseJournalTests(unittest.TestCase):
    def test_reconcile_only_clears_after_positive_exact_name_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = GpuLeaseJournal(tmp)
            journal.reserve(NAME)
            for result in (subprocess.CompletedProcess([], 1, stdout=''),
                           subprocess.CompletedProcess([], 0, stdout=NAME+'\n'),
                           subprocess.CompletedProcess([], 0, stdout='unexpected\n')):
                with self.subTest(result=result), patch('subprocess.run', return_value=result) as query:
                    with self.assertRaisesRegex(MotionError, 'durable planner lease retained'):
                        journal.reconcile()
                    self.assertTrue(journal.path.exists())
                    self.assertEqual(query.call_args.args[0], ['docker', 'ps', '-a', '--filter',
                        'name=^/'+NAME+'$', '--format', '{{.Names}}'])
            with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, stdout='')):
                journal.reconcile()
            self.assertFalse(journal.path.exists())

    def test_daemon_error_does_not_clear_or_remove_previous_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = GpuLeaseJournal(tmp)
            journal.reserve(NAME)
            with patch('subprocess.run', side_effect=OSError('daemon unavailable')) as query:
                with self.assertRaises(MotionError):
                    journal.reconcile()
            self.assertTrue(journal.path.exists())
            self.assertEqual(query.call_count, 1)
            self.assertNotIn('rm', query.call_args.args[0])

    def test_malformed_symlink_or_changed_ownership_never_queries_docker(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = GpuLeaseJournal(tmp)
            for raw in ('{}', '{', json.dumps({'protocol': True, 'container_name': NAME}),
                        json.dumps({'protocol': 1, 'container_name': 'unowned-container'})):
                journal.path.write_text(raw)
                with self.subTest(raw=raw), patch('subprocess.run') as query:
                    with self.assertRaises(MotionError): journal.reconcile()
                    query.assert_not_called()
            journal.path.unlink()
            journal.path.symlink_to(Path(tmp)/'missing')
            with patch('subprocess.run') as query:
                with self.assertRaises(MotionError): journal.reconcile()
                query.assert_not_called()
            journal.path.unlink()
            journal.reserve(NAME)
            with patch('subprocess.run') as query:
                with self.assertRaisesRegex(MotionError, 'changed'):
                    journal.release('rammp-warm-planner-0123456789ab')
                query.assert_not_called()

    def test_both_launchers_block_cross_process_reuse_after_pre_spawn_client_crash(self):
        # os._exit runs exactly where Docker would have been spawned. It loses
        # every Python object/flock without cleanup, but the journal must remain.
        script = r'''
import json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rammp_adl.motion.curobo import RAMMP_COMMIT
from rammp_adl.motion.curobo_process import REVIEWED_GPU_IMAGE, WarmGpuPlanner
from rammp_adl.motion.gpu_lease import JOURNAL_NAME
from rammp_adl.motion.reactive_probe import run_isolated
from rammp_adl.motion.rolling import MotionError
root,kind,phase=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
def spawn(command, **kwargs):
    marker=json.loads((root/'cache'/JOURNAL_NAME).read_text())
    assert marker['container_name']==command[command.index('--name')+1]
    assert len(kwargs['pass_fds'])==1
    if phase=='crash': os._exit(17)
    raise AssertionError('A surviving journal permitted a second worker')
def fake_run(command, **kwargs):
    if command[0]=='git':
        return subprocess.CompletedProcess(command,0,stdout=RAMMP_COMMIT+'\n' if 'rev-parse' in command else '')
    if command[:3]==['docker','ps','-a']:
        assert phase=='blocked'
        name=json.loads((root/'cache'/JOURNAL_NAME).read_text())['container_name']
        assert command[command.index('--filter')+1]=='name=^/'+name+'$'
        return subprocess.CompletedProcess(command,0,stdout=name+'\n')
    if command[:3]==['docker','image','inspect']:
        assert phase=='crash', 'Image inspection preceded unresolved lease rejection'
        return subprocess.CompletedProcess(command,0,stdout=json.dumps([{'Id':REVIEWED_GPU_IMAGE,'Config':{'Entrypoint':['python3']}}]))
    if command[:2]==['docker','run']: return spawn(command,**kwargs)
    raise AssertionError('Unexpected command: '+str(command))
with patch('subprocess.run',fake_run),patch('subprocess.Popen',spawn):
    try:
        if kind=='warm':
            WarmGpuPlanner(planner_config=root/'model/config.json',model_dir=root/'model',
                wrapper_checkout=root/'wrapper',gpu_cache=root/'cache',output=root/(phase+'-'+kind)).start()
        else:
            run_isolated(SimpleNamespace(candidate=root/'candidate.json',request=root/'request.json',
                contract=root/'contract.json',wrapper_checkout=root/'wrapper',gpu_cache=root/'cache',
                output=root/(phase+'-'+kind)))
    except MotionError as exc:
        assert phase=='blocked' and 'durable planner lease retained' in str(exc), str(exc)
        print('cross-process exclusion passed')
    else: raise AssertionError('Expected crash or durable exclusion')
'''
        for first, second in (('reactive', 'warm'), ('warm', 'reactive')):
            with self.subTest(first=first), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                (root/'model').mkdir(); (root/'wrapper').mkdir()
                for name in ('candidate.json','request.json','contract.json','model/config.json'):
                    (root/name).write_text('{}')
                environment=dict(os.environ, PYTHONNOUSERSITE='1')
                crashed=subprocess.run([sys.executable,'-c',script,tmp,first,'crash'],
                    capture_output=True,text=True,timeout=15,env=environment)
                self.assertEqual(crashed.returncode,17,crashed.stderr)
                marker=(root/'cache'/JOURNAL_NAME).read_bytes()
                blocked=subprocess.run([sys.executable,'-c',script,tmp,second,'blocked'],
                    capture_output=True,text=True,timeout=15,env=environment)
                self.assertEqual(blocked.returncode,0,blocked.stderr)
                self.assertIn('cross-process exclusion passed',blocked.stdout)
                self.assertEqual((root/'cache'/JOURNAL_NAME).read_bytes(),marker)


if __name__ == '__main__':
    unittest.main()
