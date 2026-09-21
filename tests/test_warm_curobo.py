"""Persistent worker framing/lifecycle tests; subprocess fixtures never load GPU/ROS."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from rammp_adl.contracts import digest
from rammp_adl.motion.curobo import CUROBO_VERSION, RAMMP_COMMIT
from rammp_adl.motion.curobo_process import (REVIEWED_GPU_IMAGE, WarmGpuPlanner, _file_digest,
                                               _UNRESOLVED_PROCESSES, _audit_worker_assets)
from rammp_adl.motion.curobo_worker import serve
from rammp_adl.motion.rolling import MotionError
from rammp_adl.motion.gpu_lease import JOURNAL_NAME
from test_curobo_worker import request


class WarmServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_initializes_once_and_serializes_independent_requests(self):
        source = io.StringIO(''.join(json.dumps({'request_id':i, 'request':request()})+'\n' for i in (1,2)))
        output = io.StringIO()
        adapter = object()
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'config'; config.write_text('{}')
            with patch('rammp_adl.motion.curobo_worker.RammpCuroboAdapter.load', AsyncMock(return_value=adapter)) as load, \
                 patch('rammp_adl.motion.curobo_worker.plan_loaded', AsyncMock(return_value={'status':'planned'})) as plan:
                await serve(source_root='fixture',planner_config=config,input_stream=source,output_stream=output)
            load.assert_awaited_once()
            self.assertEqual(plan.await_count,2)
            for call in plan.await_args_list:self.assertIs(call.kwargs['adapter'],adapter)
        frames = [json.loads(s) for s in output.getvalue().splitlines()]
        self.assertEqual(frames[0]['status'],'ready')
        self.assertEqual([f['request_id'] for f in frames[1:]],[1,2])

    async def test_regressed_or_unbounded_protocol_stops_session(self):
        for line in ('{"request_id":2,"request":{}}\n', 'x'*1048577, '{"request_id":1,"request_id":1,"request":{}}\n'):
            with self.subTest(line=line[:60]), tempfile.TemporaryDirectory() as tmp:
                config=Path(tmp)/'config';config.write_text('{}'); output=io.StringIO()
                with patch('rammp_adl.motion.curobo_worker.RammpCuroboAdapter.load',AsyncMock(return_value=object())), \
                     patch('rammp_adl.motion.curobo_worker.plan_loaded',AsyncMock()) as plan:
                    await serve(source_root='fixture',planner_config=config,input_stream=io.StringIO(line),output_stream=output)
                    plan.assert_not_called()
                self.assertEqual(json.loads(output.getvalue().splitlines()[-1])['status'],'protocol_error')


class WarmProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); (self.root/'model').mkdir();(self.root/'wrapper').mkdir()
        self.config=self.root/'model/config.json';self.config.write_text('{}')
        self.popen=subprocess.Popen
        self.behavior = 'normal'
        self.response_patch = {}
        self.daemon_available = True
        self.docker_calls = []
        self.child = None
        self.addCleanup(self.cleanup_child)

    def cleanup_child(self):
        if self.child is not None and self.child.poll() is None:
            self.child.kill(); self.child.wait(timeout=2.)

    def planner(self, output='output', **kwargs):
        options = {'startup_timeout_s':2., 'request_timeout_s':2.}
        options.update(kwargs)
        return WarmGpuPlanner(planner_config=self.config, model_dir=self.config.parent,
            wrapper_checkout=self.root/'wrapper',gpu_cache=self.root/'cache',output=self.root/output, **options)

    def fake_popen(self, command, **kwargs):
        # Real bounded pipe I/O, but explicitly fabricated non-GPU responses.
        journal = json.loads((self.root/'cache'/JOURNAL_NAME).read_text())
        self.assertEqual(journal['container_name'], command[command.index('--name')+1])
        self.assertEqual(len(kwargs['pass_fds']), 1)
        ready={'protocol':1,'status':'ready','hardware_commands':False,'planner_source_commit':RAMMP_COMMIT,
               'curobo_version':CUROBO_VERSION,'planner_config_digest':_file_digest(self.config),'initialization_wall_s':0.}
        robot = {'kinematics':{'urdf_path':'/usr/local/lib/python3.10/dist-packages/curobo/content/assets/robot/fixture.urdf',
                 'asset_root_path':'/usr/local/lib/python3.10/dist-packages/curobo/content/assets/robot',
                 'base_link':'base_link', 'ee_link':'tool'}}
        result = {'status':'planned','hardware_commands':False,'fixture_only':True,
                  'curobo_planned':True,'hardware_validated':False,'independent_validation_required':True,
                  'planner_source_commit':RAMMP_COMMIT,'curobo_version':CUROBO_VERSION,
                  'planner_config_digest':_file_digest(self.config),'planner_robot_config':robot,
                  'planner_robot_config_digest':digest(robot),'planner_urdf_digest':'sha256:'+'a'*64,
                  'base_frame':'base_link','ee_link':'tool'}
        script = """import sys,json,hashlib,time
print(sys.argv[1],flush=True)
behavior=sys.argv[2]
if behavior=='ignore_stdin': time.sleep(60)
for line in sys.stdin:
 f=json.loads(line)
 if behavior=='ignore_response': time.sleep(60)
 if behavior=='malformed':
  print('{"protocol":1,"protocol":1}',flush=True)
  continue
 r=json.loads(sys.argv[3])
 r['request_digest']='sha256:'+hashlib.sha256(json.dumps(f['request'],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
 r['world_identity']=f['request']['world_identity']
 patch=json.loads(sys.argv[4]);r.update(patch.pop('result',{}))
 frame={'protocol':1,'request_id':f['request_id'],'result':r}
 frame.update(patch)
 print(json.dumps(frame),flush=True)
"""
        self.child = self.popen([sys.executable,'-u','-c',script,json.dumps(ready),self.behavior,
                                  json.dumps(result),json.dumps(self.response_patch)],**kwargs)
        return self.child

    def docker(self, command, **kwargs):
        self.docker_calls.append(command)
        if command[:3] == ['docker','image','inspect']:
            self.assertEqual(command[-1],REVIEWED_GPU_IMAGE)
            return subprocess.CompletedProcess(command,0,stdout=json.dumps([{'Id':REVIEWED_GPU_IMAGE,'Config':{'Entrypoint':['python3']}}]))
        if not self.daemon_available:
            return subprocess.CompletedProcess(command,1,stdout='',stderr='Daemon unavailable')
        if command[:3] == ['docker','ps','-a']:
            return subprocess.CompletedProcess(command,0,stdout='')
        if command[:3] == ['docker','rm','-f']:
            self.assertTrue(command[-1].startswith('rammp-warm-planner-'))
            if self.child is not None and self.child.poll() is None:
                self.child.kill()
            return subprocess.CompletedProcess(command,0,stdout=command[-1])
        self.fail('Unexpected Docker request: '+str(command))

    def test_one_process_multiple_requests_and_exclusive_lease(self):
        with patch('rammp_adl.motion.curobo_process.subprocess.run',self.docker), \
             patch('rammp_adl.motion.curobo_process.subprocess.Popen',self.fake_popen) as spawn:
            with self.planner() as planner:
                process=planner._process
                for i in range(2): self.assertTrue(planner.plan(request())['fixture_only'])
                self.assertIs(planner._process,process)
                with self.assertRaisesRegex(MotionError,'already leased'):self.planner('second').start()
                self.assertEqual(len(list((self.root/'output').glob('response-*.json'))),2)
            self.assertIsNotNone(process.poll())
            self.assertFalse((self.root/'cache'/JOURNAL_NAME).exists())

    def test_changed_model_closes_process_and_refuses_reuse(self):
        with patch('rammp_adl.motion.curobo_process.subprocess.run',self.docker), \
             patch('rammp_adl.motion.curobo_process.subprocess.Popen',self.fake_popen):
            with self.planner() as planner:
                self.config.write_text('{"changed":true}')
                with self.assertRaisesRegex(MotionError,'assets changed'):planner.plan(request())
                self.assertIsNotNone(planner._process.poll())
                with self.assertRaisesRegex(MotionError,'unavailable'):planner.plan(request())

    def test_moving_request_never_starts_or_contacts_process(self):
        planner=self.planner(); r=request(); r['start']['velocity'][0]=.1
        with self.assertRaisesRegex(MotionError,'moving start'):planner.plan(r)
        self.assertIsNone(planner._process)
        self.assertFalse(planner.output.exists())

    def test_output_cannot_mutate_model_directory(self):
        with self.assertRaisesRegex(MotionError,'outside immutable'):self.planner('model/output')

    def test_client_exit_does_not_release_lease_when_daemon_is_unavailable(self):
        with patch('rammp_adl.motion.curobo_process.subprocess.run',self.docker), \
             patch('rammp_adl.motion.curobo_process.subprocess.Popen',self.fake_popen):
            planner = self.planner(); planner.start()
            self.daemon_available = False
            try:
                with self.assertRaisesRegex(MotionError,'teardown is unproven; planner lease retained'):
                    planner.close()
                self.assertEqual(planner._process.poll(),0)  # EOF exited normally.
                self.assertIsNotNone(planner._lease)
                self.assertIn(planner,_UNRESOLVED_PROCESSES)
                self.assertTrue((self.root/'cache'/JOURNAL_NAME).exists())
                with self.assertRaisesRegex(MotionError,'already leased'):
                    self.planner('second').start()
            finally:
                self.daemon_available = True
                planner.close()  # A later positive daemon response permits release.
            self.assertIsNone(planner._lease)
            self.assertFalse((self.root/'cache'/JOURNAL_NAME).exists())
            self.assertNotIn(planner,_UNRESOLVED_PROCESSES)
            removal = [c for c in self.docker_calls if c[:3] == ['docker','rm','-f']]
            self.assertEqual(removal,[['docker','rm','-f',planner.name]])

    def test_forced_client_exit_with_daemon_failure_retains_lease(self):
        self.behavior = 'ignore_stdin'
        with patch('rammp_adl.motion.curobo_process.subprocess.run',self.docker), \
             patch('rammp_adl.motion.curobo_process.subprocess.Popen',self.fake_popen), \
             patch('rammp_adl.motion.curobo_process.CLIENT_STOP_TIMEOUT_S',.05):
            planner = self.planner(); planner.start(); self.daemon_available = False
            try:
                with self.assertRaisesRegex(MotionError,'lease retained'): planner.close()
                self.assertIsNotNone(planner._process.poll())
                self.assertIsNotNone(planner._lease)
                self.assertIn(planner,_UNRESOLVED_PROCESSES)
            finally:
                self.daemon_available = True; planner.close()

    def test_successful_remove_without_positive_status_still_retains_lease(self):
        def unavailable_status(command, **kwargs):
            if command[:3] == ['docker','rm','-f']:
                return subprocess.CompletedProcess(command,0,stdout=command[-1])
            return self.docker(command,**kwargs)
        with patch('rammp_adl.motion.curobo_process.subprocess.run',unavailable_status), \
             patch('rammp_adl.motion.curobo_process.subprocess.Popen',self.fake_popen):
            planner=self.planner();planner.start();self.daemon_available=False
            try:
                with self.assertRaisesRegex(MotionError,'lease retained'):planner.close()
                self.assertIsNotNone(planner._lease)
            finally:
                self.daemon_available=True;planner.close()

    def test_only_positive_exact_name_stopped_status_establishes_quiescence(self):
        planner=self.planner()
        for row in ('', planner.name+'\texited\n', planner.name+'\tdead\n'):
            with self.subTest(row=row), patch('rammp_adl.motion.curobo_process.subprocess.run',
                    return_value=subprocess.CompletedProcess([],0,stdout=row)):
                self.assertTrue(planner._container_stopped())
        for row in ('other\texited\n', planner.name+'\trunning\n', planner.name+'\tcreated\n',
                    planner.name+'\tpaused\n', planner.name+'\texited\nother\texited\n'):
            with self.subTest(row=row), patch('rammp_adl.motion.curobo_process.subprocess.run',side_effect=[
                    subprocess.CompletedProcess([],0,stdout=row), subprocess.CompletedProcess([],1,stdout='')]):
                self.assertFalse(planner._container_stopped())
        for state,expected in (({'Status':'exited','Running':False,'Restarting':False,'Pid':0},True),
                               ({'Status':'exited','Running':False,'Restarting':False,'Pid':1},False),
                               ({'Status':'exited','Running':False},False),
                               ({'Status':'exited','Running':False,'Restarting':True,'Pid':0},False)):
            with self.subTest(state=state), patch('rammp_adl.motion.curobo_process.subprocess.run',side_effect=[
                    subprocess.CompletedProcess([],1,stdout=''),
                    subprocess.CompletedProcess([],0,stdout=json.dumps(state))]):
                self.assertEqual(planner._container_stopped(),expected)

    def test_worker_not_reading_stdin_is_deadline_bounded_and_reaped(self):
        self.behavior='ignore_stdin'
        value=request();value['world'][0]['fixture_padding']='x'*300000;value['world_identity']=digest(value['world'])
        with patch('rammp_adl.motion.curobo_process.subprocess.run',self.docker), \
             patch('rammp_adl.motion.curobo_process.subprocess.Popen',self.fake_popen), \
             patch('rammp_adl.motion.curobo_process.CLIENT_STOP_TIMEOUT_S',.05):
            with self.planner(request_timeout_s=.12) as planner:
                began=time.monotonic()
                with self.assertRaisesRegex(MotionError,'write deadline exceeded'): planner.plan(value)
                self.assertLess(time.monotonic()-began,1.)
                self.assertIsNotNone(planner._process.poll())
                self.assertIsNone(planner._lease)

    def test_send_and_receive_share_one_deadline(self):
        self.behavior='ignore_response'
        with patch('rammp_adl.motion.curobo_process.subprocess.run',self.docker), \
             patch('rammp_adl.motion.curobo_process.subprocess.Popen',self.fake_popen), \
             patch('rammp_adl.motion.curobo_process.CLIENT_STOP_TIMEOUT_S',.05):
            with self.planner(request_timeout_s=.15) as planner:
                original_write=planner._write_request
                def delayed_write(frame,deadline):
                    time.sleep(.08);original_write(frame,deadline)
                with patch.object(planner,'_write_request',delayed_write), \
                     patch.object(planner,'_receive',wraps=planner._receive) as receive:
                    with self.assertRaisesRegex(MotionError,'response deadline exceeded'):planner.plan(request())
                    self.assertGreater(receive.call_args.args[0],0.)
                    self.assertLess(receive.call_args.args[0],.08)
                self.assertIsNone(planner._lease)

    def test_malformed_protocol_and_identity_mismatches_destroy_session(self):
        patches=[{'protocol':True},{'request_id':True},{'request_id':2},
                 {'result':{'request_digest':'wrong'}},{'result':{'world_identity':'wrong'}},
                 {'result':{'planner_source_commit':'wrong'}},{'result':{'planner_config_digest':'wrong'}},
                 {'result':{'planner_robot_config_digest':'wrong'}},{'result':{'planner_urdf_digest':'wrong'}},
                 {'result':{'base_frame':'wrong'}},{'result':{'status':'unknown'}},
                 {'result':{'hardware_validated':True}},{'result':{'independent_validation_required':False}}]
        for index,change in enumerate([None,*patches]):
            with self.subTest(change=change), \
                 patch('rammp_adl.motion.curobo_process.subprocess.run',self.docker), \
                 patch('rammp_adl.motion.curobo_process.subprocess.Popen',self.fake_popen):
                self.behavior='malformed' if change is None else 'normal';self.response_patch=change or {}
                with self.planner('protocol-'+str(index)) as planner:
                    with self.assertRaises(MotionError):planner.plan(request())
                    with self.assertRaisesRegex(MotionError,'unavailable'):planner.plan(request())
                    self.assertIsNone(planner._lease)

    def test_model_symlink_and_cache_alias_are_rejected_before_spawn(self):
        (self.root/'model/alias').symlink_to(self.root/'wrapper',target_is_directory=True)
        with patch('rammp_adl.motion.curobo_process.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(MotionError,'without symlinks'):self.planner().start()
            spawn.assert_not_called()
        with self.assertRaisesRegex(MotionError,'must not alias'):
            WarmGpuPlanner(planner_config=self.config,model_dir=self.config.parent,
                           wrapper_checkout=self.root/'wrapper',gpu_cache=self.root,output=self.root/'output')

    def test_worker_rejects_yaml_and_asset_dependencies_outside_readonly_model(self):
        model=self.config.parent;robot=model/'robot.json';world=model/'world.json';urdf=model/'robot.urdf'
        urdf.write_text('<robot name="fixture"/>');world.write_text('{}')
        kin={'urdf_path':str(urdf),'asset_root_path':str(model),'collision_spheres':{}}
        robot.write_text(json.dumps({'robot_cfg':{'kinematics':kin}}))
        good={'robot':str(robot),'world':str(world)};self.config.write_text(json.dumps(good))
        loader=lambda path:json.loads(Path(path).read_text())
        evidence=_audit_worker_assets(model,self.config,load_yaml=loader)
        self.assertEqual(set(evidence),{str(p) for p in (self.config,robot,world,urdf)})
        outside=self.root/'mutable.json';outside.write_text('{}')
        for field in ('robot','world'):
            with self.subTest(field=field):
                self.config.write_text(json.dumps({**good,field:str(outside)}))
                with self.assertRaisesRegex(MotionError,'outside read-only'):
                    _audit_worker_assets(model,self.config,load_yaml=loader)
        self.config.write_text(json.dumps(good))
        for field,value in (('urdf_path',str(outside)),('asset_root_path',str(self.root)),
                            ('collision_spheres','/gpu-cache/spheres.yaml'),('use_usd_kinematics',True)):
            with self.subTest(field=field):
                robot.write_text(json.dumps({'robot_cfg':{'kinematics':{**kin,field:value}}}))
                with self.assertRaises(MotionError):_audit_worker_assets(model,self.config,load_yaml=loader)

    def test_mutable_asset_path_or_model_identity_change_is_rejected(self):
        with patch('rammp_adl.motion.curobo_process.subprocess.run',self.docker), \
             patch('rammp_adl.motion.curobo_process.subprocess.Popen',self.fake_popen):
            with self.planner() as planner:
                result=planner.plan(request())
                for asset in ('/gpu-cache/robot.urdf','/tmp/robot.urdf',str(self.config.parent/'../outside')):
                    result['planner_robot_config']['kinematics']['urdf_path']=asset
                    result['planner_robot_config_digest']=digest(result['planner_robot_config'])
                    with self.assertRaisesRegex(MotionError,'asset paths|outside the pinned'):
                        planner._validate_result_provenance(result,request())
            with self.planner('model-change') as planner:
                result=planner.plan(request());result['planner_urdf_digest']='sha256:'+'b'*64
                with self.assertRaisesRegex(MotionError,'identity changed'):
                    planner._validate_result_provenance(result,request())
