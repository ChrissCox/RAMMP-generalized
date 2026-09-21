"""Measured gripper transport through canonical handler/DAG/world, simulation only."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import unittest

from rammp_adl.contracts import Catalog, ContractError
from rammp_adl.executor import DagExecutor
from rammp_adl.handlers import ExecutionContext, SetGripperHandler, MoveToPoseHandler
from rammp_adl.hardware_gripper import MeasuredGripperBackend, MeasuredGripperProfile
from rammp_adl.resources import ResourceManager, ResourceError
from rammp_adl.safety import SafetyError, SafetySupervisor
from rammp_adl.validation import GeometryCheck, PlanValidator
from rammp_adl.world import WorldModel
from test_gripper import SimulatedGripper

ROOT=Path(__file__).resolve().parents[1]


class MeasuredGripperBackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.catalog=Catalog(ROOT)
        context=json.loads((ROOT/'examples/cabinet.context.json').read_text())
        context['available_skills']=['set_gripper']
        context['goal']={'predicate':'aperture_reached','args':{'aperture_m':.04}}
        self.world=WorldModel(context,self.catalog,trust_initial=True)
        self.sim=SimulatedGripper();self.transport=await self.sim.start()
        self.resources=ResourceManager(self.catalog.library['resources'])
        self.envelope_valid=True
        self.profile=MeasuredGripperProfile('sim_gripper',self.transport.feedback.calibration.digest,.125,.25,True)
        def command_check(context):
            try:self.resources.assert_owned(context.node_id,['GRIPPER'])
            except ResourceError:return False
            return True
        def envelope(node,snapshot,prediction,phase):
            if not self.envelope_valid:raise ContractError('Explicit simulation envelope revoked')
            return GeometryCheck(dependencies={'collision_revision':snapshot.identities()['collision_revision']},
                artifact={'simulation_only':True,'checked_aperture_range_m':[0.,.08]},
                established_facts=({'predicate':'motion_profile_valid','args':{'profile_id':'sim_gripper'},'validity':'true'},),
                valid_for_s=1.)
        self.backend=MeasuredGripperBackend(self.catalog,self.world,transport=self.transport,profiles=[self.profile],
            envelope_check=envelope,command_check=command_check,held_check=lambda:True)
        self.args={'aperture_m':.04,'profile_id':'sim_gripper'}

    async def asyncTearDown(self):await self.sim.close()

    def dispatch_context(self):
        snapshot=self.world.snapshot()
        artifact=self.backend.validate({'id':'aperture','skill':'set_gripper','args':self.args},snapshot,{},'dispatch').artifact
        self.resources.try_acquire('aperture',['GRIPPER'])
        return ExecutionContext(snapshot.context['task_id'],'aperture',execution_epoch=snapshot.execution_epoch,
                                snapshot=snapshot,validation_artifact=artifact,validation_id='explicit-test-receipt')

    async def test_actual_dag_commits_measured_aperture_and_releases_resource(self):
        registry=self.backend.registry(capabilities={'calibrated_aperture'},mode='simulation')
        validator=PlanValidator(self.catalog,registry,self.world,geometry=self.backend)
        executor=DagExecutor(self.catalog,registry,self.world,validator,self.backend)
        self.resources=executor.resources
        snapshot=self.world.snapshot()
        plan={'schema_version':'1.0.0','skill_library_hash':self.catalog.hash,'task_id':snapshot.context['task_id'],
              'snapshot_id':snapshot.snapshot_id,'execution_epoch':snapshot.execution_epoch,
              'nodes':[{'id':'aperture','skill':'set_gripper','args':self.args}],'edges':[]}
        result=await executor.run_plan(plan)
        self.assertEqual(result.status,'succeeded',result.to_dict())
        self.assertTrue(result.nodes[0].commit_receipt)
        self.assertTrue(result.nodes[0].backend_quiescent)
        self.assertGreater(self.sim.command_count,0)
        self.assertEqual(self.world.snapshot().fact('aperture_reached',{'aperture_m':.04}),'true')
        self.assertEqual(executor.resources.owners,{})
        self.assertTrue(result.simulation_only)

    async def test_handler_uses_acquired_aperture_not_command_assumption(self):
        context=self.dispatch_context()
        result=await self.backend.handlers()['set_gripper'].execute(self.args,context)
        self.assertEqual(result.status,'succeeded',result)
        self.assertEqual(result.evidence[0]['data']['measured_aperture_m'],.04)
        self.assertTrue(result.evidence[0]['data']['simulation_only'])
        self.assertFalse(result.evidence[0]['data']['retention_evaluated'])
        self.assertEqual([effect['predicate'] for effect in result.proposed_effects],['aperture_reached'])
        self.assertLessEqual(result.evidence[0]['observed_at'],self.world.clock())
        self.assertIn('exchange_sequence',result.evidence[0]['data'])
        self.assertEqual(self.world.snapshot().fact('aperture_reached',{'aperture_m':.04}),'unknown')
        # The handler only proposes; only the WorldModel executor commit writes facts.

    async def test_unissued_or_admission_only_artifact_never_commands(self):
        snapshot=self.world.snapshot()
        for artifact in (None,self.backend.validate({'id':'aperture','skill':'set_gripper','args':self.args},snapshot,{},'admission').artifact):
            context=ExecutionContext(snapshot.context['task_id'],'aperture',snapshot=snapshot,
                                     validation_artifact=artifact,validation_id='forged-json-id')
            result=await self.backend.handlers()['set_gripper'].execute(self.args,context)
            self.assertEqual(result.status,'failed')
        self.assertEqual(self.sim.command_count,0)

    async def test_aperture_artifact_is_immutable_and_single_use(self):
        context=self.dispatch_context()
        with self.assertRaises(TypeError):context.validation_artifact.dependencies['collision_revision']=2
        first=await self.backend.handlers()['set_gripper'].execute(self.args,context)
        self.assertEqual(first.status,'succeeded')
        count=self.sim.command_count
        repeated=await self.backend.handlers()['set_gripper'].execute(self.args,context)
        self.assertEqual(repeated.status,'failed')
        self.assertEqual(self.sim.command_count,count)

    async def test_epoch_and_resource_revocation_stop_before_more_publishes(self):
        context=self.dispatch_context()
        task=asyncio.create_task(self.backend.handlers()['set_gripper'].execute(self.args,context))
        while self.sim.command_count<2:await asyncio.sleep(.001)
        count=self.sim.command_count
        self.world.cancel_epoch()
        result=await task
        self.assertEqual(result.status,'failed')
        self.assertTrue(result.backend_quiescent)
        self.assertEqual(self.sim.command_count,count)
        self.assertFalse(result.proposed_effects)
        self.assertTrue(self.sim.stopped)

    async def test_changed_attempt_cannot_reuse_active_permit(self):
        context=self.dispatch_context()
        task=asyncio.create_task(self.backend.handlers()['set_gripper'].execute(self.args,context))
        while self.sim.command_count<2:await asyncio.sleep(.001)
        context.attempt+=1
        result=await task
        self.assertEqual(result.status,'failed')
        self.assertFalse(result.proposed_effects)
        self.assertTrue(result.backend_quiescent)

    async def test_cancellation_at_transport_completion_cannot_publish_success_effect(self):
        context=self.dispatch_context()
        original=self.transport.execute
        parent=None
        async def complete_and_cancel(*args,**kwargs):
            receipt=await original(*args,**kwargs)
            parent.cancel()
            return receipt
        self.transport.execute=complete_and_cancel
        parent=asyncio.create_task(self.backend.handlers()['set_gripper'].execute(self.args,context))
        outcome=await parent
        self.assertEqual(outcome.status,'cancelled')
        self.assertFalse(outcome.proposed_effects)
        self.assertTrue(outcome.backend_quiescent)
        self.assertTrue(self.sim.stopped)

    async def test_changed_calibration_fails_before_command(self):
        context=self.dispatch_context()
        self.transport.feedback.calibration=replace(self.transport.feedback.calibration,evidence_id='different-table')
        result=await self.backend.handlers()['set_gripper'].execute(self.args,context)
        self.assertEqual(result.status,'failed')
        self.assertEqual(self.sim.command_count,0)

    async def test_geometry_cannot_establish_emptiness_or_holding(self):
        self.backend.envelope_check=lambda *args:GeometryCheck(valid_for_s=1.,established_facts=(
            {'predicate':'gripper_empty','args':{'robot_id':'robot'},'validity':'true'},))
        with self.assertRaisesRegex(ContractError,'cannot invent'):
            self.dispatch_context()
        self.assertEqual(self.sim.command_count,0)

    async def test_slow_geometry_cannot_refresh_its_snapshot_expiry(self):
        now=[self.world.clock()]
        self.world.clock=lambda:now[0]
        def slow(*args):
            now[0]+=.02
            return GeometryCheck(valid_for_s=.01)
        self.backend.envelope_check=slow
        with self.assertRaisesRegex(ContractError,'expired'):
            self.dispatch_context()
        self.assertEqual(self.sim.command_count,0)

    async def test_repeated_cancellation_of_stop_retains_work_until_halt_settles(self):
        context=self.dispatch_context()
        self.sim.ack=False
        task=asyncio.create_task(self.backend.handlers()['set_gripper'].execute(self.args,context))
        while self.sim.command_count<2:await asyncio.sleep(.001)
        stopping=asyncio.create_task(self.backend.stop_skill('set_gripper','test-stop'))
        await asyncio.sleep(.003);stopping.cancel()
        await asyncio.sleep(.003);stopping.cancel()
        self.assertFalse(stopping.done())
        self.sim.ack=True
        await stopping
        result=await task
        self.assertEqual(result.status,'cancelled')
        self.assertTrue(result.backend_quiescent)

    async def test_physical_catalog_and_supervisor_remain_disabled(self):
        # set_gripper is implemented: hardware registration needs the declared
        # capability AND the operator's commissioned assertion, never one alone.
        self.assertEqual(self.backend.registry(capabilities={'calibrated_aperture'},mode='hardware',commissioned=False).available_skills,())
        self.assertEqual(self.backend.registry(capabilities=set(),mode='hardware',commissioned=True).available_skills,())
        registry=self.backend.registry(capabilities={'calibrated_aperture'},mode='hardware',commissioned=True)
        self.assertEqual(registry.available_skills,('set_gripper',))
        self.backend.hardware_commands=True;self.backend.mode='physical_gripper'
        self.assertIsInstance(SetGripperHandler(self.backend),SetGripperHandler)
        with self.assertRaises(ValueError):MoveToPoseHandler(self.backend)
        class SpoofedHandler(MoveToPoseHandler):skill_id='set_gripper'
        class SubclassedHandler(SetGripperHandler):pass
        with self.assertRaises(ValueError):SpoofedHandler(self.backend)
        with self.assertRaises(ValueError):SubclassedHandler(self.backend)
        with self.assertRaises(ContractError):self.backend.registry(capabilities={'calibrated_aperture'},mode='simulation')
        safety=SafetySupervisor(self.backend,epoch_getter=lambda:1,invalidate_epoch=lambda:2)
        with self.assertRaises(SafetyError):safety.check_dispatch(1,skills=['set_gripper'])
        class Spoof:
            hardware_commands=True
        with self.assertRaises(ValueError):SetGripperHandler(Spoof())
        class PretendMeasured(MeasuredGripperBackend):pass
        pretend=object.__new__(PretendMeasured);pretend.hardware_commands=True
        with self.assertRaises(ValueError):SetGripperHandler(pretend)


if __name__=='__main__':unittest.main()
