"""No GPU or robot I/O: independent candidate-screen and MPC boundary guards."""
import json
import asyncio
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

from rammp_adl.motion.curobo import CuroboMpcAdapter, CuroboUnavailable
from rammp_adl.motion.reactive_probe import moving_boundary, optimizer_robot_config, screen_horizon
from rammp_adl.motion.rolling import JointLimits, JointState, JointTrajectory, MotionError, TrajectoryPoint
from rammp_adl.motion.reactive_rehearsal import _PhysicsIO
from rammp_adl.simulation import MujocoReplay, fixture_joint_trajectory


class ProbeScreenTests(unittest.TestCase):
    def test_regeneration_tightens_only_optimizer_without_mutating_model_or_limits(self):
        robot = {'kinematics':{'cspace':{'max_acceleration':4.,'max_jerk':100.},'base_link':'base_link'}}
        limits = [4.]*7
        result = optimizer_robot_config(robot,limits,.2)
        self.assertEqual(result['kinematics']['cspace']['max_acceleration'],.8)
        self.assertEqual(robot['kinematics']['cspace']['max_acceleration'],4.)
        self.assertEqual(result['kinematics']['cspace']['max_jerk'],100.)
        self.assertEqual(limits,[4.]*7)
        with self.assertRaises(MotionError):
            optimizer_robot_config(robot,limits,2.)

    def test_selects_original_nonzero_velocity_and_acceleration(self):
        a = JointState((0.,), (0.,), (0.,))
        b = JointState((.1,), (.2,), (.3,))
        path = JointTrajectory(('j',), (TrajectoryPoint(0.,a),TrajectoryPoint(.5,b),TrajectoryPoint(1.,a)), 'TEST')
        self.assertIs(moving_boundary(path), b)
        with self.assertRaisesRegex(MotionError, 'no moving'):
            moving_boundary(JointTrajectory(('j',), (TrajectoryPoint(0.,a),TrajectoryPoint(1.,a)), 'TEST'))

    def test_quintic_between_knots_can_reject_bounded_endpoint_derivatives(self):
        # Endpoints satisfy acceleration=0, yet their connecting quintic needs
        # acceleration greater than the explicit 1 rad/s² screen limit.
        a = JointState((0.,), (.01,), (0.,))
        b = JointState((.1,), (.01,), (0.,))
        path = JointTrajectory(('j',), (TrajectoryPoint(0.,a),TrajectoryPoint(.1,b)), 'TEST')
        limits = JointLimits((-1.,),(1.,),(10.,),(1.,))
        verifier = NS(check_state=lambda state: None, maximum_fk_position_error_m=0., maximum_fk_rotation_error_rad=0.)
        result = screen_horizon(path, a, limits, verifier, sample_dt_s=.005)
        self.assertFalse(result['sampled_checks_passed'])
        self.assertIn('joint acceleration limit',result['rejection_reasons'])
        self.assertGreater(result['maximum_acceleration_rad_s2'], 1.)
        self.assertFalse(result['execution_permitted'])

    def test_collision_rejection_never_creates_a_motion_permit(self):
        a = JointState((0.,), (.1,), (0.,))
        b = JointState((.01,), (.1,), (0.,))
        path = JointTrajectory(('j',), (TrajectoryPoint(0.,a),TrajectoryPoint(.1,b)), 'TEST')
        def collision(state):
            raise MotionError('unpermitted penetration')
        verifier = NS(check_state=collision, maximum_fk_position_error_m=0., maximum_fk_rotation_error_rad=0.)
        result = screen_horizon(path,a,JointLimits((-1.,),(1.,),(1.,),(1.,)),verifier,sample_dt_s=.005)
        self.assertEqual(result['rejection_reasons'], ['unpermitted penetration'])
        self.assertFalse(result['stopping_envelope_validated'])
        self.assertFalse(result['execution_permitted'])

    def test_rejects_a_changed_initial_derivative(self):
        a = JointState((0.,), (.1,), (.2,))
        b = JointState((0.,), (.1,), (0.,))
        path = JointTrajectory(('j',), (TrajectoryPoint(0.,b),TrajectoryPoint(.1,b)), 'TEST')
        with self.assertRaisesRegex(MotionError, 'changed.*q/dq/ddq'):
            screen_horizon(path,a,None,None,sample_dt_s=.005)


class MpcInitializationTests(unittest.IsolatedAsyncioTestCase):
    async def _load(self, *, enabled=False, dt=.02, full=False):
        # Exercise the real adapter config boundary; these are named API doubles,
        # never source/GPU or optimization validation.
        module = ModuleType('curobo.wrap.reacher.mpc')
        seen = {}
        class Config:
            @staticmethod
            def load_from_robot_config(*args, **kwargs):
                seen['override'] = json.loads(Path(kwargs['override_particle_file']).read_text())
                return None
        class Solver:
            def __init__(self, config):
                self.joint_names = ['j']
                self.solver = NS(safety_rollout=NS(dynamics_model=NS(state_filter=NS(enable=enabled, dt=dt),
                                                                   return_full_act_buffer=full)))
        module.MpcSolverConfig, module.MpcSolver = Config, Solver
        base = ModuleType('curobo.types.base')
        base.TensorDeviceType = lambda: None
        with patch.dict(sys.modules, {'curobo':NS(__version__='0.7.8'),
                'torch':NS(cuda=NS(is_available=lambda:True)),
                'curobo.types.base':base,'curobo.wrap.reacher.mpc':module}):
            result = await CuroboMpcAdapter.load(robot_config={},world_config={},step_dt_s=.02,
                                                maximum_solve_s=1.,horizon_steps=4,full_horizon=full)
        return result, seen

    async def test_local_config_disables_hidden_state_smoothing_without_admission(self):
        adapter, seen = await self._load()
        self.assertIs(seen['override']['model']['state_filter_cfg']['enable'],False)
        self.assertFalse(adapter.selfcheck_passed)
        self.assertFalse(adapter.capabilities)

    async def test_rejects_upstream_filter_or_command_dt_changes(self):
        for kwargs in ({'enabled':True},{'dt':.01}):
            with self.subTest(**kwargs), self.assertRaises(CuroboUnavailable):
                await self._load(**kwargs)

    async def test_full_horizon_has_explicit_constant_timing_and_raw_boundary_state(self):
        adapter,seen = await self._load(full=True)
        options = seen['override']['model']
        self.assertEqual(options['horizon'],5)
        self.assertTrue(options['return_full_act_buffer'])
        self.assertEqual(options['dt_traj_params'],{'base_dt':.02,'base_ratio':1.,'max_dt':.02})
        self.assertTrue(adapter.full_horizon)
        self.assertFalse(adapter.selfcheck_passed)

    async def test_kernel_allocation_bound_is_checked_before_cuda_import(self):
        with self.assertRaisesRegex(MotionError,'fewer than 100'):
            await CuroboMpcAdapter.load(robot_config={},world_config={},step_dt_s=.02,
                maximum_solve_s=1.,horizon_steps=99,full_horizon=True)


class ReactivePhysicsPortTests(unittest.TestCase):
    def test_host_scheduling_gap_keeps_exact_curve_sampling_each_physics_tick(self):
        simulation = MujocoReplay(Path(__file__).resolve().parents[1]/'simulation/rolling_scene.xml')
        start = simulation.home()
        goal = list(start.position)
        goal[0] += .1
        # Explicit joint fixture tests receiver sampling, not arm planning.
        path = fixture_joint_trajectory(start,goal,duration_s=1.)
        controller = NS(start_at=0.,active=NS(trajectory=path),generation=0)
        io = _PhysicsIO(simulation,controller,None,{},asyncio.Event())
        io.command(start,0.)
        io._advance(.04)
        actual_time = float(simulation.data.time)
        self.assertAlmostEqual(float(simulation.data.ctrl[0]),path.sample(actual_time).position[0],places=10)
        self.assertGreater(float(simulation.data.ctrl[0]),start.position[0])
        fixed = simulation.data.ctrl.copy()
        before_q = simulation.data.qpos.copy()
        io.supervisor_stop('TEST',.04)
        io._advance(.08)
        self.assertEqual(list(simulation.data.ctrl),list(fixed))
        self.assertNotEqual(list(simulation.data.qpos),list(before_q))


class FullHorizonBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def _candidate(self, *, corrupt=False):
        class Tensor:
            def __init__(self,value): self.value=np.asarray(value,dtype=np.float32)
            @property
            def shape(self): return self.value.shape
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.value.tolist()
            def __getitem__(self,index): return Tensor(self.value[index])
        initial = JointState((.123456789,),(.0123456789,),(.023456789,))
        action = NS(position=Tensor([[[initial.position[0]+(.001 if corrupt else 0.)],[.124],[.125]]]),
                    velocity=Tensor([[[initial.velocity[0]],[.02],[.03]]]),
                    acceleration=Tensor([[[initial.acceleration[0]],[.02],[.03]]]))
        action.get_ordered_joint_state=lambda names: action
        solver=NS(solver=NS(safety_rollout=NS(dynamics_model=NS(traj_dt=Tensor([.02]*3)))),
                  update_world=lambda world:None,setup_solve_single=lambda goal,seeds:goal,
                  step=lambda *args,**kwargs:NS(action=action,metrics=NS(feasible=NS(all=lambda:NS(item=lambda:True)))))
        adapter=CuroboMpcAdapter(solver,None,step_dt_s=.02,maximum_solve_s=1.,horizon_steps=2,
                                 joint_names=['j'],full_horizon=True)
        adapter.selfcheck_passed=True
        adapter._state=lambda state:NS(clone=lambda:None)
        adapter._pose=lambda pose:None
        with patch.dict(sys.modules,{'curobo.rollout.rollout_base':NS(Goal=NS),
                'curobo.geom.types':NS(WorldConfig=NS(from_dict=lambda value:value))}):
            path=await adapter.candidate(moving_boundary=initial,goal_pose=((0.,0.,0.),(0.,0.,0.,1.)),
                                         world_config={},world_identity='TEST')
        return initial,path

    async def test_preserves_gpu_float32_boundary_without_derivative_patching(self):
        initial,path=await self._candidate()
        self.assertEqual(path.points[0].state.position[0],float(np.float32(initial.position[0])))
        self.assertNotEqual(path.points[0].state.position[0],initial.position[0])
        self.assertEqual(path.points[0].state.acceleration[0],float(np.float32(initial.acceleration[0])))
        self.assertEqual(path.duration_s,.04)

    async def test_rejects_a_changed_gpu_start_even_when_metrics_are_feasible(self):
        with self.assertRaisesRegex(MotionError,'changed the supplied moving'):
            await self._candidate(corrupt=True)


if __name__ == '__main__':
    unittest.main()
