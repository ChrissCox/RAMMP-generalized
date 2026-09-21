"""Actual cuRobo rolling-session integration with explicit MuJoCo-only I/O.

The controller may activate screened moving suffixes, then an intentional
cancel exercises the missing-stop-path escalation. MuJoCo freezes its actuator
setpoint and continues dynamics until measured settling. This is not a generated
stop path, commissioned stop behavior, physical transport or task success.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import time
import threading
from types import SimpleNamespace

import numpy as np

from ..resources import ResourceManager
from .guards import TargetObservation, TargetUpdateGuard
from .integration import MujocoCandidateVerifier
from .leasing import PlannerLeaseBroker
from .rolling import (BoundaryTolerance, JointState, JointTrajectory, MotionError, MotionIdentity,
                      RollingController, TrajectoryPoint, TrajectoryValidator)
from .session import RollingMotionSession, SessionWorld


class _PhysicsIO:
    hardware_commands = False

    def __init__(self, simulation, controller, world, goal, cancel):
        self.sim, self.controller, self.world = simulation, controller, world
        self.goal, self.cancel = goal, cancel
        self.origin, self.physics_origin = controller.start_at, float(simulation.data.time)
        self.commands, self.samples, self.stop_events = [], [], []
        self.activation_at = None
        self.maximum_error = self.maximum_contacts = 0
        self.stop_at = None
        self.settled_since = None
        self.path = self.path_start = None
        self.maximum_boundary_error = [0.,0.,0.]
        self.resources = ResourceManager({'ARM','PLANNER'})
        self.owner = 'gpu-mujoco-probe/move/1/1'
        if not self.resources.try_acquire(self.owner,{'ARM','PLANNER'}):
            raise MotionError('Simulation resource reservation unavailable')

    def owns(self, identity, resources):
        self.resources.assert_owned(self.owner,resources)
        return identity == self.controller.identity and resources == frozenset({'ARM','PLANNER'})

    def _advance(self, now):
        sim, dt = self.sim, float(self.sim.model.opt.timestep)
        if now-self.origin-(float(sim.data.time)-self.physics_origin) > .5:
            raise MotionError('MuJoCo reactive rehearsal missed its stepping budget')
        while float(sim.data.time)-self.physics_origin+dt <= now-self.origin:
            if self.path is not None and self.stop_at is None:
                command_time = self.origin+float(sim.data.time)-self.physics_origin+dt-self.path_start
                sim.data.ctrl[:] = self.path.sample(max(0.,min(command_time,self.path.duration_s))).position
            sim.mujoco.mj_step(sim.model, sim.data)
            if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
                raise MotionError('Nonfinite measured MuJoCo state')
            self.maximum_contacts = max(self.maximum_contacts, int(sim.data.ncon))
            if self.stop_at is not None:
                measured_at = self.origin+float(sim.data.time)-self.physics_origin
                if np.max(np.abs(sim.data.qvel)) < .003:
                    if self.settled_since is None:
                        self.settled_since = measured_at
                else:
                    self.settled_since = None

    def read_state(self, now):
        self._advance(now)
        sim = self.sim
        state = JointState(tuple(map(float,sim.data.qpos)),tuple(map(float,sim.data.qvel)),tuple(map(float,sim.data.qacc)))
        elapsed = now-self.controller.start_at
        if self.stop_at is None and 0 <= elapsed <= self.controller.active.trajectory.duration_s:
            expected = self.controller.expected(now)
            for i,field in enumerate(('position','velocity','acceleration')):
                self.maximum_boundary_error[i] = max(self.maximum_boundary_error[i],
                    max(abs(a-b) for a,b in zip(getattr(expected,field),getattr(state,field))))
        self.samples.append({'time_s':now-self.origin, 'q':list(state.position),'dq':list(state.velocity),
                             'generation':self.controller.generation})
        if self.controller.generation and self.activation_at is None:
            self.activation_at = now
        if self.activation_at is not None and now-self.activation_at >= .15:
            self.cancel.set()
        return state

    def read_target(self, now):
        shift = .005 if now-self.origin >= .015 else 0.
        return TargetObservation('simulated_target','approach',tuple(v+(shift if i==0 else 0.)
            for i,v in enumerate(self.goal['position_m'])),now,.001,True,tuple(self.goal['quaternion_xyzw']))

    def read_world(self, now):
        return self.world

    def continuation_valid(self, now, state, world):
        # Explicit simulation cell screen, not a physical stopping certificate.
        return self.maximum_contacts == 0 and world == self.world

    def command(self, state, now):
        # As in the existing MuJoCo driver port, interpolate the admitted curve
        # at every physics tick even when Python wakes later. Holding a 2 ms
        # point across host scheduling gaps would manufacture actuator jitter.
        self.path, self.path_start = self.controller.active.trajectory,self.controller.start_at
        # _advance applies it at the next exact simulation tick. Applying the
        # wall-clock point here can command a future point, then step backwards
        # on the next physics tick when simulation time is slightly behind.
        self.maximum_error = max(self.maximum_error,float(np.max(np.abs(self.sim.data.qpos-state.position))))
        self.commands.append({'time_s':now-self.origin,'generation':self.controller.generation})

    def supervisor_stop(self, reason, now):
        self.stop_events.append({'time_s':now-self.origin, 'reason':reason,
                                'method':'MuJoCo actuator setpoint freeze; no generated stop curve',
                                'resource_owners':self.resources.owners})
        self.stop_at = now
        self.settled_since = None
        # Keep the last actual setpoint; never overwrite simulated q or dq.

    def quiescent(self):
        now = time.monotonic()
        self._advance(now)
        # Solver drain can outlast a tracking tick. Elapsed wall time never
        # turns the last pre-stop state into a fresh measured stationary dwell.
        measured_at = self.origin+float(self.sim.data.time)-self.physics_origin
        return bool(self.stop_at is not None and self.settled_since is not None
                    and measured_at-self.settled_since >= .08
                    and 0 <= now-measured_at <= 2*float(self.sim.model.opt.timestep))

    def goal_satisfied(self, target, state, world):
        return False  # This bounded integration cancels; it is not an ADL task.


class _ProbePlanner:
    hardware_commands = False

    def __init__(self, adapter, *, cancel_during_solve=None):
        self.adapter, self.calls = adapter, []
        self.cancel_during_solve = cancel_during_solve
        self.solver_calls, self.cancel_at = [], None

    async def candidate(self, **request):
        began = time.monotonic()
        observer = None
        original_step = None
        if self.cancel_during_solve is not None:
            entered = threading.Event()
            original_step = self.adapter.solver.step
            def measured_step(*args, **kwargs):
                record = {'started':time.monotonic()}
                self.solver_calls.append(record)
                entered.set()
                try:
                    return original_step(*args, **kwargs)
                finally:
                    record['finished'] = time.monotonic()
            self.adapter.solver.step = measured_step
            async def cancel_after_solver_entry():
                deadline = time.monotonic()+.30
                while not entered.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(.0005)
                # Never add a fake solver delay just to manufacture overlap.
                if entered.is_set() and 'finished' not in self.solver_calls[-1]:
                    self.cancel_at = time.monotonic()
                    self.cancel_during_solve.set()
            observer = asyncio.create_task(cancel_after_solver_entry())
        try:
            return await self.adapter.candidate(**request, _selfcheck=True)
        finally:
            if observer is not None:
                observer.cancel()
                try:
                    await observer
                except asyncio.CancelledError:
                    pass
                self.adapter.solver.step = original_step
            self.calls.append((began,time.monotonic()))


async def run_reactive_rehearsal(*, simulation, reference, adapter, planner_metadata,
                               contract, limits, world, goal, cancel_during_gpu=False):
    if not adapter.full_horizon:
        raise MotionError('Reactive rehearsal requires the bounded full-horizon GPU adapter')
    # Use the pinned wrapper's established static-start time dilation. No
    # moving MPC suffix is retimed and its boundary derivatives stay intact.
    from rammp_curobo.retime import scale_trajectory
    from rammp_curobo.types import Trajectory
    if any(abs(v)>1e-8 for v in (*reference.points[0].state.velocity,*reference.points[0].state.acceleration)):
        raise MotionError('Only the stationary-start prefix may use execution speed scaling')
    dt = reference.points[1].time_s
    if any(abs(p.time_s-i*dt)>1e-7 for i,p in enumerate(reference.points)):
        raise MotionError('Pinned static prefix time dilation requires uniform source knots')
    scaled = scale_trajectory(Trajectory(joint_names=list(reference.joint_names),
        positions=np.asarray([p.state.position for p in reference.points]),
        velocities=np.asarray([p.state.velocity for p in reference.points]),
        accelerations=np.asarray([p.state.acceleration for p in reference.points]),dt=dt),.25)
    reference = JointTrajectory(reference.joint_names,tuple(TrajectoryPoint(i*scaled.dt,
        JointState(tuple(q),tuple(v),tuple(a))) for i,(q,v,a) in
        enumerate(zip(scaled.positions,scaled.velocities,scaled.accelerations))),
        reference.provenance+':upstream_static_start_time_dilation:0.25')
    simulation.home()
    simulation.data.qpos[:] = reference.points[0].state.position
    simulation.data.qvel[:] = reference.points[0].state.velocity
    simulation.data.ctrl[:] = reference.points[0].state.position
    for _ in range(500):
        simulation.mujoco.mj_step(simulation.model, simulation.data)
    def geometry(path, dependencies):
        # One batched cuRobo FK call removes hundreds of Python/GPU round trips.
        # MuJoCo still checks every original validation sample independently.
        from curobo.types.robot import JointState as CuJointState
        step_count = max(1,int(np.ceil(path.duration_s/contract['validation_sample_dt_s'])))
        states = [path.sample(path.duration_s*i/step_count) for i in range(step_count+1)]
        gpu_state = CuJointState(position=adapter.tensor_args.to_device([list(s.position) for s in states]),
                                joint_names=list(adapter.joint_names))
        pose = adapter.solver.compute_kinematics(gpu_state).ee_pose
        positions = pose.position.detach().cpu().tolist()
        quaternions = pose.quaternion.detach().cpu().tolist()
        cache = {s.position:(p,[*q[1:],q[0]]) for s,p,q in zip(states,positions,quaternions)}
        metadata = SimpleNamespace(joint_names=planner_metadata.joint_names,_robot_cfg=planner_metadata._robot_cfg,
                                   fk=lambda q,quat_order:cache[tuple(q)])
        verifier = MujocoCandidateVerifier(simulation,metadata,contract)
        for state in states:
            verifier.check_state(state)
        return dependencies == world.dependencies
    validator = TrajectoryValidator(limits,geometry,sample_dt_s=contract['validation_sample_dt_s'])
    initial = validator.validate(reference,world.dependencies,now=time.monotonic(),expires_at=time.monotonic()+20.)
    def missing_stop(now, measured, dependencies):
        raise MotionError('No cuRobo-generated stationary stop path is admitted; supervisor escalation required')
    started = time.monotonic()
    controller = RollingController(MotionIdentity('gpu-mujoco-probe','move',1,1),initial,validator,
        start_at=started,tolerance=BoundaryTolerance(.015,.15,5.),stop_provider=missing_stop,
        stop_budget_s=.06,switch_lead_s=.015,activation_lateness_s=.03)
    cancel = asyncio.Event()
    io = _PhysicsIO(simulation,controller,world,goal,cancel)
    original = io.read_target(started)
    guard = TargetUpdateGuard(original,max_correction_m=.01,max_uncertainty_m=.003,
                              max_age_s=.1,meaningful_change_m=.002)
    planner = _ProbePlanner(adapter,cancel_during_solve=cancel if cancel_during_gpu else None)
    broker = PlannerLeaseBroker(maximum_call_s=.30)
    session = RollingMotionSession(controller=controller,planner=planner,broker=broker,guard=guard,io=io,
        required_resources={'ARM','PLANNER'},planning_lead_s=.40,tick_period_s=.002,
        maximum_tick_gap_s=.15,maximum_duration_s=3.,evidence_lifetime_s=10.)
    result = await session.run(cancel)
    # Continue actual physics after supervisor setpoint freeze; no fabricated
    # stationary feedback and no promotion of the session's earlier result.
    deadline = time.monotonic()+2.
    while not io.quiescent() and time.monotonic() < deadline:
        io.read_state(time.monotonic())
        await asyncio.sleep(.002)
    retained_until_settled = io.resources.owners == {'ARM':io.owner,'PLANNER':io.owner}
    if session.quiescent() and broker.owner is None:
        io.resources.release(io.owner)
    commands_during_planning = sum(any(a<=started+c['time_s']<=b for a,b in planner.calls) for c in io.commands)
    cancelled_inside_solver = (planner.cancel_at is not None and any(
        call['started'] <= planner.cancel_at < call.get('finished',float('-inf')) for call in planner.solver_calls))
    return {'mode':'actual_curobo_mpc_rolling_session_mujoco','simulation_only':True,
        'requested_cancel_during_gpu':cancel_during_gpu,'cancel_observed_during_solver_call':cancelled_inside_solver,
        'cancel_at_s':None if planner.cancel_at is None else planner.cancel_at-started,
        'solver_calls':[{'started_s':call['started']-started,'finished_s':call.get('finished',started)-started}
                        for call in planner.solver_calls],
        'hardware_commands':False,'hardware_validated':False,'session_result':asdict(result),
        'moving_suffix_activated':controller.generation>0,'generation':controller.generation,
        'commands_during_gpu_planning':commands_during_planning,'measured_settled_after_supervisor':io.quiescent(),
        'claims_retained_until_settled':retained_until_settled,'claims_released_after_quiescence':not io.resources.owners,
        'planner_drained_before_release':broker.owner is None,
        'maximum_tracking_error_rad':io.maximum_error,'maximum_contact_count':io.maximum_contacts,
        'maximum_boundary_error':dict(zip(('position_rad','velocity_rad_s','acceleration_rad_s2'),io.maximum_boundary_error)),
        'planning_calls':[{'started_s':a-started,'finished_s':b-started,'elapsed_s':b-a} for a,b in planner.calls],
        'session_events':session.events,'controller_events':controller.events,'stop_events':io.stop_events,
        'controller_stop_reason':controller.stop_reason,'initial_static_speed_scale':.25,
        'samples':io.samples,'simulation_boundary_tolerances':{'position_rad':.015,'velocity_rad_s':.15,'acceleration_rad_s2':5.},
        'stop_path_validated':False,'task_goal_validated':False,
        'limitations':['Sampled bare-arm geometry and dynamics; no full assembly/contact/stopping proof',
            'Intentional cancel escalates missing generated stop to simulated actuator hold',
            'No physical transport, no physical motion profile, no ADL task completion']}
