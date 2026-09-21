"""Isolated actual cuRobo MPC moving-boundary probe; never a robot executor.

A saved source-pinned static cuRobo candidate supplies the synthetic moving
switch state. MPC alone generates each new joint horizon. Independent MuJoCo
FK, contact and sampled exact-interpolation checks record acceptance/rejection;
neither result is a rolling-motion permit or a measured stopping guarantee.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import uuid

import numpy as np

from ..contracts import Catalog, digest, strict_loads
from ..simulation import MujocoReplay
from .curobo import CUROBO_VERSION, RAMMP_COMMIT, MPC_EXACT_BOUNDARY_OVERRIDE, CuroboMpcAdapter
from .curobo_process import REVIEWED_GPU_IMAGE, IMAGE_ASSET_ROOT
from .curobo_worker import parse_request, serialize_trajectory
from .integration import (MujocoCandidateVerifier, collision_world_from_mujoco,
                          file_digest, load_contract, model_assets_digest)
from .rehearsal import checked_simulation_candidate
from .rolling import BoundaryTolerance, JointLimits, MotionError
from .gpu_lease import GpuLeaseJournal


_UNRESOLVED_LEASES = []


def optimizer_robot_config(robot, acceleration_limits, scale):
    """Tighten only source-supported cuRobo optimizer bounds, never validation."""
    if scale not in (1., .2, .05):
        raise MotionError('Probe allows only its three bounded local optimization profiles')
    result = copy.deepcopy(robot)
    cspace = result['kinematics']['cspace']
    prior = cspace['max_acceleration']
    if (type(prior) not in (int, float) or not math.isfinite(prior) or prior <= 0
            or len(acceleration_limits) != 7 or any(type(v) not in (int, float)
                or not math.isfinite(v) or v <= 0 for v in acceleration_limits)):
        raise MotionError('Probe requires explicit finite cuRobo and independent acceleration bounds')
    cspace['max_acceleration'] = min(prior, min(acceleration_limits)*scale)
    return result


def moving_boundary(path):
    """Select an original interior state with both nonzero velocity and acceleration."""
    states = [p.state for p in path.points[1:-1]
              if max(map(abs, p.state.velocity)) > 1e-3
              and max(map(abs, p.state.acceleration)) > 1e-3]
    if not states:
        raise MotionError("Reference path has no moving q/dq/ddq switch state")
    return states[len(states)//3]


def screen_horizon(path, boundary, limits, verifier, *, sample_dt_s):
    """Diagnostic sampled checks only; deliberately returns no motion certificate."""
    if not BoundaryTolerance(1e-6,1e-6,1e-6).matches(path.points[0].state, boundary):
        raise MotionError("MPC horizon changed its requested initial q/dq/ddq")
    if not 0 < sample_dt_s <= .02 or not 0 < path.duration_s <= 12.8:
        raise MotionError("Reactive screening work exceeds the explicit bound")
    errors, maximum_velocity, maximum_acceleration = set(), 0., 0.
    count = max(1, math.ceil(path.duration_s/sample_dt_s))
    for i in range(count+1):
        state = path.sample(path.duration_s*i/count)
        maximum_velocity = max(maximum_velocity, *map(abs, state.velocity))
        maximum_acceleration = max(maximum_acceleration, *map(abs, state.acceleration))
        for check in (limits.check, verifier.check_state):
            try:
                check(state)
            except MotionError as exc:
                errors.add(str(exc))
    return {"sampled_checks_passed": not errors, "rejection_reasons": sorted(errors),
            "states_sampled": count+1, "sample_dt_s": path.duration_s/count,
            "maximum_velocity_rad_s": maximum_velocity,
            "maximum_acceleration_rad_s2": maximum_acceleration,
            "maximum_knot_acceleration_rad_s2": max(abs(v) for p in path.points for v in p.state.acceleration),
            "maximum_fk_position_error_m": verifier.maximum_fk_position_error_m,
            "maximum_fk_rotation_error_rad": verifier.maximum_fk_rotation_error_rad,
            "stopping_envelope_validated": False, "execution_permitted": False}


async def worker(args):
    """Runs inside the pinned image with no network or hardware devices."""
    import curobo
    import torch
    root = Catalog().root
    output = Path(args.output)
    response = strict_loads(Path(args.candidate).read_bytes(), max_bytes=64*1024*1024)
    request = strict_loads(Path(args.request).read_bytes())
    if isinstance(request, dict) and set(request) == {'request_id', 'request'}:
        request = request['request']
    parse_request(request)
    contract = load_contract(args.contract)
    reference = checked_simulation_candidate(response, request, contract)
    if model_assets_digest(root/'simulation') != contract['mujoco_assets_digest']:
        raise MotionError('MuJoCo assets changed from the pinned reference contract')
    robot = response['planner_robot_config']
    for field in ('urdf_path', 'asset_root_path'):
        path = Path(robot['kinematics'][field])
        if not path.is_absolute() or not path.resolve().is_relative_to(IMAGE_ASSET_ROOT):
            raise MotionError('This bare-arm probe requires immutable image robot assets')
    if file_digest(robot['kinematics']['urdf_path']) != contract['planner_urdf_digest']:
        raise MotionError('Probe URDF differs from the static model contract')
    simulation = MujocoReplay(root/'simulation/rolling_scene.xml')
    world = collision_world_from_mujoco(simulation, contract)
    if digest(world) != request['world_identity']:
        raise MotionError('Reference world differs from the independent MuJoCo world')
    mpc_world = {'cuboid': {o['name']: {'dims': o['dims'], 'pose': [*o['position'], 1., 0., 0., 0.]}
                           for o in world}}
    boundary = moving_boundary(reference)
    bounds = response['joint_limits']
    limits = JointLimits(tuple(bounds['position'][0]), tuple(bounds['position'][1]),
        tuple(np.minimum(bounds['velocity'], contract['joint_velocity_rad_s'])),
        tuple(contract['joint_acceleration_rad_s2']))
    records, profiles = [], []
    # Regeneration uses at most two source-supported optimizer configurations.
    # Lowering acceleration here leaves independent limits, start derivatives,
    # command dt and returned trajectory samples untouched.
    for profile_name, scale, full in (('baseline', 1., False), ('interpolation_margin', .2, False),
                                    ('full_horizon_interpolation_margin', .2, True)):
        effective_robot = optimizer_robot_config(robot, contract['joint_acceleration_rad_s2'], scale)
        effective_contract = dict(contract, planner_robot_config_digest=digest(effective_robot))
        initialized = time.monotonic()
        adapter = await CuroboMpcAdapter.load(robot_config=effective_robot, world_config=mpc_world,
            step_dt_s=.02, maximum_solve_s=60., horizon_steps=64 if full else 16, full_horizon=full)
        profiles.append({'name':profile_name, 'initialization_wall_s':time.monotonic()-initialized,
            'acceleration_scale':scale, 'robot_config_digest':digest(effective_robot),
            'optimizer_acceleration_rad_s2':effective_robot['kinematics']['cspace']['max_acceleration']})

        def fk(q, quat_order):
            if quat_order != 'xyzw':
                raise MotionError('Probe FK order mismatch')
            state = adapter._state(type(boundary)(tuple(q), (0.,)*7, (0.,)*7))
            result = adapter.solver.compute_kinematics(state).ee_pose
            pos = result.position.detach().cpu().reshape(-1).tolist()
            wxyz = result.quaternion.detach().cpu().reshape(-1).tolist()
            return pos, [*wxyz[1:], wxyz[0]]

        planner = SimpleNamespace(joint_names=adapter.joint_names, _robot_cfg=effective_robot, fk=fk)
        # These are local simulation cases, not camera observations or task policy.
        for name, shift in (('initial_moving_boundary', 0.), ('same_target_5mm_correction', .005),
                            ('reused_solver_original_target', 0.)):
            goal = (tuple(request['goal']['position_m'][i]+(shift if i == 0 else 0.) for i in range(3)),
                    tuple(request['goal']['quaternion_xyzw']))
            began = time.monotonic()
            record = {'case':name, 'optimizer_profile':profile_name, 'target_shift_m':shift, 'goal_pose':goal}
            try:
                # Probe-only bypass obtains unadmitted output for independent screening.
                # selfcheck_passed remains False because stopping has not been proven.
                path = await adapter.candidate(moving_boundary=boundary, goal_pose=goal,
                    world_config=mpc_world, world_identity=digest(mpc_world), _selfcheck=True)
                record['planning_wall_s'] = time.monotonic()-began
                verifier = MujocoCandidateVerifier(simulation, planner, effective_contract)
                record.update(screen_horizon(path, boundary, limits, verifier,
                    sample_dt_s=contract['validation_sample_dt_s']))
                filename = profile_name+'-'+name+'-candidate.json'
                (output/filename).write_text(json.dumps(serialize_trajectory(path), indent=2)+'\n')
                record.update(trajectory_digest=path.digest, candidate_file=filename,
                              point_count=len(path.points), horizon_s=path.duration_s)
            except Exception as exc:
                record.update({'planning_wall_s': time.monotonic()-began,
                    'sampled_checks_passed': False, 'rejection_reasons': [type(exc).__name__+': '+str(exc)],
                    'execution_permitted': False})
            records.append(record)
    from .reactive_rehearsal import run_reactive_rehearsal
    from .session import SessionWorld
    # A further conservative local optimizer bound addresses measured MuJoCo
    # acceleration tracking during suffix activation. Independent path and
    # activation limits stay unchanged; moving suffixes are never retimed.
    effective_robot = optimizer_robot_config(robot,contract['joint_acceleration_rad_s2'],.05)
    effective_contract = dict(contract,planner_robot_config_digest=digest(effective_robot))
    initialized = time.monotonic()
    adapter = await CuroboMpcAdapter.load(robot_config=effective_robot,world_config=mpc_world,
        step_dt_s=.02,maximum_solve_s=60.,horizon_steps=64,full_horizon=True)
    profiles.append({'name':'full_horizon_tracking_margin','acceleration_scale':.05,
        'initialization_wall_s':time.monotonic()-initialized,'robot_config_digest':digest(effective_robot),
        'optimizer_acceleration_rad_s2':effective_robot['kinematics']['cspace']['max_acceleration']})
    planner = SimpleNamespace(joint_names=adapter.joint_names,_robot_cfg=effective_robot,fk=fk)
    for _ in range(2):
        await adapter.candidate(moving_boundary=reference.points[0].state,
            goal_pose=(tuple(request['goal']['position_m']),tuple(request['goal']['quaternion_xyzw'])),
            world_config=mpc_world,world_identity=digest(mpc_world),_selfcheck=True)
    rehearsal = await run_reactive_rehearsal(simulation=simulation,reference=reference,adapter=adapter,
        planner_metadata=planner,contract=effective_contract,limits=limits,
        world=SessionWorld({'world':digest(mpc_world),'model':contract['mujoco_assets_digest']},mpc_world,digest(mpc_world)),
        goal=request['goal'])
    (output/'rolling-rehearsal.json').write_text(json.dumps(rehearsal,indent=2,allow_nan=False)+'\n')
    cancelled = await run_reactive_rehearsal(simulation=simulation,reference=reference,adapter=adapter,
        planner_metadata=planner,contract=effective_contract,limits=limits,
        world=SessionWorld({'world':digest(mpc_world),'model':contract['mujoco_assets_digest']},mpc_world,digest(mpc_world)),
        goal=request['goal'],cancel_during_gpu=True)
    (output/'rolling-cancel-during-gpu.json').write_text(json.dumps(cancelled,indent=2,allow_nan=False)+'\n')
    package = Path(curobo.__file__).resolve().parent
    source_files = ('wrap/reacher/mpc.py', 'wrap/wrap_mpc.py', 'util/state_filter.py',
                    'content/configs/task/particle_mpc.yml')
    report = {'mode': 'isolated_curobo_mpc_moving_boundary_probe', 'simulation_only': True,
        'hardware_commands': False, 'hardware_validated': False, 'execution_permitted': False,
        'curobo_version': curobo.__version__, 'gpu_available': torch.cuda.is_available(),
        'source_files': {n: file_digest(package/n) for n in source_files},
        'adapter_digest': file_digest(Path(__file__).with_name('curobo.py')),
        'reference_candidate_digest': reference.digest, 'contract_id': contract['contract_id'],
        'optimizer_profiles':profiles, 'exact_boundary_override': MPC_EXACT_BOUNDARY_OVERRIDE,
        'moving_boundary': {f: list(getattr(boundary, f)) for f in ('position', 'velocity', 'acceleration')},
        'selfcheck_passed': adapter.selfcheck_passed, 'cases': records,
        'limitations': ['Bare Gen3 model; no gripper, wrist camera, payload or ADL contact physics',
            'Sampled interpolation and geometry checks are not swept/stopping proof',
            'MPC optimizer costs do not establish a constrained-motion contract',
            'No physical feedback, driver rolling handoff, stopping profile or execution permit']}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'cases': records, 'hardware_commands': False}))


def run_isolated(args):
    root = Catalog().root
    output, cache = Path(args.output).resolve(), Path(args.gpu_cache).resolve()
    raw_inputs = {n: Path(getattr(args,n)) for n in ('candidate','request','contract')}
    if any(not p.is_file() or p.is_symlink() for p in raw_inputs.values()):
        raise MotionError('Probe inputs must be existing local regular files')
    inputs = {n:p.resolve() for n,p in raw_inputs.items()}
    wrapper = Path(args.wrapper_checkout).resolve()
    revision = subprocess.run(['git','-C',str(wrapper),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    dirty = subprocess.run(['git','-C',str(wrapper),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if revision != RAMMP_COMMIT or dirty:
        raise MotionError('Reactive rehearsal requires the clean pinned wrapper for static-start speed scaling')
    readonly = [*inputs.values(), wrapper, *(root/n for n in ('rammp_adl','skills','schemas','config','simulation','tools'))]
    if any(cache.is_relative_to(p) or p.is_relative_to(cache) or output.is_relative_to(p)
           for p in readonly) or output.is_relative_to(cache) or cache.is_relative_to(output):
        raise MotionError('Writable probe output/cache must not alias read-only source inputs')
    if any(':' in str(p) for p in [output,cache,*readonly]):
        raise MotionError('Probe bind paths may not contain colons')
    output.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    lease = (cache/'rammp-commissioning-planner.lock').open('a')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lease.close()
        raise
    name = 'rammp-reactive-probe-'+uuid.uuid4().hex[:12]
    journal = GpuLeaseJournal(cache)
    created = False
    try:
        journal.reconcile()
        info = json.loads(subprocess.run(['docker','image','inspect',REVIEWED_GPU_IMAGE],
            capture_output=True, text=True, check=True, timeout=15).stdout)[0]
        if info['Id'] != REVIEWED_GPU_IMAGE or info['Config']['Entrypoint'] != ['python3'] or info['Config'].get('Volumes'):
            raise MotionError('GPU image identity or startup configuration changed')
        command = ['docker','run','--rm','--pull','never','--name',name,'--network','none',
            '--runtime','nvidia','--read-only','--tmpfs','/tmp:rw,size=2g','--entrypoint','python3',
            '--user',str(os.getuid())+':'+str(os.getgid()),'-e','PYTHONDONTWRITEBYTECODE=1',
            '-e','XDG_CACHE_HOME=/gpu-cache','-e','TORCH_EXTENSIONS_DIR=/gpu-cache/torch_extensions',
            '-e','PYTHONPATH=/workspace:/opt/pinned-wrapper/core']
        bindings = [(root/n,'/workspace/'+n,'ro') for n in ('rammp_adl','skills','schemas','config','simulation')]
        bindings += [(root/'tools/check_design.py','/workspace/tools/check_design.py','ro'),
                     (output,'/out','rw'),(cache,'/gpu-cache','rw'),(wrapper,'/opt/pinned-wrapper','ro')]
        bindings += [(p,'/input/'+n+'.json','ro') for n,p in inputs.items()]
        for source,target,mode in bindings:
            command += ['-v',str(source)+':'+target+':'+mode]
        command += [REVIEWED_GPU_IMAGE,'-m','rammp_adl.motion.reactive_probe','--worker',
                    '--candidate','/input/candidate.json','--request','/input/request.json',
                    '--contract','/input/contract.json','--output','/out']
        inventory = {str(p):file_digest(p) for p in inputs.values()}
        (output/'invocation.json').write_text(json.dumps({'command':command, 'image':REVIEWED_GPU_IMAGE,
            'input_digests':inventory, 'hardware_commands':False}, indent=2)+'\n')
        journal.reserve(name)
        created = True
        with (output/'worker.log').open('w') as log:
            # An orphaned Docker CLI must retain exclusion while its create/run
            # request is still outstanding after this Python client exits.
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240,
                                       pass_fds=(lease.fileno(),))
        if inventory != {str(p):file_digest(p) for p in inputs.values()}:
            raise MotionError('Probe inputs changed during GPU execution')
        if completed.returncode:
            raise MotionError('Reactive GPU worker failed; inspect its local worker.log')
    finally:
        # CLI exit alone does not prove a GPU container stopped. Keep the cache
        # lease if Docker cannot positively establish absence of this exact name.
        quiescent = not created
        if created:
            try:
                subprocess.run(['docker','rm','-f',name], capture_output=True, timeout=15)
                found = subprocess.run(['docker','container','ls','-a','--filter','name=^/'+name+'$',
                    '--format','{{.Names}}'], capture_output=True, text=True, check=True, timeout=15)
                quiescent = not found.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                quiescent = False
        if quiescent:
            if created:
                try:
                    journal.release(name)
                except BaseException:
                    _UNRESOLVED_LEASES.append(lease)
                    raise
            lease.close()
        else:
            _UNRESOLVED_LEASES.append(lease)
            raise MotionError('GPU container teardown could not be verified; planner cache lease retained')
    report = strict_loads((output/'report.json').read_bytes())
    print(json.dumps({'report':str(output/'report.json'), 'hardware_commands':False,
        'sampled_checks_passed':[c['sampled_checks_passed'] for c in report['cases']]}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','request','contract','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-cache', type=Path)
    parser.add_argument('--wrapper-checkout', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        asyncio.run(worker(args))
    else:
        if args.gpu_cache is None or args.wrapper_checkout is None:
            parser.error('--gpu-cache and --wrapper-checkout are required outside the GPU worker')
        run_isolated(args)


if __name__ == '__main__':
    main()
