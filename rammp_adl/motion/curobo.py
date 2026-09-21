"""Source-pinned optional RAMMP cuRobo planning adapter, without driver I/O.

Only static planning is exposed by the pinned upstream core. Moving boundary,
constrained-path and future suffix capabilities remain unavailable here. The
rolling module is the NEW receiver contract those extensions must satisfy.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import math
from pathlib import Path
import subprocess
import tempfile
import json
import time

from .rolling import BoundaryTolerance, JointState, JointTrajectory, MotionError, TrajectoryPoint
from .leasing import drain_nonpreemptible

RAMMP_COMMIT = "320872b709b276fc7283190d24edef7f8632bec9"
CUROBO_VERSION = "0.7.8"
MPC_EXACT_BOUNDARY_OVERRIDE = {"model": {"state_filter_cfg": {"enable": False}}}
SOURCE_URL = f"https://github.com/rammp-org/RAMMP-CuRobo/blob/{RAMMP_COMMIT}/core/rammp_curobo/planner.py"


class CuroboUnavailable(RuntimeError):
    pass


def availability():
    available = importlib.util.find_spec("rammp_curobo") is not None
    return {"importable": available, "required_source_commit": RAMMP_COMMIT,
            "required_curobo_version": CUROBO_VERSION,
            "hardware_commands": False, "online_replanning": False,
            "detail": "Source/config validation and GPU warmup required" if available else "RAMMP cuRobo is not installed; no alternative arm planner is selected"}


class RammpCuroboAdapter:
    """Atomic install-world-plus-plan using the verified pure Python API.

    Adapters return the exact path for independent validation. They never execute
    it. A static result cannot satisfy online-replanning capability gates.
    """
    capabilities = frozenset({"curobo_static_planning"})

    def __init__(self, planner, *, source_commit, installed_module_path, source_root):
        if source_commit != RAMMP_COMMIT:
            raise CuroboUnavailable("unverified RAMMP cuRobo source revision")
        if not Path(installed_module_path).resolve().is_relative_to(Path(source_root).resolve()):
            raise CuroboUnavailable("imported RAMMP planner is outside the verified checkout")
        self.planner, self._lock = planner, asyncio.Lock()

    @classmethod
    async def load(cls, *, source_root, planner_config):
        root = Path(source_root).resolve()
        try:
            revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise CuroboUnavailable("Could not verify the explicit local RAMMP source checkout") from exc
        if revision != RAMMP_COMMIT or dirty:
            raise CuroboUnavailable("RAMMP source must be the clean pinned checkout")
        try:
            module = importlib.import_module("rammp_curobo.planner")
            import torch
            import curobo
        except ImportError as exc:
            raise CuroboUnavailable(f"cuRobo dependency unavailable: {exc.name}") from exc
        if not torch.cuda.is_available():
            raise CuroboUnavailable("cuRobo requires an available CUDA device")
        version = getattr(curobo, "__version__", None)
        if version != CUROBO_VERSION:
            raise CuroboUnavailable(f"unverified cuRobo version {version!r}; require {CUROBO_VERSION}")
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise CuroboUnavailable("imported planner differs from verified checkout")
        from .curobo_stationary import load_stationary_planner
        planner = await asyncio.to_thread(load_stationary_planner, module, planner_config)
        return cls(planner, source_commit=revision, installed_module_path=module.__file__, source_root=root)

    async def plan_pose(self, *, position_m, quaternion_xyzw, start, world, world_identity):
        if any(abs(v) > 1e-8 for v in (*start.velocity, *start.acceleration)):
            raise CuroboUnavailable("TODO: confirm against planner: moving q/dq/ddq boundary support; pinned API accepts positions only")
        if not isinstance(world_identity, str) or not world_identity or len(position_m) != 3 or len(quaternion_xyzw) != 4 or not all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in (*position_m, *quaternion_xyzw)):
            raise MotionError("invalid grounded pose or world identity")
        if abs(sum(v*v for v in quaternion_xyzw)-1) > 1e-4:
            raise MotionError("pose quaternion is not normalized")
        async with self._lock:
            def solve():
                # Verified source API, not the narrower SetWorld ROS IDL.
                self.planner.update_world(world)
                return self.planner.plan_to_pose(position_m, quaternion_xyzw, list(start.position), quat_order="xyzw", apply_tool_correction=False)
            task = asyncio.create_task(asyncio.to_thread(solve))
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                # No upstream GPU cancellation contract: retain world/solver
                # lock until completion and discard the result.
                await drain_nonpreemptible(task)
                raise
        if not result.success or result.joint_traj is None:
            raise MotionError(f"cuRobo planning failed: {result.status}")
        trajectory = result.joint_traj
        if trajectory.velocities is None or trajectory.accelerations is None:
            raise MotionError("cuRobo output lacks derivative evidence")
        if len({len(trajectory.positions), len(trajectory.velocities), len(trajectory.accelerations)}) != 1:
            raise MotionError("cuRobo trajectory array dimensions disagree")
        extension = getattr(self.planner, "rammp_adl_boundary_extension", None)
        provenance = f"rammp_curobo:{RAMMP_COMMIT}:world:{world_identity}"
        if extension is not None:
            provenance += ":interpolation:" + extension
        converted = JointTrajectory(tuple(trajectory.joint_names), tuple(
            TrajectoryPoint(i*float(trajectory.dt), JointState(tuple(map(float, q)), tuple(map(float, v)), tuple(map(float, a))))
            for i, (q, v, a) in enumerate(zip(trajectory.positions, trajectory.velocities, trajectory.accelerations))
        ), provenance)
        return converted

    async def plan_rolling_suffix(self, **request):
        raise CuroboUnavailable("TODO: confirm against planner: pinned moving-state MPC/horizon adapter, constrained-path and independent stopping validation")


class CuroboMpcAdapter:
    """Optional cuRobo 0.7.8 moving-state candidate generator.

    Builds a bounded candidate horizon by rolling cuRobo's documented MPC
    command prediction forward from the FUTURE switch state. Every q/dq/ddq
    sample comes from cuRobo. No returned result is executable until the caller
    independently validates the complete interpolated path and stopping option.
    Default upstream state filtering is disabled: a predicted switch state must
    not be blended with a previous solve's state. Exact timed interpolation and
    stopping validation remain mandatory before execution.
    """
    capabilities = frozenset()
    hardware_commands = False
    source_url = "https://github.com/NVlabs/curobo/blob/v0.7.8/examples/mpc_example.py"

    def __init__(self, solver, tensor_args, *, step_dt_s, maximum_solve_s, horizon_steps, joint_names, full_horizon=False):
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (step_dt_s, maximum_solve_s))
                or not 0 < step_dt_s <= 0.1 or type(horizon_steps) is not int or not 2 <= horizon_steps <= 128 or maximum_solve_s <= 0):
            raise MotionError("invalid bounded MPC configuration")
        self.solver, self.tensor_args = solver, tensor_args
        self.step_dt_s, self.maximum_solve_s, self.horizon_steps = step_dt_s, maximum_solve_s, horizon_steps
        self.joint_names, self._lock, self._goal_buffer = tuple(joint_names), asyncio.Lock(), None
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names) or any(not isinstance(name, str) or not name for name in self.joint_names):
            raise MotionError("MPC requires unique nonempty joint names")
        self.selfcheck_passed = False
        if type(full_horizon) is not bool or (full_horizon and horizon_steps > 98):
            raise MotionError("Pinned full-horizon CUDA kernel requires fewer than 100 states")
        self.full_horizon = full_horizon

    @classmethod
    async def load(cls, *, robot_config, world_config, step_dt_s, maximum_solve_s, horizon_steps, full_horizon=False):
        # Validate bounded dimensions before allocating GPU kernels/buffers.
        cls(None, None, step_dt_s=step_dt_s, maximum_solve_s=maximum_solve_s,
            horizon_steps=horizon_steps, joint_names=['configuration_check'], full_horizon=full_horizon)
        try:
            import curobo
            import torch
            from curobo.types.base import TensorDeviceType
            from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
        except ImportError as exc:
            raise CuroboUnavailable(f"MPC dependency unavailable: {exc.name}") from exc
        if getattr(curobo, "__version__", None) != CUROBO_VERSION or not torch.cuda.is_available():
            raise CuroboUnavailable("MPC requires pinned cuRobo 0.7.8 and CUDA")
        tensor_args = TensorDeviceType()
        def initialize():
            # Source-verified v0.7.8 override_particle_file recursively merges
            # this local config. Default MPC blends q/dq/ddq before optimization,
            # which is incompatible with the caller's future switch boundary.
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as override:
                options = json.loads(json.dumps(MPC_EXACT_BOUNDARY_OVERRIDE))
                if full_horizon:
                    options['model'].update({'return_full_act_buffer':True, 'horizon':horizon_steps+1,
                        'dt_traj_params':{'base_dt':step_dt_s, 'base_ratio':1., 'max_dt':step_dt_s}})
                json.dump(options, override)
                override.flush()
                config = MpcSolverConfig.load_from_robot_config(robot_config, world_config, tensor_args=tensor_args,
                    step_dt=step_dt_s, store_rollouts=True, compute_metrics=True,
                    use_lbfgs=False, self_collision_check=True, override_particle_file=override.name)
            solver = MpcSolver(config)
            state_filter = solver.solver.safety_rollout.dynamics_model.state_filter
            if state_filter.enable or abs(state_filter.dt-step_dt_s) > 1e-9:
                raise CuroboUnavailable("MPC changed its supplied moving boundary or command time step")
            if full_horizon and not solver.solver.safety_rollout.dynamics_model.return_full_act_buffer:
                raise CuroboUnavailable("MPC did not provide the requested full command horizon")
            return solver
        initialization = asyncio.create_task(asyncio.to_thread(initialize))
        try:
            solver = await asyncio.shield(initialization)
        except asyncio.CancelledError:
            await drain_nonpreemptible(initialization)
            raise
        return cls(solver, tensor_args, step_dt_s=step_dt_s, maximum_solve_s=maximum_solve_s,
                   horizon_steps=horizon_steps, joint_names=solver.joint_names, full_horizon=full_horizon)

    def _state(self, state):
        from curobo.types.robot import JointState as CuJointState
        tensor = self.tensor_args.to_device
        return CuJointState(position=tensor([list(state.position)]), velocity=tensor([list(state.velocity)]),
                            acceleration=tensor([list(state.acceleration)]), joint_names=list(self.joint_names))

    def _pose(self, pose):
        from curobo.types.math import Pose
        position, xyzw = pose
        if len(position) != 3 or len(xyzw) != 4 or not all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in (*position, *xyzw)) or abs(sum(v*v for v in xyzw)-1) > 1e-4:
            raise MotionError("MPC goal requires finite metric pose and unit quaternion")
        tensor = self.tensor_args.to_device
        return Pose(position=tensor([list(position)]), quaternion=tensor([[xyzw[3], *xyzw[:3]]]))

    async def candidate(self, *, moving_boundary, goal_pose, world_config, world_identity, goal_at_time=None, _selfcheck=False):
        """goal_at_time is a trusted local profile callable, never model code.

        The callable can realize a fixed admitted constraint goal along a local
        reference. Exact constrained-path checks remain a separate requirement.
        Planning timeout discards output but waits for GPU completion before
        releasing solver/world ownership. The active controller keeps monitoring.
        """
        if not self.selfcheck_passed and not _selfcheck:
            raise CuroboUnavailable("MPC moving-boundary selfcheck has not passed")
        if not isinstance(moving_boundary, JointState) or len(moving_boundary.position) != len(self.joint_names) or not isinstance(world_identity, str) or not world_identity:
            raise MotionError("MPC boundary/world identity missing")
        if goal_at_time is not None and not callable(goal_at_time):
            raise MotionError("MPC moving goal profile must be a trusted local callable")
        if self.full_horizon and goal_at_time is not None:
            raise CuroboUnavailable("Full MPC horizon supports a single grounded pose; constrained references require separate validation")
        async with self._lock:
            def solve():
                from curobo.rollout.rollout_base import Goal
                from curobo.geom.types import WorldConfig
                started = time.monotonic()
                current = self._state(moving_boundary)
                world = WorldConfig.from_dict(world_config) if isinstance(world_config, dict) else world_config
                self.solver.update_world(world)
                pose = self._pose(goal_pose)
                if self._goal_buffer is None:
                    self._goal_buffer = self.solver.setup_solve_single(Goal(current_state=current.clone(), goal_pose=pose), 1)
                else:
                    self._goal_buffer.goal_pose.copy_(pose)
                    self.solver.update_goal(self._goal_buffer)
                if self.full_horizon:
                    result = self.solver.step(current, 1, max_attempts=1)
                    if result.metrics is None or not bool(result.metrics.feasible.all().item()):
                        raise MotionError("MPC optimized horizon is infeasible")
                    action = result.action.get_ordered_joint_state(list(self.joint_names))
                    arrays = []
                    for field in ('position', 'velocity', 'acceleration'):
                        tensor = getattr(action, field)
                        if tensor is None or tuple(tensor.shape) != (1, self.horizon_steps+1, len(self.joint_names)):
                            raise MotionError("MPC full command horizon has invalid state dimensions")
                        arrays.append(tensor.detach().cpu()[0].tolist())
                    dt = self.solver.solver.safety_rollout.dynamics_model.traj_dt.detach().cpu().tolist()
                    if len(dt) != self.horizon_steps+1 or any(abs(v-self.step_dt_s)>1e-8 for v in dt):
                        raise MotionError("MPC full horizon command time steps changed")
                    if time.monotonic()-started > self.maximum_solve_s:
                        raise MotionError("MPC horizon missed measured compute deadline")
                    # Preserve all GPU-returned states, including the float32
                    # start state. Independent boundary checks account for the
                    # conversion; no q/dq/ddq sample is overwritten.
                    trajectory = JointTrajectory(self.joint_names, tuple(TrajectoryPoint(i*self.step_dt_s,
                        JointState(tuple(q),tuple(v),tuple(a))) for i,(q,v,a) in enumerate(zip(*arrays))),
                        f"curobo_mpc:{CUROBO_VERSION}:full_horizon:world:{world_identity}")
                    if not BoundaryTolerance(1e-6,1e-6,1e-6).matches(moving_boundary,trajectory.points[0].state):
                        raise MotionError("MPC full horizon changed the supplied moving q/dq/ddq boundary")
                    return trajectory
                points = [TrajectoryPoint(0., moving_boundary)]
                for step in range(1, self.horizon_steps+1):
                    if time.monotonic()-started > self.maximum_solve_s:
                        raise MotionError("MPC horizon missed measured compute deadline")
                    if goal_at_time is not None:
                        self._goal_buffer.goal_pose.copy_(self._pose(goal_at_time(step*self.step_dt_s)))
                        self.solver.update_goal(self._goal_buffer)
                    result = self.solver.step(current, 1, max_attempts=1)
                    if result.metrics is None or not bool(result.metrics.feasible.all().item()):
                        raise MotionError("MPC predicted state is infeasible")
                    action = result.action.get_ordered_joint_state(list(self.joint_names))
                    if action.velocity is None or action.acceleration is None:
                        raise MotionError("MPC omitted moving-state derivatives")
                    def values(tensor):
                        return tuple(float(v) for v in tensor.detach().cpu().reshape(-1).tolist())
                    measured = JointState(values(action.position), values(action.velocity), values(action.acceleration))
                    points.append(TrajectoryPoint(step*self.step_dt_s, measured))
                    current = action.clone()
                if time.monotonic()-started > self.maximum_solve_s:
                    raise MotionError("MPC horizon missed measured compute deadline")
                return JointTrajectory(self.joint_names, tuple(points), f"curobo_mpc:{CUROBO_VERSION}:world:{world_identity}")
            operation = asyncio.create_task(asyncio.to_thread(solve))
            try:
                return await asyncio.shield(operation)
            except asyncio.CancelledError:
                await drain_nonpreemptible(operation)
                raise

    async def selfcheck(self, *, moving_boundary, goal_pose, world_config, world_identity, validator, dependencies, now, expires_at):
        """Exercise actual GPU output and its independent validator before use.

        Passing proves only the supplied simulation case. It does not enable
        hardware capability, establish a stop profile or authorize contact.
        """
        if not any(abs(v) > 1e-5 for v in moving_boundary.velocity):
            raise MotionError("moving-boundary selfcheck requires actual nonzero velocity")
        candidate = await self.candidate(moving_boundary=moving_boundary, goal_pose=goal_pose, world_config=world_config,
                                         world_identity=world_identity, _selfcheck=True)
        certificate = validator.validate(candidate, dependencies, now=now, expires_at=expires_at)
        self.selfcheck_passed = True
        return certificate
