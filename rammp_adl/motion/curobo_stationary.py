"""Explicit cuRobo trajectory interpolation extension, without robot I/O.

cuRobo 0.7.8's QUINTIC option already uses SciPy make_interp_spline, but
does not return its derivatives. Its LINEAR_CUDA option combines finite-
difference derivatives independently and omits the final optimized knot.
This extension interpolates the raw optimized knots with static velocity and
acceleration boundary constraints and evaluates q/dq/ddq from the same spline.
It does not generate IK or a new route. The changed curve must be independently
validated; it does not inherit the optimized knots' collision checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from .rolling import MotionError

EXTENSION_ID = "curobo-v078-stationary-quintic-v1"


def extension_identity():
    return EXTENSION_ID + ":sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class InterpolatedState:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    dt: float
    subdivisions: int


def interpolate_stationary_knots(positions, *, knot_dt_s, maximum_sample_dt_s):
    """Complete cuRobo's quintic interpolation with analytic derivatives.

    Every optimized knot is retained, with an integer number of subdivisions
    per knot interval. Consequently the receiving quintic Hermite trajectory
    never spans an unrepresented spline break. Work/output are bounded and no
    returned endpoint derivative is overwritten or separately estimated.
    """
    from scipy.interpolate import make_interp_spline

    positions = np.asarray(positions, dtype=np.float64)
    if (positions.ndim != 2 or not 6 <= positions.shape[0] <= 4096
            or not 1 <= positions.shape[1] <= 32 or not np.isfinite(positions).all()):
        raise MotionError("cuRobo interpolation requires finite bounded optimized position knots")
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (knot_dt_s, maximum_sample_dt_s))
            or maximum_sample_dt_s > .1):
        raise MotionError("cuRobo interpolation requires positive bounded time steps")
    duration = (len(positions) - 1) * knot_dt_s
    if not math.isfinite(duration) or duration > 120. or knot_dt_s / maximum_sample_dt_s > 100000:
        raise MotionError("cuRobo interpolation exceeds bounded trajectory work")
    subdivisions = max(1, math.ceil(knot_dt_s / maximum_sample_dt_s))
    point_count = (len(positions) - 1) * subdivisions + 1
    if point_count > 100000:
        raise MotionError("cuRobo interpolation exceeds bounded trajectory work")
    knot_times = np.arange(len(positions), dtype=np.float64) * knot_dt_s
    zero = np.zeros(positions.shape[1])
    spline = make_interp_spline(knot_times, positions, k=5,
                               bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]))
    # linspace includes the exact terminal knot; all source knots occur at
    # every subdivisions-th sample (up to floating point roundoff).
    times = np.linspace(0., duration, point_count)
    q, dq, ddq = (spline(times, nu=order, extrapolate=False) for order in (0, 1, 2))
    if not all(np.isfinite(values).all() for values in (q, dq, ddq)):
        raise MotionError("cuRobo interpolation produced nonfinite state evidence")
    return InterpolatedState(q, dq, ddq, knot_dt_s / subdivisions, subdivisions)


def load_stationary_planner(module, config_path):
    """Subclass only the result conversion of a previously verified module.

    The pinned wrapper's construction, warmup, planning, world conversion,
    controller ordering and original postchecks are unchanged. BACKWARD is
    deliberately not selected: its indexed v0.7.8 CUDA kernel asserts false.
    """
    class StationaryPlanner(module.CuRoboPlanner):
        rammp_adl_boundary_extension = extension_identity()

        def _finish(self, result, t0, q_goal=None):
            original = super()._finish(result, t0, q_goal=q_goal)
            if not original.success:
                return original
            if result.optimized_plan is None or result.optimized_dt is None:
                raise MotionError("cuRobo result lacks raw optimized knot/time evidence")
            raw = result.optimized_plan.get_ordered_joint_state(self.joint_names)
            q = raw.position.detach().cpu().numpy().astype(float).copy()
            dt = float(result.optimized_dt.item())
            dense = interpolate_stationary_knots(q, knot_dt_s=dt,
                                                 maximum_sample_dt_s=self.interpolation_dt)
            trajectory = module.Trajectory(
                joint_names=list(self.joint_names), positions=dense.positions,
                velocities=dense.velocities, accelerations=dense.accelerations, dt=dense.dt,
            )
            limits = self.joint_limits()
            problems = module.validate_trajectory(trajectory, limits["position"], limits["velocity"])
            from curobo.types.robot import JointState as CuJointState
            candidate = CuJointState(
                position=self._tensor(dense.positions), velocity=self._tensor(dense.velocities),
                acceleration=self._tensor(dense.accelerations), joint_names=list(self.joint_names),
            ).get_ordered_joint_state(self._curobo_joint_names)
            problems += self._recheck_constraints(candidate)
            # Preserve numeric source evidence before returning any rejection.
            self.boundary_debug = {
                "extension": self.rammp_adl_boundary_extension,
                "optimized_dt": dt, "optimized_position": q.tolist(),
                "optimized_velocity": raw.velocity.detach().cpu().tolist(),
                "optimized_acceleration": raw.acceleration.detach().cpu().tolist(),
                "original_interpolated_dt": original.joint_traj.dt,
                "original_interpolated_position": original.joint_traj.positions.tolist(),
                "original_interpolated_velocity": original.joint_traj.velocities.tolist(),
                "original_interpolated_acceleration": original.joint_traj.accelerations.tolist(),
                "output_dt": dense.dt, "subdivisions_per_knot_interval": dense.subdivisions,
                "upstream_postcheck_problems": problems,
                "independent_validation_required": True, "hardware_validated": False,
            }
            if problems:
                return module.PlanResult.failure("INTERPOLATION_VALIDATION_FAILED", "; ".join(problems))
            return module.PlanResult(
                success=True, joint_traj=trajectory, timing=original.timing, error=None,
                status="OK", validated=True, final_joints=dense.positions[-1].copy(),
                goal_mismatch_rad=original.goal_mismatch_rad,
            )

    return StationaryPlanner.from_config(str(config_path))
