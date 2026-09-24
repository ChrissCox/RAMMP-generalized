"""A constraint followed as one trajectory: the planner's joint solutions along it, splined and re-timed.

RAMMP-CuRobo plans from a stationary start only, so a constraint is still followed as
a sequence of plans from rest. They are made in advance: each waypoint is planned from
the joints the previous plan ends at, plan-only, while the arm holds the part still.
What is flown is one trajectory through the planner's own solutions at those
waypoints: a cubic spline in joint space, parameterised by the constraint's value
(door angle, drawer travel) and re-timed so the value moves at a bounded rate and
acceleration and no joint exceeds its share of the URDF velocity limit. The path
between waypoints is the spline's, not the planner's; the caller checks it against
the constraint with the kinematic model (deviation()) before anything is sent.

No IK is solved here: every configuration the spline passes through at a waypoint
is one the planner returned.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .rolling import MotionError
from .sheppy_client import JOINTS, trajectory_from_planner


class PathError(MotionError):
    """A chain of plans that cannot be flown as one trajectory."""


#: Largest joint change between neighbouring waypoints. The planner's solutions along a
#: door arc move a joint by at most a few tenths of a radian per waypoint; more is the
#: solver changing branch, and a spline through it would swing the arm while it holds the part.
MAX_KNOT_STEP_RAD = .6
#: Samples on the wire; the driver's cubic Hermite interpolation between them is exact to well under a millimetre.
SAMPLE_DT_S = .04


@dataclass(frozen=True)
class Timing:
    """Where along the constraint the trajectory is at each sample time."""
    times_s: tuple[float, ...]
    values: tuple[float, ...]

    def value_at(self, elapsed_s):
        return float(np.interp(float(elapsed_s), self.times_s, self.values))

    def time_at(self, value):
        """The first sample time at which the pass has reached `value`."""
        index = int(np.searchsorted(np.asarray(self.values), float(value)-1e-9))
        return self.times_s[min(index, len(self.times_s)-1)]


def knot_spline(values, knots):
    """A C2 joint path through the planner's solutions, as a function of the constraint value."""
    values, knots = np.asarray(values, dtype=float), np.asarray(knots, dtype=float)
    if values.ndim != 1 or len(values) < 2 or knots.shape != (len(values), len(JOINTS)):
        raise PathError("a path needs the start and at least one waypoint, seven joints each")
    if not (np.isfinite(values).all() and np.isfinite(knots).all()):
        raise PathError("waypoint values and joints must be finite")
    if values[0] != 0. or np.any(np.diff(values) <= 0.):
        raise PathError("waypoint values start at zero and increase")
    steps = np.abs(np.diff(knots, axis=0))
    if steps.max() > MAX_KNOT_STEP_RAD:
        index, joint = np.unravel_index(int(np.argmax(steps)), steps.shape)
        raise PathError(f"joint {joint+1} moves {steps[index, joint]:.2f} rad between waypoints {values[index]:.3f} and "
                        f"{values[index+1]:.3f}; the planner changed branch, and that is not one motion")
    from scipy.interpolate import CubicSpline
    return CubicSpline(values, knots, axis=0, bc_type="natural")


def _eroded(caps, window):
    """Each cap lowered to the smallest within `window` samples either side."""
    if window <= 0:
        return caps
    padded = np.pad(caps, window, mode="edge")
    return np.min(np.lib.stride_tricks.sliding_window_view(padded, 2*window+1), axis=1)


def retime(spline, total, *, rate_limits, value_rate, value_accel, smoothing_s=.4, dt=SAMPLE_DT_S):
    """Sample times, values and value rates for a pass from 0 to total and back to rest.

    The value's rate is capped where any joint would pass its limit, accelerated and
    braked at value_accel (a forward and a backward pass), then smoothed over smoothing_s
    so acceleration has no steps; caps are first lowered over the smoothing reach so the
    smoothed profile stays under them.
    """
    limits = np.asarray(rate_limits, dtype=float)
    if limits.shape != (len(JOINTS),) or not (limits > 0.).all():
        raise PathError("one positive rate limit per joint")
    if not (total > 0. and value_rate > 0. and value_accel > 0. and smoothing_s >= 0. and dt > 0.):
        raise PathError("the pass, its rate and acceleration and the sample time must be positive")
    count = max(200, int(math.ceil(total/.002)))
    grid = np.linspace(0., total, count+1)
    step = total/count
    slope = np.abs(spline(grid, 1))
    caps = np.minimum(value_rate, np.min(limits/np.maximum(slope, 1e-9), axis=1))
    caps = _eroded(caps, int(math.ceil(value_rate*smoothing_s/step)))
    squared = caps*caps
    squared[0] = 0.
    for i in range(count):                                 # accelerate as fast as allowed
        squared[i+1] = min(squared[i+1], squared[i]+2.*value_accel*step)
    squared[-1] = 0.
    for i in range(count-1, -1, -1):                       # and brake in time
        squared[i] = min(squared[i], squared[i+1]+2.*value_accel*step)
    rate = np.sqrt(squared)
    times = np.concatenate([[0.], np.cumsum(2.*step/np.maximum(rate[:-1]+rate[1:], 1e-12))])
    fine = min(.005, dt/4.)
    clock = np.arange(0., times[-1]+fine, fine)
    fine_rate = np.interp(clock, times, rate)
    width = int(round(smoothing_s/fine))
    if width >= 3:
        kernel = np.hanning(width+2)[1:-1]
        fine_rate = np.convolve(np.concatenate([np.zeros(width), fine_rate, np.zeros(width)]), kernel/kernel.sum(), mode="same")
        fine_rate = np.trim_zeros(np.where(fine_rate < 1e-12, 0., fine_rate))
        fine_rate = np.concatenate([[0.], fine_rate, [0.]])
        clock = np.arange(len(fine_rate))*fine
    travelled = np.concatenate([[0.], np.cumsum((fine_rate[1:]+fine_rate[:-1])*fine/2.)])
    fine_rate = fine_rate*(total/travelled[-1])            # land exactly on the last waypoint
    travelled = travelled*(total/travelled[-1])
    samples = np.arange(0., clock[-1]+dt/2., dt)
    samples[-1] = clock[-1]
    value = np.interp(samples, clock, travelled)
    value[-1] = total
    value_rate_out = np.interp(samples, clock, fine_rate)
    value_rate_out[[0, -1]] = 0.
    return samples, value, value_rate_out


def constraint_trajectory(values, knots, *, rate_limits, value_rate, value_accel, provenance, smoothing_s=.4,
                          dt=SAMPLE_DT_S):
    """The planner's waypoint solutions flown as one trajectory; returns it, its timing and the spline.

    The first waypoint is the start state, held for one sample like the planner's own
    first waypoint, so the wire message has the planner's shape: positions and
    velocities, the start one step ahead. If rounding leaves any joint over its rate
    limit, the whole pass is slowed until none is.
    """
    spline = knot_spline(values, knots)
    total = float(values[-1])
    times, value, value_rate_out = retime(spline, total, rate_limits=rate_limits, value_rate=value_rate,
                                          value_accel=value_accel, smoothing_s=smoothing_s, dt=dt)
    positions = spline(value)
    velocities = spline(value, 1)*value_rate_out[:, None]
    positions[0], positions[-1] = np.asarray(knots[0], dtype=float), np.asarray(knots[-1], dtype=float)
    velocities[[0, -1]] = 0.
    over = float(np.max(np.abs(velocities)/np.asarray(rate_limits, dtype=float)))
    if over > 1.:
        times, velocities, value_rate_out = times*over, velocities/over, value_rate_out/over
    shift = times[1]
    rows = [(float(t+shift), tuple(float(v) for v in q), tuple(float(v) for v in qd), None)
            for t, q, qd in zip(times, positions, velocities)]
    trajectory = trajectory_from_planner(JOINTS, rows, provenance=provenance)
    timing = Timing(tuple(float(t) for t in np.concatenate([[0.], times+shift])),
                    tuple(float(v) for v in np.concatenate([[0.], value])))
    return trajectory, timing, spline


def deviation(spline, total, tool_pose, constraint_pose, *, spacing=.005):
    """Largest distance (m) and turn (rad) between the spline's tool pose and the constraint's, sampled along it.

    tool_pose(joints) and constraint_pose(value) return (position, 3x3 rotation). The
    waypoints themselves are included: the planner reaches a pose within its own tolerance.
    """
    worst = {"distance_m": 0., "distance_at": 0., "turn_rad": 0., "turn_at": 0.}
    for value in np.linspace(0., total, max(3, int(math.ceil(total/spacing))+1)):
        position, rotation = tool_pose(spline(value))
        goal_position, goal_rotation = constraint_pose(float(value))
        distance = float(np.linalg.norm(np.asarray(position)-np.asarray(goal_position)))
        turn = float(np.arccos(np.clip((np.trace(np.asarray(goal_rotation).T @ np.asarray(rotation))-1.)/2., -1., 1.)))
        if distance > worst["distance_m"]:
            worst.update(distance_m=distance, distance_at=float(value))
        if turn > worst["turn_rad"]:
            worst.update(turn_rad=turn, turn_at=float(value))
    return worst
