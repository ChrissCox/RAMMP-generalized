"""What a grasped part's mechanism is, from how the hand actually moved with it.

Pulled compliantly (the driver's joint impedance mode), the hand follows the part
wherever its hinge or slide takes it; the measured tool path is then the
mechanism's own path, not the planned one. This fits that path: a hinge as a
circle about an axis (the pivot, the radius, how far it turned), a slide as a
line (the direction, how far it went), and says which fits better.

A short arc says little about its centre, so the fit is pulled toward the
perception prior with the prior's own uncertainty: early in a pull the prior
dominates, and the measurement takes over as the part turns. Nothing here
commands anything; the caller re-plans from the fit through the planner.
"""
from __future__ import annotations

import math

import numpy as np

from .rolling import MotionError


class ArticulationError(MotionError):
    """A path that says nothing usable about the mechanism."""


def _basis(axis):
    axis = np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    helper = np.array([1., 0., 0.]) if abs(axis[0]) < .9 else np.array([0., 1., 0.])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return axis, u, np.cross(axis, u)


def fit_slide(positions):
    """A line through the path: direction (from start toward end), travel (m), rms off the line (m)."""
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ArticulationError("a slide fit needs at least three positions")
    centre = points.mean(axis=0)
    direction = np.linalg.svd(points-centre)[2][0]
    if (points[-1]-points[0]) @ direction < 0:
        direction = -direction
    along = (points-centre) @ direction
    off = points-centre-np.outer(along, direction)
    return {"kind": "prismatic", "axis": direction.tolist(), "travelled": float(along[-1]-along[0]),
            "rms_m": float(np.sqrt(np.mean(np.sum(off*off, axis=1))))}


def fit_hinge(positions, *, axis, prior_pivot, prior_sigma_m=.02, measure_sigma_m=.001, iterations=30):
    """A circle about `axis` through the path, regularised toward the prior pivot.

    Each measured point counts with its own uncertainty (measure_sigma_m), the
    prior pivot with its own (prior_sigma_m): a long arc outweighs the prior,
    a short one, whose centre the points barely constrain, does not.

    Returns pivot (on the line through the prior's height), radius, the signed
    angle turned about the axis from the first to the last position, and the
    rms distance of the path from the fitted circle.
    """
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ArticulationError("a hinge fit needs at least three positions")
    axis, u, v = _basis(axis)
    prior = np.asarray(prior_pivot, dtype=float)
    flat = np.column_stack([(points-prior) @ u, (points-prior) @ v])       # in the plane, prior pivot at the origin
    radius0 = float(np.mean(np.linalg.norm(flat, axis=1)))
    centre, radius = np.zeros(2), radius0
    weight, scale = 1./max(prior_sigma_m, 1e-6), 1./max(measure_sigma_m, 1e-6)
    for _ in range(iterations):
        offsets = flat-centre
        distances = np.maximum(np.linalg.norm(offsets, axis=1), 1e-9)
        residual = np.concatenate([scale*(distances-radius), weight*centre])
        jacobian = np.zeros((len(flat)+2, 3))
        jacobian[:len(flat), :2] = -scale*offsets/distances[:, None]
        jacobian[:len(flat), 2] = -scale
        jacobian[len(flat):, :2] = weight*np.eye(2)
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        centre, radius = centre+step[:2], radius+step[2]
        if np.linalg.norm(step) < 1e-9:
            break
    if not radius > 0:
        raise ArticulationError("the path fits no circle about the given axis")
    pivot = prior+centre[0]*u+centre[1]*v
    first, last = flat[0]-centre, flat[-1]-centre
    turned = math.atan2(first[0]*last[1]-first[1]*last[0], float(first @ last))
    rms = float(np.sqrt(np.mean((np.linalg.norm(flat-centre, axis=1)-radius)**2)))
    return {"kind": "revolute", "axis": axis.tolist(), "pivot": pivot.tolist(), "radius": float(radius),
            "turned": float(turned), "rms_m": rms, "shift_m": float(np.linalg.norm(centre))}


def fit_articulation(positions, *, kind, axis=None, prior_pivot=None, prior_sigma_m=.02, measure_sigma_m=.001):
    """The mechanism the path shows: the prior's kind unless the other fits clearly better.

    A hinge whose fitted circle is larger than two metres, or whose path the
    circle misses by over a millimetre while a line fits twice as well, moved as a slide. On a
    short arc a line and a circle fit equally well, and that is not evidence of
    a slide. A slide is left a slide: a hinge needs a pivot, and a slide gives
    none to fit toward.
    """
    slide = fit_slide(positions)
    if kind == "revolute":
        hinge = fit_hinge(positions, axis=axis, prior_pivot=prior_pivot, prior_sigma_m=prior_sigma_m,
                          measure_sigma_m=measure_sigma_m)
        if hinge["radius"] > 2. or (hinge["rms_m"] > .001 and slide["rms_m"] < .5*hinge["rms_m"]):
            return {**slide, "overruled": "revolute"}
        return hinge
    return slide


def hinge_from_faces(initial_normal, normal, *, prior_axis, min_turn_rad=.15, max_axis_change_deg=10.):
    """The axis a panel turned about, from its face's normal before and after: (fit, "") or (None, why).

    A single view of a door gives its face, and a hinge lies in that face, but not where in it: a cabinet leaning
    sideways tilts its hinge within the face, which no one view shows. Once the panel has turned, the two normals
    span the plane the hinge is perpendicular to: their cross product is the axis, their angle how far it turned.
    Too small a turn leaves the cross product to the camera's noise; an axis far from the prior is a wrong plane.
    """
    first = np.asarray(initial_normal, dtype=float)
    second = np.asarray(normal, dtype=float)
    prior = np.asarray(prior_axis, dtype=float)
    first, second, prior = first/np.linalg.norm(first), second/np.linalg.norm(second), prior/np.linalg.norm(prior)
    turned = math.acos(min(1., max(-1., float(first @ second))))
    if turned < min_turn_rad:
        return None, f"the face turned {turned:.3f} rad, too little to place its axis"
    axis = np.cross(first, second)
    axis /= np.linalg.norm(axis)
    if axis @ prior < 0.:
        axis = -axis
    change = math.degrees(math.acos(min(1., float(axis @ prior))))
    if change > max_axis_change_deg:
        return None, f"the faces put the axis {change:.1f} deg from the placed one; not the same panel"
    return {"axis": [float(v) for v in axis], "turned": turned, "axis_change_deg": change}, ""
