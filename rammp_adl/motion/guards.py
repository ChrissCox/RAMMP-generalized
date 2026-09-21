"""Local target/refinement and contact geometry guards; no command generation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from .rolling import MotionError, _finite


def distance(a, b):
    if len(a) != len(b) or not len(a) or not _finite((*a, *b)):
        raise MotionError("invalid geometry")
    value = math.hypot(*(x-y for x, y in zip(a, b)))
    if not math.isfinite(value):
        raise MotionError("geometry distance overflow")
    return value


@dataclass(frozen=True)
class TargetObservation:
    entity_id: str
    pose_role: str
    position_m: tuple[float, float, float]
    captured_at: float
    uncertainty_m: float
    identity_confident: bool = True
    quaternion_xyzw: tuple[float, float, float, float] = (0., 0., 0., 1.)

    def __post_init__(self):
        object.__setattr__(self, "position_m", tuple(self.position_m))
        object.__setattr__(self, "quaternion_xyzw", tuple(self.quaternion_xyzw))
        if any(not isinstance(v, str) or not v for v in (self.entity_id, self.pose_role)) or type(self.identity_confident) is not bool:
            raise MotionError("target identity provenance is invalid")
        if len(self.position_m) != 3 or len(self.quaternion_xyzw) != 4 or not _finite((*self.position_m, *self.quaternion_xyzw, self.captured_at, self.uncertainty_m)):
            raise MotionError("invalid target pose evidence")
        if abs(sum(v*v for v in self.quaternion_xyzw)-1) > 1e-6 or self.uncertainty_m < 0:
            raise MotionError("invalid target orientation or uncertainty")


def quaternion_distance(a, b):
    if len(a) != 4 or len(b) != 4 or not _finite((*a, *b)) or any(abs(math.hypot(*q)-1.) > 1e-6 for q in (a, b)):
        raise MotionError("invalid normalized quaternion")
    return 2*math.acos(max(0., min(1., abs(sum(x*y for x, y in zip(a, b))))))


class TargetUpdateGuard:
    def __init__(self, initial, *, max_correction_m, max_uncertainty_m, max_age_s, meaningful_change_m,
                 max_rotation_rad=0.1, meaningful_rotation_rad=0.01):
        if not isinstance(initial, TargetObservation) or not _finite((max_correction_m, max_uncertainty_m, max_age_s, max_rotation_rad, meaningful_change_m, meaningful_rotation_rad)) or min(max_correction_m, max_uncertainty_m, max_age_s, max_rotation_rad) <= 0 or min(meaningful_change_m, meaningful_rotation_rad) < 0:
            raise MotionError("invalid target update profile")
        self.initial, self.latest, self.last_planned = initial, initial, initial
        self.max_correction_m, self.max_uncertainty_m = max_correction_m, max_uncertainty_m
        self.max_age_s, self.meaningful_change_m = max_age_s, meaningful_change_m
        self.max_rotation_rad, self.meaningful_rotation_rad = max_rotation_rad, meaningful_rotation_rad

    def accept(self, observation, *, now):
        if not observation.identity_confident or (observation.entity_id, observation.pose_role) != (self.initial.entity_id, self.initial.pose_role):
            raise MotionError("target identity or pose role changed")
        if not 0 <= now-observation.captured_at <= self.max_age_s:
            raise MotionError("target evidence stale or future dated")
        if observation.captured_at < self.latest.captured_at:
            raise MotionError("out-of-order target observation")
        if observation.captured_at == self.latest.captured_at and observation != self.latest:
            raise MotionError("conflicting geometry for the same capture time")
        if not 0 <= observation.uncertainty_m <= self.max_uncertainty_m:
            raise MotionError("target uncertainty exceeds envelope")
        if distance(self.initial.position_m, observation.position_m) > self.max_correction_m:
            raise MotionError("cumulative target correction exceeds envelope")
        if quaternion_distance(self.initial.quaternion_xyzw, observation.quaternion_xyzw) > self.max_rotation_rad:
            raise MotionError("cumulative target rotation exceeds envelope")
        changed = (distance(self.last_planned.position_m, observation.position_m) >= self.meaningful_change_m or
                   quaternion_distance(self.last_planned.quaternion_xyzw, observation.quaternion_xyzw) >= self.meaningful_rotation_rad)
        self.latest = observation
        if changed:
            self.last_planned = observation
        return changed


@dataclass(frozen=True)
class ConstraintGeometry:
    constraint_id: str
    entity_id: str
    kind: str
    origin_m: tuple[float, float, float]
    axis: tuple[float, float, float]
    reference_m: tuple[float, float, float]
    minimum: float
    maximum: float
    path_tolerance_m: float
    allowed_contact_pairs: frozenset[tuple[str, str]]

    def __post_init__(self):
        for field in ("origin_m", "axis", "reference_m"):
            values = tuple(getattr(self, field))
            if len(values) != 3 or not _finite(values):
                raise MotionError("constraint requires finite 3D geometry")
            object.__setattr__(self, field, values)
        pairs = tuple(tuple(pair) for pair in self.allowed_contact_pairs)
        if not pairs or any(len(pair) != 2 or any(not isinstance(v, str) or not v for v in pair) for pair in pairs):
            raise MotionError("constraint requires explicit contact pairs")
        object.__setattr__(self, "allowed_contact_pairs", frozenset(pairs))
        if any(not isinstance(v, str) or not v for v in (self.constraint_id, self.entity_id)) or not _finite((self.minimum, self.maximum, self.path_tolerance_m)):
            raise MotionError("invalid constraint identity or bounds")
        if self.kind not in ("prismatic", "revolute"):
            raise MotionError("constraint geometry is not implemented")
        if abs(distance(self.axis, (0, 0, 0))-1) > 1e-6 or self.minimum >= self.maximum or self.path_tolerance_m <= 0:
            raise MotionError("invalid constraint geometry profile")

    @property
    def unit(self):
        return "rad" if self.kind == "revolute" else "m"

    def point_at(self, coordinate):
        if not _finite((coordinate,)) or not self.minimum <= coordinate <= self.maximum:
            raise MotionError("absolute articulation coordinate outside bounds")
        if self.kind == "prismatic":
            return tuple(r + a*coordinate for r, a in zip(self.reference_m, self.axis))
        # Rodrigues geometry evaluates an expected physical relation; it is not IK.
        relative = tuple(r-o for r, o in zip(self.reference_m, self.origin_m))
        dot = sum(a*b for a, b in zip(self.axis, relative))
        x, y, z = self.axis
        a, b, c = relative
        cross = (y*c-z*b, z*a-x*c, x*b-y*a)
        return tuple(o + r*math.cos(coordinate) + cr*math.sin(coordinate) + ax*dot*(1-math.cos(coordinate))
                     for o, r, cr, ax in zip(self.origin_m, relative, cross, self.axis))

    def check(self, coordinate, measured_position_m, contact_pairs, *, target_unit, retained=True):
        if target_unit != self.unit:
            raise MotionError("articulation units disagree")
        if retained is not True:
            raise MotionError("slip")
        if not contact_pairs or not set(contact_pairs).issubset(self.allowed_contact_pairs):
            raise MotionError("unmodeled contact")
        if distance(self.point_at(coordinate), measured_position_m) > self.path_tolerance_m:
            raise MotionError("model_mismatch")

    def check_refinement(self, refined, *, max_origin_shift_m, max_axis_angle_rad):
        if not _finite((max_origin_shift_m, max_axis_angle_rad)) or min(max_origin_shift_m, max_axis_angle_rad) < 0:
            raise MotionError("invalid articulation refinement envelope")
        if (self.constraint_id, self.entity_id, self.kind, self.minimum, self.maximum, self.allowed_contact_pairs) != (refined.constraint_id, refined.entity_id, refined.kind, refined.minimum, refined.maximum, refined.allowed_contact_pairs):
            raise MotionError("refinement changes admitted mechanism or contact policy")
        axis_angle = math.acos(max(-1., min(1., sum(a*b for a, b in zip(self.axis, refined.axis)))))
        if distance(self.origin_m, refined.origin_m) > max_origin_shift_m or distance(self.reference_m, refined.reference_m) > max_origin_shift_m or axis_angle > max_axis_angle_rad:
            raise MotionError("articulation refinement exceeds envelope")
        if refined.path_tolerance_m > self.path_tolerance_m:
            raise MotionError("refinement relaxes path accuracy")
