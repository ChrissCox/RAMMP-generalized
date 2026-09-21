"""Locally validated, supervised free-space commissioning of the arm transport.

This is not an ADL handler or permission to execute contact tasks. A physical
test needs an independently established collision-free joint cell covering the
entire installed assembly, every joint combination in the cell, and measured
stopping excursions. A sample of collision-free poses is NOT such a cell.
The runtime checks the exact driver interpolation stays within that cell.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from ..contracts import strict_loads
from .driver_feedback import FeedbackBounds
from .driver_transport import TransportBounds, canonical_ros_trajectory
from .rolling import JointLimits, MotionError, TrajectoryValidator
from .curobo import RAMMP_COMMIT


def sha256(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vector(value, name, *, size=7, positive=False):
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) or (positive and v <= 0) for v in value)):
        raise MotionError("Invalid commissioned vector: " + name)
    return tuple(value)


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise MotionError("Missing/invalid positive commissioned value: " + name)
    return value


def _bernstein_bounds(power):
    """Conservative polynomial bounds on [0,1], with inward numerical margin.

    A polynomial lies in the convex hull of its Bernstein coefficients; this
    checks every point, unlike testing a grid or only the optimizer's knots.
    Rejecting a loose hull is safe. No new trajectory is generated here.
    """
    degree = len(power)-1
    values = [sum(power[j]*math.comb(k,j)/math.comb(degree,j) for j in range(k+1))
              for k in range(degree+1)]
    padding = 64*np.finfo(float).eps*max(1., sum(abs(v) for v in power))
    return min(values)-padding, max(values)+padding


def _coefficients(first, last, joint):
    dt = last.time_s-first.time_s
    q0, v0, a0 = (getattr(first.state, field)[joint] for field in ("position", "velocity", "acceleration"))
    q1, v1, a1 = (getattr(last.state, field)[joint] for field in ("position", "velocity", "acceleration"))
    c0, c1, c2 = q0, v0*dt, a0*dt*dt/2
    delta, vel, acc = q1-c0-c1-c2, v1*dt-c1-2*c2, a1*dt*dt-2*c2
    return np.asarray((c0, c1, c2, 10*delta-4*vel+acc/2, -15*delta+7*vel-acc, 6*delta-3*vel+acc/2)), dt


@dataclass(frozen=True)
class CommissioningSettings:
    profile_id: str
    simulation: bool
    feedback: FeedbackBounds
    transport: TransportBounds
    watchdog_timeout_s: float
    limits: JointLimits
    cell_lower: tuple
    cell_upper: tuple
    stop_excursion: tuple
    tracking_reserve: tuple
    jerk: tuple
    maximum_link_lever_m: tuple
    max_link_speed_m_s: float
    max_tool_speed_m_s: float
    evidence_expires_unix_s: float
    extension_build_id: str
    planner_model_digest: str
    world_digest: str
    asset_pins: tuple
    dependencies: tuple
    asset_stats: tuple

    def unchanged(self):
        if time.time() >= self.evidence_expires_unix_s:
            raise MotionError("Commissioned cell/environment evidence has expired")
        for path, expected in self.asset_stats:
            stat = path.stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != expected:
                raise MotionError("A commissioned dependency changed; new validation required: " + str(path))


def measured_transport_bounds(settings):
    """Reserve sensor uncertainty inside physical completion tolerances.

    The transport receives numerical q/dq samples, so it must use the remaining
    measured-error allowance, not spend the uncertainty budget a second time.
    Tracking reserves already explicitly include position uncertainty in the
    complete collision-free cell verifier.
    """
    b = settings.transport
    uncertainty = tuple(reserve-path for reserve,path in zip(settings.tracking_reserve, b.path_position_rad))
    return replace(b,
        goal_position_rad=tuple(goal-error for goal,error in zip(b.goal_position_rad, uncertainty)),
        start_position_rad=tuple(start-error for start,error in zip(b.start_position_rad, uncertainty)),
        stationary_velocity_rad_s=b.stationary_velocity_rad_s-settings.feedback.velocity_error_rad_s,
        stationary_position_span_rad=b.stationary_position_span_rad-2*max(uncertainty))


def load_settings(commissioning_path, profile_id, *, simulation=False, require_motion_enabled=True):
    """Use only the canonical commissioning file for trusted physical limits.

    Simulation-test profiles must identify themselves explicitly and cannot be
    promoted by changing a command-line flag. No simulation defaults exist here.
    """
    path = Path(commissioning_path).resolve()
    config = strict_loads(path.read_bytes())
    if require_motion_enabled and not simulation and config.get("hardware_motion_enabled") is not True:
        raise MotionError("Hardware motion is disabled in config/commissioning.json")
    profile = config.get("profiles", {}).get(profile_id)
    if not isinstance(profile, dict):
        raise MotionError("No commissioned profile: " + profile_id)
    missing = [name for name in config["required_profile_fields"] if profile.get(name) is None]
    if missing:
        raise MotionError("Missing commissioning fields: " + ", ".join(missing))
    test = profile.get("free_space_test")
    if not isinstance(test, dict) or test.get("simulation_only") is not simulation:
        raise MotionError("An explicit mode-matched free_space_test profile is required")
    if profile["allowed_contacts"] != [] or test.get("payload_kg") != 0:
        raise MotionError("This commissioning route supports empty-gripper free space only")
    if test.get("environment_mode") != "secured_static_test_cell":
        raise MotionError("This route requires a secured static cell; human/ADL motion is unavailable")
    feedback = FeedbackBounds(**test["feedback"])
    transport = TransportBounds(**test["transport"])
    watchdog = _positive(test["watchdog_timeout_s"], "watchdog_timeout_s")
    stop_latency = _positive(profile["stop_latency_s"], "stop_latency_s")
    if not transport.poll_period_s*3 < watchdog <= stop_latency:
        raise MotionError("Independent driver watchdog does not fit the measured stopping budget")
    if (transport.state_max_age_s != feedback.state_max_age_s
            or transport.receipt_max_age_s != feedback.receipt_max_age_s
            or transport.stationary_velocity_rad_s <= feedback.velocity_error_rad_s):
        raise MotionError("Transport and acquisition bounds disagree")
    cell_lower = _vector(test["collision_free_cell_lower_rad"], "cell lower")
    cell_upper = _vector(test["collision_free_cell_upper_rad"], "cell upper")
    stop = _vector(test["stopping_joint_excursion_rad"], "stop excursions", positive=True)
    uncertainty = _vector(test["joint_position_uncertainty_rad"], "joint uncertainty", positive=True)
    reserve = tuple(a+b for a,b in zip(uncertainty, transport.path_position_rad))
    limits = JointLimits(_vector(test["joint_lower_rad"], "joint lower"),
                         _vector(test["joint_upper_rad"], "joint upper"),
                         _vector(profile["max_joint_speed_rad_s"], "joint speed", positive=True),
                         _vector(profile["max_joint_acceleration_rad_s2"], "joint acceleration", positive=True))
    if any(not lo < a+s+r < b-s-r < hi for lo,hi,a,b,s,r in
           zip(limits.lower, limits.upper, cell_lower, cell_upper, stop, reserve)):
        raise MotionError("Collision-free cell cannot contain tracking uncertainty and stopping excursions")
    jerk = _vector(test["max_joint_jerk_rad_s3"], "joint jerk", positive=True)
    levers = _vector(test["all_assembly_point_lever_bounds_m"], "assembly lever bounds", positive=True)
    pins = [(path, sha256(path))]
    required_assets = {"cell_evidence", "robot_urdf", "planner_robot_config", "planner_config"}
    assets = test.get("assets", {})
    if set(assets) != required_assets:
        raise MotionError("Explicit cell evidence and complete planner/model assets are required")
    for name in sorted(required_assets):
        ref = assets[name]
        asset = (path.parent / ref["path"]).resolve()
        if sha256(asset) != ref["sha256"]:
            raise MotionError("Commissioning asset digest mismatch: " + name)
        pins.append((asset, ref["sha256"]))
    evidence_path = (path.parent/assets["cell_evidence"]["path"]).resolve()
    evidence = strict_loads(evidence_path.read_bytes())
    expected_scope = "simulation_transport_test" if simulation else "independently_reviewed_physical_joint_cell"
    required_coverage = {"kinova_gen3_7dof", "robotiq_2f85_all_apertures", "wrist_d405_and_mount",
                         "all_joint_combinations_in_cell", "secured_static_environment", "empty_payload"}
    if (evidence.get("scope") != expected_scope or set(evidence.get("coverage", [])) != required_coverage
            or evidence.get("profile_id") != profile_id or not evidence.get("reviewer")
            or not evidence.get("method") or not evidence.get("records")
            or evidence.get("collision_free_cell_lower_rad") != list(cell_lower)
            or evidence.get("collision_free_cell_upper_rad") != list(cell_upper)
            or evidence.get("robot_urdf_digest") != assets["robot_urdf"]["sha256"]):
        raise MotionError("Cell evidence does not cover the configured entire assembly/region/model")
    # Supporting local records are pinned too. A JSON success label, sampled
    # simulation result, single clear pose or button press is not a cell proof.
    for record in evidence["records"]:
        record_path = (evidence_path.parent/record["path"]).resolve()
        if sha256(record_path) != record["sha256"]:
            raise MotionError("Commissioning evidence record changed")
        pins.append((record_path, record["sha256"]))
    extension = test["extension_build_id"]
    if not isinstance(extension, str) or len(extension) != 64 or any(c not in "0123456789abcdef" for c in extension):
        raise MotionError("Pinned driver extension build required")
    settings = CommissioningSettings(profile_id, simulation, feedback, transport, watchdog, limits,
        cell_lower, cell_upper, stop, reserve, jerk, levers,
        _positive(profile["max_link_speed_m_s"], "link speed"), _positive(profile["max_tool_speed_m_s"], "tool speed"),
        _positive(evidence["expires_unix_s"], "cell evidence expiry"), extension,
        test["planner_loaded_robot_config_digest"], evidence["world_digest"], tuple(pins),
        tuple(sorted((str(p), d) for p,d in pins)),
        tuple((p, (p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)) for p,d in pins))
    settings.unchanged()
    measured_transport_bounds(settings)
    return settings


class CommissionedCellVerifier:
    """Check the entire timed polynomial and measured stopping reserve.

    Geometry validity comes from the reviewed full joint cell, not optimizer
    success. The conservative lever bounds cover every point of every arm,
    gripper and camera body, so the Cartesian speed bound includes attachments.
    No collision or human-tracking capability is registered for ADL execution.
    """
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, trajectory, dependencies):
        s = self.settings
        s.unchanged()
        if not trajectory.provenance.startswith("rammp_curobo:"+RAMMP_COMMIT+":"):
            raise MotionError("Arm trajectory must originate in the verified local cuRobo adapter")
        if dict(s.dependencies) != dict(dependencies):
            raise MotionError("Commissioning dependencies differ")
        for first, last in zip(trajectory.points, trajectory.points[1:]):
            velocity_bounds = []
            for joint in range(7):
                power, dt = _coefficients(first, last, joint)
                lower, upper = _bernstein_bounds(power)
                margin = s.stop_excursion[joint]+s.tracking_reserve[joint]
                if lower < s.cell_lower[joint]+margin or upper > s.cell_upper[joint]-margin:
                    raise MotionError("Full path/stopping envelope leaves the commissioned collision-free joint cell")
                bounds = []
                for order, maximum in enumerate((s.limits.velocity[joint], s.limits.acceleration[joint], s.jerk[joint]), 1):
                    derivative = np.polynomial.polynomial.polyder(power, order)/dt**order
                    lo, hi = _bernstein_bounds(derivative)
                    bound = max(abs(lo), abs(hi))
                    if bound > maximum:
                        raise MotionError("Continuous joint derivative bound exceeded (order %d)" % order)
                    bounds.append(bound)
                velocity_bounds.append(bounds[0])
            speed = sum(radius*velocity for radius,velocity in zip(s.maximum_link_lever_m, velocity_bounds))
            if speed > min(s.max_link_speed_m_s, s.max_tool_speed_m_s):
                raise MotionError("Conservative full-assembly Cartesian speed exceeds commissioned limit")
        return True


class CommissioningAdmission:
    """One in-memory certificate tied to the exact final ROS trajectory.

    Serialized optimizer output never carries a trusted permit. The caller must
    run this verifier locally, keep the profile/cell assets current, and obtain
    fresh stationary acquisition evidence before handing over the trajectory.
    """
    def __init__(self, settings, *, clock=time.monotonic):
        self.settings, self.clock = settings, clock
        self.validator = TrajectoryValidator(settings.limits, CommissionedCellVerifier(settings))
        self.certificate = None
        self.revoked = False

    def validate(self, trajectory):
        if self.certificate is not None or self.revoked:
            raise MotionError("Commissioning admission is single-use")
        trajectory = canonical_ros_trajectory(trajectory)
        self.settings.unchanged()
        if trajectory.duration_s > self.settings.transport.maximum_trajectory_s or len(trajectory.points) > 100000:
            raise MotionError("Commissioning trajectory exceeds bounded validation work")
        duration = trajectory.duration_s + self.settings.transport.send_timeout_s + self.settings.transport.result_slack_s + self.settings.transport.stop_timeout_s
        if time.time()+duration >= self.settings.evidence_expires_unix_s:
            raise MotionError("Cell evidence lifetime cannot cover execution and stopping")
        self.certificate = self.validator.validate(trajectory, dict(self.settings.dependencies),
            now=self.clock(), expires_at=self.clock()+duration)
        return self.certificate

    def check(self, permit, trajectory):
        if self.revoked or self.certificate is None or permit is not self.certificate or trajectory.digest != permit.trajectory_digest:
            raise MotionError("Unknown, revoked or changed commissioning trajectory permit")
        self.settings.unchanged()
        self.validator.check_certificate(permit, dict(self.settings.dependencies), self.clock())
        return True


def commissioning_status(path, profile_id=None):
    """Read-only status, useful before installing/launching any robot driver."""
    config = strict_loads(Path(path).read_bytes())
    result = {"hardware_motion_enabled": config.get("hardware_motion_enabled") is True,
              "commissioned_profiles": sorted(config.get("profiles", {})),
              "physical_adl_handlers_available": False,
              "test_scope": "supervised empty-gripper free space inside an independently certified static joint cell",
              "driver_started": False, "robot_command_sent": False}
    if profile_id:
        try:
            load_settings(path, profile_id)
            result["profile_status"] = "configuration_admissible; live feedback, ownership and trajectory validation still required"
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result["profile_status"] = "unavailable"
            result["reason"] = str(exc)
    return result


def profile_template(commissioning_path):
    """An intentionally inadmissible fragment, derived from canonical fields."""
    config = strict_loads(Path(commissioning_path).read_bytes())
    profile = {name: None for name in config["required_profile_fields"]}
    profile["allowed_contacts"] = []
    profile["free_space_test"] = {
        "simulation_only": False, "payload_kg": 0, "environment_mode": "secured_static_test_cell",
        "feedback": {name:None for name in FeedbackBounds.__dataclass_fields__},
        "transport": {name:None for name in TransportBounds.__dataclass_fields__},
        **{name:None for name in ("watchdog_timeout_s", "collision_free_cell_lower_rad", "collision_free_cell_upper_rad",
            "stopping_joint_excursion_rad", "joint_position_uncertainty_rad", "joint_lower_rad", "joint_upper_rad",
            "max_joint_jerk_rad_s3", "all_assembly_point_lever_bounds_m", "extension_build_id", "planner_loaded_robot_config_digest")},
        "assets": {name:{"path":None,"sha256":None} for name in
                   ("cell_evidence", "robot_urdf", "planner_robot_config", "planner_config")}}
    evidence = {"scope":"independently_reviewed_physical_joint_cell", "profile_id":None,
        "coverage":["kinova_gen3_7dof", "robotiq_2f85_all_apertures", "wrist_d405_and_mount",
                    "all_joint_combinations_in_cell", "secured_static_environment", "empty_payload"],
        **{name:None for name in ("reviewer", "method", "expires_unix_s", "collision_free_cell_lower_rad",
                                 "collision_free_cell_upper_rad", "robot_urdf_digest", "world_digest")}, "records":[]}
    return {"profile_fragment":profile, "cell_evidence_template":evidence,
            "note":"Nulls are deliberate missing measurements; this template cannot authorize motion. See docs/hardware-testing.md."}
