"""The single ROS surface for sheppy's arm module, used as a client.

Two containers answer it. The kinova-gen3-ros2 driver executes
(/execute_joint_trajectory), reports (/joint_states, gripper knuckle
included) and takes gripper commands on /setpoint/gripper. The RAMMP-CuRobo
planner plans (/rammp_curobo/plan_to_pose, plan_to_joints, set_world) and
moves nothing. TF comes from kinova_gen3_description's robot_state_publisher.

This runtime owns neither container. It starts no driver, builds no driver,
and holds no commissioning profile of its own: the gates below are the ones
the driver does not apply before it moves. Its ROS layer checks only that each
point carries seven positions, and its executor commands the first waypoint
from wherever the arm actually is (kinova-gen3-driver v1.0.0
trajectory_executor.cpp; kinova-gen3-ros2 ros2_backend.cpp). So the
start-state gap, position and velocity limits and continuity are checked here,
in front of every goal, and nothing is sent unless motion was armed explicitly.
"""
from __future__ import annotations

import math
from numbers import Real

from .rolling import JointTrajectory, MotionError


#: The driver's joint order on the wire, and the names /joint_states reports.
JOINTS = tuple(f"joint_{index}" for index in range(1, 8))
EXECUTE_ACTION = "/execute_joint_trajectory"
GRIPPER_SETPOINT_TOPIC = "/setpoint/gripper"
GRIPPER_STATE_TOPIC = "/gripper_state"
JOINT_STATE_TOPIC = "/joint_states"
PLANNER_NAMESPACE = "/rammp_curobo"
PLAN_TO_POSE = PLANNER_NAMESPACE+"/plan_to_pose"
PLAN_TO_JOINTS = PLANNER_NAMESPACE+"/plan_to_joints"
SET_WORLD = PLANNER_NAMESPACE+"/set_world"

#: Both containers build on rammp-base, which selects Cyclone; a Fast DDS
#: shell discovers them and then loses their data.
REQUIRED_RMW = "rmw_cyclonedds_cpp"

#: URDF joint velocity limits, in the driver's joint order.
JOINT_VMAX = (1.396, 1.396, 1.396, 1.396, 1.222, 1.222, 1.222)
#: Declared position ranges from the inspected driver URDF
#: (artifacts/jetson/passive-commissioning/driver-model.urdf). Joints 1, 3, 5
#: and 7 are continuous there and carry no bound; None means exactly that.
JOINT_POSITION_LIMITS = (None, (-2.41, 2.41), None, (-2.66, 2.66), None, (-2.23, 2.23), None)
VELOCITY_SLACK = 1.01
CONTINUITY_SLACK = 3.0
#: Largest wrap-aware start gap that may be sent; beyond it the driver would
#: jump to its first waypoint instead of continuing from where the arm is.
START_GATE_RAD = 0.05
#: The planner's pose target, tool_frame, sits this far along end_effector_link z.
#: Evidence: the container's kinova_gen3_7dof.urdf tool_frame_joint (xyz 0 0 0.120)
#: and rammp_curobo/configs/robot_gen3_2f85.yaml (ee_link: tool_frame).
TOOL_FRAME_FROM_FLANGE_M = 0.120
#: /joint_states reports the Robotiq knuckle as GripperSetpoint.position x 0.8.
#: Everything here stays in knuckle radians; only the wire value is normalized.
KNUCKLE_CLOSED_RAD = 0.8

#: rammp_arm_interfaces/action/ExecuteJointTrajectory Result.error_code, exactly
#: as the pinned definition declares them; no other code is named here.
RESULT_NAMES = {0: "SUCCESSFUL", -1: "INVALID_GOAL", -4: "PATH_TOLERANCE_VIOLATED",
                -5: "GOAL_TOLERANCE_VIOLATED", -6: "PREEMPTED", -8: "NOT_AUTHORIZED", -9: "HALTED"}
SUCCESSFUL = 0

SEMANTICS = ("client-side gates in front of a driver that applies none of them; "
             "not a commissioning profile and not a safety certification")


class SheppyClientError(MotionError):
    """A goal that must not be sent, or a surface that is not answering."""


def wrap_diff(first, second):
    """Shortest signed angular difference, so a wrap is not a large gap."""
    for value in (first, second):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise SheppyClientError("Joint angles must be finite numbers")
    return (float(first)-float(second)+math.pi) % (2.*math.pi) - math.pi


def rmw_refusal(environ):
    """Why this process must not talk to the containers, or None."""
    actual = (environ.get("RMW_IMPLEMENTATION") or "").strip()
    if actual != REQUIRED_RMW:
        return (f"RMW_IMPLEMENTATION is {actual or 'unset'}; the arm containers speak "
                f"{REQUIRED_RMW}, and another middleware discovers them and then loses their data")
    return None


def start_gap_rad(live, trajectory):
    """Largest wrap-aware distance from the live arm to the first waypoint."""
    if not isinstance(trajectory, JointTrajectory) or not trajectory.points:
        raise SheppyClientError("A non-empty JointTrajectory is required")
    first = trajectory.points[0].state.position
    if len(live) != len(first):
        raise SheppyClientError("Live joint count does not match the trajectory")
    return max(abs(wrap_diff(a, b)) for a, b in zip(live, first))


def executor_problems(trajectory, vmax=JOINT_VMAX, *, continuity_slack=CONTINUITY_SLACK,
                      position_limits=JOINT_POSITION_LIMITS):
    """Gates the driver's executor does not apply: position, velocity, continuity.

    Monotonic, zero-based timing is already a JointTrajectory invariant, so it
    is not re-checked here; a trajectory that violated it could not be built.

    TODO: confirm against driver - the first point carries time_from_start 0.
    The planner-side executor this driver replaced refused a zero leading
    interval, and whether kinova-gen3-ros2 commands that waypoint immediately
    or treats it as the start state is unconfirmed. The start gate keeps that
    waypoint within START_GATE_RAD of the live arm either way, so the hazard
    is bounded while the question is open.
    """
    problems = []
    if len(position_limits) != len(vmax):
        raise SheppyClientError("Position and velocity limit vectors disagree in length")
    for point in trajectory.points:
        velocity = point.state.velocity
        if len(velocity) != len(vmax) or len(point.state.position) != len(vmax):
            return ["trajectory joint dimensions disagree with the limits"]
        if any(abs(value) > limit*VELOCITY_SLACK for value, limit in zip(velocity, vmax)):
            problems.append("velocity exceeds limit")
            break
    for point in trajectory.points:
        if any(bounds is not None and not bounds[0] <= float(q) <= bounds[1]
               for q, bounds in zip(point.state.position, position_limits)):
            problems.append("position exceeds a declared joint range")
            break
    previous = None
    for point in trajectory.points:
        if previous is not None:
            step = point.time_s-previous.time_s
            if any(abs(wrap_diff(a, b)) > continuity_slack*limit*step
                   for a, b, limit in zip(point.state.position, previous.state.position, vmax)):
                problems.append("discontinuity")
                break
        previous = point
    return sorted(set(problems))


# RAMMP-CuRobo v1.0.0 fills positions and velocities and never accelerations
# (rammp_curobo_ros conversions.py); the driver interpolates from those two.
# The canonical type wants an acceleration for its own sampling, so it is
# differenced from the planner's velocity profile and the wire message stays
# exactly what the planner gave.
DIFFERENCED = "accelerations differenced from the planner's velocities"
# It also stamps waypoint k at (k+1)*dt: waypoint 0 is the start state, one step
# in the future (conversions.py). The canonical form begins at zero, so the
# same start state is placed there at rest; the wire message leaves it out.
START_PREPENDED = "start state placed at t=0"


def trajectory_from_planner(joint_names, points, *, provenance):
    """Build the canonical trajectory from planner output.

    This is the wire boundary, and the only place a missing velocity profile
    can appear: trajectory_msgs allows empty velocities, the canonical type
    does not, and the driver would fly such a path linearly and unchecked.
    Refusing here keeps that hazard from being silently filled with zeros.
    Accelerations are optional on the wire; all waypoints carry them or none do.
    """
    if not isinstance(provenance, str) or not provenance.strip():
        raise SheppyClientError("Planner provenance is required")
    names = tuple(joint_names)
    if names != JOINTS:
        raise SheppyClientError(f"Planner returned joint order {names}, not the driver's {JOINTS}")
    rows = list(points)
    if len(rows) < 2:
        raise SheppyClientError("Planner returned fewer than two waypoints")
    built = []
    for index, row in enumerate(rows):
        try:
            time_s, position, velocity, acceleration = row
        except (TypeError, ValueError) as exc:
            raise SheppyClientError(f"Waypoint {index} is malformed") from exc
        if velocity is None or len(tuple(velocity)) != len(names):
            raise SheppyClientError(
                f"Waypoint {index} has no velocity profile; the driver would fly this "
                "trajectory linearly and unchecked, so it is refused rather than zero-filled")
        if acceleration is not None and len(tuple(acceleration)) != len(names):
            raise SheppyClientError(f"Waypoint {index} has a malformed acceleration profile")
        built.append([float(time_s), tuple(float(v) for v in position), tuple(float(v) for v in velocity),
                      None if acceleration is None else tuple(float(v) for v in acceleration)])
    prepended = built[0][0] > 0.
    if prepended:
        built.insert(0, [0., built[0][1], (0.,)*len(names), None if built[0][3] is None else (0.,)*len(names)])
        provenance = f"{provenance}; {START_PREPENDED}"
    given = [row[3] is not None for row in built]
    if any(given) and not all(given):
        raise SheppyClientError("Planner gave accelerations for some waypoints only")
    if not any(given):
        for index, row in enumerate(built):
            a, b = built[max(0, index-1)], built[min(len(built)-1, index+1)]
            span = b[0]-a[0]
            if span <= 0.:
                raise SheppyClientError("Planner trajectory times do not increase")
            row[3] = tuple((vb-va)/span for va, vb in zip(a[2], b[2]))
        provenance = f"{provenance}; {DIFFERENCED}"
    if built[0][0] != 0.:
        raise SheppyClientError(f"Planner trajectory starts at t={built[0][0]}")
    from .rolling import JointState, TrajectoryPoint
    return JointTrajectory(names, tuple(
        TrajectoryPoint(time_s, JointState(position, velocity, acceleration))
        for time_s, position, velocity, acceleration in built), provenance)


def scale_trajectory_time(trajectory, factor):
    """The same path flown `factor` times slower: times stretch, rates shrink.

    The waypoints are the planner's; only the clock changes, so every
    position gate still holds and the velocity and continuity gates loosen.
    """
    if isinstance(factor, bool) or type(factor) not in (int, float) or not math.isfinite(factor) or factor < 1.:
        raise SheppyClientError("a time scale slows a trajectory; it must be a number of at least 1")
    if factor == 1.:
        return trajectory
    from .rolling import JointState, JointTrajectory, TrajectoryPoint
    points = tuple(TrajectoryPoint(point.time_s*factor,
                                   JointState(point.state.position,
                                              tuple(v/factor for v in point.state.velocity),
                                              tuple(a/(factor*factor) for a in point.state.acceleration)))
                   for point in trajectory.points)
    return JointTrajectory(trajectory.joint_names, points, f"{trajectory.provenance}; time scaled x{factor:g}")


def reversed_trajectory(trajectory):
    """The same validated path flown backwards: the way in was clear, so it is the way out.

    Waypoints reverse, the clock mirrors, velocities change sign and
    accelerations keep theirs. Nothing new is planned, so every gate that
    admitted the path still describes it.
    """
    from .rolling import JointState, JointTrajectory, TrajectoryPoint
    end = trajectory.duration_s
    points = tuple(TrajectoryPoint(end-point.time_s, JointState(point.state.position,
                                                               tuple(-v for v in point.state.velocity),
                                                               point.state.acceleration))
                   for point in reversed(trajectory.points))
    return JointTrajectory(trajectory.joint_names, points, f"{trajectory.provenance}; reversed")


def refusal(trajectory, live, vmax=JOINT_VMAX, *, start_gate_rad=START_GATE_RAD,
            position_limits=JOINT_POSITION_LIMITS):
    """Why this trajectory must not be sent from the live arm, or None."""
    if not isinstance(trajectory, JointTrajectory):
        raise SheppyClientError("A JointTrajectory is required")
    if not trajectory.points:
        return "empty trajectory"
    if tuple(trajectory.joint_names) != JOINTS:
        return (f"trajectory joint order {tuple(trajectory.joint_names)} is not the driver's "
                f"{JOINTS}; the driver maps by array order")
    gap = start_gap_rad(live, trajectory)
    if gap > start_gate_rad:
        return (f"trajectory starts {gap:.3f} rad from the live arm (gate {start_gate_rad:.2f}); "
                "the driver would jump to its first waypoint, so re-plan from the current state")
    problems = executor_problems(trajectory, vmax, position_limits=position_limits)
    return "; ".join(problems) if problems else None


def setpoint_from_knuckle(knuckle_rad):
    """Normalize a knuckle angle onto the driver's 0..1 gripper setpoint."""
    if (isinstance(knuckle_rad, bool) or type(knuckle_rad) not in (int, float)
            or not math.isfinite(knuckle_rad) or not 0. <= knuckle_rad <= KNUCKLE_CLOSED_RAD):
        raise SheppyClientError(f"Knuckle target must be between 0 and {KNUCKLE_CLOSED_RAD} rad")
    return float(knuckle_rad)/KNUCKLE_CLOSED_RAD


def knuckle_from_setpoint(position):
    if (isinstance(position, bool) or type(position) not in (int, float)
            or not math.isfinite(position) or not 0. <= position <= 1.):
        raise SheppyClientError("Gripper setpoint must be between 0 and 1")
    return float(position)*KNUCKLE_CLOSED_RAD


def result_message(error_code, error_string=""):
    """A driver result code, readable, with the driver's own reason."""
    name = RESULT_NAMES.get(error_code, f"UNKNOWN({error_code})")
    return f"{name}: {error_string}" if error_string else name


def describe_surface():
    """What this runtime expects to find, for a readiness report."""
    return {"planner_actions": [PLAN_TO_POSE, PLAN_TO_JOINTS], "planner_service": SET_WORLD,
            "driver_action": EXECUTE_ACTION,
            "driver_topics": [JOINT_STATE_TOPIC, GRIPPER_STATE_TOPIC],
            "gripper_topic": GRIPPER_SETPOINT_TOPIC, "required_rmw": REQUIRED_RMW,
            "joint_order": list(JOINTS), "owns_containers": False,
            "starts_driver": False, "semantics": SEMANTICS}
