"""Retained cuRobo -> world-bound gateway -> actual MuJoCo software rehearsal.

This module never imports ROS or connects to a robot driver. Its Python driver
port emulates action acknowledgements while position actuators track the exact
cuRobo joint curve in MuJoCo. All numeric transport bounds are software fixtures;
they cannot create physical commissioning evidence or enable catalog skills.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace as NS

import numpy as np

from ..contracts import Catalog, digest, strict_loads
from ..handlers import ExecutionContext
from ..hardware_backend import CommissioningArmSession
from ..resources import ResourceManager
from ..simulation import MujocoReplay
from ..world import MetricPose, WorldModel
from .curobo import CUROBO_VERSION, RAMMP_COMMIT
from .curobo_process import WarmGpuPlanner
from .driver_transport import (JointTrajectoryTransport, TransferredOwnership,
                               TransportBounds, VerifiedMotionState, canonical_ros_trajectory)
from .guards import quaternion_distance
from .integration import collision_world_from_mujoco, file_digest, load_contract, model_assets_digest
from .rolling import JointLimits, JointState, JointTrajectory, MotionError, TrajectoryPoint, TrajectoryValidator
from .trial_recording import TrialRecording


CASES = ("complete", "cancel", "stale_feedback", "dependency_change", "driver_fault")


def simulation_bounds():
    """Explicit software criteria, never measured limits for the physical arm."""
    return TransportBounds(path_position_rad=(.02,)*7, goal_position_rad=(.006,)*7,
        start_position_rad=(.001,)*7, stationary_velocity_rad_s=.003,
        stationary_position_span_rad=.0005, state_max_age_s=.10, receipt_max_age_s=.10,
        poll_period_s=.005, send_timeout_s=.5, cancel_timeout_s=.2,
        stop_timeout_s=2., settle_duration_s=.08, result_slack_s=.5, maximum_trajectory_s=10.)


def _future(value):
    result = asyncio.get_running_loop().create_future()
    result.set_result(value)
    return result


class _SimulationGoal:
    accepted = True

    def __init__(self, port):
        self.port = port
        self.goal_id = NS(uuid=bytes(range(1, 17)))
        self.result = asyncio.get_running_loop().create_future()

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.port.cancel_count += 1
        self.port.recording.event("cancel_requested", {"source": "simulated driver goal cancellation"})
        self.port.hold(status=5, code=-9)
        return _future(NS(return_code=0, goals_canceling=[NS(goal_id=self.goal_id)]))


class MujocoDriverPort:
    """SIMULATION ONLY: actual physics, internal Python action/status protocol.

    Cancel/stop freezes the current actuator position setpoint and physics keeps
    running. Neither q nor dq is overwritten to manufacture measured stopping.
    An action terminal result is intentionally independent of measured settling.
    """
    hardware_commands = False
    ros_transport = False

    def __init__(self, simulation):
        if not isinstance(simulation, MujocoReplay):
            raise MotionError("Rehearsal port requires the local MuJoCo model")
        self.simulation = simulation
        self.recording = TrialRecording(simulation=True)
        self.mj, self.model, self.data = simulation.mujoco, simulation.model, simulation.data
        simulation.home()
        self.data.ctrl[:] = self.data.qpos
        self.handle = self.path = self.latest = None
        self.sequence = self.sent = self.cancel_count = self.release_count = 0
        self.stop_reasons, self.samples = [], []
        self.feedback_frozen = False
        self.maximum_tracking_error_rad = self.maximum_velocity_rad_s = 0.
        self.maximum_contact_count = 0
        self.started_at = None
        self._task = None
        self._closing = False
        self._fault = ""
        self._bounds = None

    def start(self):
        if self._task is not None:
            raise MotionError("MuJoCo driver port is single-use")
        self._task = asyncio.create_task(self._pump())

    async def _pump(self):
        began = time.monotonic()
        dt = float(self.model.opt.timestep)
        try:
            while not self._closing:
                elapsed = time.monotonic()-began
                if elapsed-float(self.data.time) > .5:
                    raise MotionError("MuJoCo software fixture missed its bounded real-time stepping budget")
                while float(self.data.time)+dt <= elapsed:
                    active = self.path is not None and not self.handle.result.done()
                    if active:
                        path_time = min(float(self.data.time)-self.started_at, self.path.duration_s)
                        self.data.ctrl[:] = self.path.sample(max(0., path_time)).position
                    self.mj.mj_step(self.model, self.data)
                    if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                        raise MotionError("MuJoCo produced non-finite measured state")
                    self.sequence += 1
                    acquired = time.monotonic()
                    if not self.feedback_frozen:
                        self.latest = VerifiedMotionState(tuple(map(float, self.data.qpos)),
                            tuple(map(float, self.data.qvel)), acquired, acquired, self.sequence,
                            "mujoco-simulation-step")
                    self.maximum_velocity_rad_s = max(self.maximum_velocity_rad_s, float(np.max(np.abs(self.data.qvel))))
                    self.maximum_contact_count = max(self.maximum_contact_count, self.data.ncon)
                    if active:
                        error = float(np.max(np.abs(self.data.qpos-self.data.ctrl)))
                        self.maximum_tracking_error_rad = max(self.maximum_tracking_error_rad, error)
                        if self.data.ncon or any(abs(q-c) > b for q,c,b in
                            zip(self.data.qpos, self.data.ctrl, self._bounds.path_position_rad)):
                            self.hold(status=6, code=-4)
                        elif float(self.data.time)-self.started_at >= self.path.duration_s:
                            self.data.ctrl[:] = self.path.points[-1].state.position
                            self.hold(status=4, code=0)
                    if self.sequence % 10 == 0:
                        self.samples.append({"time_s": float(self.data.time), "q": self.data.qpos.tolist(),
                                             "dq": self.data.qvel.tolist(), "contact_count": self.data.ncon})
                await asyncio.sleep(min(dt/2., .002))
        except Exception as exc:
            self._fault = str(exc)
            self.hold(status=6, code=-10)

    def server_ready(self):
        return self._task is not None and not self._task.done() and not self._fault

    def state(self, now):
        if self._fault:
            raise MotionError(self._fault)
        if self.latest is None:
            raise MotionError("No simulated acquisition yet")
        return self.recording.sample(self.latest)

    async def stationary(self, bounds):
        anchor = latest_sequence = None
        minimum = maximum = None
        deadline = time.monotonic()+bounds.stop_timeout_s
        while time.monotonic() < deadline:
            sample = self.latest
            if sample is not None and sample.sequence != latest_sequence:
                latest_sequence = sample.sequence
                if (time.monotonic()-sample.acquired_at_monotonic_s > bounds.state_max_age_s
                        or max(map(abs, sample.velocity_rad_s)) > bounds.stationary_velocity_rad_s):
                    anchor = None
                elif anchor is None:
                    anchor = sample.acquired_at_monotonic_s
                    minimum = maximum = sample.position_rad
                else:
                    minimum = tuple(min(a,b) for a,b in zip(minimum, sample.position_rad))
                    maximum = tuple(max(a,b) for a,b in zip(maximum, sample.position_rad))
                    if max(b-a for a,b in zip(minimum, maximum)) > bounds.stationary_position_span_rad:
                        anchor = None
                    elif sample.acquired_at_monotonic_s-anchor >= bounds.settle_duration_s:
                        return sample
            await asyncio.sleep(bounds.poll_period_s)
        raise MotionError("Simulation did not establish a fresh stationary dwell")

    def send_goal(self, trajectory, ownership, bounds):
        if self.sent or not self.server_ready() or trajectory.joint_names != self.simulation.joint_names:
            raise MotionError("Simulation port unavailable, busy or joint order mismatched")
        self.sent += 1
        self.recording.event("dispatch_attempted", {"trajectory_digest": trajectory.digest})
        self.path, self._bounds = trajectory, bounds
        self.handle = _SimulationGoal(self)
        self.started_at = float(self.data.time)
        return _future(self.handle)

    def hold(self, *, status, code):
        if self.handle is not None and not self.handle.result.done():
            self.handle.result.set_result(NS(status=status, result=NS(error_code=code)))

    def publish_stop(self, owner_id, reason):
        self.stop_reasons.append(reason)
        self.recording.event("software_stop", {"reason": reason})
        self.hold(status=6, code=-10)

    def release(self, token):
        self.release_count += 1
        self.recording.event("ownership_released")
        return _future(NS(released=True))

    async def close(self):
        self._closing = True
        if self._task is not None:
            await self._task


def checked_simulation_candidate(response, request, contract):
    """Bind a worker candidate to the reviewed, explicitly bare-arm model."""
    expected = {"status": "planned", "hardware_commands": False, "curobo_planned": True,
                "request_digest": digest(request), "world_identity": request["world_identity"],
                "planner_source_commit": RAMMP_COMMIT, "curobo_version": CUROBO_VERSION,
                "base_frame": contract["base_frame"], "ee_link": "end_effector_link"}
    expected.update({key: contract[key] for key in
                    ("planner_config_digest", "planner_urdf_digest", "planner_robot_config_digest")})
    if (not isinstance(response, dict) or response.get("hardware_commands") is not False
            or response.get("curobo_planned") is not True or any(response.get(k) != v for k,v in expected.items())
            or digest(response.get("planner_robot_config")) != contract["planner_robot_config_digest"]):
        raise MotionError("Rehearsal planner request/source/world/model provenance mismatch")
    encoded = response.get("trajectory", {})
    if (encoded.get("interpolation") != "quintic-hermite-v1"
            or not isinstance(encoded.get("points"), list) or not 2 <= len(encoded["points"]) <= 100000):
        raise MotionError("Unsupported rehearsal trajectory encoding")
    path = JointTrajectory(encoded["joint_names"], tuple(TrajectoryPoint(p["time_s"],
        JointState(p["position"], p["velocity"], p["acceleration"])) for p in encoded["points"]), encoded["provenance"])
    if path.digest != encoded["digest"] or not path.provenance.startswith("rammp_curobo:"+RAMMP_COMMIT+":"):
        raise MotionError("Rehearsal trajectory digest or planner provenance mismatch")
    if path.joint_names != tuple(contract["joint_names"]):
        raise MotionError("Rehearsal trajectory joint order mismatch")
    return path


class SimulationPlanner:
    """Async facade for one retained GPU process; no trajectory admission."""
    def __init__(self, worker, contract):
        self.worker, self.contract = worker, contract
        self.last_response = self.last_request = None

    async def plan_pose(self, *, position_m, quaternion_xyzw, start, world, world_identity):
        request = {"goal": {"position_m": position_m, "quaternion_xyzw": quaternion_xyzw},
                   "start": {"position": list(start.position), "velocity": list(start.velocity),
                             "acceleration": list(start.acceleration)}, "world": world, "world_identity": world_identity}
        response = await asyncio.to_thread(self.worker.plan, request)
        path = checked_simulation_candidate(response, request, self.contract)
        self.last_request, self.last_response = request, response
        return path


class SimulationAdmission:
    """Sampled simulation verification only; never a hardware cell certificate."""
    def __init__(self, simulation, planner, contract, bounds, world):
        self.simulation, self.planner, self.contract = simulation, planner, contract
        self.mj, self.model = simulation.mujoco, simulation.model
        self.data = self.mj.MjData(self.model)
        self.site = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, contract["tool_site"])
        if self.site < 0:
            raise MotionError("Simulation endpoint site missing")
        self.settings = NS(world_digest=digest(world), transport=bounds, tracking_reserve=bounds.path_position_rad,
                           feedback=NS(velocity_error_rad_s=0.), unchanged=self.unchanged)
        self.certificate = self.validator = None
        self.revoked = False
        self.states_checked = 0
        self.endpoint_fk_error_m = None
        self._stats = {p: p.stat() for p in simulation.simulation_root.rglob('*') if p.is_file()}

    def unchanged(self):
        if self.revoked:
            raise MotionError("Simulation evidence invalidated")
        for p,s in self._stats.items():
            now = p.stat()
            if (s.st_size, s.st_mtime_ns, s.st_ctime_ns) != (now.st_size, now.st_mtime_ns, now.st_ctime_ns):
                raise MotionError("Simulation model asset changed")

    def pose(self, state):
        self.data.qpos[:] = state.position
        self.data.qvel[:] = state.velocity
        self.mj.mj_forward(self.model, self.data)
        wxyz = np.empty(4)
        self.mj.mju_mat2Quat(wxyz, self.data.site_xmat[self.site])
        return self.data.site_xpos[self.site].copy(), np.asarray([*wxyz[1:], wxyz[0]])

    def _geometry(self, path, dependencies):
        steps = max(1, math.ceil(path.duration_s/self.contract["validation_sample_dt_s"]))
        if steps > 100000:
            raise MotionError("Simulation validation budget exceeded")
        for i in range(steps+1):
            state = path.sample(path.duration_s*i/steps)
            position, _ = self.pose(state)
            if (np.any(position < self.contract["workspace_lower_m"])
                    or np.any(position > self.contract["workspace_upper_m"])):
                raise MotionError("Simulation endpoint leaves reviewed workspace")
            for joint in range(self.model.njnt):
                if self.model.jnt_limited[joint] and not self.model.jnt_range[joint,0] <= state.position[joint] <= self.model.jnt_range[joint,1]:
                    raise MotionError("Simulation joint limit exceeded")
            if any(c.dist < -1e-7 for c in self.data.contact):
                raise MotionError("Simulation candidate has unpermitted penetration")
            self.states_checked += 1
        return True

    def validate(self, path):
        if self.certificate is not None:
            raise MotionError("Simulation admission is single-use")
        self.unchanged()
        path = canonical_ros_trajectory(path)
        if path.duration_s > self.settings.transport.maximum_trajectory_s:
            raise MotionError("Simulation trajectory duration exceeds fixture bound")
        response = self.planner.last_response
        pos = np.asarray(response["joint_limits"]["position"])
        vel = np.asarray(response["joint_limits"]["velocity"])
        if pos.shape != (2,7) or vel.shape != (7,) or not np.isfinite(pos).all() or not np.isfinite(vel).all():
            raise MotionError("Malformed planner joint limits")
        endpoint, quat = self.pose(path.points[-1].state)
        fk = response["endpoint_fk"]
        self.endpoint_fk_error_m = float(np.linalg.norm(endpoint-fk["position_m"]))
        if (self.endpoint_fk_error_m > self.contract["fk_position_tolerance_m"]
                or quaternion_distance(tuple(quat), tuple(fk["quaternion_xyzw"])) > self.contract["fk_rotation_tolerance_rad"]):
            raise MotionError("cuRobo/MuJoCo endpoint model mismatch")
        goal = self.planner.last_request["goal"]
        if (np.linalg.norm(endpoint-goal["position_m"]) > self.contract["goal_position_tolerance_m"]
                or quaternion_distance(tuple(quat), tuple(goal["quaternion_xyzw"])) > self.contract["goal_rotation_tolerance_rad"]):
            raise MotionError("Planned endpoint misses simulation goal")
        self.validator = TrajectoryValidator(JointLimits(tuple(map(float,pos[0])), tuple(map(float,pos[1])),
            tuple(map(float, np.minimum(vel,self.contract["joint_velocity_rad_s"]))),
            tuple(self.contract["joint_acceleration_rad_s2"])), self._geometry,
            sample_dt_s=self.contract["validation_sample_dt_s"])
        self.certificate = self.validator.validate(path, {"model": self.contract["mujoco_assets_digest"],
            "world": self.settings.world_digest}, now=time.monotonic(), expires_at=time.monotonic()+30.)
        return self.certificate

    def check(self, permit, path):
        self.unchanged()
        if permit is not self.certificate or path.digest != permit.trajectory_digest:
            raise MotionError("Unknown simulation permit")
        self.validator.check_certificate(permit, {"model": self.contract["mujoco_assets_digest"],
            "world": self.settings.world_digest}, time.monotonic())
        return True


def simulation_world(goal, contract):
    """Explicit synthetic target evidence, never relabelled as camera evidence."""
    context = {"schema_version": "1.0.0", "task_id": "software-rehearsal", "snapshot_id": "initial",
        "execution_epoch": 1, "revision": 1, "collision_revision": 1, "base_epoch": "simulation-base",
        "calibration_id": "simulation-coordinate-definition", "robot_config_id": contract["mujoco_assets_digest"],
        "attachment_id": "bare-arm", "grasp_state_id": "empty", "constraints": [], "profiles": [],
        "available_skills": [], "hazards": [], "completed_nodes": [],
        "goal": {"predicate": "at_pose", "args": {"entity_id": "simulation_target", "pose_role": "staging"}},
        "entities": [{"entity_id": "simulation_target", "label": "explicit software-test goal", "entity_revision": 1,
            "pose_roles": ["staging"], "facts": [], "confidence": 1., "age_s": 0.,
            "position_validity": "unknown", "orientation_validity": "unknown", "source_ids": ["simulation-input"]}]}
    world = WorldModel(context, Catalog(), max_evidence_age_s=180.)
    source = world.authorize_source("synthetic-rehearsal-target", ["pose_valid"])
    now = world.clock()
    args = {"entity_id": "simulation_target", "pose_role": "staging"}
    world.register_evidence("simulation-input", source=source, predicates=[
        {"predicate": "pose_valid", "args": args, "validity": "true"}], ttl_s=180., observed_at=now)
    pose = MetricPose("simulation_target", "staging", tuple(goal["position_m"]), tuple(goal["quaternion_xyzw"]),
        (0.,)*36, now, "base_link", 2, context["calibration_id"], context["base_epoch"], "simulation-input", 180.)
    world.update_metric_pose(pose, source=source)
    return world


async def run_case(*, planner, contract, goal, case, output):
    if case not in CASES:
        raise MotionError("Unknown rehearsal fault case")
    simulation = MujocoReplay(Catalog().root/"simulation/rolling_scene.xml")
    port = MujocoDriverPort(simulation)
    bounds = simulation_bounds()
    collision_world = collision_world_from_mujoco(simulation, contract)
    world = simulation_world(goal, contract)
    admission = SimulationAdmission(simulation, planner, contract, bounds, collision_world)
    context = ExecutionContext("software-rehearsal", "move-"+case, execution_epoch=1, snapshot=world.snapshot())
    claims = tuple(world.catalog.skills["move_to_pose"]["claims"])
    resources = ResourceManager(claims)
    if not resources.try_acquire(context.node_id, claims):
        raise MotionError("Simulation motion resources unavailable")
    def command_check(ctx):
        resources.assert_owned(ctx.node_id, claims)
        return True
    session = CommissioningArmSession(world, planner, admission,
        stationary_check=lambda: port.stationary(bounds), collision_world=collision_world,
        command_check=command_check)
    owner = TransferredOwnership("mujoco-rehearsal-only", bytes(range(1,17)), 1)
    gateway = JointTrajectoryTransport(port=port, ownership=owner, admission_check=session.admission_check,
                                      state_check=port.state, bounds=bounds)
    gateway.update_control_status(NS(arbitration_enabled=True, estopped=False, owned=True,
                                     owner_id=owner.owner_id, generation=owner.generation))
    session.gateway = gateway
    port.start()
    inject = None
    began = time.monotonic()
    try:
        port.recording.event("planning_started")
        prepared = await session.prepare("simulation_target", "staging", context)
        port.recording.event("candidate_prepared", {"trajectory_digest": prepared.trajectory.digest})
        async def inject_fault():
            while port.started_at is None:
                await asyncio.sleep(.001)
            threshold = port.started_at+min(.12, prepared.trajectory.duration_s/3.)
            while float(port.data.time) < threshold:
                await asyncio.sleep(.002)
            if case == "cancel": context.cancel_event.set()
            elif case == "stale_feedback": port.feedback_frozen = True
            elif case == "dependency_change": world.cancel_epoch()
            elif case == "driver_fault": port.hold(status=6, code=-10)
        if case != "complete": inject = asyncio.create_task(inject_fault())
        receipt = await session.execute(prepared, context)
        port.recording.event("transport_receipt", asdict(receipt))
        if receipt.status == "succeeded" and case != "complete":
            raise MotionError("Injected rehearsal fault incorrectly completed successfully")
        if case == "complete" and receipt.status != "succeeded":
            raise MotionError("Rehearsal completion failed: "+receipt.reason)
        if case == "cancel" and (receipt.status != "cancelled" or not receipt.measured_quiescent):
            raise MotionError("Cancellation lacked measured settling")
        if case == "stale_feedback" and (receipt.release_permitted or receipt.measured_quiescent
                                          or not receipt.software_stop_published):
            raise MotionError("Stale feedback incorrectly established stopped state/handback")
        if case in {"dependency_change", "driver_fault"} and not receipt.measured_quiescent:
            raise MotionError("Fault did not establish simulated stopping")
        if receipt.release_permitted:
            await gateway.release()
            resources.release(context.node_id)
        state = JointState(tuple(map(float,port.data.qpos)), tuple(map(float,port.data.qvel)), (0.,)*7)
        actual, quat = admission.pose(state)
        actual_error = float(np.linalg.norm(actual-goal["position_m"]))
        actual_rotation_error = quaternion_distance(tuple(quat), tuple(goal["quaternion_xyzw"]))
        if case == "complete" and (actual_error > contract["goal_position_tolerance_m"]
                                   or actual_rotation_error > contract["goal_rotation_tolerance_rad"]):
            raise MotionError("Measured MuJoCo endpoint misses the requested goal")
        if port.maximum_contact_count:
            raise MotionError("Unexpected contact in the bare-arm free-space rehearsal")
        if port.maximum_tracking_error_rad > contract["maximum_tracking_error_rad"]:
            raise MotionError("Measured tracking exceeds reviewed simulation criterion")
        return {"case": case, "passed": True, "receipt": asdict(receipt),
            "gateway": "JointTrajectoryTransport", "binding": "CommissioningArmSession",
            "simulated_action_count": port.sent, "cancel_count": port.cancel_count, "release_count": port.release_count,
            "resources_retained": resources.owners,
            "software_stop_reasons": port.stop_reasons, "physics_steps": port.sequence,
            "maximum_tracking_error_rad": port.maximum_tracking_error_rad,
            "maximum_velocity_rad_s": port.maximum_velocity_rad_s, "maximum_contact_count": port.maximum_contact_count,
            "actual_goal_position_error_m": actual_error, "actual_goal_rotation_error_rad": actual_rotation_error,
            "endpoint_fk_error_m": admission.endpoint_fk_error_m, "states_checked": admission.states_checked,
            "planning_wall_s": planner.last_response["planning_wall_s"], "case_wall_s": time.monotonic()-began,
            "trial_directory": case+"-trial",
            "motion_sent_to_physical_robot": False, "target_source": "explicit simulation input"}
    finally:
        if inject is not None:
            inject.cancel()
            await asyncio.gather(inject, return_exceptions=True)
        await port.close()
        Path(output).write_text(json.dumps({"simulation_only": True, "samples": port.samples}, indent=2)+"\n")
        port.recording.write(Path(output).with_name(case+"-trial"))


async def run_rehearsal(args):
    if len(set(args.cases)) != len(args.cases):
        raise MotionError("Rehearsal cases must be unique")
    contract = load_contract(args.contract)
    if (model_assets_digest(Catalog().root/"simulation") != contract["mujoco_assets_digest"]
            or file_digest(args.planner_config) != contract["planner_config_digest"]):
        raise MotionError("Reviewed rehearsal model/config assets differ")
    goal = strict_loads(Path(args.goal).read_bytes())
    if not isinstance(goal, dict) or set(goal) != {"position_m", "quaternion_xyzw"}:
        raise MotionError("Rehearsal goal requires position_m and quaternion_xyzw only")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    worker = WarmGpuPlanner(planner_config=args.planner_config, model_dir=args.model_dir,
        wrapper_checkout=args.wrapper_checkout, gpu_cache=args.gpu_cache, output=output/"planner")
    results = []
    report = {"mode": "curobo_world_gateway_mujoco_rehearsal", "simulation_only": True,
        "hardware_commands": False, "hardware_validated": False, "ros_transport_tested": False,
        "camera_grounding_tested": False, "astra_tested": False, "physical_adl_available": False,
        "limitations": ["Bare arm: no installed D405/bracket/gripper or ADL contact physics",
                        "Sampled collision checks do not prove swept or stopping volumes",
                        "Static cuRobo moves; no rolling replacement",
                        "Synthetic target and Python driver port; no live camera, Astra or ROS command path"],
        "transport_criteria": asdict(simulation_bounds()), "cases": results}
    try:
        report["planner_ready"] = await asyncio.to_thread(worker.start)
        planner = SimulationPlanner(worker, contract)
        for case in args.cases:
            result = await run_case(planner=planner, contract=contract, goal=goal, case=case, output=output/(case+"-physics.json"))
            results.append(result)
            (output/(case+".json")).write_text(json.dumps(result, indent=2)+"\n")
        report.update(passed=True, curobo_planned=True, planner_initializations=1)
        return report
    except Exception as exc:
        report.update(passed=False, reason=str(exc))
        raise
    finally:
        try:
            await asyncio.to_thread(worker.close)
        except Exception as exc:
            report.update(passed=False, reason="GPU worker teardown unresolved: "+str(exc))
            raise
        finally:
            (output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wrapper-checkout", required=True)
    parser.add_argument("--gpu-cache", default="/tmp/rammp-rehearsal-gpu-cache")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--goal", required=True, help="Local simulation goal JSON, position_m and quaternion_xyzw")
    parser.add_argument("--output", required=True, help="New local artifact directory")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(asyncio.run(run_rehearsal(args)), indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"passed": False, "reason": str(exc), "hardware_commands": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
