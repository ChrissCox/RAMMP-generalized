"""Supervised cuRobo free-space test composition, disabled without commissioning.

The CLI never starts a driver. Default operation only prepares/reviews a path;
--execute additionally claims an unowned instrumented driver and sends it after
fresh local admission. This is distinct from fixture ADL task execution.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import re
from pathlib import Path
import signal
import threading
import time
import uuid

from ..contracts import Catalog, digest, strict_loads
from .commissioning import CommissioningAdmission, load_settings, measured_transport_bounds, sha256
from .curobo import RAMMP_COMMIT, CUROBO_VERSION
from .driver_feedback import DriverFeedbackBuffer, RosDriverFeedback
from .driver_transport import RosJointTrajectoryTransport, TransferredOwnership
from .leasing import drain_nonpreemptible
from .rolling import JointState, JointTrajectory, TrajectoryPoint, MotionError
from .curobo_process import REVIEWED_GPU_IMAGE, WarmGpuPlanner
from .trial_recording import TrialRecording


def run_gpu_worker(*, request, planner_config, model_dir, wrapper_checkout, gpu_cache, output):
    """Compatibility helper for one request; retained sessions use WarmGpuPlanner."""
    with WarmGpuPlanner(planner_config=planner_config, model_dir=model_dir,
                        wrapper_checkout=wrapper_checkout, gpu_cache=gpu_cache, output=output) as planner:
        response = planner.plan(request)
        if response.get("status") != "planned":
            raise MotionError("GPU planner rejected the request: "+response.get("reason", "unknown failure"))
        return response


def checked_candidate(response, request, settings, planner_config, robot_urdf):
    """Validate worker provenance before independent full trajectory admission."""
    if (response.get("status") != "planned" or response.get("request_digest") != digest(request)
            or response.get("world_identity") != settings.world_digest
            or response.get("planner_source_commit") != RAMMP_COMMIT
            or response.get("curobo_version") != CUROBO_VERSION
            or response.get("planner_config_digest") != sha256(planner_config)
            or response.get("planner_urdf_digest") != sha256(robot_urdf)
            or response.get("planner_robot_config_digest") != settings.planner_model_digest
            or digest(response.get("planner_robot_config")) != settings.planner_model_digest):
        raise MotionError("GPU request, world, source or installed assembly model identity mismatch")
    encoded = response["trajectory"]
    if encoded.get("interpolation") != "quintic-hermite-v1" or not 2 <= len(encoded["points"]) <= 100000:
        raise MotionError("Unsupported planner interpolation/point count")
    trajectory = JointTrajectory(tuple(encoded["joint_names"]), tuple(
        TrajectoryPoint(p["time_s"], JointState(tuple(p["position"]), tuple(p["velocity"]), tuple(p["acceleration"])))
        for p in encoded["points"]), encoded["provenance"])
    if trajectory.digest != encoded["digest"]:
        raise MotionError("Worker trajectory digest mismatch")
    return trajectory


class CheckedGpuPlanner:
    """Retained planner facade for CommissioningArmSession's plan_pose port.

    The caller starts/closes WarmGpuPlanner and owns the runtime planner lease.
    This facade validates exact response provenance, never grants motion itself.
    """
    capabilities = frozenset({'curobo_static_planning'})

    def __init__(self, worker, *, settings, robot_urdf):
        if not isinstance(worker, WarmGpuPlanner):
            raise MotionError('An explicit retained GPU worker is required')
        self.worker, self.settings, self.robot_urdf = worker, settings, Path(robot_urdf)
        self._lock = asyncio.Lock()

    async def plan_pose(self, *, position_m, quaternion_xyzw, start, world, world_identity):
        request = {'start': {'position':list(start.position), 'velocity':list(start.velocity),
                             'acceleration':list(start.acceleration)},
                   'goal': {'position_m':list(position_m), 'quaternion_xyzw':list(quaternion_xyzw)},
                   'world':world, 'world_identity':world_identity}
        async with self._lock:
            self.settings.unchanged()
            task = asyncio.create_task(asyncio.to_thread(self.worker.plan, request))
            try:
                response = await asyncio.shield(task)
            except asyncio.CancelledError:
                await drain_nonpreemptible(task)
                raise
            self.settings.unchanged()
            return checked_candidate(response, request, self.settings, self.worker.planner_config, self.robot_urdf)


def _require_not_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise MotionError('Operator cancelled; pending preparation discarded')


async def _wait(check, *, timeout_s, period_s=.01, cancel_event=None):
    deadline = time.monotonic()+timeout_s
    error = "No feedback"
    while time.monotonic() < deadline:
        _require_not_cancelled(cancel_event)
        try:
            result = check()
            if result is not False and result is not None:
                break
        except MotionError as exc:
            error = str(exc)
        await asyncio.sleep(period_s)
    else:
        raise MotionError("Readiness deadline exceeded: "+error)
    _require_not_cancelled(cancel_event)
    return result


def _stop_after_fault(gateway, claimed, report):
    """Never send a composition stop through authority already handed back."""
    if gateway is None or not claimed or not report['motion_sent']:
        return
    if report.get('ownership_release_attempted') or not gateway.owns_control():
        report['fault_stop_not_sent'] = 'Prior grant no longer authorizes a composition stop'
        return
    try:
        gateway._engage_stop('commissioning composition fault')
        report['software_stop_published'] = True
    except Exception as exc:
        report['software_stop_error'] = str(exc)[:1024]


def _write_run_evidence(recording, output, report):
    try:
        recording.write(output/'measurements')
        report['measurements_written'] = True
    except Exception as exc:
        report.update(motion_outcome=report.get('status'), status='recording_failed',
                      measurements_written=False, recording_error=str(exc)[:1024])
        raise
    finally:
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


async def run_test(args, *, gpu_planner=None):
    # Fail before creating any ROS clients when commissioning is absent.
    settings = load_settings(args.commissioning, args.profile, simulation=args.simulation,
                             require_motion_enabled=args.execute)
    wire_bounds = measured_transport_bounds(settings)
    namespace = args.namespace
    if not isinstance(namespace, str) or re.fullmatch(r"/[A-Za-z][A-Za-z0-9_]{1,100}", namespace) is None:
        raise MotionError("A dedicated non-root driver namespace is required")
    config = strict_loads(Path(args.commissioning).read_bytes())
    section = config["profiles"][args.profile]["free_space_test"]
    assets = section["assets"]
    base = Path(args.commissioning).resolve().parent
    planner_config = (base/assets["planner_config"]["path"]).resolve()
    robot_urdf = (base/assets["robot_urdf"]["path"]).resolve()
    if gpu_planner is not None and (not isinstance(gpu_planner, WarmGpuPlanner)
                                    or gpu_planner.planner_config != planner_config):
        raise MotionError('Retained planner does not match this commissioned configuration')
    request_input = strict_loads(Path(args.request).read_bytes())
    if set(request_input) != {"goal", "world"} or digest(request_input["world"]) != settings.world_digest:
        raise MotionError("Request must contain only goal and the commissioned static world")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rammp_common_interfaces.srv import AcquireControl
    rclpy.init(args=[])
    node = rclpy.create_node("rammp_commissioning_"+uuid.uuid4().hex[:8])
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    buffer = DriverFeedbackBuffer(bounds=settings.feedback, simulation=args.simulation,
                                  extension_build_id=settings.extension_build_id)
    feedback = RosDriverFeedback(node, buffer, feedback_topic=namespace+"/driver_feedback",
                                 heartbeat_topic=namespace+"/driver_heartbeat")
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    gateway = None
    claim_client = None
    claimed = False
    recording = TrialRecording(simulation=args.simulation)
    report = {"mode":"driver_simulation" if args.simulation else "physical_free_space_commissioning",
              "physical_adl_available":False, "driver_started":False, "motion_sent":False,
              "physical_validation_claimed":False}
    cancel = asyncio.Event()
    loop = asyncio.get_running_loop()
    def operator_cancel():
        cancel.set()
        recording.event('cancel_requested')
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, operator_cancel)
    def stationary():
        if cancel.is_set():
            raise MotionError("Operator cancelled")
        buffer.require_unowned()
        return recording.sample(buffer.stationary_start(duration_s=settings.transport.settle_duration_s,
            velocity_rad_s=settings.transport.stationary_velocity_rad_s,
            position_span_rad=wire_bounds.stationary_position_span_rad))
    try:
        start = await _wait(stationary, timeout_s=max(5., 3*settings.transport.settle_duration_s),
                            cancel_event=cancel)
        request = {**request_input, "world_identity":settings.world_digest,
                   "start":{"position":list(start.position_rad), "velocity":[0.]*7, "acceleration":[0.]*7}}
        # Nominal zero derivatives are a PLANNING boundary justified by the
        # measured dwell/uncertainty bounds, never fabricated sensor feedback.
        recording.event('planning_started')
        if gpu_planner is None:
            operation = asyncio.to_thread(run_gpu_worker, request=request, planner_config=planner_config,
                model_dir=args.model_dir, wrapper_checkout=args.wrapper_checkout, gpu_cache=args.gpu_cache,
                output=output/"planner")
        else:
            operation = asyncio.to_thread(gpu_planner.plan, request)
        planning = asyncio.create_task(operation)
        try:
            response = await asyncio.shield(planning)
        except asyncio.CancelledError:
            await drain_nonpreemptible(planning)
            raise
        if cancel.is_set():
            raise MotionError("Operator cancelled while planning; candidate discarded")
        trajectory = checked_candidate(response, request, settings, planner_config, robot_urdf)
        admission = CommissioningAdmission(settings)
        validation = asyncio.create_task(asyncio.to_thread(admission.validate, trajectory))
        try:
            permit = await asyncio.shield(validation)
        except asyncio.CancelledError:
            await drain_nonpreemptible(validation)
            raise
        _require_not_cancelled(cancel)
        trajectory = permit.trajectory
        recording.event('candidate_prepared', {'trajectory_digest':trajectory.digest, 'duration_s':trajectory.duration_s})
        report.update(status="prepared", trajectory_digest=trajectory.digest,
                      duration_s=trajectory.duration_s, profile_id=settings.profile_id,
                      note="Validated commissioning candidate; no ADL task/contact capability")
        (output/"review.json").write_text(json.dumps(report, indent=2)+"\n")
        if not args.execute:
            return report
        latest = stationary()
        if any(abs(a-b) > tol for a,b,tol in zip(latest.position_rad, trajectory.points[0].state.position,
                                                settings.transport.start_position_rad)):
            raise MotionError("Arm moved during preparation; discard candidate and plan again")
        admission.check(permit, trajectory)
        # This service is used ONLY after source-pinned extension feedback has
        # proved the patched non-stealing ownership implementation is present.
        # Never call the original upstream seizing AcquireControl as fallback.
        claim_client = node.create_client(AcquireControl, namespace+"/acquire_control")
        await _wait(claim_client.service_is_ready, timeout_s=settings.transport.send_timeout_s,
                    cancel_event=cancel)
        buffer.require_unowned()
        _require_not_cancelled(cancel)
        request_grant = AcquireControl.Request()
        request_grant.owner_id = "rammp-test-"+uuid.uuid4().hex[:16]
        report["control_claim_attempted"] = True
        recording.event('control_claim_attempted')
        grant_future = claim_client.call_async(request_grant)
        await _wait(grant_future.done, timeout_s=settings.transport.send_timeout_s)
        grant = grant_future.result()
        if not grant.accepted:
            raise MotionError("Driver refused non-stealing control claim")
        claimed = True
        recording.event('control_claimed')
        _require_not_cancelled(cancel)
        owner = TransferredOwnership(request_grant.owner_id, bytes(grant.token), grant.generation)
        def measured(now):
            sample = buffer.read(now, ownership=owner, watchdog_timeout_s=settings.watchdog_timeout_s)
            for q,lo,hi,stop,uncertainty in zip(sample.position_rad, settings.cell_lower, settings.cell_upper,
                    settings.stop_excursion, section["joint_position_uncertainty_rad"]):
                if not lo+stop+uncertainty <= q <= hi-stop-uncertainty:
                    raise MotionError("Measured arm/stopping envelope leaves commissioned cell")
            for v, maximum in zip(sample.velocity_rad_s, settings.limits.velocity):
                if abs(v)+settings.feedback.velocity_error_rad_s > maximum:
                    raise MotionError("Measured velocity exceeds commissioned bound")
            feedback.heartbeat(owner, now=now)
            return recording.sample(sample)
        gateway = RosJointTrajectoryTransport(node, ownership=owner, admission_check=admission.check,
            state_check=measured, bounds=wire_bounds, action_name=namespace+"/execute_joint_trajectory",
            control_topic=namespace+"/control_status", release_service=namespace+"/release_control",
            stop_topic=namespace+"/estop")
        await _wait(lambda: (measured(time.monotonic()), gateway.owns_control() and gateway.port.server_ready())[1],
                    timeout_s=min(settings.watchdog_timeout_s/2, settings.transport.send_timeout_s),
                    cancel_event=cancel)
        # The control-claim round trip must not make an earlier stationary
        # acceleration bound stale before dispatch.
        buffer.stationary_start(duration_s=settings.transport.settle_duration_s,
            velocity_rad_s=settings.transport.stationary_velocity_rad_s,
            position_span_rad=wire_bounds.stationary_position_span_rad)
        _require_not_cancelled(cancel)
        # Record dispatch intent before the wire call; errors cannot hide a
        # possibly accepted action. The detailed receipt distinguishes outcomes.
        report["motion_sent"] = "dispatch_attempted"
        recording.event('dispatch_attempted', {'trajectory_digest':trajectory.digest})
        receipt = await gateway.execute(trajectory, permit=permit, cancel_event=cancel)
        report.update(status=receipt.status, receipt=asdict(receipt))
        recording.event('transport_receipt', asdict(receipt))
        if receipt.release_permitted:
            report['ownership_release_attempted'] = True
            await gateway.release()
            report["ownership_released"] = True
            recording.event('ownership_released')
        return report
    except BaseException as exc:
        report.update(status="rejected_or_fault", reason=str(exc), control_claimed=claimed)
        _stop_after_fault(gateway, claimed, report)
        # Diagnostic failure must never prevent the command stop above.
        try:
            recording.event('fault', {'reason':str(exc)[:1024]})
            if report.get('software_stop_published'):
                recording.event('software_stop')
        except Exception as recording_error:
            report['recording_error'] = str(recording_error)
        raise
    finally:
        # The independent driver watchdog remains armed if a claim/stop is
        # unresolved; teardown never clears an e-stop or restores an old token.
        unresolved = bool(report.get('control_claim_attempted') and not report.get('ownership_released'))
        if unresolved:
            report['ownership_unresolved_at_exit'] = True
            report['exit_protection'] = 'Local monitoring ends; independent driver watchdog remains armed. No release or measured stop is inferred.'
        cleanup = [('executor_shutdown', lambda: executor.shutdown(timeout_sec=2.)),
                   ('spin_join', lambda: spin.join(timeout=2.)), ('feedback_close', feedback.close)]
        if claim_client is not None:
            cleanup.append(('claim_client_close', lambda: node.destroy_client(claim_client)))
        if gateway is not None and not unresolved:
            cleanup.append(('gateway_close', gateway.close))
        cleanup.extend([('node_destroy', node.destroy_node), ('ros_shutdown', rclpy.shutdown)])
        for name, action in cleanup:
            try:
                action()
            except Exception as exc:
                report.setdefault('cleanup_errors', []).append({'step':name, 'reason':str(exc)[:1024]})
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        _write_run_evidence(recording, output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commissioning", default=str(Catalog().root/"config/commissioning.json"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--namespace", required=True, help="Dedicated driver namespace; global command endpoints are never used")
    parser.add_argument("--request", required=True, help="Local JSON with goal and commissioned world; no images uploaded")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wrapper-checkout", required=True)
    parser.add_argument("--gpu-cache", required=True)
    parser.add_argument("--output", required=True, help="New local artifact directory")
    parser.add_argument("--simulation", action="store_true", help="Requires explicit simulation-only commissioning profile and driver")
    parser.add_argument("--execute", action="store_true", help="Claim and command the already-running instrumented driver after admission")
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(run_test(args))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["status"] in {"prepared", "succeeded"} else 2
    except (ValueError, RuntimeError, KeyError, TypeError, OSError, ImportError) as exc:
        print(json.dumps({"status":"unavailable_or_rejected", "reason":str(exc),
                          "note":"Inspect result.json if a control claim was attempted"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
