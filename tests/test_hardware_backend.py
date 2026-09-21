"""World/handler/gateway integration with explicit test doubles; no robot I/O."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time
import unittest

from rammp_adl.contracts import Catalog, ContractError, digest
from rammp_adl.handlers import BackendFailure, ExecutionContext
from rammp_adl.hardware_backend import (
    HardwareObservationBackend, ObservationMeasurement, CommissioningArmSession,
    observation_runtime,
)
from rammp_adl.motion.commissioning import CommissioningAdmission, measured_transport_bounds
from rammp_adl.motion.rolling import MotionError
from rammp_adl.safety import SafetyError, SafetySupervisor
from rammp_adl.world import MetricPose, WorldModel

from test_commissioning import settings, trajectory
from test_driver_transport import FakePort, make_transport


ROOT = Path(__file__).resolve().parents[1]
ENTITY = "cabinet_handle_1"


def world_fixture():
    catalog = Catalog(ROOT)
    context = json.loads((ROOT/"examples/cabinet.context.json").read_text())
    context["available_skills"] = ["observe"]
    return catalog, WorldModel(context, catalog, trust_initial=True, max_evidence_age_s=5.)


def execution_context(world, node="observation"):
    snapshot = world.snapshot()
    return ExecutionContext(snapshot.context["task_id"], node,
                            execution_epoch=snapshot.execution_epoch, snapshot=snapshot)


def metric(world, *, captured_at=None, evidence_id="local-fit", revision=None):
    snapshot = world.snapshot()
    if revision is None:
        revision = snapshot.identities()["entity:"+ENTITY] + 1
    source = world.authorize_source("unit-test-local-fit", ["pose_valid"])
    observed_at = world.clock() if captured_at is None else captured_at
    assertion = {"predicate": "pose_valid", "args": {"entity_id": ENTITY, "pose_role": "pregrasp"},
                 "validity": "true"}
    world.register_evidence(evidence_id, source=source, predicates=[assertion], ttl_s=5., observed_at=observed_at)
    pose = MetricPose(ENTITY, "pregrasp", (.4, .1, .3), (0., 0., 0., 1.), (0.,)*36,
                      observed_at, "base_link", revision, snapshot.context["calibration_id"],
                      snapshot.context["base_epoch"], evidence_id, 5.)
    world.update_metric_pose(pose, source=source)
    return pose


def measurement(world, *, purpose="state", captured_at=None, extra=()):
    snapshot = world.snapshot()
    identities = snapshot.identities()
    return ObservationMeasurement(ENTITY, "wrist", purpose, "local-observation-1",
        world.clock() if captured_at is None else captured_at, 4.,
        ({"predicate": "observation_valid", "args": {"entity_id": ENTITY, "purpose": purpose},
          "validity": "true"}, *extra), {"test_only": True, "hardware_validated": False},
        {name: identities[name] for name in ("execution_epoch", "calibration_id", "base_epoch", "entity:"+ENTITY)})


def pending_measurement(world):
    measured = measurement(world, purpose="pose", extra=(
        {"predicate": "pose_valid", "args": {"entity_id": ENTITY, "pose_role": "pregrasp"},
         "validity": "true"},))
    identities = world.snapshot().identities()
    pose = MetricPose(ENTITY, "pregrasp", (.4, .1, .3), (0., 0., 0., 1.), (0.,)*36,
        measured.captured_at, "base_link", identities["entity:"+ENTITY]+1,
        identities["calibration_id"], identities["base_epoch"], measured.evidence_id, measured.valid_for_s)
    return replace(measured, metric_poses=(pose,))


class ObservationBackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.catalog, self.world = world_fixture()
        self.context = execution_context(self.world)
        self.args = {"entity_id": ENTITY, "camera": "wrist", "purpose": "state"}

    def backend(self, result=None, *, observer=None, held=True):
        return HardwareObservationBackend(self.catalog, self.world,
            observer=observer or (lambda args, context: result or measurement(self.world)),
            held_check=lambda: held)

    async def test_observation_commits_through_existing_dag_world_authority(self):
        runtime = observation_runtime(self.world, observer=lambda args, ctx: measurement(self.world),
            held_check=lambda: True, capabilities={"local_geometry", "cloud_grounding"}, mode="simulation")
        executor = runtime.executor
        # Explicit test assent to exercise the unchanged consent gate. No cloud
        # provider or image egress is used by this local evaluator fixture.
        executor.confirmation_callback = lambda plan, node, bound_digest: executor.confirmations.grant(
            task_id=plan["task_id"], epoch=plan["execution_epoch"], node_id=node["id"],
            digest=bound_digest, ttl_s=1., accepted=True).confirmation_id
        snapshot = self.world.snapshot()
        plan = {"schema_version": "1.0.0", "skill_library_hash": self.catalog.hash,
                "task_id": snapshot.context["task_id"], "snapshot_id": snapshot.snapshot_id,
                "execution_epoch": snapshot.execution_epoch,
                "nodes": [{"id": "watch", "skill": "observe", "args": self.args}], "edges": []}
        result = await executor.run_plan(plan)
        self.assertTrue(result.nodes, result.to_dict())
        self.assertEqual(result.nodes[0].status, "succeeded", result.to_dict())
        self.assertTrue(result.nodes[0].commit_receipt)
        self.assertTrue(result.nodes[0].backend_quiescent)
        self.assertEqual(self.world.snapshot().fact("observation_valid", {
            "entity_id": ENTITY, "purpose": "state"}), "true")
        self.assertNotEqual(result.status, "succeeded")  # Observation is not cabinet opening.
        self.assertEqual(executor.resources.owners, {})

    async def test_no_capabilities_are_inferred_or_hardware_gates_weakened(self):
        backend = self.backend()
        self.assertEqual(set(backend.handlers()), {"observe"})
        self.assertEqual(backend.registry(capabilities=()).available_skills, ())
        # observe is implemented: it registers on hardware only when the
        # operator both declares its capabilities and asserts commissioning.
        self.assertEqual(backend.registry(capabilities={"local_geometry", "cloud_grounding"},
                                         commissioned=False).available_skills, ())
        self.assertEqual(backend.registry(capabilities={"local_geometry", "cloud_grounding"},
                                         commissioned=True).available_skills, ("observe",))
        self.assertFalse(backend.hardware_commands)
        self.assertFalse(backend.physical_capabilities)
        self.assertFalse(backend.fixture_capabilities)
        with self.assertRaisesRegex(ContractError, "advertise exactly"):
            observation_runtime(self.world, observer=lambda args, ctx: measurement(self.world),
                                held_check=lambda: True, capabilities=(), mode="hardware")

    async def test_readonly_gate_rejects_motion_missing_skill_and_mode_spoofing(self):
        backend = self.backend()
        supervisor = SafetySupervisor(backend,
            epoch_getter=lambda: self.context.execution_epoch, invalidate_epoch=self.world.cancel_epoch)
        supervisor.check_dispatch(self.context.execution_epoch, skills=("observe",))
        for skills in ((), ("move_to_pose",), ("observe", "set_gripper")):
            with self.assertRaises(SafetyError):
                supervisor.check_dispatch(self.context.execution_epoch, skills=skills)
        class Pretend:
            mode = "local_measured_observation"
            hardware_commands = False
        supervisor.backend = Pretend()
        with self.assertRaises(SafetyError):
            supervisor.check_dispatch(self.context.execution_epoch, skills=("observe",))

    async def test_stale_mismatched_or_unproven_observation_rejected(self):
        good = measurement(self.world)
        for bad in (replace(good, captured_at=time.monotonic()-10),
                    replace(good, captured_at=time.monotonic()+10),
                    replace(good, camera="scene"), replace(good, purpose="pose"),
                    replace(good, dependencies={}),
                    replace(good, assertions=())):
            backend = self.backend(bad)
            outcome = await backend.handlers()["observe"].execute(self.args, self.context)
            self.assertEqual(outcome.status, "failed", outcome)
            self.assertEqual(outcome.proposed_effects, [])

    async def test_observer_cannot_manufacture_holding_or_another_entity(self):
        for fact in ({"predicate": "holding", "args": {"entity_id": ENTITY}, "validity": "true"},
                     {"predicate": "entity_exists", "args": {"entity_id": "robot"}, "validity": "true"}):
            backend = self.backend(measurement(self.world, extra=(fact,)))
            outcome = await backend.handlers()["observe"].execute(self.args, self.context)
            self.assertEqual(outcome.failure_code, "geometry_invalid")
            self.assertFalse(outcome.proposed_effects)

    async def test_pose_observation_requires_matching_metric_evidence(self):
        self.args["purpose"] = "pose"
        assertion = {"predicate": "pose_valid", "args": {"entity_id": ENTITY, "pose_role": "pregrasp"},
                     "validity": "true"}
        backend = self.backend(measurement(self.world, purpose="pose", extra=(assertion,)))
        outcome = await backend.handlers()["observe"].execute(self.args, self.context)
        self.assertEqual(outcome.failure_code, "geometry_invalid")
        pose = metric(self.world)
        backend = self.backend(measurement(self.world, purpose="pose", captured_at=pose.captured_at,
                                          extra=(assertion,)))
        outcome = await backend.handlers()["observe"].execute(self.args, self.context)
        self.assertEqual(outcome.status, "succeeded", outcome)

    async def test_new_pose_observation_commits_geometry_and_facts_atomically_through_dag(self):
        staged = []
        async def observer(args, context):
            measured = pending_measurement(self.world)
            staged.append(measured)
            self.assertNotIn((ENTITY, "pregrasp"), self.world.snapshot().metric_poses)
            return measured
        runtime = observation_runtime(self.world, observer=observer, held_check=lambda: True,
            capabilities={"local_geometry", "cloud_grounding"}, mode="simulation")
        executor = runtime.executor
        executor.confirmation_callback = lambda plan, node, bound_digest: executor.confirmations.grant(
            task_id=plan["task_id"], epoch=plan["execution_epoch"], node_id=node["id"],
            digest=bound_digest, ttl_s=1., accepted=True).confirmation_id
        snapshot = self.world.snapshot()
        plan = {"schema_version": "1.0.0", "skill_library_hash": self.catalog.hash,
                "task_id": snapshot.context["task_id"], "snapshot_id": snapshot.snapshot_id,
                "execution_epoch": snapshot.execution_epoch,
                "nodes": [{"id": "fit", "skill": "observe", "args": dict(self.args, purpose="pose")}],
                "edges": []}
        result = await executor.run_plan(plan)
        self.assertEqual(result.nodes[0].status, "succeeded", result.to_dict())
        final = self.world.snapshot()
        self.assertEqual(final.metric_poses[(ENTITY, "pregrasp")], staged[0].metric_poses[0])
        self.assertEqual(final.revision, snapshot.revision+1)
        self.assertEqual(final.context["collision_revision"], snapshot.context["collision_revision"]+1)
        self.assertEqual(final.fact("observation_valid", {"entity_id": ENTITY, "purpose": "pose"}), "true")
        self.assertTrue(result.nodes[0].commit_receipt)
        self.assertEqual(executor.resources.owners, {})
        self.assertNotEqual(result.status, "succeeded")

    async def test_hold_loss_discards_success_and_no_robot_stop_is_available(self):
        backend = self.backend(held=False)
        outcome = await backend.handlers()["observe"].execute(self.args, self.context)
        self.assertEqual(outcome.failure_code, "safety_fault")
        self.assertFalse(await backend.quiescent())
        held = [True]
        async def observer(args, context):
            held[0] = False
            return measurement(self.world)
        backend = HardwareObservationBackend(self.catalog, self.world, observer=observer,
                                             held_check=lambda: held[0])
        outcome = await backend.handlers()["observe"].execute(self.args, self.context)
        self.assertEqual(outcome.failure_code, "safety_fault")
        self.assertFalse(outcome.proposed_effects)

    async def test_cancel_keeps_backend_busy_until_observer_drains(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        async def observer(args, context):
            entered.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                await finish.wait()  # Deliberately noncooperative local test source.
            return measurement(self.world)
        backend = self.backend(observer=observer)
        work = asyncio.create_task(backend.handlers()["observe"].execute(self.args, self.context))
        await entered.wait()
        await backend.stop("test cancellation")
        self.assertFalse(await backend.skill_quiescent("observe"))
        finish.set()
        outcome = await work
        self.assertEqual(outcome.status, "cancelled")
        self.assertFalse(outcome.proposed_effects)
        self.assertTrue(await backend.skill_quiescent("observe"))

    async def test_observation_reserves_before_waiting_for_hold_and_until_final_check(self):
        entered, release = asyncio.Event(), asyncio.Event()
        checks = 0
        async def held():
            nonlocal checks
            checks += 1
            entered.set()
            await release.wait()
            return True
        backend = HardwareObservationBackend(self.catalog, self.world,
            observer=lambda args, context: measurement(self.world), held_check=held)
        work = asyncio.create_task(backend.observe(self.args, self.context))
        await entered.wait()
        self.assertFalse(await backend.skill_quiescent("observe"))
        self.assertFalse(await backend.quiescent())
        with self.assertRaisesRegex(BackendFailure, "already owned"):
            await backend.observe(self.args, self.context)
        self.assertEqual(checks, 1)
        release.set()
        self.assertEqual((await work).status, "succeeded")
        self.assertTrue(await backend.skill_quiescent("observe"))

    async def test_stop_during_initial_hold_check_never_starts_observer(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def held():
            entered.set()
            await release.wait()
            return True
        backend = HardwareObservationBackend(self.catalog, self.world,
            observer=lambda args, context: calls.append(args), held_check=held)
        work = asyncio.create_task(backend.observe(self.args, self.context))
        await entered.wait()
        await backend.stop()
        self.assertFalse(await backend.skill_quiescent("observe"))
        release.set()
        with self.assertRaisesRegex(BackendFailure, "revoked"):
            await work
        self.assertFalse(calls)
        self.assertTrue(await backend.skill_quiescent("observe"))

    async def test_direct_task_cancel_cannot_be_swallowed_by_observer(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def observer(args, context):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return measurement(self.world)
        backend = self.backend(observer=observer)
        work = asyncio.create_task(backend.handlers()["observe"].execute(self.args, self.context))
        await entered.wait()
        work.cancel()
        await asyncio.sleep(0)
        self.assertFalse(await backend.skill_quiescent("observe"))
        release.set()
        outcome = await work
        self.assertEqual(outcome.status, "cancelled")
        self.assertFalse(outcome.proposed_effects)
        self.assertTrue(await backend.skill_quiescent("observe"))


class PlannerDouble:
    """Protocol fixture; its output does NOT establish actual GPU planning."""
    def __init__(self):
        self.calls = []
        self.wait = None
        self.started = asyncio.Event()

    async def plan_pose(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        if self.wait is not None:
            await self.wait.wait()
        return trajectory(duration=.5)


class CommissioningSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.catalog, self.world = world_fixture()
        self.pose = metric(self.world)
        self.context = execution_context(self.world, "commissioning-test")
        self.planner, self.port = PlannerDouble(), FakePort()
        collision_world = [{"name": "unit-test-only-cuboid"}]
        self.settings = replace(settings(), world_digest=digest(collision_world))
        self.admission = CommissioningAdmission(self.settings)
        self.authorized = True
        self.session = CommissioningArmSession(self.world, self.planner, self.admission,
            stationary_check=lambda: self.port.state(time.monotonic()), collision_world=collision_world,
            command_check=lambda context: self.authorized)
        self.gateway, _ = make_transport(self.port, check=self.session.admission_check)
        self.gateway.bounds = measured_transport_bounds(self.settings)
        self.session.gateway = self.gateway

    async def prepare(self):
        return await self.session.prepare(ENTITY, "pregrasp", self.context)

    async def test_exact_world_bound_plan_reaches_real_gateway_and_measured_receipt(self):
        prepared = await self.prepare()
        request = self.planner.calls[0]
        self.assertEqual(request["position_m"], list(self.pose.position_m))
        self.assertEqual(request["quaternion_xyzw"], list(self.pose.orientation_xyzw))
        self.assertEqual(request["world_identity"], self.settings.world_digest)
        self.assertFalse(self.port.sent)
        receipt = await self.session.execute(prepared, self.context)
        self.assertEqual(receipt.status, "succeeded", receipt)
        self.assertTrue(receipt.terminal_acknowledged and receipt.measured_quiescent)
        self.assertEqual(self.port.sent, [prepared.trajectory])
        self.assertEqual(self.port.released, 0)  # Supervisor retains ownership.
        self.assertFalse(prepared.review()["physical_adl_available"])
        self.assertFalse(self.session.physical_adl_skills)
        self.assertNotEqual(self.world.snapshot().fact("at_pose", {
            "entity_id": ENTITY, "pose_role": "pregrasp"}), "true")
        with self.assertRaisesRegex(MotionError, "consumed"):
            await self.session.execute(prepared, self.context)

    async def test_forged_prepared_object_wrong_context_and_start_drift_send_nothing(self):
        prepared = await self.prepare()
        with self.assertRaises(MotionError):
            await self.session.execute(replace(prepared), self.context)
        with self.assertRaisesRegex(MotionError, "different"):
            await self.session.execute(prepared, replace(self.context, node_id="another-node"))
        self.port.position = .1
        with self.assertRaisesRegex(MotionError, "start changed"):
            await self.session.execute(prepared, self.context)
        self.assertFalse(self.port.sent)

    async def test_pose_or_epoch_change_rejects_candidate_before_send(self):
        prepared = await self.prepare()
        self.world.cancel_epoch()
        with self.assertRaisesRegex(MotionError, "epoch"):
            await self.session.execute(prepared, self.context)
        self.assertFalse(self.port.sent)

    async def test_binding_replacement_during_planning_discards_candidate(self):
        self.planner.wait = asyncio.Event()
        work = asyncio.create_task(self.prepare())
        await self.planner.started.wait()
        metric(self.world, evidence_id="new-local-fit")
        self.planner.wait.set()
        with self.assertRaisesRegex(MotionError, "dependencies changed"):
            await work
        self.assertFalse(self.port.sent)
        self.assertIsNone(self.session._prepared)

    async def test_cancelled_planning_keeps_lease_until_nonpreemptible_completion(self):
        self.planner.wait = asyncio.Event()
        work = asyncio.create_task(self.prepare())
        await self.planner.started.wait()
        work.cancel()
        await asyncio.sleep(0)
        with self.assertRaisesRegex(MotionError, "leased"):
            await self.prepare()
        self.assertFalse(work.done())
        self.planner.wait.set()
        with self.assertRaises(asyncio.CancelledError):
            await work
        self.assertFalse(self.port.sent)
        self.assertIsNone(self.session._prepared)

    async def test_revoked_binding_during_motion_cancels_through_gateway(self):
        prepared = await self.prepare()
        self.port.auto_finish = False
        work = asyncio.create_task(self.session.execute(prepared, self.context))
        while not self.port.sent:
            await asyncio.sleep(.001)
        self.authorized = False
        receipt = await work
        self.assertEqual(receipt.status, "failed")
        self.assertIn("authorization", receipt.reason)
        self.assertTrue(receipt.terminal_acknowledged and receipt.measured_quiescent)
        self.assertEqual(self.port.cancelled, 1)
        self.assertFalse(self.port.released)

    async def test_final_gate_cannot_be_replaced_by_unbound_admission(self):
        prepared = await self.prepare()
        self.gateway.admission_check = self.admission.check
        with self.assertRaisesRegex(MotionError, "world-bound"):
            await self.session.execute(prepared, self.context)
        self.assertFalse(self.port.sent)

    async def test_stale_stationary_state_rejected_without_planning(self):
        self.port.stale = True
        with self.assertRaisesRegex(MotionError, "stale"):
            await self.prepare()
        self.assertFalse(self.planner.calls)

    async def test_concurrent_execute_cannot_wait_through_first_and_replay_candidate(self):
        prepared = await self.prepare()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0
        async def stationary():
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return self.port.state(time.monotonic())
        self.session.stationary_check = stationary
        first = asyncio.create_task(self.session.execute(prepared, self.context))
        await entered.wait()
        with self.assertRaisesRegex(MotionError, "busy"):
            await self.session.execute(prepared, self.context)
        self.assertEqual(calls, 1)
        release.set()
        self.assertEqual((await first).status, "succeeded")
        self.port.position = 0.  # A matching later boundary still cannot replay.
        with self.assertRaisesRegex(MotionError, "consumed"):
            await self.session.execute(prepared, self.context)
        self.assertEqual(len(self.port.sent), 1)

    async def test_gateway_requires_exact_measured_commissioned_bounds(self):
        prepared = await self.prepare()
        self.gateway.bounds = replace(self.gateway.bounds, path_position_rad=(.1,)*7)
        with self.assertRaisesRegex(MotionError, "bounds differ"):
            await self.session.execute(prepared, self.context)
        self.assertFalse(self.port.sent)
        self.assertFalse(self.session._busy)

    async def test_gateway_replacement_during_state_check_sends_nothing(self):
        prepared = await self.prepare()
        replacement, _ = make_transport(self.port, check=self.session.admission_check)
        replacement.bounds = self.gateway.bounds
        async def stationary():
            self.session.gateway = replacement
            return self.port.state(time.monotonic())
        self.session.stationary_check = stationary
        with self.assertRaisesRegex(MotionError, "gateway was replaced"):
            await self.session.execute(prepared, self.context)
        self.assertFalse(self.port.sent)

    async def test_gateway_bounds_change_during_motion_cancels_exact_goal(self):
        prepared = await self.prepare()
        self.port.auto_finish = False
        work = asyncio.create_task(self.session.execute(prepared, self.context))
        while not self.port.sent:
            await asyncio.sleep(.001)
        self.gateway.bounds = replace(self.gateway.bounds, path_position_rad=(.1,)*7)
        receipt = await work
        self.assertEqual(receipt.status, "failed")
        self.assertIn("bounds differ", receipt.reason)
        self.assertEqual(self.port.cancelled, 1)

    async def test_execution_rejects_stale_dwell_even_if_current_gateway_state_is_fresh(self):
        prepared = await self.prepare()
        stale = replace(self.port.state(time.monotonic()), acquired_at_monotonic_s=time.monotonic()-1.)
        self.session.stationary_check = lambda: stale
        with self.assertRaisesRegex(MotionError, "stale"):
            await self.session.execute(prepared, self.context)
        self.assertFalse(self.port.sent)

    async def test_collision_world_views_cannot_change_commissioned_planning_snapshot(self):
        exposed = self.session.collision_world
        exposed[0]["name"] = "mutated-obstacle"
        await self.prepare()
        request = self.planner.calls[0]
        self.assertEqual(digest(request["world"]), self.settings.world_digest)
        self.assertEqual(request["world"], [{"name": "unit-test-only-cuboid"}])


if __name__ == "__main__":
    unittest.main()
