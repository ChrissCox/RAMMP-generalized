"""Guarded constraint following on the sheppy backend, time scaling, contact exclusions."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rammp_adl.constraints import ConstraintStore, metric_constraint
from rammp_adl.contracts import Catalog, ContractError
from rammp_adl.handlers import BackendFailure
from rammp_adl.intake import draft_context, seed_articulation
from rammp_adl.motion.collision_guard import EffortGuard, GuardSet, GuardError
from rammp_adl.motion.kinematics import UrdfChain
from rammp_adl.motion.sheppy_client import SheppyClientError, executor_problems, scale_trajectory_time
from rammp_adl.sheppy_backend import SheppyArmBackend, SheppyGeometry
from rammp_adl.world import WorldModel

from synthetic_scene import looking_at
from test_constraints import GEOMETRY, PROPOSAL
from test_sheppy_backend import FakeClient, execution_context, trajectory

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2"
BENCH = json.loads((ROOT/"config/sheppy-bench.context.json").read_text())
PROFILES = [{"profile_id": "bench_transit", "safety_class": "transit", "simulation_only": False},
            {"profile_id": "bench_gripper", "safety_class": "gripper", "simulation_only": False},
            {"profile_id": "bench_contact", "safety_class": "contact", "simulation_only": False}]
CAMERA = looking_at([.1, .1, .35], [.6, .1, .3])


def door_record():
    return metric_constraint(PROPOSAL, GEOMETRY, CAMERA, constraint_id="cabinet_door_constraint",
                             entity_id="handle_1", label="cabinet door", surface_entity_id="cabinet_door_surface")


class TrackingClient(FakeClient):
    """Plans that land exactly where they are asked, so waypoint arithmetic is visible."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = []
        self.trip_after = None

    async def plan_to_pose(self, position_m, quaternion_xyzw, *, cancel_event=None, timeout_s=None):
        if self.plan_error:
            raise SheppyClientError(self.plan_error)
        self.targets.append((tuple(position_m), tuple(quaternion_xyzw)))
        end = tuple(q+.02 for q in self.joints)
        return trajectory(self.joints, end), {"message": "ok", "planning_time_s": .1}

    async def execute(self, path, *, cancel_event=None, guard=None, **kwargs):
        if self.trip_after is not None and len(self.sent) >= self.trip_after:
            self.execute_status, self.trip = "guard_trip", {"kind": "contact", "deviation_nm": 9.1, "touch_nm": 8.}
        return await super().execute(path, cancel_event=cancel_event, guard=guard, **kwargs)


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class FollowConstraintTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(ROOT)
        self.chain = UrdfChain.from_path(BUNDLE/"arm-gripper-locked.urdf")
        self.folder = tempfile.TemporaryDirectory()
        self.store = ConstraintStore(self.folder.name)
        self.record = door_record()
        self.guards = []

        def guard_factory(touch_nm=3., exclusions=(), tool_exclusion_m=0.):
            guard = GuardSet(effort=EffortGuard(touch_nm), exclusions=exclusions, tool_exclusion_m=tool_exclusion_m)
            self.guards.append({"touch_nm": touch_nm, "exclusions": list(exclusions), "tool_exclusion_m": tool_exclusion_m})
            return guard
        self.guard_factory = guard_factory

    def tearDown(self):
        self.folder.cleanup()

    def world(self):
        descriptors = [{"entity_id": "handle_1", "label": "handle", "pose_roles": ["grasp", "pregrasp", "retract", "staging"]},
                       {"entity_id": "cabinet_door_surface", "label": "cabinet door", "pose_roles": []}]
        from rammp_adl.constraints import context_constraint
        context = draft_context(BENCH, descriptors, task_id="task-door", camera_id="wrist_d405",
                                constraints=[context_constraint(self.record)])
        return WorldModel(context, self.catalog, trust_initial=False, max_evidence_age_s=600.)

    def backend(self, client, **options):
        world = self.world()
        backend = SheppyArmBackend(self.catalog, world, client=client, profiles=PROFILES, chain=self.chain,
                                   constraints={self.record["constraint_id"]: self.record}, constraint_store=self.store,
                                   guard_factory=self.guard_factory, speed_scales={"transit": 2.5, "contact": 4.}, **options)
        backend.holding_id, backend.grasp_knuckle, backend.current_pose = "handle_1", .45, ("handle_1", "grasp")
        return backend, world

    def follow(self, backend, world, target=.5):
        args = {"entity_id": "handle_1", "constraint_id": "cabinet_door_constraint", "target_value": target,
                "target_unit": "rad", "profile_id": "bench_contact"}
        return asyncio.run(backend.follow_constraint(args, execution_context(world)))

    def test_the_arc_is_followed_as_slow_guarded_steps_and_recorded(self):
        client = TrackingClient(knuckle=.45)
        backend, world = self.backend(client)
        outcome = self.follow(backend, world, .5)
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(len(client.targets), 6)                                # 5 degree steps to 0.5 rad
        self.assertEqual(len(client.sent), 6)
        self.assertTrue(all(g["touch_nm"] == 8. and g["tool_exclusion_m"] == .15 for g in self.guards))
        sent = client.sent[0]
        self.assertIn("time scaled x4", sent.provenance)
        self.assertEqual(executor_problems(sent), [])
        facts = {(f["predicate"], f["validity"]) for f in outcome.proposed_effects}
        self.assertIn(("constraint_goal_verified", "true"), facts)
        self.assertIn(("at_grasp_pose", "false"), facts)
        self.assertIsNone(backend.current_pose)
        evidence = outcome.evidence[0]["data"]
        self.assertEqual(evidence["achieved"], .5)
        self.assertIn("not measured", evidence["evidence_basis"])
        saved = self.store.load("cabinet door")
        self.assertEqual(saved["attempts"][-1]["status"], "succeeded")
        self.assertEqual(saved["attempts"][-1]["achieved"], .5)
        # The commanded tool poses stay on the circle around the hinge.
        pivot = np.asarray(self.record["pivot_base"])
        first = np.asarray(client.targets[0][0])
        radius = np.linalg.norm(first[:2]-pivot[:2])
        for position, _ in client.targets:
            self.assertAlmostEqual(np.linalg.norm(np.asarray(position)[:2]-pivot[:2]), radius, places=9)

    def test_a_step_out_of_reach_lets_the_part_swivel_between_the_pads(self):
        # The wrist cannot turn with the door past 0.3 rad here; a pull parallel to the hinge can swivel in the grasp.
        from rammp_adl.motion.kinematics import quaternion_matrix
        client = TrackingClient(knuckle=.45)
        backend, world = self.backend(client)
        start_rotation = quaternion_matrix(backend._tool_pose(client.joints)[1])
        plan = client.plan_to_pose

        async def reach_limited(position_m, quaternion_xyzw, **kwargs):
            turn = np.arccos(np.clip((np.trace(start_rotation.T @ quaternion_matrix(tuple(quaternion_xyzw)))-1.)/2., -1., 1.))
            if turn > .3+1e-6:
                raise SheppyClientError("planner refused: MotionGenStatus.IK_FAIL: no collision-free joint solution AT the goal")
            return await plan(position_m, quaternion_xyzw, **kwargs)
        client.plan_to_pose = reach_limited
        outcome = self.follow(backend, world, .6)
        self.assertEqual(outcome.status, "succeeded")
        steps = outcome.evidence[0]["data"]["steps"]
        self.assertEqual(steps[0]["swivel"], 0.)
        self.assertEqual(steps[-1]["swivel"], .5)
        self.assertEqual(outcome.evidence[0]["data"]["achieved"], .6)
        pivot = np.asarray(self.record["pivot_base"])
        radii = {round(float(np.linalg.norm(np.asarray(p)[:2]-pivot[:2])), 9) for p, _ in client.targets}
        self.assertEqual(len(radii), 1)                                                    # the hand still follows the arc

    def test_a_contact_trip_is_a_model_mismatch_with_the_achieved_value_recorded(self):
        client = TrackingClient(knuckle=.45)
        client.trip_after = 2
        backend, world = self.backend(client)
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .5)
        self.assertEqual(caught.exception.code, "model_mismatch")
        self.assertIn("stopped at", str(caught.exception))
        saved = self.store.load("cabinet door")
        attempt = saved["attempts"][-1]
        self.assertEqual(attempt["status"], "tripped")
        self.assertAlmostEqual(attempt["achieved"], 2*self.record["step"], places=9)
        self.assertEqual(attempt["trip"]["kind"], "contact")
        self.assertEqual(len(client.cancelled), 0) if isinstance(client.cancelled, list) else None

    def test_slip_and_missing_record_and_unit_mismatch_are_refused(self):
        client = TrackingClient(knuckle=.45)
        backend, world = self.backend(client)
        client.knuckle = .75                                                    # closed far past the grasp: nothing held
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .2)
        self.assertEqual(caught.exception.code, "slip")
        backend.holding_id = None
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .2)
        self.assertEqual(caught.exception.code, "slip")
        backend.holding_id = "handle_1"
        args = {"entity_id": "handle_1", "constraint_id": "cabinet_door_constraint", "target_value": .1,
                "target_unit": "m", "profile_id": "bench_contact"}
        with self.assertRaises(BackendFailure) as caught:
            asyncio.run(backend.follow_constraint(args, execution_context(world)))
        self.assertEqual(caught.exception.code, "model_mismatch")
        backend.constraints.clear()
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .2)
        self.assertEqual(caught.exception.code, "stale_state")

    def test_geometry_establishes_contact_readiness_and_attached_release(self):
        backend, world = self.backend(TrackingClient(knuckle=.45))
        geometry = SheppyGeometry(backend)
        snapshot = world.snapshot()
        node = {"id": "n", "skill": "follow_constraint", "args": {"entity_id": "handle_1", "constraint_id": "cabinet_door_constraint",
                                                                    "target_value": .5, "target_unit": "rad", "profile_id": "bench_contact"}}
        check = geometry.validate(node, snapshot, {"geometry": None}, "admission")
        self.assertIn({"predicate": "contact_ready", "args": node["args"] | {} and {"entity_id": "handle_1", "constraint_id": "cabinet_door_constraint", "profile_id": "bench_contact"}, "validity": "true"},
                      list(check.established_facts))
        release = {"id": "r", "skill": "release", "args": {"entity_id": "handle_1", "support_id": "cabinet_door_surface", "profile_id": "bench_gripper"}}
        established = [f["predicate"] for f in geometry.validate(release, snapshot, {"geometry": None}, "admission").established_facts]
        self.assertIn("at_release_pose", established)
        self.assertIn("support_verified", established)
        free = {"id": "r2", "skill": "release", "args": {"entity_id": "handle_1", "support_id": "robot", "profile_id": "bench_gripper"}}
        self.assertNotIn("support_verified", [f["predicate"] for f in geometry.validate(free, snapshot, {"geometry": None}, "admission").established_facts])
        bad = dict(node, args=dict(node["args"], target_value=5.))
        with self.assertRaises(ContractError):
            geometry.validate(bad, snapshot, {"geometry": None}, "admission")
        self.assertIn("contact_monitor", backend.provided_capabilities)

    def test_grasp_role_moves_exclude_the_target_and_seeding_makes_the_constraint_valid(self):
        client = TrackingClient(knuckle=.01)
        backend, world = self.backend(client)
        backend.holding_id, backend.current_pose = None, None
        from rammp_adl.world import MetricPose
        identities = world.snapshot().identities()
        authority = world.authorize_source("test_observation", self.catalog.predicates)
        pose = MetricPose("handle_1", "grasp", (.58, .1, .3), (0., 0., 0., 1.), tuple([0.]*36), world.clock(), "base_link",
                          identities["entity:handle_1"]+1, identities["calibration_id"], identities["base_epoch"], "pose-1", 60.)
        world.register_evidence("pose-1", source=authority, ttl_s=60., observed_at=world.clock(),
                                predicates=[{"predicate": "pose_valid", "validity": "true", "args": {"entity_id": "handle_1", "pose_role": "grasp"}}])
        world.update_metric_pose(pose, source=authority)
        args = {"target": {"entity_id": "handle_1", "pose_role": "grasp"}, "profile_id": "bench_transit"}
        outcome = asyncio.run(backend.move_to_pose(args, execution_context(world)))
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(self.guards[-1]["exclusions"], [((.58, .1, .3), .10)])
        self.assertIn("time scaled x4", client.sent[-1].provenance)                      # the last centimetres into a grasp: contact speed
        # Parked at the handle, the next move away exempts what the fingers are beside, once it has left it no more.
        self.assertTrue(backend.at_contact)
        context = execution_context(world)
        asyncio.run(backend.look("back", context))
        departing = self.guards[-1]["exclusions"]
        self.assertEqual(len(departing), 1)
        self.assertEqual(departing[0][1], .10)
        self.assertFalse(backend.at_contact)
        asyncio.run(backend.look("left", context))
        self.assertEqual(list(self.guards[-1]["exclusions"]), [])
        seeded = seed_articulation(world, [{"record": self.record, "constraint": {"constraint_id": "cabinet_door_constraint"}, "history": []}],
                                   now=world.clock())
        self.assertEqual(len(seeded), 1)
        self.assertEqual(world.snapshot().fact("constraint_valid", {"constraint_id": "cabinet_door_constraint"}), "true")


class ScalingAndExclusionTests(unittest.TestCase):
    def test_time_scaling_keeps_the_path_and_slows_the_rates(self):
        path = trajectory((0.,)*7, (.5,)*7)
        slow = scale_trajectory_time(path, 2.)
        self.assertEqual(slow.duration_s, path.duration_s*2.)
        for fast_point, slow_point in zip(path.points, slow.points):
            self.assertEqual(slow_point.state.position, fast_point.state.position)
            np.testing.assert_allclose(slow_point.state.velocity, np.asarray(fast_point.state.velocity)/2.)
            np.testing.assert_allclose(slow_point.state.acceleration, np.asarray(fast_point.state.acceleration)/4.)
        self.assertIs(scale_trajectory_time(path, 1.), path)
        with self.assertRaises(SheppyClientError):
            scale_trajectory_time(path, .5)

    def test_guard_exclusions_are_validated_and_applied(self):
        with self.assertRaises(GuardError):
            GuardSet(effort=EffortGuard(3.), exclusions=[((0., 0., 0.), 0.)])
        with self.assertRaises(GuardError):
            GuardSet(effort=EffortGuard(3.), tool_exclusion_m=-1.)
        from rammp_adl.motion.collision_guard import CollisionGuard
        points = np.array([[0., 0., 0.], [.05, 0., 0.], [1., 0., 0.]])
        kept = CollisionGuard.excluded(points, [((0., 0., 0.), .1)])
        np.testing.assert_allclose(kept, [[1., 0., 0.]])
        self.assertEqual(len(CollisionGuard.excluded(np.zeros((0, 3)), [((0., 0., 0.), .1)])), 0)


if __name__ == "__main__":
    unittest.main()
