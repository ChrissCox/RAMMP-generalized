"""Guarded constraint following on the sheppy backend, time scaling, contact exclusions."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from rammp_adl.constraints import ConstraintStore, metric_constraint
from rammp_adl.contracts import Catalog, ContractError
from rammp_adl.handlers import BackendFailure
from rammp_adl.intake import draft_context, seed_articulation
from rammp_adl.motion.collision_guard import EffortGuard, GuardSet, GuardError
from rammp_adl.motion.kinematics import UrdfChain, quaternion_matrix
from rammp_adl.motion.sheppy_client import JOINT_VMAX, JOINTS, SheppyClientError, executor_problems, scale_trajectory_time
from rammp_adl.sheppy_backend import PullGuard, SheppyArmBackend, SheppyGeometry
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

    async def plan_to_pose(self, position_m, quaternion_xyzw, *, cancel_event=None, timeout_s=None, start_joints=None):
        if self.plan_error:
            raise SheppyClientError(self.plan_error)
        self.targets.append((tuple(position_m), tuple(quaternion_xyzw)))
        end = tuple(q+.02 for q in self.joints)
        return trajectory(self.joints, end), {"message": "ok", "planning_time_s": .1}


# Run 5 on the bench (2026-09-24): the joints holding the cabinet door's pull, and that door's hinge.
GRASP_JOINTS = (-.5330, .6570, -2.5670, -1.6130, -.9090, .6160, 1.9680)
HINGE = {"axis_base": [.04188549703488399, -.01753357612509984, .9989685574863736],
         "pivot_base": [.8433071780380824, .16606076659318864, .428801964838091], "direction": -1.}
IK_FAIL = "planner refused: MotionGenStatus.IK_FAIL: no collision-free joint solution AT the goal"


def tool_frame(chain, joints):
    frame = chain.base_from_link(dict(zip(JOINTS, joints)), "end_effector_link")
    return frame @ np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., .12], [0., 0., 0., 1.]])


def solve(chain, seed, position, rotation):
    """The planner's stand-in: damped least squares from the start it was given. Tests only; the runtime solves no IK."""
    from scipy.spatial.transform import Rotation
    q = np.array(seed, dtype=float)
    for _ in range(50):
        frame = tool_frame(chain, q)
        error = np.concatenate([position-frame[:3, 3], Rotation.from_matrix(rotation @ frame[:3, :3].T).as_rotvec()])
        if np.linalg.norm(error) < 1e-10:
            break
        jacobian = np.zeros((6, 7))
        for j in range(7):
            nudged = q.copy()
            nudged[j] += 1e-7
            moved = tool_frame(chain, nudged)
            jacobian[:3, j] = (moved[:3, 3]-frame[:3, 3])/1e-7
            jacobian[3:, j] = Rotation.from_matrix(moved[:3, :3] @ frame[:3, :3].T).as_rotvec()/1e-7
        q += jacobian.T @ np.linalg.solve(jacobian @ jacobian.T+1e-9*np.eye(6), error)
    return q


class ArcClient(FakeClient):
    """A planner that solves each pose from the start it is given; a driver that ticks the guard along the path."""
    def __init__(self, chain, *, solve_live=True, **kwargs):
        super().__init__(joints=GRASP_JOINTS, **kwargs)
        self.chain, self.solve_live = chain, solve_live     # solve_live=False: moves from the live joints arrive where they start
        self.plans, self.ends = [], []
        self.reach = None                     # (position, rotation) -> False when out of reach
        self.distort = None                   # (plan index, position) -> the position the planner solves for instead
        self.knuckle_at = self.effort_at = None

    async def plan_to_pose(self, position_m, quaternion_xyzw, *, cancel_event=None, timeout_s=None, start_joints=None):
        start = tuple(self.joints if start_joints is None else start_joints)
        if start_joints is None and not self.solve_live:
            return trajectory(start, start), {"message": "ok", "planning_time_s": .2}
        self.plans.append({"start": start, "position": tuple(position_m), "orientation": tuple(quaternion_xyzw)})
        rotation = quaternion_matrix(tuple(quaternion_xyzw))
        if self.reach is not None and not self.reach(np.asarray(position_m), rotation):
            raise SheppyClientError(IK_FAIL)
        position = np.asarray(position_m) if self.distort is None else self.distort(len(self.plans)-1, np.asarray(position_m))
        end = tuple(float(v) for v in solve(self.chain, start, position, rotation))
        self.ends.append(end)
        return trajectory(start, end), {"message": "ok", "planning_time_s": .2}

    async def execute(self, path, *, cancel_event=None, guard=None, **kwargs):
        self.sent.append(path)
        self.guards_seen = getattr(self, "guards_seen", [])+[guard]
        self.renewal_times, self.duration = list(getattr(guard, "renew_at_s", ())), path.duration_s
        for progress in np.linspace(.02, 1., 50):
            live = self.live_joints()
            if self.knuckle_at is not None:
                live["knuckle_rad"] = self.knuckle_at(progress)
            if self.effort_at is not None:
                live["effort_nm"] = self.effort_at(progress)
            if guard is not None:
                guard.on_progress(progress)
                trip = guard.check(live=live, trajectory=path, elapsed_s=progress*path.duration_s, now=time.monotonic())
                if trip is not None:
                    return {"status": "guard_trip", "message": f"guard tripped ({trip.get('kind')})", "progress": float(progress),
                            "error_code": None, "sent": True, "cancel_requested": False, "final_position_rad": None,
                            "goal_gap_rad": None, "trip": trip}
        self.joints = tuple(path.points[-1].state.position)
        return {"status": "succeeded", "message": "SUCCESSFUL", "progress": 1., "error_code": 0, "sent": True,
                "cancel_requested": False, "final_position_rad": list(self.joints), "goal_gap_rad": 0.}


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

    def arc(self, **options):
        """The bench door's own hinge, held where run 5 held it."""
        client = ArcClient(self.chain, knuckle=.45, **options)
        self.record.update(HINGE)
        backend, world = self.backend(client)
        return client, backend, world

    def on_the_arc(self, path, backend):
        """Horizontal distance of the tool from the hinge line, sampled along the flown path."""
        pivot, axis = np.asarray(HINGE["pivot_base"]), np.asarray(HINGE["axis_base"])
        radii = []
        for time_s in np.linspace(0., path.duration_s, 120):
            position = np.asarray(backend._tool_pose(path.sample(time_s).position)[0])
            offset = position-pivot
            radii.append(float(np.linalg.norm(offset-(offset @ axis)*axis)))
        return np.asarray(radii)

    def test_the_arc_is_planned_first_then_flown_as_one_guarded_pull(self):
        client, backend, world = self.arc()
        outcome = self.follow(backend, world, .5)
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(len(client.sent), 1)                                   # one pull, not six stops
        self.assertEqual(len(client.plans), 7)                                  # reach checked once, then 5 degree waypoints
        chain_plans = client.plans[1:]
        self.assertEqual(chain_plans[0]["start"], GRASP_JOINTS)
        for plan, previous_end in zip(chain_plans[1:], client.ends[1:]):
            self.assertEqual(plan["start"], previous_end)                       # each planned from where the last one ends
        pull = client.sent[0]
        self.assertEqual(executor_problems(pull), [])
        np.testing.assert_allclose(pull.points[0].state.position, GRASP_JOINTS)
        np.testing.assert_allclose(pull.points[-1].state.position, client.ends[-1])
        velocities = np.abs([p.state.velocity for p in pull.points])
        self.assertTrue((velocities <= np.asarray(JOINT_VMAX)/4.*(1+1e-9)).all())  # the contact share of every joint's limit
        self.assertLess(pull.duration_s, 6.)
        radii = self.on_the_arc(pull, backend)
        self.assertLess(radii.max()-radii.min(), .004)                          # the hand stays on the door's circle
        guard = client.guards_seen[-1]
        self.assertIsInstance(guard, PullGuard)
        self.assertEqual(guard.renewals, 5)                                     # baseline moved at each interior waypoint
        self.assertTrue(all(g["touch_nm"] == 8. and g["tool_exclusion_m"] == .15 for g in self.guards))
        facts = {(f["predicate"], f["validity"]) for f in outcome.proposed_effects}
        self.assertIn(("constraint_goal_verified", "true"), facts)
        self.assertIn(("at_grasp_pose", "false"), facts)
        self.assertIsNone(backend.current_pose)
        evidence = outcome.evidence[0]["data"]
        self.assertEqual(evidence["achieved"], .5)
        self.assertEqual(evidence["pull"]["wrist_lag"], 0.)
        self.assertEqual([step["status"] for step in evidence["steps"]], ["succeeded"]*6)
        self.assertIn("not measured", evidence["evidence_basis"])
        saved = self.store.load("cabinet door")
        self.assertEqual(saved["attempts"][-1]["status"], "succeeded")
        self.assertEqual(saved["attempts"][-1]["achieved"], .5)

    def test_out_of_reach_the_part_swivels_evenly_between_the_pads(self):
        # The wrist cannot turn more than 0.3 rad from where it holds the pull; the pull runs along the hinge.
        client, backend, world = self.arc()
        start_rotation = quaternion_matrix(backend._tool_pose(GRASP_JOINTS)[1])
        client.reach = lambda position, rotation: np.arccos(np.clip((np.trace(start_rotation.T @ rotation)-1.)/2., -1., 1.)) <= .3+1e-6
        outcome = self.follow(backend, world, .6)
        self.assertEqual(outcome.status, "succeeded")
        evidence = outcome.evidence[0]["data"]
        self.assertEqual(evidence["pull"]["wrist_lag"], .5)                     # 0 and 0.25 cannot reach 0.6 rad; half can
        self.assertEqual({step["swivel"] for step in evidence["steps"]}, {.5})   # the same lag all the way: no sudden twist
        self.assertEqual(evidence["achieved"], .6)
        pull = client.sent[0]
        radii = self.on_the_arc(pull, backend)
        self.assertLess(radii.max()-radii.min(), .004)                          # the hand still follows the arc
        end_rotation = quaternion_matrix(backend._tool_pose(pull.points[-1].state.position)[1])
        self.assertAlmostEqual(float(np.arccos((np.trace(start_rotation.T @ end_rotation)-1.)/2.)), .3, places=4)

    def test_without_swivel_the_reachable_part_is_pulled_and_the_rest_refused(self):
        client, backend, world = self.arc()
        backend.swivel_lags = ()
        start_rotation = quaternion_matrix(backend._tool_pose(GRASP_JOINTS)[1])
        client.reach = lambda position, rotation: np.arccos(np.clip((np.trace(start_rotation.T @ rotation)-1.)/2., -1., 1.)) <= .3+1e-6
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .6)
        self.assertEqual(caught.exception.code, "planning_failed")
        self.assertIn("pulled to 0.262", str(caught.exception))
        self.assertEqual(len(client.sent), 1)
        attempt = self.store.load("cabinet door")["attempts"][-1]
        self.assertEqual(attempt["status"], "planning_failed")
        self.assertAlmostEqual(attempt["achieved"], 3*self.record["step"], places=9)

    def test_the_wrist_gravity_drift_is_not_contact_but_a_push_is(self):
        # 12 Nm of slow change over the pull (2 Nm a waypoint) is the wrist's own weight turning: no trip.
        client, backend, world = self.arc()
        client.effort_at = lambda progress: (0., 0., 0., 0., 0., 12.*progress, 0.)
        self.assertEqual(self.follow(backend, world, .5).status, "succeeded")
        self.assertEqual(client.guards_seen[-1].renewals, 5)
        # 9 Nm arriving at once is, even on the very tick the pull passes a waypoint: the baseline is not moved onto a push.
        client, backend, world = self.arc()
        client.effort_at = lambda progress: (0., 0., 0., 0., 0., 9. if progress*client.duration >= client.renewal_times[2] else 0., 0.)
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .5)
        self.assertEqual(caught.exception.code, "model_mismatch")
        self.assertIn("stopped at", str(caught.exception))
        attempt = self.store.load("cabinet door")["attempts"][-1]
        self.assertEqual(attempt["status"], "tripped")
        self.assertEqual(attempt["trip"]["kind"], "contact")
        self.assertTrue(0. < attempt["achieved"] < .5)                          # where the pull was, not a whole waypoint
        self.assertEqual(client.cancelled, 0)

    def test_a_cancel_while_planning_the_pull_moves_nothing_and_is_recorded_as_cancelled(self):
        client, backend, world = self.arc()
        context = execution_context(world)
        plan = client.plan_to_pose

        async def cancelled_midway(*args, **kwargs):
            if len(client.plans) == 3:
                context.cancel_event.set()
            return await plan(*args, **kwargs)
        client.plan_to_pose = cancelled_midway
        args = {"entity_id": "handle_1", "constraint_id": "cabinet_door_constraint", "target_value": .5,
                "target_unit": "rad", "profile_id": "bench_contact"}
        with self.assertRaises(BackendFailure) as caught:
            asyncio.run(backend.follow_constraint(args, context))
        self.assertEqual(caught.exception.code, "cancelled")
        self.assertEqual(client.sent, [])
        attempt = self.store.load("cabinet door")["attempts"][-1]
        self.assertEqual((attempt["status"], attempt["achieved"]), ("cancelled", 0.))

    def test_a_grip_closing_during_the_pull_stops_it_as_a_slip(self):
        client, backend, world = self.arc()
        client.knuckle_at = lambda progress: .45 if progress < .5 else .75
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .5)
        self.assertEqual(caught.exception.code, "slip")
        self.assertEqual(self.store.load("cabinet door")["attempts"][-1]["status"], "slipped")

    def test_a_path_off_the_arc_between_waypoints_is_refused_before_anything_moves(self):
        client, backend, world = self.arc()
        client.distort = lambda index, position: position+np.array([0., 0., .015]) if index == 3 else position
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .5)
        self.assertEqual(caught.exception.code, "planning_failed")
        self.assertIn("leaves", str(caught.exception))
        self.assertEqual(client.sent, [])

    def test_slip_and_missing_record_and_unit_mismatch_are_refused(self):
        client = TrackingClient(knuckle=.45)
        backend, world = self.backend(client)
        client.knuckle = .75                                                    # closed far past the grasp: nothing held
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, .2)
        self.assertEqual(caught.exception.code, "slip")
        self.assertEqual(client.sent, [])
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
