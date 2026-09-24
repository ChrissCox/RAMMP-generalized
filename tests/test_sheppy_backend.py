"""The six skills against a fake sheppy arm module; no ROS, no robot."""
import asyncio
import json
from pathlib import Path
import time
import unittest

from rammp_adl.contracts import Catalog, ContractError
from rammp_adl.handlers import BackendFailure, ExecutionContext
from rammp_adl.motion.rolling import JointState, JointTrajectory, TrajectoryPoint
from rammp_adl.motion.sheppy_client import JOINTS, KNUCKLE_CLOSED_RAD, SheppyClientError
from rammp_adl.sheppy_backend import (
    GUARDED_CAPABILITIES, KNOWN_GAPS, PROVIDED_CAPABILITIES, NominalApertureMap, SheppyArmBackend,
    SheppyGeometry, bootstrap_robot_facts)
from rammp_adl.world import MetricPose, WorldModel


ROOT = Path(__file__).resolve().parents[1]
PROFILES = [{"profile_id": "bench_transit", "safety_class": "transit", "simulation_only": False},
            {"profile_id": "bench_gripper", "safety_class": "gripper", "simulation_only": False}]


def run(coroutine):
    return asyncio.new_event_loop().run_until_complete(coroutine)


def context_document():
    """A minimal physical world: one marker-tagged entity with a grasp pose role."""
    base = json.loads((ROOT/"examples/cabinet.context.json").read_text())
    return {**base, "task_id": "bench-task", "snapshot_id": "bench-snapshot-1",
            "attachment_id": "empty", "calibration_id": "bench-cal-1", "base_epoch": "bench-base-1",
            "robot_config_id": "sheppy-gen3-2f85", "profiles": PROFILES, "constraints": [],
            "available_skills": ["move_to_pose", "set_gripper", "grasp", "release"],
            "entities": [
                {"entity_id": "cup_1", "label": "cup on the bench", "entity_revision": 1,
                 "pose_roles": ["grasp", "pregrasp"], "confidence": .9, "age_s": 0,
                 "position_validity": "true", "orientation_validity": "true", "source_ids": ["bench"],
                 "facts": [{"predicate": "entity_exists", "validity": "true", "evidence_id": "e-cup", "age_s": 0,
                            "args": {"entity_id": "cup_1"}}]},
                {"entity_id": "table_1", "label": "bench top", "entity_revision": 1, "pose_roles": ["placement"],
                 "confidence": .9, "age_s": 0, "position_validity": "unknown", "orientation_validity": "unknown",
                 "source_ids": ["bench"],
                 "facts": [{"predicate": "support_verified", "validity": "true", "evidence_id": "e-support", "age_s": 0,
                            "args": {"entity_id": "cup_1", "support_id": "table_1"}}]},
                {"entity_id": "robot", "label": "arm", "entity_revision": 1, "pose_roles": [], "confidence": .9,
                 "age_s": 0, "position_validity": "unknown", "orientation_validity": "unknown", "source_ids": ["bench"],
                 "facts": [{"predicate": "held_state", "validity": "true", "evidence_id": "e-held", "age_s": 0,
                            "args": {"robot_id": "robot"}},
                           {"predicate": "gripper_empty", "validity": "true", "evidence_id": "e-empty", "age_s": 0,
                            "args": {"robot_id": "robot"}}]}]}


def trajectory(start, end):
    return JointTrajectory(JOINTS, (
        TrajectoryPoint(0., JointState(tuple(start), (0.,)*7, (0.,)*7)),
        TrajectoryPoint(1., JointState(tuple((a+b)/2. for a, b in zip(start, end)), (.05,)*7, (0.,)*7)),
        TrajectoryPoint(2., JointState(tuple(end), (0.,)*7, (0.,)*7))), "rammp_curobo:test")


class FakeClient:
    """The client surface, scripted; records every send it would have made."""

    def __init__(self, *, armed=True, joints=(0.,)*7, knuckle=0.):
        self._armed = armed
        self.joints, self.knuckle = tuple(joints), knuckle
        self.still = True
        self.sent, self.gripper_commands, self.cancelled = [], [], 0
        self.plan_error = None
        self.execute_status = "succeeded"
        self.gripper_script = None

    @property
    def motion_enabled(self):
        return self._armed

    def live_joints(self, max_age_s=None):
        return {"position_rad": self.joints, "knuckle_rad": self.knuckle, "effort_nm": (0.,)*7,
                "velocity_rad_s": (0.,)*7, "received_at_monotonic_s": 0.}

    def still_since_s(self):
        return time.monotonic()-10. if self.still else None

    async def stationary(self, **kwargs):
        return self.still

    async def plan_to_pose(self, position_m, quaternion_xyzw, *, cancel_event=None, timeout_s=None):
        if self.plan_error:
            raise SheppyClientError(self.plan_error)
        end = tuple(q+.1 for q in self.joints)
        return trajectory(self.joints, end), {"message": "ok", "planning_time_s": .2}

    async def execute(self, path, *, cancel_event=None, guard=None, **kwargs):
        self.sent.append(path)
        self.guards_seen = getattr(self, "guards_seen", []) + [guard]
        if self.execute_status == "guard_trip":
            return {"status": "guard_trip", "message": "guard tripped", "progress": .4, "error_code": None,
                    "sent": True, "cancel_requested": False, "final_position_rad": list(self.joints),
                    "goal_gap_rad": None, "trip": self.trip}
        if self.execute_status == "succeeded":
            self.joints = tuple(path.points[-1].state.position)
        return {"status": self.execute_status, "message": self.execute_status, "progress": 1.,
                "error_code": 0 if self.execute_status == "succeeded" else -5, "sent": True,
                "cancel_requested": False, "final_position_rad": list(self.joints), "goal_gap_rad": 0.}

    async def gripper(self, knuckle_rad, **kwargs):
        self.gripper_commands.append(knuckle_rad)
        if self.gripper_script is not None:
            outcome = self.gripper_script(knuckle_rad)
        else:
            outcome = {"ok": True, "knuckle_rad": knuckle_rad, "stalled": False, "sent": True, "message": "at target"}
        self.knuckle = outcome["knuckle_rad"] if outcome["knuckle_rad"] is not None else self.knuckle
        return outcome

    async def cancel(self):
        self.cancelled += 1
        return True


def make_world(catalog, *, with_grasp_pose=True):
    world = WorldModel(context_document(), catalog, trust_initial=True, max_evidence_age_s=120.)
    if with_grasp_pose:
        identities = world.snapshot().identities()
        pose = MetricPose("cup_1", "grasp", (.45, .0, .3), (0., 0., 0., 1.), tuple([0.]*36),
                          world.clock(), "base_link", identities["entity:cup_1"]+1,
                          identities["calibration_id"], identities["base_epoch"], "pose-evidence-1", 60.)
        authority = world.authorize_source("test_observation", catalog.predicates)
        world.register_evidence("pose-evidence-1", source=authority, ttl_s=60., observed_at=world.clock(),
                                predicates=[{"predicate": "pose_valid", "validity": "true",
                                             "args": {"entity_id": "cup_1", "pose_role": "grasp"}}])
        world.update_metric_pose(pose, source=authority)
    return world


def execution_context(world, node_id="n1"):
    snapshot = world.snapshot()
    return ExecutionContext(task_id=snapshot.context["task_id"], node_id=node_id,
                            execution_epoch=snapshot.execution_epoch, snapshot=snapshot)


class ApertureMapTests(unittest.TestCase):
    def test_nominal_relation_is_labelled_and_invertible(self):
        m = NominalApertureMap()
        self.assertTrue(m.nominal)
        self.assertAlmostEqual(m.to_knuckle(.085), 0., places=12)
        self.assertAlmostEqual(m.to_knuckle(0.), KNUCKLE_CLOSED_RAD, places=12)
        self.assertAlmostEqual(m.to_aperture(m.to_knuckle(.04)), .04, places=12)
        with self.assertRaises(BackendFailure):
            m.to_knuckle(.09)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(ROOT)
        self.world = make_world(self.catalog)
        self.client = FakeClient()
        self.backend = SheppyArmBackend(self.catalog, self.world, client=self.client, profiles=PROFILES)

    def test_construction_refuses_simulation_profiles_and_non_clients(self):
        with self.assertRaises(ContractError):
            SheppyArmBackend(self.catalog, self.world, client=self.client,
                             profiles=[{"profile_id": "sim", "safety_class": "transit", "simulation_only": True}])
        with self.assertRaises(ContractError):
            SheppyArmBackend(self.catalog, self.world, client=object(), profiles=PROFILES)

    def test_move_to_pose_plans_from_the_snapshot_pose_and_commits_arrival(self):
        outcome = run(self.backend.move_to_pose(
            {"target": {"entity_id": "cup_1", "pose_role": "grasp"}, "profile_id": "bench_transit"},
            execution_context(self.world)))
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(len(self.client.sent), 1)
        predicates = {(f["predicate"], f["args"].get("pose_role")) for f in outcome.proposed_effects}
        self.assertIn(("at_pose", "grasp"), predicates)
        self.assertIn(("at_grasp_pose", None), predicates)
        self.assertFalse(outcome.evidence[0]["data"]["simulation_only"])
        self.assertFalse(outcome.evidence[0]["data"]["support_verified"])
        self.assertEqual(self.backend.current_pose, ("cup_1", "grasp"))

    def test_move_to_pose_refuses_without_a_measured_pose(self):
        world = make_world(self.catalog, with_grasp_pose=False)
        backend = SheppyArmBackend(self.catalog, world, client=self.client, profiles=PROFILES)
        with self.assertRaises(BackendFailure) as caught:
            run(backend.move_to_pose({"target": {"entity_id": "cup_1", "pose_role": "grasp"},
                                      "profile_id": "bench_transit"}, execution_context(world)))
        self.assertEqual(caught.exception.code, "stale_state")
        self.assertEqual(self.client.sent, [])

    def test_move_to_pose_refuses_when_not_armed_or_not_still(self):
        self.client._armed = False
        with self.assertRaises(BackendFailure):
            run(self.backend.move_to_pose({"target": {"entity_id": "cup_1", "pose_role": "grasp"},
                                           "profile_id": "bench_transit"}, execution_context(self.world)))
        self.client._armed, self.client.still = True, False
        with self.assertRaises(BackendFailure) as caught:
            run(self.backend.move_to_pose({"target": {"entity_id": "cup_1", "pose_role": "grasp"},
                                           "profile_id": "bench_transit"}, execution_context(self.world)))
        self.assertEqual(caught.exception.code, "stale_state")
        self.assertEqual(self.client.sent, [])

    def test_a_planner_refusal_and_a_missed_goal_are_reported_not_committed(self):
        self.client.plan_error = "start state in collision"
        with self.assertRaises(BackendFailure) as caught:
            run(self.backend.move_to_pose({"target": {"entity_id": "cup_1", "pose_role": "grasp"},
                                           "profile_id": "bench_transit"}, execution_context(self.world)))
        self.assertEqual(caught.exception.code, "planning_failed")
        self.client.plan_error, self.client.execute_status = None, "goal_not_reached"
        with self.assertRaises(BackendFailure) as caught:
            run(self.backend.move_to_pose({"target": {"entity_id": "cup_1", "pose_role": "grasp"},
                                           "profile_id": "bench_transit"}, execution_context(self.world)))
        self.assertEqual(caught.exception.code, "safety_fault")
        self.assertIsNone(self.backend.current_pose)

    def test_wrong_profile_class_is_refused(self):
        with self.assertRaises(BackendFailure):
            run(self.backend.move_to_pose({"target": {"entity_id": "cup_1", "pose_role": "grasp"},
                                           "profile_id": "bench_gripper"}, execution_context(self.world)))

    def test_set_gripper_maps_aperture_through_the_nominal_relation(self):
        outcome = run(self.backend.set_gripper({"aperture_m": .04, "profile_id": "bench_gripper"},
                                               execution_context(self.world)))
        self.assertEqual(outcome.status, "succeeded")
        self.assertAlmostEqual(self.client.gripper_commands[0], NominalApertureMap().to_knuckle(.04), places=12)
        self.assertTrue(outcome.evidence[0]["data"]["aperture_map"]["nominal"])
        self.assertEqual(outcome.proposed_effects[0]["predicate"], "aperture_reached")

    def test_set_gripper_refuses_a_stall_and_a_retained_grip(self):
        self.client.gripper_script = lambda k: {"ok": True, "knuckle_rad": k+.2, "stalled": True, "sent": True, "message": "stall"}
        with self.assertRaises(BackendFailure):
            run(self.backend.set_gripper({"aperture_m": .04, "profile_id": "bench_gripper"},
                                         execution_context(self.world)))
        self.client.gripper_script = None
        self.backend.holding_id = "cup_1"
        with self.assertRaises(BackendFailure):
            run(self.backend.set_gripper({"aperture_m": .04, "profile_id": "bench_gripper"},
                                         execution_context(self.world)))

    def test_grasp_requires_a_stall_short_of_closed(self):
        self.backend.current_pose = ("cup_1", "grasp")
        self.client.gripper_script = lambda k: {"ok": True, "knuckle_rad": .55, "stalled": True, "sent": True, "message": "stall"}
        outcome = run(self.backend.grasp({"entity_id": "cup_1", "profile_id": "bench_gripper"},
                                         execution_context(self.world)))
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(self.backend.holding_id, "cup_1")
        self.assertIn(("holding", "true"), {(f["predicate"], f["validity"]) for f in outcome.proposed_effects})
        self.assertIn(("gripper_empty", "false"), {(f["predicate"], f["validity"]) for f in outcome.proposed_effects})

    def test_grasp_that_closes_fully_is_an_empty_grasp(self):
        self.backend.current_pose = ("cup_1", "grasp")
        self.client.gripper_script = lambda k: {"ok": True, "knuckle_rad": KNUCKLE_CLOSED_RAD, "stalled": False, "sent": True, "message": "at target"}
        with self.assertRaises(BackendFailure) as caught:
            run(self.backend.grasp({"entity_id": "cup_1", "profile_id": "bench_gripper"},
                                   execution_context(self.world)))
        self.assertEqual(caught.exception.code, "empty_grasp")
        self.assertIsNone(self.backend.holding_id)

    def test_a_miss_and_a_jam_reopen_the_hand_and_say_which(self):
        for knuckle, stalled, words in ((KNUCKLE_CLOSED_RAD, False, "closed fully"), (.05, True, "pressed on the part")):
            self.backend.current_pose = ("cup_1", "grasp")
            self.client.gripper_commands.clear()
            self.client.gripper_script = lambda k, knuckle=knuckle, stalled=stalled: (
                {"ok": True, "knuckle_rad": 0., "stalled": False, "sent": True, "message": "open"} if k == 0. else
                {"ok": True, "knuckle_rad": knuckle, "stalled": stalled, "sent": True, "message": "closed"})
            with self.assertRaises(BackendFailure) as caught:
                run(self.backend.grasp({"entity_id": "cup_1", "profile_id": "bench_gripper"}, execution_context(self.world)))
            self.assertEqual(caught.exception.code, "empty_grasp")
            self.assertIn(words, str(caught.exception))
            self.assertEqual(self.client.gripper_commands[-1], 0.)                # reopened for the retry
            self.assertIsNone(self.backend.holding_id)

    def test_a_refused_start_backs_out_along_the_last_path_once_and_plans_again(self):
        args = {"target": {"entity_id": "cup_1", "pose_role": "grasp"}, "profile_id": "bench_transit"}
        run(self.backend.move_to_pose(args, execution_context(self.world)))
        way_in, arrived = self.client.sent[-1], self.client.joints
        calls = {"n": 0}
        plan = self.client.plan_to_pose

        async def refuses_once(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise SheppyClientError("planner refused: INVALID_START_STATE_WORLD_COLLISION: start in collision")
            return await plan(*a, **k)
        self.client.plan_to_pose = refuses_once
        run(self.backend.move_to_pose(args, execution_context(self.world, "n2")))
        backed = self.client.sent[-2]
        self.assertIn("reversed", backed.provenance)
        self.assertEqual(backed.points[0].state.position, way_in.points[-1].state.position)
        self.assertEqual(backed.points[0].state.position, arrived)
        self.assertEqual(calls["n"], 2)
        # No way in to retrace, or a part in the hand: the refusal stands.
        self.backend.last_trajectory, calls["n"] = None, 0
        with self.assertRaises(BackendFailure) as caught:
            run(self.backend.move_to_pose(args, execution_context(self.world, "n3")))
        self.assertEqual(caught.exception.code, "planning_failed")

    def test_grasp_away_from_the_grasp_pose_is_refused(self):
        with self.assertRaises(BackendFailure) as caught:
            run(self.backend.grasp({"entity_id": "cup_1", "profile_id": "bench_gripper"},
                                   execution_context(self.world)))
        self.assertEqual(caught.exception.code, "empty_grasp")
        self.assertEqual(self.client.gripper_commands, [])

    def test_release_opens_and_clears_the_hold_without_claiming_support(self):
        self.backend.holding_id = "cup_1"
        outcome = run(self.backend.release({"entity_id": "cup_1", "support_id": "table_1", "profile_id": "bench_gripper"},
                                           execution_context(self.world)))
        self.assertEqual(outcome.status, "succeeded")
        self.assertIsNone(self.backend.holding_id)
        self.assertEqual(self.client.gripper_commands, [0.])
        self.assertFalse(outcome.evidence[0]["data"]["support_detected"])
        with self.assertRaises(BackendFailure):
            run(self.backend.release({"entity_id": "cup_1", "support_id": "table_1", "profile_id": "bench_gripper"},
                                     execution_context(self.world)))

    def test_follow_constraint_needs_a_contact_profile_and_an_installed_record(self):
        with self.assertRaises(BackendFailure) as caught:
            run(self.backend.follow_constraint({"entity_id": "cup_1", "constraint_id": "c", "target_value": .1,
                                                "target_unit": "m", "profile_id": "bench_transit"},
                                               execution_context(self.world)))
        self.assertIn("contact profile", str(caught.exception))
        self.assertEqual(self.backend.constraints, {})
        self.assertNotIn("contact_monitor", SheppyArmBackend(self.catalog, self.world, client=self.client,
                                                              profiles=PROFILES).provided_capabilities)

    def test_observe_without_an_observer_is_refused(self):
        with self.assertRaises(BackendFailure) as caught:
            run(self.backend.observe({"entity_id": "cup_1", "camera": "wrist", "purpose": "pose"},
                                     execution_context(self.world)))
        self.assertEqual(caught.exception.code, "no_detection")

    def test_stop_cancels_the_client_and_quiescence_follows_the_arm(self):
        run(self.backend.stop_skill("move_to_pose", "test"))
        self.assertEqual(self.client.cancelled, 1)
        self.assertTrue(run(self.backend.quiescent()))
        self.client.still = False
        self.assertFalse(run(self.backend.quiescent()))

    def test_a_motion_skill_gets_a_bounded_moment_to_come_to_rest_and_says_why_when_it_does_not(self):
        # Just after arriving near a contact the joints can ring for a moment: the strict still window restarts.
        arrived = time.monotonic()
        self.client.still_since_s = lambda: max(arrived+.3, time.monotonic()-10.) if time.monotonic() > arrived+.3 else None
        self.assertTrue(run(self.backend.skill_quiescent("move_to_pose")))
        self.assertGreaterEqual(time.monotonic()-arrived, .3+self.backend.stationary_duration_s-.05)
        logged = []
        self.backend.log = logged.append
        self.client.still_since_s = lambda: None                                        # never comes to rest
        started = time.monotonic()
        self.assertFalse(run(self.backend.skill_quiescent("grasp")))
        self.assertLess(time.monotonic()-started, self.backend.quiescence_grace_s+.2)
        self.assertIn("not at rest", logged[-1])
        self.assertTrue(run(self.backend.skill_quiescent("observe")))                   # observing moves nothing

    def test_declared_capabilities_name_their_gaps(self):
        registry = self.backend.registry(capabilities=PROVIDED_CAPABILITIES | {"live_collision_guard"},
                                         commissioned=False)
        self.assertEqual(set(self.backend.declared_gaps), {"live_collision_guard"})
        self.assertTrue(PROVIDED_CAPABILITIES.isdisjoint(KNOWN_GAPS))
        # Nothing registers on hardware without the operator's commissioned flag.
        self.assertEqual(registry.available_skills, ())

    def test_commissioned_registration_admits_only_skills_whose_capabilities_are_declared(self):
        # Honest capabilities alone register nothing: grasp needs calibrated_grasp
        # (a known gap) and observe has no observer configured here.
        honest = self.backend.registry(capabilities=PROVIDED_CAPABILITIES, commissioned=True)
        self.assertEqual(honest.available_skills, ())
        self.assertIn("missing capabilities", honest.unavailable["move_to_pose"])
        self.assertIn("missing capabilities", honest.unavailable["grasp"])
        self.assertEqual(honest.unavailable["observe"], "no implemented handler")
        # Declaring a gap by name admits the skill and is recorded as a declared gap.
        declared = self.backend.registry(capabilities=PROVIDED_CAPABILITIES | {"calibrated_grasp"},
                                         commissioned=True)
        self.assertIn("grasp", declared.available_skills)
        self.assertEqual(set(self.backend.declared_gaps), {"calibrated_grasp"})
        self.assertNotIn("follow_constraint", declared.available_skills)
        self.assertEqual(declared.unavailable["follow_constraint"],
                         "missing capabilities: contact_monitor, continuous_trajectory_handoff, "
                         "curobo_constrained_path, curobo_online_replanning, live_collision_guard")


class GuardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(ROOT)
        self.world = make_world(self.catalog)
        self.client = FakeClient()

    def backend(self, factory, *, collision_guarded=None):
        guarded = factory is not None if collision_guarded is None else collision_guarded
        return SheppyArmBackend(self.catalog, self.world, client=self.client, profiles=PROFILES,
                                guard_factory=factory, collision_guarded=guarded)

    def move(self, backend):
        return run(backend.move_to_pose({"target": {"entity_id": "cup_1", "pose_role": "grasp"},
                                         "profile_id": "bench_transit"}, execution_context(self.world)))

    def test_a_fresh_guard_is_handed_to_every_move(self):
        made = []
        backend = self.backend(lambda: made.append(object()) or made[-1])
        self.move(backend)
        self.assertEqual(len(made), 1)
        self.assertIs(self.client.guards_seen[0], made[0])
        self.assertIn("live_collision_guard", backend.provided_capabilities)
        self.assertEqual(GUARDED_CAPABILITIES, {"live_collision_guard"})

    def test_without_a_factory_the_guard_capability_is_a_declared_gap(self):
        backend = self.backend(None)
        self.move(backend)
        self.assertEqual(self.client.guards_seen, [None])
        self.assertNotIn("live_collision_guard", backend.provided_capabilities)
        backend.registry(capabilities=PROVIDED_CAPABILITIES | {"live_collision_guard"}, commissioned=True)
        self.assertEqual(set(backend.declared_gaps), {"live_collision_guard"})

    def test_a_collision_trip_is_stale_state_and_contact_is_a_safety_fault(self):
        backend = self.backend(lambda: object())
        self.client.execute_status = "guard_trip"
        self.client.trip = {"kind": "collision", "distance_m": .01, "link": "bracelet_link",
                            "point_base_m": [.4, 0., .3], "margin_m": .03, "time_s": .8, "obstacle_points": 3}
        with self.assertRaises(BackendFailure) as caught:
            self.move(backend)
        self.assertEqual(caught.exception.code, "stale_state")
        self.assertEqual(caught.exception.evidence[0]["data"]["trip"]["kind"], "collision")
        self.assertIsNone(backend.current_pose)
        for kind in ("contact", "depth_blind", "depth_stale", "state_stale"):
            self.client.trip = {"kind": kind}
            with self.assertRaises(BackendFailure) as caught:
                self.move(backend)
            self.assertEqual(caught.exception.code, "safety_fault", kind)

    def test_a_non_callable_factory_is_refused(self):
        with self.assertRaises(ContractError):
            self.backend("not callable")

    def test_an_effort_only_guard_does_not_claim_the_collision_capability(self):
        backend = self.backend(lambda: object(), collision_guarded=False)
        self.assertNotIn("live_collision_guard", backend.provided_capabilities)
        with self.assertRaises(ContractError):
            SheppyArmBackend(self.catalog, self.world, client=self.client, profiles=PROFILES,
                             guard_factory=None, collision_guarded=True)


class GeometryTests(unittest.TestCase):
    def test_geometry_establishes_only_physical_profiles(self):
        catalog = Catalog(ROOT)
        world = make_world(catalog)
        backend = SheppyArmBackend(catalog, world, client=FakeClient(), profiles=PROFILES)
        geometry = SheppyGeometry(backend)
        snapshot = world.snapshot()
        check = geometry.validate({"skill": "move_to_pose", "args": {"target": {"entity_id": "cup_1", "pose_role": "grasp"},
                                                                     "profile_id": "bench_transit"}},
                                  snapshot, {"geometry": None}, "admission")
        self.assertEqual(check.established_facts[0]["predicate"], "motion_profile_valid")
        self.assertEqual(check.dependencies["execution_epoch"], snapshot.identities()["execution_epoch"])
        self.assertFalse(check.artifact["simulation_only"])
        with self.assertRaises(ContractError):
            geometry.validate({"skill": "move_to_pose", "args": {"profile_id": "absent"}}, snapshot, {}, "admission")
        with self.assertRaises(ValueError):
            SheppyGeometry(object())


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_asserts_measured_hold_and_declared_emptiness(self):
        catalog = Catalog(ROOT)
        world = WorldModel({**context_document(), "entities": [
            {"entity_id": "robot", "label": "arm", "entity_revision": 1, "pose_roles": [], "confidence": .9,
             "age_s": 0, "position_validity": "unknown", "orientation_validity": "unknown", "source_ids": ["bench"],
             "facts": []}]}, catalog, max_evidence_age_s=120.)
        result = run(bootstrap_robot_facts(world, FakeClient(knuckle=.01)))
        self.assertTrue(result["gripper_empty_asserted"])
        self.assertEqual(world.snapshot().fact("held_state", {"robot_id": "robot"}), "true")
        self.assertEqual(world.snapshot().fact("gripper_empty", {"robot_id": "robot"}), "true")

    def test_bootstrap_refuses_a_moving_arm_and_withholds_emptiness_when_closed(self):
        catalog = Catalog(ROOT)
        world = WorldModel({**context_document(), "entities": [
            {"entity_id": "robot", "label": "arm", "entity_revision": 1, "pose_roles": [], "confidence": .9,
             "age_s": 0, "position_validity": "unknown", "orientation_validity": "unknown", "source_ids": ["bench"],
             "facts": []}]}, catalog, max_evidence_age_s=120.)
        moving = FakeClient()
        moving.still = False
        with self.assertRaises(ContractError):
            run(bootstrap_robot_facts(world, moving))
        result = run(bootstrap_robot_facts(world, FakeClient(knuckle=.5)))
        self.assertFalse(result["gripper_empty_asserted"])
        self.assertEqual(world.snapshot().fact("gripper_empty", {"robot_id": "robot"}), "unknown")


if __name__ == "__main__":
    unittest.main()
