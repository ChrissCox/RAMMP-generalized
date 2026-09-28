"""Guarded constraint following on the sheppy backend, time scaling, contact exclusions."""
import asyncio
import json
import math
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
from rammp_adl.motion.sheppy_client import (JOINT_VMAX, JOINTS, KNUCKLE_CLOSED_RAD, SheppyClientError, executor_problems,
                                             scale_trajectory_time, wrap_diff)
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
    for _ in range(80):
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
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T+1e-9*np.eye(6), error)
        q += step*min(1., .2/max(np.abs(step).max(), 1e-12))            # bounded steps, as a planner's would be
    seed = np.asarray(seed, dtype=float)
    for joint in (0, 2, 4, 6):                                          # continuous joints: the turn nearest the start
        q[joint] = seed[joint]+(q[joint]-seed[joint]+np.pi) % (2.*np.pi)-np.pi
    frame = tool_frame(chain, q)
    if np.linalg.norm(position-frame[:3, 3]) > 1e-4 or np.linalg.norm(Rotation.from_matrix(rotation @ frame[:3, :3].T).as_rotvec()) > 1e-4:
        raise SheppyClientError(IK_FAIL)                                # out of reach: the planner says so too
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
                    self.joints = tuple(float(v) for v in path.sample(progress*path.duration_s).position)   # stopped there
                    return {"status": "guard_trip", "message": f"guard tripped ({trip.get('kind')})", "progress": float(progress),
                            "error_code": None, "sent": True, "cancel_requested": False, "final_position_rad": None,
                            "goal_gap_rad": None, "trip": trip}
        self.joints = tuple(path.points[-1].state.position)
        return {"status": "succeeded", "message": "SUCCESSFUL", "progress": 1., "error_code": 0, "sent": True,
                "cancel_requested": False, "final_position_rad": list(self.joints), "goal_gap_rad": 0.}


class YieldingClient(ArcClient):
    """A driver in impedance mode against a stiff door: the hand ends up on the door's own circle, not the plan's."""
    def __init__(self, chain, *, true_pivot, axis, **kwargs):
        super().__init__(chain, **kwargs)
        self.true_pivot, self.axis = np.asarray(true_pivot, dtype=float), np.asarray(axis, dtype=float)/np.linalg.norm(axis)
        self.options_seen = []

    def on_the_door(self, desired, radius):
        frame = tool_frame(self.chain, desired)
        offset = frame[:3, 3]-self.true_pivot
        along = (offset @ self.axis)*self.axis
        flat = offset-along
        return solve(self.chain, desired, self.true_pivot+along+flat/np.linalg.norm(flat)*radius, frame[:3, :3])

    async def execute(self, path, *, cancel_event=None, guard=None, impedance=None, path_tolerance_rad=None, **kwargs):
        self.sent.append(path)
        self.guards_seen = getattr(self, "guards_seen", [])+[guard]
        self.options_seen.append({"impedance": impedance, "path_tolerance_rad": path_tolerance_rad, **kwargs})
        offset = tool_frame(self.chain, self.joints)[:3, 3]-self.true_pivot
        radius = np.linalg.norm(offset-(offset @ self.axis)*self.axis)
        actual = self.joints
        for progress in np.linspace(.02, 1., 50):
            desired = path.sample(progress*path.duration_s).position
            actual = tuple(float(v) for v in self.on_the_door(desired, radius)) if impedance is not None else tuple(desired)
            live = {**self.live_joints(), "position_rad": actual, "tracking_error_rad": tuple(np.subtract(desired, actual))}
            guard.on_progress(progress)
            trip = guard.check(live=live, trajectory=path, elapsed_s=progress*path.duration_s, now=time.monotonic())
            if trip is not None:
                self.joints = actual
                return {"status": "guard_trip", "message": f"guard tripped ({trip.get('kind')})", "progress": float(progress),
                        "error_code": None, "sent": True, "cancel_requested": False, "final_position_rad": list(actual),
                        "goal_gap_rad": None, "trip": trip}
        self.joints = actual
        return {"status": "succeeded", "message": "SUCCESSFUL", "progress": 1., "error_code": 0, "sent": True,
                "cancel_requested": False, "final_position_rad": list(actual), "goal_gap_rad": 0.}


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class DoorScene:
    """The wrist camera's view of the door face: its normal turned about the door's own axis as far as the door has.

    The door's own axis is the placed hinge's leaned sideways within the face by lean_deg, which no single view shows.
    The door turns with the hand; let go, it springs back by spring_back and is gripped again there.
    """
    def __init__(self, backend, client, *, lean_deg=2., spring_back=0.):
        from rammp_adl.constraints import rotation_about
        self.backend, self.client, self.rotation_about = backend, client, rotation_about
        self.pivot = np.asarray(HINGE["pivot_base"], dtype=float)
        self.axis = np.asarray(HINGE["axis_base"], dtype=float)/np.linalg.norm(HINGE["axis_base"])
        self.lean_deg, self.spring_back, self.offset = lean_deg, spring_back, 0.
        self.start = self.n0 = self.true_axis = None
        self.looks = []

    def _flat(self):
        tool = np.asarray(self.backend._tool_pose(self.client.joints)[0])-self.pivot
        return tool-(tool @ self.axis)*self.axis

    async def surface_normal(self, **kwargs):
        flat = self._flat()
        if self.start is None:
            self.start = flat
            n0 = np.cross(self.axis, flat/np.linalg.norm(flat))
            self.n0 = n0 if n0[0] < 0 else -n0                         # the face looks back at the robot
            self.true_axis = self.rotation_about(self.n0, math.radians(self.lean_deg)) @ self.axis
        turned = abs(math.atan2(np.cross(self.start, flat) @ self.axis, self.start @ flat))
        if self.client.knuckle < .1:
            self.offset = self.spring_back                              # let go: the door relaxes back
        door = turned-self.offset
        self.looks.append(door)
        return list(self.rotation_about(self.true_axis, HINGE["direction"]*door) @ self.n0), None

    def crop_for(self, keyframe):
        raise RuntimeError("no frames in this test")


def regripping(knuckle):
    closing = knuckle > .5
    return {"ok": True, "knuckle_rad": .45 if closing else 0., "stalled": closing, "sent": True,
            "message": "moved then settled short of the target" if closing else "at target"}


class BackendCase(unittest.TestCase):
    """The bench's catalog, kinematic model and door record behind a sheppy backend, with every guard recorded."""
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


class FollowConstraintTests(BackendCase):
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

    def test_the_hinge_is_placed_from_the_close_view_not_from_discovery(self):
        # Run 8 on the bench: discovery put the hinge 17 mm in front of and 7 mm short of the line the door swung
        # about, with its axis tilted 1.6 degrees; the arm dragged the pull off the door's arc until 5 Nm tripped.
        from rammp_adl.constraints import rotation_about
        client, backend, world = self.arc()
        position, orientation = backend._tool_pose(GRASP_JOINTS)
        rotation = quaternion_matrix(orientation)
        p0, normal = np.asarray(position), -rotation[:, 2]                         # the face, square to the approach
        face = p0-normal*.015                                                        # the pull stands 15 mm proud of it
        up = np.array([0., 0., 1.])-normal[2]*normal
        up /= np.linalg.norm(up)
        toward = np.asarray(HINGE["pivot_base"])-p0
        toward -= (toward @ normal)*normal+(toward @ up)*up
        toward /= np.linalg.norm(toward)
        true_pivot = face+toward*(.25-backend.hinge_inset_m)-normal*backend.hinge_depth_m
        self.record.update(pivot_base=(true_pivot+normal*.017-toward*.007).tolist(),
                           axis_base=(rotation_about(toward, np.radians(1.6)) @ up).tolist(),
                           handle_position_m=p0.tolist(),
                           measured_door={"entity_id": "cabinet_door_surface", "width_m": .27, "height_m": .43,
                                          "handle_offsets_m": {"left": .25, "right": .02, "top": .3, "bottom": .13}})
        backend.contact_support = (rotation.T @ (face-p0), rotation.T @ normal)       # the standoff's view, in the tool frame
        outcome = self.follow(backend, world, .5)
        self.assertEqual(outcome.status, "succeeded")
        hinge = outcome.evidence[0]["data"]["hinge"]
        self.assertEqual(hinge["placed_by"], "close view")
        self.assertAlmostEqual(hinge["moved_mm"], np.hypot(17., 7.), delta=.5)
        np.testing.assert_allclose(hinge["pivot_base"], true_pivot, atol=1e-4)
        radii = []
        for time_s in np.linspace(0., client.sent[0].duration_s, 80):
            offset = np.asarray(backend._tool_pose(client.sent[0].sample(time_s).position)[0])-true_pivot
            radii.append(np.linalg.norm(offset-(offset @ up)*up))
        self.assertLess(max(radii)-min(radii), .002)                                  # on the door's own circle
        self.assertEqual(self.store.load("cabinet door")["pivot_base"], self.record["pivot_base"])   # the stored record is not rewritten

    def compliant_door(self, shift):
        """The bench door's hinge as the record has it; the real one `shift` metres away in the plane."""
        from rammp_adl.constraints import rotation_about
        axis = np.asarray(HINGE["axis_base"])/np.linalg.norm(HINGE["axis_base"])
        truth = np.asarray(HINGE["pivot_base"])+np.asarray(shift)
        client = YieldingClient(self.chain, true_pivot=truth, axis=axis, knuckle=.45)
        self.record.update(HINGE)
        backend, world = self.backend(client)
        backend.compliant_pull = True
        return client, backend, world, truth, axis

    def test_a_compliant_pull_finds_the_real_hinge_and_finishes_on_it(self):
        client, backend, world, truth, axis = self.compliant_door([.012, -.009, 0.])      # 15 mm from the modelled hinge
        outcome = self.follow(backend, world, 1.)
        self.assertEqual(outcome.status, "succeeded")
        data = outcome.evidence[0]["data"]
        pull = data["pull"]
        self.assertTrue(pull["compliant"])
        self.assertEqual(len(pull["stretches"]), 2)                                   # a first stretch, then the rest re-planned
        first, rest = client.options_seen
        self.assertEqual((first["impedance"], first["path_tolerance_rad"]), (backend.pull_impedance, backend.pull_path_tolerance_rad))
        # The rest takes joint 7 past 2.89 rad, where the driver's impedance reference wraps the long way: flown stiffly, on the fit.
        self.assertIsNone(rest["impedance"])
        self.assertEqual([stretch["compliant"] for stretch in pull["stretches"]], [True, False])
        fitted = np.asarray(pull["stretches"][-1]["fit"]["pivot"])
        self.assertLess(np.linalg.norm((fitted-truth)-((fitted-truth) @ axis)*axis), .004)
        self.assertAlmostEqual(data["achieved"], 1., delta=.03)                         # measured, not commanded
        self.assertGreater(pull["peak_force_n"], 0.)
        self.assertLess(pull["peak_force_n"], backend.pull_force_limit_n)

        def off_the_door(path):
            worst = 0.
            for time_s in np.linspace(0., path.duration_s, 40):
                offset = tool_frame(self.chain, path.sample(time_s).position)[:3, 3]-truth
                flat = offset-(offset @ axis)*axis
                worst = max(worst, abs(np.linalg.norm(flat)-np.linalg.norm(
                    (tool_frame(self.chain, GRASP_JOINTS)[:3, 3]-truth)-((tool_frame(self.chain, GRASP_JOINTS)[:3, 3]-truth) @ axis)*axis)))
            return worst
        self.assertLess(off_the_door(client.sent[1]), off_the_door(client.sent[0])/2)   # the re-planned stretch follows the door

    def test_a_door_that_pushes_back_past_the_force_limit_stops_the_pull(self):
        client, backend, world, truth, axis = self.compliant_door([.03, -.03, 0.])
        backend.pull_force_limit_n = 5.
        with self.assertRaises(BackendFailure) as caught:
            self.follow(backend, world, 1.)
        self.assertEqual(caught.exception.code, "model_mismatch")
        attempt = self.store.load("cabinet door")["attempts"][-1]
        self.assertEqual(attempt["trip"]["kind"], "contact")
        self.assertGreater(attempt["trip"]["force_n"], 5.)
        self.assertEqual(len(client.sent), 1)

    def test_the_spring_force_reads_a_joint_across_pi_as_near_not_a_turn_away(self):
        backend, _ = self.backend(TrackingClient(knuckle=.45))
        start = (0., .262, -3.1414, -2.269, 0., .96, 1.571)                   # the bench start pose: joint 3 on the wrap
        from rammp_adl.motion.rolling import JointState, JointTrajectory, TrajectoryPoint
        still = JointState(start, (0.,)*7, (0.,)*7)
        path = JointTrajectory(JOINTS, (TrajectoryPoint(0., still), TrajectoryPoint(1., still)), "rammp_curobo:test")
        guard = PullGuard(None, stiffness=backend.pull_impedance["kq"], force_limit_n=25., jacobian=backend._tool_jacobian)
        wrapped = (0., .262, 3.1414, -2.269, 0., .96, 1.571)                  # the same pose, reported the other side of pi
        live = {"position_rad": start, "knuckle_rad": None, "effort_nm": None}
        guard.on_progress(0.)
        self.assertIsNone(guard.check(live=live, trajectory=path, elapsed_s=0., now=0.))
        self.assertIsNone(guard.check(live={**live, "position_rad": wrapped}, trajectory=path, elapsed_s=.5, now=0.))
        self.assertLess(guard.peak_force_n, 1.)

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

    def test_a_pull_that_stopped_part_way_continues_from_there_when_asked_again(self):
        client, backend, world = self.arc()
        client.effort_at = lambda progress: (0., 0., 0., 0., 0., 9. if progress*client.duration >= client.renewal_times[2] else 0., 0.)
        with self.assertRaises(BackendFailure):
            self.follow(backend, world, .5)
        first = backend.constraint_progress["cabinet_door_constraint"]
        self.assertTrue(.1 < first < .4, first)
        stopped_at = client.joints
        client.effort_at = None
        outcome = self.follow(backend, world, .5)                                # the same absolute goal, asked again
        data = outcome.evidence[0]["data"]
        self.assertAlmostEqual(data["achieved"], .5, places=6)
        self.assertEqual(data["hinge"], {"placed_by": "earlier pull this task"})
        retry = client.sent[-1]
        np.testing.assert_allclose(retry.points[0].state.position, stopped_at, atol=1e-9)
        pivot, axis = np.asarray(HINGE["pivot_base"]), np.asarray(HINGE["axis_base"])/np.linalg.norm(HINGE["axis_base"])
        ends = [np.asarray(backend._tool_pose(p.state.position)[0])-pivot for p in (retry.points[0], retry.points[-1])]
        flat = [v-(v @ axis)*axis for v in ends]
        turned = np.arctan2(np.cross(flat[0], flat[1]) @ axis, flat[0] @ flat[1])
        self.assertAlmostEqual(abs(turned), .5-first, delta=.01)                 # only what was left
        self.assertAlmostEqual(backend.constraint_progress["cabinet_door_constraint"], .5, places=6)
        again = self.follow(backend, world, .5)                                  # nothing left: no motion
        self.assertEqual(len(client.sent), 2)
        self.assertEqual(again.status, "succeeded")

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

    def radius_about(self, path, backend, axis):
        pivot, axis = np.asarray(HINGE["pivot_base"]), np.asarray(axis)/np.linalg.norm(axis)
        radii = []
        for time_s in np.linspace(0., path.duration_s, 60):
            offset = np.asarray(backend._tool_pose(path.sample(time_s).position)[0])-pivot
            radii.append(float(np.linalg.norm(offset-(offset @ axis)*axis)))
        return np.asarray(radii)

    def test_a_stiff_pull_stops_short_so_the_camera_sees_the_hinge_lean_and_the_rest_follows_it(self):
        client, backend, world = self.arc()
        backend.scene = scene = DoorScene(backend, client, lean_deg=2.)
        outcome = self.follow(backend, world, .9)
        self.assertEqual(outcome.status, "succeeded")
        data = outcome.evidence[0]["data"]
        self.assertEqual(len(client.sent), 2)                                   # to the check, then the rest
        self.assertAlmostEqual(data["pull"]["stretches"][0]["commanded"], .35, places=6)
        refit = data["pull"]["refits"][0]
        self.assertTrue(refit["fitted"])
        self.assertFalse(refit["let_go"])
        self.assertAlmostEqual(refit["axis_change_deg"], 2., delta=.05)
        np.testing.assert_allclose(refit["axis_base"], scene.true_axis, atol=1e-4)
        rest = self.radius_about(client.sent[1], backend, scene.true_axis)
        self.assertLess(rest.max()-rest.min(), .001)                            # the rest turns about the door's own axis
        self.assertAlmostEqual(data["achieved"], .9, places=6)
        self.assertTrue(data["verified_locally"])
        np.testing.assert_allclose(backend.constraint_arcs["cabinet_door_constraint"]["axis_base"], refit["axis_base"], atol=1e-5)

    def test_a_pull_the_door_stops_lets_go_sees_where_it_went_grips_again_and_finishes(self):
        client, backend, world = self.arc()
        backend.scene = DoorScene(backend, client, lean_deg=2., spring_back=.03)
        client.gripper_script = regripping
        client.effort_at = lambda progress: (0.,)*5+(9. if len(client.sent) == 1 and progress > .6 else 0., 0.)
        outcome = self.follow(backend, world, .9)
        self.assertEqual(outcome.status, "succeeded")
        data = outcome.evidence[0]["data"]
        self.assertEqual(client.gripper_commands, [0., KNUCKLE_CLOSED_RAD])     # let go, then grip again: the arm stays
        refit = data["pull"]["refits"][0]
        self.assertTrue(refit["let_go"] and refit["fitted"])
        self.assertAlmostEqual(refit["turned"], data["pull"]["stretches"][0]["commanded"]-.03, delta=.002)
        self.assertEqual(backend.grasp_knuckle, .45)
        self.assertAlmostEqual(data["achieved"], .9, places=6)                  # the door's own angle, not the hand's
        self.assertTrue(data["verified_locally"])
        self.assertEqual(data["pull"]["stretches"][0]["status"], "guard_trip")

    def test_a_part_gone_from_the_fingers_when_let_go_is_a_slip(self):
        client, backend, world = self.arc()
        backend.scene = DoorScene(backend, client, spring_back=.2)
        client.gripper_script = lambda knuckle: {"ok": True, "knuckle_rad": .8 if knuckle > .5 else 0., "stalled": False,
                                                  "sent": True, "message": "at target"}
        client.effort_at = lambda progress: (0.,)*5+(9. if progress > .6 else 0., 0.)
        with self.assertRaises(BackendFailure) as failed:
            self.follow(backend, world, .9)
        self.assertEqual(failed.exception.code, "slip")
        self.assertIsNone(backend.holding_id)

    def test_a_door_this_task_opened_is_closed_on_the_same_hinge_and_its_handle_poses_move_with_it(self):
        client, backend, world = self.arc()
        shut_tool = np.asarray(backend._tool_pose(GRASP_JOINTS)[0])
        self.assertEqual(self.follow(backend, world, .9).status, "succeeded")
        arc = dict(backend.constraint_arcs["cabinet_door_constraint"])
        moved, _ = backend._moved_part_pose("handle_1", tuple(shut_tool), (0., 0., 0., 1.))
        open_tool = np.asarray(backend._tool_pose(client.joints)[0])
        self.assertLess(np.linalg.norm(np.asarray(moved)-open_tool), .005)       # the grasp measured shut is where the handle is now
        closed = self.follow(backend, world, 0.)
        self.assertEqual(closed.status, "succeeded")
        data = closed.evidence[0]["data"]
        self.assertAlmostEqual(data["achieved"], 0., places=6)
        self.assertAlmostEqual(backend.constraint_progress["cabinet_door_constraint"], 0., places=6)
        back = np.asarray(backend._tool_pose(client.sent[-1].points[-1].state.position)[0])
        self.assertLess(np.linalg.norm(back-shut_tool), .005)                    # the hand is back where it gripped the shut door
        radii = self.on_the_arc(client.sent[-1], backend)
        self.assertLess(radii.max()-radii.min(), .004)                          # on the door's circle all the way back
        self.assertEqual(backend.constraint_arcs["cabinet_door_constraint"]["direction"], arc["direction"])
        unmoved, _ = backend._moved_part_pose("handle_1", tuple(shut_tool), (0., 0., 0., 1.))
        np.testing.assert_allclose(unmoved, shut_tool, atol=1e-6)

    def test_a_door_an_earlier_task_left_open_is_sought_where_it_was_left_and_closed_from_where_the_hand_grips_it(self):
        from types import SimpleNamespace
        from rammp_adl.constraints import left_moved, rotation_about
        client, backend, world = self.arc()
        shut_tool = np.asarray(backend._tool_pose(GRASP_JOINTS)[0])
        self.assertEqual(self.follow(backend, world, .7).status, "succeeded")
        sprung_back = client.joints                                              # where the door comes to rest once let go
        self.assertEqual(self.follow(backend, world, .9).status, "succeeded")
        open_tool = np.asarray(backend._tool_pose(client.joints)[0])
        # A new task: a fresh backend knows only what the store kept. The last attempt left the door open on its hinge.
        state = left_moved(self.store.load("cabinet door"))
        self.assertAlmostEqual(state["at"], .9, places=6)
        np.testing.assert_allclose(state["pivot_base"], HINGE["pivot_base"], atol=1e-5)
        np.testing.assert_allclose(state["reference"]["tool_m"], shut_tool, atol=1e-9)   # where the hand held it shut
        later, world = self.backend(client)
        face = np.array([-1., 0., 0.])
        later.scene = SimpleNamespace(entities={"handle_1": {"grasp": {"support": {"point_m": list(shut_tool+.03*face),
                                                                                    "normal": list(face)}}}})
        self.assertEqual(later._entity_support("handle_1"), (list(shut_tool+.03*face), list(face)))
        later.inherit_part_state("cabinet_door_constraint", state)
        moved, _ = later._moved_part_pose("handle_1", tuple(shut_tool), (0., 0., 0., 1.))
        self.assertLess(np.linalg.norm(np.asarray(moved)-open_tool), .005)       # the handle is sought where the door was left
        point, normal = later._entity_support("handle_1")                       # and the face it stands on turned with it
        np.testing.assert_allclose(point, later._moved_part_pose("handle_1", tuple(shut_tool+.03*face), (0., 0., 0., 1.))[0], atol=1e-9)
        axis = np.asarray(HINGE["axis_base"])/np.linalg.norm(HINGE["axis_base"])
        np.testing.assert_allclose(normal, rotation_about(axis, HINGE["direction"]*.9) @ face, atol=1e-4)   # the store keeps 5 decimals
        client.joints = sprung_back                                              # the close view put the grip where it came to rest
        logs = []
        later.log = logs.append
        closed = self.follow(later, world, 0.)
        self.assertEqual(closed.status, "succeeded")
        self.assertTrue(any("the grip puts the part at 0.700" in line for line in logs), logs)
        back = np.asarray(later._tool_pose(client.sent[-1].points[-1].state.position)[0])
        self.assertLess(np.linalg.norm(back-shut_tool), .005)                    # shut where it was gripped shut, not .2 rad past
        radii = self.on_the_arc(client.sent[-1], later)
        self.assertLess(radii.max()-radii.min(), .004)                          # on the door's circle all the way back
        self.assertAlmostEqual(later.constraint_progress["cabinet_door_constraint"], 0., places=6)
        self.assertIsNone(left_moved(self.store.load("cabinet door")))           # left shut: the next task carries nothing

    def test_a_rehearsal_plans_each_move_from_where_the_last_ended_and_refuses_what_is_out_of_reach(self):
        from types import SimpleNamespace
        from rammp_adl.learned import run_skill
        from rammp_adl.learned.host import DryRunHost
        from rammp_adl.motion.kinematics import quaternion_xyzw_from_matrix
        client, backend, world = self.arc()
        start = client.joints
        position, orientation = backend._tool_pose(GRASP_JOINTS)
        plain = lambda values: tuple(float(v) for v in values)
        self.install_pose(world, "grasp", plain(position), plain(orientation))   # one pose: a second would re-revise the entity
        runtime = SimpleNamespace(world=world, backend=backend, catalog=self.catalog, validator=SimpleNamespace(admit=lambda plan: None))
        source = 'def run(robot):\n    """Reach the handle twice."""\n    robot.move_to("handle_1", "grasp")\n    robot.move_to("handle_1", "grasp")\n'
        dry = DryRunHost(runtime)
        result = asyncio.run(run_skill(source, {}, dry))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual(len(client.plans), 2)
        self.assertEqual(client.plans[0]["start"], tuple(start))
        self.assertEqual(client.plans[1]["start"], client.ends[0])              # chained, not from the live joints again
        self.assertEqual(client.sent, [])                                       # a rehearsal: nothing flown
        client.reach = lambda at, rotation: False
        refused = asyncio.run(run_skill(source, {}, DryRunHost(runtime)))
        self.assertEqual(refused.status, "failed")
        self.assertIn("out of reach", refused.error)

    def install_pose(self, world, role, position, orientation=(0., 0., 0., 1.)):
        from rammp_adl.world import MetricPose
        identities = world.snapshot().identities()
        authority = world.authorize_source("test_observation", self.catalog.predicates)
        pose = MetricPose("handle_1", role, tuple(position), tuple(orientation), tuple([0.]*36), world.clock(), "base_link",
                          identities["entity:handle_1"]+1, identities["calibration_id"], identities["base_epoch"], f"pose-{role}", 60.)
        world.register_evidence(f"pose-{role}", source=authority, ttl_s=60., observed_at=world.clock(),
                                predicates=[{"predicate": "pose_valid", "validity": "true", "args": {"entity_id": "handle_1", "pose_role": role}}])
        world.update_metric_pose(pose, source=authority)

    def retract(self, backend, world):
        args = {"target": {"entity_id": "handle_1", "pose_role": "retract"}, "profile_id": "bench_transit"}
        return asyncio.run(backend.move_to_pose(args, execution_context(world)))

    def test_after_the_door_swung_the_hand_backs_straight_out_not_to_where_the_handle_was(self):
        client, backend, world = self.arc()
        backend.at_contact = True                                                # parked on the handle by the grasp
        self.assertEqual(self.follow(backend, world, .5).status, "succeeded")
        release = {"entity_id": "handle_1", "support_id": "cabinet_door_surface", "profile_id": "bench_gripper"}
        self.assertEqual(asyncio.run(backend.release(release, execution_context(world))).status, "succeeded")
        self.install_pose(world, "retract", (.5, .1, .3))                        # measured before the door swung
        position, orientation = backend._tool_pose(client.joints)
        approach = quaternion_matrix(orientation)[:, 2]
        full = np.asarray(position)-approach*.10
        client.reach = lambda at, rotation: np.linalg.norm(at-full) > 1e-6       # the full standoff is out of reach
        data = self.retract(backend, world).evidence[0]["data"]
        self.assertEqual(data["way_out"]["back_m"], .07)                          # the next way out the planner reaches
        self.assertEqual(len(data["way_out"]["refused"]), 1)
        np.testing.assert_allclose(data["commanded"]["position_m"], np.asarray(position)-approach*.07, atol=1e-9)
        np.testing.assert_allclose(data["commanded"]["orientation_xyzw"], orientation, atol=1e-12)
        self.assertEqual(client.plans[-1]["position"], tuple(data["commanded"]["position_m"]))   # flown as planned, not planned again
        self.assertEqual(self.guards[-1]["exclusions"][0][1], .10)               # leaving the handle exempts what the fingers were beside
        client.reach = lambda at, rotation: False
        backend.at_contact = False
        with self.assertRaises(BackendFailure) as refused:
            self.retract(backend, world)
        self.assertEqual(refused.exception.code, "planning_failed")

    def test_a_part_that_has_not_moved_is_left_to_its_measured_retract_pose(self):
        client = TrackingClient(knuckle=.01)
        backend, world = self.backend(client)
        backend.holding_id = None
        self.install_pose(world, "retract", (.5, .1, .3))
        data = self.retract(backend, world).evidence[0]["data"]
        self.assertIsNone(data["way_out"])
        self.assertEqual(client.targets, [((.5, .1, .3), (0., 0., 0., 1.))])

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


class HomingClient(TrackingClient):
    """Plans to joints as asked, so the target the arm is sent home to is visible."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.joint_targets = []

    async def plan_to_joints(self, target_joints, *, timeout_s=None, cancel_event=None):
        self.joint_targets.append(tuple(target_joints))
        return trajectory(self.joints, tuple(target_joints)), {"message": "ok", "planning_time_s": .1}


HOME = (0., .262, -3.141, -2.269, 0., .96, 1.571)


class HomingTests(BackendCase):
    """Every task ends at home: what the hand lets go of, how it leaves, and the way round it takes."""

    def homed(self, client, **state):
        backend, _ = self.backend(client)
        backend.home_joints = HOME
        for name, value in state.items():
            setattr(backend, name, value)
        return backend, asyncio.run(backend.return_home())

    def test_a_handle_is_let_go_the_hand_backs_out_and_goes_home_the_short_way_round(self):
        client = HomingClient(joints=(.2, .5, 3.10, -1.5, .1, .9, .5), knuckle=.45)
        backend, report = self.homed(client, at_contact=True)
        self.assertTrue(report["at_home"])
        self.assertEqual(report["done"][:2], ["let go of handle_1", "backed 10 cm out"])
        self.assertEqual(client.gripper_commands, [0.])
        self.assertEqual(len(client.targets), 1)                              # the way out, along the approach
        self.assertEqual(len(client.sent), 2)                                 # out, then home
        self.assertAlmostEqual(client.joint_targets[0][2], 3.10+.02+wrap_diff(-3.141, 3.12), places=9)
        self.assertGreater(client.joint_targets[0][2], 3.)                   # joint 3 does not swing round to -3.141
        self.assertIsNone(backend.holding_id)
        self.assertFalse(backend.at_contact)
        self.assertEqual(self.guards[0]["exclusions"][0][1], .10)            # leaving the handle exempts it once

    def test_at_home_nothing_moves_and_a_free_object_is_carried_home_not_dropped(self):
        client = HomingClient(joints=HOME, knuckle=0.)
        _, report = self.homed(client, holding_id=None, at_contact=False, current_pose=None)
        self.assertEqual(report, {"at_home": True, "done": ["already home"]})
        self.assertEqual((client.sent, client.gripper_commands), ([], []))
        client = HomingClient(joints=(.3, .6, -2.9, -1.4, .2, .9, .6), knuckle=.4)
        backend, report = self.homed(client, holding_id="cup_1", at_contact=False, current_pose=None)
        self.assertTrue(report["at_home"])
        self.assertEqual(client.gripper_commands, [])                        # the cup stays in the hand
        self.assertEqual(self.guards[-1]["tool_exclusion_m"], backend.tool_exclusion_m)
        self.assertEqual(backend.holding_id, "cup_1")

    def test_a_task_is_done_only_with_the_arm_home_and_a_cancel_or_fault_leaves_it_where_it_stopped(self):
        from rammp_adl.ros_bridge import finish_at_home

        class Safety:
            def __init__(self, fault=False, stopped=False):
                self.fault_latched, self.resets = fault, 0
                self.stop_requested = asyncio.Event()
                if stopped:
                    self.stop_requested.set()

            async def reset(self):
                self.resets += 1
                self.stop_requested.clear()

        class Backend:
            home_joints = HOME

            def __init__(self, fails=None):
                self.fails, self.calls = fails, 0

            async def return_home(self):
                self.calls += 1
                if self.fails:
                    raise BackendFailure("planning_failed", self.fails)
                return {"at_home": True, "done": ["home in 3.0 s"]}

        def finish(outcome, backend, safety):
            return asyncio.run(finish_at_home(dict(outcome), backend=backend, safety=safety))
        done = {"task_id": "t", "status": "succeeded", "reason": "Measured task goal satisfied"}
        safety = Safety(stopped=True)
        home = finish(done, Backend(), safety)
        self.assertEqual((home["status"], home["home"]["at_home"], safety.resets), ("succeeded", True, 1))
        stuck = finish(done, Backend(fails="no plan home"), Safety())
        self.assertEqual(stuck["status"], "incomplete")                     # the goal was met, the arm is not home
        self.assertIn("the arm did not get home", stuck["reason"])
        gave_up = finish({"task_id": "t", "status": "incomplete", "reason": "every way on was tried"}, backend := Backend(), Safety())
        self.assertTrue(gave_up["home"]["at_home"] and backend.calls == 1)  # giving up still ends at home
        for outcome, safety in (({"task_id": "t", "status": "cancelled"}, Safety()), (done, Safety(fault=True))):
            backend = Backend()
            left = finish(outcome, backend, safety)
            self.assertEqual(backend.calls, 0)
            self.assertIn("not moved", left["home"]["detail"])

    def prepared(self, client, **options):
        backend, _ = self.backend(client)
        backend.home_joints = HOME
        backend.holding_id, backend.at_contact, backend.current_pose = None, False, None
        return backend, asyncio.run(backend.prepare_for_task(**options))

    def test_first_of_all_the_hand_opens_and_the_empty_grippers_stop_is_measured_at_home(self):
        client = HomingClient(joints=HOME, knuckle=.636)                     # left closed by another program
        client.gripper_script = lambda k: (
            {"ok": True, "knuckle_rad": 0., "stalled": False, "sent": True, "message": "at target"} if k == 0. else
            {"ok": True, "knuckle_rad": .636, "stalled": True, "sent": True, "message": "moved then settled short of the target"})
        backend, report = self.prepared(client, measure_gripper=True)
        self.assertEqual(client.gripper_commands, [0., KNUCKLE_CLOSED_RAD, 0.])   # open, close on nothing, open again
        self.assertEqual(client.sent, [])                                       # already home: the arm does not move
        self.assertEqual((report["measured_stop_rad"], report["moved"]), (.636, False))
        self.assertEqual(backend.closed_empty_knuckle_rad, .636)
        client = HomingClient(joints=HOME, knuckle=0.)
        backend, report = self.prepared(client)
        self.assertEqual((client.gripper_commands, client.sent, report["done"]), ([], [], ["already home, hand open"]))

    def test_away_from_home_the_hand_opens_before_anything_moves_backs_out_and_goes_home(self):
        client = HomingClient(joints=(.2, .5, 3.10, -1.5, .1, .9, .5), knuckle=.45)
        order = []
        gripper, execute = client.gripper, client.execute

        async def gripping(knuckle, **kwargs):
            order.append(("gripper", knuckle))
            return await gripper(knuckle, **kwargs)

        async def flying(path, **kwargs):
            order.append(("move", len(path.points)))
            return await execute(path, **kwargs)
        client.gripper, client.execute = gripping, flying
        backend, report = self.prepared(client)
        self.assertEqual(order[0], ("gripper", 0.))                             # the hand opens first
        self.assertEqual([step[0] for step in order[1:]], ["move", "move"])     # out along the approach, then home
        self.assertEqual(report["done"][:2], ["opened the hand where it was left", "backed 10 cm out"])
        self.assertTrue(report["moved"])
        self.assertGreater(client.joint_targets[0][2], 3.)                     # the short way round
        client = HomingClient(joints=(.2, .5, 3.10, -1.5, .1, .9, .5), knuckle=.45)
        _, report = self.prepared(client, keep_grip=True)                      # the last task ended holding it
        self.assertEqual(client.gripper_commands, [])

    def test_the_way_home_is_planned_around_the_door_the_arm_swung_open_where_the_camera_sees_it(self):
        import tempfile
        from rammp_adl.motion.planner_world import BASE_WORLD
        client, backend, world = self.arc()
        self.record["measured_door"] = {"width_m": .27, "height_m": .45,
                                        "handle_offsets_m": {"left": .245, "right": .022, "bottom": .14, "top": .3}}
        self.record["hinge_side"] = "left"
        backend.scene = scene = DoorScene(backend, client, lean_deg=0., spring_back=.2)
        client.gripper_script = regripping
        self.assertEqual(self.follow(backend, world, .9).status, "succeeded")
        installed = []

        async def set_world(path, **_):
            installed.append(str(path))
            return True, "world set"
        homing = HomingClient(joints=client.joints, knuckle=.45)
        homing.set_world = set_world
        homing.world_held = None
        backend.client, backend.home_joints, backend.at_contact = homing, HOME, True
        scene.offset = .2                                                       # let go at 0.9, the door springs back
        with tempfile.TemporaryDirectory() as folder:
            backend.planner_world_dir = folder
            report = asyncio.run(backend.return_home())
            self.assertTrue(report["at_home"], report)
            self.assertEqual(len(installed), 2)
            self.assertEqual(installed[1], BASE_WORLD)                          # restored once home
            scene_text = Path(installed[0]).read_text()
        self.assertIn('"name": "table"', scene_text)                            # the bench's base obstacles stay
        self.assertIn("moved_cabinet_door_constraint_0", scene_text)
        seen = next(line for line in report["done"] if "planner's world" in line)
        self.assertIn("as seen", seen)                                          # the camera saw it sprung back
        self.assertIn("0.70", seen)

    def test_a_task_does_not_start_while_another_program_can_command_the_arm(self):
        from types import SimpleNamespace as NS

        class Graph:
            def __init__(self, publishers, actions):
                self.publishers, self.actions = publishers, actions

            def get_name(self):
                return "rammp_adl_runtime"

            def get_topic_names_and_types(self):
                return [(topic, ["x"]) for topic in self.publishers]+[("/joint_states", ["x"])]

            def get_publishers_info_by_topic(self, topic):
                return [NS(node_name=name) for name in self.publishers.get(topic, ())]

            def get_node_names_and_namespaces(self):
                return [(name, "/") for name in self.actions]
        client = HomingClient(joints=HOME, knuckle=0.)
        backend, _ = self.backend(client)
        backend.home_joints = HOME
        backend._action_clients = lambda node, name, namespace: [(action, ["x"]) for action in node.actions[name]]
        client.node = Graph({"/setpoint/gripper": ["rammp_adl_runtime"], "/setpoint/twist": []},
                            {"rammp_adl_runtime": ["/execute_joint_trajectory"], "rammp_curobo": [], "kinova_gen3_node": ["/rammp_curobo/plan_to_pose"]})
        self.assertEqual(backend.other_controllers(), [])
        client.node = Graph({"/setpoint/gripper": ["rammp_adl_runtime", "space_teleop"], "/setpoint/twist": ["space_teleop"]},
                            {"rammp_adl_runtime": ["/execute_joint_trajectory"], "press_demo_mission": ["/execute_joint_trajectory"]})
        self.assertEqual(backend.other_controllers(), ["press_demo_mission", "space_teleop"])
        with self.assertRaises(BackendFailure) as refused:
            asyncio.run(backend.prepare_for_task())
        self.assertIn("space_teleop", str(refused.exception))
        self.assertEqual((client.sent, client.gripper_commands), ([], []))         # nothing moved

    def test_without_a_home_pose_nothing_moves(self):
        backend, _ = self.backend(HomingClient(joints=(.3, .6, -2.9, -1.4, .2, .9, .6)))
        report = asyncio.run(backend.return_home())
        self.assertFalse(report["at_home"])
        self.assertIn("no home pose", report["detail"])


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
