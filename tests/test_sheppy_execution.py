"""The whole door plan through the executor on the sheppy backend with a scripted client.

This is the path the arm takes: admission, per-node confirmation, the dispatch
gate, resource claims, geometry receipts, handler execution, effect commits.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from rammp_adl.app import sheppy_runtime
from rammp_adl.constraints import ConstraintStore, context_constraint
from rammp_adl.intake import draft_context, seed_articulation, seed_visibility
from rammp_adl.motion.collision_guard import EffortGuard, GuardSet
from rammp_adl.motion.kinematics import UrdfChain
from rammp_adl.sheppy_backend import bootstrap_robot_facts
from rammp_adl.world import MetricPose

from test_follow_constraint import TrackingClient, door_record

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2"
BENCH = json.loads((ROOT/"config/sheppy-bench.context.json").read_text())
CAPABILITIES = ("local_geometry,cloud_grounding,curobo_transit,trajectory_execution,local_retention_monitor,live_collision_guard,"
                "contact_monitor,curobo_online_replanning,continuous_trajectory_handoff,calibrated_aperture,calibrated_grasp,"
                "support_detection,curobo_constrained_path").split(",")


def run(coroutine):
    return asyncio.run(coroutine)


class SeenScene:
    camera_id = "wrist_d405"

    def __init__(self, visible):
        self.visible = set(visible)

    def visible_entities(self, now, *, max_age_s=2.):
        return {name: {"visible": True, "capture_id": "cap", "report_digest": "d"} for name in self.visible}


@unittest.skipUnless(BUNDLE.exists(), "The assembly sphere bundle is separate evidence")
class DoorPlanExecutionTests(unittest.TestCase):
    def setUp(self):
        self.chain = UrdfChain.from_path(BUNDLE/"arm-gripper-locked.urdf")
        self.folder = tempfile.TemporaryDirectory()
        self.store = ConstraintStore(self.folder.name)
        self.record = door_record()
        self.client = TrackingClient(knuckle=.01)
        # The gripper closes onto the handle and stalls short of closed; opening reaches its target.
        self.client.gripper_script = lambda k: ({"ok": True, "knuckle_rad": .45, "stalled": True, "sent": True, "message": "stalled"}
                                                if k > .7 else {"ok": True, "knuckle_rad": k, "stalled": False, "sent": True, "message": "at target"})
        descriptors = [{"entity_id": "handle_1", "label": "black door handle", "pose_roles": ["grasp", "pregrasp", "retract", "staging"]},
                       {"entity_id": "cabinet_door_surface", "label": "cabinet door", "pose_roles": []}]
        context = draft_context(BENCH, descriptors, task_id="task-door", camera_id="wrist_d405",
                                constraints=[context_constraint(self.record)])
        self.runtime = sheppy_runtime(context, root=ROOT, client=self.client, capabilities=CAPABILITIES, commissioned=True,
                                      observer=self.observe,
                                      guard_factory=lambda touch_nm=3., exclusions=(), tool_exclusion_m=0.: GuardSet(
                                          effort=EffortGuard(touch_nm), exclusions=exclusions, tool_exclusion_m=tool_exclusion_m),
                                      chain=self.chain, constraints={self.record["constraint_id"]: self.record},
                                      constraint_store=self.store, speed_scales={"transit": 2.5, "contact": 4.},
                                      max_evidence_age_s=600.)
        self.grants = []
        def confirm(plan, node, digest):
            grant = self.runtime.executor.confirmations.grant(task_id=plan["task_id"], epoch=plan["execution_epoch"],
                                                              node_id=node["id"], digest=digest, ttl_s=30., accepted=True)
            self.grants.append(node["id"])
            return grant.confirmation_id
        self.runtime.executor.confirmation_callback = confirm
        world = self.runtime.world
        run(bootstrap_robot_facts(world, self.client))
        seed_visibility(world, SeenScene({"handle_1", "cabinet_door_surface", "ghost_9"}), now=world.clock())
        seed_articulation(world, [{"record": self.record, "constraint": context_constraint(self.record), "history": []}], now=world.clock())

    async def observe(self, args, context):
        """A wrist observation of the handle with its three roles, as the grounded scene returns one."""
        from rammp_adl.hardware_backend import ObservationMeasurement
        snapshot = context.snapshot
        identities = snapshot.identities()
        dependencies = {key: identities[key] for key in ("execution_epoch", "calibration_id", "base_epoch", "entity:handle_1")}
        captured_at = self.runtime.world.clock()-.05
        evidence_id = "observation-"+context.node_id
        poses = tuple(MetricPose("handle_1", role, position, (0., 0., 0., 1.), tuple([0.]*36), captured_at, "base_link",
                                 identities["entity:handle_1"]+1, identities["calibration_id"], identities["base_epoch"], evidence_id, 120.)
                      for role, position in (("pregrasp", (.48, .1, .3)), ("grasp", (.58, .1, .3)), ("retract", (.48, .1, .3))))
        assertions = [{"predicate": "entity_exists", "args": {"entity_id": "handle_1"}, "validity": "true"},
                      {"predicate": "observation_valid", "args": {"entity_id": "handle_1", "purpose": "pose"}, "validity": "true"}]
        assertions += [{"predicate": "pose_valid", "args": {"entity_id": "handle_1", "pose_role": p.pose_role}, "validity": "true"} for p in poses]
        return ObservationMeasurement("handle_1", "wrist", "pose", evidence_id, captured_at, 120., tuple(assertions),
                                      {"kind": "test"}, dependencies, metric_poses=poses)

    def tearDown(self):
        self.folder.cleanup()

    def plan(self):
        snapshot = self.runtime.world.snapshot()
        nodes = [
            {"id": "n0", "skill": "observe", "args": {"entity_id": "handle_1", "camera": "wrist", "purpose": "pose"}},
            {"id": "n1", "skill": "move_to_pose", "args": {"target": {"entity_id": "handle_1", "pose_role": "pregrasp"}, "profile_id": "bench_transit"}},
            {"id": "n2", "skill": "set_gripper", "args": {"aperture_m": .06, "profile_id": "bench_gripper"}},
            {"id": "n3", "skill": "move_to_pose", "args": {"target": {"entity_id": "handle_1", "pose_role": "grasp"}, "profile_id": "bench_transit"}},
            {"id": "n4", "skill": "grasp", "args": {"entity_id": "handle_1", "profile_id": "bench_gripper"}},
            {"id": "n5", "skill": "follow_constraint", "args": {"entity_id": "handle_1", "constraint_id": self.record["constraint_id"],
                                                                 "target_value": .3, "target_unit": "rad", "profile_id": "bench_contact"}},
            {"id": "n6", "skill": "release", "args": {"entity_id": "handle_1", "support_id": "cabinet_door_surface", "profile_id": "bench_gripper"}},
            {"id": "n7", "skill": "move_to_pose", "args": {"target": {"entity_id": "handle_1", "pose_role": "retract"}, "profile_id": "bench_transit"}},
        ]
        edges = [{"from": f"n{i}", "to": f"n{i+1}"} for i in range(0, 7)]
        return {"schema_version": "1.0.0", "skill_library_hash": self.runtime.catalog.hash, "task_id": snapshot.context["task_id"],
                "snapshot_id": snapshot.snapshot_id, "execution_epoch": snapshot.execution_epoch, "nodes": nodes, "edges": edges}

    def test_the_door_plan_runs_end_to_end(self):
        self.assertEqual(set(self.runtime.registry.available_skills), {"observe", "move_to_pose", "set_gripper", "grasp", "release", "follow_constraint"})
        result = run(self.runtime.executor.run_plan(self.plan()))
        self.assertEqual(result.status, "succeeded", result.to_dict())
        self.assertEqual([n.status for n in result.nodes], ["succeeded"]*8)
        self.assertEqual(self.grants, ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"])
        facts = self.runtime.world.snapshot()
        self.assertEqual(facts.fact("constraint_goal_verified", {"constraint_id": self.record["constraint_id"], "target_value": .3, "target_unit": "rad"}), "true")
        self.assertEqual(facts.fact("released", {"entity_id": "handle_1", "support_id": "cabinet_door_surface"}), "true")
        self.assertEqual(facts.fact("gripper_empty", {"robot_id": "robot"}), "true")
        self.assertNotEqual(facts.fact("holding", {"entity_id": "handle_1"}), "true")
        # Transits slowed, contact steps slowed more, the grasp exclusion applied on the grasp move only.
        provenance = [t.provenance for t in self.client.sent]
        self.assertTrue(provenance[0].endswith("x2.5") and provenance[1].endswith("x2.5") and provenance[-1].endswith("x2.5"))
        self.assertTrue(all(p.endswith("x4") for p in provenance[2:-1]) and len(provenance) > 4)
        self.assertEqual(self.store.load("cabinet door")["attempts"][-1]["status"], "succeeded")

    def test_without_confirmation_nothing_moves(self):
        self.runtime.executor.confirmation_callback = None
        result = run(self.runtime.executor.run_plan(self.plan()))
        self.assertNotEqual(result.status, "succeeded")
        self.assertEqual(self.client.sent, [])


if __name__ == "__main__":
    unittest.main()
