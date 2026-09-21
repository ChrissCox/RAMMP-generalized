"""Contract/semantic tests with explicit test doubles, never robot validation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from rammp_adl.contracts import (Catalog, ContractError, SkillRegistry, canonical_json,
                                 digest, fact_key, strict_loads, validate_schema)
from rammp_adl.validation import GeometryCheck, PlanValidator
from rammp_adl.world import EvidenceAuthority, MetricPose, WorldConflict, WorldModel

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.now = 100.0
    def __call__(self):
        return self.now


class ContractGeometryDouble:
    """Tests validator orchestration only. Provides no motion/collision validation."""
    def __init__(self):
        self.calls = []
    def validate(self, node, snapshot, predicted_state, phase):
        self.calls.append((node["id"], phase, copy.deepcopy(predicted_state["geometry"])))
        facts = []
        if "profile_id" in node["args"]:
            facts.append({"predicate": "motion_profile_valid", "args": {"profile_id": node["args"]["profile_id"]}})
        return GeometryCheck(end_state={"test_end_of": node["id"]}, established_facts=tuple(facts), valid_for_s=2.0)


class RuntimeContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog(ROOT)
        cls.template = json.loads((ROOT / "examples/cabinet.context.json").read_text())

    def setUp(self):
        self.clock = Clock()
        self.context = copy.deepcopy(self.template)
        self.world = WorldModel(self.context, self.catalog, clock=self.clock,
                                trust_initial=True, max_evidence_age_s=60.0)
        handlers = {skill: lambda *args: None for skill in self.catalog.skills}
        caps = {cap for skill in self.catalog.skills.values() for cap in skill["capabilities_required"]}
        self.registry = SkillRegistry(self.catalog, handlers, caps)
        self.geometry = ContractGeometryDouble()
        self.validator = PlanValidator(self.catalog, self.registry, self.world, self.geometry)
        self.authority = self.world.authorize_source("test_evaluator", self.catalog.predicates)

    def plan(self, nodes, edges=()):
        snapshot = self.world.snapshot()
        return {"schema_version": "1.0.0", "skill_library_hash": self.catalog.hash,
                "task_id": snapshot.context["task_id"], "snapshot_id": snapshot.snapshot_id,
                "execution_epoch": snapshot.execution_epoch, "nodes": nodes,
                "edges": [{"from": a, "to": b} for a, b in edges]}

    def move(self, node_id, role="pregrasp"):
        return {"id": node_id, "skill": "move_to_pose", "args": {
            "target": {"entity_id": "cabinet_handle_1", "pose_role": role}, "profile_id": "sim_transit"}}

    def gripper(self, node_id="g", aperture=0.06):
        return {"id": node_id, "skill": "set_gripper", "args": {"aperture_m": aperture, "profile_id": "sim_gripper"}}

    def evidence(self, predicate, args, *, validity="true", evidence_id="measured-1"):
        effect = {"predicate": predicate, "args": args, "validity": validity, "evidence_id": evidence_id}
        self.world.register_evidence(evidence_id, source=self.authority, predicates=[effect], ttl_s=30.0)
        return effect

    def test_strict_json_rejects_duplicates_nonfinite_overflow_depth_and_size(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '[' * 66 + '0' + ']' * 66):
            with self.subTest(value=value[:30]), self.assertRaises(ContractError):
                strict_loads(value)
        with self.assertRaises(ContractError):
            strict_loads('"abcd"', max_bytes=3)

    def test_full_schema_supports_keywords_beyond_artifact_subset(self):
        with self.assertRaises(ContractError):
            validate_schema("abc", {"type": "string", "pattern": "^[0-9]+$"})
        with self.assertRaises(ContractError):
            validate_schema(True, {"type": "integer"})

    def test_catalog_hash_and_runtime_filtered_schema(self):
        self.assertEqual(self.catalog.hash, digest(self.catalog.library))
        schema = self.catalog.response_schema(["observe"], context=self.context)
        with self.assertRaises(ContractError):
            validate_schema({"result": {"status": "OK", "plan": self.plan([self.gripper()])}}, schema)

    def test_registry_excludes_missing_deferred_and_hardware(self):
        self.assertEqual(SkillRegistry(self.catalog, self.registry.handlers, set()).available_skills, ())
        # Hardware mode still needs the commissioned assertion, and planned or
        # deferred entries never register there regardless of capabilities.
        uncommissioned = SkillRegistry(self.catalog, self.registry.handlers, self.registry.capabilities,
                                       mode="hardware", commissioned=False)
        self.assertEqual(uncommissioned.available_skills, ())
        hardware = SkillRegistry(self.catalog, self.registry.handlers, self.registry.capabilities,
                                 mode="hardware", commissioned=True)
        # Every current skill is implemented; commissioning plus capabilities registers all of them.
        self.assertEqual(set(hardware.available_skills), set(self.registry.available_skills))
        self.assertEqual(uncommissioned.unavailable["follow_constraint"], "hardware capability is not implemented and commissioned")

    def test_numeric_fact_args_have_one_key(self):
        self.assertEqual(fact_key("aperture_reached", {"aperture_m": 0}), fact_key("aperture_reached", {"aperture_m": 0.0}))
        self.assertNotEqual(fact_key("at_pose", {"entity_id": "x", "pose_role": "grasp"}),
                            fact_key("at_pose", {"entity_id": "x", "pose_role": "pregrasp"}))

    def test_untrusted_initial_facts_never_establish_truth(self):
        world = WorldModel(self.context, self.catalog, clock=self.clock)
        self.assertEqual(world.snapshot().fact("gripper_empty", {"robot_id": "robot"}), "unknown")

    def test_snapshots_are_detached_and_evidence_expires(self):
        snapshot = self.world.snapshot()
        external = snapshot.context
        external["execution_epoch"] = 999
        external["entities"][0]["entity_id"] = "forged"
        self.assertEqual(snapshot.execution_epoch, 1)
        self.assertEqual(self.world.snapshot().context["entities"][0]["entity_id"], "cabinet_handle_1")
        self.clock.now += 61.0
        self.assertEqual(self.world.snapshot().fact("gripper_empty", {"robot_id": "robot"}), "unknown")
        self.assertEqual(snapshot.fact("gripper_empty", {"robot_id": "robot"}), "true")

    def test_evidence_requires_issued_authority_and_exact_argument_scope(self):
        forged = EvidenceAuthority("test_evaluator", frozenset(self.catalog.predicates))
        with self.assertRaises(ContractError):
            self.world.register_evidence("forged", source=forged, predicates=[], ttl_s=2)
        effect = self.evidence("aperture_reached", {"aperture_m": 0.06})
        self.world.register_operation("op", 1)
        effect["args"]["aperture_m"] = 0.08
        with self.assertRaises(ContractError):
            self.world.commit_effects("op", [effect], 1, 1)
        self.assertEqual(self.world.snapshot().revision, 1)

    def test_evidence_unknown_and_false_do_not_satisfy_goal(self):
        goal = self.context["goal"]
        for index, validity in enumerate(("unknown", "false", "true")):
            effect = self.evidence(goal["predicate"], goal["args"], validity=validity, evidence_id=f"goal-{index}")
            self.world.register_operation(f"op-{index}", 1)
            self.world.commit_effects(f"op-{index}", [effect], self.world.snapshot().revision, 1)
            self.assertEqual(self.world.goal_satisfied(), validity == "true")

    def test_idempotent_commit_rejects_changed_payload(self):
        effect = self.evidence("aperture_reached", {"aperture_m": 0.06})
        self.world.register_operation("op", 1)
        first = self.world.commit_effects("op", [effect], 1, 1, completed_node="g")
        self.assertEqual(first, self.world.commit_effects("op", [effect], 1, 1, completed_node="g"))
        self.assertEqual(self.world.snapshot().revision, 2)
        with self.assertRaises(ContractError):
            self.world.commit_effects("op", [], 1, 1, completed_node="g")

    def test_disjoint_completed_effects_rebase_without_repeating_motion(self):
        aperture = self.evidence("aperture_reached", {"aperture_m": 0.06}, evidence_id="gripper")
        pose = self.evidence("at_pose", {"entity_id": "cabinet_handle_1", "pose_role": "pregrasp"}, evidence_id="arm")
        dependencies = {"entity:cabinet_handle_1": 1}
        self.world.register_operation("arm", 1, dependencies)
        self.world.register_operation("gripper", 1, dependencies)
        self.world.commit_effects("gripper", [aperture], 1, 1)
        receipt = self.world.commit_effects("arm", [pose], 1, 1)
        self.assertTrue(receipt.rebased)
        self.assertEqual(self.world.snapshot().fact("at_pose", pose["args"]), "true")

    def test_changed_dependency_blocks_completed_effect_rebase(self):
        deps = {"grasp_state_id": self.context["grasp_state_id"]}
        self.world.register_operation("old", 1, deps)
        self.world.register_operation("grasp", 1)
        effect = self.evidence("holding", {"entity_id": "cabinet_handle_1"})
        self.world.commit_effects("grasp", [effect], 1, 1)
        with self.assertRaises(WorldConflict):
            self.world.commit_effects("old", [], 1, 1)
        self.assertEqual(self.world.snapshot().context["attachment_id"], "empty")
        self.assertNotEqual(self.world.snapshot().context["grasp_state_id"], "empty")

    def test_cancel_revokes_commands_but_reconciles_registered_terminal_evidence(self):
        self.world.register_operation("inflight", 1, {"execution_epoch": 1})
        effect = self.evidence("aperture_reached", {"aperture_m": 0.06})
        self.assertEqual(self.world.invalidate_epoch(), 2)
        with self.assertRaises(WorldConflict):
            self.world.seal_epoch(1)
        receipt = self.world.commit_effects("inflight", [effect], 1, 1)
        self.assertEqual(receipt.execution_epoch, 1)
        self.world.seal_epoch(1)
        with self.assertRaises(WorldConflict):
            self.world.register_operation("late", 1)

    def test_terminal_commit_requires_quiescence(self):
        self.world.register_operation("op", 1)
        with self.assertRaises(ContractError):
            self.world.commit_effects("op", [], 1, 1, backend_quiescent=False)

    def test_prerequisite_expiry_during_motion_does_not_discard_fresh_terminal_evidence(self):
        identities = self.world.snapshot().identities()
        dependency_name = "fact:" + canonical_json(list(fact_key("gripper_empty", {"robot_id": "robot"})))
        self.world.register_operation("long-action", 1, {dependency_name: identities[dependency_name]})
        self.clock.now += 61.0
        effect = self.evidence("aperture_reached", {"aperture_m": 0.06}, evidence_id="fresh-completion")
        self.world.commit_effects("long-action", [effect], 1, 1)
        self.assertEqual(self.world.snapshot().fact("aperture_reached", effect["args"]), "true")

    def test_prerequisite_invalidation_is_not_treated_as_expiry(self):
        identities = self.world.snapshot().identities()
        dependency_name = "fact:" + canonical_json(list(fact_key("gripper_empty", {"robot_id": "robot"})))
        self.world.register_operation("old-action", 1, {dependency_name: identities[dependency_name]})
        self.world.register_operation("grasp", 1)
        effect = self.evidence("holding", {"entity_id": "cabinet_handle_1"}, evidence_id="retention")
        self.world.commit_effects("grasp", [effect], 1, 1)
        with self.assertRaises(WorldConflict):
            self.world.commit_effects("old-action", [], 1, 1)

    def test_bound_rolling_update_refreshes_only_current_operation_dependencies(self):
        initial = self.world.snapshot()
        dependency_keys = ("entity:cabinet_handle_1", "collision_revision", "execution_epoch", "calibration_id")
        dependencies = {key: initial.identities()[key] for key in dependency_keys}
        self.world.register_operation("motion", 1, dependencies)
        pose = MetricPose("cabinet_handle_1", "grasp", (0.1, 0.2, 0.3), (0, 0, 0, 1),
                          (0.0,) * 36, 100.0, "base_link", 2, self.context["calibration_id"],
                          self.context["base_epoch"], "new-fit", 10.0)
        self.evidence("pose_valid", {"entity_id": "cabinet_handle_1", "pose_role": "grasp"}, evidence_id="new-fit")
        self.world.update_metric_pose(pose, source=self.authority)
        refreshed = {key: self.world.snapshot().identities()[key] for key in dependency_keys}
        with self.assertRaises(ContractError):
            self.world.refresh_operation_dependencies("motion", refreshed, source=self.authority, validated_update_id="generation-2")
        gateway = self.world.authorize_source("command_gateway", [])
        with self.assertRaises(ContractError):
            self.world.refresh_operation_dependencies("motion", {}, source=gateway, validated_update_id="generation-2")
        with self.assertRaises(WorldConflict):
            self.world.refresh_operation_dependencies("motion", dict(refreshed, calibration_id="changed"),
                                                      source=gateway, validated_update_id="generation-2")
        self.world.refresh_operation_dependencies("motion", refreshed, source=gateway, validated_update_id="generation-2")
        effect = self.evidence("at_pose", {"entity_id": "cabinet_handle_1", "pose_role": "grasp"}, evidence_id="arrival")
        receipt = self.world.commit_effects("motion", [effect], initial.revision, 1)
        self.assertTrue(receipt.rebased)

    def test_atomic_commit_rejects_missing_entity_in_any_argument(self):
        effect = self.evidence("released", {"entity_id": "cabinet_handle_1", "support_id": "missing"})
        self.world.register_operation("release", 1)
        with self.assertRaises(ContractError):
            self.world.commit_effects("release", [effect], 1, 1)
        self.assertEqual(self.world.snapshot().revision, 1)

    def test_plan_rejects_cycle_unknown_fields_and_wrong_catalog(self):
        candidate = self.plan([self.move("a"), self.move("b")], [("a", "b"), ("b", "a")])
        with self.assertRaises(ContractError):
            self.validator.admit(candidate)
        candidate = self.plan([self.move("a")])
        candidate["nodes"][0]["claims"] = []
        with self.assertRaises(ContractError):
            self.validator.admit(candidate)
        candidate = self.plan([self.move("a")])
        candidate["skill_library_hash"] = "sha256:forged"
        with self.assertRaises(ContractError):
            self.validator.admit(candidate)

    def test_sequential_geometry_uses_predicted_predecessor_state(self):
        candidate = self.plan([self.move("a"), self.move("b", "grasp")], [("a", "b")])
        receipt = self.validator.admit(candidate)
        self.assertEqual(receipt.order, ("a", "b"))
        self.assertEqual(self.geometry.calls, [("a", "admission", None), ("b", "admission", {"test_end_of": "a"})])
        self.assertEqual(self.world.snapshot().fact("at_grasp_pose", {"entity_id": "cabinet_handle_1"}), "unknown")

    def test_parallel_arm_moves_are_state_conflict_even_if_claims_serialize(self):
        with self.assertRaisesRegex(ContractError, "unsequenced state conflict"):
            self.validator.admit(self.plan([self.move("a"), self.move("b", "grasp")]))

    def test_parallel_empty_gripper_preshape_and_move_admit(self):
        candidate = self.plan([self.move("a"), self.gripper()])
        receipt = self.validator.admit(candidate)
        self.assertEqual(set(receipt.order), {"a", "g"})
        self.assertTrue(self.registry.claims("move_to_pose").isdisjoint(self.registry.claims("set_gripper")))

    def test_preconditions_only_come_from_ancestors(self):
        grasp = {"id": "g", "skill": "grasp", "args": {"entity_id": "cabinet_handle_1", "profile_id": "sim_gripper"}}
        with self.assertRaises(ContractError):
            self.validator.admit(self.plan([self.move("a", "grasp"), grasp]))
        receipt = self.validator.admit(self.plan([self.move("a", "grasp"), grasp], [("a", "g")]))
        self.assertEqual(receipt.ancestors["g"], frozenset({"a"}))

    def test_unsatisfied_preconditions_reject_whole_plan(self):
        grasp = {"id": "g", "skill": "grasp", "args": {"entity_id": "cabinet_handle_1", "profile_id": "sim_gripper"}}
        candidate = self.plan([self.move("a", "pregrasp"), grasp], [("a", "g")])
        with self.assertRaisesRegex(ContractError, "unsatisfied precondition at_grasp_pose"):
            self.validator.admit(candidate)
        with self.assertRaises(ContractError):
            self.validator.dispatch(candidate, "a")

    def test_missing_geometry_allows_observation_only(self):
        validator = PlanValidator(self.catalog, self.registry, self.world)
        with self.assertRaises(ContractError):
            validator.admit(self.plan([self.move("a")]))
        observe = {"id": "o", "skill": "observe", "args": {"entity_id": "cabinet_handle_1", "camera": "scene", "purpose": "pose"}}
        self.assertIsNotNone(validator.admit(self.plan([observe])))

    def test_dispatch_requires_committed_dependencies_and_fresh_predicates(self):
        candidate = self.plan([self.move("a"), self.move("b", "grasp")], [("a", "b")])
        admission = self.validator.admit(candidate)
        with self.assertRaises(ContractError):
            self.validator.dispatch(candidate, "b", admission)
        receipt = self.validator.dispatch(candidate, "a", admission)
        self.assertTrue(self.validator.verify_receipt(receipt, candidate, "a"))
        self.clock.now += 3.0
        with self.assertRaises(ContractError):
            self.validator.verify_receipt(receipt, candidate, "a")

    def test_receipt_cannot_be_forged_retagged_or_reused_in_new_epoch(self):
        from dataclasses import replace
        candidate = self.plan([self.move("a")])
        admission = self.validator.admit(candidate)
        receipt = self.validator.dispatch(candidate, "a", admission)
        with self.assertRaises(ContractError):
            self.validator.verify_receipt(replace(receipt, expires_at=1000.0), candidate, "a")
        altered = copy.deepcopy(candidate)
        altered["nodes"][0]["args"]["target"]["pose_role"] = "grasp"
        with self.assertRaises(ContractError):
            self.validator.verify_receipt(receipt, altered, "a")
        self.world.invalidate_epoch()
        with self.assertRaises(ContractError):
            self.validator.verify_receipt(receipt, candidate, "a")

    def test_bindings_reject_wrong_units_entity_profile_and_role(self):
        for mutate in (
            lambda n: n["args"]["target"].update(entity_id="missing"),
            lambda n: n["args"].update(profile_id="sim_gripper"),
            lambda n: n["args"]["target"].update(pose_role="staging"),
        ):
            node = self.move("a")
            mutate(node)
            with self.assertRaises(ContractError):
                self.validator.admit(self.plan([node]))
        constraint = {"id": "c", "skill": "follow_constraint", "args": {"entity_id": "cabinet_handle_1", "constraint_id": "hinge_1", "target_value": 1.0, "target_unit": "m", "profile_id": "sim_cabinet_contact"}}
        with self.assertRaisesRegex(ContractError, "wrong unit"):
            self.validator.admit(self.plan([constraint]))

    def test_metric_pose_does_not_invent_orientation_or_accept_bad_covariance(self):
        kwargs = dict(entity_id="cabinet_handle_1", pose_role="grasp", position_m=(0.1, 0.2, 0.3),
                      orientation_xyzw=(0, 0, 0, 1), covariance=(0.0,) * 36, captured_at=100.0,
                      frame_id="base_link", entity_revision=2, calibration_id=self.context["calibration_id"],
                      base_epoch=self.context["base_epoch"], evidence_id="pose", valid_for_s=10.0)
        with self.assertRaises(ContractError):
            MetricPose(**dict(kwargs, orientation_xyzw=(0, 0, 0, 0)))
        covariance = [0.0] * 36
        covariance[0] = covariance[7] = 1.0
        covariance[1] = covariance[6] = 2.0
        with self.assertRaises(ContractError):
            MetricPose(**dict(kwargs, covariance=tuple(covariance)))
        pose = MetricPose(**kwargs)
        self.evidence("pose_valid", {"entity_id": "cabinet_handle_1", "pose_role": "grasp"}, evidence_id="pose")
        old = self.world.snapshot()
        self.world.update_metric_pose(pose, source=self.authority)
        snapshot = self.world.snapshot()
        self.assertNotEqual(snapshot.snapshot_id, old.snapshot_id)
        self.assertEqual(snapshot.metric_poses[("cabinet_handle_1", "grasp")], pose)
        self.assertEqual(snapshot.fact("pose_valid", {"entity_id": "cabinet_handle_1", "pose_role": "pregrasp"}), "unknown")


if __name__ == "__main__":
    unittest.main()
