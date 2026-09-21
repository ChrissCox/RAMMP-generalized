"""Atomic local observation commits, using synthetic measurements and no I/O."""
from dataclasses import replace
import unittest

from rammp_adl.contracts import ContractError
from rammp_adl.world import WorldConflict, WorldModel
from test_hardware_backend import ENTITY, metric, pending_measurement, world_fixture


class AtomicObservationCommitTests(unittest.TestCase):
    def setUp(self):
        self.catalog, self.world = world_fixture()
        self.source = self.world.authorize_source("handler:observe", self.catalog.predicates.keys())
        self.initial = self.world.snapshot()
        self.measured = pending_measurement(self.world)
        identities = self.initial.identities()
        self.dependencies = {name: identities[name] for name in (
            "execution_epoch", "collision_revision", "calibration_id", "base_epoch", "entity:"+ENTITY)}
        self.world.register_operation("observe", self.initial.execution_epoch, self.dependencies)
        self.register(self.measured)

    def register(self, measured, source=None):
        return self.world.register_evidence(measured.evidence_id, source=source or self.source,
            predicates=measured.assertions, observed_at=measured.captured_at,
            ttl_s=measured.valid_for_s, dependencies=measured.dependencies, data=measured.data)

    def commit(self, measured=None, **kwargs):
        measured = measured or self.measured
        return self.world.commit_effects("observe",
            [{**fact, "evidence_id": measured.evidence_id} for fact in measured.assertions],
            self.initial.revision, self.initial.execution_epoch, completed_node="fit",
            metric_poses=measured.metric_poses, metric_source=kwargs.pop("metric_source", self.source), **kwargs)

    def assert_no_commit(self, previous=None):
        previous = previous or self.initial
        current = self.world.snapshot()
        self.assertEqual(current.revision, previous.revision)
        self.assertEqual(dict(current.metric_poses), dict(previous.metric_poses))
        self.assertEqual(dict(current.facts), dict(previous.facts))
        self.assertEqual(current.context["completed_nodes"], previous.context["completed_nodes"])
        self.assertEqual(current.context["collision_revision"], previous.context["collision_revision"])

    def test_geometry_facts_and_single_revision_commit_together_and_replay_is_idempotent(self):
        receipt = self.commit()
        current = self.world.snapshot()
        self.assertEqual(current.revision, self.initial.revision+1)
        self.assertEqual(current.metric_poses[(ENTITY, "pregrasp")], self.measured.metric_poses[0])
        self.assertEqual(current.fact("pose_valid", {"entity_id": ENTITY, "pose_role": "pregrasp"}), "true")
        self.assertEqual(self.commit(), receipt)
        self.assertEqual(self.world.snapshot().revision, current.revision)
        changed = replace(self.measured.metric_poses[0], position_m=(.5, .1, .3))
        with self.assertRaisesRegex(ContractError, "idempotency"):
            self.commit(replace(self.measured, metric_poses=(changed,)))

    def test_an_invalid_later_pose_leaves_every_geometry_and_fact_unmodified(self):
        assertion = {"predicate": "pose_valid", "args": {"entity_id": ENTITY, "pose_role": "grasp"},
                     "validity": "true"}
        first = replace(self.measured.metric_poses[0], evidence_id="invalid-batch")
        second = replace(first, pose_role="grasp", frame_id="untransformed-camera-frame")
        bad = replace(self.measured, evidence_id="invalid-batch", metric_poses=(first, second),
                      assertions=(*self.measured.assertions, assertion))
        self.register(bad)
        with self.assertRaisesRegex(ContractError, "planning frame"):
            self.commit(bad)
        self.assert_no_commit()

    def test_external_target_revision_is_preserved_and_never_retagged(self):
        external = metric(self.world, evidence_id="external-fit")
        before = self.world.snapshot()
        with self.assertRaisesRegex(WorldConflict, "changed dependency"):
            self.commit()
        self.assert_no_commit(before)
        self.assertEqual(self.world.snapshot().metric_poses[(ENTITY, "pregrasp")], external)

    def test_external_collision_change_cannot_be_dropped_from_original_dependencies(self):
        context = self.initial.context
        next(e for e in context["entities"] if e["entity_id"] == "cabinet_door_1")["pose_roles"] = ["pregrasp"]
        world = WorldModel(context, self.catalog, trust_initial=True)
        observer = world.authorize_source("handler:observe", self.catalog.predicates.keys())
        local = world.authorize_source("external-fit", ["pose_valid"])
        world.register_operation("observe", self.initial.execution_epoch, self.dependencies)
        world.register_evidence(self.measured.evidence_id, source=observer,
            predicates=self.measured.assertions, observed_at=self.measured.captured_at,
            ttl_s=self.measured.valid_for_s, dependencies=self.measured.dependencies)
        other = replace(self.measured.metric_poses[0], entity_id="cabinet_door_1", evidence_id="other-obstacle",
                        entity_revision=world.snapshot().identities()["entity:cabinet_door_1"]+1)
        world.register_evidence(other.evidence_id, source=local, ttl_s=other.valid_for_s,
            observed_at=other.captured_at, predicates=[{"predicate": "pose_valid", "args": {
                "entity_id": other.entity_id, "pose_role": other.pose_role}, "validity": "true"}])
        world.update_metric_pose(other, source=local)
        previous = world.snapshot()
        with self.assertRaisesRegex(WorldConflict, "collision_revision"):
            world.commit_effects("observe",
                [{**fact, "evidence_id": self.measured.evidence_id} for fact in self.measured.assertions],
                self.initial.revision, self.initial.execution_epoch,
                metric_poses=self.measured.metric_poses, metric_source=observer)
        self.assertEqual(world.snapshot().revision, previous.revision)
        self.assertNotIn((ENTITY, "pregrasp"), world.snapshot().metric_poses)
        self.assertEqual(world.snapshot().metric_poses[(other.entity_id, "pregrasp")], other)

    def test_unsatisfied_postcondition_leaks_no_geometry_or_completion(self):
        with self.assertRaisesRegex(ContractError, "postcondition"):
            self.commit(expected_postconditions=[{"predicate": "holding", "args": {"entity_id": ENTITY}}])
        self.assert_no_commit()
        self.assertTrue(self.commit().receipt_id)  # Retained evidence can still be reconciled.

    def test_unregistered_or_wrong_evaluator_cannot_install_atomic_geometry(self):
        other = self.world.authorize_source("other-geometry-evaluator", ["pose_valid"])
        for source in (None, other):
            with self.subTest(source=source), self.assertRaisesRegex(ContractError, "handler authority"):
                self.commit(metric_source=source)
            self.assert_no_commit()

    def test_atomic_observation_cannot_claim_physical_effects(self):
        pose = replace(self.measured.metric_poses[0], evidence_id="not-physical-evidence")
        bad = replace(self.measured, evidence_id=pose.evidence_id, metric_poses=(pose,),
            assertions=(*self.measured.assertions,
                {"predicate": "holding", "args": {"entity_id": ENTITY}, "validity": "true"}))
        self.register(bad)
        with self.assertRaisesRegex(ContractError, "physical"):
            self.commit(bad)
        self.assert_no_commit()


if __name__ == "__main__":
    unittest.main()
