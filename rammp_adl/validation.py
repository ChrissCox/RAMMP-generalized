"""Whole-DAG symbolic admission and fresh per-dispatch geometric validation.

Geometry is an injected, trusted adapter. The core never substitutes a symbolic
success label for IK, swept-path, human-envelope or stopping-volume validation.
"""
from __future__ import annotations

import copy
import math
import threading
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import (Catalog, ContractError, SkillRegistry, canonical_json,
                        checked_copy, digest, fact_key)
from .world import WorldModel


@dataclass(frozen=True)
class GeometryCheck:
    """Exact artifact and predicted end state produced by the trusted adapter.

    `phase` passed to the adapter is admission or dispatch. Admission must use the
    supplied predecessor geometry/end state, not replan each node from initial
    joints. `established_facts` are locally evaluated preconditions only.
    """
    end_state: Any = None
    dependencies: Mapping[str, Any] = field(default_factory=dict)
    artifact: Any = None
    established_facts: tuple[dict, ...] = ()
    valid_for_s: float = 1.0
    status: str = "validated"


@dataclass(frozen=True)
class AdmissionReceipt:
    receipt_id: str
    plan_digest: str
    task_id: str
    execution_epoch: int
    snapshot_id: str
    order: tuple[str, ...]
    ancestors: Mapping[str, frozenset[str]]
    node_checks: Mapping[str, GeometryCheck]
    created_at: float


@dataclass(frozen=True)
class DispatchReceipt:
    receipt_id: str
    admission_id: str
    plan_digest: str
    node_id: str
    args_digest: str
    execution_epoch: int
    snapshot_id: str
    dependencies: Mapping[str, Any]
    geometry: GeometryCheck
    expires_at: float


def _bound_keys(conditions):
    return {fact_key(c["predicate"], c["args"]) for c in conditions}


def _graph(plan):
    nodes = {node["id"]: node for node in plan["nodes"]}
    if len(nodes) != len(plan["nodes"]):
        raise ContractError("duplicate node IDs")
    incoming = {key: set() for key in nodes}
    outgoing = {key: set() for key in nodes}
    seen = set()
    for edge in plan["edges"]:
        start, end = edge["from"], edge["to"]
        if start not in nodes or end not in nodes or start == end or (start, end) in seen:
            raise ContractError("invalid, duplicate or self dependency edge")
        incoming[end].add(start)
        outgoing[start].add(end)
        seen.add((start, end))
    counts = {key: len(value) for key, value in incoming.items()}
    ready = sorted(key for key, count in counts.items() if count == 0)
    order = []
    ancestors = {key: set() for key in nodes}
    while ready:
        key = ready.pop(0)
        order.append(key)
        for child in sorted(outgoing[key]):
            ancestors[child] |= ancestors[key] | {key}
            counts[child] -= 1
            if counts[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(nodes):
        raise ContractError("dependency graph contains a cycle")
    return nodes, tuple(order), ancestors, incoming


def _entity_references(node):
    result = set()
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"entity_id", "support_id"}:
                    result.add(child)
                else:
                    walk(child)
    walk(node["args"])
    return result


def _state_access(node):
    """Trusted physical state categories supplement exact typed fact conflicts.

    Reads of gripper aperture during an empty transit are represented by the
    geometry adapter's full aperture envelope, permitting legitimate preshaping.
    """
    skill, args = node["skill"], node["args"]
    reads = {"entity:" + entity for entity in _entity_references(node)}
    writes = set()
    if "profile_id" in args:
        reads.add("profile:" + args["profile_id"])
    if skill == "observe":
        writes.add("entity:" + args["entity_id"])
    elif skill == "move_to_pose":
        writes.add("robot_pose")
        reads.add("retention")
    elif skill == "set_gripper":
        reads.add("retention")
        writes.add("aperture")
    elif skill in {"grasp", "release"}:
        reads.add("robot_pose")
        writes |= {"retention", "aperture"}
    elif skill == "follow_constraint":
        reads.add("retention")
        reads.add("constraint:" + args["constraint_id"])
        writes |= {"robot_pose", "entity:" + args["entity_id"], "constraint:" + args["constraint_id"]}
    return reads, writes


def _apply_success(catalog, node, facts, entities=None):
    """Prediction only: never written into WorldModel by admission.

    Closed invariants invalidate obsolete alternatives. Measurement at dispatch
    and commit is still required. A move whose exact goal is the grasp role
    entails at_grasp_pose on its successful branch; an observation for pose
    entails the entity's declared pose roles, which dispatch re-measures.
    """
    skill, args = node["skill"], node["args"]
    for key in list(facts):
        predicate, args_json = key
        bound = checked_copy(args_json)
        invalidate = False
        if skill in {"move_to_pose", "follow_constraint"}:
            invalidate = predicate in {"at_pose", "at_grasp_pose", "at_release_pose"}
        elif skill == "set_gripper":
            invalidate = predicate == "aperture_reached"
        elif skill == "grasp":
            invalidate = predicate in {"gripper_empty", "holding", "released"}
        elif skill == "release":
            invalidate = predicate == "holding" and bound.get("entity_id") == args["entity_id"]
        if skill == "follow_constraint":
            invalidate |= predicate == "constraint_goal_verified" and bound.get("constraint_id") == args["constraint_id"]
        if invalidate:
            facts[key] = "unknown"
    for condition in catalog.bind_conditions(skill, args, "postconditions"):
        facts[fact_key(condition["predicate"], condition["args"])] = "true"
    if skill == "observe" and args.get("purpose") in ("pose", "grasp") and entities:
        entity = entities.get(args["entity_id"])
        for role in (entity or {}).get("pose_roles", ()):
            facts[fact_key("pose_valid", {"entity_id": args["entity_id"], "pose_role": role})] = "true"
        if entity is not None:
            facts[fact_key("entity_exists", {"entity_id": args["entity_id"]})] = "true"
    if skill == "move_to_pose" and args["target"]["pose_role"] == "grasp":
        facts[fact_key("at_grasp_pose", {"entity_id": args["target"]["entity_id"]})] = "true"
    elif skill == "grasp":
        facts[fact_key("gripper_empty", {"robot_id": "robot"})] = "false"
    elif skill == "release":
        facts[fact_key("holding", {"entity_id": args["entity_id"]})] = "false"
        facts[fact_key("gripper_empty", {"robot_id": "robot"})] = "true"


class PlanValidator:
    def __init__(self, catalog: Catalog, registry: SkillRegistry, world: WorldModel, geometry=None):
        self.catalog, self.registry, self.world, self.geometry = catalog, registry, world, geometry
        self._admissions: dict[str, AdmissionReceipt] = {}
        self._dispatches: dict[str, DispatchReceipt] = {}
        self._lock = threading.RLock()

    def _bindings(self, node, snapshot):
        context = snapshot.context
        entities = {e["entity_id"]: e for e in context["entities"]}
        for entity_id in _entity_references(node):
            if entity_id not in entities:
                raise ContractError("node references unknown entity: " + entity_id)
        args = node["args"]
        target = args.get("target")
        if isinstance(target, dict) and target["pose_role"] not in entities[target["entity_id"]]["pose_roles"]:
            raise ContractError("target pose role is unavailable")
        if "profile_id" in args:
            profiles = {p["profile_id"]: p for p in context["profiles"]}
            profile = profiles.get(args["profile_id"])
            if profile is None:
                raise ContractError("unknown profile ID")
            if profile["safety_class"] != self.catalog.skills[node["skill"]]["safety_profile"]:
                expected = self.catalog.skills[node["skill"]]["safety_profile"]
                raise ContractError(f"Node {node['id']} ({node['skill']}): profile {args['profile_id']} "
                                    f"has safety class {profile['safety_class']}; required {expected}")
            if self.registry.mode == "hardware" and profile["simulation_only"]:
                raise ContractError("simulation profile cannot authorize hardware")
        if "constraint_id" in args:
            constraints = {c["constraint_id"]: c for c in context["constraints"]}
            constraint = constraints.get(args["constraint_id"])
            if constraint is None or constraint["entity_id"] != args["entity_id"]:
                raise ContractError("unknown constraint or wrong entity binding")
            if constraint["validity"] != "true":
                raise ContractError("constraint estimate is unknown or invalid")
            if constraint["unit"] != args["target_unit"]:
                raise ContractError("constraint target uses wrong unit")
            if not constraint["minimum"] <= args["target_value"] <= constraint["maximum"]:
                raise ContractError("constraint target is outside locally admitted bounds")

    def _geometry(self, node, snapshot, predicted_state, phase):
        if node["skill"] == "observe":
            return GeometryCheck(end_state=predicted_state.get("geometry"),
                                 valid_for_s=self.world.max_evidence_age_s)
        if self.geometry is None:
            raise ContractError("motion geometry adapter is unavailable; observation-only admission permitted")
        evaluator = getattr(self.geometry, "validate", self.geometry)
        check = evaluator(copy.deepcopy(node), snapshot, predicted_state, phase)
        if not isinstance(check, GeometryCheck) or check.status != "validated":
            raise ContractError("geometry adapter did not return a validated GeometryCheck")
        if type(check.valid_for_s) not in (int, float) or not math.isfinite(check.valid_for_s) or check.valid_for_s <= 0:
            raise ContractError("geometry receipt expiry must be finite and positive")
        if "ARM" in self.registry.claims(node["skill"]) and check.end_state is None:
            raise ContractError("geometry adapter omitted predicted arm end state")
        identities = snapshot.identities()
        for key, value in check.dependencies.items():
            if identities.get(key) != value:
                raise ContractError("geometry dependency is not from the supplied world snapshot: " + key)
        return check

    def _check_preconditions(self, node, facts, geometry):
        conditions = self.catalog.bind_conditions(node["skill"], node["args"])
        required = _bound_keys(conditions)
        for fact in geometry.established_facts:
            if set(fact) not in ({"predicate", "args"}, {"predicate", "args", "validity"}):
                raise ContractError("geometry established fact has unexpected fields")
            self.catalog.validate_predicate(fact["predicate"], fact["args"])
            key = fact_key(fact["predicate"], fact["args"])
            if key not in required:
                raise ContractError("geometry attempted to establish a non-precondition fact")
            if fact.get("validity", "true") != "true":
                raise ContractError("geometry did not establish required fact truth")
            facts[key] = "true"
        for condition in conditions:
            if facts.get(fact_key(condition["predicate"], condition["args"]), "unknown") != "true":
                raise ContractError(f"{node['id']}: unsatisfied precondition {condition['predicate']} {canonical_json(condition['args'])}")

    def admit(self, plan) -> AdmissionReceipt:
        candidate = self.catalog.validate_plan_json(plan, available=self.registry.available_skills)
        snapshot = self.world.snapshot()
        context = snapshot.context
        for key in ("task_id", "snapshot_id", "execution_epoch"):
            if candidate[key] != context[key]:
                raise ContractError("plan has stale or mismatched " + key)
        nodes, order, ancestors, _ = _graph(candidate)
        available_context = set(context["available_skills"])
        accesses = {}
        conditions = {}
        effects = {}
        for node_id in order:
            node = nodes[node_id]
            self.registry.require(node["skill"])
            if node["skill"] not in available_context:
                raise ContractError("skill not advertised by current context")
            self._bindings(node, snapshot)
            accesses[node_id] = _state_access(node)
            conditions[node_id] = _bound_keys(self.catalog.bind_conditions(node["skill"], node["args"]))
            effects[node_id] = _bound_keys(self.catalog.bind_conditions(node["skill"], node["args"], "postconditions"))
        for index, left in enumerate(order):
            for right in order[index + 1:]:
                if left in ancestors[right] or right in ancestors[left]:
                    continue
                lr, lw = accesses[left]
                rr, rw = accesses[right]
                if lw & (rr | rw) or rw & lr or effects[left] & (conditions[right] | effects[right]) or effects[right] & conditions[left]:
                    raise ContractError(f"unsequenced state conflict between {left} and {right}")
        checks = {}
        for node_id in order:
            facts = dict(snapshot.facts)
            for ancestor in order:
                if ancestor in ancestors[node_id]:
                    _apply_success(self.catalog, nodes[ancestor], facts,
                                   entities={e["entity_id"]: e for e in snapshot.context["entities"]})
            arm_ancestors = [ancestor for ancestor in order if ancestor in ancestors[node_id]
                             and nodes[ancestor]["skill"] != "observe"
                             and "ARM" in self.registry.claims(nodes[ancestor]["skill"])]
            prior = checks[arm_ancestors[-1]].end_state if arm_ancestors else None
            parallel = [copy.deepcopy(nodes[other]) for other in order
                        if other != node_id and other not in ancestors[node_id] and node_id not in ancestors[other]]
            prediction = {"facts": facts, "geometry": prior, "parallel_nodes": parallel,
                          "ancestor_nodes": tuple(copy.deepcopy(nodes[a]) for a in order if a in ancestors[node_id])}
            geometry = self._geometry(nodes[node_id], snapshot, prediction, "admission")
            self._check_preconditions(nodes[node_id], facts, geometry)
            checks[node_id] = geometry
        # Observations may have changed while an expensive GPU admission ran.
        current = self.world.snapshot()
        if (current.snapshot_id != snapshot.snapshot_id or current.execution_epoch != snapshot.execution_epoch
                or current.facts != snapshot.facts):
            raise ContractError("world changed during whole-plan admission; regenerate/revalidate")
        receipt = AdmissionReceipt("admission-" + uuid.uuid4().hex, digest(candidate), candidate["task_id"],
                                   candidate["execution_epoch"], candidate["snapshot_id"], order,
                                   MappingProxyType({n: frozenset(a) for n, a in ancestors.items()}),
                                   MappingProxyType(checks), self.world.clock())
        with self._lock:
            self._admissions[receipt.receipt_id] = receipt
        return receipt

    def _lookup_admission(self, plan, admission):
        plan_digest = digest(plan)
        with self._lock:
            if admission is None:
                matches = [a for a in self._admissions.values() if a.plan_digest == plan_digest]
                if not matches:
                    raise ContractError("plan has no whole-candidate admission")
                admission = matches[-1]
            if self._admissions.get(admission.receipt_id) is not admission or admission.plan_digest != plan_digest:
                raise ContractError("admission receipt does not authorize this exact plan")
        return admission

    def dispatch(self, plan, node_id: str, admission=None, *, completed_nodes=None) -> DispatchReceipt:
        candidate = self.catalog.validate_plan_json(plan, available=self.registry.available_skills)
        admission = self._lookup_admission(candidate, admission)
        nodes, _, ancestors, incoming = _graph(candidate)
        if node_id not in nodes:
            raise ContractError("unknown dispatch node")
        snapshot = self.world.snapshot()
        context = snapshot.context
        if admission.execution_epoch != snapshot.execution_epoch or admission.task_id != context["task_id"]:
            raise ContractError("admission belongs to a revoked task or epoch")
        completed = set(context["completed_nodes"] if completed_nodes is None else completed_nodes)
        if not incoming[node_id] <= completed:
            raise ContractError("predecessor effects have not been committed")
        if node_id in completed:
            raise ContractError("cannot dispatch a completed node")
        node = nodes[node_id]
        self.registry.require(node["skill"])
        self._bindings(node, snapshot)
        prediction = {"facts": dict(snapshot.facts), "geometry": None,
                      "parallel_nodes": [copy.deepcopy(other) for other in nodes.values()
                                         if other["id"] != node_id and other["id"] not in ancestors[node_id]
                                         and node_id not in ancestors[other["id"]]]}
        geometry = self._geometry(node, snapshot, prediction, "dispatch")
        self._check_preconditions(node, dict(snapshot.facts), geometry)
        identities = snapshot.identities()
        keys = {"execution_epoch", "base_epoch", "calibration_id", "robot_config_id"}
        if node["skill"] != "observe":
            keys |= {"collision_revision", "attachment_id", "grasp_state_id"}
        keys |= {"entity:" + entity for entity in _entity_references(node)}
        for name in ("profile_id", "constraint_id"):
            if name in node["args"]:
                keys.add(name.removesuffix("_id") + ":" + node["args"][name])
        for condition in self.catalog.bind_conditions(node["skill"], node["args"]):
            key = "fact:" + canonical_json(list(fact_key(condition["predicate"], condition["args"])))
            if key in identities:
                keys.add(key)
        dependencies = {key: identities[key] for key in keys}
        dependencies.update(geometry.dependencies)
        receipt = DispatchReceipt("dispatch-" + uuid.uuid4().hex, admission.receipt_id, digest(candidate), node_id,
                                  digest(node["args"]), snapshot.execution_epoch, snapshot.snapshot_id,
                                  MappingProxyType(dependencies), geometry,
                                  snapshot.captured_at + min(geometry.valid_for_s, self.world.max_evidence_age_s))
        with self._lock:
            self._dispatches[receipt.receipt_id] = receipt
        self.verify_receipt(receipt, candidate, node_id)
        return receipt

    def verify_receipt(self, receipt: DispatchReceipt, plan, node_id: str) -> bool:
        """Recheck at the final command gate, immediately before backend acceptance."""
        with self._lock:
            if self._dispatches.get(receipt.receipt_id) is not receipt:
                raise ContractError("unknown or forged dispatch receipt")
        if receipt.plan_digest != digest(plan) or receipt.node_id != node_id:
            raise ContractError("dispatch receipt does not match exact plan/node")
        nodes = {n["id"]: n for n in plan["nodes"]}
        if node_id not in nodes or receipt.args_digest != digest(nodes[node_id]["args"]):
            raise ContractError("dispatch receipt arguments changed")
        self.registry.require(nodes[node_id]["skill"])
        snapshot = self.world.snapshot()
        if self.world.clock() >= receipt.expires_at or snapshot.execution_epoch != receipt.execution_epoch:
            raise ContractError("dispatch receipt expired or revoked")
        identities = snapshot.identities()
        for key, expected in receipt.dependencies.items():
            if identities.get(key) != expected:
                raise ContractError("dispatch receipt dependency changed: " + key)
        return True
