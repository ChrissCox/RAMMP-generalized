"""Sole-writer scene state and evidence-based, idempotent effect commits.

All timestamps in this in-process API use the injected monotonic clock. Camera or
ROS time must be converted by its trusted adapter; mixing clock domains is rejected
by freshness checks, not silently interpreted as fresh evidence.
"""
from __future__ import annotations

import copy
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .contracts import (Catalog, ContractError, MAX_COUNTER, canonical_json,
                        checked_copy, digest, fact_key, validate_schema)


class WorldConflict(ContractError):
    """A completed operation's dependencies changed; never repeat its motion."""


@dataclass(frozen=True)
class MetricPose:
    entity_id: str
    pose_role: str
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    covariance: tuple[float, ...]
    captured_at: float
    frame_id: str
    entity_revision: int
    calibration_id: str
    base_epoch: str
    evidence_id: str
    valid_for_s: float

    def __post_init__(self):
        object.__setattr__(self, "position_m", tuple(self.position_m))
        object.__setattr__(self, "orientation_xyzw", tuple(self.orientation_xyzw))
        object.__setattr__(self, "covariance", tuple(self.covariance))
        if len(self.position_m) != 3 or len(self.orientation_xyzw) != 4 or len(self.covariance) != 36:
            raise ContractError("metric pose requires position[3], quaternion xyzw[4], covariance[36]")
        numbers = (*self.position_m, *self.orientation_xyzw, *self.covariance, self.captured_at, self.valid_for_s)
        if any(type(n) not in (int, float) or not math.isfinite(n) for n in numbers):
            raise ContractError("metric pose contains non-finite or non-numeric values")
        if abs(sum(q * q for q in self.orientation_xyzw) - 1.0) > 1e-6:
            raise ContractError("metric quaternion is not normalized")
        if self.valid_for_s <= 0 or any(self.covariance[i * 6 + i] < 0 for i in range(6)):
            raise ContractError("invalid pose validity or covariance")
        for i in range(6):
            for j in range(6):
                if abs(self.covariance[i * 6 + j] - self.covariance[j * 6 + i]) > 1e-9:
                    raise ContractError("pose covariance is not symmetric")
        # A symmetric matrix with positive diagonal can still encode impossible
        # negative uncertainty. Check positive semidefiniteness by Schur updates.
        residual = [[self.covariance[i * 6 + j] for j in range(6)] for i in range(6)]
        for pivot in range(6):
            value = residual[pivot][pivot]
            if value < -1e-10:
                raise ContractError("pose covariance is not positive semidefinite")
            if abs(value) <= 1e-10:
                if any(abs(residual[row][pivot]) > 1e-9 for row in range(pivot + 1, 6)):
                    raise ContractError("pose covariance is not positive semidefinite")
                continue
            for row in range(pivot + 1, 6):
                for column in range(pivot + 1, 6):
                    residual[row][column] -= residual[row][pivot] * residual[pivot][column] / value
        if not all((self.entity_id, self.pose_role, self.frame_id, self.calibration_id,
                    self.base_epoch, self.evidence_id)):
            raise ContractError("metric pose identities must be nonempty")
        if type(self.entity_revision) is not int or not 0 <= self.entity_revision <= MAX_COUNTER:
            raise ContractError("invalid entity revision")


@dataclass(frozen=True, eq=False)
class EvidenceAuthority:
    """Identity token issued only to a trusted local evaluator, never a model."""
    name: str
    predicates: frozenset[str]
    _nonce: str = field(default_factory=lambda: uuid.uuid4().hex, repr=False)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source: str
    source_id: str
    assertions: frozenset[tuple[str, str, str]]
    observed_at: float
    expires_at: float
    data_json: str
    dependencies_json: str


@dataclass(frozen=True)
class FactRecord:
    predicate: str
    args_json: str
    validity: str
    evidence_id: str
    owner_id: str


@dataclass(frozen=True)
class CommitReceipt:
    receipt_id: str
    idempotency_key: str
    revision: int
    snapshot_id: str
    execution_epoch: int
    rebased: bool
    effects_digest: str


@dataclass(frozen=True)
class WorldSnapshot:
    """A detached context and immutable metric/fact records at one instant."""
    _context_json: str
    captured_at: float
    metric_poses: Mapping[tuple[str, str], MetricPose]
    _facts: Mapping[tuple[str, str], str]
    _dependencies: Mapping[str, Any]

    @property
    def context(self) -> dict:
        from .contracts import strict_loads
        return strict_loads(self._context_json)

    @property
    def revision(self) -> int:
        return self.context["revision"]

    @property
    def execution_epoch(self) -> int:
        return self.context["execution_epoch"]

    @property
    def snapshot_id(self) -> str:
        return self.context["snapshot_id"]

    @property
    def facts(self) -> Mapping[tuple[str, str], str]:
        return self._facts

    def fact(self, predicate: str, args: Mapping[str, Any]) -> str:
        return self._facts.get(fact_key(predicate, args), "unknown")

    def identities(self) -> dict:
        return copy.deepcopy(dict(self._dependencies))


class WorldModel:
    def __init__(self, context, catalog: Catalog, *, clock: Callable[[], float] = time.monotonic,
                 trust_initial: bool = False, max_evidence_age_s: float = 5.0):
        self.catalog = catalog
        self.clock = clock
        if not math.isfinite(max_evidence_age_s) or max_evidence_age_s <= 0:
            raise ContractError("evidence age limit must be finite and positive")
        self.max_evidence_age_s = max_evidence_age_s
        self._context = checked_copy(context)
        validate_schema(self._context, catalog.world_schema, "context")
        self._lock = threading.RLock()
        self._authorities: set[EvidenceAuthority] = set()
        self._evidence: dict[str, EvidenceRecord] = {}
        self._facts: dict[tuple[str, str], FactRecord] = {}
        self._poses: dict[tuple[str, str], MetricPose] = {}
        self._commits: dict[str, tuple[str, CommitReceipt]] = {}
        self._committed_effect_payloads: dict[str, str] = {}
        self._operations: dict[str, tuple[int, dict]] = {}
        self._reconciling: set[int] = set()
        self._sealed: set[int] = set()
        self._created_at = self.clock()
        self._entity_ages = {}
        self._validate_ids()
        initial = {}
        for entity in self._context["entities"]:
            self._entity_ages[entity["entity_id"]] = self._created_at - entity["age_s"]
            for fact in entity["facts"]:
                self.catalog.validate_predicate(fact["predicate"], fact["args"])
                key = fact_key(fact["predicate"], fact["args"])
                if key in self._facts:
                    raise ContractError("duplicate bound fact in initial world")
                self._facts[key] = FactRecord(fact["predicate"], canonical_json(fact["args"]),
                                              fact["validity"], fact["evidence_id"], entity["entity_id"])
                initial.setdefault(fact["evidence_id"], []).append(fact)
        if trust_initial:
            authority = self.authorize_source("explicit_simulation_bootstrap", catalog.predicates)
            for evidence_id, facts in initial.items():
                oldest = max(f["age_s"] for f in facts)
                self.register_evidence(evidence_id, source=authority, predicates=facts,
                                       ttl_s=max_evidence_age_s,
                                       observed_at=self._created_at - oldest)

    def _validate_ids(self):
        for name, key in (("entities", "entity_id"), ("constraints", "constraint_id"), ("profiles", "profile_id")):
            values = [record[key] for record in self._context[name]]
            if len(values) != len(set(values)):
                raise ContractError("duplicate " + key)
        entity_ids = {e["entity_id"] for e in self._context["entities"]}
        for constraint in self._context["constraints"]:
            if constraint["entity_id"] not in entity_ids or constraint["minimum"] > constraint["maximum"]:
                raise ContractError("constraint has missing entity or inverted bounds")

    def authorize_source(self, name: str, predicates) -> EvidenceAuthority:
        """Trusted composition-root API. Never expose this through model/ROS JSON."""
        allowed = frozenset(predicates)
        if not name or not allowed <= self.catalog.predicates.keys():
            raise ContractError("invalid evaluator name or predicate scope")
        with self._lock:
            authority = EvidenceAuthority(name, allowed)
            self._authorities.add(authority)
            return authority

    def register_evidence(self, evidence_id: str, *, source: EvidenceAuthority,
                          predicates, ttl_s: float, data=None, observed_at: float | None = None,
                          dependencies: Mapping[str, Any] | None = None) -> EvidenceRecord:
        with self._lock:
            if not isinstance(source, EvidenceAuthority) or source not in self._authorities:
                raise ContractError("evidence source is not an authorized local evaluator")
            if not isinstance(evidence_id, str) or not evidence_id:
                raise ContractError("evidence ID must be nonempty")
            now = self.clock()
            observed_at = now if observed_at is None else observed_at
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (ttl_s, observed_at)):
                raise ContractError("invalid evidence timing")
            if not 0 < ttl_s <= self.max_evidence_age_s or observed_at > now:
                raise ContractError("evidence TTL exceeds policy or capture time is in the future")
            assertions = set()
            for statement in predicates:
                name, args = statement["predicate"], statement["args"]
                validity = statement.get("validity", "true")
                self.catalog.validate_predicate(name, args)
                if name not in source.predicates or validity not in {"true", "false", "unknown"}:
                    raise ContractError("evidence assertion exceeds evaluator scope")
                key = fact_key(name, args)
                if any(a[:2] == key and a[2] != validity for a in assertions):
                    raise ContractError("contradictory evidence assertions")
                assertions.add((*key, validity))
            record = EvidenceRecord(evidence_id, source.name, source._nonce, frozenset(assertions), observed_at,
                                    observed_at + ttl_s, canonical_json(checked_copy(data)),
                                    canonical_json(checked_copy(dict(dependencies or {}))))
            if evidence_id in self._evidence and self._evidence[evidence_id] != record:
                raise ContractError("evidence ID reused with different content")
            self._evidence[evidence_id] = record
            return record

    def _effective(self, record: FactRecord, now: float) -> str:
        evidence = self._evidence.get(record.evidence_id)
        assertion = (*fact_key(record.predicate, checked_copy(record.args_json)), record.validity)
        if evidence is None or evidence.expires_at <= now or assertion not in evidence.assertions:
            return "unknown"
        return record.validity

    def _identities(self, now: float) -> dict:
        identities = {key: self._context[key] for key in (
            "execution_epoch", "collision_revision", "base_epoch", "calibration_id", "robot_config_id",
            "attachment_id", "grasp_state_id")}
        for entity in self._context["entities"]:
            identities["entity:" + entity["entity_id"]] = entity["entity_revision"]
        for constraint in self._context["constraints"]:
            identities["constraint:" + constraint["constraint_id"]] = digest(constraint)
        for profile in self._context["profiles"]:
            identities["profile:" + profile["profile_id"]] = digest(profile)
        for key, record in self._facts.items():
            identities["fact:" + canonical_json(list(key))] = canonical_json({
                "evidence_id": record.evidence_id, "validity": record.validity,
                "effective": self._effective(record, now),
            })
        return identities

    def snapshot(self) -> WorldSnapshot:
        with self._lock:
            now = self.clock()
            context = copy.deepcopy(self._context)
            entities = {e["entity_id"]: e for e in context["entities"]}
            for entity in entities.values():
                entity["facts"] = []
                entity["age_s"] = max(0.0, now - self._entity_ages[entity["entity_id"]])
            facts = {}
            for key, record in self._facts.items():
                validity = self._effective(record, now)
                facts[key] = validity
                evidence = self._evidence.get(record.evidence_id)
                entities[record.owner_id]["facts"].append({
                    "predicate": record.predicate, "args": checked_copy(record.args_json),
                    "validity": validity, "evidence_id": record.evidence_id,
                    "age_s": max(0.0, now - evidence.observed_at) if evidence else self.max_evidence_age_s,
                })
            poses = {key: pose for key, pose in self._poses.items()
                     if pose.captured_at <= now < pose.captured_at + pose.valid_for_s
                     and pose.evidence_id in self._evidence and self._evidence[pose.evidence_id].expires_at > now}
            return WorldSnapshot(canonical_json(context), now, MappingProxyType(poses),
                                 MappingProxyType(facts), MappingProxyType(self._identities(now)))

    def replace_goal(self, goal):
        """Install the task's normalized goal; only before any node completed."""
        with self._lock:
            goal = checked_copy(goal)
            if not isinstance(goal, dict) or set(goal) != {"predicate", "args"}:
                raise ContractError("a goal is exactly a predicate with args")
            self.catalog.validate_predicate(goal["predicate"], goal["args"])
            pending = [key for key in self._operations if key not in self._commits]
            if self._context["completed_nodes"] or self._sealed or pending:
                raise ContractError("the goal cannot change once execution has begun")
            self._context["goal"] = goal
            validate_schema(self._context, self.catalog.world_schema, "context")
            self._advance()
            return copy.deepcopy(goal)

    def goal_satisfied(self) -> bool:
        snapshot = self.snapshot()
        goal = snapshot.context["goal"]
        return snapshot.fact(goal["predicate"], goal["args"]) == "true"

    def _advance(self):
        if self._context["revision"] == MAX_COUNTER:
            raise ContractError("world revision exhausted; new session required")
        self._context["revision"] += 1
        self._context["snapshot_id"] = "world-" + uuid.uuid4().hex

    def register_operation(self, key: str, epoch: int, dependencies: Mapping[str, Any] | None = None):
        with self._lock:
            if epoch != self._context["execution_epoch"] or epoch in self._sealed:
                raise WorldConflict("cannot register operation in inactive epoch")
            if not isinstance(key, str) or not key or key in self._operations or key in self._commits:
                raise ContractError("operation key must be fresh and nonempty")
            deps = checked_copy(dict(dependencies or {}))
            self._check_dependencies(deps)
            self._operations[key] = (epoch, deps)

    register_inflight = register_operation

    def _check_dependencies(self, dependencies, *, reconcile=False, terminal=False):
        current = self._identities(self.clock())
        for key, expected in dependencies.items():
            if reconcile and key == "execution_epoch":
                continue
            if terminal and key.startswith("fact:") and key in current:
                # Start-condition evidence may age while the action runs. Fresh
                # terminal evidence is checked independently below. Actual fact
                # replacement/invalidation still prevents a blind rebase.
                previous_fact, current_fact = checked_copy(expected), checked_copy(current[key])
                if all(previous_fact.get(name) == current_fact.get(name) for name in ("evidence_id", "validity")):
                    continue
            if current.get(key) != expected:
                raise WorldConflict("changed dependency: " + key)

    def refresh_operation_dependencies(self, key: str, dependencies: Mapping[str, Any], *,
                                       source: EvidenceAuthority, validated_update_id: str):
        """Accept dependencies of a command-gate-validated active-skill update.

        This trusted adapter API does not validate a trajectory itself. The caller
        must hold the command gateway's authority and supply its accepted update
        identity. Stable target IDs, epoch, base, calibration and attachments are
        preserved. Calling it from model output would violate the trust boundary.
        """
        with self._lock:
            if source not in self._authorities or source.name != "command_gateway":
                raise ContractError("operation dependency refresh requires command gateway authority")
            if not isinstance(validated_update_id, str) or not validated_update_id:
                raise ContractError("validated trajectory update identity required")
            operation = self._operations.get(key)
            if operation is None or operation[0] != self._context["execution_epoch"]:
                raise WorldConflict("cannot refresh a revoked or unknown operation")
            updated = checked_copy(dict(dependencies))
            if set(updated) != set(operation[1]):
                raise ContractError("rolling update cannot drop or add dependency identities")
            for name, previous in operation[1].items():
                mutable = name == "collision_revision" or name.startswith(("entity:", "constraint:", "fact:"))
                if not mutable and updated[name] != previous:
                    raise WorldConflict("rolling update changed stable identity: " + name)
            self._check_dependencies(updated)
            self._operations[key] = (operation[0], updated)

    def commit_effects(self, key: str, effects, base_revision: int, execution_epoch: int,
                       *, backend_quiescent: bool = True, completed_node: str | None = None,
                       dependencies: Mapping[str, Any] | None = None,
                       expected_postconditions=None, metric_poses=(),
                       metric_source: EvidenceAuthority | None = None) -> CommitReceipt:
        """Atomically commit measured terminal effects; rebase only compatible state.

        A conflicting effect remains in the evidence registry for reconciliation.
        The caller must stop/replan, never rerun an already-completed physical action.
        """
        with self._lock:
            effects = checked_copy(effects)
            if type(effects) is not list:
                raise ContractError("effects must be an array")
            if not isinstance(metric_poses, (list, tuple)) or any(
                    not isinstance(pose, MetricPose) for pose in metric_poses):
                raise ContractError("metric updates require canonical MetricPose records")
            metric_poses = tuple(metric_poses)
            payload = {"effects": effects, "epoch": execution_epoch, "completed_node": completed_node}
            if metric_poses:
                payload["metric_poses"] = [
                    {name: list(value) if isinstance(value, tuple) else value
                     for name, value in asdict(pose).items()} for pose in metric_poses]
            payload_digest = digest(payload)
            previous = self._commits.get(key)
            if previous:
                if previous[0] != payload_digest:
                    raise ContractError("idempotency key reused with changed effects")
                return previous[1]
            if not backend_quiescent:
                raise ContractError("terminal effects require verified backend quiescence")
            operation = self._operations.get(key)
            if operation is None or operation[0] != execution_epoch:
                raise WorldConflict("unregistered terminal operation")
            current_epoch = self._context["execution_epoch"]
            reconcile = execution_epoch in self._reconciling
            if execution_epoch in self._sealed or (execution_epoch != current_epoch and not reconcile):
                raise WorldConflict("terminal result belongs to a sealed or unknown epoch")
            if type(base_revision) is not int or not 0 <= base_revision <= self._context["revision"]:
                raise WorldConflict("invalid base revision")
            all_dependencies = dict(operation[1])
            for name, value in dict(dependencies or {}).items():
                if name in all_dependencies and all_dependencies[name] != value:
                    raise WorldConflict("cannot replace an operation dependency")
                all_dependencies[name] = value
            self._check_dependencies(all_dependencies, reconcile=reconcile, terminal=True)
            now = self.clock()
            pending = {}
            for effect in effects:
                if set(effect) != {"predicate", "args", "validity", "evidence_id"}:
                    raise ContractError("effect fields do not match typed contract")
                predicate, args = effect["predicate"], effect["args"]
                self.catalog.validate_predicate(predicate, args)
                validity = effect["validity"]
                if validity not in {"true", "false", "unknown"}:
                    raise ContractError("invalid fact validity")
                fact = fact_key(predicate, args)
                if fact in pending:
                    raise ContractError("duplicate bound effect")
                evidence = self._evidence.get(effect["evidence_id"])
                if evidence is None or evidence.expires_at <= now or (*fact, validity) not in evidence.assertions:
                    raise ContractError("effect lacks fresh authorized matching evidence")
                self._check_dependencies(checked_copy(evidence.dependencies_json), reconcile=reconcile)
                owner = self._owner(args)
                pending[fact] = FactRecord(predicate, canonical_json(args), validity, effect["evidence_id"], owner)
            updated_entities = set()
            if metric_poses:
                # Geometry and terminal facts share the original operation's
                # dependency check above. No self-produced revision is installed
                # early and no external revision can be silently rebased over.
                if (metric_source not in self._authorities
                        or metric_source.name != "handler:observe"):
                    raise ContractError("atomic observation geometry requires its registered handler authority")
                updated_entities = {pose.entity_id for pose in metric_poses}
                if len(updated_entities) != 1:
                    raise ContractError("one observation cannot update multiple entities")
                target = next(iter(updated_entities))
                if any(record.predicate not in {"observation_valid", "entity_exists", "pose_valid"}
                       or checked_copy(record.args_json).get("entity_id") != target
                       for record in pending.values()):
                    raise ContractError("observation geometry cannot establish physical or other-entity effects")
                if not any(record.predicate == "observation_valid" and record.validity == "true"
                           for record in pending.values()):
                    raise ContractError("atomic geometry requires its measured observation assertion")
                roles = set()
                revisions = set()
                captures = set()
                for pose in metric_poses:
                    self._validate_metric_pose(pose, source=metric_source)
                    if pose.pose_role in roles:
                        raise ContractError("duplicate observed pose role")
                    roles.add(pose.pose_role)
                    revisions.add(pose.entity_revision)
                    captures.add((pose.captured_at, pose.valid_for_s, pose.evidence_id))
                    record = pending.get(fact_key("pose_valid", {
                        "entity_id": pose.entity_id, "pose_role": pose.pose_role}))
                    evidence = self._evidence[pose.evidence_id]
                    if (record is None or record.validity != "true" or record.evidence_id != pose.evidence_id
                            or evidence.observed_at != pose.captured_at
                            or evidence.expires_at != pose.captured_at + pose.valid_for_s):
                        raise ContractError("metric observation lacks exact capture and terminal pose evidence")
                if len(revisions) != 1 or len(captures) != 1:
                    raise ContractError("one observed entity requires one resulting revision and capture")
                if any(record.predicate == "pose_valid" and record.validity == "true"
                       and checked_copy(record.args_json)["pose_role"] not in roles
                       for record in pending.values()):
                    raise ContractError("new geometry cannot preserve an older pose role")
                if self._context["collision_revision"] == MAX_COUNTER:
                    raise ContractError("world collision counter exhausted")
            if completed_node is not None and (not isinstance(completed_node, str) or not completed_node):
                raise ContractError("completed node ID must be nonempty")
            true_holdings = [record for record in pending.values()
                             if record.predicate == "holding" and record.validity == "true"]
            if len(true_holdings) > 1:
                raise ContractError("single-gripper commit has incompatible simultaneous holding effects")
            if expected_postconditions is not None:
                conditions = checked_copy(expected_postconditions)
                if type(conditions) is not list:
                    raise ContractError("expected postconditions must be a typed array")
                for condition in conditions:
                    if set(condition) != {"predicate", "args"}:
                        raise ContractError("invalid expected postcondition")
                    self.catalog.validate_predicate(condition["predicate"], condition["args"])
                # Preview the exact invalidation rules atomically. A contradictory
                # effect set must never mark a node complete and then discover
                # that its success predicates were deleted by another effect.
                original_facts = self._facts
                self._facts = dict(original_facts)
                try:
                    self._invalidate_metric_facts(updated_entities)
                    for fact, record in pending.items():
                        self._invalidate_conflicting_facts(record)
                        self._facts[fact] = record
                    for condition in conditions:
                        record = self._facts.get(fact_key(condition["predicate"], condition["args"]))
                        if record is None or self._effective(record, now) != "true":
                            raise ContractError("terminal effects do not preserve every required postcondition")
                finally:
                    self._facts = original_facts
            # All checks precede mutation, including counter exhaustion.
            if self._context["revision"] >= MAX_COUNTER:
                raise ContractError("world revision exhausted")
            rebased = base_revision != self._context["revision"]
            if metric_poses:
                self._install_metric_poses(metric_poses)
            for fact, record in pending.items():
                self._invalidate_conflicting_facts(record)
                self._facts[fact] = record
                if record.validity == "true" and record.predicate in {"holding", "released"}:
                    args = checked_copy(record.args_json)
                    self._context["grasp_state_id"] = "grasp-" + uuid.uuid4().hex
                    articulated = any(c["entity_id"] == args["entity_id"] for c in self._context["constraints"])
                    if not articulated:
                        self._context["attachment_id"] = ("attached-" + args["entity_id"] + "-" + uuid.uuid4().hex
                                                          if record.predicate == "holding" else "empty")
            if completed_node is not None and completed_node not in self._context["completed_nodes"]:
                self._context["completed_nodes"].append(completed_node)
            self._advance()
            receipt = CommitReceipt("commit-" + uuid.uuid4().hex, key, self._context["revision"],
                                    self._context["snapshot_id"], execution_epoch, rebased, payload_digest)
            self._commits[key] = (payload_digest, receipt)
            self._committed_effect_payloads[key] = digest({"effects": effects, "epoch": execution_epoch})
            del self._operations[key]
            return receipt

    def committed_receipt(self, key: str, effects, execution_epoch: int) -> CommitReceipt:
        """Read-only exact duplicate lookup for unauthenticated ROS commit clients."""
        with self._lock:
            expected = digest({"effects": checked_copy(effects), "epoch": execution_epoch})
            if key not in self._commits or self._committed_effect_payloads.get(key) != expected:
                raise ContractError("no already-committed operation matches this exact effect payload")
            return self._commits[key][1]

    def _owner(self, args) -> str:
        ids = {e["entity_id"] for e in self._context["entities"]}
        owner = None
        for member in ("entity_id", "robot_id", "support_id"):
            if member in args:
                if args[member] not in ids:
                    raise ContractError("fact references an unknown entity")
                if owner is None:
                    owner = args[member]
        for name, records, identity in (("constraint_id", "constraints", "constraint_id"),
                                        ("profile_id", "profiles", "profile_id")):
            if name in args and args[name] not in {r[identity] for r in self._context[records]}:
                raise ContractError("fact references an unknown " + name)
        if owner is not None:
            return owner
        if not ids:
            raise ContractError("world has no entity record for facts")
        return "robot" if "robot" in ids else sorted(ids)[0]

    def _invalidate_conflicting_facts(self, incoming: FactRecord):
        """Closed physical invariants; invalidation never manufactures new truth."""
        if incoming.validity != "true":
            return
        args = checked_copy(incoming.args_json)
        for key, old in list(self._facts.items()):
            old_args = checked_copy(old.args_json)
            invalidate = False
            if incoming.predicate in {"at_pose", "at_grasp_pose", "at_release_pose"}:
                invalidate = old.predicate in {"at_pose", "at_grasp_pose", "at_release_pose"} and key != fact_key(incoming.predicate, args)
                same_grasp = (args.get("entity_id") == old_args.get("entity_id")
                              and {incoming.predicate, old.predicate} == {"at_pose", "at_grasp_pose"}
                              and (args.get("pose_role") == "grasp" or old_args.get("pose_role") == "grasp"))
                if same_grasp:
                    invalidate = False
            if incoming.predicate == "aperture_reached":
                invalidate |= old.predicate == "aperture_reached" and old_args != args
            if incoming.predicate == "holding":
                invalidate |= old.predicate in {"gripper_empty", "released"}
                invalidate |= old.predicate == "holding" and old_args != args
            if incoming.predicate in {"released", "gripper_empty"}:
                invalidate |= old.predicate == "holding" and (incoming.predicate == "gripper_empty" or old_args.get("entity_id") == args.get("entity_id"))
            if incoming.predicate == "constraint_goal_verified":
                invalidate |= old.predicate == incoming.predicate and old_args.get("constraint_id") == args["constraint_id"] and old_args != args
            if invalidate:
                self._facts[key] = FactRecord(old.predicate, old.args_json, "unknown", old.evidence_id, old.owner_id)

    def _validate_metric_pose(self, pose: MetricPose, *, source: EvidenceAuthority):
        """Validate without mutation while the world writer lock is held."""
        if not isinstance(source, EvidenceAuthority) or source not in self._authorities or "pose_valid" not in source.predicates:
            raise ContractError("metric update requires authorized geometry evaluator")
        if not isinstance(pose, MetricPose):
            raise ContractError("metric update requires a canonical MetricPose")
        now = self.clock()
        entities = {e["entity_id"]: e for e in self._context["entities"]}
        entity = entities.get(pose.entity_id)
        if entity is None or pose.pose_role not in entity["pose_roles"]:
            raise ContractError("unknown entity or pose role")
        if pose.frame_id != self.catalog.library["frames"]["planning"]:
            raise ContractError("metric pose must be transformed into configured planning frame")
        if pose.calibration_id != self._context["calibration_id"] or pose.base_epoch != self._context["base_epoch"]:
            raise WorldConflict("metric pose calibration/base identity mismatch")
        if not pose.captured_at <= now < pose.captured_at + pose.valid_for_s:
            raise WorldConflict("metric pose stale or clock domain mismatch")
        if pose.valid_for_s > self.max_evidence_age_s or pose.entity_revision <= entity["entity_revision"]:
            raise WorldConflict("metric update must advance revision within freshness policy")
        evidence = self._evidence.get(pose.evidence_id)
        assertion = (*fact_key("pose_valid", {"entity_id": pose.entity_id, "pose_role": pose.pose_role}), "true")
        if evidence is None or evidence.expires_at <= now or assertion not in evidence.assertions:
            raise ContractError("metric pose lacks fresh exact pose evidence")
        if evidence.source_id != source._nonce:
            raise ContractError("metric pose evidence evaluator mismatch")

    def _invalidate_metric_facts(self, entity_ids):
        for key, record in list(self._facts.items()):
            if record.predicate == "pose_valid" and checked_copy(record.args_json).get("entity_id") in entity_ids:
                self._facts[key] = FactRecord(record.predicate, record.args_json, "unknown", record.evidence_id, record.owner_id)

    def _install_metric_poses(self, poses):
        """Install a fully validated batch; caller owns all checks and the lock."""
        entity_ids = {pose.entity_id for pose in poses}
        entities = {e["entity_id"]: e for e in self._context["entities"]}
        # All supplied roles share the new entity estimate; other roles expire.
        self._poses = {key: value for key, value in self._poses.items() if key[0] not in entity_ids}
        for pose in poses:
            entity = entities[pose.entity_id]
            entity["entity_revision"] = pose.entity_revision
            entity["position_validity"] = entity["orientation_validity"] = "true"
            self._entity_ages[pose.entity_id] = pose.captured_at
            self._poses[(pose.entity_id, pose.pose_role)] = pose
        self._invalidate_metric_facts(entity_ids)
        self._context["collision_revision"] += 1

    def update_metric_pose(self, pose: MetricPose, *, source: EvidenceAuthority):
        with self._lock:
            self._validate_metric_pose(pose, source=source)
            if self._context["collision_revision"] == MAX_COUNTER or self._context["revision"] == MAX_COUNTER:
                raise ContractError("world counter exhausted")
            self._install_metric_poses((pose,))
            args = {"entity_id": pose.entity_id, "pose_role": pose.pose_role}
            self._facts[fact_key("pose_valid", args)] = FactRecord("pose_valid", canonical_json(args), "true", pose.evidence_id, pose.entity_id)
            self._advance()

    def invalidate_epoch(self) -> int:
        with self._lock:
            old = self._context["execution_epoch"]
            if old == MAX_COUNTER or self._context["revision"] == MAX_COUNTER:
                raise ContractError("execution epoch exhausted; new session required")
            self._reconciling.add(old)
            self._context["execution_epoch"] += 1
            self._advance()
            return self._context["execution_epoch"]

    cancel_epoch = invalidate_epoch

    def reconcile_operation(self, key: str, *, backend_quiescent: bool):
        """Seal an operation with no further effects only after verified stopping."""
        with self._lock:
            if not backend_quiescent:
                raise ContractError("cannot reconcile a moving backend")
            if key not in self._operations:
                raise ContractError("unknown in-flight operation")
            del self._operations[key]

    def seal_epoch(self, epoch: int):
        with self._lock:
            if epoch in self._sealed:
                return
            if epoch not in self._reconciling:
                raise ContractError("epoch is not awaiting reconciliation")
            if any(operation[0] == epoch for operation in self._operations.values()):
                raise WorldConflict("epoch still has unreconciled in-flight operations")
            self._reconciling.remove(epoch)
            self._sealed.add(epoch)
