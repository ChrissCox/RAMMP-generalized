"""Task intake: task text -> one normalized goal -> a fresh world for the task.

The runtime's world holds a trusted goal that plans cannot change. Intake is
where that goal comes from when the operator types a task: Astra binds the
text to one goal predicate over the entities the operator declared, and the
binding is validated locally before it is installed. Entity existence is
seeded from what the grounded scene currently sees, never from the context JSON.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from .contracts import ContractError, checked_copy, digest
from .sheppy_backend import assertion

ID_ARGS = {"entity_id", "support_id", "robot_id"}
VISIBILITY_SOURCE = "local_grounded_scene"


class IntakeError(Exception):
    def __init__(self, status, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def new_task_id(clock=time.time):
    stamp = datetime.fromtimestamp(clock(), tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"task-{stamp}-{uuid.uuid4().hex[:6]}"


def goal_candidates(catalog, available_skills):
    """Postcondition predicates of the skills that can actually run, catalog order."""
    available = set(available_skills)
    names = []
    for skill_id, skill in catalog.skills.items():
        if skill_id not in available:
            continue
        for condition in skill["postconditions"]:
            if condition["predicate"] not in names:
                names.append(condition["predicate"])
    return tuple(names)


def draft_context(base, descriptors, *, task_id, camera_id, goal=None, constraints=()):
    """A task context from the bench context and the scene's entity descriptors ({entity_id, label, pose_roles})."""
    context = checked_copy(base)
    robot = [entity for entity in context["entities"] if entity["entity_id"] == "robot"]
    if len(robot) != 1:
        raise ContractError("the base context must declare exactly one robot entity")
    entities = [dict(robot[0], facts=[], age_s=0)]
    for entity in descriptors:
        if entity["entity_id"] == "robot":
            raise ContractError("a scene entity cannot be called robot")
        entities.append({"entity_id": entity["entity_id"], "label": entity["label"], "entity_revision": 1,
                         "pose_roles": sorted(entity["pose_roles"]), "facts": [], "confidence": 1.0, "age_s": 0,
                         "position_validity": "unknown", "orientation_validity": "unknown",
                         "source_ids": [camera_id]})
    known = {entity["entity_id"] for entity in entities}
    for constraint in constraints:
        if constraint["entity_id"] not in known:
            raise ContractError(f"constraint {constraint['constraint_id']} binds an entity that is not in the scene")
    context.update(task_id=task_id, snapshot_id=f"{task_id}-1", execution_epoch=1, revision=1,
                   entities=entities, constraints=[checked_copy(c) for c in constraints], available_skills=[], completed_nodes=[],
                   grasp_state_id="empty-at-intake", goal=goal if goal is not None else context["goal"])
    return context


def validate_goal(catalog, goal, context, candidates):
    """The goal is one offered predicate bound to ids that exist in this context."""
    if not isinstance(goal, dict) or set(goal) != {"predicate", "args"}:
        raise IntakeError("INVALID_OUTPUT", "a goal is exactly a predicate with args")
    if goal["predicate"] not in candidates:
        raise IntakeError("UNSUPPORTED", f"{goal['predicate']} is not an outcome any available skill establishes")
    try:
        catalog.validate_predicate(goal["predicate"], goal["args"])
    except ContractError as exc:
        raise IntakeError("INVALID_OUTPUT", str(exc)) from exc
    entity_ids = {entity["entity_id"] for entity in context["entities"]}
    constraint_ids = {constraint["constraint_id"] for constraint in context["constraints"]}
    profile_ids = {profile["profile_id"] for profile in context["profiles"]}
    for name, value in goal["args"].items():
        pool = entity_ids if name in ID_ARGS else constraint_ids if name == "constraint_id" else profile_ids if name == "profile_id" else None
        if pool is not None and value not in pool:
            raise IntakeError("INVALID_OUTPUT", f"goal argument {name}={value!r} names nothing in this scene")
    return checked_copy(goal)


async def normalize_task(reasoner, catalog, context, task_text, *, available_skills):
    """Ask Astra for the goal, then accept it only if it validates locally."""
    candidates = goal_candidates(catalog, available_skills)
    if not candidates:
        raise IntakeError("NEED_CAPABILITY", "no registered skill establishes any outcome; nothing can be a goal")
    result = await reasoner.normalize_goal(context, task_text, predicates=candidates)
    if result.status != "OK" or result.goal is None:
        raise IntakeError(result.status, result.detail)
    return validate_goal(catalog, result.goal, context, candidates)


def seed_visibility(world, scene, *, now, max_age_s=60., ttl_s=None):
    """Commit entity_exists for every entity the scene placed recently."""
    visible = scene.visible_entities(now, max_age_s=max_age_s)
    known = {entity["entity_id"] for entity in world.snapshot().context["entities"]}
    present = sorted(entity_id for entity_id, record in visible.items() if record["visible"] and entity_id in known)
    if not present:
        return {"visible": [], "evidence_id": None, "detail": visible}
    facts = [assertion("entity_exists", {"entity_id": entity_id}) for entity_id in present]
    authority = world.authorize_source(VISIBILITY_SOURCE, ["entity_exists"])
    evidence_id = "visibility-"+digest({"now": now, "visible": {k: visible[k]["report_digest"] for k in present}})
    world.register_evidence(evidence_id, source=authority, predicates=facts,
                            ttl_s=world.max_evidence_age_s if ttl_s is None else ttl_s, observed_at=now,
                            data={"camera_id": scene.camera_id, "captures": {k: visible[k]["capture_id"] for k in present}})
    snapshot = world.snapshot()
    key = "operation-"+evidence_id
    world.register_operation(key, snapshot.execution_epoch)
    world.commit_effects(key, [{**fact, "evidence_id": evidence_id} for fact in facts],
                         snapshot.revision, snapshot.execution_epoch, backend_quiescent=True)
    return {"visible": present, "evidence_id": evidence_id, "detail": visible}


CONSTRAINT_SOURCE = "constraint_library"


def seed_articulation(world, articulations, *, now, ttl_s=None):
    """Commit constraint_valid for each installed record; the plan may then use it."""
    committed = []
    for item in articulations:
        record, constraint = item["record"], item["constraint"]
        facts = [assertion("constraint_valid", {"constraint_id": constraint["constraint_id"]})]
        authority = world.authorize_source(CONSTRAINT_SOURCE, ["constraint_valid"])
        evidence_id = "constraint-"+digest({"record": record.get("digest"), "version": record.get("parameters_version"), "now": now})
        world.register_evidence(evidence_id, source=authority, predicates=facts,
                                ttl_s=world.max_evidence_age_s if ttl_s is None else ttl_s, observed_at=now,
                                data={"constraint_id": constraint["constraint_id"], "kind": record["kind"],
                                      "parameters_version": record.get("parameters_version"), "history": item.get("history", [])})
        snapshot = world.snapshot()
        key = "operation-"+evidence_id
        world.register_operation(key, snapshot.execution_epoch)
        world.commit_effects(key, [{**fact, "evidence_id": evidence_id} for fact in facts],
                             snapshot.revision, snapshot.execution_epoch, backend_quiescent=True)
        committed.append(evidence_id)
    return committed


async def seed_observations(runtime, entity_ids, *, node_id="intake-observe", camera="wrist"):
    """Commit a pose observation for each entity before planning.

    The plan's own observe nodes refresh these; having them first lets the
    planner's move targets pass dispatch even when the model omits an observe,
    and shows the world what discovery measured. Each observation goes
    through the backend's observe path and is committed exactly as the
    executor commits a node's outcome.
    """
    from .handlers import BackendFailure, ExecutionContext
    world, executor = runtime.world, runtime.executor
    authority = executor._authorities.get("observe")
    if authority is None:
        return {"observed": [], "skipped": {entity_id: "observe is not registered" for entity_id in entity_ids}}
    observed, skipped = [], {}
    for index, entity_id in enumerate(entity_ids):
        snapshot = world.snapshot()
        entity = next((e for e in snapshot.context["entities"] if e["entity_id"] == entity_id), None)
        if entity is None or not entity.get("pose_roles"):
            skipped[entity_id] = "no pose roles"
            continue
        owner = f"{node_id}-{index}"
        context = ExecutionContext(task_id=snapshot.context["task_id"], node_id=owner, execution_epoch=snapshot.execution_epoch,
                                   snapshot=snapshot)
        try:
            outcome = await runtime.backend.observe({"entity_id": entity_id, "camera": camera, "purpose": "pose"}, context)
        except BackendFailure as exc:
            skipped[entity_id] = f"{exc.code}: {exc}"
            continue
        if outcome.status != "succeeded":
            skipped[entity_id] = f"{outcome.status}: {getattr(outcome, 'detail', '')}"
            continue
        try:
            for record in outcome.evidence:
                world.register_evidence(record["evidence_id"], source=authority, predicates=record["predicates"],
                                        ttl_s=record["ttl_s"], data=record.get("data"), observed_at=record.get("observed_at"),
                                        dependencies=record.get("dependencies"))
            operation = f"{snapshot.context['task_id']}/{snapshot.execution_epoch}/{owner}/1"
            world.register_operation(operation, snapshot.execution_epoch)
            world.commit_effects(operation, outcome.proposed_effects, snapshot.revision, snapshot.execution_epoch,
                                 backend_quiescent=True, metric_poses=outcome.metric_poses, metric_source=authority)
        except ContractError as exc:
            skipped[entity_id] = f"commit refused: {exc}"
            continue
        observed.append(entity_id)
    return {"observed": observed, "skipped": skipped}


async def search_until_visible(scene, backend, reasoner, context, task_text, *, max_viewpoints=6, still_s=.5, log=None):
    """Discover; while the model says the target is out of view, turn the camera as it suggests and look again.

    Each viewpoint costs one discovery request and one guarded transit. Entities
    seen from every viewpoint accumulate in the scene. Returns the last
    discovery result and the moves made; raises IntakeError when the target
    never appeared.
    """
    from .handlers import BackendFailure, ExecutionContext
    from .perception.grounded_scene import SceneError
    moves, last_hint = [], None
    say = log or (lambda message: None)
    for viewpoint in range(max_viewpoints+1):
        try:
            found = await scene.discover(reasoner, context, task_text)
        except SceneError as exc:
            raise IntakeError(exc.status, exc.detail)
        search = found.get("search") or {}
        found["viewpoints"] = list(moves)
        if search.get("target_visible", True) or search.get("search_hint", "none") == "none":
            return found
        if viewpoint == max_viewpoints:
            break
        hint = search["search_hint"]
        if hint == last_hint and hint in ("back", "closer"):
            hint = "left"                                    # do not walk the arm along one line
        say(f"target not in view ({search.get('search_note', '')}); looking {hint}")
        execution = ExecutionContext(task_id=context["task_id"], node_id=f"search-{viewpoint}",
                                     execution_epoch=context["execution_epoch"])
        try:
            move = await backend.look(hint, execution)
        except BackendFailure as exc:
            say(f"look {hint} refused ({exc.code}: {exc}); trying the other way")
            moves.append({"hint": hint, "status": f"refused: {exc.code}"})
            opposite = {"left": "right", "right": "left", "up": "down", "down": "up", "back": "left", "closer": "back"}[hint]
            try:
                move = await backend.look(opposite, execution)
                hint = opposite
            except BackendFailure as second:
                raise IntakeError("NEED_OBSERVATION", f"could not turn the camera ({second.code}: {second})") from second
        moves.append({"hint": hint, "status": move["receipt_status"]})
        last_hint = hint
        if not await backend.client.stationary(duration_s=still_s):
            raise IntakeError("NEED_OBSERVATION", "the arm did not settle after a look move")
    raise IntakeError("NEED_OBSERVATION", f"the target was not found after {max_viewpoints} viewpoints: "
                                          f"{(search.get('search_note') or '')[:120]}")
