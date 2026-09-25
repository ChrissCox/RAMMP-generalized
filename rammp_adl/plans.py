"""Plans for common goals from local templates, and local recovery plans after a failed step.

The sequencing for opening a door, picking up and holding a part, reaching a
pose, looking at a part or setting the gripper is the same every time; only the
entities, the constraint, the target and the profiles change, and the scene
holds all of those. Building those plans locally spares a planner request per
task; Astra is still asked for anything a template does not cover, and every
template plan passes the same admission as Astra's before it runs.

Recovery: after a step fails, the plan's remaining steps can be run again from
the failed one, or after looking at the part again. Which one (or Astra, or
stop) is a decision for the caller; this only builds them.
"""
from __future__ import annotations

import copy

OPEN_APERTURE_M = .08                               # a 2F-85 opened nearly fully before reaching for a handle


def _profiles(context):
    chosen = {}
    for profile in context.get("profiles", ()):
        chosen.setdefault(profile.get("safety_class"), profile["profile_id"])
    return chosen


def _frame(plan_nodes, context, catalog):
    snapshot_keys = {key: context[key] for key in ("task_id", "snapshot_id", "execution_epoch")}
    nodes = [node for node, _ in plan_nodes]
    edges = [{"from": before, "to": node["id"]} for node, afters in plan_nodes for before in afters]
    return {"schema_version": "1.0.0", "skill_library_hash": catalog.hash, **snapshot_keys, "nodes": nodes, "edges": edges}


def _reach_and_grasp(entity_id, profiles, *, prefix="t"):
    return [({"id": f"{prefix}_pregrasp", "skill": "move_to_pose",
              "args": {"target": {"entity_id": entity_id, "pose_role": "pregrasp"}, "profile_id": profiles["transit"]}}, []),
            ({"id": f"{prefix}_open_hand", "skill": "set_gripper",
              "args": {"aperture_m": OPEN_APERTURE_M, "profile_id": profiles["gripper"]}}, []),
            ({"id": f"{prefix}_to_grasp", "skill": "move_to_pose",
              "args": {"target": {"entity_id": entity_id, "pose_role": "grasp"}, "profile_id": profiles["transit"]}},
             [f"{prefix}_pregrasp", f"{prefix}_open_hand"]),
            ({"id": f"{prefix}_grasp", "skill": "grasp", "args": {"entity_id": entity_id, "profile_id": profiles["gripper"]}},
             [f"{prefix}_to_grasp"])]


def template_for_goal(goal, context, catalog, *, support_of=None, release=True, retract=True):
    """The plan a common goal always takes, or None when no template covers it.

    support_of maps a handled part to the surface it is attached to (a handle to its door): the part is
    released onto that surface after it has been moved. release and retract are the template's two open
    choices: whether to let go when done, and whether to move the hand back afterwards.
    """
    profiles = _profiles(context)
    if not {"transit", "gripper"} <= set(profiles):
        return None
    predicate, args = goal["predicate"], goal["args"]
    entities = {entity["entity_id"]: entity for entity in context["entities"]}
    steps = None
    if predicate == "constraint_goal_verified":
        constraint = next((c for c in context.get("constraints", ()) if c["constraint_id"] == args["constraint_id"]), None)
        if constraint is None or "contact" not in profiles:
            return None
        part = constraint["entity_id"]
        roles = set(entities.get(part, {}).get("pose_roles") or ())
        if not {"pregrasp", "grasp"} <= roles:
            return None
        steps = _reach_and_grasp(part, profiles)
        steps.append(({"id": "t_follow", "skill": "follow_constraint",
                       "args": {"entity_id": part, "constraint_id": constraint["constraint_id"],
                                "target_value": args["target_value"], "target_unit": args["target_unit"],
                                "profile_id": profiles["contact"]}}, ["t_grasp"]))
        support = (support_of or {}).get(part)
        last = "t_follow"
        if release and support in entities:
            steps.append(({"id": "t_release", "skill": "release",
                           "args": {"entity_id": part, "support_id": support, "profile_id": profiles["gripper"]}}, [last]))
            last = "t_release"
            if retract and "retract" in roles:
                steps.append(({"id": "t_retract", "skill": "move_to_pose",
                               "args": {"target": {"entity_id": part, "pose_role": "retract"},
                                        "profile_id": profiles["transit"]}}, [last]))
    elif predicate == "holding":
        part = args["entity_id"]
        if not {"pregrasp", "grasp"} <= set(entities.get(part, {}).get("pose_roles") or ()):
            return None
        steps = _reach_and_grasp(part, profiles)
    elif predicate == "at_pose":
        steps = [({"id": "t_reach", "skill": "move_to_pose",
                   "args": {"target": {"entity_id": args["entity_id"], "pose_role": args["pose_role"]},
                            "profile_id": profiles["transit"]}}, [])]
    elif predicate == "observation_valid":
        steps = [({"id": "t_look", "skill": "observe",
                   "args": {"entity_id": args["entity_id"], "camera": "wrist", "purpose": args["purpose"]}}, [])]
    elif predicate == "aperture_reached":
        steps = [({"id": "t_hand", "skill": "set_gripper",
                   "args": {"aperture_m": args["aperture_m"], "profile_id": profiles["gripper"]}}, [])]
    return None if steps is None else _frame(steps, context, catalog)


def recovery_plans(plan, failed_node_id, context, catalog):
    """Local ways to go on after a failed step: {key: (what it does, plan)}.

    retry_from_failed runs the failed step and everything after it again from where the arm is; with
    look_again first, the part the failed step acts on is observed afresh before that. Both keep the
    original arguments: a constraint's target is the part's absolute goal, so a pull that stopped
    part way continues from there rather than starting over.
    """
    nodes = {node["id"]: node for node in plan["nodes"]}
    if failed_node_id not in nodes:
        return {}
    after = {edge["to"]: [] for edge in plan["edges"]}
    for edge in plan["edges"]:
        after.setdefault(edge["from"], []).append(edge["to"])
    keep, frontier = [], [failed_node_id]
    while frontier:
        node_id = frontier.pop()
        if node_id not in keep:
            keep.append(node_id)
            frontier.extend(after.get(node_id, ()))
    order = [node["id"] for node in plan["nodes"] if node["id"] in keep]
    renamed = {node_id: f"retry_{node_id}" for node_id in order}          # fresh ids: this is a new plan
    rest = []
    for node_id in order:
        node = copy.deepcopy(nodes[node_id])
        node["id"] = renamed[node_id]
        before = [renamed[edge["from"]] for edge in plan["edges"] if edge["to"] == node_id and edge["from"] in keep]
        rest.append((node, before))
    failed = nodes[failed_node_id]
    target = failed["args"].get("entity_id") or (failed["args"].get("target") or {}).get("entity_id")
    options = {"retry_from_failed": (f"run the failed step ({failed['skill']}) and the steps after it again from where the arm is now",
                                     _frame(rest, context, catalog))}
    if target and failed["skill"] in ("move_to_pose", "grasp", "observe"):
        look = ({"id": "retry_look", "skill": "observe", "args": {"entity_id": target, "camera": "wrist", "purpose": "pose"}}, [])
        relinked = [look]+[(node, before or ["retry_look"]) for node, before in rest]
        options["look_again_then_retry"] = (f"look at the {target} again, then retry the failed step and the steps after it",
                                            _frame(relinked, context, catalog))
    return options
