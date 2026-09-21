"""Offline artifact checks. No ROS, model calls, geometry or hardware authorization.

The small schema interpreter supports only keywords used by these generated artifacts.
A production runtime must use a complete JSON Schema implementation plus semantic,
kinematic and safety validation. --generate writes derived schema/fixture hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("non-finite JSON number: " + value)


def loads(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def read(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def digest(library):
    data = json.dumps(library, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def object_schema(properties):
    return dict(type="object", properties=properties,
                required=list(properties), additionalProperties=False)


def literal_schema(value):
    """Keep explicit scalar types, as emitted by the provider SDK's schemas."""
    types = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}
    require(type(value) in types, "identity literals must be JSON scalars")
    return {"type": types[type(value)], "const": value}


def plan_schema(library, available=None):
    selected = library["skills"] if available is None else [
        s for s in library["skills"] if s["id"] in available]
    require(selected, "no available skills: return NEED_CAPABILITY, not an empty schema")
    variants = [object_schema({
        "id": {"type": "string", "minLength": 1},
        "skill": literal_schema(s["id"]),
        "args": s["arguments"],
    }) for s in selected]
    text = {"type": "string", "minLength": 1}
    schema = object_schema({
        "schema_version": literal_schema("1.0.0"),
        "skill_library_hash": literal_schema(digest(library)),
        "task_id": text,
        "snapshot_id": text,
        "execution_epoch": {"type": "integer", "minimum": 0, "maximum": 9007199254740991},
        "nodes": {"type": "array", "minItems": 1, "maxItems": 64,
                  "items": {"anyOf": variants}},
        "edges": {"type": "array", "maxItems": 512,
                  "items": object_schema({"from": text, "to": text})},
    })
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema



def response_schema(library, available=None):
    plan = plan_schema(library, available)
    plan.pop("$schema")
    schema = object_schema({"result": {"anyOf": [
        object_schema({"status": literal_schema("OK"), "plan": plan}),
        object_schema({
            "status": {"type": "string", "enum": ["NEED_CAPABILITY", "INFEASIBLE", "NEED_OBSERVATION",
                                "AMBIGUOUS", "REFUSED"]},
            "detail": {"type": "string", "minLength": 1},
        }),
    ]}})
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


KEYWORDS = {"$schema", "title", "description", "type", "const", "enum", "anyOf",
            "properties", "required", "additionalProperties", "items", "minItems",
            "maxItems", "minLength", "minimum", "maximum"}


def validate(value, schema, path="$"):
    require(not set(schema) - KEYWORDS, "unsupported schema keywords at " + path)
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                validate(value, branch, path)
                break
            except ValueError:
                pass
        else:
            raise ValueError(path + ": no schema branch accepts value")
    if "const" in schema:
        require(type(value) is type(schema["const"]) and value == schema["const"],
                path + ": wrong constant")
    if "enum" in schema:
        require(any(type(value) is type(v) and value == v for v in schema["enum"]),
                path + ": invalid enum")
    expected = schema.get("type")
    matches = {
        "object": lambda x: type(x) is dict,
        "array": lambda x: type(x) is list,
        "string": lambda x: type(x) is str,
        "boolean": lambda x: type(x) is bool,
        "integer": lambda x: type(x) is int,
        "number": lambda x: type(x) in (int, float) and math.isfinite(x),
    }
    if expected:
        require(expected in matches and matches[expected](value),
                path + ": expected " + expected)
    if type(value) is dict:
        props = schema.get("properties", {})
        require(set(schema.get("required", [])) <= set(value), path + ": missing fields")
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(props), path + ": unknown fields")
        for key in value.keys() & props.keys():
            validate(value[key], props[key], path + "." + key)
    if type(value) is list:
        require(len(value) >= schema.get("minItems", 0), path + ": too few items")
        require(len(value) <= schema.get("maxItems", float("inf")), path + ": too many items")
        for index, child in enumerate(value):
            if "items" in schema:
                validate(child, schema["items"], path + "[" + str(index) + "]")
    if type(value) is str:
        require(len(value) >= schema.get("minLength", 0), path + ": empty string")
    if type(value) in (int, float):
        require(math.isfinite(value), path + ": non-finite number")
        require(value >= schema.get("minimum", -float("inf")), path + ": below minimum")
        require(value <= schema.get("maximum", float("inf")), path + ": above maximum")


def check_catalog(library):
    require(library["status"] in {"design_only", "implementation"}, "unknown catalog status")
    ids = [s["id"] for s in library["skills"]]
    require(len(ids) == len(set(ids)), "duplicate skill ID")
    for skill in library["skills"]:
        require(skill["implementation_status"] in {"planned", "implemented"}, "unknown implementation status")
        require(len(skill["claims"]) == len(set(skill["claims"])), "duplicate claim")
        require(set(skill["claims"]) <= set(library["resources"]), "unknown resource")
        require(all(library["resources"][r]["exclusive"] for r in skill["claims"]), "shared claim")
        require(skill["timing"]["timeout_s"] >= skill["timing"]["nominal_s"] > 0,
                "invalid skill timing")
        require(skill["execution"]["interruptible"], "safety cannot interrupt skill")
        for key in ("arguments", "outputs"):
            schema = skill[key]
            require(schema["type"] == "object" and schema["additionalProperties"] is False,
                    "skill schema must be closed")
            require(set(schema["required"]) == set(schema["properties"]), "untyped optional arg")
        for predicate in skill["preconditions"] + skill["postconditions"]:
            require(predicate["predicate"] in library["predicates"], "unknown predicate")
            parameters = library["predicates"][predicate["predicate"]]["arguments"]["properties"]
            require(set(predicate["bindings"]) == set(parameters), "predicate binding fields differ")
            for name, binding in predicate["bindings"].items():
                if set(binding) == {"literal"}:
                    validate(binding["literal"], parameters[name])
                else:
                    require(set(binding) == {"arg_path"} and binding["arg_path"],
                            "invalid predicate binding")
                    current = skill["arguments"]
                    for segment in binding["arg_path"]:
                        require(segment in current.get("properties", {}), "unbound predicate argument")
                        current = current["properties"][segment]
        codes = [e["code"] for e in skill["failures"]]
        require(len(codes) == len(set(codes)), "duplicate failure code")
        require(all(e["local_retries"] >= 0 for e in skill["failures"]), "negative retry cap")


def check_plan(plan, library, context):
    validate(plan, plan_schema(library, context["available_skills"]))
    for key, context_key in (("task_id", "task_id"), ("snapshot_id", "snapshot_id"),
                             ("execution_epoch", "execution_epoch")):
        require(plan[key] == context[context_key], "stale or wrong " + key)
    nodes = {n["id"]: n for n in plan["nodes"]}
    require(len(nodes) == len(plan["nodes"]), "duplicate node ID")
    successors = {key: set() for key in nodes}
    incoming = dict.fromkeys(nodes, 0)
    seen_edges = set()
    for edge in plan["edges"]:
        a, b = edge["from"], edge["to"]
        require(a in nodes and b in nodes and a != b, "invalid dependency edge")
        require((a, b) not in seen_edges, "duplicate edge")
        seen_edges.add((a, b))
        successors[a].add(b)
        incoming[b] += 1
    ready = [key for key, degree in incoming.items() if degree == 0]
    order = []
    while ready:
        key = ready.pop()
        order.append(key)
        for dest in successors[key]:
            incoming[dest] -= 1
            if incoming[dest] == 0:
                ready.append(dest)
    require(len(order) == len(nodes), "dependency cycle")
    entities = {e["entity_id"]: e for e in context["entities"]}
    profiles = {p["profile_id"]: p for p in context["profiles"]}
    constraints = {c["constraint_id"]: c for c in context["constraints"]}
    require(len(entities) == len(context["entities"]), "duplicate entity ID")
    require(len(profiles) == len(context["profiles"]), "duplicate profile ID")
    require(len(constraints) == len(context["constraints"]), "duplicate constraint ID")
    goal = context["goal"]
    for key, value in goal["args"].items():
        if key.endswith("_id"):
            require(value in entities or value in constraints or value in profiles,
                    "unknown goal subject")
    if goal["predicate"] == "constraint_goal_verified":
        constraint = constraints[goal["args"]["constraint_id"]]
        require(goal["args"]["target_unit"] == constraint["unit"], "goal unit mismatch")
        require(constraint["minimum"] <= goal["args"]["target_value"] <= constraint["maximum"],
                "goal outside constraint bounds")
    for node in plan["nodes"]:
        args = node["args"]
        for key in ("entity_id", "support_id"):
            if key in args:
                require(args[key] in entities, "unknown entity: " + args[key])
        if "profile_id" in args:
            require(args["profile_id"] in profiles, "unknown profile")
        if node["skill"] == "move_to_pose":
            target = args["target"]
            require(target["entity_id"] in entities, "unknown target entity")
            require(target["pose_role"] in entities[target["entity_id"]]["pose_roles"],
                    "unknown pose role")
        if node["skill"] == "follow_constraint":
            require(args["constraint_id"] in constraints, "unknown constraint")
            c = constraints[args["constraint_id"]]
            require(c["validity"] == "true", "constraint evidence is invalid or unknown")
            require(c["entity_id"] == args["entity_id"], "constraint target mismatch")
            require(c["unit"] == args["target_unit"], "constraint units mismatch")
            require(c["minimum"] <= args["target_value"] <= c["maximum"], "constraint bounds")
    return order


def check_links(root):
    pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for path in root.rglob("*.md"):
        for raw in pattern.findall(path.read_text(encoding="utf-8")):
            link = raw.split("#", 1)[0]
            if not link or "://" in link or link.startswith("mailto:"):
                continue
            require((path.parent / link).exists(), "broken local link in " + str(path) + ": " + link)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    library = read(ROOT / "skills/adl_skill_library.yaml")
    check_catalog(library)
    schema = plan_schema(library)
    schema_path = ROOT / "schemas/plan.schema.json"
    response_path = ROOT / "schemas/reasoning-response.schema.json"
    response = response_schema(library)
    plans = sorted((ROOT / "examples").glob("*.plan.json"))
    if args.generate:
        schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        for path in plans:
            plan = read(path)
            plan["skill_library_hash"] = digest(library)
            path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    require(read(schema_path) == schema, "stale generated plan schema; run --generate")
    require(read(response_path) == response, "stale generated reasoning response schema")
    context_schema = read(ROOT / "schemas/world-context.schema.json")
    for path in plans:
        context = read(path.with_name(path.name.replace(".plan.json", ".context.json")))
        validate(context, context_schema)
        check_plan(read(path), library, context)
    for path in ROOT.rglob("*.json"):
        read(path)
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".md", ".py", ".json", ".yaml", ".msg", ".srv", ".action"}:
            content = path.read_text(encoding="utf-8")
            require(not any(ord(c) < 32 and c not in "\r\n\t" for c in content), "control character")
    check_links(ROOT)
    print(f"Design checks passed: {len(library['skills'])} skills, {len(plans)} example plans.")
    print("Offline only: no runtime precondition, collision, ROS, provider or hardware validation.")


if __name__ == "__main__":
    main()
