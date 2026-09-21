"""Strict, catalog-owned runtime contracts. No model output is executable.

The artifact generator is reused only for grammar construction. Runtime validation
uses the complete Draft 2020-12 implementation, never the offline subset checker.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from jsonschema import Draft202012Validator

MAX_COUNTER = 9007199254740991
CURRENT_SKILLS = frozenset({"observe", "move_to_pose", "set_gripper", "grasp", "release", "follow_constraint"})


class ContractError(ValueError):
    """An untrusted value violates a syntactic or semantic contract."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ContractError("non-finite JSON number: " + value)


def _check_tree(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ContractError("JSON exceeds maximum nesting depth")
    if type(value) is float and not math.isfinite(value):
        raise ContractError("non-finite number")
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ContractError("JSON object keys must be strings")
            _check_tree(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _check_tree(child, depth + 1)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ContractError("not a JSON value: " + type(value).__name__)


def strict_loads(value: str | bytes, *, max_bytes: int = 262144) -> Any:
    """Reject duplicate keys, non-finite/overflow numbers, excessive size/depth."""
    if not isinstance(value, (str, bytes)):
        raise ContractError("JSON input must be text or UTF-8 bytes")
    try:
        raw = value.encode("utf-8") if isinstance(value, str) else value
        if len(raw) > max_bytes:
            raise ContractError("JSON exceeds byte limit")
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                            parse_constant=_nonfinite)
        _check_tree(parsed)
        return parsed
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ContractError("invalid JSON: " + str(exc)) from exc


def canonical_json(value: Any) -> str:
    _check_tree(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def checked_copy(value: Any, *, max_bytes: int = 262144) -> Any:
    if isinstance(value, (str, bytes)):
        return strict_loads(value, max_bytes=max_bytes)
    return strict_loads(canonical_json(value), max_bytes=max_bytes)


def validate_schema(value: Any, schema: Mapping[str, Any], label: str = "value") -> None:
    _check_tree(value)
    errors = sorted(Draft202012Validator(schema).iter_errors(value),
                    key=lambda error: "/".join(str(p) for p in error.absolute_path))
    if errors:
        error = errors[0]
        location = "/".join(str(p) for p in error.absolute_path)
        raise ContractError(f"{label}/{location}: {error.message[:500]}")


def _numeric_key(value):
    # JSON numbers 1 and 1.0 bind the same physical predicate argument.
    if type(value) is float and value.is_integer():
        return int(value)
    if type(value) is dict:
        return {k: _numeric_key(v) for k, v in value.items()}
    if type(value) is list:
        return [_numeric_key(v) for v in value]
    return value


def fact_key(predicate: str, args: Mapping[str, Any]) -> tuple[str, str]:
    return predicate, canonical_json(_numeric_key(dict(args)))


class Catalog:
    """Load the single catalog authority and verify its generated grammar."""

    def __init__(self, root_path: str | Path | None = None):
        if root_path is not None:
            self.root = Path(root_path)
        else:
            candidates = (Path(__file__).resolve().parents[1], Path(sys.prefix) / "share/rammp_adl")
            self.root = next((path for path in candidates if (path / "skills/adl_skill_library.yaml").is_file()), candidates[0])
        self.library = strict_loads((self.root / "skills/adl_skill_library.yaml").read_bytes())
        spec = importlib.util.spec_from_file_location("_rammp_artifact_grammar", self.root / "tools/check_design.py")
        if spec is None or spec.loader is None:
            raise ContractError("artifact grammar generator unavailable")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        self._generator = generator
        self.hash = digest(self.library)
        skill_list = self.library["skills"]
        self.skills = {s["id"]: copy.deepcopy(s) for s in skill_list}
        if len(self.skills) != len(skill_list):
            raise ContractError("duplicate catalog skill IDs")
        self.predicates = copy.deepcopy(self.library["predicates"])
        self._validate_metadata()
        self.schema = generator.plan_schema(self.library)
        stored = strict_loads((self.root / "schemas/plan.schema.json").read_bytes())
        if stored != self.schema:
            raise ContractError("generated plan schema is stale; run tools/check_design.py --generate")
        self.world_schema = strict_loads((self.root / "schemas/world-context.schema.json").read_bytes())
        for schema in [self.schema, self.world_schema, *[s["arguments"] for s in skill_list]]:
            Draft202012Validator.check_schema(schema)

    def _validate_metadata(self):
        """Check trusted policy consistency without using the offline interpreter."""
        if self.library.get("status") not in {"design_only", "implementation"}:
            raise ContractError("unknown catalog implementation status")
        resources = self.library["resources"]
        for skill_id, skill in self.skills.items():
            if skill["implementation_status"] not in {"planned", "implemented"}:
                raise ContractError("unknown skill implementation status")
            claims = skill["claims"]
            if len(claims) != len(set(claims)) or not set(claims) <= resources.keys():
                raise ContractError("duplicate or unknown resource claim in " + skill_id)
            if any(resources[claim].get("exclusive") is not True for claim in claims):
                raise ContractError("runtime claims require exclusive resources")
            if skill["execution"].get("interruptible") is not True:
                raise ContractError("skill must allow safety interruption")
            nominal, timeout = skill["timing"]["nominal_s"], skill["timing"]["timeout_s"]
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in (nominal, timeout)) or not 0 < nominal <= timeout:
                raise ContractError("invalid catalog skill timing")
            for schema_name in ("arguments", "outputs"):
                schema = skill[schema_name]
                Draft202012Validator.check_schema(schema)
                if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
                    raise ContractError("skill argument/output schema must be a closed object")
                if set(schema.get("required", [])) != set(schema.get("properties", {})):
                    raise ContractError("skill fields must all be explicitly required")
            for condition in skill["preconditions"] + skill["postconditions"]:
                if set(condition) != {"predicate", "bindings"} or condition["predicate"] not in self.predicates:
                    raise ContractError("unknown or malformed catalog predicate condition")
                argument_schema = self.predicates[condition["predicate"]]["arguments"]
                Draft202012Validator.check_schema(argument_schema)
                if set(condition["bindings"]) != set(argument_schema["properties"]):
                    raise ContractError("predicate condition has incomplete bindings")
                for name, binding in condition["bindings"].items():
                    if set(binding) == {"literal"}:
                        validate_schema(binding["literal"], argument_schema["properties"][name], "catalog binding")
                    elif set(binding) == {"arg_path"} and type(binding["arg_path"]) is list and binding["arg_path"]:
                        source_schema = skill["arguments"]
                        for segment in binding["arg_path"]:
                            if type(segment) is not str or segment not in source_schema.get("properties", {}):
                                raise ContractError("catalog predicate has an unbound argument path")
                            source_schema = source_schema["properties"][segment]
                    else:
                        raise ContractError("catalog condition is not a literal or typed argument path")
            failure_codes = [failure["code"] for failure in skill["failures"]]
            if len(failure_codes) != len(set(failure_codes)):
                raise ContractError("duplicate catalog failure code")
            if any(type(failure["local_retries"]) is not int or failure["local_retries"] < 0 for failure in skill["failures"]):
                raise ContractError("retry policy must be a bounded nonnegative integer")

    def planning_skills(self, available, *, context):
        """Narrow registered availability by existing catalog/profile compatibility."""
        return tuple(skill_id for skill_id, skill in self.skills.items()
                     if (available is None or skill_id in available)
                     and ("profile_id" not in skill["arguments"]["properties"]
                          or any(profile["safety_class"] == skill["safety_profile"]
                                 for profile in context["profiles"])))

    def _bind_plan_context(self, schema, context):
        # Generated argument branches reference the catalog. Never mutate them
        # while narrowing one request to its trusted context.
        schema = copy.deepcopy(schema)
        for name in ("task_id", "snapshot_id", "execution_epoch"):
            schema["properties"][name] = self._generator.literal_schema(context[name])
        for variant in schema["properties"]["nodes"]["items"]["anyOf"]:
            skill = self.skills[variant["properties"]["skill"]["const"]]
            args = variant["properties"]["args"]["properties"]
            if "profile_id" in args:
                args["profile_id"]["enum"] = [profile["profile_id"] for profile in context["profiles"]
                                              if profile["safety_class"] == skill["safety_profile"]]
        return schema

    def plan_schema(self, available=None, *, context=None):
        if context is not None:
            available = self.planning_skills(available, context=context)
            if not available:
                raise ContractError("No available skills have compatible profiles")
        schema = self._generator.plan_schema(self.library, available)
        return self._bind_plan_context(schema, context) if context is not None else schema

    def response_schema(self, available=None, *, context=None):
        if context is not None:
            available = self.planning_skills(available, context=context)
            if not available:
                raise ContractError("No available skills have compatible profiles")
        schema = self._generator.response_schema(self.library, available)
        if context is not None:
            outcome = schema["properties"]["result"]["anyOf"][0]["properties"]
            outcome["plan"] = self._bind_plan_context(outcome["plan"], context)
        return schema

    def validate_plan_json(self, plan, *, available=None):
        data = checked_copy(plan)
        validate_schema(data, self.plan_schema(available), "plan")
        return data

    def validate_predicate(self, predicate: str, args: Mapping[str, Any]) -> None:
        if predicate not in self.predicates:
            raise ContractError("unknown predicate: " + str(predicate))
        validate_schema(dict(args), self.predicates[predicate]["arguments"], predicate)

    def bind_conditions(self, skill_id: str, args: Mapping[str, Any], kind: str = "preconditions"):
        if skill_id not in self.skills or kind not in {"preconditions", "postconditions"}:
            raise ContractError("unknown skill or condition kind")
        validate_schema(dict(args), self.skills[skill_id]["arguments"], skill_id)
        result = []
        for condition in self.skills[skill_id][kind]:
            bound = {}
            for name, binding in condition["bindings"].items():
                if set(binding) == {"literal"}:
                    bound[name] = copy.deepcopy(binding["literal"])
                elif set(binding) == {"arg_path"}:
                    value = args
                    try:
                        for segment in binding["arg_path"]:
                            value = value[segment]
                    except (KeyError, TypeError) as exc:
                        raise ContractError("invalid catalog argument binding") from exc
                    bound[name] = copy.deepcopy(value)
                else:
                    raise ContractError("unrecognized catalog binding")
            self.validate_predicate(condition["predicate"], bound)
            result.append({"predicate": condition["predicate"], "args": bound})
        return result


class SkillRegistry:
    """Only concrete handlers with all capabilities enter the active registry.

    Simulation registration is explicit and cannot authorize hardware. Current
    catalog entries are planned, so hardware registration remains unavailable even
    if a caller accidentally passes commissioned=True.
    """

    def __init__(self, catalog: Catalog, handlers: Mapping[str, Any], capabilities,
                 *, mode: str = "simulation", commissioned: bool = False):
        if mode not in {"simulation", "hardware"}:
            raise ContractError("registry mode must be simulation or hardware")
        self.catalog = catalog
        self.mode = mode
        self.capabilities = frozenset(capabilities)
        self.handlers = MappingProxyType(dict(handlers))
        self.unavailable = {}
        available = []
        for skill_id, skill in catalog.skills.items():
            handler = handlers.get(skill_id)
            if skill_id not in CURRENT_SKILLS:
                reason = "deferred capability"
            elif not callable(handler) and not callable(getattr(handler, "execute", None)):
                reason = "no implemented handler"
            elif not set(skill["capabilities_required"]) <= self.capabilities:
                reason = "missing capabilities: " + ", ".join(sorted(set(skill["capabilities_required"]) - self.capabilities))
            elif mode == "hardware" and (not commissioned or skill["implementation_status"] != "implemented"):
                reason = "hardware capability is not implemented and commissioned"
            else:
                available.append(skill_id)
                continue
            self.unavailable[skill_id] = reason
        self.available_skills = tuple(available)

    def require(self, skill_id: str):
        if skill_id not in self.available_skills:
            raise ContractError(f"unavailable skill {skill_id}: {self.unavailable.get(skill_id, 'unknown skill')}")
        return self.handlers[skill_id]

    def claims(self, skill_id: str) -> frozenset[str]:
        self.require(skill_id)
        return frozenset(self.catalog.skills[skill_id]["claims"])
