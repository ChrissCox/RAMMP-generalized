"""The primitive API a learned skill can call, and the host-side check of every call's arguments.

The table is the whole of what a skill can ask for. Arguments are checked here, in the host, against this
table: a skill cannot pass a speed, a force, a guard setting or anything the table does not name. What a
call then does is the host's binding (validated plans run by the executor under the guards).
"""
from __future__ import annotations

import math

ROLES = ("pregrasp", "grasp", "retract", "above")
DIRECTIONS = ("left", "right", "up", "down", "back", "closer")
UNITS = ("rad", "m")


class PrimitiveError(ValueError):
    """A call the host refuses; the skill receives it as RobotError and may handle it."""


def _text(limit):
    def check(value):
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise PrimitiveError(f"expected text of at most {limit} characters")
        return value.strip()
    return check


def _choice(options):
    def check(value):
        if value not in options:
            raise PrimitiveError(f"expected one of {', '.join(options)}")
        return value
    return check


def _number(low, high):
    def check(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise PrimitiveError(f"expected a number from {low} to {high}")
        return float(value)
    return check


def _optional(check):
    return lambda value: None if value is None else check(value)


OBJECT = _text(64)
#: name -> (argument checks in order, required argument names, kind). kind "motion" moves the arm or hand.
PRIMITIVES = {
    "find": ({"query": _text(120)}, ("query",), "perception"),
    "look": ({"direction": _choice(DIRECTIONS)}, ("direction",), "motion"),
    "move_to": ({"object": OBJECT, "role": _choice(ROLES)}, ("object", "role"), "motion"),
    "open_hand": ({"aperture_m": _number(0., .085)}, ("aperture_m",), "motion"),
    "grasp": ({"object": OBJECT}, ("object",), "motion"),
    "release": ({"object": OBJECT, "onto": _optional(OBJECT)}, ("object",), "motion"),
    "move_part": ({"object": OBJECT, "amount": _number(-3.2, 3.2), "unit": _choice(UNITS)}, ("object", "amount", "unit"), "motion"),
    "go_home": ({}, (), "motion"),
    "holding": ({}, (), "state"),
    "gripper_opening": ({}, (), "state"),
    "check": ({"condition": _text(64), "object": _optional(OBJECT), "value": _optional(_number(-10., 10.))}, ("condition",), "state"),
    "log": ({"text": _text(500)}, ("text",), "log"),
}


def validate_call(name, args):
    """The call's arguments, checked and normalised; PrimitiveError names what is wrong."""
    if name not in PRIMITIVES:
        raise PrimitiveError(f"there is no primitive {name!r}; the primitives are {', '.join(sorted(PRIMITIVES))}")
    checks, required, _ = PRIMITIVES[name]
    if not isinstance(args, dict):
        raise PrimitiveError("arguments must be named")
    unknown = sorted(set(args)-set(checks))
    if unknown:
        raise PrimitiveError(f"{name} takes no argument {', '.join(unknown)}")
    missing = [key for key in required if args.get(key) is None]
    if missing:
        raise PrimitiveError(f"{name} needs {', '.join(missing)}")
    checked = {}
    for key, check in checks.items():
        if key in args:
            try:
                checked[key] = check(args[key])
            except PrimitiveError as exc:
                raise PrimitiveError(f"{name}({key}=...): {exc}") from None
    return checked


def api_card():
    """The API as the skill writer reads it."""
    lines = []
    for name, (checks, required, kind) in PRIMITIVES.items():
        params = ", ".join(key if key in required else f"{key}=None" for key in checks)
        lines.append(f"robot.{name}({params})  [{kind}]")
    lines.append("robot.use(skill, **args)  [calls another library skill]")
    return "\n".join(lines)
