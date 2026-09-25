"""The code gate: what a learned skill's source may contain, checked before anything runs.

This is one of three layers (the jail's operating-system isolation and its restricted builtins are the
others); it refuses the ways out of an interpreter that are easiest to write: imports beyond a few pure
modules, underscore names and attributes (the route to __class__, __globals__ and the rest), reflection
and code-making builtins, class definitions, and global state.
"""
from __future__ import annotations

import ast

#: Pure modules a skill may import: arithmetic and data handling, nothing that reaches the machine.
ALLOWED_IMPORTS = frozenset({"math", "statistics", "itertools", "collections", "functools", "random"})
#: Names a skill may not mention at all.
FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "input", "breakpoint", "help", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr", "type", "object", "memoryview", "bytearray", "super",
    "classmethod", "staticmethod", "property", "__import__", "__builtins__", "__loader__", "__spec__"})
MAX_SOURCE_BYTES = 64_000


class GateError(ValueError):
    """The source is refused; the message lists every reason, for the writer to fix."""


def check_source(source):
    """Refuse or accept a skill's source; returns its description (the run function's docstring)."""
    if not isinstance(source, str) or not source.strip():
        raise GateError("the skill is empty")
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise GateError(f"the skill is longer than {MAX_SOURCE_BYTES} bytes")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise GateError(f"syntax error at line {exc.lineno}: {exc.msg}") from exc
    problems = []

    def refuse(node, why):
        problems.append(f"line {getattr(node, 'lineno', '?')}: {why}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    refuse(node, f"import {alias.name} is not allowed (only {', '.join(sorted(ALLOWED_IMPORTS))})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                refuse(node, f"from {node.module} import is not allowed")
            elif any(alias.name == "*" or alias.name.startswith("_") for alias in node.names):
                refuse(node, "star and underscore imports are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            refuse(node, f"the attribute {node.attr} is not allowed")
        elif isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in FORBIDDEN_NAMES):
            refuse(node, f"the name {node.id} is not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.arg)):
            name = node.arg if isinstance(node, ast.arg) else node.name
            if name.startswith("_"):
                refuse(node, f"the name {name} is not allowed")
        elif isinstance(node, ast.ClassDef):
            refuse(node, "classes are not allowed; write functions")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            refuse(node, "global and nonlocal are not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            refuse(node, "async code is not allowed; robot calls are ordinary calls")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            refuse(node, "strings containing double underscores are not allowed")
    runs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if not runs:
        refuse(tree, "the skill must define run(robot, **args)")
    else:
        run = runs[0]
        if not run.args.args or run.args.args[0].arg != "robot":
            refuse(run, "run's first parameter must be robot")
        if not ast.get_docstring(run):
            refuse(run, "run needs a docstring saying what the skill does")
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)) and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            refuse(node, "only imports, constants and function definitions may be at the top level")
    if problems:
        raise GateError("; ".join(problems))
    return ast.get_docstring(runs[0]).strip()
