"""The program that runs inside the jail (passed to python -c; it imports nothing from this project).

It reads one JSON line with the skill's source, arguments and the primitive names, runs the skill with a
bare builtins table, and turns every robot call into a JSON line to the host and back. print becomes a log
line. It writes nothing else to stdout.
"""
import json
import sys

_OUT, _IN = sys.stdout, sys.stdin
_ALLOWED_IMPORTS = ("math", "statistics", "itertools", "collections", "functools", "random")


def _send(message):
    _OUT.write(json.dumps(message, default=str)+"\n")
    _OUT.flush()


def _receive():
    line = _IN.readline()
    if not line:
        raise SystemExit(3)
    return json.loads(line)


class RobotError(Exception):
    """A primitive call the host refused or that failed; the message says why."""


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in _ALLOWED_IMPORTS:
        raise ImportError(f"import {name} is not allowed")
    return __import__(name, globals, locals, fromlist, level)


_SAFE = ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list",
         "map", "max", "min", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "divmod",
         "pow", "frozenset", "Exception", "ValueError", "RuntimeError", "KeyError", "IndexError", "TypeError",
         "ZeroDivisionError", "ArithmeticError", "StopIteration", "True", "False", "None")


def _builtins():
    import builtins
    table = {name: getattr(builtins, name) for name in _SAFE if hasattr(builtins, name)}
    table["__import__"] = _import
    table["print"] = lambda *parts, **_: _send({"log": " ".join(str(p) for p in parts)[:500]})
    table["RobotError"] = RobotError
    return table


def _load(source, name):
    namespace = {"__builtins__": _builtins(), "__name__": "skill_"+name, "RobotError": RobotError}
    exec(compile(source, f"<skill {name}>", "exec"), namespace)
    return namespace["run"]


class _Robot:
    def __init__(self, api):
        self._api = api

    def __getattr__(self, name):
        if name == "use":
            def use(skill, **args):
                _send({"use": skill, "args": args})
                reply = _receive()
                if "error" in reply:
                    raise RobotError(reply["error"])
                return _load(reply["source"], skill)(self, **args)
            return use
        if name not in self._api:
            raise AttributeError(f"robot has no primitive {name}")
        params = self._api[name]

        def call(*positional, **named):
            if len(positional) > len(params):
                raise TypeError(f"{name} takes at most {len(params)} positional arguments")
            args = dict(zip(params, positional))
            clash = set(args) & set(named)
            if clash:
                raise TypeError(f"{name} got {', '.join(sorted(clash))} twice")
            args.update(named)
            _send({"call": name, "args": args})
            reply = _receive()
            if "error" in reply:
                raise RobotError(reply["error"])
            return reply.get("result")
        return call


def _main():
    start = _receive()
    try:
        run = _load(start["source"], start["name"])
        result = run(_Robot(start["api"]), **start["args"])
        _send({"done": result})
    except RobotError as exc:
        _send({"failed": str(exc)[:1000], "kind": "RobotError"})
    except BaseException as exc:                             # noqa: BLE001 - reported to the host, which decides
        import traceback
        _send({"failed": f"{type(exc).__name__}: {exc}"[:1000], "kind": "exception",
               "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-4:])[:2000]})


_main()
