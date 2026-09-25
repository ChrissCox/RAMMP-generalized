"""Run a learned skill in the jail and serve its primitive calls.

The skill runs in /usr/bin/python3 under bubblewrap: new user, network, process, IPC, UTS and mount
namespaces, only /usr read-only, a private /tmp, no environment, and CPU, memory, file-size and open-file
limits. Its one channel is a pipe of JSON lines. The host checks every call against the primitive table,
counts it against the budget, and hands it to the binding (host.call), which runs it as validated,
guarded execution. The run ends when the skill returns or fails, or on the budget or the clock.
"""
from __future__ import annotations

import asyncio
import json
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

from .api import PRIMITIVES, PrimitiveError, validate_call
from .gate import GateError, check_source

HARNESS = (Path(__file__).parent/"harness.py").read_text()
JAIL_PYTHON = "/usr/bin/python3"


class JailError(RuntimeError):
    """The jail could not be started or broke its protocol."""


class SkillAbort(Exception):
    """Raised by a binding to end the run at once (a safety fault, a cancel): the skill gets no reply."""


@dataclass
class SkillRun:
    name: str
    status: str                      # succeeded, failed, refused, budget, timeout, aborted, crashed
    result: object = None
    error: str = ""
    calls: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    duration_s: float = 0.
    stderr: str = ""

    def to_dict(self):
        return {key: getattr(self, key) for key in ("name", "status", "result", "error", "calls", "logs", "duration_s", "stderr")}


def jail_command():
    return ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp",
            JAIL_PYTHON, "-I", "-S", "-B", "-c", HARNESS]


def _limits(cpu_s, memory_mb):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


async def run_skill(source, args, host, *, name="skill", library=None, max_calls=60, timeout_s=300., cpu_s=60,
                    memory_mb=768, max_uses=20):
    """Gate the source, then run it in the jail against host.call(name, args); returns a SkillRun."""
    try:
        check_source(source)
    except GateError as exc:
        return SkillRun(name, "refused", error=str(exc))
    return await _run_in_jail(source, args, host, name=name, library=library, max_calls=max_calls, timeout_s=timeout_s,
                              cpu_s=cpu_s, memory_mb=memory_mb, max_uses=max_uses)


async def _run_in_jail(source, args, host, *, name, library, max_calls, timeout_s, cpu_s, memory_mb, max_uses):
    """The jail itself, for source that has already been gated."""
    try:
        json.dumps(args)
    except (TypeError, ValueError) as exc:
        raise JailError(f"skill arguments must be plain data: {exc}") from exc
    began = time.monotonic()
    run = SkillRun(name, "crashed")
    process = await asyncio.create_subprocess_exec(
        *jail_command(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        preexec_fn=_limits(cpu_s, memory_mb), env={}, limit=1 << 20)
    stderr_task = asyncio.ensure_future(process.stderr.read(64_000))

    async def send(message):
        process.stdin.write((json.dumps(message)+"\n").encode())
        await process.stdin.drain()
    uses = 0
    try:
        await send({"name": name, "source": source, "args": args, "api": {n: list(spec[0]) for n, spec in PRIMITIVES.items()}})
        while True:
            left = timeout_s-(time.monotonic()-began)
            if left <= 0:
                run.status, run.error = "timeout", f"the skill ran past {timeout_s:.0f} s"
                break
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=left)
            except asyncio.TimeoutError:
                run.status, run.error = "timeout", f"the skill ran past {timeout_s:.0f} s"
                break
            if not line:
                run.status, run.error = "crashed", "the skill's process ended without a result (CPU or memory limit?)"
                break
            try:
                message = json.loads(line)
            except ValueError:
                run.status, run.error = "crashed", "the jail broke its protocol"
                break
            if "log" in message:
                run.logs.append(str(message["log"])[:500])
            elif "call" in message:
                call = {"primitive": message["call"], "args": message.get("args"), "at_s": round(time.monotonic()-began, 3)}
                run.calls.append(call)
                if sum(1 for c in run.calls if c.get("primitive") not in (None, "log")) > max_calls:
                    run.status, run.error = "budget", f"more than {max_calls} primitive calls"
                    break
                try:
                    checked = validate_call(message["call"], message.get("args") or {})
                except PrimitiveError as exc:
                    call["refused"] = str(exc)
                    await send({"error": str(exc)})
                    continue
                if message["call"] == "log":
                    run.logs.append(checked["text"])
                    await send({"result": None})
                    continue
                try:
                    result = await host.call(message["call"], checked)
                except SkillAbort as exc:
                    run.status, run.error = "aborted", str(exc)
                    break
                except Exception as exc:                    # noqa: BLE001 - the skill decides what to do about it
                    call["error"] = str(exc)[:300]
                    await send({"error": f"{message['call']} failed: {exc}"[:1000]})
                    continue
                call["result"] = result
                await send({"result": result})
            elif "use" in message:
                uses += 1
                if library is None:
                    await send({"error": "there is no skill library"})
                elif uses > max_uses:
                    await send({"error": f"more than {max_uses} uses of other skills"})
                else:
                    try:
                        used = library.load(str(message["use"]))
                        check_source(used["source"])
                        run.calls.append({"use": message["use"], "version": used["version"], "at_s": round(time.monotonic()-began, 3)})
                        await send({"source": used["source"]})
                    except (KeyError, GateError) as exc:
                        await send({"error": f"cannot use {message['use']}: {exc}"})
            elif "done" in message:
                run.status, run.result = "succeeded", message["done"]
                break
            elif "failed" in message:
                run.status, run.error = "failed", message["failed"]+(("\n"+message["traceback"]) if message.get("traceback") else "")
                break
            else:
                run.status, run.error = "crashed", "the jail broke its protocol"
                break
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
        try:
            run.stderr = (await asyncio.wait_for(stderr_task, 2.)).decode(errors="replace")[-2000:]
        except asyncio.TimeoutError:
            stderr_task.cancel()
        transport = getattr(process, "_transport", None)   # closed here, not by a collector after the loop is gone
        if transport is not None:
            transport.close()
        run.duration_s = round(time.monotonic()-began, 3)
    return run
