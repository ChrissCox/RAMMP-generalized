"""Voyager's loop on the arm: write a skill, gate it, rehearse it, run it, verify it, keep it.

A writer model gets the task, the primitive API, the scene (what find() would return), the library's closest
skills (with their source, to call with robot.use) and, from the second round, what went wrong last time: the
gate's refusal, the rehearsal's refused step, the run's error and calls, the verifier's judgement. It answers
with one Python function. The first skill that runs and is verified is kept in the library as provisional;
verified runs later promote it. Nothing here moves the arm: rehearse and execute are the caller's hosts.
"""
from __future__ import annotations

import json
import re

from .api import api_card
from .gate import GateError, check_source

RULES = """Rules for the code (it is checked before it runs, and runs in a sandbox):
- Define run(robot, **args) with a docstring saying what the skill does. Keyword arguments need defaults.
- Only these imports: math, statistics, itertools, collections, functools, random.
- No classes, no global/nonlocal, no names or attributes starting with an underscore, no eval/exec/open/getattr.
- The robot is changed only through robot.<call>(...). A call that is refused or fails raises RobotError(message):
  catch it when you can do something else, else let it end the skill.
- Objects are named by the ids find() returns. move_to roles: pregrasp (just short of it), grasp, retract, above.
- move_part(object, amount, unit) moves an articulated part to that position from closed (0), within its range.
- Return a small plain value (a dict of what happened). print() writes to the log."""

PERSONA = ("You write skills for a robot arm with a two-finger gripper, as short Python functions over a fixed robot API. "
           "The arm is real: plan each step, check what you can measure, and prefer the simplest sequence that works.")


class WriterError(RuntimeError):
    """The writer gave nothing usable."""


def extract_skill(text):
    """(name, source) from the writer's answer: the last ```python block, named by a '# name:' line or run's first line."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        raise WriterError("the answer had no ```python block")
    source = blocks[-1].strip()+"\n"
    named = re.search(r"^#\s*name:\s*([a-z][a-z0-9_]{1,63})\s*$", source, flags=re.M)
    name = named.group(1) if named else "skill"
    return name, source


def prompt(task, *, scene, examples, feedback):
    """The writer's input: persona, API, rules, scene, known skills, the task and last round's outcome."""
    parts = [PERSONA, "Robot API:\n"+api_card(), RULES,
             "Scene now (what robot.find returns for these words):\n"+json.dumps(scene, indent=1)[:4000]]
    if examples:
        shown = []
        for example in examples:
            shown.append(f"# skill {example['name']} ({'verified' if example.get('verified') else 'provisional'}): "
                         f"{example['description']}\n{example.get('source', '')}")
        parts.append("Skills you can call with robot.use(name, **args), or learn from:\n"+"\n\n".join(shown)[:8000])
    parts.append(f"Task: {task}")
    if feedback:
        parts.append("Your last attempt and what happened:\n"+feedback[:6000]+"\nFix the cause; do not repeat what failed.")
    parts.append("Answer with one ```python block. Its first line is '# name: <snake_case_name>' naming what the skill does "
                 "in general (not this scene), then the function.")
    return "\n\n".join(parts)


class ModelWriter:
    """The writer as a Responses API model (config/learning.json); transport injectable for tests."""

    def __init__(self, config, *, transport=None):
        self.config = dict(config["writer"])
        if transport is None:
            import os
            from ..reasoning import OpenAIResponsesTransport
            key = os.environ.get(self.config["api_key_env"])
            if not key:
                raise WriterError(f"{self.config['api_key_env']} is not set; the skill writer has no model")
            transport = OpenAIResponsesTransport(timeout_s=float(self.config["request_timeout_s"]), api_key=key)
        self.transport, self.requests = transport, 0

    async def write(self, text):
        self.requests += 1
        response = await self.transport.create(model=self.config["model"], input=text, store=bool(self.config.get("store")),
                                               reasoning={"effort": self.config.get("reasoning_effort", "medium")},
                                               max_output_tokens=int(self.config["max_output_tokens"]))
        output = getattr(response, "output_text", None)
        if output is None and isinstance(response, dict):
            output = "".join(part.get("text", "") for item in response.get("output", ()) for part in item.get("content", ())
                             if isinstance(part, dict))
        if not output:
            raise WriterError("the writer returned no text")
        return output


def describe_run(run):
    """A run as the writer reads it: status, error, and each call with its answer or refusal."""
    lines = [f"status: {run.status}"+(f"; error: {run.error[:1500]}" if run.error else "")]
    for call in run.calls[-40:]:
        if "use" in call:
            lines.append(f"  used skill {call['use']} v{call.get('version')}")
            continue
        answer = (f"REFUSED {call['refused']}" if "refused" in call else f"FAILED {call['error']}" if "error" in call
                  else json.dumps(call.get("result"))[:200])
        lines.append(f"  robot.{call['primitive']}({json.dumps(call.get('args'))[:160]}) -> {answer}")
    lines += [f"  log: {line}" for line in run.logs[-10:]]
    return "\n".join(lines)


async def learn_skill(task, *, writer, library, scene, rehearse, execute=None, verify=None, max_rounds=4, examples=4,
                      log=None):
    """Write, gate, rehearse, run and verify until a skill works or the rounds run out. Returns the story.

    rehearse(source) and execute(source) run a skill and return a SkillRun (the caller binds the hosts);
    execute None stops at rehearsal. verify(run) -> (True | False | None, detail): None is "cannot measure".
    """
    say = log or (lambda message: None)
    known = []
    for hit in library.search(task, k=examples):
        try:
            loaded = library.load(hit["name"])
        except KeyError:
            continue
        known.append({**hit, "source": loaded["source"]})
    rounds, feedback = [], ""
    for number in range(1, max_rounds+1):
        entry = {"round": number}
        rounds.append(entry)
        try:
            answer = await writer.write(prompt(task, scene=scene, examples=known, feedback=feedback))
            name, source = extract_skill(answer)
        except WriterError as exc:
            entry.update(stage="write", error=str(exc))
            feedback = f"Your answer could not be used: {exc}"
            continue
        entry.update(name=name, source=source)
        try:
            check_source(source)
        except GateError as exc:
            entry.update(stage="gate", error=str(exc))
            say(f"learn round {number}: {name} refused by the gate: {str(exc)[:160]}")
            feedback = f"```python\n{source}```\nThe code check refused it: {exc}"
            continue
        rehearsal = await rehearse(source)
        entry["rehearsal"] = rehearsal.status
        if rehearsal.status != "succeeded":
            entry.update(stage="rehearsal", error=rehearsal.error)
            say(f"learn round {number}: {name} failed its rehearsal: {rehearsal.error[:160]}")
            feedback = f"```python\n{source}```\nIn rehearsal (nothing moved):\n{describe_run(rehearsal)}"
            continue
        if execute is None:
            entry["stage"] = "rehearsed"
            return {"status": "rehearsed", "name": name, "source": source, "rounds": rounds}
        run = await execute(source)
        entry["run"] = run.status
        verified, detail = (None, "no verifier") if verify is None else await verify(run)
        entry.update(verified=verified, verification=detail)
        if run.status == "succeeded" and verified is not False:
            version = library.save(name, source, task=task)
            status = library.record_outcome(name, version, verified=bool(verified), evidence={"task": task, "detail": detail[:200]})
            entry["stage"] = "kept"
            say(f"learn round {number}: {name} v{version} {'verified' if verified else 'ran (unverified)'}; kept as {status}")
            return {"status": "learned" if verified else "kept_unverified", "name": name, "version": version,
                    "library_status": status, "rounds": rounds, "result": run.result}
        entry.update(stage="run", error=run.error or detail)
        say(f"learn round {number}: {name} ran ({run.status}) but was not verified: {(run.error or detail)[:160]}")
        feedback = (f"```python\n{source}```\nOn the robot:\n{describe_run(run)}\nVerification: "
                    f"{'not met' if verified is False else 'unknown'}: {detail}")
    return {"status": "failed", "rounds": rounds}
