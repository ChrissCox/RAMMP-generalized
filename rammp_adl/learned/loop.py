"""The learning loop on the arm: write skills, gate them, rehearse them, run one, verify it, keep it, learn from failures.

From Voyager: a writer model gets the task, the primitive API, the scene (what find() would return), the
library's closest skills (with their source, to call with robot.use) and, from the second round, what went
wrong last time. From RSIAgent: each round the writer offers a few different approaches, all gated and
rehearsed side by side (broad exploration, done where it costs nothing: in rehearsal); only one passing
candidate runs on the arm (deep, sequential practice); verification is separate from the writer; and the
failures of a learning session are distilled into short general lessons shown to the writer on similar tasks.
The first skill that runs and is verified is kept as provisional; verified runs later promote it. Nothing here
moves the arm: rehearse and execute are the caller's hosts.
"""
from __future__ import annotations

import asyncio
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


def extract_skills(text):
    """[(name, source)] for every ```python block in the answer, in order; each named by its '# name:' line."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        raise WriterError("the answer had no ```python block")
    skills = []
    for block in blocks:
        source = block.strip()+"\n"
        named = re.search(r"^#\s*name:\s*([a-z][a-z0-9_]{1,63})\s*$", source, flags=re.M)
        skills.append((named.group(1) if named else "skill", source))
    return skills


def extract_skill(text):
    """(name, source) of the last ```python block."""
    return extract_skills(text)[-1]


def prompt(task, *, scene, examples, feedback, lessons=(), candidates=1):
    """The writer's input: persona, API, rules, scene, known skills, lessons, the task and last round's outcome."""
    parts = [PERSONA, "Robot API:\n"+api_card(), RULES,
             "Scene now (what robot.find returns for these words):\n"+json.dumps(scene, indent=1)[:4000]]
    if examples:
        shown = []
        for example in examples:
            shown.append(f"# skill {example['name']} ({'verified' if example.get('verified') else 'provisional'}): "
                         f"{example['description']}\n{example.get('source', '')}")
        parts.append("Skills you can call with robot.use(name, **args), or learn from:\n"+"\n\n".join(shown)[:8000])
    if lessons:
        parts.append("Lessons from earlier attempts on this robot:\n"+"\n".join(f"- {lesson}" for lesson in lessons))
    parts.append(f"Task: {task}")
    if feedback:
        parts.append("Your last attempt and what happened:\n"+feedback[:6000]+"\nFix the cause; do not repeat what failed.")
    if candidates > 1:
        parts.append(f"Give up to {candidates} different approaches, most promising first, each in its own ```python block "
                     "whose first line is '# name: <snake_case_name>' naming what the skill does in general (not this scene). "
                     "They are all rehearsed and one is run.")
    else:
        parts.append("Answer with one ```python block. Its first line is '# name: <snake_case_name>' naming what the skill does "
                     "in general (not this scene), then the function.")
    return "\n\n".join(parts)


def lessons_prompt(task, story):
    """Ask for the general lessons a learning session's failures teach."""
    failures = []
    for entry in story["rounds"]:
        for attempt in entry.get("attempts", [entry]):
            if attempt.get("error"):
                failures.append(f"- {attempt.get('name', '?')} ({attempt.get('stage')}): {str(attempt['error'])[:400]}")
    return (f"{PERSONA}\n\nTask: {task}\nOutcome: {story['status']}\nWhat failed along the way:\n" + "\n".join(failures[-12:]) +
            "\n\nWrite up to 3 short lessons, one per line starting with '- ', that would help write skills for other tasks "
            "on this robot. General facts about the robot, its API or the world; no code, nothing about this one scene.")


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
                      candidates=3, choose=None, distil=True, log=None):
    """Write, gate, rehearse, run and verify until a skill works or the rounds run out. Returns the story.

    rehearse(source) and execute(source) run a skill and return a SkillRun (the caller binds the hosts);
    execute None stops at rehearsal. verify(run) -> (True | False | None, detail): None is "cannot measure".
    choose(task, [(name, description)]) -> index picks which rehearsed candidate runs; default the writer's first.
    """
    say = log or (lambda message: None)
    known = []
    for hit in library.search(task, k=examples):
        try:
            loaded = library.load(hit["name"])
        except KeyError:
            continue
        known.append({**hit, "source": loaded["source"]})
    lessons = library.lessons(task)
    rounds, feedback, story = [], "", None
    for number in range(1, max_rounds+1):
        entry = {"round": number, "attempts": []}
        rounds.append(entry)
        try:
            answer = await writer.write(prompt(task, scene=scene, examples=known, feedback=feedback, lessons=lessons,
                                               candidates=candidates))
            offered = extract_skills(answer)[:max(1, candidates)]
        except WriterError as exc:
            entry.update(stage="write", error=str(exc))
            feedback = f"Your answer could not be used: {exc}"
            continue
        gated, notes = [], []
        for name, source in offered:
            attempt = {"name": name, "source": source}
            entry["attempts"].append(attempt)
            try:
                attempt["description"] = check_source(source)
                gated.append(attempt)
            except GateError as exc:
                attempt.update(stage="gate", error=str(exc))
                notes.append(f"```python\n{source}```\nThe code check refused it: {exc}")
        runs = await asyncio.gather(*(rehearse(attempt["source"]) for attempt in gated))
        passing = []
        for attempt, rehearsal in zip(gated, runs):
            attempt["rehearsal"] = rehearsal.status
            if rehearsal.status == "succeeded":
                passing.append(attempt)
            else:
                attempt.update(stage="rehearsal", error=rehearsal.error)
                notes.append(f"```python\n{attempt['source']}```\nIn rehearsal (nothing moved):\n{describe_run(rehearsal)}")
        say(f"learn round {number}: {len(offered)} offered, {len(gated)} past the gate, {len(passing)} rehearsed cleanly")
        if not passing:
            entry.update(stage="gate" if not gated else "rehearsal", error="; ".join(a.get("error", "")[:200] for a in entry["attempts"]))
            feedback = "\n\n".join(notes)
            continue
        pick = 0
        if choose is not None and len(passing) > 1:
            try:
                pick = int(await choose(task, [(a["name"], a["description"]) for a in passing]))
            except Exception:                                   # noqa: BLE001 - unsure or failed: the writer's order
                pick = 0
            pick = pick if 0 <= pick < len(passing) else 0
        chosen = passing[pick]
        chosen["chosen"] = True
        entry.update(name=chosen["name"], source=chosen["source"])
        if execute is None:
            entry["stage"] = "rehearsed"
            story = {"status": "rehearsed", "name": chosen["name"], "source": chosen["source"], "rounds": rounds}
            break
        run = await execute(chosen["source"])
        chosen["run"] = entry["run"] = run.status
        verified, detail = (None, "no verifier") if verify is None else await verify(run)
        entry.update(verified=verified, verification=detail)
        if run.status == "succeeded" and verified is not False:
            version = library.save(chosen["name"], chosen["source"], task=task)
            status = library.record_outcome(chosen["name"], version, verified=bool(verified),
                                            evidence={"task": task, "detail": detail[:200]})
            entry["stage"] = "kept"
            say(f"learn round {number}: {chosen['name']} v{version} {'verified' if verified else 'ran (unverified)'}; kept as {status}")
            story = {"status": "learned" if verified else "kept_unverified", "name": chosen["name"], "version": version,
                     "library_status": status, "rounds": rounds, "result": run.result}
            break
        chosen.update(stage="run", error=run.error or detail)
        entry.update(stage="run", error=run.error or detail)
        say(f"learn round {number}: {chosen['name']} ran ({run.status}) but was not verified: {(run.error or detail)[:160]}")
        feedback = (f"```python\n{chosen['source']}```\nOn the robot:\n{describe_run(run)}\nVerification: "
                    f"{'not met' if verified is False else 'unknown'}: {detail}")
    story = story or {"status": "failed", "rounds": rounds}
    if distil and any(a.get("error") for r in rounds for a in r.get("attempts", [r])):
        try:
            answer = await writer.write(lessons_prompt(task, story))
            learned = [line[2:].strip() for line in answer.splitlines() if line.strip().startswith("- ")][:3]
            if learned:
                library.add_lessons(task, learned, source=story.get("name", ""))
                story["lessons"] = learned
                say("lessons kept: "+" | ".join(learned))
        except Exception:                                       # noqa: BLE001 - lessons are a bonus, never the outcome
            pass
    return story


def curriculum_prompt(scene, library, *, count=3):
    """Ask for the next practice tasks: what the scene allows, aimed at what the library cannot do yet."""
    skills = []
    for name in library.names():
        try:
            loaded = library.load(name)
            skills.append(f"- {name} ({loaded['status']}): {loaded['description']}")
        except KeyError:
            skills.append(f"- {name} (retired)")
    lessons = library.lessons(" ".join(str(e.get("label", "")) for e in scene), k=8)
    return (f"{PERSONA}\n\nYou choose what the robot practises next, to become able to help with everyday tasks.\n"
            f"Robot API:\n{api_card()}\n\nScene now:\n{json.dumps(scene, indent=1)[:4000]}\n\n"
            f"Skills it has:\n" + ("\n".join(skills) or "- none yet") + "\n\n"
            + ("Lessons so far:\n" + "\n".join(f"- {lesson}" for lesson in lessons) + "\n\n" if lessons else "")
            + f"Propose {count} tasks for the robot to practise next in this scene: each achievable with the API, each teaching "
            "something the skills above do not cover yet, small first. Prefer tasks that leave the scene as it was, or that "
            "a following task undoes. Answer only with a JSON list of objects with the keys task (the words an operator "
            "would type), why, and undo (how the scene is put back, or \"none needed\").")


async def propose_tasks(writer, library, scene, *, count=3):
    """The curriculum's next practice tasks, for an operator to choose from: [{task, why, undo}]."""
    answer = await writer.write(curriculum_prompt(scene, library, count=count))
    found = re.search(r"\[.*\]", answer, flags=re.S)
    if not found:
        raise WriterError("the curriculum answer had no JSON list")
    try:
        proposals = json.loads(found.group(0))
    except ValueError as exc:
        raise WriterError(f"the curriculum answer was not JSON: {exc}") from exc
    return [{"task": str(p.get("task", ""))[:200], "why": str(p.get("why", ""))[:300], "undo": str(p.get("undo", ""))[:200]}
            for p in proposals if isinstance(p, dict) and p.get("task")][:count]
