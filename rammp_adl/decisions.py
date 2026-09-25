"""Fast typed decisions from TypeSafe's Jev, beside Astra, never instead of what is measured.

Jev answers typed questions about text (pick one option, with calibrated
probabilities and a confidence); it sees no images and writes no text. It is
used where the runtime can state the whole question as text and list every
answer itself:

- which goal a task names, among the goals this scene can bind (goal_options);
- after a failed step, whether a new plan could help or the task should stop.

Every option is built locally, so a pick can only be something the runtime
already allows, and a pick is still validated like Astra's answer would be.
Below the configured confidence, or on any transport failure, the caller falls
back to Astra, so switching Jev on can make a task cheaper and faster but never
changes what may happen. Nothing here moves the arm.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

NONE = "none_of_these"
CONFIG = "config/decisions.json"


@dataclass(frozen=True)
class Decision:
    choice: str | None
    confidence: float
    probabilities: dict
    latency_s: float
    model: str = ""
    error: str = ""

    def accepted(self, threshold):
        return self.choice is not None and self.choice != NONE and self.confidence >= threshold


def load_config(root):
    path = Path(root)/CONFIG
    config = json.loads(path.read_text())
    for key in ("endpoint", "model", "api_key_env", "timeout_s", "goal_min_confidence", "stop_min_confidence"):
        if key not in config:
            raise ValueError(f"{CONFIG} lacks {key}")
    if not (0. < float(config["timeout_s"]) <= 10.):
        raise ValueError("a Jev decision must be quick: timeout_s in (0, 10]")
    for key in ("goal_min_confidence", "stop_min_confidence"):
        if not .5 <= float(config[key]) <= 1.:
            raise ValueError(f"{key} must be in [0.5, 1]")
    return config


class JevDecider:
    """One Choice question per call to POST /v1/systemone; failures come back as a Decision with no choice."""

    def __init__(self, config, *, environ=None, transport=None, log=None):
        self.config = dict(config)
        environ = os.environ if environ is None else environ
        self.api_key = environ.get(self.config["api_key_env"], "")
        self.transport = transport                  # async (url, headers, body, timeout_s) -> dict; tests replace it
        self.log = log or (lambda message: None)
        self.calls, self.input_tokens = 0, 0

    @property
    def available(self):
        return bool(self.api_key) or self.transport is not None

    async def _post(self, body):
        if self.transport is not None:
            return await self.transport(self.config["endpoint"], {}, body, float(self.config["timeout_s"]))
        return await asyncio.to_thread(self._post_blocking, body)

    def _post_blocking(self, body):
        """The standard library's HTTP, off the event loop: the node runs without user site-packages."""
        import urllib.request
        request = urllib.request.Request(self.config["endpoint"], data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=float(self.config["timeout_s"])) as response:
            return json.loads(response.read().decode())

    async def choose(self, state, *, question, instructions, options):
        """Ask one Choice; options maps each key to what choosing it means."""
        if not options or len(options) > 255:
            return Decision(None, 0., {}, 0., error="a choice needs between 1 and 255 options")
        body = {"model": self.config["model"], "state": state,
                "questions": {question: {"type": "choice", "instructions": instructions, "criteria": dict(options)}}}
        began = time.monotonic()
        try:
            reply = await self._post(body)
            answer = reply["answers"][question]
            choice, confidence = answer["choice"], float(answer["confidence"])
            probabilities = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
            if choice not in options or not math.isfinite(confidence):
                raise ValueError(f"answer {choice!r} is not one of the options")
        except Exception as exc:                    # noqa: BLE001 - any failure means: ask Astra as before
            decision = Decision(None, 0., {}, time.monotonic()-began, error=f"{type(exc).__name__}: {str(exc)[:160]}")
            self.log(f"jev {question}: no decision ({decision.error}); falling back")
            return decision
        self.calls += 1
        self.input_tokens += int((reply.get("usage") or {}).get("input_tokens") or 0)
        return Decision(choice, confidence, probabilities, time.monotonic()-began, str(reply.get("model", "")))


def _label(context, entity_id):
    for entity in context["entities"]:
        if entity["entity_id"] == entity_id:
            return entity.get("label") or entity_id
    return entity_id


def goal_options(context, candidates):
    """Every goal this scene can bind, keyed, with what each means in words: {key: (description, goal)}.

    Targets come from the scene: a hinge or slide is opened to its installed
    range, or halfway; nothing numeric is left for the model to choose freely.
    """
    options = {}
    entities = [e for e in context["entities"] if e.get("entity_id") and e["entity_id"] != "robot"]
    if "constraint_goal_verified" in candidates:
        for constraint in context.get("constraints", []):
            label, unit = _label(context, constraint["entity_id"]), constraint["unit"]
            verb = "open" if constraint["kind"] == "revolute" else "slide out"
            for share, how in ((1., "fully"), (.5, "halfway")):
                value = round(float(constraint["maximum"])*share, 3)
                options[f"move_{constraint['constraint_id']}_{how}"] = (
                    f"{verb} {how}: move the part the {label} belongs to by {value} {unit}, as far as {'it goes' if share == 1. else 'half of that'}",
                    {"predicate": "constraint_goal_verified",
                     "args": {"constraint_id": constraint["constraint_id"], "target_value": value, "target_unit": unit}})
    for entity in entities:
        roles = set(entity.get("pose_roles") or ())
        if "holding" in candidates and "grasp" in roles:
            options[f"hold_{entity['entity_id']}"] = (f"pick up or grab the {entity['label']} and keep holding it",
                                                      {"predicate": "holding", "args": {"entity_id": entity["entity_id"]}})
        if "at_pose" in candidates:
            for role in sorted(roles & {"pregrasp", "grasp", "viewpoint", "retract"}):
                options[f"reach_{entity['entity_id']}_{role}"] = (
                    f"move the hand to the {entity['label']}'s {role} pose without grasping it",
                    {"predicate": "at_pose", "args": {"entity_id": entity["entity_id"], "pose_role": role}})
        if "observation_valid" in candidates:
            options[f"look_{entity['entity_id']}"] = (f"look at the {entity['label']} to find where it is",
                                                      {"predicate": "observation_valid",
                                                       "args": {"entity_id": entity["entity_id"], "purpose": "pose"}})
    if "aperture_reached" in candidates:
        options["open_gripper"] = ("open the gripper fully", {"predicate": "aperture_reached", "args": {"aperture_m": .085}})
        options["close_gripper"] = ("close the gripper", {"predicate": "aperture_reached", "args": {"aperture_m": 0.}})
    options[NONE] = ("the task asks for something none of the other options describe", None)
    return options


def task_state(task_text, context):
    """The text Jev reads: the request and the parts in view, nothing it cannot use."""
    parts = [f"{e['label']} ({e['entity_id']})" for e in context["entities"] if e.get("entity_id") not in (None, "robot")]
    return f"Request to a robot arm: {task_text.strip()}\nParts in view: {', '.join(parts) or 'none'}"


async def decide_goal(decider, task_text, context, candidates, *, threshold):
    """The goal Jev picks, or None to ask Astra; returns (goal or None, Decision)."""
    options = goal_options(context, candidates)
    decision = await decider.choose(task_state(task_text, context), question="goal",
                                    instructions="Which outcome does the request ask the robot arm to bring about?",
                                    options={key: description for key, (description, _) in options.items()})
    if not decision.accepted(threshold):
        return None, decision
    return options[decision.choice][1], decision


STOP_OPTIONS = {
    "replan": "Another plan with the same skills could plausibly succeed: the failure looks transient, like a "
              "perception, placement or grasp error, a timeout, or something a different sequence of steps would avoid.",
    "stop": "No new plan can fix it: a skill or capability the robot lacks is needed, the part is blocked, stuck or "
            "at its limit, or the same step already failed the same way.",
}


async def decide_after_failure(decider, *, task_text, skill, failure_code, detail, attempt, threshold):
    """True when Jev is confident a new plan cannot help; returns (stop, Decision)."""
    state = (f"Request to a robot arm: {task_text.strip()}\nAttempt {attempt+1} failed at the step {skill}: "
             f"{failure_code}. {detail[:400]}")
    decision = await decider.choose(state, question="after_failure",
                                    instructions="Should the robot ask its planner for a new plan, or stop and report?",
                                    options=STOP_OPTIONS)
    return decision.choice == "stop" and decision.confidence >= threshold, decision
