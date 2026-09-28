"""Fast typed decisions from TypeSafe's Jev, beside Astra, never instead of what is measured.

Jev answers typed questions about text (pick one option, with calibrated
probabilities and a confidence); it sees no images and writes no text. It is
used where the runtime can state the whole question as text and list every
answer itself:

- which goal a task names, among the goals this scene can bind (goal_options);
- how a handled part moves: hinge or slide, which edge, pull or push, how far,
  from its label and the door the depth measured (JevConstraintReasoner);
- the choices a plan template leaves open, such as letting go when done;
- after a failed step, which local recovery to run, or to ask Astra, or to stop.

Several questions about the same state go in one call and are answered
independently and in parallel: that is what Jev is fast at.

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

    async def ask(self, state, questions):
        """Several questions about one state in one call: {id: ("choice", instructions, {key: meaning})
        or ("noul", instructions)}. Returns {id: Decision}; a noul's choice is "yes" or "no" and its
        confidence the probability of that answer. Any failure answers every question with no choice."""
        body = {"model": self.config["model"], "state": state, "questions": {}}
        for key, (kind, instructions, *rest) in questions.items():
            if kind == "choice":
                options = dict(rest[0])
                if not options or len(options) > 255:
                    return {k: Decision(None, 0., {}, 0., error="a choice needs between 1 and 255 options") for k in questions}
                body["questions"][key] = {"type": "choice", "instructions": instructions, "criteria": options}
            elif kind == "noul":
                body["questions"][key] = {"type": "noul", "instructions": instructions}
            else:
                raise ValueError(f"unsupported question type {kind}")
        began = time.monotonic()
        try:
            reply = await self._post(body)
            answers = {}
            for key, (kind, *_) in questions.items():
                answer = reply["answers"][key]
                if kind == "choice":
                    choice, confidence = answer["choice"], float(answer["confidence"])
                    if choice not in body["questions"][key]["criteria"] or not math.isfinite(confidence):
                        raise ValueError(f"{key}: answer {choice!r} is not one of the options")
                    probabilities = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
                else:
                    yes = float(answer["noul"])
                    if not 0. <= yes <= 1.:
                        raise ValueError(f"{key}: {yes} is not a probability")
                    choice, confidence, probabilities = ("yes" if yes >= .5 else "no"), max(yes, 1.-yes), {"yes": yes}
                answers[key] = Decision(choice, confidence, probabilities, time.monotonic()-began, str(reply.get("model", "")))
        except Exception as exc:                    # noqa: BLE001 - any failure means: decide as without Jev
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
            self.log(f"jev {', '.join(questions)}: no decision ({error}); falling back")
            return {key: Decision(None, 0., {}, time.monotonic()-began, error=error) for key in questions}
        self.calls += 1
        self.input_tokens += int((reply.get("usage") or {}).get("input_tokens") or 0)
        return answers

    async def choose(self, state, *, question, instructions, options):
        """Ask one Choice; options maps each key to what choosing it means."""
        return (await self.ask(state, {question: ("choice", instructions, options)}))[question]


def _label(context, entity_id):
    for entity in context["entities"]:
        if entity["entity_id"] == entity_id:
            return entity.get("label") or entity_id
    return entity_id


def goal_options(context, candidates):
    """Every goal this scene can bind, keyed, with what each means in words: {key: (description, goal)}.

    Targets come from the scene: a hinge or slide is opened to its installed
    range, or halfway, or brought back to where it rests shut; nothing numeric
    is left for the model to choose freely.
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
            shut = round(float(constraint["minimum"]), 3)
            options[f"move_{constraint['constraint_id']}_shut"] = (
                f"{'close' if constraint['kind'] == 'revolute' else 'slide in'}: bring the part the {label} belongs to back "
                f"to {shut} {unit}, where it rests shut, from wherever it is now",
                {"predicate": "constraint_goal_verified",
                 "args": {"constraint_id": constraint["constraint_id"], "target_value": shut, "target_unit": unit}})
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


REVOLUTE_RANGES = {"quarter_turn": ("opens about a quarter turn, like a cabinet or wardrobe door", 1.57),
                   "wide_swing": ("swings wider than a quarter turn, like a fridge or room door", 1.92),
                   "part_way": ("tilts or drops open only part way, like an oven door, flap or lid", .79)}
PRISMATIC_RANGES = {"short": ("slides out a short way, like a shallow drawer or tray", .15),
                    "medium": ("slides out a medium way, like a kitchen drawer", .30),
                    "long": ("slides out a long way, like a deep drawer or a sliding door", .45)}


def constraint_state(label, geometry, door):
    """What Jev reads about a handled part: its name, which way its bar runs, and the door the depth measured."""
    major = [float(v) for v in (geometry or {}).get("major_axis", (0., 0., 1.))]
    bar = "vertical" if abs(major[2]) > .7 else "horizontal" if abs(major[2]) < .3 else "diagonal"
    lines = [f"A robot arm faces a {label} and holds out its gripper toward the {bar} handle on it."]
    if door:
        offsets = door["handle_offsets_m"]
        lines.append(f"Measured by depth: the {label} is {door['width_m']:.2f} m wide and {door.get('height_m') or 0.:.2f} m tall; "
                     f"the handle is {offsets['left']:.2f} m from its left edge, {offsets['right']:.2f} m from its right edge, "
                     f"{offsets['bottom']:.2f} m from its bottom and {offsets['top']:.2f} m from its top (as seen by the robot).")
    return "\n".join(lines)


def measured_hinge(geometry, door):
    """The hinge edge the depth implies: the door edge farther from the handle, across the handle's bar."""
    if not door:
        return None
    offsets = door["handle_offsets_m"]
    major = [float(v) for v in (geometry or {}).get("major_axis", (0., 0., 1.))]
    if abs(major[2]) > .7:                              # a vertical bar: the hinge is on a side edge
        return "left" if offsets["left"] > offsets["right"] else "right"
    if abs(major[2]) < .3:                              # a horizontal bar: on the top or bottom edge
        return "bottom" if offsets["bottom"] > offsets["top"] else "top"
    return None


async def decide_constraint(decider, *, label, geometry, door, threshold, push_threshold=.9):
    """A constraint proposal in Astra's format from one call, or None when Jev is unsure of what it is best at.

    Each part of the answer comes from whoever is good at it. Jev: what kind of part this is (swings or
    slides) and how far it opens, judged from its name. The depth: which edge the hinge is on (the edge
    farther from the handle, across its bar) and the hinge-to-handle width. A local prior: a handled door is
    pulled unless Jev is sure it is pushed, because pulling a push door only stalls, while pushing a pull door
    forces it. A slide's direction has no safe default, so a slide needs Jev sure of that too.
    """
    state = constraint_state(label, geometry, door)
    questions = {
        "kind": ("choice", "How does this part move when opened?",
                 {"revolute": "it swings open on hinges", "prismatic": "it slides straight out or sideways"}),
        "opening": ("choice", "Which way does it open?",
                    {"pull": "toward the robot", "push": "away from the robot, into the furniture",
                     "slide_left": "sliding to the robot's left", "slide_right": "sliding to the robot's right"}),
        "swing": ("choice", "If it swings, how far does it open?", {k: v[0] for k, v in REVOLUTE_RANGES.items()}),
        "slide": ("choice", "If it slides, how far does it come out?", {k: v[0] for k, v in PRISMATIC_RANGES.items()}),
    }
    answers = await decider.ask(state, questions)
    sure = {key: answers[key].choice is not None and answers[key].confidence >= threshold for key in answers}
    if not sure["kind"]:
        return None, answers
    notes = [f"kind {answers['kind'].confidence:.2f}"]
    if answers["kind"].choice == "revolute":
        side = measured_hinge(geometry, door)
        if side is None:
            return None, answers                        # no measured door: let Astra look at the picture
        opening = "push" if answers["opening"].choice == "push" and answers["opening"].confidence >= push_threshold else "pull"
        range_key = answers["swing"].choice if sure["swing"] else "quarter_turn"
        text, value = REVOLUTE_RANGES[range_key]
        proposal = {"status": "OK", "kind": "revolute", "hinge_side": side, "opening": opening,
                    "door_width_m": float(door["handle_offsets_m"][side]), "range": value, "contact_effort_nm": 5.}
        notes += [f"hinge {side} from the depth", f"{opening} ({'jev' if opening == 'push' else 'a handle is pulled'})",
                  f"range {range_key} ({'jev ' + format(answers['swing'].confidence, '.2f') if sure['swing'] else 'default'})"]
    else:
        if not (sure["opening"] and answers["opening"].choice in ("pull", "slide_left", "slide_right")):
            return None, answers
        range_key = answers["slide"].choice if sure["slide"] else "medium"
        text, value = PRISMATIC_RANGES[range_key]
        proposal = {"status": "OK", "kind": "prismatic", "hinge_side": "none", "opening": answers["opening"].choice,
                    "door_width_m": .45, "range": value, "contact_effort_nm": 5.}
        notes += [f"opening {answers['opening'].choice} {answers['opening'].confidence:.2f}", f"range {range_key}"]
    proposal["rationale"] = f"jev: {proposal['kind']}, {text}; " + "; ".join(notes)
    return proposal, answers


class JevConstraintReasoner:
    """Astra's reasoner with constraint proposals asked of Jev first; everything else passes straight through."""

    def __init__(self, reasoner, decider, *, threshold=.8, log=None):
        self._reasoner, self._decider, self._threshold = reasoner, decider, float(threshold)
        self._log = log or (lambda message: None)

    def __getattr__(self, name):
        return getattr(self._reasoner, name)

    async def propose_constraint(self, context, entity_id, *, label, geometry, history=(), images=(), door=None, **kwargs):
        from .reasoning import ReasoningResult
        if self._decider is not None and self._decider.available:
            proposal, answers = await decide_constraint(self._decider, label=label, geometry=geometry, door=door,
                                                        threshold=self._threshold)
            if proposal is not None:
                self._log(f"constraint from jev for {entity_id}: {proposal['rationale']}")
                return ReasoningResult("OK", detail=proposal["rationale"], proposal=proposal)
            unsure = {k: (d.choice or d.error[:40], round(d.confidence, 2)) for k, d in answers.items()}
            self._log(f"jev unsure of {entity_id}'s motion {unsure}; asking astra")
        return await self._reasoner.propose_constraint(context, entity_id, label=label, geometry=geometry, history=history,
                                                       images=images, door=door, **kwargs)


def constraint_goal_text(goal, context):
    """What the arm will do for a constraint goal, in words: open the part, or bring it back shut."""
    constraint = next((c for c in context.get("constraints", ()) if c["constraint_id"] == goal["args"]["constraint_id"]), None)
    if constraint is not None and float(goal["args"]["target_value"]) <= float(constraint["minimum"]):
        return f"{'close' if constraint['kind'] == 'revolute' else 'slide in'} the part, back to where it rests shut"
    return "open the part and hold it at its goal"


async def decide_plan_variants(decider, task_text, goal_text, *, threshold):
    """What a plan template leaves open, in one call: let go when done? move away afterwards? Defaults when unsure."""
    answers = await decider.ask(f"Request to a robot arm: {task_text.strip()}\nThe arm will {goal_text}.", {
        "release": ("noul", "When that is done, should the gripper let go of what it holds?"),
        "retract": ("noul", "Afterwards, should the arm move its hand back away from the object?")})
    variants = {"release": True, "retract": True}
    for key in variants:
        if answers[key].choice is not None and answers[key].confidence >= threshold:
            variants[key] = answers[key].choice == "yes"
    return variants, answers


async def decide_recovery(decider, *, task_text, skill, failure_code, detail, attempt, options, threshold, tried=()):
    """Which way on to take after a failed step, among those not yet tried; (key or None, Decision).

    The options are the caller's whole menu (local recoveries and, while the planner may still be asked,
    asking it); stopping is not one of them: the task goes on while a way on is left.
    """
    state = (f"Request to a robot arm: {task_text.strip()}\nAttempt {attempt+1} failed at the step {skill}: "
             f"{failure_code}. {detail[:400]}" + (f"\nAlready tried after this failure: {', '.join(tried)}." if tried else ""))
    decision = await decider.choose(state, question="recovery", instructions="What should the robot try next?", options=options)
    if decision.choice is None or decision.confidence < threshold:
        return None, decision
    return decision.choice, decision
