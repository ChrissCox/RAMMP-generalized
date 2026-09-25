# 15 Learned skills: model-written code in a sandbox

Status: 2026-09-25. Implemented in [rammp_adl/learned/](../../rammp_adl/learned/): the jail, the code gate,
the call protocol, the skill store, the binding of the primitive API to the executor
([host.py](../../rammp_adl/learned/host.py): `ExecutorHost` runs each motion primitive as a one-step plan
through admission, the executor and the guards; `DryRunHost` moves nothing and has the validator admit the
skill's steps as one growing chain), the write-gate-rehearse-run-verify loop
([loop.py](../../rammp_adl/learned/loop.py), writer model in [config/learning.json](../../config/learning.json)),
and the node's entry points (`task --skill`, `--library`, `--learn`, `--dry-run`, `--rehearse-only`). The loop
and the binding are tested against the fixture runtime; the writer model answered one live request with a
skill the gate admitted. Also implemented: RSIAgent's parallel candidates, separate verifier, failure lessons and curriculum (below), and
local perception (OWLv2) behind discovery and grounding. Not yet run on the robot. Not yet built: cuRobo reach in
the rehearsal.

## Why

The runtime should zero-shot tasks it was never built for and get better with use. The approach is
[Voyager](https://github.com/MineDojo/Voyager)'s, carried from Minecraft to a real arm:

- **A skill library of code.** Each skill that worked is kept as a Python function with a description, found
  again by what it does, and called from bigger skills. The library grows; tasks get cheaper as it does.
- **Iterative writing.** A model writes a skill for a task, it runs, and the outcome comes back to the model:
  the error, the primitive that refused and why, the measurements, the verifier's judgement. The model rewrites.
- **Self-verification.** A skill joins the library only when the task is judged done, and here that judgement
  is by measurement first, a vision model second.
- **An automatic curriculum.** A model proposes the next task from the scene and what the library can already do.

Voyager's skills can do anything the game API allows, and a bad one costs nothing. On this arm a bad one can
hurt someone or break something, so the one thing that changes is where the line is drawn.

## The line: the sandbox exposes only guarded primitives

Two kinds of code exist, and only one of them is ever written by a model.

1. **Primitives** are written by people, tested, and registered as today. Every speed, force limit, effort
   budget, guard, planner request, resource claim and retry policy lives in them or below them. A primitive
   call is a one-step plan (or a template plan) that the validator admits and the executor runs, with the
   collision and effort guards, the supervisor and the e-stop exactly as for any plan.
2. **Learned skills** are written by a model. They are Python functions that run in a jail and can do one
   thing to the world: ask the host for a primitive call. They decide *which* primitives, in what order,
   with what arguments, and what to do with the results; they cannot change *how* a primitive moves.

A learned skill therefore has exactly the power the planner has today (choosing and ordering validated
steps) plus loops, conditions and arithmetic over measurements. A learned skill that is wrong, careless or
hostile can at worst request primitive calls, each of which is admitted or refused on its own.

### What the jail is

Three independent layers; each would have to fail for code to reach the robot another way.

1. **The operating system.** The skill runs in a separate process under bubblewrap: its own user, network,
   process, IPC and mount namespaces, no network interface, no `/home`, no `/opt/ros`, no credentials or
   environment, a private empty `/tmp`, the system `/usr` read-only, and CPU-time, memory, file and process
   limits. It dies with its parent. Its only channel is a pipe to the host.
2. **The code gate.** Before anything runs, the source is parsed and refused if it imports anything outside a
   short list of pure modules (`math`, `statistics`, `itertools`, `collections`, `functools`, `random`),
   touches any name or attribute beginning with an underscore, or names `exec`, `eval`, `compile`, `open`,
   `getattr` and the other reflective builtins. The skill must define `run(robot, **args)` with a docstring.
3. **The interpreter.** The code runs with a builtins table holding only harmless builtins, and `robot`, a
   stub whose every method is a message to the host.

The host, not the jail, enforces the call budget, the wall-clock limit, argument types and bounds for each
primitive (from the API table below, never from the skill), and turns every call into validated execution.

### The primitive API (version 0)

| Call | What it does | Bound by |
|---|---|---|
| `find(query)` | objects matching words: id, label, position, size, kind | local perception |
| `look(direction)` | turn the wrist camera: left, right, up, down, back, closer | guarded transit |
| `move_to(object, role)` | hand to an object's pregrasp, grasp, retract or above pose | validator, guards |
| `open_hand(aperture_m)` | set the gripper, 0 to 0.085 m | catalog aperture |
| `grasp(object)` | close on an object at its grasp pose | stall check against the measured stop |
| `release(object, onto)` | let go, onto a surface or the part's own support | support rules |
| `move_part(object, amount, unit)` | move an articulated part along its measured joint | constraint limits, refits |
| `go_home()` | the arm to the home joints | guarded transit |
| `holding()`, `gripper_opening()` | the hand's state | measurement |
| `check(condition, **args)` | a catalog predicate, measured | world model |
| `use(skill, **args)` | call another library skill | the same jail and budget |
| `log(text)` | a line in the task log | length limit |

New primitives (place, press, push, pour, wipe, insert, hand over) are added by people when the curriculum
keeps asking for them; the model can write a request for one, never the primitive itself.

## The loop

1. **Curriculum.** From the scene and the library, a model proposes a task. With an operator present, each
   proposal is shown and needs a go; unattended, only tasks with an automatic reset are proposed.
2. **Retrieve.** The library's skills closest to the task (by description) are given to the writer with the
   API table.
3. **Write.** A strong model writes `run(robot, **args)`. Writing is rare, so a paid model is cheap here;
   running a written skill afterwards costs no model calls unless it asks for perception.
4. **Gate and rehearse.** The code gate; then a dry run against a host that plans every motion primitive
   without moving (cuRobo plan-only, the validator's admission) and answers perception from the latest scene.
   Errors go back to the writer.
5. **Run.** The skill runs against the real host. Every refusal, guard trip and measurement is recorded.
6. **Verify.** Local measurement decides whether the task's condition holds; a vision model gives a second
   opinion from the last frame. Both are shown to the writer on failure, with the log, for a rewrite (at most
   a few rounds per task).
7. **Keep.** A skill that verified is stored as `provisional`; after it verifies on the bench a set number of
   times it becomes `verified`; repeated failures demote it. Every version is kept with its evidence.

## From RSIAgent

[RSIAgent](https://aetherlabsai.github.io/RSIAgent/) (arXiv:2609.15364) improves an agent without touching its
weights. A curriculum agent, an actor that works in code, and a verifier kept apart from the actor build a memory
of procedures, settings and failure lessons. Broad exploration runs several complementary projects in parallel,
and deep exploration then refines sequentially. What carries over to a real arm:

- **Broad in rehearsal, deep on the arm.** Each round the writer offers up to three different approaches in one
  request. All are gated and rehearsed side by side at no cost. Jev picks among those that rehearse cleanly,
  scoring all of them in one call, and only that one runs.
- **A verifier apart from the actor.** A measured goal decides where one binds. Otherwise a model judges from
  the view from home before and after, knowing the request and never the skill's code; done means 4 of 4 at
  confidence 0.7 (`_judge_from_home`).
- **Failure lessons.** After a session with failures, one request distils up to three general lessons, which
  are kept in the library and shown to the writer on similar tasks.
- **A curriculum** (`task --propose`): from what the robot sees, the skills it has and the lessons, a model
  proposes practice tasks aimed at what is missing, each with how the scene is put back. An operator picks one
  and runs it with `--learn`.

Its environment resets between attempts (VMs) have no counterpart on a real arm. The jail, rehearsal, the guards
and going home stand in for them, and the operator's choice of practice keeps unattended physical exploration off.

## What stays## Open

- Local perception (the detector, outlines, a small vision model) behind `find` and verification, so running
  skills is fast and free.
- cuRobo plan-only reachability in the rehearsal (today it checks admission, not reach).
- Unattended curriculum runs: they need automatic resets (a door put back, objects returned), which only
  some tasks have yet.
- Which model writes skills; how the curriculum is paced; how many verified runs promote a skill.
