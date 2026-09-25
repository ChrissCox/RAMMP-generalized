# 15 Learned skills: model-written code in a sandbox

Status: design, 2026-09-25. The jail, the code gate, the call protocol and the skill store are implemented
([rammp_adl/learned/](../../rammp_adl/learned/)). The binding of the primitive API to the executor, the
curriculum and the write-verify loop are not yet implemented.

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

## What stays

The validator, the executor, the supervisor, the guards, the recovery loop, going home before and after every
task, and the rule that metric poses, planning and world facts are local. The existing catalog skills become
the first primitives; the door work becomes the first learned skill ("open a hinged part by its handle").

## Open

- Local perception (the detector, outlines, a small vision model) behind `find` and verification, so running
  skills is fast and free.
- The executor binding of the primitive API, and the dry-run host.
- Which model writes skills; how the curriculum is paced; how many verified runs promote a skill.
