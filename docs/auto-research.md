# Automatic research on the bench

This is the brief for an automatic research loop (Onyx, or any agent that edits code and measures the result) working on this repository with the real arm. Read [AGENTS.md](../AGENTS.md) first; everything there applies. Runs that move the arm are attended by the operator, or run under the operator's standing unattended GO described below.

## Goal

Make two typed tasks succeed on the bench, one after the other in the same node: `open the cabinet door`, then `close the cabinet door`. The second task must find that the door is open (the first task left it so; nothing tells it) and close it. A cycle that works leaves the door as it found it, so the bench resets itself. Reliably and quickly. Generalization matters more than this door: a change that hardcodes this cabinet, this pose, this label or the order of the two tasks is a regression.

## Metric

One command, one cycle, one JSON line on stdout with a `score` from 0 to 100 (the campaign's `cycle`):

```zsh
python tools/bench_eval.py cycle
```

Each task is scored by stage reached, as before (intake 10, plan 10, standoff 15, at the handle 15, grasped 15, followed 30 scaled by the fraction of the target, released 5; halved when the supervisor ended it). The evaluator also looks for itself: from the start pose it compares the wrist camera's depth with its view before the cycle, after each task. The open half keeps its followed points only when the camera saw the door move. The close task is sent only after a door that moved; its half is 80 % its stages and 20 % the camera seeing the scene back as it was. The cycle's score is the mean of the two halves. The JSON also carries `door` (the changed fractions), both halves with their `first_failure`, `duration_s` and the tail of the node's log. Maximise `score`; among equal scores prefer the shorter `duration_s` and fewer replans. One run is one sample: confirm an improvement with a second run before building on it.

Before spending a hardware run, the change must pass the no-motion tier:

```zsh
python tools/bench_eval.py offline --plan      # unit tests, design check, a live plan through every send gate
```

## How a hardware run goes

1. The command checks the pinned safety files and waits for the operator.
2. The operator makes sure the door is closed, stands at the e-stop and runs `python tools/bench_eval.py go`. A GO is used once. Without one the run records `operator_absent` and nothing is sent to the arm.
3. The command restarts the `adl` node so it loads the experiment's code, reads back the node's speed and effort limits, opens the hand and returns the arm to the recorded start pose through the same planner, gates and effort guard as every skill, then sends the open task, returns to the start pose to look, sends the close task, looks again and scores the cycle.

Use one worker. There is one arm; two experiments cannot share it.

Unattended: `python tools/bench_eval.py unattended --hours 12` is the operator's standing GO until then; runs then go without step 2. The bench halts itself, until the operator runs `attended` or `unattended` again, the first time a run ends in a safety fault, cannot be reset, returns no result, leaves the door or the view from the start pose other than it found them, or the window expires. A halted bench answers every hardware run at once with status `halted` and moves nothing, and a worker that sees it finishes the session. Nobody is at the e-stop in this mode: the guards, the speed and effort limits and the pinned files are the only protection.

## Scope

Editable: `rammp_adl/` except the files below, `skills/adl_skill_library.yaml` (regenerate schemas with `python tools/check_design.py --generate`), `tests/`, `docs/`.

Not editable. The command refuses to run when any of these differs from the operator's pin, and a change here is a request to the operator, written up with its reason, never a commit:

- `rammp_adl/safety.py`, `rammp_adl/motion/collision_guard.py`, `rammp_adl/motion/sheppy_client.py`, `rammp_adl/motion/sheppy_arm.py`
- `config/sheppy-bench.context.json`, `config/imagery-locality.json`, `config/reasoning.json`
- `rammp_adl/learned/jail.py`, `gate.py` and `api.py`: the sandbox and the primitive table model-written skills see
- `tools/bench_eval.py`, and the pin itself (`bench_eval.py freeze` is the operator's command)
- the sheppy manifest, and the node's `transit_speed_scale` (0.4), `contact_speed_scale` (0.25) and `touch_nm` (3.0), which may be tightened and never loosened

Also out of scope whatever the score says: removing or bypassing a guard, a gate, a confirmation or the face screen; raising an effort budget past 15 Nm; hand-written IK, Cartesian servo loops or any motion that is not a validated cuRobo trajectory; sending anything but screened, downscaled keyframes off the machine; weakening a test to make it pass.

## Rules for an experiment

- One idea per experiment, with the failure it addresses named from a run's `first_failure` or log.
- Add or update a unit test with every behavioural change; `offline` must pass before `hardware`.
- Read the run's log before the next idea. A failure that repeats unchanged after one fix is a wrong diagnosis.
- Report outcomes as they are. A fixture result, a plan-only result and a mocked provider response are not robot results.
- Record each session's findings in `docs/jetson.md` as a dated entry.

## Operator setup, once

```zsh
python tools/bench_eval.py record-start     # with the arm where every run should begin, looking toward the cabinet
python tools/bench_eval.py freeze           # pins the files above at ~/.config/rammp-bench/frozen.json
```

Re-run `freeze` only after reviewing a change to a pinned file yourself.
