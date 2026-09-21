# Automatic research on the bench

This is the brief for an automatic research loop (Onyx, or any agent that edits code and measures the result) working on this repository with the real arm. Read [AGENTS.md](../AGENTS.md) first; everything there applies. The operator is present at the e-stop for every run that moves the arm.

## Goal

Make the typed task `open the cabinet door in front of you` succeed on the bench, reliably and quickly: find the pull, reach the standoff, align, grasp, follow the door to its target angle, release. Generalization matters more than this door: a change that hardcodes this cabinet, this pose or this label is a regression.

## Metric

One command, one run, one JSON line on stdout with a `score` from 0 to 100:

```zsh
python tools/bench_eval.py hardware
```

The score is by stage reached (intake 10, plan 10, standoff 15, at the handle 15, grasped 15, followed 30 scaled by the fraction of the target angle, released 5) and is halved when the supervisor had to end the run. The JSON also carries `first_failure`, the phases, the task's duration and the tail of the node's log, which names every move, alignment look and followed waypoint. Maximise `score`; among equal scores prefer the shorter `duration_s` and fewer model requests. One run is one sample: confirm an improvement with a second run before building on it.

Before spending a hardware run, the change must pass the no-motion tier:

```zsh
python tools/bench_eval.py offline --plan      # unit tests, design check, a live plan through every send gate
```

## How a hardware run goes

1. The command checks the pinned safety files and waits for the operator.
2. The operator closes the door, stands at the e-stop and runs `python tools/bench_eval.py go`. A GO is used once. Without one the run records `operator_absent` and nothing is sent to the arm.
3. The command restarts the `adl` node so it loads the experiment's code, reads back the node's speed and effort limits, opens the hand and returns the arm to the recorded start pose through the same planner, gates and effort guard as every skill, then sends the task and scores the result.

Use one worker. There is one arm; two experiments cannot share it.

## Scope

Editable: `rammp_adl/` except the files below, `skills/adl_skill_library.yaml` (regenerate schemas with `python tools/check_design.py --generate`), `tests/`, `docs/`.

Not editable. The command refuses to run when any of these differs from the operator's pin, and a change here is a request to the operator, written up with its reason, never a commit:

- `rammp_adl/safety.py`, `rammp_adl/motion/collision_guard.py`, `rammp_adl/motion/sheppy_client.py`, `rammp_adl/motion/sheppy_arm.py`
- `config/sheppy-bench.context.json`, `config/imagery-locality.json`, `config/reasoning.json`
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
