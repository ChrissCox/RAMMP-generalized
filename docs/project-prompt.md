# Task-level autonomy for ADL manipulation on a Kinova Gen3

Design and implement one reusable ROS 2 autonomy layer for a wheelchair user's Activities of Daily Living. Given a natural-language goal, perceive the actual scene, maintain an uncertainty-aware world model, use OpenAI GPT-6 Astra (`gpt-6-astra`) to propose a DAG of registered skills, validate it locally, execute with monitoring, and recover or return control to the user.

Prioritize speed and accuracy together: minimize command-to-first-motion latency and avoidable pauses during execution. Plan the upcoming motion while the current motion runs, including local cuRobo updates to the remaining trajectory and grounded endpoint when observations change within the admitted skill's limits. Execute a validated portion while preparing and validating its continuation, with smooth moving-state handoff and a bounded stopping option. This reactive capability is required, not merely an optional later optimization. Keep planning resources warm, local perception active and dispatch prompt. Measure waiting by cause alongside task success and geometric accuracy; do not improve timing by weakening validation or adding pointless motion. See [performance requirements](design/14-performance.md).

## Scope and baseline

Use one codebase for cabinet/drawer/fridge/microwave doors, supported container lids, feeding, drinking, pick-and-place, buttons/switches, handover, and wiping. Derive shared physical skills from that concrete coverage set. A fixed, versioned skill library is available during a task. Developers can extend it only with implemented handlers, typed contracts, and tests. Task recipes belong in plans, not separate pipelines.

Current scope is observe, move_to_pose, set_gripper, grasp, release and follow_constraint. Feeding and human transfer are out of scope; their former skills were removed from the catalog on 2026-09-16.

Assume Jetson AGX Orin 64 GB in MAXN provisionally; benchmark the actual module, cooling, power mode, GPU memory, and latency. Do not declare smaller Orin models infeasible without measurements. The arm is a Kinova Gen3 7-DoF with joint torque sensing; verify actual command/state rates, firmware, stop behavior, and installed gripper. The gripper supports continuous aperture; map physical meters through calibration. The user has confirmed the installed gripper is a Robotiq 2F-85. Its identity is known; aperture, tool-frame and force/current calibration remain separate verification work.

Use an Intel RealSense D405 wrist camera (nominal short range about 7-50 cm; no IMU). The Orbbec scene camera about one meter to the left is temporary testing equipment; camera model, usable depth range, and FOV remain deployment parameters. Assume a nominal 90-degree horizontal FOV only for test-layout planning, never for geometry; measured CameraInfo governs. Do not make a scene-camera identity mandatory for the executor.

The base is table-mounted now. Future wheelchair mounting requires calibrated transforms and timestamped motion; it does not imply that every rigid mounting transform changes continuously. Use base_link as a local planning frame; invalidate relevant scene geometry on external base motion.

Use ROS 2 Humble. The prior design identifies [rammp-org/kinova-gen3-ros2](https://github.com/rammp-org/kinova-gen3-ros2) as the candidate driver; confirm it is the intended installed driver because the original prompt's link was broken. Use [RAMMP-CuRobo](https://github.com/rammp-org/RAMMP-CuRobo). Confirm source revisions and actual interfaces before coding adapters.

cuRobo owns arm IK and trajectory generation. A driver tracking a cuRobo trajectory is consistent with that requirement. If the wrapper lacks constrained contact or human-proximate planning, expose the gap, implement/validate an adapter extension, and keep those skills unavailable until then. Do not silently replace cuRobo with hand-written IK, Cartesian servoing, learned action output, or MoveIt.

## Architecture and contracts

Keep perception, scene state, cloud reasoning, plan validation, skill execution, and independent safety supervision distinct. Reasoning uses one stable local GeneratePlan service. GroundTarget is a separate bounded inference operation behind the same OpenAI adapter. No API calls belong in robot control or tracking loops. Plan at task start and replan on explicit events. Before any cloud wait, cancel/settle active motion and verify held state; never assume cancellation acknowledgement alone proves stopping.

Distinguish cloud task planning from local motion replanning: the latter runs during execution inside the same skill and does not require a cloud wait, new DAG node or task replan. Adjust the same grounded target only within trusted commissioned bounds. Preparing a later skill never assumes its predecessors have succeeded; cross-skill ownership and terminal effects remain explicit.

Use a DAG with explicit dependency edges. Derive exclusive resource claims, limits, confirmations, timeout, and success criteria from the trusted skill catalog; the model cannot set them. Two executing skills may overlap only with disjoint claims and compatible state dependencies. Empty-gripper preshaping may overlap transit only if both are validated for the aperture envelope.

Represent stable object IDs, pose and orientation uncertainty, timestamps, articulation hypotheses, collision geometry, attachment state, calibrated TF, and the user's body/keep-out volumes. Cloud detections are hypotheses. A depth pixel gives a 3D point, not a full 6-DoF pose. Estimate orientation and articulation locally or report them unknown. Use compact, bounded task-relevant context and optional redacted crops; document exactly what leaves the device.

Validate complete plan syntax, argument types, IDs, graph, data/state dependencies, available capabilities, preconditions, and safety policy before admission. Use cuRobo for all resolvable reachability and swept-path checks. Future perception cannot be certified in advance: unknown geometry allows only an explicit perception phase; every later motion needs fresh binding and dispatch validation. A rejected plan executes no nodes. A previously admitted plan may be halted after partial success when new observations invalidate it.

Stop, hold, e-stop, confirmation, and ownership belong to the supervisor; model output cannot disable them. Check full-arm, gripper, attachment, and stopping volumes. Keep force in N, torque in Nm, and uncalibrated joint residuals distinct. Require measured contact sensing and stop envelopes for human-proximate execution; do not invent safe contact thresholds.

## Required deliverables

1. Architecture diagram, placement and ROS mechanism table.
2. ADL decomposition followed by a small shared skill set; a machine-readable catalog with typed arguments, frames/units, preconditions, evidence-based postconditions, sensors, exclusive claims, interruption, timing, failures, and bounded recovery.
3. World model and calibration/freshness policy; serialized reasoning context.
4. Astra API adapter configuration, exact JSON plan schema, grounding, local validation, failure handling and retry/replan caps.
5. DAG executor with task-state machine, preemption, idempotent completion commits, and safety supervision.
6. Commissioning requirements for per-class speed/force/torque limits, keep-outs, confirmations, watchdogs and deterministic fallback.
7. Risk-ordered implementation phases with measurable capability and validation criteria.
8. Architecture-changing assumptions and capability gaps.
9. Package layout and project-owned ROS IDL.
10. Checked example DAGs and full traces for opening a cabinet and bringing a water bottle to the mouth, including parallelism and injected failure recovery.

Preserve hardware safety and real unknowns while removing duplicate definitions, unverifiable claims, stale instructions, and unnecessary framework requirements. Ask at most five clarifying questions only when answers change architecture; otherwise proceed with explicit assumptions.
