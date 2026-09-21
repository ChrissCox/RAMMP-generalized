# Roadmap

Advance each phase when its exit criteria are met. Implement the six currently scoped skills through shared foundations, tabletop manipulation and articulated contact. Treat fast, accurate execution as an acceptance requirement throughout; use [the performance contract](14-performance.md) to distinguish useful task activity from avoidable waiting.

| Phase | Work | Exit criteria |
|---|---|---|
| 1. Foundation | Pin external sources; confirm hardware; implement registry/schema, mocks, Astra adapter, ownership, stop path, rolling adapter contracts and timing instrumentation | Checked offline plans; malformed/stale commands rejected; simulated trajectory-generation races, late updates and cancellation handled; provider access, stop characterization and wait-reason logging recorded |
| 2. Tabletop manipulation | Calibrate tool/cameras, world model, full-robot/attachment guard, cuRobo transit and gripper support; implement and commission rolling motion updates | Tabletop pick-and-place with verified release; bounded target shifts corrected during transit; delayed/invalid continuation causes bounded stop; safe preshaping overlap and immediate ready-node dispatch demonstrated |
| 3. Articulated contact | Constrained cuRobo adapter, local articulation fit and contact commissioning; progress from drawer translation to a supported hinge | Cabinet opens through the shared executor with validated in-flight path/model refinement; inconsistent contact or model mismatch stops with measured partial state |
| 4. Integration | Integrate cabinet interaction and tabletop object manipulation; implement planning ahead after its ownership and trajectory-validation contracts; sustained Jetson testing | Repeated runs with measured task accuracy, first-motion latency, avoidable idle time, memory, thermals, stop distance and false-success evidence |
| 5. Evaluation | Run repeatable integrated scenarios with supported hardware, objects and tools | Supported autonomous sequence, visible confirmation/handback and reproducible bounded recovery |

Initial integration scope: cabinet interaction plus supported tabletop object manipulation using observe, move_to_pose, set_gripper, grasp, release and follow_constraint. Their catalog entries and future traces do not require their implementation in these phases.

If constrained motion support through cuRobo is unavailable, report the gate and continue work on independent capabilities. Existing human keep-out and stop requirements still apply to the current skills. Human-proximate operation requires its own sensing/stopping commissioning before use. Dependent phases advance only after the required capabilities are validated.

Measure completion, interventions, false success, grounding error, local/cloud latency, avoidable idle time, transition latency, peak memory, sustained thermals, stop distance and recovery outcomes, tied to hardware/software/config identities. Later extend constraint types, human interaction, food profiles, camera configurations and wheelchair deployment with tested handlers.
