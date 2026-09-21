# Runtime and ROS package layout

The implemented core is one Python package, [rammp_adl](../../rammp_adl/), with two ROS build packages. This preserves module boundaries without introducing a separate package or node for every design layer. See the [runbook](../runtime.md) for installation and execution.

| Build package | Source and responsibility |
|---|---|
| rammp_adl_interfaces | [interfaces](../../interfaces/): canonical project-owned messages, services and actions; ament_cmake and rosidl generation |
| rammp_adl_runtime | [ros2/rammp_adl_runtime](../../ros2/rammp_adl_runtime/): ament_cmake_python installation of the same Python core, shared catalog/configuration assets, executable and simulation launch |

The Python package also installs through [pyproject.toml](../../pyproject.toml) and provides the `rammp-adl` command. Both installation paths use the canonical catalog and schemas; there is no independently maintained ROS copy. Explicit `--root` or the ROS `project_root` parameter selects an installed asset directory or source checkout.

| Python module | Implemented responsibility |
|---|---|
| [contracts](../../rammp_adl/contracts.py) | Strict parsing, catalog integrity/hash, full JSON Schema validation and capability-filtered registry |
| [world](../../rammp_adl/world.py) | Detached snapshots, typed metric poses, evaluator authority, evidence freshness, exact bound facts, atomic commits and epoch reconciliation |
| [validation](../../rammp_adl/validation.py) | Whole-DAG admission, ancestor-dependent preconditions, state conflicts, trusted geometry hooks and fresh dispatch receipts |
| [executor](../../rammp_adl/executor.py), [resources](../../rammp_adl/resources.py) | Task ownership, event-driven DAG scheduling, bounded retries/replans, cancellation and resource release after quiescence/effect commit |
| [handlers](../../rammp_adl/handlers.py) | Six explicit backend-facing skill handlers; no food acquisition or user-transfer handler |
| [safety](../../rammp_adl/safety.py) | Local supervision, liveness, stop/held checks, epoch gate and bound single-use confirmations; independent deployed hardware stopping remains external work |
| [reasoning](../../rammp_adl/reasoning.py) | Astra Responses client, generated grammar, bounded cloud requests, refusal/error handling and grounding of known entities |
| [perception](../../rammp_adl/perception/) | Camera ingestion, capture/crop provenance, local geometric fitting and track storage; no automatic end-to-end scene reconstruction |
| [motion](../../rammp_adl/motion/) | Rolling trajectory contracts, validation/activation guards, bounded target/constraint refinement, planner leasing and optional pinned static cuRobo adapter |
| [simulation](../../rammp_adl/simulation.py) | Synthetic ADL backend and separate MuJoCo Gen3 joint-physics replay |
| [ros_bridge](../../rammp_adl/ros_bridge.py), [ros_node](../../rammp_adl/ros_node.py) | Project-owned ROS services/actions, held reasoning/grounding leases, fresh status, read-only duplicate commit lookup and simulation-only composition |
| [app](../../rammp_adl/app.py), [cli](../../rammp_adl/cli.py), [telemetry](../../rammp_adl/telemetry.py) | Composition roots, local commands, structured events and timing summaries |

The current ROS node constructs the fixture backend only. `hardware_motion_enabled=true` is rejected. Its startup parameters are immutable, and no Kinova command publisher, driver action client or control-token acquisition is constructed. UserConfirm currently reports that no local assent provider is connected; a ROS request is never treated as user assent.

The existing Kinova driver and RAMMP cuRobo remain external pinned dependencies. The optional cuRobo adapter provides static planning boundaries only and cannot qualify online constrained motion or future suffix replacement. The logical safety module currently shares the runtime process; it does not replace an independent deployed driver watchdog or physical e-stop.

Python contracts, executor integration, rolling protocols and MuJoCo joint replay have been exercised locally. ROS interface generation, launch and transport execution have not been compiled/run in this Windows environment. A [Humble container recipe](../../deployment/Dockerfile.simulation) is supplied for that next validation step, but Docker and a usable Linux/WSL runtime were unavailable here. Integrated cuRobo ADL simulation, Jetson benchmarks and commissioned hardware work remain distinct acceptance steps.
