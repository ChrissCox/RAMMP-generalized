# Simulation and motion integration

The Python runtime has three deliberately distinct validation surfaces:

| Surface | What runs | What it establishes |
|---|---|---|
| `FixtureBackend` + `FixtureGeometry` | Real DAG executor and local contracts against synthetic task state | Resource overlap, typed effects, support/retention policy, cancellation and injected failure recovery |
| `rolling_fixture_trace()` | The real project-owned suffix acceptance/activation state machine against ideal joint-state fixtures | Generation, epoch, freshness, continuity, bounded stop and late-result handling |
| `MujocoReplay` | Actual MuJoCo rigid-body dynamics with the vendored bare Gen3 | Joint tracking under position actuators; `replay_rolling_fixture()` also checks a moving suffix switch against actual simulated q/dq/ddq |

Neither synthetic task-state success nor a joint replay establishes cuRobo
planning, Cartesian ADL success, gripper physics, calibrated contact sensing,
full swept/stopping collision protection, or hardware validation. The analytic
joint fixture helper is test input only and cannot register as a planner.

Run from the repository root after installing its Python dependencies:

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_motion_runtime.py -v
.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_simulation.py -v
.venv/Scripts/python.exe -m rammp_adl simulate --scenario cabinet
.venv/Scripts/python.exe -m rammp_adl physics
.venv/Scripts/python.exe -m rammp_adl rolling
```

The physics dependency is optional; absent MuJoCo/numpy skips the two physics
tests with an explicit reason. Other runtime tests require the project's runtime
dependencies. Use the project's virtual environment, not a different global
Python installation.

`MujocoReplay().replay_rolling_fixture()` uses `rolling_scene.xml`, a project
overlay that excludes the known overlap between the directly adjacent base and
shoulder meshes. The upstream XML and meshes remain unchanged. Raw upstream
replay reports those contacts explicitly. No broad collision exclusion or
geometry removal is used. Both models lack the installed gripper and D405.

## Optional cuRobo code

`rammp_adl.motion.curobo.RammpCuroboAdapter` uses the pinned RAMMP pure Python
planner API, retains a lock across world installation plus planning, preserves
joint derivatives, and returns an exact trajectory for independent validation.
It rejects a moving start because that upstream API accepts only positions.
Its source revision and import origin are checked before loading.

`CuroboMpcAdapter` targets cuRobo 0.7.8 and constructs a bounded candidate horizon
from the documented MPC joint-state outputs. It keeps solver/world ownership
until a cancelled GPU operation actually completes. A successful actual GPU
moving-boundary selfcheck plus independent trajectory validation is required
before candidate generation is exposed; hardware capabilities remain empty.
No GPU selfcheck was run on the current native Windows environment.

`RollingMotionSession` in `rammp_adl/motion/session.py` runs planning and independent
validation concurrently with local tracking. Its simulation transport callback
contract requires the full catalog resource set, versioned world/target reads,
measured state, and a local supervisor. It rejects stale pending target updates,
stops on compute deadlines, and retains ownership until cancelled work drains.
The scheduling tests use a joint fixture planner, not cuRobo.

`python -m rammp_adl.motion.integration --describe-model` prints the installed
model identity. The same module implements static cuRobo-to-MuJoCo execution
when given explicit `--source-root`, `--planner-config`, `--contract` and `--goal`
files. `load_contract` defines the required source/config/URDF/model identities
and local simulation criteria. Actual FK agreement and sampled geometric checks
must pass before replay. The implemented wiring has been exercised with a mock
upstream API and actual MuJoCo; the GPU path remains untested. This runner does
not establish constrained ADL contact physics or continuous swept/stopping safety.

Source references:

- [RAMMP planner, pinned commit](https://github.com/rammp-org/RAMMP-CuRobo/blob/320872b709b276fc7283190d24edef7f8632bec9/core/rammp_curobo/planner.py)
- [cuRobo 0.7.8 MPC example](https://github.com/NVlabs/curobo/blob/v0.7.8/examples/mpc_example.py)
- [cuRobo 0.7.8 MPC interface](https://github.com/NVlabs/curobo/blob/v0.7.8/src/curobo/wrap/reacher/mpc.py)
- [Vendored Gen3 source and license](vendor/kinova_gen3/SOURCE.md)

The project-owned `RollingController` defines future suffix installation and
measured-state activation. It is not an adapter to an existing driver action.
TODO: confirm against driver: atomic switch acceptance, generation feedback,
interpolation, stopping behavior and independent watchdog integration.
TODO: confirm against planner: installed configuration and model agreement,
constrained path coverage, bounded GPU deadlines and independent swept/stopping
validation. These missing checks cannot be replaced by a boolean success flag.
