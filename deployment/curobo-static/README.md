# Actual cuRobo static simulation probe

This isolated Jetson probe generates a real cuRobo trajectory for the existing
bare Gen3 simulation model. It uses the production source/version checks and
independent MuJoCo admission checks. It has no ROS launch, robot transport,
camera access, network, privileged mode, or physical commissioning profile.

The first 2026-09-11 probe generated a 31-point, 0.60-second trajectory for a 3 cm
Cartesian target in 0.2415 seconds after warmup. Fifteen independent model FK
comparisons passed: maximum position error 3.80 micrometres, rotation error
0.000430 radians. **The trajectory was rejected before physics replay** because
its endpoint derivatives did not meet the existing stationary-boundary contract.
This is actual GPU planning evidence, not ADL contact simulation or robot evidence.

The subsequent [stationary interpolation extension](../../rammp_adl/motion/curobo_stationary.py)
now passes the unchanged boundary gates and actual MuJoCo replay. The accepted
32-point, 0.62-second path underwent 126 independent state checks, with maximum
FK disagreement 3.65 micrometres, final simulated goal error 3.53 mm / 0.00910 rad,
and no contact. [Accepted result](../../artifacts/jetson/curobo-boundaries/quintic-1/probe/result.json)
and [raw/original interpolation evidence](../../artifacts/jetson/curobo-boundaries/quintic-1/probe/curobo-optimized.json)
are retained separately from the original rejection.

| Boundary quantity | Returned maximum | Required maximum |
|---|---:|---:|
| Initial position error, rad | 9.10e-8 | 1e-5 |
| Initial velocity, rad/s | 0.002207 | 1e-5 |
| Initial acceleration, rad/s² | 0.110813 | 1e-4 |
| Terminal velocity, rad/s | 0.002158 | 1e-5 |
| Terminal acceleration, rad/s² | 0.107550 | 1e-5 |

Run from the project root; use a new output directory each time:

```zsh
python deployment/curobo-static/run.py \
  --output artifacts/manual/curobo-static-1 \
  --wrapper-checkout /home/abra/.cache/rammp-adl/curobo-static/pinned-wrapper \
  --gpu-cache /home/abra/.cache/rammp-adl/curobo-static/gpu
```

The current probe writes a structured `probe/result.json` and returns exit code
2 when independent admission rejects a generated candidate. GPU initialization
failures also return nonzero. It computes consistent derivatives from the same
boundary-constrained quintic spline and retains the existing independent boundary
gates. The host runner enforces a 600-second process bound and
stops only its own container on timeout/interruption. A local cache directory keeps
GPU compilation products warm between runs.

Evidence from the first actual solve: [result](../../artifacts/jetson/curobo-static/attempt-3/probe/result.json),
[exact candidate](../../artifacts/jetson/curobo-static/attempt-3/probe/curobo-trajectory.json),
[FK checks](../../artifacts/jetson/curobo-static/attempt-3/probe/fk-probe.json),
[invocation](../../artifacts/jetson/curobo-static/attempt-3/invocation.json), and
[original log](../../artifacts/jetson/curobo-static/attempt-3/probe.log).
That original invocation returned exit code 1 with a traceback; its structured
result was subsequently derived from the saved candidate and failed gate. The
executed harness is preserved alongside the invocation. Later runs use explicit
exit code 2 for this rejection.

## Source, packaging, and model provenance

The existing `rammp-curobo:jp6` image is unchanged. Its installed package reported
0.0.0 although its clean cuRobo checkout was exact v0.7.8. Reinstalling from that
unchanged checkout with setuptools_scm already present repaired both module and
distribution metadata to 0.7.8. No version field was edited manually. All 445
installed tracked source files and five CUDA binaries match the retained source
and build outputs. The source commit is
`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`; the isolated RAMMP wrapper checkout is
exact `320872b709b276fc7283190d24edef7f8632bec9` and is imported directly from that
clean checkout. [Package hashes](../../artifacts/jetson/curobo-static/package-provenance.json)
and [image/source summary](../../artifacts/jetson/curobo-static/summary.json) preserve the evidence.

The reviewed new image is `rammp-adl-curobo-static:jp6-v078`, ID
`sha256:b34c1bcf9fc094ecb39e7de3591f4e22112b9713e2892783259a2ac0cc7d949e`.
The runner checks that exact ID. The base image ID is
`sha256:081953f22faa1f815196ea1726a717ea73a690449ef8175e2949aec4629db28d`.

The generated bare-arm config selects the source URDF's `end_effector_link`:
its bracelet-relative transform exactly matches the existing MuJoCo `pinch_site`.
The wrapper's default `tool_frame` is another 120 mm beyond that point and is
therefore unsuitable for the bare-arm fixture. This derivation is simulation
model reconciliation, not physical tool or camera calibration. Only the eight
arm links are included; their sphere values come unchanged from the upstream
v0.7.8 config. MuJoCo independently checks its own full arm meshes. The generated
URDF/config/world/model contract hashes, local simulation limits, and explicit
source derivation are retained in the probe directory. No gripper is simulated.

## Rebuild independently

The build helper refuses existing context directories, wrapper checkout
directories, and derived image tags. It verifies the original base image ID,
copies only the hash-locked dependency wheels, creates an isolated exact-pinned
wrapper checkout, and builds with networking disabled. For example:

```zsh
python deployment/curobo-static/build.py \
  --wheel-dir artifacts/jetson/curobo-static/build-context/wheels \
  --context artifacts/manual/curobo-image-build-1 \
  --wrapper-repository /home/abra/RAMMP-CuRobo \
  --wrapper-checkout /home/abra/.cache/rammp-adl/curobo-static/wrapper-rebuild-1 \
  --image rammp-adl-curobo-static:rebuild-1
```

The wheel bundle contains pinned MuJoCo 3.13.0, glfw 2.10.2, etils 1.13.0,
absl-py 2.5.0, and PyOpenGL 3.1.5. The retained build log is
[image-build.log](../../artifacts/jetson/curobo-static/image-build.log).
Review a rebuilt image's source and binary provenance before updating the host
runner's image ID pin. Runtime provenance/frame verification refuses Python
optimization mode, so `-O` cannot remove its checks.

## Stationary boundary integration

The pinned wrapper constructs cuRobo `JointState.from_position`, which sets
velocity and acceleration to zero. These zeros reach the planner request. It
then returns cuRobo's trimmed interpolated plan unchanged. In v0.7.8 the default
LINEAR_CUDA interpolation linearly combines raw position, velocity, and
acceleration independently, rescaling derivatives with the optimized time step.
At sample zero its weight selects the raw first derivative, so interpolation
alone cannot explain away the observed nonzero starting derivative. The raw
optimized state buffers were not captured in this probe. The default trajectory
optimization configuration uses CENTRAL state finite differences.

Source references within the clean upstream checkout are
`src/curobo/types/state.py:108`, `src/curobo/util/warp_interpolation.py:25`, and
`src/curobo/content/configs/task/finetune_trajopt.yml:27`; within the pinned
wrapper they are `core/rammp_curobo/planner.py:361` and `:482`.
The CPU interpolation alternatives only update position arrays and do not supply
matching derivative evidence, so selecting them is not an acceptable fix.
The project-owned extension completes cuRobo's SciPy quintic interpolation route:
raw optimized position knots are retained, endpoint velocity/acceleration
constraints are applied to the spline itself, and every returned q/dq/ddq sample
comes from that same spline. The changed path is rechecked by cuRobo and the
independent simulation verifier. Selecting BACKWARD instead was tested and
rejected because the pinned indexed CUDA kernel explicitly asserts false.

[curobo_worker.py](../../rammp_adl/motion/curobo_worker.py) exposes the same bounded
planning-only route for an explicit start, metric goal and local world. An actual
GPU worker invocation also passed. Its output is a candidate, never an execution
permit. The [hardware commissioning composition](../../rammp_adl/motion/hardware_test.py)
performs separate local admission before any ROS command. Rolling boundaries,
contact physics, camera grounding and physical commissioning remain separate work.
