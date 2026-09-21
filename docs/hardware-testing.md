# Hardware execution and physical testing

There is now a guarded ROS trajectory command path, an instrumented build of the custom Kinova driver, a retained cuRobo planner, and a free-space commissioning composition. Separate measured observation and empty-gripper aperture adapters now connect to canonical handlers, with explicit simulation rehearsals. The command overlay uses the source revision and split interface packages inspected in the current installation; it does not modify or adopt the running driver. No physical robot was commanded during implementation. The default ADL DAG composition still runs fixtures; `simulate --reasoner astra` does not command the robot. Cabinet contact, measured grasp/release behavior, calibrated physical scene integration and rolling hardware handoffs remain unavailable.

The first physical route is a supervised, empty-gripper, free-space move inside an independently established static test region. It is deliberately separate from declaring an ADL skill implemented. The installed Robotiq 2F-85 and wrist D405/mount must be covered by its geometry evidence. Camera calibration is not invented to make this route pass.

## What was tested

* The physical-capable driver compiled against the local Kortex 2.8.0 AArch64 SDK. Compilation did not connect to the robot or run its driver.
* Actual ROS DDS with an explicit Kortex-disabled SimTransport build passed fresh exchange feedback, non-stealing ownership, full trajectory delivery, cancellation with measured stop, rejection of a false success, and independent watchdog expiry.
* Actual GPU cuRobo planning and a separate MuJoCo replay passed. The corrected cuRobo interpolation preserves a consistent position/velocity/acceleration polynomial, with stationary endpoint constraints and independent revalidation. It does not replace IK or zero only the derivative fields.

Evidence and validation limits are recorded in [Jetson setup](jetson.md). Upstream SimTransport leaves arm positions/velocities fixed: its action test is **not** trajectory tracking or physical stopping validation. MuJoCo uses the bare-arm simulation model and is not an installed gripper/camera model approval.

## Rehearse GPU planning and monitored motion now

This integrated software rehearsal runs actual retained cuRobo planning, binds its target and collision-world identities, validates the trajectory, and executes through the same Python command gateway against actual MuJoCo dynamics. It exercises completion, cancellation, stale feedback, changed execution epoch and driver fault. Resource claims remain held when stopping cannot be verified. It starts no ROS node or robot driver and has no hardware mode.

```zsh
cd /home/abra/RAMMP-generalized
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python -m rammp_adl.motion.rehearsal \
  --planner-config artifacts/jetson/hardware-execution/worker-model/planner.yaml \
  --model-dir artifacts/jetson/hardware-execution/worker-model \
  --wrapper-checkout /home/abra/.cache/rammp-adl/curobo-static/pinned-wrapper \
  --gpu-cache /home/abra/.cache/rammp-adl/real-world-ready-gpu \
  --contract artifacts/jetson/real-world-ready/rehearsal-inputs/model-contract.json \
  --goal artifacts/jetson/real-world-ready/rehearsal-inputs/goal.json \
  --output artifacts/manual/integrated-rehearsal
```

Choose a new output directory. The first plan includes GPU initialization; later plans reuse that worker. This tests a synthetic target and the bare-arm model. The Python simulation port supplies action messages internally: actual DDS is tested separately below. Camera grounding, Astra task planning, physical gripper/contact behavior and installed camera geometry are outside this rehearsal. Its numerical tolerances are simulation criteria and cannot populate a physical profile.

## Test the ROS command path now, without hardware

The checked driver sources are isolated from the existing installations. Build a new simulation workspace:

```zsh
cd /home/abra/RAMMP-generalized
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.zsh
source .venv/bin/activate
python deployment/driver-hardware/build.py --mode sim \
  --workspace /tmp/rammp-driver-test --jobs 2
source /tmp/rammp-driver-test/install/setup.zsh
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=91
python deployment/driver-hardware/test_transport.py \
  --workspace /tmp/rammp-driver-test \
  --output artifacts/manual/driver-transport-test
```

Choose a new output directory each run. Build the current sources as above: old `/tmp` workspaces may have disappeared or contain an earlier extension/IDL. This build uses `rammp_arm_interfaces` for trajectories and `rammp_common_interfaces` for ownership and stop. The reproducer checks `KINOVA_ENABLE_KORTEX=OFF`, launches the binary explicitly with `--sim`, and shuts down only that process. It cannot select physical mode. Domain 91 is reserved for this isolated test; every command/feedback endpoint used by the test is also remapped into a unique random namespace, so it cannot address another driver's global services.

### Test calibrated aperture transport in ROS simulation

With the current simulation workspace built and sourced, run:

```zsh
ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=93 \
python deployment/driver-hardware/test_gripper_transport.py \
  --workspace /tmp/rammp-driver-test \
  --output artifacts/manual/gripper-transport-test
```

This reproducer checks the executable hash and Kortex-disabled manifest before starting its isolated `--sim` driver. It uses an explicitly synthetic aperture table and requires measured aperture dwell. Cancellation must receive the new exchange-backed gripper halt acknowledgement and fresh settling measurements. Later exchanges must preserve the halt target. [Actual DDS results](../artifacts/jetson/gripper-execution/dds-gripper/result.json) and the unchanged [arm transport regression](../artifacts/jetson/gripper-execution/dds-arm-regression/result.json) remain separate from physical contact/stopping validation.

The [gripper gateway](../rammp_adl/motion/gripper.py) accepts calibrated aperture/speed/current bounds and current ownership/evidence, with no default physical table. The [measured backend](../rammp_adl/hardware_gripper.py) reuses the canonical `set_gripper` handler, checks empty-gripper and full aperture/stopping geometry dependencies before each publication, and commits only measured `aperture_reached` evidence. It does not infer grasp, retention or supported release. Hardware skill registration and dispatch remain disabled.

The driver extension holds the last measured normalized finger position after a commanded gripper halt; it does not open the fingers or increase the current ceiling. Its acknowledgement proves a successful transport exchange, not physical stopping or object retention. Once a gripper command has been admitted and then halted, a new ownership grant is refused until a new driver process is started. Ordinary arm-only release/reacquisition remains supported. A production gripper session reset/handback contract and physical hold characterization are still required; do not restart or adopt the installed physical driver as part of this simulation command.

### Rehearse moving GPU trajectory replacement

The [reactive probe](../rammp_adl/motion/reactive_probe.py) uses actual pinned cuRobo 0.7.8 MPC and MuJoCo, with no ROS or hardware port:

```zsh
python -m rammp_adl.motion.reactive_probe \
  --candidate artifacts/jetson/real-world-ready/rehearsal-final/planner/response-1.json \
  --request artifacts/jetson/real-world-ready/rehearsal-final/planner/request-1.json \
  --contract artifacts/jetson/real-world-ready/rehearsal-inputs/model-contract.json \
  --gpu-cache /home/abra/.cache/rammp-adl/real-world-ready-gpu \
  --wrapper-checkout /home/abra/.cache/rammp-adl/curobo-static/pinned-wrapper \
  --output artifacts/manual/reactive-gpu
```

Choose a new output directory. It preserves nonzero initial position/velocity/acceleration, checks interpolated trajectories independently and exercises a bounded 5 mm same-target correction. The full-horizon adapter returns the optimized horizon in one solve. Conservative optimizer acceleration settings address overshoot without relaxing independent limits or modifying the moving suffix. The initial stationary-start reference uses the pinned wrapper's consistent time dilation for the dynamics rehearsal.

The [accepted rolling rehearsal](../artifacts/jetson/independent-readiness/reactive-final-2/rolling-rehearsal.json) activated generation 1 while tracking continued during GPU work, then intentionally cancelled. It retained claims through planner drain and measured settling. The stop trajectory is still unavailable: cancellation explicitly escalates to a MuJoCo actuator hold and does not report ADL task success. This is bare-arm simulation with sampled geometry checks; it establishes neither continuous physical collision/stopping bounds nor driver suffix support. Earlier rejected runs remain in the adjacent numbered artifact directories.

The same command also saves `rolling-cancel-during-gpu.json`, cancelling while the actual solver call is still running. It discards that suffix and retains claims through solver drain and a fresh measured stationary dwell. Both GPU launchers now preserve a durable cache journal across process crashes; a later client cannot reuse a potentially active worker's cache merely because the old Python process exited. [Scoped acceptance checks](../artifacts/jetson/independent-readiness/reactive-final-2/acceptance.json) distinguish these results from physical stopping or task completion.

## Physical commissioning route

### Collect feedback without moving the arm

`record-driver-state` subscribes to an already-running driver's joint and EE publications. It starts no driver, creates no command or ownership client, and sends no stop or heartbeat. The current Jetson driver publishes `rammp_arm_interfaces/msg/EeState`; the older project-pinned build uses `kinova_gen3_interfaces/msg/EeState`. Select the observed interface explicitly. Both use the same inspected Header/Pose/Twist fields for this diagnostic adapter; this diagnostic selection does not approve physical command compatibility.

For the current Jetson deployment:

```zsh
cd /home/abra/RAMMP-generalized
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.zsh
source /home/abra/ros2_ws/install/setup.zsh
source .venv/bin/activate
ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
python -m rammp_adl record-driver-state \
  --joint-topic /joint_states --ee-topic /ee_state \
  --ee-message-type rammp_arm_interfaces/msg/EeState \
  --duration-s 30 --output-dir artifacts/manual/passive-state
```

Choose a new output directory for each recording. Raw paired samples and a statistical report remain local. The recorder bounds duration and memory, preserves original publication stamps, records both receipt times, and reports rejected/unpaired data. Its default one-second receipt timeout is diagnostic buffering configuration, not a hardware safety limit. Rates and gaps describe retained publications; acquisition age is unknown. Position variation during a stationary observation is not motion tracking error, and reported joint torque is not external contact force. Nothing from this command populates or enables a commissioning profile.

The [actual passive recording](../artifacts/jetson/passive-commissioning/live-state-2/report.json) captured 2,931 pairs at approximately 98.3 Hz, with a maximum receipt gap of 38.7 ms and zero rejected messages. Stopping latency/distance, dynamic tracking, watchdog effectiveness and hardware acquisition uncertainty remain unmeasured. [Implementation](../rammp_adl/motion/driver_diagnostics.py)

### Record camera and wrist observations together

The passive `record-calibration` command now records marker observations from both cameras alongside the existing custom driver's joint/EE publications. [Command, files and interpretation](runtime.md#passive-camera-wrist-calibration-recording). Leave the fixed fiducial in place. One recording is one labelled window; it neither moves the wrist nor proves calibration.

If the camera publishers are absent, these camera-only commands reproduce the reviewed local setup. Source `/opt/ros/humble/setup.zsh` in each terminal first. Start each command in its own terminal and allow both streams to become available before recording; stop these camera launches with Ctrl-C afterward. Do not launch or restart the robot driver as part of recording.

```zsh
ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=87 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=adl_capture camera_name:=d405 serial_no:=_260322274242 \
  enable_color:=true enable_depth:=true \
  depth_module.color_profile:=640,480,15 depth_module.depth_profile:=640,480,15 \
  align_depth.enable:=true enable_sync:=true publish_tf:=false pointcloud.enable:=false
```

```zsh
ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=87 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
ros2 launch orbbec_camera gemini_330_series.launch.py \
  camera_name:=adl_orbbec serial_number:=CPC99450003K \
  color_width:=1280 color_height:=720 color_fps:=15 \
  depth_width:=1280 depth_height:=720 depth_fps:=15 \
  enable_point_cloud:=false enable_colored_point_cloud:=false \
  depth_registration:=true align_mode:=SW publish_tf:=false \
  enumerate_net_device:=false enable_heartbeat:=false
```

These camera launches stay process-wide localhost-only, which is what actually confines imagery to the loopback interface. The recorder itself runs with the per-domain [locality document](../config/cyclonedds-local-imagery.xml) instead, because the custom driver container binds a routable interface and cannot be discovered from a localhost-only process. The recorder requires only subscriptions in domains 0 and 87. It keeps both candidate marker poses, model-derived wrist feedback and original clocks; receipt-near data remains unapproved. Multiple diverse poses, frame/model reconciliation and measured timing/uncertainty are still needed for robot-referenced calibration. Hardware commissioning and motion remain separate.

### Reuse the preliminary fixed marker

The user confirmed the marker will remain fixed and authorized the earlier observations as preliminary geometry. The fixed Orbbec observation now supplies a [marker-frame reference](../artifacts/jetson/passive-commissioning/fixed-marker-reference.json), retaining both planar pose branches, original capture stamps and nominal 50 mm scale. Reproduce it locally with:

```zsh
python -m rammp_adl.perception.fiducial fixed-reference \
  --observation artifacts/jetson/single-marker/orbbec/observation.json \
  --marker-fixed-confirmed --camera-fixed-confirmed \
  --output artifacts/manual/fixed-marker-reference.json
```

This exports `marker_from_camera` candidates for preliminary scene work. It does not make the marker frame `base_link` or refresh historical scene observations. The fixed reference is invalidated if the Orbbec, marker or its intrinsics change. Wrist-to-scene geometry still depends on wrist posture. No robot-base anchor was observed in the earlier capture, so `base_from_marker` remains unknown. This exporter does not connect calibrated scene geometry to the ADL executor.

The [running model audit](../artifacts/jetson/passive-commissioning/model-audit.json) found an existing wrist bracket reference but **D415** camera geometry and a fixed gripper chain. The installed D405 manufacturer model is available locally; its nominal case and internal offsets do not establish the actual wrist mounting transform. The running model is therefore not verified geometry for the D405 and moving 2F-85. Three bracket photographs have now been received and inspected. They show a white bracket supporting the D405 immediately in front of the built-in wrist camera beside the gripper base, with an attached cable. [Photo observations](../artifacts/jetson/passive-commissioning/bracket-photo-observations.json) resolve the missing-photo request; exact bracket dimensions, mesh identity and the wrist-to-camera transform remain unmeasured.

The [assembly preparation tool](../rammp_adl/motion/assembly.py) now bundles the official Gen3 and articulated 2F-85 meshes, an explicit frozen-aperture planner model, conservative collision spheres and a separate D405 component. Arm FK matched the legacy model in 500 seeded comparisons. The D405 remains detached until its actual mount transform is established; the bracket and cable envelope are unresolved. The driver's fixed gripper chain is deliberate for its seven-DOF dynamics model. See [reproduction and model limits](../artifacts/jetson/real-world-ready/assembly/REPRODUCE.md). These assets were not installed on the physical driver.

### Prepare a physical profile

Check readiness without starting a driver:

```zsh
python -m rammp_adl hardware-check --profile free_space
python -m rammp_adl hardware-template \
  --output artifacts/manual/hardware-profile-template.json
```

The template derives required fields from [canonical commissioning](../config/commissioning.json) and the implemented acquisition/transport types. Unknown measurements remain `null`. The template is not an approval and is intentionally inadmissible. Put a completed reviewed profile in `profiles.free_space` of the canonical configuration. `hardware_motion_enabled` remains false in this repository.

A physical profile requires the existing commissioning fields and the `free_space_test` extension, whose implementation is in [commissioning.py](../rammp_adl/motion/commissioning.py). In particular:

* State acquisition uncertainty, measured stopping/hold behavior, joint limits, tracking uncertainty, derivative bounds and watchdog deadline must be established for this robot. ROS publication timestamps and motor current are not substitute measurements.
* `collision_free_cell_lower_rad` / `upper_rad` describe a verified collision-free **joint region**, including all combinations within that region and the entire assembly in a secured static environment. One clear pose, a sampled simulation path or a confirmation button does not establish that region. Supporting local records and their hashes are required. The runtime reserves measured joint stopping excursions and tracking error inside it.
* `all_assembly_point_lever_bounds_m` bounds each joint's maximum lever arm to every relevant assembly point throughout that region. These bounds turn continuous joint-speed bounds into conservative full-assembly Cartesian-speed bounds. The evidence must justify them, including the gripper and wrist camera. They are not guessed link lengths.
* The profile pins the reviewed URDF, robot/planner configuration, loaded planner-model digest, driver extension build, static-world digest and the cell's supporting evidence. Evidence expiry must cover the move and stopping. Changes revoke the permit.

This mode assumes a secured static test cell. It does not provide person tracking, intended contact, gripper actuation, payload transport, or camera-grounded ADL execution. Those need their own implemented and commissioned capabilities.

Compile the physical-capable driver without launching it:

```zsh
python deployment/driver-hardware/build.py --mode hardware \
  --workspace /tmp/rammp-driver-physical \
  --kortex-sdk /home/abra/kortex_api_2.8.0_aarch64 --jobs 2
```

The build manifest contains the executable hash, pinned upstream revisions, SDK identity and `extension_build_id`. The SDK's entire archive was inspected as AArch64. It is still necessary to validate real feedback and stopping behavior; compiling it is not hardware validation. A build of the migrated interface packages already exists at `/tmp/rammp-driver-split-hardware` with its manifest under that workspace and a copy in the Jetson artifacts.

The physical driver must be deliberately started by the operator with the reviewed robot model, its connection setting, enforced arbitration and commissioned `heartbeat_timeout_s`. Its real-mode startup connects and switches to low-level servoing, so **starting it is itself hardware interaction**. The autonomy CLI never starts it, clears faults/e-stop, goes home or seizes an existing owner. The source-confirmed binary arguments are `--ip`, `--urdf`, optional `--ee-frame`, and ROS parameters `arbitration_mode:=enforced` / `heartbeat_timeout_s:=...`; do not copy simulation timeout/geometry values into a physical launch.

The test CLI requires a dedicated namespace, for example `/rammp_physical_test`. Add these ROS remappings to that deliberately configured driver launch; they isolate the selected instance's command and feedback paths:

```zsh
-r __ns:=/rammp_physical_test \
-r /execute_joint_trajectory:=/rammp_physical_test/execute_joint_trajectory \
-r /control_status:=/rammp_physical_test/control_status \
-r /acquire_control:=/rammp_physical_test/acquire_control \
-r /release_control:=/rammp_physical_test/release_control \
-r /estop:=/rammp_physical_test/estop \
-r /rammp/driver_feedback:=/rammp_physical_test/driver_feedback \
-r /rammp/driver_heartbeat:=/rammp_physical_test/driver_heartbeat
```

These are launch argument fragments, not a command that starts the robot. An already-running unpatched/global driver is not silently adopted.

Once that driver and a complete profile are available, create a local request JSON with exactly `goal` (`position_m`, normalized `quaternion_xyzw`) and `world` (the reviewed cuRobo obstacle records). The world must match the profile evidence. Keep all planner assets in an explicit model directory, with absolute paths that also work when mounted read-only into the GPU container. Preview a fresh, independently admitted path:

```zsh
python -m rammp_adl.motion.hardware_test \
  --profile free_space --namespace /rammp_physical_test \
  --request /path/to/local-request.json \
  --model-dir /path/to/reviewed-model \
  --wrapper-checkout /home/abra/.cache/rammp-adl/curobo-static/pinned-wrapper \
  --gpu-cache /home/abra/.cache/rammp-adl/hardware-test-gpu \
  --output artifacts/manual/physical-preview
```

The default only prepares the candidate and writes `review.json` plus local planner evidence. It does not acquire control or send a trajectory. No images or robot data leave the machine; the planner container has networking disabled.

For an intentional physical run, use the same command with a new output directory and `--execute`, after enabling the fully commissioned profile. It replans from fresh measured stationary state and validates the exact final ROS interpolation before requesting non-stealing control. A stale start, changed model/world/profile, expired evidence, ownership loss or unavailable watchdog refuses or stops the action. Ctrl+C requests cancellation; the driver heartbeat watchdog remains effective if the Python monitor stops. No stale reviewed trajectory is replayed just because a JSON file says it was validated.

Every run now also records bounded local acquisition samples and command/receipt events in `measurements/`. Recording happens in memory during monitoring and is written after monitoring ends. These samples support later commissioning analysis; they never generate or enable a profile. The terminal result records driver acknowledgement separately from fresh measured goal attainment and stationary dwell. Unresolved stopping retains authority and latches a fault; a successful action result alone never permits another move. If this standalone CLI must exit with unresolved ownership, the report explicitly records that local monitoring ends and the independent driver watchdog remains responsible; it never records a fictitious release. A physical emergency stop remains distinct from the ROS software-stop message.

The composition reserves sensor uncertainty inside start, goal and stationary-dwell tolerances before handing them to the driver gateway. It also rechecks the measured stationary acceleration bound after the control-claim round trip. A bounded acquisition estimate is kept distinct from a zero-derivative planning boundary.

### Run as a sheppy client

With the arm, planner and cameras up under sheppy, `sheppy up adl` starts the runtime as a client with motion disabled; it bootstraps `held_state` from `/joint_states` and registers no physical skill. Motion requires editing the `adl` node's command in the manifest by hand to `commissioned:=true` and `arm_motion:=true` and declaring the acknowledged gaps in `capabilities`; no shipped profile does this. Before starting it: physical e-stop within reach, the cell clear, and the startup log read for the acknowledged gaps it prints. Every trajectory is planned from a verified stationary start, gated against the driver's declared joint ranges and velocity limits, executed once with the effort guard armed after the first quarter of progress, and verified stationary at the goal. A guard trip cancels the goal and the skill fails as `stale_state` for a collision-guard trip or `safety_fault` for contact. See [runtime](runtime.md#sheppy-client-deployment) for the per-skill gates.

### Typed tasks on the bench

One-time inputs: `OPENAI_API_KEY` exported from the login shell profile, and the face-screen model present under [artifacts/models](../artifacts/models/README.md). Objects need no markers: put them on the bench in the wrist camera's view. Then, in order:

```bash
# 1. Astra plans the fixture cabinet offline; no robot, no cameras.
rammp-adl simulate --reasoner astra --task "open the cabinet door"
# 2. The stack, motion disabled. Read the startup log: scene, gaps, unavailable skills.
cd /home/abra/rammp-deployments/december_2026 && sheppy up adl --manifest sheppy-manifest.yaml
# 3. A typed task with nothing registered exercises intake up to the goal:
#    bootstrap, visibility, then NEED_CAPABILITY because no physical skill is commissioned.
ros2 action send_goal /rammp/execute_task rammp_adl_interfaces/action/ExecuteTask \
  "{task_id: '', task_text: 'pick up the water bottle', plan_json: ''}" --feedback
# 4. Arm it by hand in the manifest command: commissioned:=true arm_motion:=true and the
#    acknowledged gaps in capabilities (the full list is in runtime.md). sheppy down; sheppy up adl; repeat 3 with the e-stop in hand.
```

For a door: type `open the cabinet door in front of you`. The cabinet need not be in view: if Astra does not see it, the arm pans, tilts or backs off as Astra suggests and looks again, up to six times, before intake gives up with `NEED_OBSERVATION`; those are real transits, so the e-stop applies from the first phase. Intake adds `INTAKE_PROPOSING_CONSTRAINT`; the node logs the constraint it installed (kind, hinge side, opening, width, effort budget, parameters version). The plan is observe, pregrasp, preshape, grasp transit, grasp, follow_constraint, release, retract. During follow_constraint the arm moves in short stop-and-go steps; a contact trip ends the task as `model_mismatch` with the achieved angle recorded, and the next attempt starts from that history. Before the final approach the arm may shift sideways a few centimetres at the standoff once or twice: that is the model aligning the fingers to the handle from the wrist image; the log prints each shift. Watch the tool: the fingertips must stay on the handle throughout; if they slide, stop with the e-stop and lower `contact_speed_scale`. A follow that ends `goal_unobserved` means the tool moved but the door face did not turn with it. The transit speed is 0.4 of the planner's and contact steps run at 0.25 by default; both are node parameters.

Expected on the first armed pick: `INTAKE_DISCOVERING` (one keyframe to Astra, entities named from it), `INTAKE_BOOTSTRAP`, `INTAKE_NORMALIZING_GOAL`, a goal such as `holding(water_bottle_1)`, then observe, the pregrasp and grasp transits, the gripper preshape and the grasp, each planned from a verified still start and stopped by the effort guard on contact. Watch the startup log for the scene line: it says whether the face screen loaded; without it no keyframe is sent and intake returns `EGRESS_REFUSED`. Every frame that goes to Astra is kept as a JPEG under `artifacts/keyframes/` (the last forty), so when discovery names nothing you can see exactly what the wrist camera showed.

## Remaining engineering

The command path and measured observation/aperture seams are implemented. Full autonomous physical ADL execution still needs complete installed-assembly geometry/calibration, physical commissioning evidence, live semantic/collision scene construction, the remaining measured physical skill backends, constrained/contact planning, generated stopping paths and a validated rolling driver protocol. The current static trajectory transport explicitly does not advertise rolling replacement. Camera capture and the single-marker provisional relation do not establish wrist/base or scene/base calibration. Offline extrinsic fitting and recorded-camera/Astra observation rehearsal are now available in the [runtime runbook](runtime.md); their outputs remain explicitly outside physical calibration/commissioning admission.
