# Runtime runbook

Work in `/home/abra/RAMMP-generalized` on the Jetson; the Windows backup is not synchronized automatically. See [Jetson setup](jetson.md) for environment and dated evidence. Unit, camera, provider, GPU, ROS simulation and physical validation are separate checks.

The Python runtime executes validated task DAGs with resource ownership, measured-effect commits, bounded recovery and local supervision. Current ADL demonstrations use a deliberately synthetic scene/backend. MuJoCo runners exercise Gen3 joint dynamics, rolling handoff and retained static cuRobo planning through the guarded Python gateway. The integrated rehearsal has synthetic target geometry and no ROS transport; real DDS is a separate simulation test. A guarded physical ROS trajectory transport is now implemented behind a dedicated free-space commissioning composition; the ADL node remains fixture-only. See [hardware testing](hardware-testing.md) for prerequisites, the available ROS test, and physical execution commands.

## What runs today

| Component | Implemented and exercised scope | Remaining integration |
|---|---|---|
| [Catalog/registry](../rammp_adl/contracts.py) | Strict JSON, full JSON Schema, canonical catalog hash, six explicit fixture handlers, capability filtering | Hardware entries remain planned and disabled |
| [World/validator](../rammp_adl/world.py), [plan validation](../rammp_adl/validation.py) | Immutable snapshot data, typed metric poses, trusted evidence, ancestor-dependent preconditions, state conflicts, exact dispatch receipts and atomic effects | Live scene reconstruction, commissioned geometric/safety evaluators and cuRobo swept-path checks |
| [Executor/safety](../rammp_adl/executor.py), [supervision](../rammp_adl/safety.py) | Parallel disjoint skills, task ownership across cloud waits, bounded retries/replans, cancellation, verified quiescence; separate driver gateway has ownership and a C++ heartbeat watchdog | Physical stop/hold commissioning and integration of physical skill handlers |
| [Astra gateway](../rammp_adl/reasoning.py) | Responses adapter, context-bound profile choices, bounded requests/replan feedback, malformed/refusal/error handling; live Astra plans validated in the synthetic cabinet and recorded-marker observation rehearsal | Robot-referenced semantic/collision scene integration and provider reliability across tasks |
| [Perception](../rammp_adl/perception/) | Camera buffers/adapters, bounded ROS RGB-D subscriptions and local captures, provisional single-marker camera poses, calibration admission, timestamp checks, crop provenance, local depth geometry, plane/prismatic/revolute fitting and track storage | Verified rectification, capture-clock mapping, robot-referenced extrinsics and a connected task-relevant perception pipeline |
| [Rolling motion](../rammp_adl/motion/rolling.py), [async session](../rammp_adl/motion/session.py) | Typed trajectories, actual GPU moving-horizon/MuJoCo rehearsal, concurrent planning/tracking, q/dq/ddq handoff, target invalidation and worker drain | Constrained/contact and stopping validation; physical driver suffix replacement |
| [Simulation](../rammp_adl/simulation.py), [integrated rehearsal](../rammp_adl/motion/rehearsal.py) | Synthetic ADL state transitions; Gen3 MuJoCo joint replay, moving suffix activation and retained cuRobo static paths through the guarded Python gateway, including fault injection | A cuRobo-planned robot interacting with cabinet/gripper/payload/contact physics |
| [ROS transport](../rammp_adl/ros_node.py), [driver gateway](../rammp_adl/motion/driver_transport.py) | Humble compilation, actual DDS SimTransport trajectory/ownership/cancellation/watchdog tests; physical-capable driver compiled | Measured physical execution; ADL ROS node still fixture-only |

The available fixture skills are `observe`, `move_to_pose`, `set_gripper`, `grasp`, `release` and `follow_constraint`. The catalog remains the only definition of their arguments and policies.

A task context must already contain a trusted normalized goal and initial stable entity/profile/constraint bindings. The current CLI selects such a context from a fixture or an explicit file. Natural-language text does not automatically construct a metric scene or replace that goal. Astra grounding refines a known entity reference; it does not turn an RGB detection or depth point into a verified grasp pose. The ROS node does not automatically subscribe to every attached camera or ingest preview images into its grounding registry.

A separate [local observation and commissioning seam](../rammp_adl/hardware_backend.py) binds measured poses, execution identity and world dependencies to existing handlers and transport. The marker adapter now feeds this observation seam; a separate calibrated gripper backend reuses `set_gripper`. Neither registers an enabled physical motion skill. The six hardware catalog entries remain planned. The [hardware runbook](hardware-testing.md) contains the current integrated GPU/physics rehearsal and isolated ROS commands.

## Install and check

Use the existing Jetson Python 3.10 environment from Zsh. For fresh installation instructions, see [Jetson setup](jetson.md).

```zsh
cd /home/abra/RAMMP-generalized
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.zsh
source .venv/bin/activate
python -m rammp_adl doctor
python -m rammp_adl registry
python tools/check_design.py
python -m unittest discover -s tests -v
```

`doctor` reports dependency availability and whether `OPENAI_API_KEY` is configured; it does not print the key or establish provider/device access. The base package installs runtime JSON Schema/numerical dependencies. Extras add the SDK, images/camera tooling and MuJoCo. Vendor RGB-D SDKs and ROS/cuRobo are separate installations.

The design checker remains dependency-free and checks generated artifacts, references and fixture grammar. Runtime tests require the installed dependencies. Some physics tests may be skipped when MuJoCo is absent; inspect the test output before claiming physics coverage. After a catalog edit, regenerate with `python tools/check_design.py --generate`, then run the design and runtime checks. Do not change generated schemas independently.

## Synthetic ADL execution

Run the cabinet DAG through the real scheduler/world/validator against synthetic task-state evidence:

```zsh
python -m rammp_adl simulate --scenario cabinet --output artifacts/cabinet.json --trace artifacts/cabinet-trace.jsonl
```

This demonstrates dependency execution, overlapping empty-gripper preshaping and transit, evidence commits, retention/support logic and the normalized goal check. It does not test cabinet contact physics, vision-driven object discovery or cuRobo reachability.

Inject one locally recoverable observation failure:

```zsh
python -m rammp_adl simulate --scenario cabinet --inject c1:no_detection --output artifacts/observation-retry.json --trace artifacts/observation-retry-trace.jsonl
```

Inject an articulation-model failure after partial progress:

```zsh
python -m rammp_adl simulate --scenario cabinet --inject c6:model_mismatch --output artifacts/partial-failure.json --trace artifacts/partial-failure-trace.jsonl
```

The latter run is expected to return exit code 2 with an incomplete task, committed partial evidence and held state. It must not release the retained handle or report the original opening goal satisfied. The fixture CLI executes one supplied plan; the executor's bounded task-level replan path is exercised separately by integration tests and by Astra mode.

`--context PATH` and `--plan PATH` select explicit contracts. `--time-scale` scales synthetic skill duration only; it does not change robot speed, physics time or safety limits. The bottle-to-mouth example remains a design reference and is rejected by the current runtime because user transfer is deferred.

## Rolling updates and MuJoCo

Run the receiver protocol against explicit joint test stimuli:

```zsh
python -m rammp_adl rolling --output artifacts/rolling.json
```

Run actual MuJoCo Gen3 joint dynamics and optionally render the model:

```zsh
python -m rammp_adl physics --output artifacts/physics.json --image artifacts/gen3-simulation.png
python -m rammp_adl physics --rolling --output artifacts/rolling-physics.json
```

The rolling physics report records a future suffix activation against simulated joint state, derivative/activation errors and final tracking error. Trajectories in these commands are explicit test inputs, not generated arm plans. No gripper, bottle, cabinet contact or human interaction is validated by these joint replays. Model source/license provenance lives under [simulation/vendor](../simulation/vendor/).

The [cuRobo adapters](../rammp_adl/motion/curobo.py) provide two distinct paths. `RammpCuroboAdapter` verifies the pinned RAMMP checkout/import and supports static planning; that wrapper rejects moving starts. `CuroboMpcAdapter` implements optional cuRobo 0.7.8 moving-state candidate generation and requires an actual GPU selfcheck plus independent validation before use. Both GPU paths have now run on this Jetson in their documented simulation scopes and expose no physical capabilities. The current [reactive rehearsal](hardware-testing.md#rehearse-moving-gpu-trajectory-replacement) records actual MPC replacement and its remaining stopping limitation.

`RollingMotionSession` connects candidate generation and independent validation to a concurrent tracking loop through explicit local simulation callbacks. It retains the active skill's complete catalog resource set, validates future moving-state boundaries, rejects obsolete targets, and stops before a late worker releases its lease. Scheduling tests use a clearly labeled joint fixture planner and ideal tracking. The session is a backend integration component; the fixture DAG does not silently acquire real cuRobo capability. Successor speculation remains disabled while the catalog reserves PLANNER for each whole action. See [performance requirements](design/14-performance.md).

An additional [static integration runner](../rammp_adl/motion/integration.py) can pass a real pinned cuRobo trajectory through independent MuJoCo model/FK/joint/contact checks and physics replay in a compatible GPU environment:

```zsh
python -m rammp_adl.motion.integration --describe-model
```

Its execution arguments are `--source-root`, `--planner-config`, `--contract` and `--goal`. The closed local manifest is defined by `load_contract` in that module: it binds source/config/robot/URDF/model hashes, exact joint order and frames, and explicit simulation criteria. The goal file contains `position_m` and `quaternion_xyzw`. A compatible planner model/URDF must be supplied; the default gripper/tool configuration cannot be assumed to match the bare Gen3 model. Wiring tests use a mocked upstream planner API with actual MuJoCo dynamics and report that distinction.

The [isolated Jetson static probe](../deployment/curobo-static/README.md) now supplies an explicitly reconciled bare-arm model and exact source/package provenance in a separate image. It generated an actual 31-point cuRobo trajectory for a 3 cm simulation target in approximately 0.2415 s after warmup, with fifteen independent FK comparisons passing. Independent admission rejected its nonzero start/terminal derivatives before physics replay. [Candidate and rejection evidence](../artifacts/jetson/curobo-static/attempt-3/probe/result.json) record that initial rejection. The stationary boundary defect was subsequently fixed in the planner interpolation and the [integrated rehearsal](hardware-testing.md#rehearse-gpu-planning-and-monitored-motion-now) passed; historical rejected candidates remain unchanged. Neither result establishes physical robot readiness or a continuous swept/stopping proof.

Timing JSON and JSONL traces describe this local process. Use their wait reasons and overlap events to find software pauses, but do not extrapolate fixture durations or workstation joint replay to Jetson throughput or safe physical speed.

## Cameras and local imagery

Probe local cameras and optionally capture one local RGB preview:

```zsh
python -m rammp_adl cameras --output artifacts/cameras.json
python -m rammp_adl cameras --capture-index 0 --image artifacts/camera-preview.png --output artifacts/cameras.json
```

An earlier Windows preview used a Brio 105 and did not establish depth-camera access. On the Jetson, both D405 and Orbbec ROS captures are now verified again after reconnection: each detected marker ID 0 in 30/30 frames at approximately 15 Hz. See the latest [camera and observation replay results](../artifacts/jetson/camera-restored-2026-09-14/results.json). Capture alone does not approve metric geometry.

Local capture does not upload anything to Astra. Provider imagery requires an explicitly registered minimized crop with capture identity, source calibration and crop-to-original mapping. Raw depth, point clouds and continuous video remain local. An OpenCV receipt timestamp is insufficient for synchronized metric geometry. Configure actual serial IDs, RGB/depth alignment, intrinsics, capture-clock mapping and camera-to-base/tool calibration before using metric observations. The generic ROS camera buffer accepts already aligned/calibrated input; it does not guess Orbbec topics or calibrate the cameras automatically.

### Live ROS RGB-D on the Jetson

The [ROS RGB-D source](../rammp_adl/perception/ros_rgbd.py) now connects explicitly selected `Image` and `CameraInfo` topics to a bounded local capture path. It starts subscriptions only. Inspect any camera driver launch before running it. The camera-only launches and actual topics used on this Jetson are recorded in [Jetson setup](jetson.md). They must run with the same `ROS_DOMAIN_ID` and `ROS_LOCALHOST_ONLY=1` as the subscriber; setting localhost only on the subscriber does not constrain another process's publishers.

With the checked D405 driver already running:

```zsh
export PYTHONNOUSERSITE=1 ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=87
source /opt/ros/humble/setup.zsh
source .venv/bin/activate
python -m rammp_adl capture-rgbd \
  --rgb-topic /adl_capture/d405/color/image_raw \
  --depth-topic /adl_capture/d405/aligned_depth_to_color/image_raw \
  --rgb-info-topic /adl_capture/d405/color/camera_info \
  --depth-info-topic /adl_capture/d405/aligned_depth_to_color/camera_info \
  --reliability reliable --samples 30 --timeout-s 15 \
  --output-dir artifacts/jetson/d405-new-capture
```

The output directory must be new. The command saves the last pair to local `capture.npz` (RGB, depth in metres, original stamps and CameraInfo metadata) and writes `report.json`. It reports missing messages, rejected frames, queue drops, pair delivery rate and wait time; incomplete capture returns exit code 2. `frames_transmitted=0` means no external image egress; local DDS messages are received. No image is printed, uploaded, registered with Astra or admitted to the world model by this command. Saved observations are historical evidence and cannot be made fresh by assigning them a new timestamp.

The source keeps its own ROS context/thread warm while the consumer works, retains bounded queues and only the latest completed pair, and joins before teardown. It checks increasing nonzero acquisition stamps, pair skew, image/CameraInfo frame and dimension agreement, row stride, byte order and supported encodings (`rgb8`, `bgr8`, `16UC1` millimetres, `32FC1` metres). Invalid depth remains unknown. Subscription QoS defaults to best effort for compatibility. Use `--reliability reliable` only with confirmed reliable publishers: on this Jetson that change restored D405 delivery from about 1.5 to 15 paired frames/s in short diagnostic runs, without changing freshness or pairing bounds. This is a measured camera transport observation, not an autonomy throughput benchmark.

`RgbdPair` is deliberately separate from `CameraFrame`. A trusted composition can wrap `RosRgbdSource` in `MetricRgbdAdapter` and feed the existing `PerceptionLoop`/`CaptureRegistry`, but must first supply `RgbdCalibration`, a characterized capture-to-monotonic clock mapper, and explicit freshness/uncertainty bounds. The calibration binds camera/frame identity, both CameraInfo hashes, and externally verified alignment, rectification and clock evidence. IDs/hashes detect identity changes; they do not prove calibration accuracy. The adapter checks both exposure times and rejects changed calibration, incompatible projection, excessive uncertainty and stale frames. Its current pinhole seam rejects ROI/binning/stereo transforms that need a separate verified adapter.

The live captures remain diagnostic: both color streams report nonzero distortion. Orbbec color and depth CameraInfo report different distortion/ROI metadata despite common frame names, which also needs verification. Matching dimensions, a driver alignment setting or close header stamps do not establish spatial alignment, rectification, clock accuracy or camera-to-base geometry. No calibration IDs, extrinsics, object poses or task facts were invented. Semantic association, calibrated transforms and locally validated pose estimation are still required before these observations can ground arm planning. Tests cover calibrated admission using explicit synthetic evidence plus local DDS transport with fabricated images; that test is separate from the real camera reports.

### Local calibration preparation

The [target utility](../rammp_adl/perception/calibration_target.py) uses OpenCV's ChArUco generator/detector and the local [target specification](../config/calibration-target.json). It saves physically sized PDF pages and 2D observation reports; it does not output robot transforms or a calibration approval. See [OpenCV's ChArUco documentation](https://docs.opencv.org/5.0/tutorials/objdetect/charuco_detection/charuco_detection.html).

```zsh
python -m rammp_adl.perception.calibration_target generate \
  --output-dir artifacts/jetson/new-calibration-target
python -m rammp_adl.perception.calibration_target inspect-capture \
  --capture artifacts/jetson/d405-new-capture/capture.npz \
  --output artifacts/jetson/d405-target-observation.json
```

Both destinations must be new. The second command reads an existing local `capture-rgbd` archive and preserves its acquisition timestamps, capture identity and CameraInfo digest. Output includes detected corner IDs/pixel coordinates and whether correspondences are noncollinear; neither a detected board nor a detector success flag certifies calibration. Unknown/partial visibility remains explicit. No local camera imagery is uploaded. Installed OpenCV must supply `aruco.CharucoBoard` and `aruco.CharucoDetector`; no dependency or algorithm fallback is selected silently. The printing instructions and current mounting information are in [Jetson setup](jetson.md).

### Provisional calibration with the available single marker

The user-selected [marker specification](../config/calibration-marker.json) is `DICT_4X4_50`, ID 0, with a nominal **50 mm black outer edge** from its label; the white margin is excluded. A larger board is optional for later refinement. The [single-marker observer](../rammp_adl/perception/fiducial.py) uses raw color `CameraInfo` K/D and OpenCV `SOLVEPNP_IPPE_SQUARE`, retaining both positive-depth pose branches and their pixel reprojection errors. It does not recalibrate intrinsics or use aligned depth to bypass uncertain color/depth geometry. Unsupported intrinsics, duplicate matching IDs and degenerate corners cannot produce an accepted pose observation. See [OpenCV's pose-estimation documentation](https://docs.opencv.org/4.9.0/d5/d1f/calib3d_solvePnP.html).

Add `--inspect-marker` to `capture-rgbd` to reuse one detector throughout the bounded capture. `report.json` then includes each observation and detection counts; `observation.json` corresponds to the last saved `capture.npz`. A successful RGB-D capture alone does not mean the marker was detected: check `observation_summary` and the observation status. Source stamps, capture identity and camera/marker definition hashes remain attached to the numeric results. Images stay local.

Existing captures can also be inspected without running a driver:

```zsh
python -m rammp_adl.perception.fiducial inspect-capture \
  --capture artifacts/jetson/single-marker/d405/capture.npz \
  --output artifacts/jetson/d405-marker-reinspection.json
python -m rammp_adl.perception.fiducial relative \
  --first-observation artifacts/jetson/single-marker/d405/observation.json \
  --second-observation artifacts/jetson/single-marker/orbbec/observation.json \
  --stationary-setup-confirmed \
  --output artifacts/jetson/new-relative-marker-candidates.json
```

Outputs must be new. The stationarity flag records an operator statement that both cameras and the same marker remained stationary between captures; it does not establish clock synchronization. The user confirmed that condition for the saved `single-marker` captures. Composition is `second_camera_from_marker @ inverse(first_camera_from_marker)`, preserving all planar branch combinations. The marker frame has its origin at the center, x right, y up and z out of the printed face. Candidate ordering by summed pixel error does not resolve ambiguity or approve calibration.

These provisional estimates depend on nominal print scale and factory intrinsics. They never create `CalibratedTransform`, publish TF, admit metric observations or enable motion. The camera-to-camera relation applies only at the captured wrist posture and is invalid after wrist/base/camera motion; it does not establish wrist-link or robot-base extrinsics. [Live results and remaining calibration dependencies](jetson.md#single-marker-capture-follow-up) are recorded separately from synthetic numerical tests.

### Recorded camera through task reasoning and the world model

The [observation rehearsal](../rammp_adl/perception/observation_replay.py) detects the configured marker in a local archive, constructs a scoped marker entity, validates a canonical observation plan and commits measured visibility through the existing executor and world writer. The default plan is local; `--reasoner astra` uses the existing text-only cloud gateway. No images or marker pose branches are sent to the provider.

```zsh
python -m rammp_adl.perception.observation_replay \
  --capture artifacts/jetson/single-marker/orbbec/capture.npz \
  --camera-role scene --reasoner astra \
  --output-dir artifacts/manual/recorded-marker-astra
```

Use a new output directory. The report, exact initial context, proposed plans, reasoning records and trace stay local. Omit `--reasoner astra` for an offline run; supply `--plan` only with local mode to inspect a separately written canonical DAG. This composition has only `observe` available. It uses an explicitly frozen playback clock and a synthetic held-state source, preserving the original capture stamps. It creates no robot-frame pose, collision geometry or motion transport. A table marker is not automatically a cabinet handle. Both offline and actual Astra runs succeeded; [evidence](../artifacts/jetson/independent-readiness/camera-world/observation-replay-results.json) distinguishes these tests from live camera operation.

The reusable [local marker adapter](../rammp_adl/perception/measured_marker.py) also accepts live `RgbdPair` objects through `on_pair` or its bounded warm capture loop. Metric admission additionally requires an exact marker-to-entity binding, independently resolved planar branch, calibrated camera-to-base transform, characterized capture clock and explicit final pose covariance. A wrist camera cannot use a static camera-to-base transform. Source identity/intrinsics changes and capture-clock regressions latch rejection; stopping the source immediately revokes new observations. No such physical transform or uncertainty was inferred from the existing archive.

### Passive camera/wrist calibration recording

`record-calibration` collects one labelled observation window from existing camera and custom-driver publishers. It reuses the local marker detector and strict joint/EE adapter. The [subscription configuration](../config/calibration-recording.json) selects the current driver in ROS domain 0 and both cameras in domain 87 through separate contexts; it never changes the process's domain setting, starts a driver or constructs a motion client. Camera detection stays warm on independent workers while robot subscriptions continue.

With the camera-only publishers running under the topic names in that configuration:

```zsh
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.zsh
source /home/abra/ros2_ws/install/setup.zsh
source .venv/bin/activate
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
CYCLONEDDS_URI=file://$PWD/config/cyclonedds-local-imagery.xml \
python -m rammp_adl record-calibration \
  --pose-label pose-01 --duration-s 15 \
  --output-dir artifacts/manual/calibration/pose-01
```

Use a new output directory for each recording. The [camera setup commands](hardware-testing.md#record-camera-and-wrist-observations-together) start only the reviewed camera packages. The recorder alone starts no publishers. For another deployment, supply `--config` with explicit absolute topics, domain IDs and the inspected EE interface type. Missing streams produce an `incomplete` report and exit code 2, retaining any received data. Ctrl-C also saves an incomplete record and closes the recorder's subscriptions.

Imagery locality is resolved per ROS domain. Every camera domain must be confined to the loopback interface, either by process-wide `ROS_LOCALHOST_ONLY=1` or by a `CYCLONEDDS_URI` document that pins that domain to 127.0.0.1; [config/cyclonedds-local-imagery.xml](../config/cyclonedds-local-imagery.xml) is the generated document for the recorded camera domains. The driver domain carries no imagery, so it may keep the routable interface its publisher already binds. On this Jetson that separation is required: `ROS_LOCALHOST_ONLY=1` pins Cyclone DDS to `lo`, which is not multicast-capable here, while the custom driver container selects the first non-loopback multicast-capable interface. When both mechanisms are present they must agree, because the middleware prepends its own stanza to the same configuration list and a later stanza can reselect a routable interface. Confinement is a claim about advertised DDS locators, not a packet capture, and it does not constrain a publisher another process started: launch camera drivers with the same document or with `ROS_LOCALHOST_ONLY=1`. Choosing another domain does not bridge a remote host. Terminal receipt checks use each camera's actual recording cutoff when its explicit sample cap is reached, and separately report age at the overall session cutoff. They do not count an intentionally completed camera recording as a later stream dropout.

`report.json` records the resolved per-domain locality evidence, with the mechanism, interface selectors and configuration hashes for each domain. It also contains source mappings, publisher discovery, source rejection/drop statistics, raw joint variation, per-camera observation/association counts and file hashes. `driver.jsonl` preserves every retained same-publication joint/EE pair, including original integer nanosecond stamps, both monotonic receipt times, empty frame names and model-derived pose/twist. Each camera folder has `observations.jsonl` (every detected result, both IPPE branches, corners and original RGB-D/CameraInfo metadata), `associations.jsonl` (indices into the bracketing driver records) and `last-capture.npz` (only the last local RGB/depth image pair). Camera imagery remains local.

`receipt_bracketed` means two driver publications bracket the RGB callback receipt within the configured diagnostic receipt gap (default 0.1 s). It does **not** establish common acquisition time. The recorder never interpolates poses, selects a marker branch or converts an empty driver frame into `base_link`. Cross-source header deltas are recorded with their clock relation explicitly unknown. Changed camera intrinsics/frames, clock regressions, source rejections or ambiguous terminal publisher discovery invalidate associations. Duration, storage caps and receipt gaps are diagnostic settings, never safety limits.

`recorded_unapproved` means raw evidence was collected successfully. Every output still has zero admitted calibration samples and unknown clock uncertainty, robot model/frame agreement, base epoch and marker-placement identity. A pose label neither certifies stationarity nor turns many frames of one posture into independent hand-eye samples. The recorder's [raw format and implementation](../rammp_adl/perception/calibration_recording.py) is intentionally distinct from the solver input below. Later, operator-controlled stationary windows at multiple diverse wrist orientations, reviewed frame/model identities, timestamp bounds and independent branch resolution are required before fitting. No autonomous repositioning is part of this command.

### Offline wrist-camera extrinsic candidates

The [offline solver](../rammp_adl/perception/extrinsics.py) accepts 5–128 paired `base_from_wrist` and `camera_from_marker` measurements. It solves `base_from_wrist * wrist_from_camera * camera_from_marker = base_from_marker`, rejecting insufficient independent rotation axes, poor translation conditioning, changed identities, excessive pairing uncertainty and inconsistent fixed-marker closure. Each square-marker branch must already have independent resolution evidence. It creates an **unapproved candidate**, never an admitted `CalibratedTransform` or a TF publication.

The exact versioned input contract is validated by that module. An explicitly synthetic input illustrating its fields is [synthetic-paired-poses.json](../artifacts/jetson/independent-readiness/extrinsics/synthetic-paired-poses.json). Its values are a software fixture, not calibration records for this robot. Run it without any camera or robot access:

```zsh
python -m rammp_adl.perception.extrinsics \
  --input artifacts/jetson/independent-readiness/extrinsics/synthetic-paired-poses.json \
  --output artifacts/manual/synthetic-hand-eye-candidate.json
```

The optional `calibration` dependency provides SciPy. Input `timestamp_uncertainty_s` must bound the combined error of both capture clocks and their mapping; the solver adds it once to the time difference. Named robot/model/intrinsics/marker/placement/base/clock identities remain attached to the output. Fit residuals describe consistency and cannot become physical uncertainty estimates. The fixed-camera composition also preserves its own intrinsics and fixed-mount evidence and rejects the wrist camera. Actual multi-pose measurements and uncertainty are still missing; one stationary shared-marker view cannot supply them. This utility issues no robot command.

## Astra task reasoning

The selected model and bounds are in [config/reasoning.json](../config/reasoning.json). Configure `OPENAI_API_KEY` through local credential/environment setup, then run:

```zsh
python -m rammp_adl simulate --scenario cabinet --reasoner astra --task "Open the cabinet door" --output artifacts/astra-cabinet.json --trace artifacts/astra-cabinet-trace.jsonl
```

This requests an Astra plan for the selected synthetic context and executes only locally admitted fixture skills. Basic access was first verified by a small text-only [request](../artifacts/jetson/calibration-preparation/astra-text-probe.json). A subsequent [live cabinet run](../artifacts/jetson/astra-feedback/after-astra-cabinet.json) completed seven model-selected nodes on the first plan, with no task replans. It used synthetic scene information and no images. That single successful run does not establish camera grounding, provider reliability or robot readiness. Mock transport tests establish fault-handling behavior separately.

Profile IDs in each skill's proposal schema are restricted to compatible existing context profiles using the canonical catalog safety class. Rejected/incomplete attempts now pass bounded diagnostic feedback to the next task plan, and exhausted replan budgets preserve the last cause in the final result. The original user run's four rejected proposals were caused by mismatched profile safety classes; neither increased retry budgets nor weaker local validation were needed to correct that failure.

Timing reports now measure first-motion latency from the local task request, including initial cloud planning; process startup is excluded. Older reports started that metric at plan admission and could omit the cloud wait. The successful live run's unchanged trace gives approximately 21.23 s to first simulated motion, rather than its original 0.032 s post-cloud figure. [Corrected historical timing](../artifacts/jetson/astra-feedback/after-astra-timing-corrected.json) preserves the original report and trace.

Every request is built the same way in [reasoning.py](../rammp_adl/reasoning.py): one persona (decide like a competent technician, commit to the likeliest reading, answer only in the schema, free text under 15 words), one short rule block for the kind of decision, and a compact scene brief: IDs, labels, pose roles, the facts currently true, valid constraints, profiles by class, hazards and completed nodes. Evidence IDs, ages, revisions and calibration identities stay local because they cost tokens and carry no decision. Skill cards (needs, gives, claims) are generated from the catalog, not written by hand. All requests run at low reasoning effort; planning keeps the larger output bound and everything else is capped at 2000 output tokens. Measured on 2026-09-21 against the live provider: a door plan took 6.1 s with 1487 input and 285 output tokens at low effort against 10.8 s and 349 output tokens at medium, both valid on the first attempt; goal binding took 3.8 s (965 in, 66 out) and discovery 13.3 s (703 text tokens in plus the image, 335 out). These are single requests, not a reliability figure.

Task reasoning runs while held at task start and admitted replan events. A skill that owns the arm may also ask (alignment before a grasp, the progress score after a follow): there the hold it must show is the arm measured still for the dwell, since the backend is by definition not idle. Local rolling motion updates do not call Astra. Whole-task ownership and the ROS bridge's held resource lease prevent cloud waits from racing another action into motion. Refusal, budget exhaustion, malformed output, unsupported capabilities and transport failures remain typed outcomes; there is no fallback model or direct model-to-robot command path.

## ROS Humble and container recipes

The read-only [driver-state adapter](../rammp_adl/motion/driver_state.py) now subscribes to explicit upstream `joint_states` and `ee_state` topics through a caller-owned ROS node. It checks exact arm joint identities, reorders by name, separates the optional Robotiq joint, preserves publication and local receipt times, and rejects stale/out-of-order/malformed observations. The bounded buffer pairs the driver's common publication tick. It creates no command clients or publishers and does not launch a driver.

These records remain diagnostic: upstream ROS stamps are publication times, hardware acquisition age is unknown, acceleration is absent and the EE pose comes from unverified driver model/FK. `require_motion_state()` refuses conversion into a motion-state certificate. A freshly received message or small reported velocity cannot establish a fresh measured hold. [Generated-message and fabricated local DDS evidence](../artifacts/jetson/driver-state/results.json) is separate from real robot-state validation. The test-only upstream IDL build is at `/tmp/rammp-driver-state-idl/install`; source its `setup.zsh` after ROS Humble to instantiate this subscriber, or provide a separately verified build of the pinned `kinova_gen3_interfaces` package.

The two build packages are [interfaces](../interfaces/) and [ros2/rammp_adl_runtime](../ros2/rammp_adl_runtime/). The latter installs the same Python implementation and canonical assets into its package share. Both packages have compiled on the Jetson, including the additive gripper halt feedback fields. Building still does not start the physical robot; use a separate build/install prefix for isolated verification.

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[simulation,reasoning,cameras]'
colcon build --base-paths interfaces ros2/rammp_adl_runtime --merge-install --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash
ros2 interface show rammp_adl_interfaces/action/ExecuteTask
ros2 launch rammp_adl_runtime simulation.launch.py enable_astra:=false
```

For a separate container-based Humble recipe, [deployment/Dockerfile.simulation](../deployment/Dockerfile.simulation) provides:

```zsh
docker build -f deployment/Dockerfile.simulation -t rammp-adl-sim .
docker run --rm --network none rammp-adl-sim check
docker run --rm --network none rammp-adl-sim launch
```

The container recipe has not been built or run here. Its runtime commands require no robot device passthrough, host networking or GPU access. The launch constructs the fixture backend and fixes hardware motion off; the node independently rejects attempts to enable hardware.

Use ExecuteTask for whole DAG execution. The direct ExecuteSkill compatibility path requires admission plus fresh dispatch validation, consumes a node attempt and retains one manual plan per epoch. The public CommitEffects endpoint is exact duplicate-receipt lookup only; trusted in-process handlers perform initial commits. UserConfirm reports unavailable until a real assent provider is connected. ROS endpoint names and message definitions are in [source](../rammp_adl/ros_node.py) and [canonical IDL](../interfaces/), rather than a second editable contract table here.

## Sheppy client deployment

The runtime runs as a client of the sheppy arm module. It starts none of the containers; it subscribes to `/joint_states`, plans through `/rammp_curobo`, and sends `/execute_joint_trajectory` and `/setpoint/gripper` only when its `arm_motion` parameter is true. The [manifest](/home/abra/rammp-deployments/december_2026/sheppy-manifest.yaml) carries the `adl` node and the [profiles](/home/abra/rammp-deployments/december_2026/profiles/) select one:

```bash
cd /home/abra/rammp-deployments/december_2026
sheppy up adl --manifest sheppy-manifest.yaml   # client only: subscribes, plans, never moves
sheppy status; sheppy down
sheppy restart adl                              # after any code change: `down` takes no node name, and `up` leaves a running node alone
```

No shipped profile arms motion. To move, edit the `adl` node's command in the manifest by hand: set `-p commissioned:=true -p arm_motion:=true`, and add the acknowledged gaps `curobo_online_replanning,continuous_trajectory_handoff,calibrated_aperture` to `capabilities`. Both flags default to false; the physical e-stop must be within reach.

The process is `python -m rammp_adl.ros_node` with `backend:=sheppy`. Its parameters are `context_path` (default [sheppy-bench.context.json](../config/sheppy-bench.context.json)), `capabilities` (the operator's declaration, comma separated), `commissioned`, `arm_motion`, `imagery_policy_path`, `sphere_bundle_dir`, `touch_nm` and the four wrist-camera topics. On startup it logs every declared capability that is a known gap, every unavailable skill with its reason, and, once `/joint_states` flows, the robot facts it bootstrapped from measurement. Nothing in the context JSON is trusted as a robot fact.

Motion follows the planner's real output. RAMMP-CuRobo v1.0.0 returns positions and velocities, no accelerations, with waypoint k stamped at (k+1)*dt and waypoint 0 equal to the start. The [client](../rammp_adl/motion/sheppy_client.py) builds its canonical trajectory from that (start state at t=0, accelerations differenced from the velocity profile, both named in the provenance), runs the start, velocity, range and continuity gates on it, slows it by the class's speed scale, and sends the driver only the planner's own waypoints. After the driver reports success the client waits up to 6 s for the arm to settle and checks arrival within 0.02 rad. A move leaving a grasp pose exempts the tool's neighbourhood from the depth guard for that move; a start the planner refuses as in collision is answered once by retracing the last completed path; a grasp that misses or jams reopens the hand; and the same failure twice ends the task rather than spending the replan budget.

What each skill needs before it registers in hardware mode, all of it operator-supplied:

| Skill | Gate | Where it comes from |
|---|---|---|
| any physical skill | `commissioned:=true` | the operator's assertion, set by hand in the manifest command |
| `move_to_pose` | `curobo_online_replanning`, `continuous_trajectory_handoff` declared | acknowledged gaps: RAMMP-CuRobo v1.0.0 plans from a stationary start, so neither exists client-side; declaring them is the acknowledgement and the node logs it |
| `follow_constraint` | `curobo_constrained_path` declared, `contact_monitor` | the effort guard is the contact monitor; the constrained path is an acknowledged gap, followed as waypoints from rest |
| `move_to_pose` | `live_collision_guard` | real with the wrist depth stream, admitted by the [imagery policy](../config/imagery-locality.json), and the sphere bundle; otherwise the effort guard runs alone and the name is logged as a gap |
| `set_gripper`, `release` | `calibrated_aperture` | the nominal 2F-85 stroke map, 8 mm error; a measured map replaces it through `aperture_map` |
| `grasp` | `calibrated_grasp` | success is a knuckle stall short of closed, not a grip model |
| `release` | `support_detection` | not measured for free objects; a part the constraint record says is attached to its support is released as supported |
| `observe` | the wrist camera under the imagery policy, the face-screen model | wired: keyframes to Astra for labels and boxes, local depth for poses |

### Typing a task

With Astra enabled (the manifest passes `-p enable_astra:=true`; `OPENAI_API_KEY` must be exported by the login shell that `bash -lc` starts, for example from `~/.profile`, and is never read from source), send the task text alone. An empty `task_id` selects intake:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; source /opt/ros/humble/setup.bash; source /home/abra/RAMMP-generalized/artifacts/jetson/ros-install/setup.bash
ros2 action send_goal /rammp/execute_task rammp_adl_interfaces/action/ExecuteTask \
  "{task_id: '', task_text: 'pick up the water bottle', plan_json: ''}" --feedback
```

Intake first discovers the scene: the current wrist keyframe goes to Astra with the task text and comes back as labelled boxes, one per object, plus a verdict on whether the thing the task names is in view. When it is not, Astra names the one camera move most likely to reveal it (pan left or right, tilt up or down, back off, closer); the arm makes that move as a guarded transit from its current tool pose, looks again, and repeats up to `max_viewpoints` times (six by default), keeping every entity seen along the way. A refused move tries the opposite direction; a target that never appears returns `NEED_OBSERVATION`. Local depth under each box, placed through the still-arm camera pose, gives the object's centroid, footprint axes, extents and height, and from those the `pregrasp`, `grasp` and `staging` roles for the planner's `tool_frame` (a top-down grasp for short objects that fit between the fingers, a side grasp along a footprint axis otherwise, no grasp roles if nothing fits). Those entities become the task's world; held/empty is bootstrapped from `/joint_states`; Astra binds the text to one goal predicate among the outcomes the registered skills establish, validated locally; then the ordinary plan, validate, execute and replan loop runs. Feedback phases are `INTAKE_DISCOVERING`, `INTAKE_SEARCHING`, `INTAKE_PROPOSING_CONSTRAINT`, `INTAKE_BOOTSTRAP`, `INTAKE_OBSERVING`, `INTAKE_NORMALIZING_GOAL`, `RUNNING` and `STOPPING`; the result JSON carries the goal, the visible entities and every node result. A declined request returns `incomplete` with `UNSUPPORTED`, `AMBIGUOUS`, `NEED_OBSERVATION`, `EGRESS_REFUSED` or `NEED_CAPABILITY` and its reason.

Keyframes ([keyframes.py](../rammp_adl/perception/keyframes.py)) are captures taken while the arm is verifiably still, placed by forward kinematics through the measured mount, and selected at most once per `keyframe_min_interval_s` when the camera moved more than 2 cm or 3 degrees or more than 5 % of the depth changed by over 3 cm. Every frame that leaves the machine is downscaled to 640 px, re-encoded and passed through the local YuNet face screen ([artifacts/models](../artifacts/models/README.md), sha256 pinned); any detection withholds it, and without the model nothing is sent. Astra sees a keyframe at intake, when a plan node re-observes an entity whose keyframe is no longer current, and, while the arm is idle and no task runs, when a selected keyframe is still unsent (`scene_refresh`, at most twelve per hour by the request budget). Between keyframes an entity is carried by its base-frame position: `observe(purpose=state)` projects it into the current keyframe and checks the depth there, with no cloud call, and a pose observation from the same keyframe reuses the measured geometry.

What runs end to end is observe, move_to_pose over the measured roles, set_gripper, grasp, follow_constraint and release of an attached part: pick up and hold, or open a door or drawer by its handle. Releasing a free object onto a surface still needs support detection.

### Closing the loop with the model

Five mechanisms taken from the closed-loop tool-calling harnesses (RoboCurve, GPT-Policy-Eval and the keyboard and painting cases) sit inside the validated skills, so the model corrects what it sees without ever commanding the arm directly:

- **Look-act before the grasp.** At the standoff, `move_to_pose(grasp)` sends the wrist image and the measured state to Astra, which answers with a small camera-frame shift, DONE or ABORT. Each shift is bounded (3 cm per step, 8 cm in all, 15 degrees of yaw), moves the standoff pose by the same amount at contact speed under the guards so the next image shows its effect, and moves the grasp target with it. DONE releases the final straight approach; ABORT or five calls without DONE fail the node as `target_changed`, which replans.
- **The model marks the grasp point.** Discovery returns, per graspable object, the image point the fingertips should close around. Local depth lifts it to metres and moves the closing point across the object; the approach depth still comes from the measured geometry.
- **Verified following with a progress score.** During `follow_constraint` the wrist depth is read at every waypoint; the door face's normal must have turned as far as the tool had at the waypoint where it was measured, within 0.17 rad, or the node fails `goal_unobserved`. A waypoint where no plane is visible says nothing; if the last waypoint is one of those, the outcome reports the part's displacement as not measured. The attempt record carries a 0 to 4 progress score by rubric, computed locally and, from the last frame, by Astra.
- **Replans see the scene.** A replan request carries the latest screened keyframe alongside the failure feedback.
- **Successes become demonstrations.** A successful follow keeps its start, middle and end frames under `artifacts/constraints/<label>/demo-NNN/`; the next proposal for that kind of part sees the end frame together with the current view.

### Constraints the AI invents and refines

When discovery labels a part as a handle attached to a door or drawer, intake asks Astra for a motion model: hinge or slide, which side the hinge is on as seen in the image, whether it pulls or pushes, the hinge-to-handle width, how far to move and the effort budget. [Local geometry](../rammp_adl/constraints.py) turns that into a base-frame axis and pivot using the handle's measured position and the door face's normal and image axes, and the record goes into `artifacts/constraints/<label>.json`. The plan then reads `follow_constraint(handle, constraint, target)`: the backend traces the arc as cuRobo plans from rest between five-degree or two-centimetre waypoints, each executed at the contact speed under the record's effort budget with the tool's own neighbourhood excluded from the depth guard, and checks after every step that the grip has not slipped. Success means the tool traced the path with the grip retained; the part's own angle is not observed, and the outcome says so.

Every attempt is appended to the record: parameters, target, how far it got, why it stopped, peak effort. The next task on the same kind of part shows Astra that history and asks again; if Astra repeats parameters that already tripped, a local rule moves the hinge-to-handle width toward whichever neighbouring attempt got further. That is the learning loop: measured outcomes and a growing store, not gradient training. `curobo_constrained_path` stays an acknowledged gap because the container has no constrained planner; stop-and-go waypoints are what it gets.



Imagery locality: the deployment publishes both cameras on ROS domain 0 over the host network, which this runtime cannot confine. The [imagery policy](../config/imagery-locality.json) lists that domain as the deployment's own, and that listing admits the subscriptions; nothing else is required.

## Inputs and remaining work before deployment

The Jetson camera identities, user-confirmed Robotiq 2F-85 and local Astra access are recorded in [Jetson setup](jetson.md). No previous camera calibration exists; the single-marker estimates remain provisional. Robot-referenced transforms, clock alignment, physical profiles and installed-assembly geometry remain integration dependencies. Static cuRobo boundary handling and the physical trajectory command transport are implemented and tested in the documented simulation scopes. Driver/firmware compatibility and gripper calibration are engineering verification work; the user does not need to supply the web-interface address or firmware to continue that work. See [hardware testing](hardware-testing.md) and the [fact sheet](design/00-interface-fact-sheet.md). Package maintainer contact and repository license metadata also need real project values before distribution.

Engineering work still includes automatic task-relevant entity/goal initialization, live semantic/metric scene construction, integrated cuRobo ADL physics with articulated objects and gripper contact, validation of constrained moving-boundary planning, driver suffix acknowledgement/streaming, independent stopping supervision, measured safety profiles and Jetson performance. Human-proximate food acquisition and delivery remain deferred. These dependencies must stay visible; clearing a flag or adding a YAML capability cannot resolve them.
