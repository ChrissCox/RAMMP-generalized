# Architecture-changing assumptions

| Question | Assumption and consequence |
|---|---|
| Is the installed driver the pinned candidate? | The prompt link was broken; prior documents identify rammp-org/kinova-gen3-ros2. Confirm installation/revision, state stamps, frames, ownership and stop behavior before its adapter is enabled. |
| Does the project have Astra API access? | Public documentation verifies OpenAI gpt-6-astra with image input and structured output. Project credentials, regional access, quotas and measured latency remain untested. Failure returns unavailable; do not silently substitute a model. |
| Which Jetson/software/power/cooling setup? | Provisionally AGX Orin 64 GB MAXN. Exact JetPack/CUDA/cuRobo/ROS compatibility and sustained benchmarks gate deployment. |
| Which gripper is physically fitted? | User baseline says Kinova two-finger; pinned driver describes Robotiq 2F-85. Confirm model/aperture/current semantics and collision geometry. |
| Which temporary Orbbec, FOV and location? | Nominal 90-degree horizontal FOV is only a layout assumption. Calibration/coverage dictate capability. Removing scene camera cannot remove required human/workspace visibility. |
| Can cuRobo cover contact paths and guarded human delivery? | Current wrapper provides transit. Implement and verify constrained paths, contact allowances, attachments and dynamic-world guarding before dependent skills run. |
| Can the selected planner and driver update motion continuously? | Required: bounded local cuRobo replanning from moving joint states, current geometry, validated continuous suffix replacement and measured stopping on underrun. Pinned position-only planning and preemption fields do not establish these capabilities; adapter extensions are required. |
| Is contact evidence adequate near a face? | Torque presence does not establish contact sensitivity. Characterization failure disables dependent tasks pending suitable sensing. |
| Do stop/hold and runtime geometry cover every command? | Full robot/tool/attachment and stopping horizon are required, independent of the cloud. Missing coverage blocks that execution mode. |
| Which fixtures support lids and one-arm tasks? | Containers needing counterholding require secured fixtures or assistance. General task planning does not remove single-arm embodiment limits. |
| What imagery may leave the device? | Bounded object crops by default; face images disabled. If required images are disallowed, use local evidence or report insufficient perception. Provider retention/account controls must be verified. |
| How will the base move later? | Table mount now; external wheelchair repositioning invalidates geometry. Persistent world localization or autonomous base motion changes calibration/versioning needs. |
| What evaluation protocol defines readiness? | Specify supported objects, intervention limits and repeat counts for reproducible validation. Safety capability gates remain. |

These unknowns must become capability checks or configuration failures, not just prose TODOs while executable code assumes success.
