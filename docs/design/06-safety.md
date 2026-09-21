# Safety

[Commissioning configuration](../../config/commissioning.json) disables hardware motion. No numerical contact limit has been validated for this robot/user. An accepted Astra plan alone never authorizes motion.

The local safety supervisor and mandatory command gate enforce ownership, registry capabilities, limits, geometry, freshness, confirmations and stopping. A passive subscriber cannot veto an already forwarded command; every command path must pass the gate. Independent driver watchdogs and the physical stop remain effective if the main ROS/Python process stalls.

## Limits and evidence

| Class | Required commissioning |
|---|---|
| Transit | Joint speed/acceleration, relevant link/tool/attachment Cartesian speed, stopping latency/distance, collision uncertainty margins |
| Environment contact | Transit requirements plus contact sensing and explicit permitted contact geometry, force/torque detection and abort behavior |
| Human proximity | Above plus tracking uncertainty/latency, protective separation, specific tool corridor, contact sensitivity and confirmation lifetime |
| Gripper | Calibrated aperture/current/speed, pinch and retention behavior, compatible object/payload range |

Simulation starting candidates are 0.10 m/s transit, 0.03 m/s environment-contact path and 0.03 m/s human-proximate tool speed. They are not approved hardware limits. No guessed N/Nm values are provided: unresolved force/torque limits leave hardware capability disabled. Required configuration fields include both physical force/torque limits and the separately characterized joint-residual thresholds.

Joint effort does not automatically measure external contact force. Establish provenance, gravity/payload compensation, noise, singularity sensitivity, response latency and posture dependence. N, Nm and normalized gripper motor current are distinct quantities. A user button cannot replace insufficient contact sensing.

A speed multiplier does not prove Cartesian limits. Check/retime the cuRobo trajectory against all configured joint/link/tool/attachment limits and validate the final timed path. Recovery has the same limits as ordinary motion.

## Geometry and human interaction

Check the full arm, gripper, tool and held object, including swept and stopping volumes. Maintain current protection for head, torso and observable hands. Margins include calibration error, tracking uncertainty, observation age, measured stopping behavior and bounded human motion assumptions. Loss of visibility never removes a protective volume.

Both transit and contact execution need runtime stopping-horizon checks. A static plan against an old head pose is insufficient. If the installed planner/guard cannot express intended contact or an authorized tool corridor while protecting other links/object portions, that skill stays unavailable.

In-flight replanning must preserve safety during computation and at trajectory replacement. Independently validate the timed candidate, moving-state continuity, current swept/stopping volumes and applicable contact constraints before the command gate installs it. Keep a measured stopping budget if no valid continuation arrives. A new obstacle invalidating current protection causes immediate supervisor action; continuing an obsolete path while waiting for cuRobo is not allowed. GPU scheduling must leave adequate capacity for the guard. The implemented rolling protocol tests and its still-unavailable deployment capabilities are described in [performance requirements](14-performance.md).

Mouth delivery uses a narrowly defined corridor for a specific tool/target/direction during a confirmed transfer. Do not erase the head obstacle or attempt a Boolean hole using positive collision boxes. A bottle at standoff is the default trace outcome; tilting, fluid delivery and mouth contact require their own commissioned capability and explicit task intent.

## Confirmation and stop

Explicit, single-use assent is required before any action classified by local policy (including recipient release, pinching, heating or other enabled high-consequence operations). Bind assent to task, epoch, node, item/target, profile and resolved behavior digest. Expiry, cancellation, geometry change, tracking loss or recoil revokes it. A held button is not a fresh assent.

Watchdog deadlines come from measured stop budgets. Watch robot-state age, command ownership, controller/stream health, gripper retention, required perception, safety process liveness and backend acknowledgements. Cloud failure remains held and bounded; networking is never part of a servo watchdog.

Stop dispatch, invalidate commands, stop the backend, verify stationary state, retain grip where supported, then hold or latch a fault. A physical e-stop has precedence at every stage. Hold mode/payload retention must themselves be commissioned. Do not automatically soften the controller, open the gripper, or go home after a fault. If hold cannot be established, use verified driver/hardware stop behavior and hand back.

Reset never resumes an old plan. Require fresh observations, capability checks, validation and any needed assent.

## Implemented free-space commissioning seam

The [hardware test composition](../../rammp_adl/motion/hardware_test.py) now provides a conditional arm-only route, separate from the fixture ADL node. It requires an independently reviewed collision-free joint region covering the entire installed assembly and all joint combinations in a secured static test cell. The [local verifier](../../rammp_adl/motion/commissioning.py) bounds the exact quintic polynomial continuously with Bernstein coefficient hulls, reserves commissioned stopping excursions/tracking uncertainty inside that region, and applies joint/attachment speed bounds. It does not infer a region from a sampled clear trajectory or claim live human protection.

The [instrumented driver](../../deployment/driver-hardware/) provides non-stealing ownership, successful-exchange timing and a separate C++ heartbeat watchdog. The [gateway](../../rammp_adl/motion/driver_transport.py) requires terminal acknowledgement and fresh measured goal/stop evidence before handback. These paths passed explicit simulation transport checks; physical profiles, acquisition uncertainty, actual stopping/hold behavior and installed geometry remain uncommissioned. See the [physical testing runbook](../hardware-testing.md). This seam does not advertise physical ADL, contact, gripper or rolling capabilities.
