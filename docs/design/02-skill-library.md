# Derived skill library

[The catalog](../../skills/adl_skill_library.yaml) is the only source for skill IDs, arguments, resources, timing, failures and predicates. It is a fixed vocabulary during execution. [Six explicit handlers](../../rammp_adl/handlers.py) now run against the [synthetic ADL backend](../../rammp_adl/simulation.py). Hardware implementations and commissioned capabilities remain unavailable. YAML does not implement control code or establish deployment readiness.

## Derivation from ADLs

| ADL family | Physical motion and observable events | Shared skills and limits |
|---|---|---|
| Cabinet, fridge, hinged microwave | Find handle/hinge; pregrasp; seat and retain handle; constrained arc; observe opening; release supported handle | observe, move_to_pose, set_gripper, grasp, follow_constraint(revolute), release |
| Drawer, sliding door | Same sequence with translation and stroke/latch evidence | Same skills; prismatic constraint |
| Screw lid | Fixture holds container; retain cap; helical displacement; observe release; supported placement | Same; helical constraint, fixture required |
| Snap/Tupperware lid | Identify tab/seam; fixture supports container; bounded pry/peel path with contact and separation evidence | follow_constraint only for a validated lid/tool profile; arbitrary deformable peeling is not claimed covered |
| Pick-and-place | Observe grasp/support; preshape during free traverse; grasp verification; attached-object transit; support; release | observe, move_to_pose, set_gripper, grasp, release |
| Button/switch | Observe state; configure end effector; approach; constrained contact stroke; retract; observe actual actuation | move_to_pose, set_gripper, follow_constraint, observe; force threshold alone never proves actuation |
| Wiping | Retain suitable pad; establish surface contact; bounded surface path; observe coverage/result | follow_constraint(planar path) plus common manipulation |

This yields six skills. Free transit and empty-gripper aperture are separate to allow useful parallelism. grasp and release have distinct retention/support effects. observe has no hidden positioning motion. follow_constraint factors environment contact/path-following; it is not a universal free-form controller. Food acquisition and human delivery remain separate because their sensing, endpoint geometry and confirmation differ.

All six skills are in scope. move_to_pose and follow_constraint require continuous cuRobo motion with local monitoring and validated rolling updates to the remaining trajectory. A permitted same-target pose refinement stays within the active skill. Computational path samples or horizon boundaries must not introduce mandatory physical stops. Short stop-and-check motions are commissioning/fallback behavior where continuous motion has not yet been validated. [Rolling receiver contracts](../../rammp_adl/motion/rolling.py) and their fixture/physics tests are implemented; cuRobo generation and the actual driver handoff needed to qualify these capabilities remain unavailable. Follow [the performance contract](14-performance.md) for bounded endpoint/model updates, internal ownership, planner lookahead and cross-skill transitions.

Stop, hold, waiting, confirmation and joins belong to supervisor/DAG semantics. There is no learned-policy escape hatch, arbitrary predicate skill, per-task script skill or dynamic skill creation.

## Types and units

Skill arguments are actual nested JSON Schemas in the catalog. Pose arguments are stable entity IDs plus a finite pose_role. ResolveBinding converts a role into a local metric PoseWithCovarianceStamped with capture/calibration/robot/attachment provenance. The plan never supplies raw joint commands or arbitrary target coordinates. All physical scalar lengths are meters, angles radians and times seconds.

follow_constraint references a locally registered/fitted constraint: revolute/helical targets use rad; prismatic targets use m; a planar/guarded path uses signed path arclength in m. Its profile supplies bounded geometry, intended contact pairs, stroke and completion test. target_unit must match that definition, and target_value must fall within its measured bounds. Generic schema numeric ranges are only serialization bounds, never permission to use the maximum on hardware.

profile_id is an allowlisted local configuration reference with hardware identity, object/tool compatibility, limits, tolerances, completion predicates and provenance. Astra cannot create profiles or embed settings inside IDs. Missing/unknown profile disables motion. Aperture is converted using the installed gripper calibration.

## Predicate semantics

Predicates are named registered evaluators, not strings passed to eval. Each declares its closed argument schema in the catalog. Condition bindings are literal values or arg_path arrays into typed skill arguments, never expressions. Facts, effects and goals preserve these bound arguments; predicate plus canonical args identifies the exact condition. Every evaluator consumes timestamped evidence and returns true/false/unknown with an evidence ID. Unknown never satisfies a precondition.

| Predicate group | Required evidence |
|---|---|
| entity_exists, pose_valid, observation_valid | Stable association; valid local pose/uncertainty for required role and observation purpose; source timestamp/TF |
| held_state, gripper_empty, aperture_reached | Verified controller quiescence and measured gripper state/empty-object evidence within calibrated tolerance |
| motion_profile_valid | Profile exists, matches hardware/tool/object and includes required commissioned limits/capabilities |
| at_pose, at_grasp_pose, at_release_pose | Actual TCP/attachment pose within the applicable local role tolerance; collision validity; release alignment consistent with declared support |
| holding | Measured closure plus local retention evidence and stable attachment; motor current alone is insufficient |
| support_verified, released | Known load-bearing support or recipient retention; verified detachment and free/attached collision transition |
| constraint_valid, contact_ready, constraint_goal_verified | Locally fitted constraint with units/bounds; readiness for bounded guarded approach with allowed contact and valid sensing; measured articulation/state/coverage change appropriate to profile |

A move to pose_role=grasp establishes at_grasp_pose for that entity only if the measured role tolerance passes. pose_role=placement can establish at_release_pose only with a matching verified support; the target support ID is not sufficient by itself. Grasp commits holding and invalidates gripper_empty. For a freely carried object it also changes free-object pose/attachment facts. Grasping a fixed handle records a constrained grasp relation and retains the articulated door/handle in world geometry; it does not attach the door as a carried payload. Release invalidates holding/retention and commits supported free-object state only after evidence. Observation refreshes only properties it actually measured. follow_constraint invalidates outdated articulation/obstacle poses and commits measured progress even on failure.

Precondition entailment uses these typed effect rules; it cannot assume success effects on a failed branch. [Runtime validation](../../rammp_adl/validation.py) evaluates ancestor-dependent success branches and rejects unsequenced state conflicts. The [world writer](../../rammp_adl/world.py) accepts exact typed assertions only from issued local evaluator authorities and fresh evidence, checks success predicates atomically, and commits measured effects before dependency release. Fixture evaluators are synthetic; real grasp, support and contact evaluators still require their sensor/backend implementations.

## Resources, timing and recovery

Catalog claims are exclusive; sensor reads are separate. observe reserves ARM and GRIPPER to enforce the held-cloud-wait barrier, even though it commands neither. Grasp/release reserve both to prevent moving during retention/support verification. move_to_pose and empty set_gripper have disjoint claims, but only overlap when the full aperture envelope was checked.

The Python handlers are asynchronous and the ROS transport exposes ExecuteSkill/ExecuteTask actions; ROS compilation and execution remain unverified. Async does not imply simultaneous arm ownership. Nominal times are planning estimates; deadlines are bounded. Safety stop overrides all skills and the supervisor; an independent physical e-stop still requires the deployed robot path.

Failure codes, observable signatures and retry caps are in the catalog. Current local retry is one stationary no-detection retry. Any extra viewpoint, regrasp, articulation change, food retry or retreat is a newly validated plan step, not an unclaimed hidden action. Human-tracking/retention/safety faults hand back. Post-motion failures commit measured partial effects before deciding recovery.

Routine bounded metric refinement and cuRobo trajectory refresh within an active motion are normal execution, not a new viewpoint, mechanism substitution or failed-skill retry. Candidate rejection is nonterminal only while an admissible continuation remains. Refinement never changes the admitted target identity, pose role, constraint type or absolute articulation goal, and never resets the skill timeout. The catalog distinguishes loss of continuation and corrections outside the trusted envelope from normal updates.

For a supported door handle, a verified grasp establishes the expected retained contact relation; contact_ready additionally checks that relation against the locally fitted articulation/contact profile. Releasing a handle may use its door/hinge as verified support without a placement motion, provided local at_release_pose verifies load support at the current articulation. A portable object's placement pose role explicitly includes the intended support relationship; resolve that role from the current goal/profile, not a hidden offset.

follow_constraint owns a profile's bounded guarded approach, verifies expected contact, follows the cuRobo-generated path and performs any profile-defined validated retract. contact_ready means those prerequisites are valid, not that a missing earlier skill already created contact. Hand/recipient_ready transfer ends stationary in a verified shared-support pose with grip retained; release then verifies recipient retention, followed by a separate retract. Mouth/user_done and mouth/bite_removed include validated withdrawal. Other target/completion combinations are rejected.
