# Interface contracts

Canonical NEW project-owned ROS definitions are under [interfaces](../../interfaces/), with ament/rosidl build packaging. They describe this project's APIs, not upstream Kinova or cuRobo APIs. The Python [bridge](../../rammp_adl/ros_bridge.py) and [ROS node](../../rammp_adl/ros_node.py) implement the transport boundary for simulation; ROS compilation and execution remain unverified.

| Definition | Semantics |
|---|---|
| GeneratePlan.srv | Request ID, task/epoch, bounded context and catalog identity in; explicit status and complete plan out |
| GroundTarget.srv | Candidate entity and captured image identity in; bounded 2D candidates/error status out; metric geometry stays local |
| GetWorldSnapshot.srv / WorldState.msg | Immutable snapshot identity, entity/collision revisions and task-relevant context |
| ResolveBinding.srv / ResolvedPose.msg | Exact entity/pose role to timestamped local pose/covariance with relevant configuration identities |
| ValidatePlan.srv | Admission or dispatch validation, explicit status, receipt ID; receipt only for requested state/plan/epoch |
| ExecuteSkill.action | Registry ID and typed JSON args with validation receipt/epoch; typed status, outputs/evidence and effect proposal |
| ExecuteTask.action | Whole task entry with task text and optional candidate plan; bounded task outcome and completion feedback; the trusted normalized goal already exists in the world context |
| CommitEffects.srv | Sole-writer compare-and-commit, idempotency key and typed effect records; assigned revision/receipt |
| UserConfirm.srv | Fresh user assent bound to resolved action digest and expiry; single use |
| RequestStop.srv / SafetyStatus.msg | Supervisor stop request and durable status; accepted request is not proof of stopped motion |
| DriverFeedback.msg | NEW instrumented driver exchange sequence, same-host monotonic exchange bracket, boot/session/build identity, measured q/dq and health/ownership; not a device acquisition timestamp or measured acceleration |
| DriverHeartbeat.msg | NEW owner/token/generation-bound monitor heartbeat; sender-time expiry and replay rejection in the independent driver watchdog |

JSON boundaries use schemas in [schemas](../../schemas/). Skill args/outputs derive from the catalog. Strings carrying JSON are transport envelopes, not opaque instructions: strict parse, size limits and schema validation occur at each boundary. No eval, executable predicates or arbitrary expression language exists.

The rolling-motion protocol in [performance requirements](14-performance.md) is implemented as NEW typed internal contracts in [motion/rolling.py](../../rammp_adl/motion/rolling.py), with synthetic and MuJoCo joint tests. It is not an existing upstream ROS service or an extra model skill. Current IDL does not define trajectory generations, future switch acknowledgement or moving-state suffix validation across processes. The actual cuRobo/driver integration remains required before qualifying those capabilities. ExecuteSkill remains the outer action; a successful internal update does not emit terminal success or release claims. If the internal boundary later crosses ROS processes, add explicitly project-owned IDL rather than invent an upstream action field.

All identities are strings or uint64 monotonic revisions as defined in the files; no negative latest sentinel. GetWorldSnapshot explicitly requests the current snapshot for a task. ResolveBinding uses that immutable snapshot ID and returns STALE if no longer usable. Runtime validation must compare entity/collision/base/calibration/attachment dependencies, not merely a global counter.

Outputs and proposed effects never become world truth merely because a skill returned SUCCESS. CommitEffects verifies evidence, checks the prior revision and performs atomic idempotent commit before scheduler lock release. Fact keys come from registered predicates. FactEffect carries predicate, schema-validated bound_args_json, true/false/unknown validity and evidence. Keys include the full canonical bound args, so different pose roles or constraint goals cannot collapse into one fact. False/unknown values invalidate facts; geometry changes invalidate the relevant collision snapshot before the next motion. The service does not accept arbitrary cuboid edits from Astra.

In the current implementation, trusted in-process handlers propose effects and the executor commits them through WorldModel. The public ROS CommitEffects endpoint only returns an exact already-committed receipt; it cannot commit an in-flight operation on behalf of a remote caller. A writable remote backend protocol requires authenticated evaluator/operation ownership. Direct ExecuteSkill also consumes node-attempt authority and keeps one admitted manual plan per epoch, so obtaining another receipt cannot replay a physical action or interleave a competing plan.

GeneratePlan and GroundTarget share held command-resource ownership through the bridge. External cloud requests cannot race a task or manual skill into motion. Captures must come from the local image registry, and a preview file or model-provided URL is not automatically an authorized metric image input. UserConfirm currently rejects requests because the deferred human-facing skills have no connected assent provider.

The first implementation uses stable entity IDs plus typed pose roles; there is no second FROM_NODE/binding grammar. Dependency edges order observations and effects. A new object discovery creates an ID in local perception, followed by replanning; an old ID is never rebound to a different object.

These IDL sources have not been compiled against ROS 2 Humble in this Windows workspace. XML/entry-point syntax and transport-independent behavior were checked. The [runtime runbook](../runtime.md) provides the build/launch recipe and separates those checks from ROS conformance testing.

JSON revision/epoch counters are restricted to 0 through 9007199254740991, an exactly representable subset of ROS uint64. Reject overflow and start a new session identity before rollover. This prevents JavaScript precision loss as well as ROS serialization overflow.

GeneratePlan's no-plan outcomes are schema-valid Astra responses in the response envelope. The local adapter additionally maps transport failures and provider refusals into service status codes; it never invents an empty successful DAG to represent failure.
