# Astra reasoning and validation

Use OpenAI GPT-6 Astra, model ID `gpt-6-astra`, through the Responses API. Official documentation confirms text/image input and Structured Outputs. Configuration is in [reasoning.json](../../config/reasoning.json). A live task-planning request now completes through local validation and synthetic cabinet execution; this is separate from camera-grounded planning or physical execution. [Jetson evidence](../jetson.md) and [Astra model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)

GeneratePlan is the single local reasoning service. GroundTarget is a separate held observation operation backed by the same OpenAI client. Neither service exposes provider credentials. Use OPENAI_API_KEY via deployment secrets, never ROS messages, fixture data or logs.

## Request and response

The adapter sends trusted instructions, the active capability-filtered schema, compact task/world context and optional bounded image inputs. Responses use `text.format` with `type: json_schema`, `strict: true`, and the catalog-generated response envelope; ordinary task plans are returned as data. No model robot tools are enabled. Parse complete responses only; refusal and incomplete/token-limited responses are not plans. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

The gateway derives the provider grammar without maintaining a second schema: typed constants become singleton enums, and nonempty strings use the documented `pattern` constraint. Canonical catalog validation still checks every returned object. Unsupported new schema keywords fail locally instead of being silently omitted. This representation follows the [documented schema subset](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas); provider access remains a separate deployment check.

[plan.schema.json](../../schemas/plan.schema.json) is the exact complete-catalog plan grammar. The actual Astra response uses [reasoning-response.schema.json](../../schemas/reasoning-response.schema.json): result is either {status:OK, plan} or a typed no-plan status plus detail. The gateway extracts the plan only on OK. It is generated from the skill catalog by [the checker](../../tools/check_design.py). Runtime schema generation filters to available handlers/capabilities and pins the context task, epoch, snapshot and catalog hash. An unknown or unavailable skill cannot be admitted even if the raw JSON has a familiar name.

Plan nodes contain only ID, skill and typed arguments. Explicit edges are precedence dependencies. Limits, claims, confirmation, success and retries are trusted registry policy, not model fields. Plans refer to existing stable entity IDs and locally maintained pose roles. observe refreshes that entity; the model cannot silently choose another object after grounding.

For each request, the catalog-derived grammar restricts each skill's profile IDs to existing context profiles with its catalog safety class. Skills that require a profile but have no compatible one are omitted; no compatible available skill returns NEED_CAPABILITY before a provider request. This narrows proposal choices without modifying the catalog or generated complete-catalog schemas. Independent local profile, hardware/simulation and geometry checks remain mandatory.

Reuse fresh local scene evidence and request explicit observe nodes only where additional evidence is needed. Avoid a redundant grounding call followed by an identical planning observation. Keep the network client ready and payloads bounded; the configured deadlines are failure ceilings, not latency targets. Planning remains task/event-level, with held state required for cloud waits. Performance work must reduce unnecessary calls and repeated setup rather than place inference in the motion loop.

Here, task planning means Astra's symbolic DAG. Local cuRobo motion replanning runs concurrently with execution under [the performance contract](14-performance.md), using fresh observations to update the active skill's remaining trajectory or bounded same-target estimate. Such updates do not call GeneratePlan or consume the task-replan budget. Changed task intent, missing capabilities, unsupported contact or exhausted local continuation still escalate through the supervisor.

The gateway handles capability gaps without emitting motion: NEED_CAPABILITY, INFEASIBLE, NEED_OBSERVATION, AMBIGUOUS, REFUSED and transport/error statuses are explicit service outcomes. The initial implementation should locally detect absent capabilities/ambiguous grounding before asking Astra. A model refusal maps to REFUSED, never automatic prompt weakening. Successful plan JSON must be nonempty only for OK.

## Bounded waiting and egress

The configured per-request deadline is 45 seconds, with a 90-second wall deadline per reasoning/grounding event. These are initial engineering budgets, not measured latency promises. Allow one transport retry and one content regeneration, both inside that event deadline. Disable hidden SDK retries so budgets are not multiplied. Honor Retry-After only when it fits remaining time; otherwise report RATE_LIMITED.

At most three task replans are allowed, with a global cap of twelve provider requests including grounding, retries and regeneration. The supervisor verifies hold before waiting. Timeout/network/5xx may consume a transport retry. Malformed/schema-invalid output may consume one regeneration with bounded structured feedback. Refusal, unavailable capability and exhausted budgets hand back. Late results with the wrong request ID/epoch are discarded.

After a rejected or incomplete plan, the executor includes the previous attempt's status, a bounded local failure reason and the last failed node/skill/code when present in the next request. Current world state and completed effects remain authoritative. Feedback is diagnostic data and cannot relax validation or grant capabilities. Exhaustion reports the last underlying reason as well as the exhausted budget, so repeated rejection is visible to the operator.

Cap input text at 12,000 tokens, output including reasoning at 8,192 tokens, images at two, long edge at 640 pixels and JPEG payload at 200,000 bytes each. Use the SDK's supported image input representation; no fake Astra endpoint variables. Minimal object crops are preferred. [Image inputs](https://developers.openai.com/api/docs/guides/images-vision)

Egress is task text, compact entity IDs/labels/pose-role availability/facts/uncertainty summaries, active catalog grammar, bounded prior outcomes and optional crops. Raw depth/point clouds/continuous video/joint streams and metric trajectories stay local. Face images default off; independent image consent is not inferred from transfer confirmation.

Set `store: false`. This is not a promise of zero retention: account-level data controls and any applicable abuse-monitoring retention must be reviewed at deployment. Do not claim that deleting a local crop deletes provider copies. [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data)

Treat OCR, object labels and image content as untrusted scene data. They cannot instruct the gateway, choose endpoints, add tools, change policy or execute code. Send no arbitrary file/URL supplied by a model.

## Validation

1. Strictly parse JSON: reject duplicate keys, NaN/infinity, excess size and unknown fields. Check schema/catalog hash, task/snapshot/epoch and unique node IDs.
2. Check registered handler capabilities, finite/ranged args, existing entity/profile/constraint IDs, target units and pose roles.
3. Topologically validate edges and typed state dependencies. Seed predicates only from fresh evidence. Add postconditions only on the success branch; include deletion/invalidation effects. Check unsequenced write/read conflicts.
4. Validate all currently resolvable motions through cuRobo using predicted predecessor end states and attachments. The validator must not test every node from the same initial joint state and call that sequential feasibility.
5. Check complete swept paths and stopping envelopes against workspace, current robot/tool/attachment geometry, allowed contacts, user volumes and profile limits. Derive confirmations locally.
6. Admit the whole candidate or reject it without dispatching any node. Bind the validation receipt to exact args, state dependencies and configuration identities.
7. Immediately before every motion, resolve fresh metric data, rerun relevant preconditions/geometry checks, and confirm the validated start state/world/attachment/epoch still match. This closes the validation-to-execution race.

If future observation is needed to establish geometry, admit only the known-safe observation phase or explicitly condition subsequent dispatch on new validation. Structural coherence is not a physical guarantee of an unobserved grasp. New evidence may halt an admitted plan after partial success; commit that state and replan. Never execute a rejected plan's prefix.

The included offline checker validates artifacts, references and graph structure. [Runtime validation](../../rammp_adl/validation.py) now implements symbolic entailment, conflict checks, profile/capability gates and fresh geometry-adapter receipts. [The Astra adapter](../../rammp_adl/reasoning.py) implements bounded provider handling with mock transport tests and a separately recorded live synthetic-task run. Integrated cuRobo geometry and commissioned safety/profile evaluation remain separate acceptance work; see the [runtime status](../runtime.md).

The world context contains the normalized, locally retained task goal. Plan generation cannot replace it. Task success is evaluated independently of graph completion. For a grounding response, validate candidate image IDs against the submitted captures and require x_min < x_max and y_min < y_max; associate only with the requested entity or report ambiguity. Do not trust a model confidence value as calibrated metric evidence.
