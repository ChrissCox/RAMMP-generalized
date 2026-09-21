# Skill-spec foundation implementation plan

Updated by the 2026-09-08 audit. This replaces the former code recipe and unavailable external-skill requirements.

Implement the design in small verified increments, beginning with contract loading and mock execution. The existing tools/check_design.py is an offline design checker, not the production runtime validator.

Implement the first six catalog skills; acquire_bite and transfer_to_user are deferred. Track [performance requirements](../design/14-performance.md) from the first runtime slice, including event-driven dispatch, idle reasons and validated in-flight motion/endpoint updates inside move_to_pose and follow_constraint. Keep deferred bottle-transfer examples as contract fixtures, not a dependency for initial runtime completion.

1. Build rammp_skill_spec for Python 3.10/ROS 2 Humble. Strictly parse the canonical catalog, reject duplicate/unknown metadata and preserve complete nested argument/output schemas, units, frames, bounds, claims, failure policies and effect semantics.
2. Generate the plan schema from that catalog; deploy only implemented handlers with all required capabilities. No dynamic implementation from YAML and no edited duplicate roster.
3. Implement semantic admission: unique IDs, edges/cycles, active vocabulary, entity/profile/constraint lookup, typed unit checks, predicate entailment/invalidation, state conflicts, and request/epoch/hash identity. Use a complete JSON Schema validator in runtime.
4. Compile interfaces/ through rosidl. Add mock tests for refusal/timeout/late results, observation failure, invalid geometry, control loss, stale validation receipts, duplicate effects and fresh single-use confirmation.
5. Implement the DAG scheduler and mandatory command gate against mocks. Demonstrate legal empty-gripper preshaping plus arm transit, conflicting-resource serialization, and stop acknowledgement/effect commit before releasing claims.
6. Connect Astra with the exact model/config and official Responses schema adapter; record real project access, text/image behavior and latency. Disable automatic hidden retries; exercise configured budgets.
7. Follow the risk-ordered roadmap for measured hardware adapters. No motion until the applicable commissioned capabilities exist.

Performance acceptance includes no arbitrary sleep between ready nodes, persistent planner initialization, continuous local sensing, and measurements separating useful action time from avoidable software waiting. Implement same-skill rolling replanning within its existing claims: first test generation/switch semantics, moving-state continuity, stale candidate rejection and bounded stopping against mocks, then prove the selected cuRobo/driver integration in simulation and commissioned hardware. In-flight metric updates must preserve target identity, profile bounds and the original skill deadline.

Implement catalog-derived planner phase leases and prepared-trajectory validation before preparing a different skill during motion; current full-action resource claims cannot be bypassed. Active-motion continuation takes priority over speculative planning. Cross-skill motion blending needs its own verified handoff contract.

Acceptance: the root offline checks pass, runtime tests reject malformed/unsafe semantic plans, generated grammar preserves argument types, fake/unknown capabilities fail closed, and the actual interfaces compile. Report simulation, provider and hardware validation separately.

No per-task implementation packages, mandatory unavailable plugins, forced commit cadence, stale command transcripts or generic universal-controller slots are part of this plan.
