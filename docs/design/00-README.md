# Design index

This design targets the ADL scope in [the project prompt](../project-prompt.md). A Python runtime and simulation checks now implement much of the control flow and contracts. The [runtime runbook](../runtime.md) separates implemented components from unavailable robot/GPU integration.

The architecture is suitable with the changes recorded in [the audit](../audit-2026-09-08.md): a cloud proposer, deterministic local validator, DAG scheduler, skill implementations, and independent safety/ownership path. Hardware contact and human-proximate execution remain capability-gated.

| Authority | Purpose |
|---|---|
| [01 architecture](01-architecture.md) | Data flow, placement, interface table |
| [02 skill library](02-skill-library.md) and [catalog](../../skills/adl_skill_library.yaml) | ADL derivation, typed fixed vocabulary |
| [03 world model](03-world-model.md) | Geometry, TF, freshness and evidence |
| [04 reasoning](04-reasoning.md) | Astra, context, plan validation |
| [05 execution](05-execution-engine.md) | Scheduling, stop and effects |
| [06 safety](06-safety.md) | Commissioning and human protection |
| [07 roadmap](07-roadmap.md) | Risk-ordered capability phases |
| [08 questions](08-open-questions.md) | Architecture-changing unknowns |
| [09 cabinet](09-trace-cabinet.md), [10 bottle](10-trace-bottle.md) | Worked traces and fixture links |
| [11 packages](11-package-layout.md), [12 contracts](12-interface-contracts.md) | Package responsibilities and actual IDL source |
| [13 kitchen](13-worked-example-kitchen.md) | Bounded integration scenario |
| [14 performance](14-performance.md) | Continuous motion, reduced idle time and measured latency |
| [external facts](00-interface-fact-sheet.md) | Pinned source facts and deployment checks |

Schemas and ROS definitions live in files; prose links to them instead of carrying alternate field lists. The YAML is JSON-compatible YAML 1.2 so the offline checker needs only Python's standard library. Runtime validation uses full JSON Schema plus semantic, evidence and geometry gates. Six explicit fixture handlers are implemented; every catalog entry retains planned hardware status. Neither fixture capability emulation nor joint-physics replay establishes deployed capability availability.
