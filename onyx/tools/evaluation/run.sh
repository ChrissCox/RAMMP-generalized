#!/usr/bin/env bash
# One attended hardware run. A run that never moved (no operator GO, refused, node did not start) is not a
# measurement: it exits non-zero with no METRIC, so it is never ranked as a zero.
set -euo pipefail
source onyx/tools/env.sh
out="$(python tools/bench_eval.py hardware | tail -1)"
echo "$out" | cut -c1-3000
python - "$out" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
if r["status"] in ("operator_absent", "refused", "node_failed_to_start", "reset_failed", "no_result"):
    sys.exit(f"not a measurement: {r['status']}: {r.get('reason', '')}")
print(f"METRIC score={r['score']}")
print(f"METRIC duration_s={r.get('duration_s', 0)}")
print(f"METRIC followed_fraction={r.get('followed_fraction', 0)}")
print(f"METRIC task_replans={r.get('task_replans') or 0}")
print(f"METRIC safety_fault={int(bool(r.get('safety_fault')))}")
PY
