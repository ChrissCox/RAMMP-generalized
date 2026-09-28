#!/usr/bin/env bash
# One hardware cycle: open the cabinet door, then close it in a second task in the same node. A run that never
# measured anything (no operator GO, refused, node did not start, reset failed, no view) exits non-zero with no
# METRIC, so it is never ranked as a zero.
set -euo pipefail
source onyx/tools/env.sh
out="$(python tools/bench_eval.py cycle | tail -1)"
echo "$out" | cut -c1-3500
python - "$out" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
if r["status"] in ("operator_absent", "refused", "halted", "node_failed_to_start", "reset_failed", "no_result", "no_view"):
    sys.exit(f"not a measurement: {r['status']}: {r.get('reason', '')}")
print(f"METRIC cycle={r['score']}")
print(f"METRIC open_score={r.get('open_score', 0)}")
print(f"METRIC close_score={r.get('close_score', 0)}")
print(f"METRIC door_back={int(bool(r.get('door_back')))}")
print(f"METRIC duration_s={r.get('duration_s', 0)}")
print(f"METRIC task_replans={r.get('task_replans') or 0}")
print(f"METRIC safety_fault={int(bool(r.get('safety_fault')))}")
PY
