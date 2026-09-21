#!/usr/bin/env bash
# No-motion guardrail: pinned safety files unchanged, tests and design check pass, a live plan passes every send gate.
set -euo pipefail
source onyx/tools/env.sh
out="$(python tools/bench_eval.py offline --plan | tail -1)"
echo "$out" | cut -c1-1500
python - "$out" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
bad = [*r["frozen_problems"]] + ([] if r["tests_passed"] else ["unit tests failed"]) + ([] if r["design_check"] else ["design check failed"]) \
    + ([] if r.get("plan_only", {}).get("fraction") == 1. else ["live plan-only failed: "+str(r.get("plan_only", {}).get("detail"))])
if bad:
    sys.exit("guard failed: "+"; ".join(bad))
PY
