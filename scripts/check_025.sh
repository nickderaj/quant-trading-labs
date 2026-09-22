#!/usr/bin/env bash
# Prints the Phase 2 cross-validation pass/fail matrix for notebook 025 in
# under 30 lines, so the model never has to open the JSON to know whether
# the pricers are sound. See NEXT_PROMPT.md section 4, Phase 2.
#
# Usage: scripts/check_025.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REPORT="src/research/tmp/phase_2_25_crossval.json"

if [ ! -f "$REPORT" ]; then
  echo "no $REPORT yet -- run scripts/run_025.sh (or src/research/tmp/run_phase_2_25_crossval.py) first"
  exit 1
fi

.venv/bin/python -c "
import json
d = json.load(open('$REPORT'))
for c in d['matrix']:
    status = 'PASS' if c['passed'] else 'FAIL'
    print(f\"{status}  {c['name']}  ({c['tolerance']})\")
print('---')
print('ALL PASSED' if d['all_passed'] else 'SOME FAILED -- see details above')
"
