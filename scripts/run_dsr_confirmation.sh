#!/usr/bin/env bash
# Notebook 019, Phase 4 (NEXT_PROMPT.md sec 4 row 4, sec 5.1, sec 7).
# Idempotent: re-running skips every cell already on disk. Never touches
# scratch/017/. Must run AFTER scratch/019/phase4_cells.txt exists
# (build_phase4_grid_19.py) and AFTER phase_3_19_prediction.json exists.
# Usage: scripts/run_dsr_confirmation.sh [--smoke] [--force]
set -euo pipefail
cd "$(dirname "$0")/.."

M=20000
FORCE=""
for arg in "$@"; do
  case "$arg" in
    --smoke) M=500 ;;
    --force) FORCE="--force" ;;
  esac
done

if [ ! -f "src/research/tmp/phase_3_19_prediction.json" ]; then
  echo "phase_3_19_prediction.json missing -- Phase 4 must not start before Phase 3's prediction is on disk (sec 4)" >&2
  exit 1
fi
if [ ! -f "scratch/019/phase4_cells.txt" ]; then
  echo "scratch/019/phase4_cells.txt missing -- run build_phase4_grid_19.py first" >&2
  exit 1
fi

OUT="scratch/019/cells/phase4"
STATUS="scratch/019/status.json"
mkdir -p "$OUT" "$(dirname "$STATUS")"

# sec 7 rule 1: default to 1 on this machine.
CONCURRENCY="${CONCURRENCY:-1}"
# sec 7 rule 2: CHUNK defaults to 200, dropping to 50 for N>=95 & T>=3840
# (only C3 can ever hit that combination, and build_phase4_grid_19.py
# already excludes it from C3 -- kept here as a defensive floor anyway).
CHUNK="${CHUNK:-200}"

PYTHON="$(pwd)/.venv/bin/python"

run_cell() {  # N T rho skew kurt mode subset
  local key="$1-$2-$3-$4-$5-$6-$7"
  [ -f "${OUT}/${key}.json" ] && return 0
  local chunk="$CHUNK"
  if (( $(echo "$1 >= 95" | bc -l) )) && (( $(echo "$2 >= 3840" | bc -l) )); then
    chunk=50
  fi
  "$PYTHON" src/research/tmp/run_phase_4_19_confirmation.py \
      --n-trials "$1" --n-obs "$2" --rho "$3" --skew "$4" --kurt "$5" \
      --mode "$6" --subset "$7" --reps "$M" --chunk "$chunk" \
      --out "${OUT}/${key}.json"
}
export -f run_cell; export OUT M PYTHON CHUNK

TOTAL=$(wc -l < scratch/019/phase4_cells.txt)

# sec 7 rule 5: predicted wall time before any work starts, only cells not
# already on disk, divided by CONCURRENCY. Reuses 017's fitted cost model.
PREDICTED=$("$PYTHON" -c "
import json, os
with open('scratch/019/phase4_manifest.json') as f:
    manifest = json.load(f)
a, b = 2.5395633012820515e-08, 1.6332131410256405e-07
M = $M
concurrency = $CONCURRENCY
remaining = 0.0
for c in manifest['cells']:
    key = f\"{c['n_trials']}-{c['n_obs']}-{c['rho']}-{c['skew']}-{c['kurt']}-{c['mode']}-{c['subset']}\"
    if not os.path.exists(f'$OUT/{key}.json'):
        remaining += (a * c['n_trials'] + b) * c['n_obs'] * M
print(f'{remaining / concurrency:.1f}')
")
echo "predicted wall time on ${CONCURRENCY} proc(s) for remaining cells: ${PREDICTED}s ($(echo "$PREDICTED / 3600" | bc -l | xargs printf '%.2f')h), ${TOTAL} cells total"
if [ "${PREDICTED%.*}" -gt 21600 ] && [ -z "$FORCE" ]; then
  echo "predicted wall time exceeds 6h -- refusing to start without --force" >&2
  exit 1
fi

( while :; do
    DONE=$(ls "$OUT" 2>/dev/null | wc -l)
    "$PYTHON" src/research/tmp/write_status_19.py --phase 4 \
        --done "$DONE" --total "$TOTAL" --state running
    sleep 30
  done ) & HEARTBEAT=$!
trap 'kill $HEARTBEAT 2>/dev/null' EXIT

xargs -P "$CONCURRENCY" -n 7 bash -c 'run_cell "$@"' _ < scratch/019/phase4_cells.txt

"$PYTHON" src/research/tmp/run_phase_4_19_confirmation.py --collate \
    --cells "$OUT" --out src/research/tmp/phase_4_19_confirmation.json
"$PYTHON" src/research/tmp/write_status_19.py --phase 4 --done "$TOTAL" --total "$TOTAL" --state done
