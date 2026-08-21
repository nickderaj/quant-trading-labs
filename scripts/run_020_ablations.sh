#!/usr/bin/env bash
# Notebook 020, Phase 5 (NEXT_PROMPT.md sec 8). Idempotent, same shape as
# run_020_books.sh. Requires Phase 4's results (headline books) to exist.
#
# Usage: scripts/run_020_ablations.sh [--force]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f "src/research/tmp/phase_4_20_results.json" ]; then
  echo "phase_4_20_results.json missing -- run Phase 4 (scripts/run_020_books.sh) first" >&2
  exit 1
fi

OUT="scratch/020/cells/phase5"
STATUS="scratch/020/status.json"
mkdir -p "$OUT" "$(dirname "$STATUS")"

CONCURRENCY="${CONCURRENCY:-3}"
PYTHON="$(pwd)/.venv/bin/python"

VARIANTS=(
  A3_no_hysteresis A3_nmin2 A3_nmin5 A3_cost_0bp A3_cost_17bp A3_cost_51bp A3_excl_luna_ftt
  B0_no_hysteresis B0_one_venue_leg_only B0_cost_0bp "B0_cost_12.5bp" "B0_cost_37.5bp" B0_excl_top2
)
TOTAL=${#VARIANTS[@]}

run_cell() {  # variant
  [ -f "${OUT}/$1.json" ] && return 0
  "$PYTHON" src/research/tmp/run_phase_5_20_ablations.py --variant "$1" --out "${OUT}/$1.json"
}
export -f run_cell; export OUT PYTHON

( while :; do
    DONE=$(ls "$OUT" 2>/dev/null | grep -c '\.json$' || true)
    "$PYTHON" src/research/tmp/write_status_20.py --phase 5 \
        --done "$DONE" --total "$TOTAL" --state running
    sleep 30
  done ) & HEARTBEAT=$!
trap 'kill $HEARTBEAT 2>/dev/null' EXIT

printf '%s\n' "${VARIANTS[@]}" | xargs -P "$CONCURRENCY" -I{} bash -c 'run_cell "$@"' _ {}

"$PYTHON" src/research/tmp/run_phase_5_20_ablations.py --collate \
    --cells "$OUT" --out src/research/tmp/phase_5_20_ablations.json
"$PYTHON" src/research/tmp/write_status_20.py --phase 5 --done "$TOTAL" --total "$TOTAL" --state done
