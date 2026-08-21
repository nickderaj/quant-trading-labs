#!/usr/bin/env bash
# Notebook 020, Phase 4 (NEXT_PROMPT.md sec 8, sec 3 rule 7). Idempotent:
# re-running skips every cell already on disk. Refuses to start (exit 1) if
# phase_3_20_prediction.json is missing (sec 3b's ordering guard) or if the
# predicted wall time for remaining cells exceeds 6h without --force.
#
# Usage: scripts/run_020_books.sh [--smoke] [--force]
set -euo pipefail
cd "$(dirname "$0")/.."

FORCE=""
for arg in "$@"; do
  case "$arg" in
    --force) FORCE="--force" ;;
  esac
done

if [ ! -f "src/research/tmp/phase_3_20_prediction.json" ]; then
  echo "phase_3_20_prediction.json missing -- Phase 4 must not start before Phase 3b's prediction is on disk (sec 8)" >&2
  exit 1
fi
if [ ! -f "scratch/020/phase4_cells.txt" ]; then
  echo "scratch/020/phase4_cells.txt missing -- run build_phase4_grid_20.py first" >&2
  exit 1
fi

OUT="scratch/020/cells/phase4"
STATUS="scratch/020/status.json"
mkdir -p "$OUT" "$(dirname "$STATUS")"

CONCURRENCY="${CONCURRENCY:-3}"
PYTHON="$(pwd)/.venv/bin/python"

run_cell() {  # variant offset
  local key="$1-$2"
  [ -f "${OUT}/${key}.json" ] && return 0
  "$PYTHON" src/research/tmp/run_phase_4_20_book.py \
      --variant "$1" --offset "$2" --out "${OUT}/${key}.json"
}
export -f run_cell; export OUT PYTHON

TOTAL=$(wc -l < scratch/020/phase4_cells.txt)

# sec 3 rule 7: predicted wall time before starting, for cells not already
# on disk, using Phase 2b's measured bars/sec. Book builds are ~1-2s each on
# this repo's panel sizes, so this refusal is a defensive floor, not
# expected to ever trigger.
PREDICTED=$("$PYTHON" -c "
import json, os
with open('src/research/tmp/phase_2_20_panels.json') as f:
    panels = json.load(f)
bars_per_second = panels.get('book_build_timing', {}).get('bars_per_second', 1000.0)
remaining = 0
with open('scratch/020/phase4_cells.txt') as f:
    for line in f:
        variant, offset = line.split()
        key = f'{variant}-{offset}'
        if not os.path.exists(f'$OUT/{key}.json'):
            remaining += 1
n_bars = panels.get('binance_panels', {}).get('21', {}).get('n_rows', 461431) / 126
per_cell_s = n_bars / bars_per_second + 0.5  # + fixed overhead per invocation
print(f'{remaining * per_cell_s / ${CONCURRENCY}:.1f}')
")
echo "predicted wall time on ${CONCURRENCY} proc(s) for remaining cells: ${PREDICTED}s, ${TOTAL} cells total"
if [ "${PREDICTED%.*}" -gt 21600 ] && [ -z "$FORCE" ]; then
  echo "predicted wall time exceeds 6h -- refusing to start without --force" >&2
  exit 1
fi

( while :; do
    DONE=$(ls "$OUT" 2>/dev/null | grep -c '\.json$' || true)
    "$PYTHON" src/research/tmp/write_status_20.py --phase 4 \
        --done "$DONE" --total "$TOTAL" --state running
    sleep 30
  done ) & HEARTBEAT=$!
trap 'kill $HEARTBEAT 2>/dev/null' EXIT

xargs -P "$CONCURRENCY" -n 2 bash -c 'run_cell "$@"' _ < scratch/020/phase4_cells.txt

"$PYTHON" src/research/tmp/run_phase_4_20_book.py --collate \
    --cells "$OUT" --out src/research/tmp/phase_4_20_results.json
"$PYTHON" src/research/tmp/write_status_20.py --phase 4 --done "$TOTAL" --total "$TOTAL" --state done
