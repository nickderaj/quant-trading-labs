#!/usr/bin/env bash
# Notebook 017, Phase 3. Idempotent: re-running skips every cell already on disk.
# Usage: scripts/run_dsr_calibration.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."

M=20000; [ "${1:-}" = "--smoke" ] && M=500
OUT="scratch/017/cells"
STATUS="scratch/017/status.json"
mkdir -p "$OUT" "$(dirname "$STATUS")"

# CONCURRENCY (-P) and CHUNK (mc_cell's replications-per-chunk) are
# overridable via env vars. Defaults (4, 200) match the original design;
# lowered after repeated crashes traced to peak memory at N=95/122 under
# 4-way parallelism (~9GB across workers on a 15GB Pi) -- see the Phase 3
# commit message / write-up for the numbers.
CONCURRENCY="${CONCURRENCY:-4}"
CHUNK="${CHUNK:-200}"

# .venv/bin/python directly, not `uv run python`: at 756 per-cell process
# launches, uv's per-invocation dependency sync check (~0.5-1s fixed tax
# each) dominated a smoke-test run entirely disproportionate to the actual
# M=500 compute cost. The venv is already synced by the `uv run` calls
# above/below; cells only need the interpreter.
PYTHON="$(pwd)/.venv/bin/python"

run_cell() {  # N T rho skew kurt mode
  local key="$1-$2-$3-$4-$5-$6"
  [ -f "${OUT}/${key}.json" ] && return 0        # already have it -> skip
  "$PYTHON" src/research/tmp/run_phase_3_17_calibration.py \
      --n-trials "$1" --n-obs "$2" --rho "$3" --skew "$4" --kurt "$5" \
      --mode "$6" --reps "$M" --chunk "$CHUNK" --out "${OUT}/${key}.json"
}
export -f run_cell; export OUT M PYTHON CHUNK

uv run python src/research/tmp/build_cells_17.py > scratch/017/cells.txt
TOTAL=$(wc -l < scratch/017/cells.txt)

# predicted wall time (sec 9.1 rule 7): empirical cost model fit in Phase 3
# prep from two measured (N, T, M) points (18/3840 and 122/3840, both after
# the mc_cell optimization that dropped the grid's dominant cost -- see
# Phase 3 commit message), cost(N,T,M) ~= (2.54e-8*N + 1.63e-7)*T*M. Only
# cells NOT already on disk are counted (resumability-aware), divided by
# CONCURRENCY.
PREDICTED=$(uv run python3 -c "
import json
with open('src/research/tmp/phase_0_17_preregistration.json') as f:
    g = json.load(f)['grid']
a, b = 2.5395633012820515e-08, 1.6332131410256405e-07
M = $M
concurrency = $CONCURRENCY
remaining = 0.0
for n in g['N']:
    for t in g['T']:
        for rho in g['rho']:
            for m in g['moments']:
                for mode in ('null', 'edge'):
                    key = f\"{n}-{t}-{rho}-{m['skew']}-{m['kurtosis']}-{mode}\"
                    import os
                    if not os.path.exists(f'$OUT/{key}.json'):
                        remaining += (a * n + b) * t * M
print(f'{remaining / concurrency:.1f}')
")
echo "predicted wall time on ${CONCURRENCY} proc(s) for remaining cells: ${PREDICTED}s ($(echo "$PREDICTED / 3600" | bc -l | xargs printf '%.2f')h), ${TOTAL} cells total"
if [ "${PREDICTED%.*}" -gt 21600 ] && [ "${2:-}" != "--force" ]; then
  echo "predicted wall time exceeds 6h -- refusing to start without --force" >&2
  exit 1
fi

( while :; do                                     # heartbeat, independent of the workers
    DONE=$(ls "$OUT" 2>/dev/null | wc -l)
    uv run python src/research/tmp/write_status_17.py --phase 3 \
        --done "$DONE" --total "$TOTAL" --state running
    sleep 30
  done ) & HEARTBEAT=$!
trap 'kill $HEARTBEAT 2>/dev/null' EXIT

xargs -P "$CONCURRENCY" -n 6 bash -c 'run_cell "$@"' _ < scratch/017/cells.txt

uv run python src/research/tmp/run_phase_3_17_calibration.py --collate \
    --cells "$OUT" --out src/research/tmp/phase_3_17_calibration.json
uv run python src/research/tmp/write_status_17.py --phase 3 --done "$TOTAL" --total "$TOTAL" --state done
