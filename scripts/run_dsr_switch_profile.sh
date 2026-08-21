#!/usr/bin/env bash
# Notebook 019, Phase 2 (NEXT_PROMPT.md sec 4 row 2, sec 7). Idempotent:
# re-running skips every cell already on disk. Never touches scratch/017/.
# Usage: scripts/run_dsr_switch_profile.sh [--smoke] [--force]
set -euo pipefail
cd "$(dirname "$0")/.."

M=2000
FORCE=""
for arg in "$@"; do
  case "$arg" in
    --smoke) M=200 ;;
    --force) FORCE="--force" ;;
  esac
done

OUT="scratch/019/cells/phase2"
STATUS="scratch/019/status.json"
mkdir -p "$OUT" "$(dirname "$STATUS")"

# sec 7 rule 1: default to 1 on this machine (4 cores, 15GB, ~9GB free) --
# 017's script still defaults to 4 and had to be overridden by hand after
# repeated OOM kills at N in {95,122}, T=3840.
CONCURRENCY="${CONCURRENCY:-1}"
CHUNK="${CHUNK:-200}"

# sec 7 rule 4: .venv/bin/python directly, not `uv run python`.
PYTHON="$(pwd)/.venv/bin/python"

run_cell() {  # N T rho skew kurt mode
  local key="$1-$2-$3-$4-$5-$6"
  [ -f "${OUT}/${key}.json" ] && return 0
  "$PYTHON" src/research/tmp/run_phase_2_19_switch_profile.py \
      --n-trials "$1" --n-obs "$2" --rho "$3" --skew "$4" --kurt "$5" \
      --mode "$6" --reps "$M" --chunk "$CHUNK" --out "${OUT}/${key}.json"
}
export -f run_cell; export OUT M PYTHON CHUNK

"$PYTHON" src/research/tmp/build_cells_19_phase2.py > scratch/019/phase2_cells.txt
TOTAL=$(wc -l < scratch/019/phase2_cells.txt)

# sec 7 rule 5: predicted wall time before any work starts, only cells not
# already on disk, divided by CONCURRENCY. Reuses 017's fitted cost model
# (Phase 3's per-cell cost, sec 9.1) as a conservative upper bound -- Phase
# 2's per-replication work (draw + one correlation estimate) is strictly
# cheaper than 017's mc_cell (draw + best-of-N selection + 5-7 variants), so
# this print is an overestimate, not a fresh fit.
PREDICTED=$("$PYTHON" -c "
import os
n_axes = [4, 8, 12, 18, 36, 95, 122]
t_axes = [300, 1000, 3840]
rho_axes = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]
n_moments = 3
a, b = 2.5395633012820515e-08, 1.6332131410256405e-07
M = $M
concurrency = $CONCURRENCY
remaining = 0.0
for n in n_axes:
    for t in t_axes:
        for rho in rho_axes:
            for _ in range(n_moments):
                for mode in ('null', 'edge'):
                    remaining += (a * n + b) * t * M
print(f'{remaining / concurrency:.1f}')
")
echo "predicted wall time on ${CONCURRENCY} proc(s): ${PREDICTED}s ($(echo "$PREDICTED / 3600" | bc -l | xargs printf '%.2f')h), ${TOTAL} cells total"
if [ "${PREDICTED%.*}" -gt 21600 ] && [ -z "$FORCE" ]; then
  echo "predicted wall time exceeds 6h -- refusing to start without --force" >&2
  exit 1
fi

( while :; do
    DONE=$(ls "$OUT" 2>/dev/null | wc -l)
    "$PYTHON" src/research/tmp/write_status_19.py --phase 2 \
        --done "$DONE" --total "$TOTAL" --state running
    sleep 30
  done ) & HEARTBEAT=$!
trap 'kill $HEARTBEAT 2>/dev/null' EXIT

xargs -P "$CONCURRENCY" -n 6 bash -c 'run_cell "$@"' _ < scratch/019/phase2_cells.txt

"$PYTHON" src/research/tmp/run_phase_2_19_switch_profile.py --collate \
    --cells "$OUT" --out src/research/tmp/phase_2_19_switch_profile.json
"$PYTHON" src/research/tmp/write_status_19.py --phase 2 --done "$TOTAL" --total "$TOTAL" --state done
