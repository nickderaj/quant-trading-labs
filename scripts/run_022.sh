#!/usr/bin/env bash
# Notebook 022 driver (Hyperliquid/Binance CEX-DEX funding spread,
# NEXT_PROMPT.md Candidate 1). Five phases, sequential, foreground,
# idempotent by output-file existence -- mirrors scripts/run_021.sh's own
# precedent: this notebook's book grid is 12 configurations, small enough
# that no background runner / xargs -P / status-heartbeat machinery is
# warranted (021's own reasoning, one order of magnitude below 020's own
# already-cheap 126-cell grid).
#
# Phase 1b (the network fetch) is the one genuinely slow step (~10-30
# minutes for 55 symbols at <=5 req/s) and is run with its own nohup
# backgrounding by convention -- this driver still calls it in the
# foreground because it is itself idempotent (skips any symbol/series
# already cached) and safe to just wait on.
#
# Usage: scripts/run_022.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(pwd)/.venv/bin/python"
TMP="src/research/tmp"

run_step() {  # description, out_path, cmd...
  local desc="$1" out="$2"
  shift 2
  if [ -f "$out" ]; then
    echo "skip: $desc ($out exists)"
    return 0
  fi
  echo "run:  $desc"
  "$@"
}

run_step "Phase 1a: universe mapping" \
  "scratch/022/phase1a_probe.json" \
  "$PYTHON" "$TMP/run_phase_1a_22_universe_map.py"

run_step "Phase 1b: fetch Hyperliquid funding + candles" \
  "scratch/022/phase1_manifest.json" \
  "$PYTHON" "$TMP/run_phase_1b_22_fetch_hyperliquid.py"

echo "run:  Phase 2: library tests"
"$PYTHON" -m pytest tests/test_basis_lib22.py -q

run_step "Phase 2b: frozen-feed screen (unplanned -- added after a tripwire; see src/results/022_hyperliquid_cex_dex_funding_spread.md)" \
  "scratch/022/frozen_feed_exclusions.json" \
  "$PYTHON" "$TMP/run_phase_2b_22_frozen_feed_screen.py"

run_step "Phase 2: gross spread (HD-1) + power (HD-3)" \
  "$TMP/phase_2_22_results.json" \
  "$PYTHON" "$TMP/run_phase_2_22_spread.py"

run_step "Phase 4/5: book grid + gates HD-2/HD-4/HD-5/HD-6/HD-7/FUND-HL + JELLY" \
  "$TMP/phase_4_22_results.json" \
  "$PYTHON" "$TMP/run_phase_4_22_books.py"

run_step "Phase 6: notebook" \
  "src/research/022_hyperliquid_cex_dex_funding_spread.ipynb" \
  "$PYTHON" "$TMP/build_notebook22.py"

echo "022 driver complete."
