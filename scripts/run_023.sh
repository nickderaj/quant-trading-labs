#!/usr/bin/env bash
# Notebook 023 driver (NEXT_PROMPT.md). Ten phases, sequential, foreground,
# idempotent by output-file existence, mirroring scripts/run_021.sh's pattern.
# Phase 2 (S/L estimation across 3 liquidity variants) is the slowest step at
# roughly 6-7 minutes; every other phase is under 3 minutes. Phase 9 touches
# the holdout and must only ever run once against real data -- idempotency by
# output-file existence is the mechanism that enforces that.
#
# Usage: scripts/run_023.sh
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

echo "run:  lib23 tests"
"$PYTHON" -m pytest tests/test_lib23.py -q

run_step "Phase 0: pre-registration" \
  "$TMP/phase_0_23_preregistration.json" \
  "$PYTHON" "$TMP/run_phase_0_23_preregistration.py"

run_step "Phase 1: curve panel construction" \
  "$TMP/phase_1_23_panel_report.json" \
  "$PYTHON" "$TMP/run_phase_1_23_panel.py"

run_step "Phase 2: estimate S and L (3 estimators x 3 liquidity variants)" \
  "$TMP/phase_2_23_SL_report.json" \
  "$PYTHON" "$TMP/run_phase_2_23_SL.py"

run_step "Phase 3: fit model (26), train sample" \
  "$TMP/phase_3_23_fit26_report.json" \
  "$PYTHON" "$TMP/run_phase_3_23_fit26.py"

run_step "Phase 4: benchmarks, in-sample full-curve RMSE" \
  "$TMP/phase_4_23_benchmarks_report.json" \
  "$PYTHON" "$TMP/run_phase_4_23_benchmarks.py"

run_step "Phase 5: model (28), short-term shock" \
  "$TMP/phase_5_23_model28_report.json" \
  "$PYTHON" "$TMP/run_phase_5_23_model28.py"

run_step "Phase 6: held-out-maturity test (primary gate, train sample)" \
  "$TMP/phase_6_23_heldout_report.json" \
  "$PYTHON" "$TMP/run_phase_6_23_heldout_maturity.py"

run_step "Phase 7: volatility identity eq (23)" \
  "$TMP/phase_7_23_vol_identity_report.json" \
  "$PYTHON" "$TMP/run_phase_7_23_vol_identity.py"

run_step "Phase 8: seasonality, cross-product" \
  "$TMP/phase_8_23_seasonality_report.json" \
  "$PYTHON" "$TMP/run_phase_8_23_seasonality.py"

run_step "Phase 9: spend the holdout once" \
  "$TMP/phase_9_23_holdout_report.json" \
  "$PYTHON" "$TMP/run_phase_9_23_holdout.py"

run_step "Notebook" \
  "src/research/023_gabillon_two_factor_oil_curve.ipynb" \
  "$PYTHON" "$TMP/build_notebook23.py"

echo "023 driver complete."
