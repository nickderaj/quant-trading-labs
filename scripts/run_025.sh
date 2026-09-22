#!/usr/bin/env bash
# Notebook 025 driver (NEXT_PROMPT.md). Seven phases, sequential, foreground,
# idempotent by output-file existence, mirroring scripts/run_024.sh's pattern.
# This is a pedagogical study: no holdout, no trading gate, no Sharpe ratio --
# every premium shown is a model price from realised vol, never a market quote.
#
# Usage: scripts/run_025.sh
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

echo "run:  lib25 and pricer module tests"
"$PYTHON" -m pytest tests/test_lib25.py tests/test_pricers_25_barrier.py \
  tests/test_pricers_25_asian.py tests/test_pricers_25_american.py \
  tests/test_pricers_25_exotic.py -q

run_step "Phase 0: pre-registration" \
  "$TMP/phase_0_25_preregistration.json" \
  "$PYTHON" "$TMP/run_phase_0_25_preregistration.py"

run_step "Phase 1: market inputs and the vol term structure" \
  "$TMP/phase_1_25_inputs.json" \
  "$PYTHON" "$TMP/run_phase_1_25_inputs.py"

run_step "Phase 2: the cross-validation matrix (the referee)" \
  "$TMP/phase_2_25_crossval.json" \
  "$PYTHON" "$TMP/run_phase_2_25_crossval.py"

run_step "Phase 3: primer mechanics" \
  "$TMP/phase_3_25_primer.json" \
  "$PYTHON" "$TMP/run_phase_3_25_primer.py"

run_step "Phase 4: pricing-method behaviour" \
  "$TMP/phase_4_25_methods.json" \
  "$PYTHON" "$TMP/run_phase_4_25_methods.py"

run_step "Phase 5: case studies A-D" \
  "$TMP/phase_5_25_cases_abcd.json" \
  "$PYTHON" "$TMP/run_phase_5_25_cases_abcd.py"

run_step "Phase 6: case studies E-G and the variance aside" \
  "$TMP/phase_6_25_cases_efg.json" \
  "$PYTHON" "$TMP/run_phase_6_25_cases_efg.py"

run_step "Phase 7: notebook" \
  "src/research/025_pricing_bespoke_commodity_derivatives.ipynb" \
  "$PYTHON" "$TMP/build_notebook25.py"

echo "run:  execute notebook"
( cd src/research && "../../.venv/bin/jupyter" nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=1800 \
    025_pricing_bespoke_commodity_derivatives.ipynb )

echo "025 driver complete."
