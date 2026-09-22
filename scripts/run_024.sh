#!/usr/bin/env bash
# Notebook 024 driver (NEXT_PROMPT.md). Seven phases, sequential, foreground,
# idempotent by output-file existence, mirroring scripts/run_023.sh's pattern.
# This is a descriptive study: no holdout, no trading gate, no Sharpe ratio.
#
# Usage: scripts/run_024.sh
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

echo "run:  lib24 tests"
"$PYTHON" -m pytest tests/test_lib24.py -q

run_step "Phase 0: pre-registration" \
  "$TMP/phase_0_24_preregistration.json" \
  "$PYTHON" "$TMP/run_phase_0_24_preregistration.py"

run_step "Phase 1: panel construction and data-quality census" \
  "$TMP/phase_1_24_panel_report.json" \
  "$PYTHON" "$TMP/run_phase_1_24_panels.py"

run_step "Phase 2: vol-vs-maturity profiles (Section A)" \
  "$TMP/phase_2_24_profiles.json" \
  "$PYTHON" "$TMP/run_phase_2_24_profiles.py"

run_step "Phase 3: Samuelson slopes and the metals inversion" \
  "$TMP/phase_3_24_slopes.json" \
  "$PYTHON" "$TMP/run_phase_3_24_slopes.py"

run_step "Phase 4: the time dimension (Section B)" \
  "$TMP/phase_4_24_timeseries.json" \
  "$PYTHON" "$TMP/run_phase_4_24_timeseries.py"

run_step "Phase 5: seasonality (Section C)" \
  "$TMP/phase_5_24_seasonality.json" \
  "$PYTHON" "$TMP/run_phase_5_24_seasonality.py"

run_step "Phase 6: the smile/skew exploration (Section D)" \
  "$TMP/phase_6_24_smile_skew.json" \
  "$PYTHON" "$TMP/run_phase_6_24_smile_skew.py"

run_step "Phase 7: notebook" \
  "src/research/024_samuelson_effect_across_futures.ipynb" \
  "$PYTHON" "$TMP/build_notebook24.py"

echo "run:  execute notebook"
( cd src/research && "../../.venv/bin/jupyter" nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=900 \
    024_samuelson_effect_across_futures.ipynb )

echo "024 driver complete."
