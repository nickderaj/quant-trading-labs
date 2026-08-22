#!/usr/bin/env bash
# Notebook 021 driver (NEXT_PROMPT.md sec 4). Five phases, sequential,
# foreground, idempotent by output-file existence. Total measured compute
# budget is under three minutes -- no background runner, no heartbeat, no
# status.json, no xargs -P (sec 5: 021 is an order of magnitude cheaper
# again than 020, which was already cheap enough that its own runner
# machinery was overhead).
#
# Usage: scripts/run_021.sh
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

run_step "Phase 0: pre-registration + reproduction tripwire" \
  "$TMP/phase_0_21_tripwire.json" \
  "$PYTHON" "$TMP/run_phase_0_21_preregistration.py"

run_step "Phase 1: mechanical frozen-feed catalogue" \
  "$TMP/phase_1_21_catalogue.json" \
  "$PYTHON" "$TMP/run_phase_1_21_catalogue.py"

echo "run:  Phase 2: power_lib21 tests"
"$PYTHON" -m pytest tests/test_power_lib21.py -q

run_step "Phase 3: diff CIs, MDE, placebo" \
  "$TMP/phase_3_21_results.json" \
  "$PYTHON" "$TMP/run_phase_3_21_results.py"

run_step "Phase 4: notebook" \
  "src/research/021_rc3_power_and_data_quality.ipynb" \
  "$PYTHON" "$TMP/build_notebook21.py"

echo "021 driver complete."
