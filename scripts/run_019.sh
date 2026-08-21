#!/usr/bin/env bash
# Notebook 019 orchestrator (NEXT_PROMPT.md sec 7): phases 1->6 unattended,
# safe to nohup. Each phase is idempotent; re-running after an interruption
# resumes at zero cost for anything already on disk. Refuses to continue
# past Phase 4 without --force if the predicted wall time exceeds 6h (that
# check lives inside run_dsr_confirmation.sh itself).
#
# Usage: nohup scripts/run_019.sh [--smoke] [--force] > scratch/019/run_019.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=""
FORCE=""
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE="--smoke" ;;
    --force) FORCE="--force" ;;
  esac
done

PYTHON="$(pwd)/.venv/bin/python"
mkdir -p scratch/019

echo "== Phase 1: dsr_lib17.py v3 extension -- verified via tests =="
uv run pytest tests/test_dsr_lib17.py -q

echo "== Phase 0: pre-registration =="
"$PYTHON" src/research/tmp/run_phase_0_19_preregistration.py

echo "== Phase 2: switch-activation profile =="
bash scripts/run_dsr_switch_profile.sh $SMOKE $FORCE

echo "== Phase 3: prediction (must precede Phase 4) =="
"$PYTHON" src/research/tmp/run_phase_3_19_prediction.py

echo "== Phase 4 grid build =="
"$PYTHON" src/research/tmp/build_phase4_grid_19.py

echo "== Phase 4: confirmation grid (the only new-Monte-Carlo phase) =="
bash scripts/run_dsr_confirmation.sh $SMOKE $FORCE

echo "== Phase 5: adoption =="
"$PYTHON" src/research/tmp/run_phase_5_19_adoption.py

echo "== Phase 6: rescore =="
"$PYTHON" src/research/tmp/run_phase_6_19_rescore.py

echo "== 019 pipeline complete =="
