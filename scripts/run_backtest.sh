#!/usr/bin/env bash
# Notebook 018, Phase 4. Requires basis18/dev to be fully fetched
# (scripts/fetch_basis_data.sh) first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scratch/018
python3 -c "import json; json.dump({'phase':'4','state':'running'}, open('scratch/018/status.json','w'))"
uv run python src/research/tmp/run_phase_4_18_backtest.py 2>&1 | tee scratch/018/phase4.log
python3 -c "import json; json.dump({'phase':'4','state':'done'}, open('scratch/018/status.json','w'))"
