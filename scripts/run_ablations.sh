#!/usr/bin/env bash
# Notebook 018, Phase 5. Requires basis18/dev to be fully fetched first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scratch/018
python3 -c "import json; json.dump({'phase':'5','state':'running'}, open('scratch/018/status.json','w'))"
uv run python src/research/tmp/run_phase_5_18_ablations.py 2>&1 | tee scratch/018/phase5.log
python3 -c "import json; json.dump({'phase':'5','state':'done'}, open('scratch/018/status.json','w'))"
