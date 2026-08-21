#!/usr/bin/env bash
# Notebook 020, Phase 1b (NEXT_PROMPT.md sec 4.3). Single-threaded, rate-limited
# fetch of Bybit klines+funding for the dev/holdout windows. Idempotent: each
# symbol/series/window is one cached parquet, skipped if already on disk.
# Time-boxed at 90 minutes inside the python script itself.
#
# Usage: nohup scripts/fetch_bybit_data.sh [--smoke] > scratch/020/phase1_fetch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(pwd)/.venv/bin/python"
mkdir -p scratch/020 src/research/cache/bybit20/dev src/research/cache/bybit20/holdout

"$PYTHON" src/research/tmp/run_phase_1_20_fetch_bybit.py "$@"
