#!/usr/bin/env bash
# Notebook 018, Phase 1. Idempotent: re-running skips everything already
# cached (basis_lib18's own per-month parquet caching handles resume).
#
# Usage: scripts/fetch_basis_data.sh
#
# Deliberately takes no [dev|holdout] argument, unlike NEXT_PROMPT.md's own
# sec 9.2 skeleton: the run instructions that supersede it ("Also fetch the
# holdout window to basis18/holdout/ in the same pass ... fetching it early
# costs nothing and saves a second download later") ask for both windows in
# one pass, so run_phase_1_18_fetch.py does both internally rather than
# taking a window argument. Only run_phase_6_18_holdout.py ever turns the
# pre-fetched holdout data into a backtest result -- this script performs
# raw I/O only.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p scratch/018
exec uv run python src/research/tmp/run_phase_1_18_fetch.py
