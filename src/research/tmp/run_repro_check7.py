"""Phase 0 (checkpoint 0): reproduction check on notebook 3's cost/turnover
numbers, before notebook 7 builds anything on top of them.

Per NEXT_RUN_PROMPT.md's own stop condition (mirrored in the notebook-7
runbook): if any assertion below fails, this script's raised AssertionError
IS the stop-and-report signal. Do not patch the number and carry on - report
it and stop.

Re-derives, directly from the already-committed
src/research/tmp/backtest_results.json (notebook 3's own artifact, never
regenerated here), the headline net Sharpe / gross Sharpe / turnover numbers
`src/results/3_cross_sectional_ic.md` reports for cfg2_12h - the signal
Phase A trades differently but never re-fits.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

RESULTS = json.load(open("src/research/tmp/backtest_results.json"))

cfg2 = RESULTS["cfg2_12h"]["origin_results"]

# ---- 1. cfg2_12h headline (offset 0): net Sharpe +0.42, gross Sharpe +1.32 ----
sharpe_net_0 = cfg2["0"]["stitched_metrics"]["sharpe_net"]
sharpe_gross_0 = cfg2["0"]["stitched_metrics"]["sharpe"]
print(f"cfg2_12h offset=0 sharpe_net={sharpe_net_0:.3f} (write-up: +0.42), "
      f"sharpe_gross={sharpe_gross_0:.3f} (write-up: +1.32)")
assert abs(sharpe_net_0 - 0.42) < 0.005, f"cfg2_12h offset0 net Sharpe {sharpe_net_0} != write-up's +0.42"
assert abs(sharpe_gross_0 - 1.32) < 0.005, f"cfg2_12h offset0 gross Sharpe {sharpe_gross_0} != write-up's +1.32"

# ---- 2. cfg2_12h origin-shift instability: +0.42 -> -2.45 at offset 7 ----
sharpe_net_7 = cfg2["7"]["stitched_metrics"]["sharpe_net"]
print(f"cfg2_12h offset=7 sharpe_net={sharpe_net_7:.3f} (write-up: -2.45)")
assert abs(sharpe_net_7 - (-2.45)) < 0.005, f"cfg2_12h offset7 net Sharpe {sharpe_net_7} != write-up's -2.45"

# ---- 3. cfg2_12h realized turnover/fee drag: ~0.33-0.37%/yr annual fee drag ----
fee_drags = [cfg2[str(o)]["stitched_metrics"]["annual_fee_drag_pct"] for o in (0, 7, 14, 21)]
print(f"cfg2_12h annual_fee_drag_pct across offsets: {[f'{x:.3f}' for x in fee_drags]} "
      f"(write-up: ~0.33-0.37%/yr)")
assert all(0.30 <= x <= 0.40 for x in fee_drags), (
    f"cfg2_12h annual fee drag {fee_drags} outside the write-up's ~0.33-0.37%/yr range"
)

# ---- 4. cfg1_4h / cfg3_1d: negative net Sharpe at every offset (Phase 6's
# "no config beats costs at 4h/1d" claim) ----
for cfg_id in ("cfg1_4h", "cfg3_1d"):
    origin = RESULTS[cfg_id]["origin_results"]
    sharpes = [origin[str(o)]["stitched_metrics"]["sharpe_net"] for o in (0, 7, 14, 21)]
    print(f"{cfg_id} sharpe_net across offsets: {[f'{x:.3f}' for x in sharpes]} (write-up: all negative)")
    assert all(x < 0 for x in sharpes), f"{cfg_id} expected negative net Sharpe at every offset, got {sharpes}"

# ---- 5. holdout Sharpe (Phase 7, spent once): net -0.47, gross +0.74 ----
HOLDOUT = json.load(open("src/research/tmp/holdout_results.json"))
holdout_net = HOLDOUT["holdout_metrics"]["sharpe_net"]
holdout_gross = HOLDOUT["holdout_metrics"]["sharpe"]
print(f"holdout sharpe_net={holdout_net:.3f} (write-up: -0.47), "
      f"sharpe_gross={holdout_gross:.3f} (write-up: +0.74)")
assert abs(holdout_net - (-0.47)) < 0.005, f"holdout net Sharpe {holdout_net} != write-up's -0.47"
assert abs(holdout_gross - 0.74) < 0.005, f"holdout gross Sharpe {holdout_gross} != write-up's +0.74"

print("\nAll notebook-3 headline numbers reproduced from the committed JSONs. Proceeding.")
