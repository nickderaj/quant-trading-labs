"""Phase 4 merge step: combine all six symbols' phase4_violation_{symbol}.json
files and apply Gate V (NEXT_RUN_PROMPT.md section 3). Not fanned out - the
gate verdict is the orchestrator's job alone.

Gate V: "a count/duration model for the violation process is selected over
the i.i.d.-Bernoulli null by likelihood ratio at p<0.05 on a majority of
(model, interval, symbol) cells." Two complementary tests feed this (4a
counts: NB vs Poisson; 4b durations: discrete Weibull vs geometric) - both
are tests of the same null (i.i.d. Bernoulli violations), from different
angles, so a cell "rejects the null" if EITHER test rejects it at p<0.05
(reported with each sub-test's own rate alongside, not just the OR).
"""

import json

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
INTERVALS = ["1h", "4h", "12h", "1d"]

cells = []
for symbol in SYMBOLS:
    with open(f"src/research/tmp/phase4_violation_{symbol}.json") as _f:
        d = json.load(_f)
    for interval in INTERVALS:
        models = d["intervals"][interval]["models"]
        for model_name, cell in models.items():
            counts_sig = (
                bool(cell["counts"]["nb_significantly_better"])
                if cell["counts"]
                else None
            )
            dur_sig = (
                bool(cell["durations"]["weibull_significantly_better"])
                if cell["durations"]
                else None
            )
            dur_clusters = (
                bool(cell["durations"]["clusters"]) if cell["durations"] else None
            )
            rejects_null = bool(counts_sig) or bool(dur_sig)
            cells.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "model": model_name,
                    "counts_significant": counts_sig,
                    "durations_significant": dur_sig,
                    "durations_cluster_beta_lt_1": dur_clusters,
                    "n_violations": cell["n_violations"],
                    "rejects_iid_bernoulli_null": rejects_null,
                }
            )

n_total = len(cells)
n_counts_valid = sum(1 for c in cells if c["counts_significant"] is not None)
n_counts_sig = sum(1 for c in cells if c["counts_significant"])
n_dur_valid = sum(1 for c in cells if c["durations_significant"] is not None)
n_dur_sig = sum(1 for c in cells if c["durations_significant"])
n_dur_cluster = sum(1 for c in cells if c["durations_cluster_beta_lt_1"])
n_reject = sum(1 for c in cells if c["rejects_iid_bernoulli_null"])

gate_v_fires = n_reject > n_total / 2.0

print(f"total (model,interval,symbol) cells: {n_total}")
print(f"counts (NB vs Poisson) significant: {n_counts_sig}/{n_counts_valid}")
print(
    f"durations (discrete-Weibull vs geometric) significant: {n_dur_sig}/{n_dur_valid} "
    f"(of which beta<1 / clustering: {n_dur_cluster})"
)
print(
    f"cells rejecting i.i.d.-Bernoulli null (counts OR durations): {n_reject}/{n_total} "
    f"({100 * n_reject / n_total:.1f}%)"
)
print(f"Gate V fires (majority): {gate_v_fires}")

# breakdown by model - which models' violations cluster most/least
from collections import defaultdict

by_model: defaultdict[str, dict[str, int]] = defaultdict(
    lambda: {"n": 0, "n_reject": 0}
)
for c in cells:
    by_model[c["model"]]["n"] += 1
    by_model[c["model"]]["n_reject"] += int(c["rejects_iid_bernoulli_null"])
model_summary = {
    m: {"n": v["n"], "n_reject": v["n_reject"], "frac_reject": v["n_reject"] / v["n"]}
    for m, v in sorted(by_model.items())
}
print("by model (fraction of cells rejecting the null):")
for m, v in model_summary.items():
    print(f"  {m}: {v['n_reject']}/{v['n']} ({100 * v['frac_reject']:.0f}%)")

out = {
    "n_total_cells": n_total,
    "n_counts_significant": n_counts_sig,
    "n_counts_valid": n_counts_valid,
    "n_durations_significant": n_dur_sig,
    "n_durations_valid": n_dur_valid,
    "n_durations_clustering_beta_lt_1": n_dur_cluster,
    "n_reject_null": n_reject,
    "frac_reject_null": n_reject / n_total,
    "gate_v_fires": gate_v_fires,
    "by_model": model_summary,
    "cells": cells,
}
with open("src/research/tmp/phase4_violation_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase4_violation_results.json")
