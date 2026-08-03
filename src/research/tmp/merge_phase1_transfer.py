"""Phase 1 merge step: combine BTC (already computed, notebook 5's own
phase3_density_results.json, plus this notebook's phase1_transfer_BTCUSDT.json
12h validation run confirming an exact match) with the five
phase1_transfer_{symbol}.json files the fanned-out subagents wrote, and apply
Gate T (NEXT_RUN_PROMPT.md section 3) per interval.

Not fanned out - per NEXT_RUN_PROMPT.md section 1's own instruction, gate
verdicts are single-threaded and applied by the orchestrator reading numbers,
never delegated to a subagent.
"""

import json
from typing import Any

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
TRANSFER_SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
INTERVALS = ["12h", "4h", "1h"]

with open("src/research/tmp/phase3_density_results.json") as _f:
    btc_phase3 = json.load(_f)

# The driver (run_phase1_transfer_full.py) was already validated once against
# BTC's committed 12h numbers before any transfer symbol was fanned out: its
# BTC 12h log-score-per-model output matched phase3_density_results.json's
# own d5_garch_t log score (2.622684455811527) to full float precision. BTC
# itself is not re-run here - its Gate A numbers are loaded directly from
# notebook 5's committed phase3_density_results.json (identical computation,
# already proven byte-for-byte reproducible).

out: dict[str, Any] = {"intervals": {}}

for interval in INTERVALS:
    per_symbol = {}
    per_symbol["BTCUSDT"] = {
        "best_by_log_score": btc_phase3["intervals"][interval]["gate_a_verdict"][
            "best_by_log_score"
        ],
        "beats_every_other_significantly_bootstrap_bh": btc_phase3["intervals"][
            interval
        ]["gate_a_verdict"]["beats_every_other_significantly_bootstrap_bh"],
        "beats_every_other_significantly_normal_bh": btc_phase3["intervals"][interval][
            "gate_a_verdict"
        ]["beats_every_other_significantly_normal_bh"],
        "scores": {
            k: v["log_score_mean"]
            for k, v in btc_phase3["intervals"][interval]["scores"].items()
        },
        "n_per_model": {
            k: v["n"] for k, v in btc_phase3["intervals"][interval]["scores"].items()
        },
    }
    for s in TRANSFER_SYMBOLS:
        with open(f"src/research/tmp/phase1_transfer_{s}.json") as _f:
            d = json.load(_f)
        iv = d["intervals"][interval]
        per_symbol[s] = {
            "best_by_log_score": iv["best_by_log_score"],
            "beats_every_other_significantly_bootstrap_bh": iv[
                "beats_every_other_significantly_bootstrap_bh"
            ],
            "beats_every_other_significantly_normal_bh": iv[
                "beats_every_other_significantly_normal_bh"
            ],
            "scores": {k: v["log_score_mean"] for k, v in iv["scores"].items()},
            "n_per_model": {k: v["n"] for k, v in iv["scores"].items()},
        }

    n_garch_t_best = sum(
        1 for s in SYMBOLS if per_symbol[s]["best_by_log_score"] == "d5_garch_t"
    )
    n_garch_t_sig_winner = sum(
        1
        for s in SYMBOLS
        if per_symbol[s]["best_by_log_score"] == "d5_garch_t"
        and per_symbol[s]["beats_every_other_significantly_bootstrap_bh"]
    )
    btc_included = (
        per_symbol["BTCUSDT"]["best_by_log_score"] == "d5_garch_t"
        and per_symbol["BTCUSDT"]["beats_every_other_significantly_bootstrap_bh"]
    )
    # Gate T (NEXT_RUN_PROMPT.md section 3): GARCH-t's win generalizes at an
    # interval if it is the best-log-score model AND significantly beats
    # every other competitor on >=5 of 6 symbols INCLUDING BTC.
    gate_t_fires = (n_garch_t_sig_winner >= 5) and btc_included

    # Cluster result: is the best model always fat-tailed (d5/d7) or
    # log-RV-family (d2), at every symbol, even when identity varies?
    fat_or_logrv_ids = {"d5_garch_t", "d7_gjr_t", "d2_har_log_rv"}
    cluster_holds = all(
        per_symbol[s]["best_by_log_score"] in fat_or_logrv_ids for s in SYMBOLS
    )

    out["intervals"][interval] = {
        "per_symbol": per_symbol,
        "n_garch_t_best_of_6": n_garch_t_best,
        "n_garch_t_significant_winner_of_6": n_garch_t_sig_winner,
        "btc_included_in_significant_count": btc_included,
        "gate_t_fires": gate_t_fires,
        "cluster_fat_tailed_or_log_rv_holds": cluster_holds,
    }
    print(
        f"{interval}: GARCH-t best on {n_garch_t_best}/6, significant winner on {n_garch_t_sig_winner}/6, "
        f"Gate T fires={gate_t_fires}, cluster holds={cluster_holds}"
    )

with open("src/research/tmp/phase1_transfer_full_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase1_transfer_full_results.json")
