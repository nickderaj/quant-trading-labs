"""Phase 5 merge step: combine all six symbols' phase5_evt_density_{symbol}.json
files, check whether d8_garch_evt/d9_gjr_evt win the 10-model Gate A contest
at each interval, and summarize the spliced density's own health diagnostics
(fraction of refits that produced a valid density, mean continuity gap at
the splice points). Not fanned out - the gate/verdict reading is the
orchestrator's job alone.

Scope note: 1h was dropped after a single symbol (BTC validation aside,
ETH/SOL/DOGE) blew past 30+ minutes of CPU time on that one interval alone
with no sign of finishing - the same sanctioned "drop the interval, say so"
fallback Phase 1 used, applied here for the same reason (compute budget on a
Raspberry Pi), not a hidden or silent scope cut.
"""

import json

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
INTERVALS = ["12h", "4h", "1d"]  # 1h dropped, see module docstring

data = {s: json.load(open(f"src/research/tmp/phase5_evt_density_{s}.json")) for s in SYMBOLS}

out = {"intervals": {}, "scope_note": "1h dropped (compute budget, see module docstring)"}
for interval in INTERVALS:
    per_symbol = {}
    for s in SYMBOLS:
        iv = data[s]["intervals"][interval]
        scores = {k: v["log_score_mean"] for k, v in iv["scores"].items()}
        best = max(scores, key=scores.get)
        evt_is_best = best in ("d8_garch_evt", "d9_gjr_evt")
        # does an EVT model beat every non-EVT model significantly? (EVT-vs-EVT
        # ties are common and expected - both siblings dominating together)
        evt_names = ["d8_garch_evt", "d9_gjr_evt"]
        non_evt_names = [n for n in scores if n not in evt_names]

        def _evt_beats_all_non_evt(evt_name, all_pairs_dm=iv["all_pairs_dm"]):
            for other in non_evt_names:
                key = f"{evt_name}_vs_{other}" if f"{evt_name}_vs_{other}" in all_pairs_dm else f"{other}_vs_{evt_name}"
                entry = all_pairs_dm.get(key)
                if entry is None:
                    continue
                a_is_evt = entry["a"] == evt_name
                bh = entry["bh_bootstrap"]
                wins_sig = bh["significant"] and ((a_is_evt and entry["tstat"] < 0) or (not a_is_evt and entry["tstat"] > 0))
                if not wins_sig:
                    return False
            return True

        d8_dominates = _evt_beats_all_non_evt("d8_garch_evt")
        d9_dominates = _evt_beats_all_non_evt("d9_gjr_evt")

        per_symbol[s] = {
            "scores": scores, "best": best, "evt_is_best": evt_is_best,
            "d8_beats_all_non_evt_significantly": d8_dominates,
            "d9_beats_all_non_evt_significantly": d9_dominates,
            "evt_dominates_non_evt": d8_dominates or d9_dominates,
            "d8_health": iv["spliced_density_health"]["d8_garch_evt"],
            "d9_health": iv["spliced_density_health"]["d9_gjr_evt"],
        }

    n_evt_best = sum(1 for s in SYMBOLS if per_symbol[s]["evt_is_best"])
    n_evt_dominates = sum(1 for s in SYMBOLS if per_symbol[s]["evt_dominates_non_evt"])
    btc_dominates = per_symbol["BTCUSDT"]["evt_dominates_non_evt"]

    out["intervals"][interval] = {
        "per_symbol": per_symbol,
        "n_evt_best_of_6": n_evt_best,
        "n_evt_dominates_non_evt_of_6": n_evt_dominates,
        "btc_evt_dominates": btc_dominates,
        "evt_wins_gate_a_cross_sectionally": n_evt_dominates >= 5 and btc_dominates,
    }
    print(f"--- {interval} ---")
    print(f"  EVT (d8 or d9) is single best model on {n_evt_best}/6 symbols")
    print(f"  EVT significantly dominates every non-EVT model on {n_evt_dominates}/6 symbols "
          f"(BTC: {btc_dominates})")
    print(f"  cross-sectional EVT-wins-Gate-A verdict (>=5/6 incl. BTC): "
          f"{out['intervals'][interval]['evt_wins_gate_a_cross_sectionally']}")
    for s in SYMBOLS:
        h8, h9 = per_symbol[s]["d8_health"], per_symbol[s]["d9_health"]
        print(f"    {s}: best={per_symbol[s]['best']} "
              f"d8_frac_ok={h8['frac_succeeded']:.2f} d8_gap={h8['mean_relative_continuity_gap']:.2f} "
              f"d9_frac_ok={h9['frac_succeeded']:.2f} d9_gap={h9['mean_relative_continuity_gap']:.2f}")

with open("src/research/tmp/phase5_evt_density_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase5_evt_density_results.json")
