"""Phase 3 zoo merge step: combine all six symbols' phase3_zoo_{symbol}.json
and apply Gate P (NEXT_RUN_PROMPT.md section 3). Not fanned out - the gate
verdict is the orchestrator's job alone.

Gate P: a new density family is a winner at an interval only if it (i) beats
GARCH-t on mean log score, (ii) significantly under BH-adjusted bootstrap DM,
(iii) on at least 5 of 6 symbols including BTC.
"""

import json
from typing import Any

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
INTERVALS = ["1h", "4h", "12h", "1d"]
ZOO_FAMILIES = ["garch_ged", "garch_nig", "garch_johnsonsu", "garch_hansen_skewt"]


def _load(path):
    with open(path) as f:
        return json.load(f)


data = {s: _load(f"src/research/tmp/phase3_zoo_{s}.json") for s in SYMBOLS}

out: dict[str, Any] = {"intervals": {}}
for interval in INTERVALS:
    per_symbol = {}
    for s in SYMBOLS:
        iv = data[s]["intervals"][interval]
        per_symbol[s] = {
            "scores": {k: v["log_score_mean"] for k, v in iv["scores"].items()},
            "best_by_log_score": iv["best_by_log_score"],
            "best_zoo_family": iv["best_zoo_family"],
            "gate_p_per_family": iv["gate_p_per_family"],
        }

    gate_p_verdict = {}
    for fam in ZOO_FAMILIES:
        n_sig = sum(
            1
            for s in SYMBOLS
            if per_symbol[s]["gate_p_per_family"][fam]["significantly_beats_garch_t"]
        )
        btc_sig = per_symbol["BTCUSDT"]["gate_p_per_family"][fam][
            "significantly_beats_garch_t"
        ]
        n_beats_mean = sum(
            1
            for s in SYMBOLS
            if per_symbol[s]["gate_p_per_family"][fam]["beats_garch_t_on_mean"]
        )
        gate_p_verdict[fam] = {
            "n_significantly_beats_garch_t_of_6": n_sig,
            "n_beats_garch_t_on_mean_of_6": n_beats_mean,
            "btc_significant": btc_sig,
            "fires": n_sig >= 5 and btc_sig,
        }

    out["intervals"][interval] = {
        "per_symbol": per_symbol,
        "gate_p_verdict": gate_p_verdict,
    }
    print(f"--- {interval} ---")
    for fam, v in gate_p_verdict.items():
        print(
            f"  {fam}: sig on {v['n_significantly_beats_garch_t_of_6']}/6 (BTC sig={v['btc_significant']}), "
            f"beats mean on {v['n_beats_garch_t_on_mean_of_6']}/6, Gate P fires={v['fires']}"
        )

# shape-parameter path summary (report ranges, not full paths, for readability)
shape_summary: dict[str, Any] = {}
for interval in INTERVALS:
    shape_summary[interval] = {}
    for fam in ZOO_FAMILIES:
        all_shapes = []
        for s in SYMBOLS:
            sp = data[s]["intervals"][interval]["shape_paths"].get(fam, [])
            all_shapes.extend([e["shape"] for e in sp])
        if not all_shapes:
            continue
        import numpy as np

        arr = np.array(all_shapes)
        shape_summary[interval][fam] = {
            "n_refits_pooled": len(all_shapes),
            "median": arr.mean(axis=0).tolist()
            if arr.ndim > 1
            else [float(arr.mean())],
            "per_param_median": [
                float(np.median(arr[:, i])) for i in range(arr.shape[1])
            ],
            "per_param_range": [
                [float(arr[:, i].min()), float(arr[:, i].max())]
                for i in range(arr.shape[1])
            ],
        }
out["shape_summary"] = shape_summary

with open("src/research/tmp/phase3_zoo_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase3_zoo_results.json")
