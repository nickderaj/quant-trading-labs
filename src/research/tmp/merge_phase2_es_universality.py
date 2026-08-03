"""Phase 2 merge step: combine all six symbols' phase2_es_universality_{symbol}.json
files into the full grid and apply Gate U / Gate U-fat (NEXT_RUN_PROMPT.md
section 3). Not fanned out - gate verdicts are the orchestrator's job alone.

Multiple testing: BH-adjusted across the WHOLE grid (10 models x 4 intervals
x 6 symbols x 6 levels = up to 1,440 tests), per NEXT_RUN_PROMPT.md section 4
Phase 2's explicit instruction, not per-symbol or per-interval in isolation.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
import dist_lib5 as L5

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
INTERVALS = ["1h", "4h", "12h", "1d"]
LEVELS = L5.QUANTILES
LOWER_LEVELS = [0.01, 0.025, 0.05]
UPPER_LEVELS = [0.95, 0.975, 0.99]

THIN_TAILED = {
    "d0_trailing_std",
    "d1_har_rv",
    "d2_har_log_rv",
    "d3_range",
    "d4_garch_normal",
    "d6_gjr_normal",
}
FAT_TAILED_AND_EVT = {"d5_garch_t", "d7_gjr_t", "d8_garch_evt", "d9_gjr_evt"}

# ---- load all six symbols, all cells ----
all_cells = []  # list of dicts with symbol/interval/model/level/z/pvalue/n_violations/underpowered
for symbol in SYMBOLS:
    with open(f"src/research/tmp/phase2_es_universality_{symbol}.json") as _f:
        d = json.load(_f)
    for interval in INTERVALS:
        for cell in d["intervals"][interval]["cells"].values():
            all_cells.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "model": cell["model"],
                    "level": cell["level"],
                    "z": cell["z"],
                    "raw_pvalue": cell["bootstrap_pvalue"],
                    "n": cell["n"],
                    "n_violations": cell["n_violations"],
                    "underpowered": cell["underpowered"],
                }
            )

print(f"total cells loaded: {len(all_cells)}")

# ---- BH adjustment across the WHOLE grid ----
pvalues = {i: c["raw_pvalue"] for i, c in enumerate(all_cells)}
bh = L5.benjamini_hochberg(pvalues, alpha=0.05)
for i, c in enumerate(all_cells):
    c["bh_significant"] = bh[i]["significant"]
    c["bh_rank"] = bh[i]["rank"]

n_underpowered = sum(1 for c in all_cells if c["underpowered"])
print(f"underpowered cells (n_violations < 10): {n_underpowered} / {len(all_cells)}")


def gate_u_stats(cells, exclude_underpowered=True):
    thin = [
        c for c in cells if c["model"] in THIN_TAILED and c["level"] in LOWER_LEVELS
    ]
    if exclude_underpowered:
        thin = [c for c in thin if not c["underpowered"]]
    n = len(thin)
    if n == 0:
        return {"n_cells": 0}
    frac_positive = sum(1 for c in thin if c["z"] > 0) / n
    frac_sig_positive = sum(1 for c in thin if c["z"] > 0 and c["bh_significant"]) / n
    n_sig_negative = sum(1 for c in thin if c["z"] < 0 and c["bh_significant"])
    sig_negative_examples = [
        {
            "symbol": c["symbol"],
            "interval": c["interval"],
            "model": c["model"],
            "level": c["level"],
            "z": c["z"],
        }
        for c in thin
        if c["z"] < 0 and c["bh_significant"]
    ]
    return {
        "n_cells": n,
        "frac_positive": frac_positive,
        "frac_significantly_positive": frac_sig_positive,
        "n_significantly_negative": n_sig_negative,
        "significantly_negative_examples": sig_negative_examples,
        "fires": frac_positive >= 0.90
        and frac_sig_positive >= 0.60
        and n_sig_negative == 0,
    }


gate_u_primary = gate_u_stats(all_cells, exclude_underpowered=True)
gate_u_sensitivity_incl_underpowered = gate_u_stats(
    all_cells, exclude_underpowered=False
)

print(
    f"Gate U (lower-tail, thin-tailed, powered cells only): "
    f"n={gate_u_primary['n_cells']} pos={gate_u_primary['frac_positive']:.3f} "
    f"sig_pos={gate_u_primary['frac_significantly_positive']:.3f} "
    f"sig_neg={gate_u_primary['n_significantly_negative']} "
    f"fires={gate_u_primary['fires']}"
)

# ---- upper-tail secondary panel, same thin-tailed set ----
gate_u_upper = gate_u_stats(
    [c for c in all_cells if c["level"] in UPPER_LEVELS]
    + [c for c in all_cells if False],
    exclude_underpowered=True,
)


# gate_u_stats filters to LOWER_LEVELS internally; build an upper-specific version instead
def upper_stats(cells):
    thin = [
        c
        for c in cells
        if c["model"] in THIN_TAILED
        and c["level"] in UPPER_LEVELS
        and not c["underpowered"]
    ]
    n = len(thin)
    if n == 0:
        return {"n_cells": 0}
    frac_positive = sum(1 for c in thin if c["z"] > 0) / n
    frac_sig_positive = sum(1 for c in thin if c["z"] > 0 and c["bh_significant"]) / n
    n_sig_negative = sum(1 for c in thin if c["z"] < 0 and c["bh_significant"])
    return {
        "n_cells": n,
        "frac_positive": frac_positive,
        "frac_significantly_positive": frac_sig_positive,
        "n_significantly_negative": n_sig_negative,
    }


gate_u_upper = upper_stats(all_cells)
print(f"Upper-tail panel (same thin-tailed set, powered cells): {gate_u_upper}")


def gate_u_fat_stats(cells):
    out = {}
    for model in FAT_TAILED_AND_EVT:
        m_cells = [
            c
            for c in cells
            if c["model"] == model
            and c["level"] in LOWER_LEVELS
            and not c["underpowered"]
        ]
        if not m_cells:
            out[model] = {"n_cells": 0}
            continue
        n_pass = sum(1 for c in m_cells if not c["bh_significant"])
        out[model] = {"n_cells": len(m_cells), "pass_fraction": n_pass / len(m_cells)}
    return out


gate_u_fat = gate_u_fat_stats(all_cells)
print(f"Gate U-fat pass fractions (Z not significantly different from 0): {gate_u_fat}")

out = {
    "n_total_cells": len(all_cells),
    "n_underpowered": n_underpowered,
    "gate_u_primary_excl_underpowered": gate_u_primary,
    "gate_u_sensitivity_incl_underpowered": gate_u_sensitivity_incl_underpowered,
    "gate_u_upper_tail_panel": gate_u_upper,
    "gate_u_fat": gate_u_fat,
    "cells": all_cells,
}
with open("src/research/tmp/phase2_es_universality_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase2_es_universality_results.json")
