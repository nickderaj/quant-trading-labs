"""11b Phase 4: Gate RE (NEXT_PROMPT.md sec 4.2, sec 2.1's "reversed on
corrected data" entry, their own flagged follow-up).

Sweeps the gated-reentry validity thresholds -- `half_life_max` in
{30, 45, 60} days and `adf_pmax` in {0.05, 0.10, 0.20} -- against the
pre-declared trading rule's default combination (45, 0.10, already the
control/baseline everywhere else in this notebook) on the five live
spreads, our costs, four origin offsets: a genuine 3x3x4 = 36-trial grid,
counted in full (NEXT_PROMPT.md sec 9's explicit instruction -- this is the
one DSR count this notebook may not shrink even if unreachable).

Best-performing non-baseline combination (highest offset-0 Sharpe) is
compared against baseline via the same paired block bootstrap used
everywhere else in this notebook; DSR uses n_trials=36 (the full grid, not
just the winning cell) against that best combination's Sharpe -- the
honest denominator for a search over 9 configurations x 4 offsets.

Fires if: the best combination's net Sharpe > 0 at every offset AND its
paired CI (best - baseline) excludes zero AND DSR > 0.95 at n_trials=36.

Writes phase_4_11b_results.json.
"""

import json
import sys
from typing import Any

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C8
import numpy as np
import polars as pl
import spread_lib11 as S11

import research

SPREAD_DIR = "src/research/data/market/spreads"
OUT_PATH = "src/research/tmp/phase_4_11b_results.json"
DEV_END = "2024-12-31"
ORIGIN_OFFSETS = [0, 7, 14, 21]
ANNUALIZED_RATE = float(np.sqrt(252))
N_TRIALS_RE = 36

LIVE_SPREADS = [
    "brent_wti",
    "brent_calendar",
    "corn_wheat",
    "bean_corn",
    "kc_chicago_wheat",
]
STOP_ATR_OVERRIDES = {"brent_calendar": 4.0, "kc_chicago_wheat": 12.0}
REGIME_REQUIREMENTS = {"brent_calendar": "backwardation"}
HALF_LIFE_MAX_GRID = [30.0, 45.0, 60.0]
ADF_PMAX_GRID = [0.05, 0.10, 0.20]
BASELINE = (45.0, 0.10)

research.set_seed(0)


def load_frame(name: str, offset: int) -> pl.DataFrame:
    df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
    df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    if offset > 0:
        dates = df["date"].unique().sort().to_list()
        keep_from = dates[offset] if offset < len(dates) else dates[-1]
        df = df.filter(pl.col("date") >= pl.lit(keep_from))
    return df


def build_book(offset: int, half_life_max: float, adf_pmax: float) -> dict:
    frames = {n: load_frame(n, offset) for n in LIVE_SPREADS}
    p = S11.TradingRuleParams(half_life_max=half_life_max, adf_pmax=adf_pmax)
    params = {n: p for n in LIVE_SPREADS}
    return S11.simulate_book(
        frames,
        params,
        STOP_ATR_OVERRIDES,
        REGIME_REQUIREMENTS,
        C8.round_turn_cost_per_contract,
    )


def main() -> None:
    grid: dict[str, Any] = {}
    for hl in HALF_LIFE_MAX_GRID:
        for pmax in ADF_PMAX_GRID:
            key = f"hl{int(hl)}_p{pmax}"
            by_offset = {}
            for offset in ORIGIN_OFFSETS:
                by_offset[f"offset_{offset}"] = S11.book_metrics(
                    build_book(offset, hl, pmax)
                )
            grid[key] = {"half_life_max": hl, "adf_pmax": pmax, "by_offset": by_offset}

    baseline_key = f"hl{int(BASELINE[0])}_p{BASELINE[1]}"
    non_baseline = {k: v for k, v in grid.items() if k != baseline_key}
    best_key = max(
        non_baseline, key=lambda k: non_baseline[k]["by_offset"]["offset_0"]["sharpe"]
    )
    best = grid[best_key]
    baseline = grid[baseline_key]

    best_positive_every_offset = all(
        best["by_offset"][f"offset_{o}"]["sharpe"] > 0 for o in ORIGIN_OFFSETS
    )

    best_book0 = build_book(0, best["half_life_max"], best["adf_pmax"])
    baseline_book0 = build_book(0, BASELINE[0], BASELINE[1])
    treatment_pnl = np.array([t["ret_eq"] for t in best_book0["trades"]])
    treatment_blocks = S11.trade_blocks(
        np.array([t["exit_date"] for t in best_book0["trades"]])
    )
    control_pnl = np.array([t["ret_eq"] for t in baseline_book0["trades"]])
    control_blocks = S11.trade_blocks(
        np.array([t["exit_date"] for t in baseline_book0["trades"]])
    )
    re_bootstrap = S11.paired_block_bootstrap(
        control_pnl, control_blocks, treatment_pnl, treatment_blocks
    )

    best_n = best["by_offset"]["offset_0"]["n_trades"]
    best_sharpe = best["by_offset"]["offset_0"]["sharpe"]
    re_dsr = (
        research.deflated_sharpe_prob(
            best_sharpe / ANNUALIZED_RATE, n_trials=N_TRIALS_RE, n_obs=best_n
        )
        if best_n > 1
        else float("nan")
    )

    re_fires = bool(
        best_positive_every_offset
        and re_bootstrap["delta_excludes_zero"]
        and re_dsr > 0.95
    )

    out = {
        "grid": grid,
        "baseline_key": baseline_key,
        "best_non_baseline_key": best_key,
        "best_positive_every_offset": best_positive_every_offset,
        "paired_bootstrap_best_vs_baseline": re_bootstrap,
        "deflated_sharpe_prob_best": re_dsr,
        "n_trials": N_TRIALS_RE,
        "fires": re_fires,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"Gate RE: fires={re_fires} best={best_key} "
        f"best_sharpes={[round(best['by_offset'][f'offset_{o}']['sharpe'], 3) for o in ORIGIN_OFFSETS]} "
        f"baseline_sharpes={[round(baseline['by_offset'][f'offset_{o}']['sharpe'], 3) for o in ORIGIN_OFFSETS]} "
        f"dsr={re_dsr:.4f} delta_ci={re_bootstrap['delta_ci']}"
    )


if __name__ == "__main__":
    main()
