"""11b Phase 3: Gate VA (NEXT_PROMPT.md sec 4.2, sec 2.1's "reversed on
corrected data" entry).

The external programme's 0.75x-1.25x realized-vol-percentile scaling of
`stop_atr_mult`, applied to the pre-declared trading rule's five live
spreads, our costs, dev window. Scale factor = 0.75 + 0.5 * vol_pctile,
where vol_pctile is the SAME rolling-vol percentile `spread_lib11.
_suppression_masks` already computes for the vol/vol-regime suppression
filters (rolling std of day-over-day change, percentile over a 252-day
window) -- reused here, not recomputed, since it is already the trading
rule's own volatility-regime measure. NaN vol_pctile (warm-up) falls back to
scale=1.0 (`simulate_single_spread`'s own documented behaviour for a
non-finite `stop_mult_scale`).

Fires if: net Sharpe > 0 at every offset AND a paired block-bootstrap 95% CI
on (VA - control) excludes zero AND max drawdown no worse than control's
(the axis their own corrected re-run could not measure -- sec 2.1).

Writes phase_3_11b_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C8
import numpy as np
import polars as pl
import spread_lib11 as S11

import research

SPREAD_DIR = "src/research/data/market/spreads"
OUT_PATH = "src/research/tmp/phase_3_11b_results.json"
DEV_END = "2024-12-31"
ORIGIN_OFFSETS = [0, 7, 14, 21]
ANNUALIZED_RATE = float(np.sqrt(252))
N_TRIALS_VA = 4  # 4 offsets x 1 pre-declared scaling, not swept

LIVE_SPREADS = [
    "brent_wti",
    "brent_calendar",
    "corn_wheat",
    "bean_corn",
    "kc_chicago_wheat",
]
STOP_ATR_OVERRIDES = {"brent_calendar": 4.0, "kc_chicago_wheat": 12.0}
REGIME_REQUIREMENTS = {"brent_calendar": "backwardation"}
SCALE_LO, SCALE_HI = 0.75, 1.25

research.set_seed(0)


def load_frame(name: str, offset: int) -> pl.DataFrame:
    df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
    df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
    if offset > 0:
        dates = df["date"].unique().sort().to_list()
        keep_from = dates[offset] if offset < len(dates) else dates[-1]
        df = df.filter(pl.col("date") >= pl.lit(keep_from))
    return df


def vol_adaptive_scale(df: pl.DataFrame, p: S11.TradingRuleParams) -> np.ndarray:
    value = df["value"].to_numpy().astype(float)
    roll_flag = df["roll_window_flag"].to_numpy()
    vol_pct = S11._suppression_masks(value, roll_flag, p)["vol_pct"]
    return SCALE_LO + (SCALE_HI - SCALE_LO) * vol_pct


def build_book(offset: int, adaptive: bool) -> dict:
    frames = {n: load_frame(n, offset) for n in LIVE_SPREADS}
    p = S11.TradingRuleParams()
    params = {n: p for n in LIVE_SPREADS}
    stop_mult_scales = (
        {n: vol_adaptive_scale(frames[n], p) for n in LIVE_SPREADS} if adaptive else None
    )
    return S11.simulate_book(
        frames,
        params,
        STOP_ATR_OVERRIDES,
        REGIME_REQUIREMENTS,
        C8.round_turn_cost_per_contract,
        stop_mult_scales=stop_mult_scales,
    )


def main() -> None:
    control_by_offset = {}
    va_by_offset = {}
    for offset in ORIGIN_OFFSETS:
        control_by_offset[f"offset_{offset}"] = S11.book_metrics(build_book(offset, False))
        va_by_offset[f"offset_{offset}"] = S11.book_metrics(build_book(offset, True))

    va_positive_every_offset = all(
        va_by_offset[f"offset_{o}"]["sharpe"] > 0 for o in ORIGIN_OFFSETS
    )

    control_book0 = build_book(0, False)
    va_book0 = build_book(0, True)
    treatment_pnl = np.array([t["ret_eq"] for t in va_book0["trades"]])
    treatment_blocks = S11.trade_blocks(np.array([t["exit_date"] for t in va_book0["trades"]]))
    control_pnl = np.array([t["ret_eq"] for t in control_book0["trades"]])
    control_blocks = S11.trade_blocks(np.array([t["exit_date"] for t in control_book0["trades"]]))
    va_bootstrap = S11.paired_block_bootstrap(
        control_pnl, control_blocks, treatment_pnl, treatment_blocks
    )

    control_dd = control_by_offset["offset_0"]["max_drawdown"]
    va_dd = va_by_offset["offset_0"]["max_drawdown"]
    dd_no_worse = va_dd >= control_dd  # both negative; "no worse" = less negative or equal

    va_fires = bool(
        va_positive_every_offset and va_bootstrap["delta_excludes_zero"] and dd_no_worse
    )

    out = {
        "control_by_offset": control_by_offset,
        "va_by_offset": va_by_offset,
        "net_sharpe_positive_every_offset": va_positive_every_offset,
        "paired_bootstrap": va_bootstrap,
        "control_max_drawdown": control_dd,
        "va_max_drawdown": va_dd,
        "drawdown_no_worse_than_control": dd_no_worse,
        "n_trials": N_TRIALS_VA,
        "fires": va_fires,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"Gate VA: fires={va_fires} va_sharpes={[round(va_by_offset[f'offset_{o}']['sharpe'],3) for o in ORIGIN_OFFSETS]} "
        f"control_sharpes={[round(control_by_offset[f'offset_{o}']['sharpe'],3) for o in ORIGIN_OFFSETS]} "
        f"dd_va={va_dd:.4f} dd_control={control_dd:.4f} delta_ci={va_bootstrap['delta_ci']}"
    )


if __name__ == "__main__":
    main()
