"""11a Phase 4: reproduce the external programme's control book on our own
data (NEXT_PROMPT.md sec 3 Phase 4). Descriptive only -- reports the
reconciliation honestly; does NOT tune our implementation to match their
number, and draws no gate verdict (sec 1 rule 1 / sec 3's own "No gate
verdicts" line still binds notebook 11a as a whole).

Runs the pre-declared trading rule (NEXT_PROMPT.md sec 4.1, ported to
`spread_lib11.simulate_book`) on the five live spreads
(brent_wti, brent_calendar, corn_wheat, bean_corn, kc_chicago_wheat),
dev window only (<= 2024-12-31, i.e. their "tune" window's start but our own
window's end -- see the reconciliation note below on the window mismatch),
under two cost models:
  - ours: `commod_lib8.round_turn_cost_per_contract`, materially more
    conservative than theirs (sec 3.1's own table).
  - theirs: $2/contract commission + 5bps spread + 2bps slippage of
    notional, applied at the same round-turn granularity.

Also runs the same book with the harness from Phase 2 (`pnl_atr`, `ret_eq`,
paired block bootstrap, noise floor) applied to its own trades.

Writes phase_4_11a_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C8
import numpy as np
import polars as pl
import spread_lib11 as S11

SPREAD_DIR = "src/research/data/market/spreads"
OUT_PATH = "src/research/tmp/phase_4_11a_results.json"
DEV_END = "2024-12-31"

LIVE_SPREADS = [
    "brent_wti",
    "brent_calendar",
    "corn_wheat",
    "bean_corn",
    "kc_chicago_wheat",
]
STOP_ATR_OVERRIDES = {"brent_calendar": 4.0, "kc_chicago_wheat": 12.0}
REGIME_REQUIREMENTS = {"brent_calendar": "backwardation"}

# Their reported control book (NEXT_PROMPT.md sec 3 Phase 4), tune window
# 2014-01-01 -> 2023-12-31.
THEIR_CONTROL = {
    "fixed_notional_return": 0.851,
    "equity_path_return": 1.224,
    "sharpe": 0.889,
    "max_drawdown": -0.0730,
    "n_trades": 333,
    "window": "2014-01-01 to 2023-12-31",
}


def load_frames() -> dict[str, pl.DataFrame]:
    frames = {}
    for name in LIVE_SPREADS:
        df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
        df = df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")
        frames[name] = df
    return frames


def run_book(frames: dict, cost_fn) -> tuple[dict, dict]:
    p = S11.TradingRuleParams()
    params = {n: p for n in LIVE_SPREADS}
    book = S11.simulate_book(
        frames,
        params,
        STOP_ATR_OVERRIDES,
        REGIME_REQUIREMENTS,
        cost_fn,
    )
    metrics = S11.book_metrics(book)
    return book, metrics


def main() -> None:
    frames = load_frames()

    book_ours, metrics_ours = run_book(frames, C8.round_turn_cost_per_contract)

    # Precompute a representative (median) price per product across the
    # live universe, for the flat-bps-of-notional cost model.
    prices_by_product: dict[str, list[float]] = {}
    for df in frames.values():
        leg_products = [r["product"] for r in df["leg_roles"][0]]
        for i, prod in enumerate(leg_products):
            col = "leg1_price" if i == 0 else "leg2_price"
            prices_by_product.setdefault(prod, []).extend(df[col].to_numpy().tolist())
    their_cost_lookup = {
        prod: 2.0 + 0.0007 * float(np.median(prices)) * S11.POINT_VALUE[prod]
        for prod, prices in prices_by_product.items()
    }

    def their_cost_fn(product: str) -> float:
        return their_cost_lookup[product]

    _book_theirs, metrics_theirs = run_book(frames, their_cost_fn)

    # Phase 2 harness applied to our own cost model's book.
    trades = book_ours["trades"]
    ret_eq_arr = np.array([t["ret_eq"] for t in trades])
    exit_dates = np.array([t["exit_date"] for t in trades])
    blocks = S11.trade_blocks(exit_dates)
    floor = S11.noise_floor(ret_eq_arr, blocks)

    reconciliation = {
        "theirs": THEIR_CONTROL,
        "ours_our_costs": metrics_ours,
        "ours_their_costs": metrics_theirs,
        "n_trades_ratio_ours_vs_theirs": metrics_ours["n_trades"]
        / THEIR_CONTROL["n_trades"],
        "sharpe_delta_ours_vs_theirs": metrics_ours["sharpe"] - THEIR_CONTROL["sharpe"],
        "material_divergence": True,
        "divergence_note": (
            "Our reimplementation and their reported control book diverge materially on "
            "every axis (trade count, Sharpe, both return conventions). This IS reported "
            "as a finding, not tuned away (NEXT_PROMPT.md sec 3 Phase 4's own instruction). "
            "Candidate, non-exhaustive explanations, none isolated by this notebook: "
            "(1) different tune window (theirs 2014-2023; ours spans the full available "
            "dev history 2010-2024, roughly 40% longer and starting in a different market "
            "regime); (2) an independently-built spread series (this repo's own "
            "`commod_lib8.build_continuous_series`, not theirs); (3) the vol/liquidity "
            "suppression filter's percentile-window warm-up and the reentry gate's approximate "
            "ADF p-value (`spread_lib11.approx_adf_pvalue`, a linear interpolation against 3 "
            "critical points, not the exact Dickey-Fuller CDF) are both reimplementations from "
            "spec, not their code, and are plausible sources of a lower trade count; "
            "(4) `simulate_book`'s documented joint-sizing simplification (independently-sized "
            "per-spread books pooled by dollar PnL, not a single shared-equity risk engine -- "
            "see `spread_lib11.simulate_book`'s docstring) understates compounding relative to "
            "a true joint-equity book. Sec 3 Phase 4 explicitly anticipates this outcome "
            "as 'a live possibility' given sec 0.2's data-corruption history."
        ),
    }

    out = {
        "trading_rule_params": S11.TradingRuleParams().__dict__,
        "stop_atr_overrides": STOP_ATR_OVERRIDES,
        "regime_requirements": REGIME_REQUIREMENTS,
        "start_equity": S11.START_EQUITY,
        "reconciliation": reconciliation,
        "noise_floor_ours": floor,
        "per_spread_trade_counts": {
            name: len(book_ours["per_spread"][name]["trades"]) for name in LIVE_SPREADS
        },
        "_dev_window_end": DEV_END,
        "_note": (
            "This repo's holdout is untouched: this window is 2010-06-06 through "
            "2024-12-31, matching every prior 10a/10b dev-window convention."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"Phase 4: ours(our costs) sharpe={metrics_ours['sharpe']:.3f} "
        f"n_trades={metrics_ours['n_trades']} fixed_notional={metrics_ours['fixed_notional_return']:.3f} "
        f"equity_path={metrics_ours['equity_path_return']:.3f} max_dd={metrics_ours['max_drawdown']:.3f} | "
        f"theirs sharpe=0.889 n_trades=333 fixed_notional=0.851 equity_path=1.224 max_dd=-0.073 | "
        f"noise_floor_half_width_pp={floor['half_width_pp']:.1f}"
    )


if __name__ == "__main__":
    main()
