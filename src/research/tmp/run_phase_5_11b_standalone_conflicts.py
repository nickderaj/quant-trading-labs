"""11b Phase 5: sec 4.3's standalone `es_calendar`/`gc_cal_m2m3` test
(NEXT_PROMPT.md sec 4.3, sec 2.2's two named cross-repo conflicts).

Runs each spread alone under the pre-declared trading rule (default
params -- neither spread has a declared stop-mult override or regime
requirement in sec 4.1), our costs, dev window, at TWO risk_pct levels:
the pre-declared default (0.03) and a drawdown-matched level tuned (by
1-D bisection on risk_pct) so the standalone book's own max drawdown equals
Gate TS's control book's max drawdown (`phase_0_11b_results.json`'s
structured-book-offset-0 max_drawdown) -- sec 4.3's own instruction, so
the comparison is edge-vs-edge rather than edge-vs-leverage (their own
§14.2 methodology point).

This does NOT adjudicate the external repo's own internal contradiction
for `gc_cal_m2m3` (their v4 §14.2 claims +0.56/+0.62 mean ATR; their own
atlas manifest reportedly shows -0.04 mean ATR over 224 pooled trades) --
that is a disagreement inside their own data this repo cannot see. What
this phase CAN do is report this repo's own independently measured
per-trade edge (`pnl_atr`) and Sharpe for both spreads, standalone, under
this repo's stricter cost model.

Writes phase_5_11b_results.json.
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
GATE_TS_PATH = "src/research/tmp/phase_0_11b_results.json"
OUT_PATH = "src/research/tmp/phase_5_11b_results.json"
DEV_END = "2024-12-31"
SPREADS = ["es_calendar", "gc_cal_m2m3"]

research.set_seed(0)


def load_frame(name: str) -> pl.DataFrame:
    df = pl.read_parquet(f"{SPREAD_DIR}/{name}.parquet")
    return df.filter(pl.col("date") <= pl.lit(DEV_END).str.to_date()).sort("date")


def build_book(name: str, df: pl.DataFrame, risk_pct: float) -> dict:
    p = S11.TradingRuleParams(risk_pct=risk_pct)
    return S11.simulate_book(
        {name: df}, {name: p}, {}, {}, C8.round_turn_cost_per_contract
    )


def drawdown_matched_risk_pct(
    name: str, df: pl.DataFrame, target_dd: float, lo: float = 0.001, hi: float = 0.15
) -> float:
    """Bisect risk_pct so the standalone book's max_drawdown matches
    `target_dd` (both negative numbers; matches on magnitude). Drawdown is
    monotone non-decreasing in risk_pct (more risk per trade -> equal or
    larger drawdown), so bisection is well-posed.
    """
    target_mag = abs(target_dd)
    for _ in range(20):
        mid = (lo + hi) / 2
        dd = abs(S11.book_metrics(build_book(name, df, mid))["max_drawdown"])
        if dd < target_mag:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean_pnl_atr(book: dict) -> float:
    vals = [t["pnl_atr"] for t in book["trades"] if np.isfinite(t["pnl_atr"])]
    return float(np.mean(vals)) if vals else float("nan")


def single_name_binding(name: str, df: pl.DataFrame, p: S11.TradingRuleParams) -> dict:
    """Diagnostic: is `max_single_name_pct` (not `risk_pct`) the binding
    sizing constraint for this spread? notional_per_contract = the more
    expensive leg's own |price| x point value; if 12% of START_EQUITY is
    below that notional, `qty` floors to 0 at ANY risk_pct (sec 4.3 Phase 5's
    own finding, below).
    """
    leg1 = df["leg1_price"].to_numpy().astype(float)
    leg2 = df["leg2_price"].to_numpy().astype(float)
    point_value = S11.POINT_VALUE[next(r["product"] for r in df["leg_roles"][0])]
    notional = np.maximum(np.abs(leg1), np.abs(leg2)) * point_value
    cap_dollars = p.max_single_name_pct / 100.0 * S11.START_EQUITY
    frac_uncapped = float(np.mean(notional <= cap_dollars))
    return {
        "median_notional_per_contract": float(np.median(notional)),
        "single_name_cap_dollars": cap_dollars,
        "frac_bars_notional_within_cap": frac_uncapped,
    }


def main() -> None:
    with open(GATE_TS_PATH) as f:
        control_dd = json.load(f)["gate_TS"]["structured_by_offset"]["offset_0"][
            "max_drawdown"
        ]

    results: dict[str, Any] = {}
    for name in SPREADS:
        df = load_frame(name)
        p = S11.TradingRuleParams()
        binding = single_name_binding(name, df, p)
        default_book = build_book(name, df, 0.03)
        default_metrics = S11.book_metrics(default_book)
        stressed_book = build_book(name, df, 0.15)
        stressed_metrics = S11.book_metrics(stressed_book)
        matched_risk_pct = drawdown_matched_risk_pct(name, df, control_dd)
        matched_book = build_book(name, df, matched_risk_pct)
        matched_metrics = S11.book_metrics(matched_book)
        results[name] = {
            "single_name_cap_diagnostic": binding,
            "default_risk_pct": 0.03,
            "default_metrics": default_metrics,
            "default_mean_pnl_atr": mean_pnl_atr(default_book),
            "risk_pct_0.15_metrics": stressed_metrics,
            "drawdown_matched_risk_pct": matched_risk_pct,
            "drawdown_matched_metrics": matched_metrics,
            "drawdown_matched_mean_pnl_atr": mean_pnl_atr(matched_book),
        }

    out = {
        "control_max_drawdown_target": control_dd,
        "per_spread": results,
        "note": (
            "Both spreads are trade-starved under the pre-declared risk caps: "
            "max_single_name_pct=12% of START_EQUITY=$1,000,000 caps notional per contract "
            "at $120,000, which is below es_calendar's and gc_cal_m2m3's own per-contract "
            "notional (ES/GC point values x their own price level) for nearly the entire "
            "dev window -- qty floors to 0 regardless of risk_pct, since the single-name "
            "cap (not the ATR-based risk_pct sizing) is the binding constraint here. "
            "Raising risk_pct from 0.03 to 0.15 (5x) does not meaningfully unlock trades, "
            "confirming this. The drawdown-matched risk_pct search is reported for "
            "completeness but is not a meaningful edge-vs-edge comparison for either spread "
            "given this constraint -- this notebook does not relax max_single_name_pct to "
            "work around it, since that is itself part of the pre-declared trading rule "
            "(sec 4.1). Does not adjudicate the external repo's own internal atlas-vs-v4-"
            "section contradiction for gc_cal_m2m3 (data this repo does not have)."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    for name, r in results.items():
        b = r["single_name_cap_diagnostic"]
        print(
            f"{name}: median_notional=${b['median_notional_per_contract']:,.0f} "
            f"cap=${b['single_name_cap_dollars']:,.0f} frac_within_cap={b['frac_bars_notional_within_cap']:.3f} "
            f"n_trades(default)={r['default_metrics']['n_trades']} n_trades(riskpct=0.15)={r['risk_pct_0.15_metrics']['n_trades']}"
        )


if __name__ == "__main__":
    main()
