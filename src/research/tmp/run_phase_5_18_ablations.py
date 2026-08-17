"""Notebook 018, Phase 5: controls and ablations (sec 7.2's table). Each of
the 6 exhibits below is one entry in the pre-registered n_trials count
(phase_0_18_preregistration.json: 12 baseline + 6 here = 18), evaluated at
origin_offset=0 as the primary/reported number (013's own convention for
Phase 5-style ablation exhibits) -- these are robustness checks on an
already-frozen design, not a parameter sweep for a better headline (sec 12).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib18 as bl

import research
from risk.model import fit_risk_model

OUT_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_5_18_results.json"
LEVERED_MULTIPLES = [3, 5]
COST_ROUND_TURN_LEVELS_BP = [0.0, 17.0, 34.0, 51.0]
DELISTED_SYMBOLS = ["LUNAUSDT", "1000LUNAUSDT", "FTTUSDT"]


def _json_default(o: object) -> object:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def apply_costs_at_rate(
    trade_frame: pl.DataFrame, round_turn_bp: float
) -> pl.DataFrame:
    """apply_two_leg_costs, generalized to an arbitrary round-turn cost
    (sec 7.2 Phase 5's cost-sensitivity exhibit). cost_frac is HALF the
    round-turn rate, since one round turn = one entry + one matching exit,
    each contributing |weight change| once to turnover (sec 3.4).
    """
    cost_frac = (round_turn_bp / 2.0) * 1e-4
    return (
        trade_frame.with_columns(
            (1 - cost_frac * pl.col("turnover")).log().alias("cost_log_return")
        )
        .with_columns(
            (pl.col("trade_log_return") + pl.col("cost_log_return")).alias(
                "trade_log_return_net"
            )
        )
        .with_columns(
            pl.col("trade_log_return_net").cum_sum().alias("equity_curve_net")
        )
        .with_columns(
            (pl.col("equity_curve_net") - pl.col("equity_curve_net").cum_max()).alias(
                "drawdown_log_return_net"
            )
        )
    )


def main() -> None:
    panel, manifest = bl.load_basis_panel()
    featured = bl.add_trade_features(panel)
    annualized_rate = research.sharpe_to_annualized_rate("8h")

    results: dict[str, object] = {}

    # ---- baseline (timed, offset 0) for reference ----
    timed_w = bl.build_book_weights(featured, timed=True)
    timed_tf = bl.book_trade_frame(featured, timed_w, origin_offset=0)
    timed_costed = bl.apply_two_leg_costs(timed_tf)
    baseline_metrics = bl.book_metrics(timed_costed, annualized_rate, "timed_baseline")
    results["baseline"] = baseline_metrics
    print(f"baseline timed net Sharpe: {baseline_metrics['sharpe_net']:.3f}")

    # ---- 1. no-hysteresis (theta_out == theta_in) ----
    no_hyst_w = bl.build_book_weights(
        featured, timed=True, theta_in=bl.THETA_IN, theta_out=bl.THETA_IN
    )
    no_hyst_tf = bl.book_trade_frame(featured, no_hyst_w, origin_offset=0)
    no_hyst_costed = bl.apply_two_leg_costs(no_hyst_tf)
    no_hyst_metrics = bl.book_metrics(no_hyst_costed, annualized_rate, "no_hysteresis")
    results["1_no_hysteresis"] = no_hyst_metrics
    print(
        f"no-hysteresis net Sharpe: {no_hyst_metrics['sharpe_net']:.3f}, "
        f"turnover {no_hyst_metrics['annualized_turnover']:.1f}/yr "
        f"(baseline {baseline_metrics['annualized_turnover']:.1f}/yr)"
    )

    # ---- 2. perp-leg-only (no spot hedge) ----
    perp_only = featured.sort(["symbol", "datetime"]).with_columns(
        (-pl.col("perp_log_return") + pl.col("funding_rate"))
        .over("symbol")
        .alias("perp_only_log_return")
    )
    perp_only = perp_only.with_columns(
        pl.col("perp_only_log_return")
        .shift(-1)
        .over("symbol")
        .alias("fwd_perp_only_return_1")
    )
    perp_only_tf = research.portfolio_trade_frame(
        timed_w, perp_only, target_col="fwd_perp_only_return_1"
    )
    perp_fee_frac = (
        bl.PERP_TAKER_BP + bl.SLIPPAGE_BP
    ) * 1e-4  # one leg only, no spot fee
    perp_only_costed = perp_only_tf.with_columns(
        (1 - perp_fee_frac * pl.col("turnover")).log().alias("cost_log_return")
    ).with_columns(
        (pl.col("trade_log_return") + pl.col("cost_log_return")).alias(
            "trade_log_return_net"
        )
    )
    perp_only_metrics = research._series_metrics(
        perp_only_costed["trade_log_return_net"], annualized_rate, "perp_only"
    )
    basket = research.equal_weight_basket_returns(
        featured, target_col="fwd_perp_return_1"
    ).rename({"trade_log_return": "trade_log_return_basket"})
    btc = featured.filter(pl.col("symbol") == "BTCUSDT").select(
        "datetime", pl.col("fwd_perp_return_1").alias("btc_return")
    )
    beta_frame = (
        perp_only_costed.select("datetime", "trade_log_return_net")
        .join(basket, on="datetime", how="inner")
        .join(btc, on="datetime", how="inner")
    )
    perp_only_beta_basket = bl.ols_beta(
        beta_frame["trade_log_return_net"].to_numpy(),
        beta_frame["trade_log_return_basket"].to_numpy(),
    )
    perp_only_beta_btc = bl.ols_beta(
        beta_frame["trade_log_return_net"].to_numpy(),
        beta_frame["btc_return"].to_numpy(),
    )
    hedged_beta_frame = (
        timed_costed.select("datetime", "trade_log_return_net")
        .join(basket, on="datetime", how="inner")
        .join(btc, on="datetime", how="inner")
    )
    hedged_beta_basket = bl.ols_beta(
        hedged_beta_frame["trade_log_return_net"].to_numpy(),
        hedged_beta_frame["trade_log_return_basket"].to_numpy(),
    )
    hedged_beta_btc = bl.ols_beta(
        hedged_beta_frame["trade_log_return_net"].to_numpy(),
        hedged_beta_frame["btc_return"].to_numpy(),
    )
    results["2_perp_leg_only_no_spot_hedge"] = {
        **perp_only_metrics,
        "beta_to_crypto_basket": perp_only_beta_basket,
        "beta_to_btc": perp_only_beta_btc,
        "hedged_baseline_beta_to_crypto_basket": hedged_beta_basket,
        "hedged_baseline_beta_to_btc": hedged_beta_btc,
        "confirms_hedge_removes_beta": bool(
            abs(perp_only_beta_basket) > abs(hedged_beta_basket)
            and abs(perp_only_beta_btc) > abs(hedged_beta_btc)
        ),
    }
    print(
        f"perp-only beta_basket={perp_only_beta_basket:.4f} beta_btc={perp_only_beta_btc:.4f}"
    )

    # ---- 3. excluding LUNA/FTT ----
    ex_panel = featured.filter(~pl.col("symbol").is_in(DELISTED_SYMBOLS))
    ex_w = bl.build_book_weights(ex_panel, timed=True)
    ex_tf = bl.book_trade_frame(ex_panel, ex_w, origin_offset=0)
    ex_costed = bl.apply_two_leg_costs(ex_tf)
    ex_metrics = bl.book_metrics(ex_costed, annualized_rate, "excluding_luna_ftt")
    results["3_excluding_luna_ftt"] = {
        **ex_metrics,
        "symbols_excluded": DELISTED_SYMBOLS,
        "symbols_present_in_universe": [
            s for s in DELISTED_SYMBOLS if s in manifest and manifest[s] == "ok"
        ],
    }
    print(
        f"excluding LUNA/FTT net Sharpe: {ex_metrics['sharpe_net']:.3f} (baseline {baseline_metrics['sharpe_net']:.3f})"
    )

    # ---- 4. cost sensitivity ----
    cost_sensitivity = {}
    for rt_bp in COST_ROUND_TURN_LEVELS_BP:
        costed = apply_costs_at_rate(timed_tf, rt_bp)
        m = research._series_metrics(
            costed["trade_log_return_net"], annualized_rate, f"cost_{rt_bp}bp"
        )
        cost_sensitivity[str(rt_bp)] = m
    results["4_cost_sensitivity"] = cost_sensitivity
    breakeven_bp = None
    sharpes_by_cost = [
        (rt, cost_sensitivity[str(rt)]["sharpe"]) for rt in COST_ROUND_TURN_LEVELS_BP
    ]
    for rt, sharpe in sharpes_by_cost:
        if sharpe <= 0:
            breakeven_bp = rt
            break
    results["4_cost_sensitivity_breakeven_round_turn_bp"] = breakeven_bp
    print(f"cost sensitivity sharpes: {sharpes_by_cost}")

    # ---- 5. levered variants + sec 6.3 liquidation analysis ----
    net_log_returns = timed_costed["trade_log_return_net"].to_numpy()
    levered_results = {}
    for lev in LEVERED_MULTIPLES:
        simple = np.expm1(net_log_returns)
        levered_simple = lev * simple
        levered_simple = np.clip(levered_simple, -0.999999, None)
        levered_log = np.log1p(levered_simple)
        mean = float(np.nanmean(levered_log))
        std = float(np.nanstd(levered_log))
        sharpe = (mean / std) * annualized_rate if std else 0.0
        cum = np.nancumsum(levered_log)
        dd = cum - np.maximum.accumulate(cum)
        levered_results[str(lev)] = {
            "sharpe": sharpe,
            "total_log_return": float(np.nansum(levered_log)),
            "max_drawdown": float(np.nanmin(dd)) if len(dd) else 0.0,
        }
    results["5_levered_variants"] = levered_results

    perp_shock = featured["perp_log_return"].drop_nulls().to_numpy()
    perp_shock = perp_shock[np.isfinite(perp_shock)]
    liquidation_analysis: dict[str, object] = {
        "citation": "notebook 006 (Distribution Zoo, crypto), NOT 008 (Gate RE/CE/CT is commodity futures and does not transfer -- NEXT_PROMPT sec 6.2)",
        "series_used": "pooled cross-symbol perp 8h log returns (the short perp leg's own mark-to-market -- spot and perp margin are not cross-margined on a real exchange, so an adverse perp rally can trigger liquidation even though the paired position is economically delta-neutral)",
        "n_obs": len(perp_shock),
    }
    fitted = fit_risk_model(
        perp_shock, product="crypto_perp_basis_short_leg", family="hansen_skewt"
    )
    if fitted is not None:
        es_1pct = fitted.es(0.01)
        liquidation_analysis["family"] = "hansen_skewt"
        liquidation_analysis["es_1pct_per_period"] = es_1pct
        liquidation_analysis["required_margin_buffer_note"] = (
            "1% expected shortfall of the perp leg's own 8h return, expressed as a "
            "fraction of perp notional -- a levered variant's margin buffer should be "
            "sized to survive several multiples of this before a liquidation call, "
            "not just the levered position's own VaR"
        )
        for lev in LEVERED_MULTIPLES:
            liquidation_analysis[f"leveraged_{lev}x_es_1pct_of_deployed_capital"] = (
                es_1pct * lev
            )
    else:
        liquidation_analysis["family"] = None
        liquidation_analysis["note"] = (
            "fit_risk_model returned None (insufficient obs or fit failure)"
        )
    results["5_liquidation_analysis"] = liquidation_analysis

    # ---- 6. by-year decomposition ----
    by_year_frame = timed_costed.join(
        featured.select("datetime").unique(), on="datetime", how="left"
    ).with_columns(pl.col("datetime").dt.year().alias("year"))
    by_year = {}
    for year in sorted(by_year_frame["year"].unique().to_list()):
        yr = by_year_frame.filter(pl.col("year") == year)
        m = research._series_metrics(
            yr["trade_log_return_net"], annualized_rate, f"year_{year}"
        )
        by_year[str(year)] = m
    results["6_by_year_decomposition"] = by_year
    print(
        f"by-year net Sharpe: {[(y, round(v['sharpe'], 3)) for y, v in by_year.items()]}"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
