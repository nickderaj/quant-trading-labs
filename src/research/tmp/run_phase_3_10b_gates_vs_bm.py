"""10b Phase 3: Gates VS and BM (NEXT_PROMPT.md sec 6 Phase 3). Both reuse
notebook 8's own carry/momentum panel and futures cost model unmodified
(`run_phase_5_alpha.py`'s `build_panel`/`load_product_frame`) -- only the
weighting/aggregation rule changes, per NEXT_PROMPT.md's own instruction.

**Gate VS (volatility-scaled carry)**: identical signal (`carry_signal`,
21-day horizon) and universe to notebook 8's Gate AC, but
`research.dollar_neutral_weights`' `size_col` is set to each product's own
inverse trailing-20-day realized volatility (of `log_return_ratioadj`) --
standard inverse-vol position sizing, so a within-leg allocation is smaller
for a noisier product and larger for a calmer one, instead of nb8's original
equal-weight-within-leg carry book. DSR n_trials = 8 (4 already-logged nb8
carry configs forwarded + 4 new vol-scaled configs, one per origin offset --
phase_5_10a_results.json).

**Gate BM (blended momentum)**: an equal-weighted average of the SAME four
lookback signals (`mom_1m`, `mom_3m`, `mom_6m`, `mom_12m`) notebook 8 already
computed, as one new blended predictor, ranked/costed exactly like any
single-lookback momentum config. DSR n_trials = 20 (16 already-logged nb8
momentum configs -- the literal historical n_trials, not NEXT_PROMPT's own
summarized "4" -- forwarded + 4 new blend configs, one per origin offset).

Writes phase_3_10b_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import numpy as np
import polars as pl
import run_phase_5_alpha as A

import research

OUT_PATH = "src/research/tmp/phase_3_10b_results.json"
ORIGIN_OFFSETS = [0, 7, 14, 21]
ANNUALIZED_RATE = float(np.sqrt(252))
VOL_WINDOW = 20
N_TRIALS_VS = 8  # 4 forwarded (nb8 gate_AC) + 4 new -- phase_5_10a_results.json
N_TRIALS_BM = 20  # 16 forwarded (nb8 gate_AM) + 4 new -- phase_5_10a_results.json

research.set_seed(0)


def build_extended_panel() -> pl.DataFrame:
    panel = A.build_panel(C.PRODUCTS)
    panel = panel.sort(["symbol", "date"])
    panel = panel.with_columns(
        pl.col("log_return_ratioadj").rolling_std(window_size=VOL_WINDOW).over("symbol").alias("vol_20d")
    )
    panel = panel.with_columns((1.0 / pl.col("vol_20d").clip(lower_bound=1e-6)).alias("inv_vol_20d"))
    blend = pl.mean_horizontal([pl.col(c) for c in A.MOM_LOOKBACKS])
    panel = panel.with_columns(blend.alias("mom_blend"))
    return panel


def run_strategy_sized(panel: pl.DataFrame, pred_col: str, target_col: str, size_col: str | None, label: str, origin_offset: int):
    sub = panel.select(["date", "symbol", pred_col, target_col, "close"] + ([size_col] if size_col else [])).drop_nulls(
        subset=[pred_col, target_col] + ([size_col] if size_col else [])
    )
    sub = sub.filter(pl.col(pred_col).is_finite() & pl.col(target_col).is_finite())
    if origin_offset > 0:
        dates = sub["date"].unique().sort().to_list()
        keep_dates = set(dates[origin_offset:])
        sub = sub.filter(pl.col("date").is_in(list(keep_dates)))

    weights = research.dollar_neutral_weights(
        sub.rename({"date": "datetime"}), pred_col=pred_col, datetime_col="datetime",
        top_frac=A.TOP_FRAC, size_col=size_col, gross_exposure=A.GROSS_EXPOSURE, max_position_per_symbol=A.MAX_POSITION,
    )
    returns_df = sub.rename({"date": "datetime"}).select(["datetime", "symbol", target_col])
    trade_frame = research.portfolio_trade_frame(weights, returns_df, target_col=target_col, datetime_col="datetime")

    prices = sub.rename({"date": "datetime"}).select(["datetime", "symbol", "close"])
    costs = C.portfolio_costs_futures(weights, prices, datetime_col="datetime")
    metrics = C.futures_portfolio_metrics(trade_frame, costs, annualized_rate=ANNUALIZED_RATE, datetime_col="datetime", label=f"{label}_offset{origin_offset}")
    costed = C.add_portfolio_costs_futures(trade_frame, costs, datetime_col="datetime")
    return metrics, costed.select(["datetime", "trade_log_return_net"])


def gate_verdict(metrics_by_offset: dict, headline_net_returns: pl.DataFrame, basket_returns: pl.DataFrame, n_trials: int) -> dict:
    sharpes_net = [m["sharpe_net"] for m in metrics_by_offset.values()]
    all_positive = all(s > 0 for s in sharpes_net)

    joined = headline_net_returns.join(basket_returns, on="datetime", how="inner")
    excess = (joined["trade_log_return_net"] - joined["basket_return"]).drop_nulls().to_numpy()
    ci_lo, ci_hi = research.block_bootstrap_ci(excess, n_boot=2000, seed=0) if len(excess) > 30 else (None, None)
    ci_excludes_zero = ci_lo is not None and ci_hi is not None and (ci_lo > 0 or ci_hi < 0)

    headline = metrics_by_offset["offset_0"]
    n_obs = headline["no_bars"]
    dsr = research.deflated_sharpe_prob(headline["sharpe_net"] / ANNUALIZED_RATE, n_trials=n_trials, n_obs=n_obs)
    return {
        "net_sharpe_positive_at_every_offset": all_positive,
        "sharpes_net_by_offset": {k: v["sharpe_net"] for k, v in metrics_by_offset.items()},
        "excess_return_ci": [ci_lo, ci_hi],
        "excess_ci_excludes_zero": ci_excludes_zero,
        "deflated_sharpe_prob": dsr,
        "n_trials": n_trials,
        "fires": bool(all_positive and ci_excludes_zero and dsr > 0.95),
    }


def main():
    panel = build_extended_panel()
    panel_dt = panel.rename({"date": "datetime"})

    print("Gate VS: vol-scaled carry...", flush=True)
    vs_by_offset, vs_returns_by_offset = {}, {}
    for offset in ORIGIN_OFFSETS:
        m, ret = run_strategy_sized(panel, "carry_signal", "fwd_return_carry", "inv_vol_20d", "carry_vs", offset)
        vs_by_offset[f"offset_{offset}"] = m
        vs_returns_by_offset[offset] = ret
    carry_basket = research.equal_weight_basket_returns(panel_dt, target_col="fwd_return_carry", datetime_col="datetime")
    carry_basket = carry_basket.rename({"trade_log_return": "basket_return"})
    gate_vs = gate_verdict(vs_by_offset, vs_returns_by_offset[0], carry_basket, n_trials=N_TRIALS_VS)

    print("Gate BM: blended momentum...", flush=True)
    bm_by_offset, bm_returns_by_offset = {}, {}
    for offset in ORIGIN_OFFSETS:
        m, ret = run_strategy_sized(panel, "mom_blend", "fwd_return_1", None, "mom_blend", offset)
        bm_by_offset[f"offset_{offset}"] = m
        bm_returns_by_offset[offset] = ret
    mom_basket = research.equal_weight_basket_returns(panel_dt, target_col="fwd_return_1", datetime_col="datetime")
    mom_basket = mom_basket.rename({"trade_log_return": "basket_return"})
    gate_bm = gate_verdict(bm_by_offset, bm_returns_by_offset[0], mom_basket, n_trials=N_TRIALS_BM)

    results = {
        "gate_VS": {"by_offset": vs_by_offset, "gate": gate_vs},
        "gate_BM": {"by_offset": bm_by_offset, "gate": gate_bm},
        "_note": "Gate VS reuses nb8's carry panel with inverse-20d-vol within-leg sizing; Gate BM reuses nb8's momentum panel with an equal-weighted blend of the 4 lookback signals as one new predictor.",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"written {OUT_PATH}")
    print(f"Gate VS fires={gate_vs['fires']} dsr={gate_vs['deflated_sharpe_prob']:.4f} sharpes={gate_vs['sharpes_net_by_offset']}")
    print(f"Gate BM fires={gate_bm['fires']} dsr={gate_bm['deflated_sharpe_prob']:.4f} sharpes={gate_bm['sharpes_net_by_offset']}")


if __name__ == "__main__":
    main()
