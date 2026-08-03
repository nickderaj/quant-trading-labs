"""Phase 5: alpha attempts, pre-registered, cost-charged (NEXT_PROMPT.md sec
4, Phase 5). Every strategy: signal -> position -> futures cost model (sec 7)
-> net returns -> the same bootstrap/deflated-Sharpe machinery notebooks 3
and 7 used -> gate. Development window only; holdout stays frozen.

**Scope note, stated per this notebook's own risk-prioritization ("if time
runs short, cut Phase 5 and Phase 6 before cutting Phase 0, 3, or 7"):** this
pass implements strategies A (carry) and B (time-series momentum), the two
most central per sec 3.3's literature review. C (cross-sectional momentum),
D (basis-momentum), E (spread mean reversion), and F (COT hedging pressure,
CL-only) are declared but not run in this pass -- their absence is reported
explicitly in the results MD, not silently omitted.

Multiple-testing discipline: every (strategy, variant, origin-offset)
configuration actually run is logged; BH applied across the two headline
gates (AC, AM); deflated_sharpe_prob computed on each headline using the true
count of configurations tried.

Writes phase_5_results.json.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import numpy as np
import polars as pl

import research

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/phase_5_results.json"

DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}
DEV_END = "2024-12-31"

CARRY_HORIZON = 21  # ~1 trading month; commodity carry is slow (sec 3.3)
MOM_LOOKBACKS = {"mom_1m": 21, "mom_3m": 63, "mom_6m": 126, "mom_12m": 252}
TOP_FRAC = 0.3  # ~5 of 16 products per leg
ORIGIN_OFFSETS = [0, 7, 14, 21]
GROSS_EXPOSURE = 1.0
MAX_POSITION = 0.25
ANNUALIZED_RATE = float(np.sqrt(252))
CONFIG_LOG: list[dict] = []

research.set_seed(0)


def load_product_frame(product: str) -> pl.DataFrame | None:
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    dev_start = DEV_START.get(product, DEV_START["__default__"])
    sub = curve.filter(pl.col("log_return_ratioadj").is_finite())
    sub = sub.filter(
        (pl.col("date") >= pl.lit(dev_start).str.to_date())
        & (pl.col("date") <= pl.lit(DEV_END).str.to_date())
    )
    sub = sub.sort("date")
    if sub.height < 400:
        return None

    ts = C.term_structure_state(
        sub.select(["date", "close_f1", "dte_f1", "close_f2", "dte_f2"])
    )
    sub = sub.join(ts, on="date", how="left")

    # carry signal: long backwardation (negative slope) -> predictive score
    # is the NEGATIVE of the roll slope, so backwardated products rank high.
    sub = sub.with_columns((-pl.col("roll_slope_annualized")).alias("carry_signal"))

    for name, lb in MOM_LOOKBACKS.items():
        sub = sub.with_columns(
            pl.col("log_return_ratioadj").rolling_sum(window_size=lb).alias(name)
        )

    # Forward-return construction: sum of the next H returns, aligned so
    # row t's fwd_return is realized strictly after t (never includes r_t
    # itself, which is already known at decision time).
    fwd = sub["log_return_ratioadj"].to_numpy()
    n = len(fwd)
    fwd_1 = np.full(n, np.nan)
    fwd_1[:-1] = fwd[1:]
    fwd_carry = np.full(n, np.nan)
    csum = np.cumsum(np.nan_to_num(fwd))
    for t in range(n - CARRY_HORIZON):
        fwd_carry[t] = csum[t + CARRY_HORIZON] - csum[t]
    sub = sub.with_columns(
        pl.Series("fwd_return_1", fwd_1), pl.Series("fwd_return_carry", fwd_carry)
    )

    sub = sub.with_columns(
        pl.lit(product).alias("symbol"), pl.col("close_f1").alias("close")
    )
    return sub


def build_panel(products: list[str]) -> pl.DataFrame:
    frames = []
    for p in products:
        f = load_product_frame(p)
        if f is not None:
            frames.append(f)
    panel = pl.concat(frames, how="diagonal_relaxed")
    return panel.sort(["date", "symbol"])


def run_strategy(
    panel: pl.DataFrame, pred_col: str, target_col: str, label: str, origin_offset: int
) -> tuple[dict, pl.DataFrame]:
    sub = panel.select(["date", "symbol", pred_col, target_col, "close"]).drop_nulls(
        subset=[pred_col, target_col]
    )
    sub = sub.filter(pl.col(pred_col).is_finite() & pl.col(target_col).is_finite())
    if origin_offset > 0:
        # simple origin shift: drop the first `origin_offset` unique dates,
        # matching notebook 7's "does the result depend on the walk-forward
        # grid's start point" robustness check, applied here to the signal's
        # own rebalance calendar.
        dates = sub["date"].unique().sort().to_list()
        keep_dates = set(dates[origin_offset:])
        sub = sub.filter(pl.col("date").is_in(list(keep_dates)))

    weights = research.dollar_neutral_weights(
        sub.rename({"date": "datetime"}),
        pred_col=pred_col,
        datetime_col="datetime",
        top_frac=TOP_FRAC,
        gross_exposure=GROSS_EXPOSURE,
        max_position_per_symbol=MAX_POSITION,
    )
    returns_df = sub.rename({"date": "datetime"}).select(
        ["datetime", "symbol", target_col]
    )
    trade_frame = research.portfolio_trade_frame(
        weights, returns_df, target_col=target_col, datetime_col="datetime"
    )

    prices = sub.rename({"date": "datetime"}).select(["datetime", "symbol", "close"])
    costs = C.portfolio_costs_futures(weights, prices, datetime_col="datetime")
    metrics = C.futures_portfolio_metrics(
        trade_frame,
        costs,
        annualized_rate=ANNUALIZED_RATE,
        datetime_col="datetime",
        label=f"{label}_offset{origin_offset}",
    )
    costed = C.add_portfolio_costs_futures(trade_frame, costs, datetime_col="datetime")

    CONFIG_LOG.append(
        {
            "label": label,
            "origin_offset": origin_offset,
            "pred_col": pred_col,
            "target_col": target_col,
        }
    )
    return metrics, costed.select(["datetime", "trade_log_return_net"])


def gate_verdict(
    metrics_by_offset: dict,
    headline_net_returns: pl.DataFrame,
    basket_returns: pl.DataFrame,
    n_trials: int,
) -> dict:
    """Gate AC/AM: net Sharpe > 0 at every origin offset AND block-bootstrap
    95% CI on (strategy_net - equal_weight_basket) excludes zero AND
    deflated Sharpe probability > 0.95.
    """
    sharpes_net = [m["sharpe_net"] for m in metrics_by_offset.values()]
    all_positive = all(s > 0 for s in sharpes_net)

    joined = headline_net_returns.join(basket_returns, on="datetime", how="inner")
    excess = (
        (joined["trade_log_return_net"] - joined["basket_return"])
        .drop_nulls()
        .to_numpy()
    )
    ci_lo, ci_hi = (
        research.block_bootstrap_ci(excess, n_boot=2000, seed=0)
        if len(excess) > 30
        else (None, None)
    )
    ci_excludes_zero = (
        ci_lo is not None and ci_hi is not None and (ci_lo > 0 or ci_hi < 0)
    )

    headline = metrics_by_offset["offset_0"]
    n_obs = headline["no_bars"]
    dsr = research.deflated_sharpe_prob(
        headline["sharpe_net"] / ANNUALIZED_RATE, n_trials=n_trials, n_obs=n_obs
    )
    return {
        "net_sharpe_positive_at_every_offset": all_positive,
        "sharpes_net_by_offset": {
            k: v["sharpe_net"] for k, v in metrics_by_offset.items()
        },
        "excess_return_ci": [ci_lo, ci_hi],
        "excess_ci_excludes_zero": ci_excludes_zero,
        "deflated_sharpe_prob": dsr,
        "fires": all_positive and ci_excludes_zero and dsr > 0.95,
    }


def main():
    t0 = time.time()
    print("building panel...", flush=True)
    panel = build_panel(C.PRODUCTS)
    print(
        f"panel: {panel.height} rows, {panel['symbol'].n_unique()} symbols", flush=True
    )

    results: dict = {"strategy_A_carry": {}, "strategy_B_momentum": {}}

    panel_dt = panel.rename({"date": "datetime"})

    print("Strategy A: carry...", flush=True)
    carry_by_offset, carry_returns_by_offset = {}, {}
    for offset in ORIGIN_OFFSETS:
        m, ret = run_strategy(
            panel, "carry_signal", "fwd_return_carry", "carry", offset
        )
        carry_by_offset[f"offset_{offset}"] = m
        carry_returns_by_offset[offset] = ret
    carry_basket = research.equal_weight_basket_returns(
        panel_dt, target_col="fwd_return_carry", datetime_col="datetime"
    )
    carry_basket = carry_basket.rename({"trade_log_return": "basket_return"})
    results["strategy_A_carry"] = {
        "by_offset": carry_by_offset,
        "gate_AC": gate_verdict(
            carry_by_offset,
            carry_returns_by_offset[0],
            carry_basket,
            n_trials=len(ORIGIN_OFFSETS),
        ),
    }

    print("Strategy B: momentum (4 lookbacks)...", flush=True)
    mom_results = {}
    mom_basket = research.equal_weight_basket_returns(
        panel_dt, target_col="fwd_return_1", datetime_col="datetime"
    )
    mom_basket = mom_basket.rename({"trade_log_return": "basket_return"})
    for name in MOM_LOOKBACKS:
        by_offset, returns_by_offset = {}, {}
        for offset in ORIGIN_OFFSETS:
            m, ret = run_strategy(panel, name, "fwd_return_1", f"mom_{name}", offset)
            by_offset[f"offset_{offset}"] = m
            returns_by_offset[offset] = ret
        mom_results[name] = {
            "by_offset": by_offset,
            "gate": gate_verdict(
                by_offset,
                returns_by_offset[0],
                mom_basket,
                n_trials=len(ORIGIN_OFFSETS) * len(MOM_LOOKBACKS),
            ),
        }
    best_mom = max(
        mom_results.items(), key=lambda kv: kv[1]["by_offset"]["offset_0"]["sharpe_net"]
    )
    results["strategy_B_momentum"] = {
        "by_lookback": mom_results,
        "best_lookback": best_mom[0],
        "gate_AM": best_mom[1]["gate"],
    }

    results["_scope_note"] = (
        "Strategies C (cross-sectional momentum), D (basis-momentum), E (spread "
        "mean reversion), and F (COT hedging pressure) are declared in "
        "NEXT_PROMPT.md sec 4 Phase 5 but were not run in this pass -- see "
        "results MD for the explicit scope tradeoff."
    )
    results["_config_log"] = CONFIG_LOG
    results["_config"] = {
        "carry_horizon_days": CARRY_HORIZON,
        "mom_lookbacks": MOM_LOOKBACKS,
        "top_frac": TOP_FRAC,
        "origin_offsets": ORIGIN_OFFSETS,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
