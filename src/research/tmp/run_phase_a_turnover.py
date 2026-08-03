"""Phase A (checkpoint 1): cut turnover on notebook 3's cfg2_12h signal
without touching the signal itself, then apply Gate TC.

Design decision, stated explicitly per this repo's convention of naming
deviations from the runbook rather than silently making them: NEXT_PROMPT.md
suggests fanning Phase A out by hysteresis band, one subagent per band, on
the reasoning that "each band is an independent backtest." It is not, here -
per section 6's own rule ("Phase A changes only *how the signal is traded*,
never *what the signal is*"), every intervention in this file must share one
frozen signal (the model's predictions), generated ONCE per origin offset
with a fixed seed (`research.set_seed`). Retraining per band would silently
reintroduce cfg2_12h's own well-documented problem (the model is never
seeded in backtest_configs.py, so two "identical" runs disagree - see
003_cross_sectional_ic.md's inference-correction section) and would confound a
trading-mechanics change with fresh training noise, making the whole
experiment uninterpretable. So: predictions are generated once per origin
offset (the only genuinely compute-costly step - four small walk-forward
fits, seconds each on this hardware, not the "compute-heavy" regime the
runbook's 3-concurrent-agent cap is about), then every position-construction
variant (hysteresis bands, quantization, throttling, and their combination)
is evaluated against that single frozen signal per offset - cheap vectorized
polars/numpy work, not worth subagent fan-out overhead. This IS the fan-out
the runbook asks for in spirit (each variant is independent, evaluated in
isolation) without the accidental signal-refit confound fan-out-by-band
would have introduced.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
from alpha_lib7 import hysteresis_weights, quantize_weights, throttle_weights
from backtest_configs import (
    BARS_PER_DAY,
    GROSS_EXPOSURE,
    MAX_POSITION,
    ORIGIN_OFFSETS_DAYS,
    SLIPPAGE,
    TAKER_FEE,
    TOP_FRAC,
    TRAIN_TEST_DAYS,
    build_featured_panel,
    train_predict_fold,
)

import research

CONFIG_ID = "cfg2_12h"
INTERVAL = "12h"
FEATURE_COLS_RAW = [
    "mean_reversion_1", "mean_reversion_4", "mean_reversion_12",
    "realized_vol_8", "realized_vol_24", "realized_vol_96", "vol_of_vol_96",
]
VOL_COL = "realized_vol_24"
TARGET_COL = "fwd_return_1_vol_norm"

# Pre-declared, not tuned (NEXT_PROMPT.md section 4, Phase A).
HYSTERESIS_BANDS = [0.0, 0.05, 0.10, 0.15, 0.20]
QUANTIZE_GRID = 0.05
THROTTLE_KS = [1, 2, 3, 6]
# Pre-declared "in combination" variant: mid-points of the individually
# tested grids, chosen before seeing any Phase A number.
COMBINED = {"band": 0.10, "grid": QUANTIZE_GRID, "k": 3}

SEED = 0


def build_signal_panel(offset_days: int) -> pl.DataFrame:
    """Generate cfg2_12h's OOS predictions ONCE for this origin offset (fixed
    seed), stitched across folds with a 'fold' column retained so downstream
    metrics can still compute per-fold excess return the way
    backtest_configs.py's run_config does. This is 'the signal' - every Phase
    A variant below trades it differently but never regenerates it.
    """
    research.set_seed(SEED)
    feature_cols = [f"{c}_cs_z" for c in FEATURE_COLS_RAW]
    featured = build_featured_panel(INTERVAL, funding_by_symbol={})
    featured = featured.with_columns(
        research.vol_normalized_target(target_col="fwd_return_1", vol_col=VOL_COL)
    )
    needed = ["datetime", "symbol", "fwd_return_1", VOL_COL, *feature_cols, TARGET_COL]
    df = featured.select(needed).drop_nulls().sort(["datetime", "symbol"])
    df = df.filter(pl.col(VOL_COL) > 1e-12)

    vol_target = research._as_float(df[VOL_COL].median())
    bars_per_day = BARS_PER_DAY[INTERVAL]
    train_bars = TRAIN_TEST_DAYS[0] * bars_per_day
    test_bars = TRAIN_TEST_DAYS[1] * bars_per_day
    offset_bars = offset_days * bars_per_day

    splits = research.panel_walk_forward_splits(
        df, train_bars, test_bars, origin_offset=offset_bars
    )
    fold_frames = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        train_df, test_df = df[train_idx], df[test_idx]
        preds, _, _ = train_predict_fold(train_df, test_df, feature_cols, TARGET_COL)
        scored = test_df.with_columns(pl.Series("pred", preds)).with_columns(
            research.vol_targeted_size("pred", VOL_COL, vol_target)
        )
        fold_frames.append(
            scored.select(
                "datetime", "symbol", "pred", "vol_targeted_size", "fwd_return_1", VOL_COL
            ).with_columns(pl.lit(fold_id).alias("fold"))
        )
    panel = pl.concat(fold_frames, how="diagonal_relaxed").sort(["datetime", "symbol"])
    return panel


def fold_excess_returns(trade_frame_net: pl.DataFrame, signal_panel: pl.DataFrame) -> list[float]:
    """Per-fold (net strategy total - basket total) log return, matching
    backtest_configs.py's own excess_return_net convention - what Gate TC's
    bootstrap CI is computed over."""
    out = []
    for fold_id, fold_dates in (
        signal_panel.group_by("fold").agg(pl.col("datetime")).sort("fold").rows()
    ):
        fold_dates = set(fold_dates)
        strat_total = float(
            trade_frame_net.filter(pl.col("datetime").is_in(fold_dates))["trade_log_return_net"].sum()
        )
        basket = research.equal_weight_basket_returns(
            signal_panel.filter(pl.col("fold") == fold_id)
        )
        basket_total = float(basket["trade_log_return"].sum())
        out.append(strat_total - basket_total)
    return out


def evaluate_variant(name: str, weights: pl.DataFrame, signal_panel: pl.DataFrame, annualized_rate: float) -> dict:
    trade_frame = research.portfolio_trade_frame(weights, signal_panel, target_col="fwd_return_1")
    metrics = research.portfolio_metrics(
        trade_frame, annualized_rate, taker_fee=TAKER_FEE, slippage=SLIPPAGE, label=name
    )
    costed = research.add_portfolio_costs(trade_frame, TAKER_FEE, SLIPPAGE)
    excess = fold_excess_returns(costed, signal_panel)
    ci_lo, ci_hi = research.bootstrap_ci(np.array(excess), n_boot=2000, seed=0)

    net_sum = weights.group_by("datetime").agg(pl.col("weight").sum().alias("net")).select(
        pl.col("net").abs().max()
    ).item()

    return {
        "name": name,
        "sharpe_net": metrics.get("sharpe_net"),
        "sharpe_gross": metrics.get("sharpe"),
        "mean_turnover_per_bar": metrics.get("mean_turnover_per_bar"),
        "turnover_per_year": metrics.get("turnover_per_year"),
        "annual_fee_drag_pct": metrics.get("annual_fee_drag_pct"),
        "max_drawdown_net": metrics.get("max_drawdown_net"),
        "bootstrap_ci_excess_return": [ci_lo, ci_hi],
        "n_folds": len(excess),
        "max_abs_net_position_imbalance": float(net_sum),
    }


def main():
    annualized_rate = research.sharpe_to_annualized_rate(INTERVAL)
    results: dict = {"config_id": CONFIG_ID, "interval": INTERVAL, "by_offset": {}}

    for offset in ORIGIN_OFFSETS_DAYS:
        print(f"=== offset={offset}d: building signal (frozen, seed={SEED}) ===", flush=True)
        signal_panel = build_signal_panel(offset)

        variants: dict[str, dict] = {}

        # A0: baseline (band=0.0 must reproduce dollar_neutral_weights - the
        # correctness check already covered by tests/test_alpha_lib7.py;
        # here it's simply the notebook-3 baseline itself).
        baseline_weights = hysteresis_weights(
            signal_panel, "pred", band=0.0, size_col="vol_targeted_size",
            top_frac=TOP_FRAC, gross_exposure=GROSS_EXPOSURE, max_position_per_symbol=MAX_POSITION,
        )
        variants["A0_baseline_band0"] = evaluate_variant(
            "A0_baseline_band0", baseline_weights, signal_panel, annualized_rate
        )

        # A1: hysteresis bands
        for band in HYSTERESIS_BANDS:
            if band == 0.0:
                continue
            w = hysteresis_weights(
                signal_panel, "pred", band=band, size_col="vol_targeted_size",
                top_frac=TOP_FRAC, gross_exposure=GROSS_EXPOSURE, max_position_per_symbol=MAX_POSITION,
            )
            variants[f"A1_hysteresis_band{band}"] = evaluate_variant(
                f"A1_hysteresis_band{band}", w, signal_panel, annualized_rate
            )

        # A2: weight quantization (applied to the baseline)
        q = quantize_weights(baseline_weights, grid=QUANTIZE_GRID)
        variants[f"A2_quantize_grid{QUANTIZE_GRID}"] = evaluate_variant(
            f"A2_quantize_grid{QUANTIZE_GRID}", q, signal_panel, annualized_rate
        )

        # A3: rebalance throttling (applied to the baseline)
        for k in THROTTLE_KS:
            if k == 1:
                continue
            t = throttle_weights(baseline_weights, k=k)
            variants[f"A3_throttle_k{k}"] = evaluate_variant(
                f"A3_throttle_k{k}", t, signal_panel, annualized_rate
            )

        # Combined: pre-declared mid-point of each grid, tested once.
        combo_w = hysteresis_weights(
            signal_panel, "pred", band=COMBINED["band"], size_col="vol_targeted_size",
            top_frac=TOP_FRAC, gross_exposure=GROSS_EXPOSURE, max_position_per_symbol=MAX_POSITION,
        )
        combo_w = quantize_weights(combo_w, grid=COMBINED["grid"])
        combo_w = throttle_weights(combo_w, k=COMBINED["k"])
        variants["A4_combined"] = evaluate_variant("A4_combined", combo_w, signal_panel, annualized_rate)

        results["by_offset"][str(offset)] = variants

        print(f"  baseline: net={variants['A0_baseline_band0']['sharpe_net']:.3f} "
              f"turnover/yr={variants['A0_baseline_band0']['turnover_per_year']:.1f}", flush=True)
        for name, v in variants.items():
            if name == "A0_baseline_band0":
                continue
            print(f"  {name}: net={v['sharpe_net']:.3f} turnover/yr={v['turnover_per_year']:.1f} "
                  f"ci={v['bootstrap_ci_excess_return']}", flush=True)

    with open("src/research/tmp/phase_a_turnover_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten phase_a_turnover_results.json")


if __name__ == "__main__":
    main()
