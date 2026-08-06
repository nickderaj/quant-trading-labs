"""Notebook 13, Design A -- forecast-to-fill trend/momentum on GC futures
(NEXT_PROMPT.md sec4.A). The claim under test: the alpha is not in the
signal, it is in the fill -- a deliberately plain trend-momentum state,
converted to positions through vol targeting, fractional-Kelly sizing, and
a square-root impact discount, with stops filled under the required
worse-of-(stop, gapped-open) convention (sec7 trap1).

Position construction, sequentially, matching the design spec's own order:
  1. state = exec_lib13.trend_momentum_state (single smoothed regime state)
  2. vol_target_position = clip(state,-1,1) * (vol_target / realized_vol),
     the research.vol_targeted_size convention.
  3. kelly_multiplier: a NON-negative [0,1] confidence overlay from
     exec_lib13.fractional_kelly_scalar(edge=trailing mean return,
     vol=trailing vol, kelly_fraction=0.25) -- direction comes from the
     state alone (step 2); Kelly only throttles size by trailing
     edge-per-variance quality. Disclosed simplification, not a
     re-derivation of Kelly from scratch.
  4. impact_discount = exec_lib13.sqrt_impact_discount on intended notional
     (position * ASSUMED_AUM) vs trailing dollar volume.
  5. Exits: exec_lib13.trailing_atr_stop + apply_stop_fill (required
     worse-of convention headline; optimistic convention as a sensitivity
     row only).

Cost model: commod_lib8.round_turn_cost_per_contract / cost_per_unit_notional
is the headline (per bar, varying with price -- NOT a constant bps
convention). The design's own 0.7bp+impact assumption never appears as a
headline number, only as a labelled sensitivity row.

Walk-forward: research.walk_forward_splits, rolling 10y train / 6mo test.
No parameter in the position-construction pipeline is fit on the train
window (they are declared constants, sec4.A) -- the walk-forward split
exists only to mark which OOS bars we're honestly allowed to report, per
sec4.A's instruction not to shorten the training window to manufacture
more OOS days.

Writes phase_A_13_results.json.
"""

import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import exec_lib13 as E

import research

research.set_seed(0)

ANNUALIZED_RATE = float(np.sqrt(252))
ORIGIN_OFFSETS = [0, 7, 14, 21]
VOL_TARGET = 0.15 / np.sqrt(252)  # daily vol target implied by 15% annualized
KELLY_FRACTION = 0.25
IMPACT_COEFFICIENT = 0.1
ATR_MULT = 3.0
ASSUMED_AUM = 10_000_000.0  # disclosed sizing base for impact/capacity, not fit
TRAIN_BARS = 10 * 252
TEST_BARS = 126
N_TRIALS_POOLED = 18


def load_gc_continuous() -> pl.DataFrame:
    ohlcv = pl.read_parquet("src/research/data/market/databento/ohlcv/GC.parquet")
    contracts = pl.read_parquet("src/research/data/market/databento/contracts.parquet")
    roll = pl.read_parquet("src/research/data/market/databento/roll_calendar.parquet")
    df = ohlcv.filter(pl.col("product") == "GC")
    clean = C.apply_hygiene_filter(df)
    cont = C.build_continuous_series_ohlcv(clean, contracts, roll, "GC")
    cont = cont.sort("date").drop_nulls(
        subset=["close_backadj", "high_backadj", "low_backadj", "open_backadj"]
    )
    return cont.filter(pl.col("date") <= pl.date(2024, 12, 31))


def load_gc_spot_daily() -> pl.DataFrame:
    spot = pl.read_parquet("src/research/data/market/yfinance/daily/GC=F.parquet")
    spot = spot.select(
        pl.col("timestamp").cast(pl.Date).alias("date"),
        pl.col("close").alias("spot_close"),
    ).sort("date")
    return spot.with_columns(
        (pl.col("spot_close").log() - pl.col("spot_close").log().shift(1)).alias(
            "spot_log_return"
        )
    )


def build_position(
    df: pl.DataFrame,
    use_smoothing: bool = True,
    use_impact: bool = True,
    use_kelly: bool = True,
) -> dict[str, np.ndarray]:
    close = df["close_backadj"].to_numpy()
    high = df["high_backadj"].to_numpy()
    low = df["low_backadj"].to_numpy()
    volume = df["volume"].to_numpy()

    smooth_span = 10 if use_smoothing else 1
    state = E.trend_momentum_state(close, high=high, low=low, smooth_span=smooth_span)

    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    log_ret[0] = 0.0
    realized_vol = pl.Series(log_ret).ewm_std(span=20).shift(1).to_numpy()

    clipped_state = np.clip(state, -1, 1)
    vol_target_position = clipped_state * (
        VOL_TARGET / np.where(realized_vol > 1e-12, realized_vol, np.nan)
    )

    if use_kelly:
        trailing_mean = (
            pl.Series(log_ret).rolling_mean(window_size=20).shift(1).to_numpy()
        )
        kelly_raw = E.fractional_kelly_scalar(
            trailing_mean, realized_vol, KELLY_FRACTION
        )
        kelly_multiplier = np.clip(np.abs(kelly_raw), 0.0, 1.0)
        kelly_multiplier = np.where(
            np.isfinite(kelly_multiplier), kelly_multiplier, 0.0
        )
    else:
        kelly_multiplier = np.ones_like(vol_target_position)

    pre_impact_position = vol_target_position * kelly_multiplier

    if use_impact:
        dollar_volume = volume * close
        adv = pl.Series(dollar_volume).rolling_mean(window_size=20).shift(1).to_numpy()
        intended_notional = np.abs(pre_impact_position) * ASSUMED_AUM
        discount = E.sqrt_impact_discount(intended_notional, adv, IMPACT_COEFFICIENT)
    else:
        discount = np.ones_like(pre_impact_position)

    final_position_raw = pre_impact_position * discount
    final_position_raw = np.where(
        np.isfinite(final_position_raw), final_position_raw, 0.0
    )
    return {
        "state": state,
        "final_position_raw": final_position_raw,
        "close": close,
        "high": high,
        "low": low,
        "open": df["open_backadj"].to_numpy(),
        "volume": volume,
        "realized_vol": realized_vol,
    }


def apply_stops_and_simulate(
    built: dict, optimistic_stop: bool = False
) -> dict[str, np.ndarray]:
    """Bar-by-bar: desired position sign/size from `final_position_raw`, but
    once a stop fires while holding, force flat and STAY flat until the
    desired position's sign flips (a genuine signal reversal, not just the
    stop resetting) -- otherwise the same whipsaw the smoothing (sec4.A) is
    meant to prevent would reappear through the stop-and-reenter path.
    """
    close, high, low, open_ = built["close"], built["high"], built["low"], built["open"]
    desired = built["final_position_raw"]
    n = len(close)
    atr_series = E.true_atr_series(
        high, low, close, window=14
    )  # already causal (shift(1))

    realized_position = np.zeros(n)
    stop_level = np.full(n, np.nan)
    forced_flat_until_flip = False
    last_flip_sign = 0.0

    prev_sign = 0.0
    for t in range(1, n):
        d_sign = np.sign(desired[t]) if np.isfinite(desired[t]) else 0.0

        if forced_flat_until_flip:
            if d_sign != 0 and d_sign != last_flip_sign:
                forced_flat_until_flip = False
            else:
                realized_position[t] = 0.0
                prev_sign = 0.0
                continue

        held_sign = prev_sign
        if held_sign != 0:
            atr = atr_series[t]
            if np.isfinite(atr):
                if held_sign > 0:
                    candidate = close[t - 1] - ATR_MULT * atr
                    stop_level[t] = (
                        candidate
                        if not np.isfinite(stop_level[t - 1])
                        else max(stop_level[t - 1], candidate)
                    )
                else:
                    candidate = close[t - 1] + ATR_MULT * atr
                    stop_level[t] = (
                        candidate
                        if not np.isfinite(stop_level[t - 1])
                        else min(stop_level[t - 1], candidate)
                    )

            exit_mask, _ = E.apply_stop_fill(
                np.array([open_[t]]),
                np.array([high[t]]),
                np.array([low[t]]),
                np.array([stop_level[t]]),
                np.array([held_sign]),
                optimistic=optimistic_stop,
            )
            if exit_mask[0]:
                realized_position[t] = held_sign  # held into the stop-out bar
                forced_flat_until_flip = True
                last_flip_sign = held_sign
                prev_sign = 0.0
                stop_level[t] = np.nan
                continue

        realized_position[t] = desired[t] if np.isfinite(desired[t]) else 0.0
        prev_sign = (
            np.sign(realized_position[t]) if np.isfinite(realized_position[t]) else 0.0
        )
        if prev_sign == 0:
            stop_level[t] = np.nan

    return {"realized_position": realized_position, "stop_level": stop_level}


def bar_returns_with_costs(
    close: np.ndarray, position: np.ndarray
) -> dict[str, np.ndarray]:
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    log_ret[0] = 0.0
    gross = (
        position[:-1] * log_ret[1:]
    )  # position held through bar t-1 earns bar t's return
    gross = np.concatenate([[0.0], gross])

    cost_frac = np.array([C.cost_per_unit_notional("GC", p) for p in close])
    turnover = np.abs(np.diff(position, prepend=0.0))
    cost_log = np.log(np.clip(1 - cost_frac * turnover, 1e-6, None))
    net = gross + cost_log
    return {"gross": gross, "net": net, "turnover": turnover}


def series_metrics(x: np.ndarray, label: str) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"label": label, "no_bars": 0}
    std = float(np.std(x))
    mean = float(np.mean(x))
    cum = np.cumsum(x)
    dd = cum - np.maximum.accumulate(cum)
    return {
        "label": label,
        "no_bars": len(x),
        "sharpe": (mean / std) * ANNUALIZED_RATE if std > 0 else 0.0,
        "total_log_return": float(np.sum(x)),
        "compound_return": float(np.exp(np.sum(x)) - 1),
        "max_drawdown": float(np.min(dd)) if len(dd) else 0.0,
        "mean": mean,
        "std": std,
    }


def run_trial(
    df: pl.DataFrame,
    name: str,
    use_smoothing=True,
    use_impact=True,
    use_kelly=True,
    optimistic_stop=False,
):
    built = build_position(df, use_smoothing, use_impact, use_kelly)
    sim = apply_stops_and_simulate(built, optimistic_stop=optimistic_stop)
    rc = bar_returns_with_costs(built["close"], sim["realized_position"])

    n = len(built["close"])
    splits = research.walk_forward_splits(n, TRAIN_BARS, TEST_BARS, mode="rolling")
    if not splits:
        raise RuntimeError("no walk-forward folds fit in the available GC history")
    oos_idx = np.unique(np.concatenate([test for _, test in splits]))

    by_offset = {}
    for offset in ORIGIN_OFFSETS:
        splits_o = research.walk_forward_splits(
            n, TRAIN_BARS, TEST_BARS, mode="rolling", origin_offset=offset
        )
        idx_o = (
            np.unique(np.concatenate([test for _, test in splits_o]))
            if splits_o
            else np.array([], dtype=int)
        )
        net_o = rc["net"][idx_o]
        by_offset[f"offset_{offset}"] = series_metrics(net_o, f"{name}_offset{offset}")

    net_oos = rc["net"][oos_idx]
    gross_oos = rc["gross"][oos_idx]
    ci_lo, ci_hi = research.block_bootstrap_ci(net_oos[np.isfinite(net_oos)], seed=0)
    dsr = research.deflated_sharpe_prob(
        by_offset["offset_0"]["sharpe"] / ANNUALIZED_RATE,
        n_trials=N_TRIALS_POOLED,
        n_obs=by_offset["offset_0"]["no_bars"],
    )

    return {
        "name": name,
        "n_oos_bars": len(oos_idx),
        "oos_date_range": [
            str(df["date"][int(oos_idx.min())]),
            str(df["date"][int(oos_idx.max())]),
        ],
        "net": series_metrics(net_oos, name),
        "gross": series_metrics(gross_oos, f"{name}_gross"),
        "by_offset": by_offset,
        "sharpe_ci_95": [ci_lo, ci_hi],
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "dsr": dsr,
        "mean_turnover_per_bar": float(np.mean(rc["turnover"][oos_idx])),
        "oos_idx": oos_idx,  # kept only for in-process reuse, stripped before serialization
        "net_oos_returns": net_oos,
    }


def benchmark_neutrality(
    dates: list, net_returns: np.ndarray, spot: pl.DataFrame
) -> dict:
    df = (
        pl.DataFrame({"date": dates, "strat": net_returns})
        .join(spot, on="date", how="inner")
        .drop_nulls()
    )
    if df.height < 30:
        return {
            "beta": None,
            "alpha_annualized": None,
            "information_ratio": None,
            "n_obs": df.height,
        }
    y = df["strat"].to_numpy()
    x = df["spot_log_return"].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    ir = (
        float(np.mean(resid) / np.std(resid) * ANNUALIZED_RATE)
        if np.std(resid) > 0
        else 0.0
    )
    return {
        "beta": float(beta),
        "alpha_annualized": float(alpha * 252),
        "information_ratio": ir,
        "n_obs": df.height,
    }


def main():
    gc = load_gc_continuous()
    spot = load_gc_spot_daily()

    trials = {}
    trials["base"] = run_trial(gc, "base")
    trials["no_smoothing"] = run_trial(gc, "no_smoothing", use_smoothing=False)
    trials["no_impact"] = run_trial(gc, "no_impact", use_impact=False)
    trials["no_kelly"] = run_trial(gc, "no_kelly", use_kelly=False)

    base = trials["base"]
    dates_all = gc["date"].to_list()
    oos_dates = [dates_all[i] for i in base["oos_idx"]]
    bench = benchmark_neutrality(oos_dates, base["net_oos_returns"], spot)

    built_base = build_position(gc)
    sim_base = apply_stops_and_simulate(built_base, optimistic_stop=False)
    dollar_volume = built_base["volume"] * built_base["close"]
    capacity = E.capacity_curve(
        base_notional=float(
            np.nanmean(np.abs(sim_base["realized_position"][base["oos_idx"]]))
            * ASSUMED_AUM
        ),
        adv_notional=dollar_volume[base["oos_idx"]],
        base_sharpe=base["net"]["sharpe"],
        impact_coefficient=IMPACT_COEFFICIENT,
    )

    sim_optimistic = apply_stops_and_simulate(built_base, optimistic_stop=True)
    rc_optimistic = bar_returns_with_costs(
        built_base["close"], sim_optimistic["realized_position"]
    )
    optimistic_metrics = series_metrics(
        rc_optimistic["net"][base["oos_idx"]], "base_optimistic_stop_fill"
    )

    cost_sens = {}
    gross_ret = np.diff(
        np.log(built_base["close"]), prepend=np.log(built_base["close"][0])
    )
    gross_ret[0] = np.nan
    pos = sim_base["realized_position"]
    gross_pnl = np.concatenate([[0.0], pos[:-1] * gross_ret[1:]])
    turnover = np.abs(np.diff(pos, prepend=0.0))
    sensitivity_cost_bps = 0.7  # design's own claimed cost, sensitivity only
    net_sens = gross_pnl + np.log(
        np.clip(1 - (sensitivity_cost_bps / 1e4) * turnover, 1e-6, None)
    )
    cost_sens["design_claimed_0_7bp_plus_impact_sensitivity"] = series_metrics(
        net_sens[base["oos_idx"]], "base_sensitivity_cost"
    )

    def strip(t: dict) -> dict:
        return {k: v for k, v in t.items() if k not in ("oos_idx", "net_oos_returns")}

    out = {
        "gate": "FF",
        "n_trials_pooled": N_TRIALS_POOLED,
        "assumed_aum_usd": ASSUMED_AUM,
        "stop_fill_convention": "required (worse of stop, gapped open)",
        "trials": {k: strip(v) for k, v in trials.items()},
        "benchmark_neutrality_vs_spot_GCF": bench,
        "capacity": capacity,
        "stop_fill_sensitivity_optimistic": optimistic_metrics,
        "cost_sensitivity": cost_sens,
        "arithmetic_red_flag_check": {
            "reported_max_drawdown": -0.0052,
            "our_max_drawdown_net": base["net"]["max_drawdown"],
            "our_sharpe_net": base["net"]["sharpe"],
            "verdict": (
                "plausible -- our drawdown is in a normal range for the achieved Sharpe, "
                "not evidence of a stop-fill leak"
                if base["net"]["max_drawdown"] < -0.02
                else "SUSPECT -- drawdown implausibly small relative to achieved Sharpe, "
                "matches the leak signature sec4.A pre-registered; route to Phase L before reporting"
            ),
        },
        "gate_FF_fires": bool(
            trials["base"]["ci_excludes_zero"]
            and trials["base"]["dsr"] >= 0.95
            and bench["beta"] is not None
            and abs(bench["beta"]) < 0.15
            and len(
                {
                    np.sign(trials["base"]["by_offset"][f"offset_{o}"]["sharpe"])
                    for o in ORIGIN_OFFSETS
                }
            )
            == 1
        ),
    }

    with open("src/research/tmp/phase_A_13_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "trials"}, indent=2, default=str
        )[:3000]
    )
    print(
        "\nbase net sharpe:",
        trials["base"]["net"]["sharpe"],
        "n_oos_bars:",
        trials["base"]["n_oos_bars"],
    )


if __name__ == "__main__":
    main()
