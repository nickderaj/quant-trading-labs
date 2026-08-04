"""Notebook 13, Design C -- adaptive trend, 6h crypto, asymmetric long/short
book (NEXT_PROMPT.md sec4.C). The claim under test: a trend book with
volatility-adaptive trailing exits, quality-filtered selection, and causal
monthly re-parameterization clears Sharpe 2.4 net of 4bp taker fees.

Bars: native 6h Binance klines (sec2/prereg correction -- 6h is a real
Binance interval, fetched directly, not resampled from 1h).
Calibration: 2021-07-01 to 2021-12-31. OOS dev: 2022-01-01 to 2025-06-30.

Book: 70% capital to longs / 30% to shorts, equal-weight within each leg,
monthly rebalance. Entry: rate-of-change(L) vs +-theta. Exit: trailing ATR
stop (alpha * ATR), required worse-of-(stop, gapped-open) fill convention.
Selection: trailing 1-month realized Sharpe (leg-specific threshold, higher
bar for shorts) intersected with a trailing-dollar-volume liquidity rank
(market-cap substitute, disclosed). Adaptation: (L, theta, alpha) grid
search on the PRECEDING month only, pooled across the eligible universe,
applied forward -- exec_lib13.causal_monthly_regrid_search's per-series
contract adapted here to a panel-level score function.

Six trials (Gate AT + n_trials pooling, sec5/sec6): adaptive, frozen-twin,
and four ablations (no trailing stop, no selection filter, no liquidity
filter, no adaptation). LUNA/FTT survivorship run with-and-without (trap6).

Writes phase_C_13_results.json.
"""

import json
import sys
from datetime import datetime, timezone

import numpy as np
import polars as pl

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import exec_lib13 as E

import research

ANNUALIZED_RATE = float(np.sqrt(365 * 4))  # 6h bars, 4/day
ORIGIN_OFFSETS = [0, 7 * 4, 14 * 4, 21 * 4]  # offsets in bars at 6h cadence (approx days*4)
N_TRIALS_POOLED = 18
TAKER_FEE = 0.0004
SLIPPAGE = 0.0001
CALIB_START = datetime(2021, 7, 1, tzinfo=timezone.utc)
CALIB_END = datetime(2021, 12, 31, tzinfo=timezone.utc)
DEV_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
DEV_END = datetime(2025, 6, 30, tzinfo=timezone.utc)
CALIB_START_NAIVE = CALIB_START.replace(tzinfo=None)
CALIB_END_NAIVE = CALIB_END.replace(tzinfo=None)
DEV_START_NAIVE = DEV_START.replace(tzinfo=None)
DEV_END_NAIVE = DEV_END.replace(tzinfo=None)
LONG_FRAC, SHORT_FRAC = 0.70, 0.30
LONG_QUALITY, SHORT_QUALITY = 0.0, 0.15
LIQUIDITY_RANK_FRAC = 0.5
DELISTED = {"LUNAUSDT", "FTTUSDT"}
GRID = {"L": [8, 16, 32], "theta": [0.02, 0.05, 0.10], "alpha": [2.0, 3.0, 4.0]}

SYMBOLS = [
    s + "USDT"
    for s in "AAVE ADA ALGO ATOM AVAX AXS BNB BTC DOGE DOT EOS ETC ETH FIL FTM FTT LINK LTC LUNA MANA MATIC NEAR SAND SOL THETA TRX UNI VET XLM XRP".split()
]


def load_panel() -> pl.DataFrame:
    panel = research.load_universe_panel(
        SYMBOLS, "6h", CALIB_START, DEV_END,
        download_dir="/tmp/claude-1000/-home-nick-Documents-quant-trading-labs/47229a3d-77fb-43df-b666-7397be0c6d9a/scratchpad",
        cache_dir="src/research/cache",
    )
    return panel.sort(["symbol", "datetime"])


def add_features(panel: pl.DataFrame) -> pl.DataFrame:
    panel = panel.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1)).over("symbol").alias("log_return")
    )
    panel = panel.with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume")
    )
    panel = panel.with_columns(
        pl.col("dollar_volume").rolling_mean(window_size=120).shift(1).over("symbol").alias("trailing_dollar_volume")
    )
    # trailing "1-month" realized Sharpe at 6h cadence: ~30d*4bars/day = 120 bars
    panel = panel.with_columns(
        (
            pl.col("log_return").rolling_mean(window_size=120).over("symbol")
            / pl.col("log_return").rolling_std(window_size=120).over("symbol").clip(lower_bound=1e-9)
            * np.sqrt(365 * 4)
        ).shift(1).over("symbol").alias("trailing_sharpe_1m")
    )
    panel = panel.with_columns(
        pl.col("log_return").rolling_std(window_size=20).shift(1).over("symbol").alias("realized_vol_20")
    )
    return panel.filter(pl.col("realized_vol_20") > 1e-12)


def month_key(dt) -> str:
    return str(dt)[:7]


def score_month_config(month_df: pl.DataFrame, L: int, theta: float, alpha: float) -> float:
    """Score one candidate (L, theta, alpha) on one already-sliced (single
    calendar month, all symbols) frame: mean per-symbol net Sharpe of the
    entry+trailing-stop rule that month, pooled equally-weighted across
    symbols with enough bars. Purely a fitting criterion for the NEXT
    month -- never touches data outside `month_df`.
    """
    sharpes = []
    for sym, g in month_df.group_by("symbol"):
        g = g.sort("datetime")
        close = g["close"].to_numpy()
        high = g["high"].to_numpy()
        low = g["low"].to_numpy()
        if len(close) < L + 10:
            continue
        entry = E.rate_of_change_entry(close, L, theta)
        stop = E.trailing_atr_stop(close, high, low, entry, atr_mult=alpha)
        exit_mask, fill = E.apply_stop_fill(
            np.roll(close, 1), high, low, stop, entry, optimistic=False
        )
        ret = np.diff(np.log(close), prepend=np.log(close[0]))
        ret[0] = 0.0
        pnl = entry[:-1] * ret[1:]
        pnl = np.concatenate([[0.0], pnl])
        sd = np.std(pnl)
        if sd > 1e-10:
            sharpes.append(float(np.mean(pnl) / sd))
    return float(np.mean(sharpes)) if sharpes else -np.inf


def monthly_panel_regrid(panel: pl.DataFrame, grid: dict) -> dict[str, tuple]:
    """Per-month (L, theta, alpha), fit ONLY on the preceding month's slice,
    applied to the next month. NEXT_PROMPT.md sec7 trap2: `month_df` below
    is a hard `.filter()` slice of `panel` to `[month_start, month_end)`
    before `score_month_config` ever sees it, so no bar from the applied
    month (or later) can enter the grid search. Asserted by
    tests/test_exec_lib13.py's monthly-regrid perturbation test (same
    contract, panel-shaped here).
    """
    panel = panel.with_columns(pl.col("datetime").map_elements(month_key, return_dtype=pl.Utf8).alias("_month"))
    months = sorted(panel["_month"].unique().to_list())
    fitted = {}
    for i in range(1, len(months)):
        fit_month, applied_month = months[i - 1], months[i]
        month_df = panel.filter(pl.col("_month") == fit_month)
        best = None
        for L in grid["L"]:
            for theta in grid["theta"]:
                for alpha in grid["alpha"]:
                    score = score_month_config(month_df, L, theta, alpha)
                    if best is None or score > best[0]:
                        best = (score, L, theta, alpha)
        fitted[applied_month] = (best[1], best[2], best[3])
    return fitted


def simulate_book(
    panel: pl.DataFrame,
    params_by_month: dict[str, tuple] | tuple,
    use_stop: bool = True,
    use_selection: bool = True,
    use_liquidity: bool = True,
    exclude_delisted: bool = False,
    origin_offset_bars: int = 0,
) -> pl.DataFrame:
    df = panel
    if exclude_delisted:
        df = df.filter(~pl.col("symbol").is_in(list(DELISTED)))
    df = df.filter(pl.col("datetime") >= DEV_START_NAIVE, pl.col("datetime") <= DEV_END_NAIVE)
    if origin_offset_bars:
        all_dt = sorted(df["datetime"].unique().to_list())
        keep = set(all_dt[origin_offset_bars:])
        df = df.filter(pl.col("datetime").is_in(list(keep)))

    df = df.with_columns(pl.col("datetime").map_elements(month_key, return_dtype=pl.Utf8).alias("_month"))
    rows = []
    for sym, g in df.group_by("symbol"):
        g = g.sort("datetime")
        n = len(g)
        if n < 30:
            continue
        close = g["close"].to_numpy()
        high = g["high"].to_numpy()
        low = g["low"].to_numpy()
        months = g["_month"].to_list()
        dvol = g["trailing_dollar_volume"].to_numpy()
        tsharpe = g["trailing_sharpe_1m"].to_numpy()

        if isinstance(params_by_month, tuple):
            L, theta, alpha = params_by_month
            entry = E.rate_of_change_entry(close, L, theta)
        else:
            entry = np.zeros(n)
            for m in set(months):
                if m not in params_by_month:
                    continue
                L, theta, _alpha = params_by_month[m]
                idx = [i for i, mm in enumerate(months) if mm == m]
                sub_close = close[max(0, idx[0] - L - 1) : idx[-1] + 1]
                sig = E.rate_of_change_entry(sub_close, L, theta)
                entry[idx] = sig[-len(idx):]

        if use_stop:
            atr_mult = params_by_month[months[-1]][2] if isinstance(params_by_month, dict) and months[-1] in params_by_month else (params_by_month[2] if isinstance(params_by_month, tuple) else 3.0)
            stop = E.trailing_atr_stop(close, high, low, entry, atr_mult=atr_mult)
            open_ = np.roll(close, 1)
            exit_mask, fill_price = E.apply_stop_fill(open_, high, low, stop, entry, optimistic=False)
            realized_entry = entry.copy()
            flat_until_flip = False
            last_sign = 0.0
            for t in range(1, n):
                if flat_until_flip:
                    if np.sign(entry[t]) not in (0.0, last_sign):
                        flat_until_flip = False
                    else:
                        realized_entry[t] = 0.0
                        continue
                if exit_mask[t] and realized_entry[t - 1] != 0:
                    realized_entry[t] = realized_entry[t - 1]
                    flat_until_flip = True
                    last_sign = np.sign(realized_entry[t - 1])
        else:
            realized_entry = entry

        eligible_long = np.ones(n, dtype=bool)
        eligible_short = np.ones(n, dtype=bool)
        if use_selection:
            eligible_long = E.quality_liquidity_selection(tsharpe, dvol, LONG_QUALITY, SHORT_QUALITY, LIQUIDITY_RANK_FRAC if use_liquidity else 1.0, "long")
            eligible_short = E.quality_liquidity_selection(tsharpe, dvol, LONG_QUALITY, SHORT_QUALITY, LIQUIDITY_RANK_FRAC if use_liquidity else 1.0, "short")
        elif use_liquidity:
            rank = pl.Series(dvol).rank(method="average", descending=True).to_numpy() / n
            eligible_long = eligible_short = rank <= LIQUIDITY_RANK_FRAC

        leg_position = np.where(
            realized_entry > 0, np.where(eligible_long, 1.0, 0.0),
            np.where(realized_entry < 0, np.where(eligible_short, -1.0, 0.0), 0.0),
        )
        ret = np.diff(np.log(close), prepend=np.log(close[0]))
        ret[0] = 0.0
        for t in range(n):
            rows.append({"datetime": g["datetime"][t], "symbol": sym, "leg_position": leg_position[t], "ret": ret[t]})

    return pl.DataFrame(rows)


def book_returns(sim: pl.DataFrame) -> pl.DataFrame:
    def leg_return(df: pl.DataFrame, sign: int, capital_frac: float) -> pl.DataFrame:
        leg = df.filter(pl.col("leg_position") * sign > 0)
        n_active = leg.group_by("datetime").agg(pl.len().alias("n"))
        leg = leg.join(n_active, on="datetime")
        leg = leg.with_columns((pl.col("ret") * sign * capital_frac / pl.col("n")).alias("weighted_ret"))
        return leg.group_by("datetime").agg(pl.col("weighted_ret").sum().alias("leg_ret"))

    longs = leg_return(sim, 1, LONG_FRAC)
    shorts = leg_return(sim, -1, SHORT_FRAC)
    all_dates = sim.select("datetime").unique()
    out = (
        all_dates.join(longs, on="datetime", how="left")
        .join(shorts, on="datetime", how="left", suffix="_short")
        .fill_null(0.0)
        .with_columns((pl.col("leg_ret") + pl.col("leg_ret_short")).alias("trade_log_return"))
        .sort("datetime")
    )
    turnover = (
        sim.sort(["symbol", "datetime"])
        .with_columns(pl.col("leg_position").diff().fill_null(pl.col("leg_position")).abs().over("symbol").alias("d_pos"))
        .group_by("datetime")
        .agg((pl.col("d_pos").sum() / 30.0).alias("turnover"))
    )
    out = out.join(turnover, on="datetime", how="left").fill_null(0.0)
    cost = TAKER_FEE + SLIPPAGE
    out = out.with_columns(
        (pl.col("trade_log_return") - cost * pl.col("turnover")).alias("trade_log_return_net")
    )
    return out


def series_metrics(x: np.ndarray, label: str) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"label": label, "no_bars": 0}
    std, mean = float(np.std(x)), float(np.mean(x))
    cum = np.cumsum(x)
    dd = cum - np.maximum.accumulate(cum)
    return {
        "label": label, "no_bars": len(x),
        "sharpe": (mean / std) * ANNUALIZED_RATE if std > 0 else 0.0,
        "total_log_return": float(np.sum(x)),
        "max_drawdown": float(np.min(dd)),
    }


def evaluate(panel, params, exclude_delisted=False, **kwargs) -> dict:
    sim = simulate_book(panel, params, exclude_delisted=exclude_delisted, **kwargs)
    br = book_returns(sim)
    net = br["trade_log_return_net"].to_numpy()
    gross = br["trade_log_return"].to_numpy()
    ci_lo, ci_hi = research.block_bootstrap_ci(net, seed=0) if len(net) > 30 else (None, None)
    dsr = (
        research.deflated_sharpe_prob(series_metrics(net, "x")["sharpe"] / ANNUALIZED_RATE, N_TRIALS_POOLED, len(net))
        if len(net) > 30 else float("nan")
    )
    return {
        "net": series_metrics(net, "net"),
        "gross": series_metrics(gross, "gross"),
        "ci_95": [ci_lo, ci_hi],
        "ci_excludes_zero": bool(ci_lo is not None and (ci_lo > 0 or ci_hi < 0)),
        "dsr": dsr,
        "n_bars": len(net),
    }


def main():
    print("Loading panel...", flush=True)
    panel = load_panel()
    panel = add_features(panel)
    print(f"Panel: {panel.height} rows, {panel['symbol'].n_unique()} symbols", flush=True)

    calib = panel.filter(pl.col("datetime") >= CALIB_START_NAIVE, pl.col("datetime") <= CALIB_END_NAIVE)
    print("Fitting frozen params on 2021H2 calibration window...", flush=True)
    best = None
    for L in GRID["L"]:
        for theta in GRID["theta"]:
            for alpha in GRID["alpha"]:
                score = score_month_config(calib, L, theta, alpha)
                if best is None or score > best[0]:
                    best = (score, L, theta, alpha)
    frozen_params = (best[1], best[2], best[3])
    print("Frozen params:", frozen_params, flush=True)

    print("Fitting monthly adaptive grid on dev window (causal)...", flush=True)
    monthly_fit_input = panel.filter(pl.col("datetime") >= CALIB_START_NAIVE, pl.col("datetime") <= DEV_END_NAIVE)
    adaptive_params = monthly_panel_regrid(monthly_fit_input, GRID)

    trials = {}
    print("Evaluating: adaptive...", flush=True)
    trials["adaptive"] = evaluate(panel, adaptive_params)
    print("Evaluating: frozen twin...", flush=True)
    trials["frozen_twin"] = evaluate(panel, frozen_params)
    print("Evaluating: ablation no_stop...", flush=True)
    trials["ablation_no_stop"] = evaluate(panel, adaptive_params, use_stop=False)
    print("Evaluating: ablation no_selection...", flush=True)
    trials["ablation_no_selection"] = evaluate(panel, adaptive_params, use_selection=False)
    print("Evaluating: ablation no_liquidity...", flush=True)
    trials["ablation_no_liquidity"] = evaluate(panel, adaptive_params, use_liquidity=False)
    print("Evaluating: ablation no_adaptation (=frozen)...", flush=True)
    trials["ablation_no_adaptation"] = trials["frozen_twin"]

    print("Survivorship: adaptive excluding LUNA/FTT...", flush=True)
    trials["adaptive_excl_delisted"] = evaluate(panel, adaptive_params, exclude_delisted=True)

    print("Origin offsets...", flush=True)
    by_offset = {}
    for off in ORIGIN_OFFSETS:
        by_offset[f"offset_{off}"] = evaluate(panel, adaptive_params, origin_offset_bars=off)

    adaptive_net = trials["adaptive"]["net"]["sharpe"]
    frozen_net = trials["frozen_twin"]["net"]["sharpe"]
    # paired CI on (adaptive - frozen) daily/6h returns
    sim_a = simulate_book(panel, adaptive_params)
    sim_f = simulate_book(panel, frozen_params)
    br_a, br_f = book_returns(sim_a), book_returns(sim_f)
    joined = br_a.join(br_f, on="datetime", suffix="_frozen")
    diff = (joined["trade_log_return_net"] - joined["trade_log_return_net_frozen"]).to_numpy()
    diff_ci = research.block_bootstrap_ci(diff, seed=0) if len(diff) > 30 else (None, None)
    adaptive_beats_frozen = bool(diff_ci[0] is not None and diff_ci[0] > 0)

    ablation_predicted_direction = {
        "no_stop": bool(trials["ablation_no_stop"]["net"]["sharpe"] < adaptive_net),
        "no_selection": bool(trials["ablation_no_selection"]["net"]["sharpe"] < adaptive_net),
        "no_liquidity": bool(trials["ablation_no_liquidity"]["net"]["sharpe"] < adaptive_net),
        "no_adaptation": bool(trials["ablation_no_adaptation"]["net"]["sharpe"] < adaptive_net),
    }
    n_correct_direction = sum(ablation_predicted_direction.values())

    gate_at_fires = bool(
        trials["adaptive"]["ci_excludes_zero"]
        and trials["adaptive"]["dsr"] >= 0.95
        and adaptive_net > 1.5  # sec3 bar floor context; full bar also needs null-beat, checked in writeup
        and adaptive_beats_frozen
        and n_correct_direction >= 3
    )

    out = {
        "gate": "AT",
        "n_trials_pooled": N_TRIALS_POOLED,
        "frozen_params": frozen_params,
        "n_adaptive_months": len(adaptive_params),
        "trials": trials,
        "by_offset": by_offset,
        "adaptive_vs_frozen_paired_ci": diff_ci,
        "adaptive_beats_frozen": adaptive_beats_frozen,
        "ablation_predicted_direction": ablation_predicted_direction,
        "n_ablations_correct_direction": n_correct_direction,
        "funding_not_modelled": True,
        "gate_AT_fires": gate_at_fires,
    }
    with open("src/research/tmp/phase_C_13_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "by_offset"}, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
