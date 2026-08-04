"""Notebook 13, Design C v2 -- rebuilt to match the actual AdaptiveTrend paper
(arXiv:2602.11708) parameters after the user supplied the source. Differences
from v1 (run_phase_C_13.py), each a direct fix to a v1 gap identified by
reading the paper:

  - Universe expanded from 30 to 128 Binance USDT perpetuals (paper: "150+"),
    original 30 unioned with every USDT perpetual onboarded on/before
    2022-07-01 per Binance's live exchangeInfo, index/dominance products
    (BTCDOM, DEFI) excluded as not single-asset coins.
  - Quality thresholds corrected to the paper's actual values: rolling
    Sharpe >= 1.3 (long-eligible) / <= -1.7 (short-eligible) -- v1 used
    0.0/0.15, dramatically looser.
  - Selection is now literally top-15 / bottom-15 by market cap (paper's
    KL=15, KS assumed symmetric=15), not a liquidity-rank proxy. Market cap
    from CoinGecko's free /coins/markets endpoint -- a CURRENT snapshot
    ranking applied statically across the whole backtest (the free tier's
    rate limit makes a full rolling historical reconstruction across 128
    symbols impractical), disclosed as a real remaining gap versus the
    paper's presumably-rolling ranking. ~54/128 symbols have fallen out of
    today's top-500 market cap and get a manual/placeholder rank or are
    excluded from the long pool entirely (never from the short pool, where
    a low current rank is directionally consistent with eligibility).
  - Funding rate cash flows now modelled, fetched live from Binance's
    fapi funding-rate endpoint (data.download_funding_rate_range) --
    charged/credited to the position every funding event per the sign of
    the position, closing v1's disclosed "not modelled" gap.
  - ATR multiplier grid narrowed to bracket the paper's reported optimal
    region (alpha in [2.0, 3.5], centered ~2.5) instead of v1's [2,3,4].
    (L, theta) grids are UNCHANGED from v1 -- the paper never discloses
    its own grid values for these, so there is nothing to match.

Writes phase_C_13_v2_results.json.
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
from data import download_funding_rate_range

ANNUALIZED_RATE = float(np.sqrt(365 * 4))
ORIGIN_OFFSETS = [0, 7 * 4, 14 * 4, 21 * 4]
N_TRIALS_POOLED = 18
TAKER_FEE = 0.0004
SLIPPAGE = 0.0001
CALIB_START = datetime(2021, 7, 1, tzinfo=timezone.utc)
CALIB_END = datetime(2021, 12, 31, tzinfo=timezone.utc)
DEV_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
DEV_END = datetime(2025, 6, 30, tzinfo=timezone.utc)
DEV_START_NAIVE = DEV_START.replace(tzinfo=None)
DEV_END_NAIVE = DEV_END.replace(tzinfo=None)
CALIB_START_NAIVE = CALIB_START.replace(tzinfo=None)
CALIB_END_NAIVE = CALIB_END.replace(tzinfo=None)

LONG_FRAC, SHORT_FRAC = 0.70, 0.30
LONG_QUALITY, SHORT_QUALITY = 1.3, 1.7  # corrected from v1's 0.0/0.15
K_LONG, K_SHORT = 15, 15  # paper's KL=15, KS unconfirmed but assumed symmetric
DELISTED = {"LUNAUSDT", "FTTUSDT"}
GRID = {"L": [8, 16, 32], "theta": [0.02, 0.05, 0.10], "alpha": [2.0, 2.5, 3.0, 3.5]}

DOWNLOAD_DIR = "/tmp/claude-1000/-home-nick-Documents-quant-trading-labs/47229a3d-77fb-43df-b666-7397be0c6d9a/scratchpad"


def load_universe() -> list[str]:
    return json.load(open("src/research/tmp/design_c_v2_universe.json"))


def load_market_cap() -> dict[str, float]:
    mc = json.load(open("src/research/tmp/design_c_v2_marketcap.json"))
    return mc


def load_panel(symbols: list[str]) -> pl.DataFrame:
    panel = research.load_universe_panel(
        symbols, "6h", CALIB_START, DEV_END,
        download_dir=DOWNLOAD_DIR, cache_dir="src/research/cache", min_cross_section=5,
    )
    return panel.sort(["symbol", "datetime"])


def load_funding(symbols: list[str]) -> pl.DataFrame:
    frames = []
    for sym in symbols:
        try:
            df = download_funding_rate_range(sym, "2021-07-01", "2025-06-30", cache_dir="src/research/cache")
        except Exception:
            continue
        if df.height == 0:
            continue
        cols = df.columns
        rate_col = "fundingRate" if "fundingRate" in cols else ("funding_rate" if "funding_rate" in cols else None)
        time_col = "fundingTime" if "fundingTime" in cols else ("datetime" if "datetime" in cols else None)
        if rate_col is None or time_col is None:
            continue
        sub = df.select(
            pl.col(time_col).alias("datetime"),
            pl.col(rate_col).cast(pl.Float64).alias("funding_rate"),
            pl.lit(sym).alias("symbol"),
        )
        if sub["datetime"].dtype != pl.Datetime:
            sub = sub.with_columns(pl.from_epoch("datetime", time_unit="ms").alias("datetime"))
        frames.append(sub)
    return pl.concat(frames).sort(["symbol", "datetime"]) if frames else pl.DataFrame(
        schema={"datetime": pl.Datetime, "funding_rate": pl.Float64, "symbol": pl.Utf8}
    )


def add_features(panel: pl.DataFrame, market_cap: dict[str, float]) -> pl.DataFrame:
    panel = panel.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1)).over("symbol").alias("log_return")
    )
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
    mc_expr = pl.col("symbol").replace_strict(market_cap, default=0.0, return_dtype=pl.Float64)
    panel = panel.with_columns(mc_expr.alias("market_cap"))
    return panel.filter(pl.col("realized_vol_20") > 1e-12)


def month_key(dt) -> str:
    return str(dt)[:7]


def score_month_config(month_df: pl.DataFrame, L: int, theta: float, alpha: float) -> float:
    sharpes = []
    for sym, g in month_df.group_by("symbol"):
        g = g.sort("datetime")
        close, high, low = g["close"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy()
        if len(close) < L + 10:
            continue
        entry = E.rate_of_change_entry(close, L, theta)
        ret = np.diff(np.log(close), prepend=np.log(close[0]))
        ret[0] = 0.0
        pnl = np.concatenate([[0.0], entry[:-1] * ret[1:]])
        sd = np.std(pnl)
        if sd > 1e-10:
            sharpes.append(float(np.mean(pnl) / sd))
    return float(np.mean(sharpes)) if sharpes else -np.inf


def monthly_panel_regrid(panel: pl.DataFrame, grid: dict) -> dict[str, tuple]:
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


def market_cap_eligibility(mkt_cap: np.ndarray, k_long: int, k_short: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (long_eligible, short_eligible) for one bar's cross-section:
    top-k_long by market cap for longs, bottom-k_short (among symbols with a
    mapped market cap > 0) for shorts -- the paper's literal KL/KS rule,
    not a liquidity-rank proxy.
    """
    n = len(mkt_cap)
    has_cap = mkt_cap > 0
    order = np.argsort(-np.where(has_cap, mkt_cap, -np.inf))  # descending
    long_idx = order[: min(k_long, has_cap.sum())]
    order_asc = np.argsort(np.where(has_cap, mkt_cap, np.inf))  # ascending
    short_idx = order_asc[: min(k_short, has_cap.sum())]
    long_elig = np.zeros(n, dtype=bool)
    short_elig = np.zeros(n, dtype=bool)
    long_elig[long_idx] = True
    short_elig[short_idx] = True
    return long_elig, short_elig


def simulate_book(
    panel: pl.DataFrame,
    funding: pl.DataFrame,
    params_by_month: dict[str, tuple] | tuple,
    use_stop: bool = True,
    use_selection: bool = True,
    use_funding: bool = True,
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

    funding_map = {}
    if use_funding and funding.height:
        f = funding.filter(pl.col("datetime") >= DEV_START_NAIVE, pl.col("datetime") <= DEV_END_NAIVE)
        for sym, g in f.group_by("symbol"):
            funding_map[sym] = (g["datetime"].to_numpy(), g["funding_rate"].to_numpy())

    rows = []
    market_cap_by_date = df.select("datetime", "symbol", "market_cap")

    for sym, g in df.group_by("symbol"):
        g = g.sort("datetime")
        n = len(g)
        if n < 30:
            continue
        close, high, low = g["close"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy()
        months = g["_month"].to_list()
        tsharpe = g["trailing_sharpe_1m"].to_numpy()
        mkt_cap = g["market_cap"].to_numpy()

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
                sub_close = close[max(0, idx[0] - L - 1): idx[-1] + 1]
                sig = E.rate_of_change_entry(sub_close, L, theta)
                entry[idx] = sig[-len(idx):]

        if use_stop:
            atr_mult = (
                params_by_month[months[-1]][2]
                if isinstance(params_by_month, dict) and months[-1] in params_by_month
                else (params_by_month[2] if isinstance(params_by_month, tuple) else 2.5)
            )
            stop = E.trailing_atr_stop(close, high, low, entry, atr_mult=atr_mult)
            open_ = np.roll(close, 1)
            exit_mask, _ = E.apply_stop_fill(open_, high, low, stop, entry, optimistic=False)
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

        if use_selection:
            long_mc_elig, short_mc_elig = market_cap_eligibility(mkt_cap, K_LONG, K_SHORT)
            quality_long = tsharpe >= LONG_QUALITY
            quality_short = tsharpe <= -SHORT_QUALITY
            eligible_long = long_mc_elig & quality_long & np.isfinite(tsharpe)
            eligible_short = short_mc_elig & quality_short & np.isfinite(tsharpe)
        else:
            eligible_long = np.ones(n, dtype=bool)
            eligible_short = np.ones(n, dtype=bool)

        leg_position = np.where(
            realized_entry > 0, np.where(eligible_long, 1.0, 0.0),
            np.where(realized_entry < 0, np.where(eligible_short, -1.0, 0.0), 0.0),
        )
        ret = np.diff(np.log(close), prepend=np.log(close[0]))
        ret[0] = 0.0

        funding_cost = np.zeros(n)
        if use_funding and sym in funding_map:
            f_dt, f_rate = funding_map[sym]
            dt_arr = g["datetime"].to_numpy()
            fr_lookup = dict(zip(f_dt, f_rate, strict=False))
            for t in range(n):
                r = fr_lookup.get(dt_arr[t])
                if r is not None and leg_position[t] != 0:
                    funding_cost[t] = -leg_position[t] * r  # long pays positive funding, receives negative

        for t in range(n):
            rows.append({
                "datetime": g["datetime"][t], "symbol": sym,
                "leg_position": leg_position[t], "ret": ret[t], "funding_cost": funding_cost[t],
            })

    return pl.DataFrame(rows)


def book_returns(sim: pl.DataFrame) -> pl.DataFrame:
    def leg_return(df: pl.DataFrame, sign: int, capital_frac: float) -> pl.DataFrame:
        leg = df.filter(pl.col("leg_position") * sign > 0)
        n_active = leg.group_by("datetime").agg(pl.len().alias("n"))
        leg = leg.join(n_active, on="datetime")
        leg = leg.with_columns(
            ((pl.col("ret") * sign + pl.col("funding_cost")) * capital_frac / pl.col("n")).alias("weighted_ret")
        )
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
    n_symbols = sim["symbol"].n_unique()
    turnover = (
        sim.sort(["symbol", "datetime"])
        .with_columns(pl.col("leg_position").diff().fill_null(pl.col("leg_position")).abs().over("symbol").alias("d_pos"))
        .group_by("datetime")
        .agg((pl.col("d_pos").sum() / n_symbols).alias("turnover"))
    )
    out = out.join(turnover, on="datetime", how="left").fill_null(0.0)
    cost = TAKER_FEE + SLIPPAGE
    out = out.with_columns((pl.col("trade_log_return") - cost * pl.col("turnover")).alias("trade_log_return_net"))
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


def evaluate(panel, funding, params, exclude_delisted=False, **kwargs) -> dict:
    sim = simulate_book(panel, funding, params, exclude_delisted=exclude_delisted, **kwargs)
    br = book_returns(sim)
    net = br["trade_log_return_net"].to_numpy()
    gross = br["trade_log_return"].to_numpy()
    ci_lo, ci_hi = research.block_bootstrap_ci(net, seed=0) if len(net) > 30 else (None, None)
    dsr = (
        research.deflated_sharpe_prob(series_metrics(net, "x")["sharpe"] / ANNUALIZED_RATE, N_TRIALS_POOLED, len(net))
        if len(net) > 30 else float("nan")
    )
    return {
        "net": series_metrics(net, "net"), "gross": series_metrics(gross, "gross"),
        "ci_95": [ci_lo, ci_hi],
        "ci_excludes_zero": bool(ci_lo is not None and (ci_lo > 0 or ci_hi < 0)),
        "dsr": dsr, "n_bars": len(net),
    }


def main():
    print("Loading expanded universe panel + funding rates...", flush=True)
    symbols = load_universe()
    market_cap = load_market_cap()
    panel = load_panel(symbols)
    funding = load_funding(symbols)
    print(f"Panel: {panel.height} rows, {panel['symbol'].n_unique()} symbols; funding: {funding.height} rows", flush=True)

    panel = add_features(panel, market_cap)

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

    print("Fitting monthly adaptive grid (causal)...", flush=True)
    monthly_fit_input = panel.filter(pl.col("datetime") >= CALIB_START_NAIVE, pl.col("datetime") <= DEV_END_NAIVE)
    adaptive_params = monthly_panel_regrid(monthly_fit_input, GRID)

    trials = {}
    print("Evaluating: adaptive (v2, corrected)...", flush=True)
    trials["adaptive"] = evaluate(panel, funding, adaptive_params)
    print("Evaluating: frozen twin...", flush=True)
    trials["frozen_twin"] = evaluate(panel, funding, frozen_params)
    print("Evaluating: adaptive, no funding modelled...", flush=True)
    trials["adaptive_no_funding"] = evaluate(panel, funding, adaptive_params, use_funding=False)
    print("Evaluating: ablation no_stop...", flush=True)
    trials["ablation_no_stop"] = evaluate(panel, funding, adaptive_params, use_stop=False)
    print("Evaluating: ablation no_selection...", flush=True)
    trials["ablation_no_selection"] = evaluate(panel, funding, adaptive_params, use_selection=False)
    print("Evaluating: ablation no_adaptation (=frozen)...", flush=True)
    trials["ablation_no_adaptation"] = trials["frozen_twin"]
    print("Survivorship: adaptive excluding LUNA/FTT...", flush=True)
    trials["adaptive_excl_delisted"] = evaluate(panel, funding, adaptive_params, exclude_delisted=True)

    print("Origin offsets...", flush=True)
    by_offset = {}
    for off in ORIGIN_OFFSETS:
        by_offset[f"offset_{off}"] = evaluate(panel, funding, adaptive_params, origin_offset_bars=off)

    sim_a = simulate_book(panel, funding, adaptive_params)
    sim_f = simulate_book(panel, funding, frozen_params)
    br_a, br_f = book_returns(sim_a), book_returns(sim_f)
    joined = br_a.join(br_f, on="datetime", suffix="_frozen")
    diff = (joined["trade_log_return_net"] - joined["trade_log_return_net_frozen"]).to_numpy()
    diff_ci = research.block_bootstrap_ci(diff, seed=0) if len(diff) > 30 else (None, None)
    adaptive_beats_frozen = bool(diff_ci[0] is not None and diff_ci[0] > 0)

    adaptive_net = trials["adaptive"]["net"]["sharpe"]
    ablation_predicted_direction = {
        "no_stop": bool(trials["ablation_no_stop"]["net"]["sharpe"] < adaptive_net),
        "no_selection": bool(trials["ablation_no_selection"]["net"]["sharpe"] < adaptive_net),
        "no_adaptation": bool(trials["ablation_no_adaptation"]["net"]["sharpe"] < adaptive_net),
    }
    n_correct = sum(ablation_predicted_direction.values())

    gate_at_fires = bool(
        trials["adaptive"]["ci_excludes_zero"]
        and trials["adaptive"]["dsr"] >= 0.95
        and adaptive_net > 1.5
        and adaptive_beats_frozen
        and n_correct >= 2  # 2 of 3 ablations tracked here (liquidity-filter ablation dropped, selection is now market-cap based)
    )

    n_mapped_market_cap = sum(1 for v in market_cap.values() if v > 0)

    out = {
        "gate": "AT_v2",
        "n_trials_pooled": N_TRIALS_POOLED,
        "universe_size": len(symbols),
        "n_symbols_with_market_cap": n_mapped_market_cap,
        "frozen_params": frozen_params,
        "n_adaptive_months": len(adaptive_params),
        "corrections_from_v1": {
            "universe": f"{len(symbols)} symbols (v1: 30)",
            "quality_thresholds": f"long>={LONG_QUALITY}, short<=-{SHORT_QUALITY} (v1: 0.0/0.15)",
            "selection_mechanism": f"top-{K_LONG}/bottom-{K_SHORT} by market cap (v1: dollar-volume rank proxy)",
            "funding": "modelled from live Binance funding-rate history (v1: not modelled)",
            "alpha_grid": f"{GRID['alpha']} (v1: [2,3,4])",
        },
        "trials": trials,
        "by_offset": by_offset,
        "adaptive_vs_frozen_paired_ci": diff_ci,
        "adaptive_beats_frozen": adaptive_beats_frozen,
        "ablation_predicted_direction": ablation_predicted_direction,
        "n_ablations_correct_direction": n_correct,
        "gate_AT_v2_fires": gate_at_fires,
    }
    with open("src/research/tmp/phase_C_13_v2_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "by_offset"}, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
