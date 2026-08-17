"""Notebook 018 -- the crypto perpetual funding basis trade (Gate FA).

Long spot, short perp, same symbol, delta-neutral. The return is the 8h
funding payment less the basis mark-to-market and costs (NEXT_PROMPT sec
3.2).

This is NOT 007 Phase C (cross-sectional dollar-neutral in perps, betting
funding predicts price) and NOT 013 Design C (AdaptiveTrend, already null).
See NEXT_PROMPT sec 2.1.

Data-loading fence (NEXT_PROMPT sec 9.3): `load_basis_panel` is the only
panel loader this module exposes for Phase 3/4/5 use, and it always reads
from `DEV_CACHE_DIR` -- there is no parameter on it that can reach
`basis18/holdout`. `_load_basis_panel` (the shared assembly logic, taking
an explicit `cache_dir`) is a private implementation detail; only
`run_phase_6_18_holdout.py` calls it directly, with the holdout directory
literal.
"""

from __future__ import annotations

import functools
import sys
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import data
import research

# --------------------------------------------------------------------------
# Frozen constants, from NEXT_PROMPT sec 3.4 / 5.2 and
# phase_0_18_preregistration.json. Declared, not swept.
# --------------------------------------------------------------------------
SPOT_TAKER_BP = 10.0
PERP_TAKER_BP = 5.0
SLIPPAGE_BP = 1.0
ROUND_TURN_BP = 2 * (SPOT_TAKER_BP + SLIPPAGE_BP + PERP_TAKER_BP + SLIPPAGE_BP)  # 34.0
TARGET_HOLD_PERIODS = 45
THETA_IN = (ROUND_TURN_BP / TARGET_HOLD_PERIODS) * 1e-4  # 7.5556e-05
THETA_OUT = THETA_IN / 2.0  # 3.7778e-05
CARRY_EWMA_PERIODS = 21  # a HALF-LIFE, not a span/window (sec 5.1)
MAX_POSITIONS = 10
LIQUIDITY_FLOOR_USD = 5_000_000.0

DEV_START = datetime(2021, 7, 1, tzinfo=UTC)
DEV_END = datetime(2025, 6, 30, tzinfo=UTC)
DEV_CACHE_DIR = "src/research/cache/basis18/dev"
DEV_DOWNLOAD_DIR = "src/research/tmp_dl/basis18/dev"
FUNDING_CACHE_DIR = "src/research/cache"  # existing repo-wide funding cache, sec 4.2
UNIVERSE_SEED_PATH = str(Path(__file__).resolve().parent / "design_c_v2_universe.json")

INTERVAL = "8h"


# --------------------------------------------------------------------------
# Core trade mechanics (NEXT_PROMPT sec 7.3)
# --------------------------------------------------------------------------


def carry_estimate(funding: pl.Expr, periods: int = CARRY_EWMA_PERIODS) -> pl.Expr:
    """Causal EWMA of settled funding. Strictly backward-looking: the value
    at t uses only funding settled at or before t. Proven by
    test_carry_estimate_is_causal.
    """
    return funding.ewm_mean(half_life=periods, adjust=False, ignore_nulls=False)


def paired_log_return(
    spot_close: pl.Expr, perp_close: pl.Expr, funding: pl.Expr
) -> pl.Expr:
    """One 8h period of a held paired position, per unit of paired notional.

    +spot leg, -perp leg, + funding received (Binance: positive funding pays
    shorts). Written as the explicit difference of two legs rather than the
    sec 3.2 basis approximation, so that identity can be *checked* against
    this in Phase 3 rather than assumed by it.
    """
    spot_leg = (spot_close / spot_close.shift(1)).log()
    perp_leg = (perp_close / perp_close.shift(1)).log()
    return spot_leg - perp_leg + funding


def qualifies(carry: pl.Expr, liquid: pl.Expr, held: pl.Expr) -> pl.Expr:
    """Hysteresis entry/exit (sec 5.3). Enter above THETA_IN, hold until
    below THETA_OUT -- never the same threshold in both directions, which
    is the whole point (007 sec 357-359).
    """
    return liquid & pl.when(held).then(carry > THETA_OUT).otherwise(carry > THETA_IN)


def apply_two_leg_costs(trade_frame: pl.DataFrame) -> pl.DataFrame:
    """Explicit two-leg cost accounting (NEXT_PROMPT sec 7.3's "one genuine
    gap"): `research.add_portfolio_costs` charges a single blended
    `cost_frac` per unit of turnover, which is fine numerically here since
    one unit of *paired-position* turnover trades equal notional on both
    legs -- but the blend must be built from both legs' real fees, not one
    leg's fee reused for both (012's mistake, in a new shape).

    cost_frac = (spot_taker + slippage) + (perp_taker + slippage), i.e. the
    sec 3.4 "entry (both legs)" cost of 17bp for one unit of turnover in one
    direction; a full open+close round turn costs 2x that = 34bp, matching
    ROUND_TURN_BP exactly (pinned by test_round_turn_bp_is_34).
    """
    spot_leg_cost = (SPOT_TAKER_BP + SLIPPAGE_BP) * 1e-4
    perp_leg_cost = (PERP_TAKER_BP + SLIPPAGE_BP) * 1e-4
    cost_frac = spot_leg_cost + perp_leg_cost
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


# --------------------------------------------------------------------------
# Data acquisition -- spot/perp klines (reusing data.download_and_unzip_klines
# via its new `market` param) and premiumIndexKlines (not in data.py: a
# different endpoint family, not just a different market -- sec 4.1/sec 10).
# --------------------------------------------------------------------------


def _month_range(start_date: datetime, end_date: datetime) -> list[str]:
    months = []
    cur = start_date.replace(day=1)
    while cur <= end_date:
        months.append(cur.strftime("%Y-%m"))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def fetch_premium_index_month(
    symbol: str,
    interval: str,
    month: str,
    download_dir: str = "tmp",
    cache_dir: str = "cache",
) -> pl.DataFrame | None:
    """Download+unzip one month of Binance futures/um premiumIndexKlines --
    the venue-authoritative basis series (sec 4.1). Verified live to be the
    identical 12-column monthly kline CSV shape as regular klines while
    building this module, so `data.KLINE_SCHEMA` is reused directly for
    parsing. Returns just (datetime, premium_index) -- open/high/low of the
    premium index are not used anywhere in this notebook.
    """
    cache_path_dir = Path(cache_dir)
    cache_path_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_dir / f"{symbol}-premium-{interval}-{month}.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    url = (
        "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    )
    download_path_dir = Path(download_dir) / "premium"
    download_path_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_path_dir / f"{symbol}-{interval}-{month}.zip"
    csv_path = download_path_dir / f"{symbol}-{interval}-{month}.csv"

    df = None
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(download_path_dir)
        with open(csv_path) as f:
            has_header = f.readline().startswith("open_time")
        df = pl.read_csv(
            csv_path,
            has_header=has_header,
            new_columns=None if has_header else list(data.KLINE_SCHEMA.keys()),
            schema_overrides=data.KLINE_SCHEMA,
        )
        df = (
            df.with_columns(
                pl.from_epoch("open_time", time_unit="ms").alias("datetime")
            )
            .select("datetime", "close")
            .rename({"close": "premium_index"})
        )
        df.write_parquet(cache_path)
    finally:
        zip_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
    return df


def _download_monthly_series_range(
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
    download_dir: str,
    cache_dir: str,
    fetch_month: Callable[[str, str, str, str, str], pl.DataFrame | None],
    series_tag: str,
) -> pl.DataFrame | None:
    """Shared month-range loop + combined-range parquet cache, generalized
    over which per-month fetch function is used. Mirrors
    `data.download_klines_range`'s own caching shape without modifying that
    function (NEXT_PROMPT sec 10 permits only the `market` param change to
    `download_and_unzip_klines` itself) -- this is the "add it to
    basis_lib18.py" fallback that section names.
    """
    range_cache_path = (
        Path(cache_dir)
        / f"{symbol}-{series_tag}-{interval}-{start_date:%Y-%m-%d}-{end_date:%Y-%m-%d}.parquet"
    )
    if range_cache_path.exists():
        return pl.read_parquet(range_cache_path)

    dfs = []
    for month in _month_range(start_date, end_date):
        df = fetch_month(symbol, interval, month, download_dir, cache_dir)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None

    result = (
        pl.concat(dfs, how="diagonal_relaxed")
        .sort("datetime")
        .unique(subset=["datetime"])
        .filter(
            (pl.col("datetime") >= start_date.replace(tzinfo=None))
            & (pl.col("datetime") <= end_date.replace(tzinfo=None))
        )
    )
    result.write_parquet(range_cache_path)
    return result


def fetch_symbol_series(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    download_dir: str,
    cache_dir: str,
) -> dict[str, pl.DataFrame | None]:
    """Fetch (or load from cache) spot klines, perp klines, and premium
    index klines for one symbol over [start_date, end_date]. Returns a dict
    with keys "spot"/"perp"/"premium", any of which may be None (a 404
    across every month -- data, not an error, sec 4.1/9.2).
    """
    spot = _download_monthly_series_range(
        symbol,
        INTERVAL,
        start_date,
        end_date,
        download_dir,
        cache_dir,
        functools.partial(data.download_and_unzip_klines, market="spot"),
        "spot",
    )
    perp = _download_monthly_series_range(
        symbol,
        INTERVAL,
        start_date,
        end_date,
        download_dir,
        cache_dir,
        functools.partial(data.download_and_unzip_klines, market="futures/um"),
        "perp",
    )
    premium = _download_monthly_series_range(
        symbol,
        INTERVAL,
        start_date,
        end_date,
        download_dir,
        cache_dir,
        fetch_premium_index_month,
        "premium",
    )
    return {"spot": spot, "perp": perp, "premium": premium}


def load_universe_seed() -> list[str]:
    import json

    with open(UNIVERSE_SEED_PATH) as f:
        return list(json.load(f))


def load_dev_funding(symbol: str) -> pl.DataFrame | None:
    """The already-cached, repo-wide funding history (sec 4.2) -- never
    re-downloaded for the dev window. All 128 universe-seed symbols already
    have this exact file.
    """
    path = Path(FUNDING_CACHE_DIR) / f"{symbol}-funding-2021-07-01-2025-06-30.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------


def _assemble_symbol_frame(
    symbol: str,
    series: dict[str, pl.DataFrame | None],
    funding: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Join spot/perp/premium/funding onto one 8h grid for a single symbol.

    spot/perp/premium are already on the native 8h Binance grid (sec 4.1),
    so an inner join on datetime is exact, not an asof/resample. Funding's
    own timestamps carry small millisecond jitter (see its own cache), so
    it is asof-joined backward exactly as `features.add_funding_rate_feature`
    already does for the rest of this repo -- never a rate published after
    the bar closes.

    dollar_volume approximation: `data.download_and_unzip_klines`'s select()
    does not retain Binance's own quote_volume column, so
    volume * close is used as the per-bar dollar-volume proxy for both
    legs' liquidity screen -- stated here, not silently assumed.
    """
    spot, perp, premium = series["spot"], series["perp"], series["premium"]
    if spot is None or perp is None or funding is None:
        return None

    spot_s = spot.select(
        "datetime",
        pl.col("close").alias("spot_close"),
        (pl.col("volume") * pl.col("close")).alias("spot_dollar_volume"),
    )
    perp_s = perp.select(
        "datetime",
        pl.col("close").alias("perp_close"),
        (pl.col("volume") * pl.col("close")).alias("perp_dollar_volume"),
    )
    joined = spot_s.join(perp_s, on="datetime", how="inner")

    if premium is not None:
        joined = joined.join(
            premium.rename({"premium_index": "basis_premium"}),
            on="datetime",
            how="left",
        )
    else:
        joined = joined.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("basis_premium")
        )

    funding_s = funding.sort("datetime")
    joined = joined.sort("datetime").join_asof(
        funding_s, on="datetime", strategy="backward"
    )

    return joined.with_columns(pl.lit(symbol).alias("symbol")).sort("datetime")


def _load_basis_panel(
    cache_dir: str,
    download_dir: str,
    symbols: Sequence[str],
    start_date: datetime,
    end_date: datetime,
    funding_loader: Callable[[str], pl.DataFrame | None] = load_dev_funding,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Shared panel-assembly logic (private -- see module docstring for the
    holdout fence this enforces). Returns (panel, manifest) where manifest
    maps symbol -> "ok" | "no_spot" | "no_perp" | "no_funding", so a caller
    can report the sec 4.3 "N of 128 symbols have no spot leg" finding.
    """
    frames = []
    manifest: dict[str, str] = {}
    for symbol in symbols:
        series = fetch_symbol_series(
            symbol, start_date, end_date, download_dir, cache_dir
        )
        funding = funding_loader(symbol)
        if series["spot"] is None:
            manifest[symbol] = "no_spot"
            continue
        if series["perp"] is None:
            manifest[symbol] = "no_perp"
            continue
        if funding is None:
            manifest[symbol] = "no_funding"
            continue
        frame = _assemble_symbol_frame(symbol, series, funding)
        if frame is None or len(frame) == 0:
            manifest[symbol] = "empty_after_join"
            continue
        manifest[symbol] = "ok"
        frames.append(frame)

    if not frames:
        raise ValueError("No symbols survived spot/perp/funding assembly")

    panel = pl.concat(frames, how="diagonal_relaxed").sort(["datetime", "symbol"])
    return panel, manifest


def load_basis_panel(
    symbols: Sequence[str] | None = None,
    start_date: datetime = DEV_START,
    end_date: datetime = DEV_END,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """The only basis panel loader Phase 3/4/5 may import. Always reads
    DEV_CACHE_DIR -- no parameter here can reach basis18/holdout
    (NEXT_PROMPT sec 9.3); only run_phase_6_18_holdout.py calls
    `_load_basis_panel` directly, with the holdout directory literal.
    """
    if end_date > research.HOLDOUT_START:
        raise ValueError(
            f"end_date {end_date} reaches into the frozen holdout period "
            f"(>= {research.HOLDOUT_START:%Y-%m-%d}). load_basis_panel never "
            "accepts allow_holdout -- only run_phase_6_18_holdout.py may read "
            "past this boundary."
        )
    if symbols is None:
        symbols = load_universe_seed()
    return _load_basis_panel(
        DEV_CACHE_DIR, DEV_DOWNLOAD_DIR, symbols, start_date, end_date
    )


# --------------------------------------------------------------------------
# Liquidity / universe screening (sec 4.3)
# --------------------------------------------------------------------------


def liquidity_screen(
    panel: pl.DataFrame, floor_usd: float = LIQUIDITY_FLOOR_USD, window_bars: int = 90
) -> pl.DataFrame:
    """Adds a `liquid` boolean column: trailing `window_bars`-bar (90 bars
    of 8h data = 30 days) median dollar volume on BOTH legs independently
    clears `floor_usd` (sec 4.3 screen 2). Causal (rolling, backward-looking
    only) so `liquid` at bar t never uses bar t's own volume alone -- it
    reflects the trailing 30 days as of, and including, bar t.
    """
    return (
        panel.sort(["symbol", "datetime"])
        .with_columns(
            [
                pl.col("spot_dollar_volume")
                .rolling_median(window_size=window_bars, min_samples=window_bars)
                .over("symbol")
                .alias("spot_dollar_volume_30d_median"),
                pl.col("perp_dollar_volume")
                .rolling_median(window_size=window_bars, min_samples=window_bars)
                .over("symbol")
                .alias("perp_dollar_volume_30d_median"),
            ]
        )
        .with_columns(
            (
                (pl.col("spot_dollar_volume_30d_median") >= floor_usd)
                & (pl.col("perp_dollar_volume_30d_median") >= floor_usd)
            ).alias("liquid")
        )
    )


# --------------------------------------------------------------------------
# Trade features: causal carry, the liquidity screen, and the forward-return
# label used as portfolio_trade_frame's target_col.
# --------------------------------------------------------------------------


def add_trade_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Adds "carry" (causal EWMA of funding, per symbol), "liquid" (sec 4.3
    screen 2), "paired_log_return" (this bar's realized paired return,
    ending at this bar), and "fwd_paired_return_1" (the label: the paired
    return that will accrue from this bar to the next one, known only once
    the next bar closes -- built with a negative/forward shift exactly like
    `features.forward_return`, never used as a feature itself, only as
    `research.portfolio_trade_frame`'s target_col so a decision weight at
    bar t is always paired with the return realized strictly after t).
    """
    panel = liquidity_screen(panel)
    panel = panel.sort(["symbol", "datetime"]).with_columns(
        carry_estimate(pl.col("funding_rate")).over("symbol").alias("carry")
    )
    paired_ret = paired_log_return(
        pl.col("spot_close"), pl.col("perp_close"), pl.col("funding_rate")
    )
    panel = panel.with_columns(paired_ret.over("symbol").alias("paired_log_return"))
    panel = panel.with_columns(
        pl.col("paired_log_return")
        .shift(-1)
        .over("symbol")
        .alias("fwd_paired_return_1")
    )
    perp_ret = (pl.col("perp_close") / pl.col("perp_close").shift(1)).log()
    panel = panel.with_columns(perp_ret.over("symbol").alias("perp_log_return"))
    panel = panel.with_columns(
        pl.col("perp_log_return").shift(-1).over("symbol").alias("fwd_perp_return_1")
    )
    return panel


# --------------------------------------------------------------------------
# Book construction (sec 5.4/5.5) -- sequential, causal, per-symbol.
# --------------------------------------------------------------------------


def build_book_weights(
    panel: pl.DataFrame,
    timed: bool,
    theta_in: float = THETA_IN,
    theta_out: float = THETA_OUT,
) -> pl.DataFrame:
    """Sequential, causal book construction.

    timed=True: hysteresis-gated entry/exit (`qualifies`), capped at
    MAX_POSITIONS by carry rank when more symbols qualify than the cap
    (sec 5.4) -- the pre-registered "timed" book, the headline.
    theta_in/theta_out default to the frozen module constants; Phase 5's
    no-hysteresis ablation passes theta_out=theta_in explicitly (a
    pre-registered robustness exhibit, not a sweep of the frozen design --
    sec 12 forbids sweeping to improve the headline, not running the one
    ablation sec 7.2 already names).
    timed=False ("always-on", sec 5.5): every currently-liquid symbol is
    held, uncapped, equal-weighted -- isolates whether timing (entry/exit
    gating + the N_max rank cap) adds anything over simply being in the
    trade the whole time, which is Gate FA-3's actual research question.
    This uncapped reading of "every screened symbol" is an implementation
    detail of an already-frozen book definition (sec 5.5), not a swept
    constant -- disclosed here and in the results write-up, not silently
    assumed.

    Sequential because hysteresis needs each symbol's own prior-bar held
    state; a pure-polars column expression can't express that persistence,
    so this loops bar-by-bar in chronological order (~4,400 dev bars x <=128
    symbols -- a few hundred thousand row-operations, not a performance
    concern at this scale).
    """
    df = panel.sort(["datetime", "symbol"]).select(
        "datetime", "symbol", "carry", "liquid"
    )
    held: dict[str, bool] = {}
    rows: list[dict[str, object]] = []

    for (dt,), group in df.group_by("datetime", maintain_order=True):
        symbols = group["symbol"].to_list()
        carries = group["carry"].to_list()
        liquids = group["liquid"].to_list()

        in_book: list[tuple[str, float]] = []
        for sym, carry, liquid in zip(symbols, carries, liquids, strict=True):
            liquid_bool = bool(liquid) if liquid is not None else False
            carry_val = carry if carry is not None else float("-inf")
            if timed:
                was_held = held.get(sym, False)
                threshold = theta_out if was_held else theta_in
                qualify = liquid_bool and carry_val > threshold
            else:
                qualify = liquid_bool
            if qualify:
                in_book.append((sym, carry_val))

        if timed and len(in_book) > MAX_POSITIONS:
            in_book.sort(key=lambda x: x[1], reverse=True)
            in_book = in_book[:MAX_POSITIONS]

        in_book_symbols = {sym for sym, _ in in_book}
        held = {sym: (sym in in_book_symbols) for sym in symbols}

        n = len(in_book_symbols)
        w = 1.0 / n if n > 0 else 0.0
        for sym in symbols:
            rows.append(
                {
                    "datetime": dt,
                    "symbol": sym,
                    "weight": w if sym in in_book_symbols else 0.0,
                }
            )

    schema: dict[str, pl.DataType | type] = {
        "datetime": panel["datetime"].dtype,
        "symbol": pl.Utf8,
        "weight": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def book_trade_frame(
    panel: pl.DataFrame, weights: pl.DataFrame, origin_offset: int = 0
) -> pl.DataFrame:
    """weights (build_book_weights output) + panel's fwd_paired_return_1 ->
    a portfolio_trade_frame-shaped frame (trade_log_return, turnover), via
    `research.portfolio_trade_frame` unchanged.

    origin_offset (in 8h periods) trims the FIRST origin_offset unique
    timestamps from the result before returning -- carry's EWMA and every
    weight decision are still computed over the FULL panel first (no
    re-fit, since this is a fixed-parameter strategy, not a model), only
    the metrics window shifts. This is the sec 7.2 Phase 4 "origin offsets
    [0,1,2,3]" robustness check, applied the cheap way a non-refit strategy
    allows.
    """
    trade_frame = research.portfolio_trade_frame(
        weights, panel, target_col="fwd_paired_return_1"
    )
    if origin_offset > 0:
        unique_times = trade_frame["datetime"].unique(maintain_order=True).sort()
        if origin_offset < len(unique_times):
            cutoff = unique_times[origin_offset]
            trade_frame = trade_frame.filter(pl.col("datetime") >= cutoff)
    return trade_frame


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def book_metrics(
    costed_trade_frame: pl.DataFrame, annualized_rate: float, label: str = "book"
) -> dict[str, Any]:
    """Gross + net summary metrics for an `apply_two_leg_costs` output
    frame, reusing `research._series_metrics` (the exact Sharpe/max-drawdown
    definitions every other backtest in this repo uses) instead of
    reimplementing them. Also reports annualized turnover in 007's own
    units (round-trips/yr, via `research.portfolio_turnover`'s sum-of-
    absolute-weight-changes convention) so this notebook's number is
    directly comparable to 007's ~674-681/yr (sec 5.3).
    """
    metrics = research._series_metrics(
        costed_trade_frame["trade_log_return"], annualized_rate, label
    )
    net_metrics = research._series_metrics(
        costed_trade_frame["trade_log_return_net"], annualized_rate, f"{label}_net"
    )
    metrics["sharpe_net"] = net_metrics["sharpe"]
    metrics["total_log_return_net"] = net_metrics["total_log_return"]
    metrics["compound_return_net"] = net_metrics["compound_return"]
    metrics["max_drawdown_net"] = net_metrics["max_drawdown"]
    periods_per_year = annualized_rate**2
    mean_turnover = research._as_float(costed_trade_frame["turnover"].mean())
    metrics["mean_turnover_per_bar"] = mean_turnover
    metrics["annualized_turnover"] = mean_turnover * periods_per_year
    return metrics


# --------------------------------------------------------------------------
# FA-4: neutrality check
# --------------------------------------------------------------------------


def ols_beta(returns: np.ndarray, benchmark: np.ndarray) -> float:
    """Simple single-factor OLS beta: Cov(returns, benchmark) / Var(benchmark).
    Both arrays must already be aligned (same bars, same order); NaNs are
    dropped pairwise first.
    """
    returns = np.asarray(returns, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    mask = np.isfinite(returns) & np.isfinite(benchmark)
    returns, benchmark = returns[mask], benchmark[mask]
    if len(returns) < 2 or np.var(benchmark) == 0:
        return float("nan")
    return float(np.cov(returns, benchmark)[0, 1] / np.var(benchmark))
