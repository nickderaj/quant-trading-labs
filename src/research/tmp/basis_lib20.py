"""Notebook 020 -- refined single-venue basis construction (Mechanism A) and
cross-venue funding-rate dispersion (Mechanism B). NEXT_PROMPT sec 5.

Imports basis_lib18 (as bl18) and research, and reuses their primitives; it
duplicates nothing it can call. Never edits either module.

Data-loading fence, mirroring bl18's own (NEXT_PROMPT sec 5.5): both
`load_xvenue_panel` here and `bl18.load_basis_panel` (used unchanged for
Mechanism A) always read from the *_CACHE_DIR/dev directories and raise if
asked to reach past `research.HOLDOUT_START`. Only run_phase_6_20_holdout.py
may name a holdout directory literal.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import basis_lib18 as bl18

import research

# --------------------------------------------------------------------------
# sec 5.1: inherited unchanged from 018 -- re-exported, not re-derived.
# --------------------------------------------------------------------------
SPOT_TAKER_BP = bl18.SPOT_TAKER_BP
PERP_TAKER_BP = bl18.PERP_TAKER_BP
SLIPPAGE_BP = bl18.SLIPPAGE_BP
ROUND_TURN_BP = bl18.ROUND_TURN_BP
LIQUIDITY_FLOOR_USD = bl18.LIQUIDITY_FLOOR_USD
LIQUIDITY_WINDOW_BARS = 90
MAX_POSITIONS = bl18.MAX_POSITIONS
CARRY_EWMA_HALF_LIFE = bl18.CARRY_EWMA_PERIODS
TARGET_HOLD_PERIODS = bl18.TARGET_HOLD_PERIODS
THETA_IN = bl18.THETA_IN
THETA_OUT = bl18.THETA_OUT
DEV_START = bl18.DEV_START
DEV_END = bl18.DEV_END
INTERVAL = bl18.INTERVAL

# --------------------------------------------------------------------------
# sec 5.2: Mechanism A -- the diversification floor.
# --------------------------------------------------------------------------
N_MIN = 3

# --------------------------------------------------------------------------
# sec 5.3: Mechanism A -- the lower-turnover carry (both scaled 2x from 018,
# preserving 018's own ratio, not a two-dimensional sweep).
# --------------------------------------------------------------------------
SLOW_CARRY_HALF_LIFE = 42
TARGET_HOLD_SLOW = 90
THETA_IN_SLOW = ROUND_TURN_BP / TARGET_HOLD_SLOW * 1e-4  # == 018's THETA_OUT, an
THETA_OUT_SLOW = (
    THETA_IN_SLOW / 2.0
)  # expected coincidence (34/90 == half of 34/45), not a bug.

# --------------------------------------------------------------------------
# sec 5.4: Mechanism B constants.
# --------------------------------------------------------------------------
BYBIT_TAKER_BP = 5.5
ROUND_TURN_BP_XV = 2 * ((PERP_TAKER_BP + SLIPPAGE_BP) + (BYBIT_TAKER_BP + SLIPPAGE_BP))
THETA_IN_XV = ROUND_TURN_BP_XV / TARGET_HOLD_PERIODS * 1e-4
THETA_OUT_XV = THETA_IN_XV / 2.0
THETA_IN_XV_SLOW = ROUND_TURN_BP_XV / TARGET_HOLD_SLOW * 1e-4
THETA_OUT_XV_SLOW = THETA_IN_XV_SLOW / 2.0
MAX_POSITIONS_XV = 10
N_MIN_XV = 3

BYBIT_DEV_CACHE_DIR = "src/research/cache/bybit20/dev"
BYBIT_HOLDOUT_CACHE_DIR = "src/research/cache/bybit20/holdout"
PHASE1_MANIFEST_PATH = "scratch/020/phase1_manifest.json"
PHASE1A_PROBE_PATH = "scratch/020/phase1a_probe.json"

BUCKET_SECONDS = 8 * 3600


# --------------------------------------------------------------------------
# Mechanism A -- sec 5.5
# --------------------------------------------------------------------------


def add_trade_features_v2(
    panel: pl.DataFrame, *, half_life: int = CARRY_EWMA_HALF_LIFE
) -> pl.DataFrame:
    """bl18.add_trade_features with a parameterised carry half-life.
    Everything else -- liquidity screen, paired_log_return,
    fwd_paired_return_1, perp_log_return -- is bl18's own, unchanged.
    """
    panel = bl18.liquidity_screen(panel)
    panel = panel.sort(["symbol", "datetime"]).with_columns(
        bl18.carry_estimate(pl.col("funding_rate"), periods=half_life)
        .over("symbol")
        .alias("carry")
    )
    paired_ret = bl18.paired_log_return(
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


def build_book_weights_v2(
    panel: pl.DataFrame,
    *,
    timed: bool,
    n_min: int = 1,
    half_life: int = CARRY_EWMA_HALF_LIFE,
    theta_in: float = THETA_IN,
    theta_out: float = THETA_OUT,
    max_positions: int = MAX_POSITIONS,
) -> pl.DataFrame:
    """A copy of bl18.build_book_weights's sequential loop with the
    diversification floor added (sec 5.2). half_life is accepted for
    interface symmetry/manifest bookkeeping only -- panel's "carry" column
    must already be computed at the desired half-life (add_trade_features_v2
    recomputes it; this loop never does).

    Reduces EXACTLY to bl18.build_book_weights when n_min=1 and the 018
    defaults are passed (test_n_min_1_reproduces_018_weights, sec 5.7 #7)
    because the floor branch is a strict no-op at n_min<=1: a candidate list
    of length 0 already produces an all-zero, all-unheld bar under the
    ordinary (non-floor) path, so n_min=1 never triggers extra behaviour.

    The floor is a STAND-DOWN, not a fill-up (sec 5.2): if fewer than n_min
    symbols qualify, ALL weights go to 0 (held resets to False for every
    symbol -- the turnover of that close is charged at the normal rate via
    the ordinary weight-diff mechanism, no special-cased cost). If n_min or
    more qualify, every qualifying symbol gets its natural 1/n share -- never
    padded up to a target count.
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

        candidates: list[tuple[str, float]] = []
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
                candidates.append((sym, carry_val))

        if timed and n_min > 1 and len(candidates) < n_min:
            in_book: list[tuple[str, float]] = []
        else:
            in_book = candidates

        if timed and len(in_book) > max_positions:
            in_book.sort(key=lambda x: x[1], reverse=True)
            in_book = in_book[:max_positions]

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


# --------------------------------------------------------------------------
# Mechanism B -- sec 5.4/5.5
# --------------------------------------------------------------------------


def xvenue_paired_log_return(
    a_close: pl.Expr, b_close: pl.Expr, a_funding: pl.Expr, b_funding: pl.Expr
) -> pl.Expr:
    """One period of the w=+1 position ("short venue A's perp, long venue
    B's perp"), A=Binance, B=Bybit, per unit paired notional:

        r_t = log(b_close_t/b_close_{t-1}) - log(a_close_t/a_close_{t-1})
              + (a_funding_t - b_funding_t)

    i.e. long the cheap-funding venue's perp, short the expensive-funding
    venue's perp, collect the spread. w=-1 flips the whole expression (done
    by the caller via a signed weight, not by a separate code path -- see
    build_xvenue_book_weights). Short A receives a_funding when positive
    (018's own sign convention, sec 4.5); long B pays b_funding when
    positive -- hence +a_funding - b_funding.
    """
    a_leg = (a_close / a_close.shift(1)).log()
    b_leg = (b_close / b_close.shift(1)).log()
    return b_leg - a_leg + (a_funding - b_funding)


def xvenue_carry_estimate(
    spread: pl.Expr, periods: int = CARRY_EWMA_HALF_LIFE
) -> pl.Expr:
    """Causal EWMA of the SIGNED spread f_binance - f_bybit. Sign selects
    direction (positive => short Binance/long Bybit is the carry-positive
    trade); magnitude drives hysteresis (xvenue_qualifies).
    """
    return spread.ewm_mean(half_life=periods, adjust=False, ignore_nulls=False)


def xvenue_qualifies(
    carry: pl.Expr,
    liquid: pl.Expr,
    held: pl.Expr,
    *,
    theta_in: float = THETA_IN_XV,
    theta_out: float = THETA_OUT_XV,
) -> pl.Expr:
    """Hysteresis on |carry| -- sign selects direction, not eligibility.
    Enter when |carry| > theta_in, hold until |carry| < theta_out.
    """
    abs_carry = carry.abs()
    return liquid & pl.when(held).then(abs_carry > theta_out).otherwise(
        abs_carry > theta_in
    )


def build_xvenue_book_weights(
    panel: pl.DataFrame,
    *,
    timed: bool,
    n_min: int = N_MIN_XV,
    half_life: int = CARRY_EWMA_HALF_LIFE,
    theta_in: float = THETA_IN_XV,
    theta_out: float = THETA_OUT_XV,
    max_positions: int = MAX_POSITIONS_XV,
) -> pl.DataFrame:
    """Sequential, causal cross-venue book construction, mirroring
    build_book_weights_v2's shape with one structural difference: carry can
    be negative, so hysteresis and the diversification floor operate on
    |carry|, and the output "weight" column is SIGNED -- weight>0 means the
    w=+1 direction (short Binance perp / long Bybit perp), weight<0 the
    opposite (sign(carry) at the bar a symbol is held).

    A held symbol's direction is re-set to sign(carry) EVERY bar it stays
    held (not frozen at entry), so a sign flip is charged exactly like a
    close+reopen through the ordinary |weight_t - weight_{t-1}| turnover
    accounting research.portfolio_turnover already does -- no special-cased
    branch needed (test_xvenue_sign_flip pins this: the resulting turnover
    on a flip bar is 2x a single position's weight, i.e. two round-turn
    halves, not a free flip).
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

        candidates: list[tuple[str, float]] = []
        for sym, carry, liquid in zip(symbols, carries, liquids, strict=True):
            liquid_bool = bool(liquid) if liquid is not None else False
            carry_val = carry if carry is not None else 0.0
            abs_carry = abs(carry_val)
            if timed:
                was_held = held.get(sym, False)
                threshold = theta_out if was_held else theta_in
                qualify = liquid_bool and abs_carry > threshold
            else:
                qualify = liquid_bool
            if qualify:
                candidates.append((sym, carry_val))

        if timed and n_min > 1 and len(candidates) < n_min:
            in_book: list[tuple[str, float]] = []
        else:
            in_book = candidates

        if timed and len(in_book) > max_positions:
            in_book.sort(key=lambda x: abs(x[1]), reverse=True)
            in_book = in_book[:max_positions]

        in_book_map = {sym: carry_val for sym, carry_val in in_book}
        held = {sym: (sym in in_book_map) for sym in symbols}

        n = len(in_book_map)
        w_mag = 1.0 / n if n > 0 else 0.0
        for sym in symbols:
            if sym in in_book_map:
                direction = 1.0 if in_book_map[sym] >= 0 else -1.0
                w = direction * w_mag
            else:
                w = 0.0
            rows.append({"datetime": dt, "symbol": sym, "weight": w})

    schema: dict[str, pl.DataType | type] = {
        "datetime": panel["datetime"].dtype,
        "symbol": pl.Utf8,
        "weight": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def apply_xvenue_costs(trade_frame: pl.DataFrame) -> pl.DataFrame:
    """cost_frac = (PERP_TAKER_BP+SLIPPAGE_BP)*1e-4 + (BYBIT_TAKER_BP+SLIPPAGE_BP)*1e-4
    per unit of one-way turnover -- exactly half of ROUND_TURN_BP_XV,
    mirroring bl18.apply_two_leg_costs' structure (sec 5.5).
    """
    a_leg_cost = (PERP_TAKER_BP + SLIPPAGE_BP) * 1e-4
    b_leg_cost = (BYBIT_TAKER_BP + SLIPPAGE_BP) * 1e-4
    cost_frac = a_leg_cost + b_leg_cost
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
# Mechanism B -- Bybit data assembly (sec 4.2/4.4)
# --------------------------------------------------------------------------


def aggregate_bybit_kline_4h_to_8h(df: pl.DataFrame) -> pl.DataFrame:
    """Group Bybit's native 4h bars (sec 4.2: interval=480 is silently
    rejected by Bybit's kline endpoint) into Binance-aligned 8h buckets:
    open=first (by time), high=max, low=min, close=last (by time),
    volume/turnover=sum. A bucket assembled from fewer than 2 sub-bars
    (possible at the very edges of a fetched window) is still emitted, with
    "n_subbars" recording how many, so a caller can decide whether to trim it.
    """
    if len(df) == 0:
        return df.with_columns(pl.lit(0).cast(pl.UInt32).alias("n_subbars"))
    return (
        df.sort("datetime")
        .with_columns(
            ((pl.col("datetime").dt.epoch("s") // BUCKET_SECONDS) * BUCKET_SECONDS)
            .cast(pl.Int64)
            .alias("_bucket_epoch_s")
        )
        .group_by("_bucket_epoch_s", maintain_order=True)
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("turnover").sum().alias("turnover"),
            pl.len().alias("n_subbars"),
        )
        .with_columns(pl.from_epoch("_bucket_epoch_s", time_unit="s").alias("datetime"))
        .select(
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "n_subbars",
        )
        .sort("datetime")
    )


def resample_funding_to_8h(
    funding: pl.DataFrame, funding_interval_min: int
) -> pl.DataFrame:
    """Sec 4.4: for each Binance settlement stamp t, Bybit's funding for
    that period is the SUM of every Bybit funding payment settling in
    (t-8h, t] -- never an average, since funding is a cash flow and a
    symbol paying 1h funding pays eight times per Binance period.

    funding_interval_min must divide 480 evenly; a non-divisor raises
    ValueError (excluded, never interpolated -- sec 4.4). Also returns
    "n_payments" per bucket so a caller can assert every bucket has the
    expected count (480 // funding_interval_min) and report the ones that
    don't as a data-quality count -- the sum rule itself handles a
    mid-history interval change correctly, it just needs disclosing.
    """
    if 480 % funding_interval_min != 0:
        raise ValueError(
            f"funding_interval_min={funding_interval_min} does not divide "
            "480 evenly -- excluding this symbol rather than interpolating (sec 4.4)"
        )
    if len(funding) == 0:
        return funding.with_columns(pl.lit(0).cast(pl.UInt32).alias("n_payments"))
    return (
        funding.sort("datetime")
        .with_columns(
            (
                ((pl.col("datetime").dt.epoch("s") - 1) // BUCKET_SECONDS + 1)
                * BUCKET_SECONDS
            )
            .cast(pl.Int64)
            .alias("_bucket_end_epoch_s")
        )
        .group_by("_bucket_end_epoch_s", maintain_order=True)
        .agg(
            pl.col("funding_rate").sum().alias("funding_rate"),
            pl.len().alias("n_payments"),
        )
        .with_columns(
            pl.from_epoch("_bucket_end_epoch_s", time_unit="s").alias("datetime")
        )
        .select("datetime", "funding_rate", "n_payments")
        .sort("datetime")
    )


def load_xvenue_universe() -> list[str]:
    """Symbols with a successfully-cached dev-window Bybit fetch (sec 4.3) --
    the natural "universe seed" for Mechanism B, read from the Phase 1b
    manifest rather than re-deriving the Binance/Bybit intersection.
    """
    with open(PHASE1_MANIFEST_PATH) as f:
        manifest = json.load(f)
    return sorted(
        sym
        for sym, v in manifest["symbols"].items()
        if isinstance(v, dict) and v.get("dev", {}).get("status") == "ok"
    )


def _funding_interval_map() -> dict[str, int]:
    with open(PHASE1A_PROBE_PATH) as f:
        probe = json.load(f)
    return dict(probe["universe_intersection"]["_funding_interval_map"])


def _load_bybit_dev_raw(symbol: str) -> dict[str, pl.DataFrame | None]:
    kline_path = (
        Path(BYBIT_DEV_CACHE_DIR)
        / f"{symbol}-kline4h-{DEV_START:%Y-%m-%d}-{DEV_END:%Y-%m-%d}.parquet"
    )
    funding_path = (
        Path(BYBIT_DEV_CACHE_DIR)
        / f"{symbol}-funding-{DEV_START:%Y-%m-%d}-{DEV_END:%Y-%m-%d}.parquet"
    )
    kline = pl.read_parquet(kline_path) if kline_path.exists() else None
    funding = pl.read_parquet(funding_path) if funding_path.exists() else None
    return {"kline": kline, "funding": funding}


def _bybit_symbol_frame(symbol: str, funding_interval_min: int) -> pl.DataFrame | None:
    raw = _load_bybit_dev_raw(symbol)
    if raw["kline"] is None or raw["funding"] is None or len(raw["kline"]) == 0:
        return None
    kline_8h = aggregate_bybit_kline_4h_to_8h(raw["kline"])
    funding_8h = resample_funding_to_8h(raw["funding"], funding_interval_min)
    joined = kline_8h.select(
        "datetime",
        pl.col("close").alias("bybit_close"),
        pl.col("turnover").alias("bybit_dollar_volume"),
    ).join(
        funding_8h.select(
            "datetime", pl.col("funding_rate").alias("bybit_funding_rate")
        ),
        on="datetime",
        how="inner",
    )
    if len(joined) == 0:
        return None
    return joined.with_columns(pl.lit(symbol).alias("symbol"))


def _binance_perp_funding_frame(symbol: str) -> pl.DataFrame | None:
    """Binance's PERP close + funding only (sec 5.4/5.5's a_close/a_funding)
    -- no spot leg is needed for Mechanism B, but the data is read from the
    same basis18/dev cache 018 already built (sec 4.1: no new Binance
    infrastructure at all).
    """
    series = bl18.fetch_symbol_series(
        symbol, DEV_START, DEV_END, bl18.DEV_DOWNLOAD_DIR, bl18.DEV_CACHE_DIR
    )
    funding = bl18.load_dev_funding(symbol)
    if series["perp"] is None or funding is None:
        return None
    perp_s = series["perp"].select(
        "datetime",
        pl.col("close").alias("binance_close"),
        (pl.col("volume") * pl.col("close")).alias("binance_dollar_volume"),
    )
    funding_s = funding.sort("datetime").rename(
        {"funding_rate": "binance_funding_rate"}
    )
    joined = perp_s.sort("datetime").join_asof(
        funding_s, on="datetime", strategy="backward"
    )
    return joined


def load_xvenue_panel(
    symbols: Sequence[str] | None = None,
    start_date: datetime = DEV_START,
    end_date: datetime = DEV_END,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Cross-venue panel: Binance perp close+funding (via bl18, from
    basis18/dev) joined to Bybit perp close+funding (from bybit20/dev,
    aggregated/resampled onto Binance's 8h grid). Reads bybit20/dev and
    basis18/dev ONLY -- no parameter here can reach a holdout directory,
    mirroring bl18.load_basis_panel's guard verbatim (sec 5.5). Only
    run_phase_6_20_holdout.py may read past this boundary.
    """
    if end_date > research.HOLDOUT_START:
        raise ValueError(
            f"end_date {end_date} reaches into the frozen holdout period "
            f"(>= {research.HOLDOUT_START:%Y-%m-%d}). load_xvenue_panel never "
            "accepts allow_holdout -- only run_phase_6_20_holdout.py may read "
            "past this boundary."
        )
    if symbols is None:
        symbols = load_xvenue_universe()

    funding_intervals = _funding_interval_map()

    frames = []
    manifest: dict[str, str] = {}
    for symbol in symbols:
        fi = funding_intervals.get(symbol)
        if fi is None:
            manifest[symbol] = "no_funding_interval_on_record"
            continue
        if 480 % fi != 0:
            manifest[symbol] = "excluded_nondivisor_funding_interval"
            continue
        bybit_frame = _bybit_symbol_frame(symbol, fi)
        if bybit_frame is None:
            manifest[symbol] = "no_bybit_data"
            continue
        binance_frame = _binance_perp_funding_frame(symbol)
        if binance_frame is None:
            manifest[symbol] = "no_binance_perp_or_funding"
            continue

        joined = binance_frame.join(bybit_frame, on="datetime", how="inner")
        joined = joined.filter(
            (pl.col("datetime") >= start_date.replace(tzinfo=None))
            & (pl.col("datetime") <= end_date.replace(tzinfo=None))
        )
        if len(joined) == 0:
            manifest[symbol] = "empty_after_join"
            continue
        manifest[symbol] = "ok"
        frames.append(joined)

    if not frames:
        raise ValueError("No symbols survived Binance/Bybit cross-venue assembly")

    panel = pl.concat(frames, how="diagonal_relaxed").sort(["datetime", "symbol"])
    return panel, manifest


def xvenue_liquidity_screen(
    panel: pl.DataFrame,
    floor_usd: float = LIQUIDITY_FLOOR_USD,
    window_bars: int = LIQUIDITY_WINDOW_BARS,
) -> pl.DataFrame:
    """Both venues independently clear the liquidity floor (sec 5.5):
    Binance on close*volume (018's own approximation, unchanged), Bybit on
    its own turnover field directly (sec 4.2) -- the asymmetry disclosed,
    not "fixed" on either side.
    """
    return (
        panel.sort(["symbol", "datetime"])
        .with_columns(
            [
                pl.col("binance_dollar_volume")
                .rolling_median(window_size=window_bars, min_samples=window_bars)
                .over("symbol")
                .alias("binance_dollar_volume_median"),
                pl.col("bybit_dollar_volume")
                .rolling_median(window_size=window_bars, min_samples=window_bars)
                .over("symbol")
                .alias("bybit_dollar_volume_median"),
            ]
        )
        .with_columns(
            (
                (pl.col("binance_dollar_volume_median") >= floor_usd)
                & (pl.col("bybit_dollar_volume_median") >= floor_usd)
            ).alias("liquid")
        )
    )


def add_xvenue_trade_features(
    panel: pl.DataFrame, *, half_life: int = CARRY_EWMA_HALF_LIFE
) -> pl.DataFrame:
    """xvenue analogue of add_trade_features_v2: liquidity screen, causal
    signed-spread carry, realized xvenue_paired_log_return, and its
    forward-shifted label (fwd_xvenue_paired_return_1, the return of the
    FIXED w=+1 direction -- a signed weight from build_xvenue_book_weights
    recovers the w=-1 case by multiplication, sec 5.5's design).
    """
    panel = xvenue_liquidity_screen(panel)
    spread = pl.col("binance_funding_rate") - pl.col("bybit_funding_rate")
    panel = panel.sort(["symbol", "datetime"]).with_columns(
        xvenue_carry_estimate(spread, periods=half_life).over("symbol").alias("carry")
    )
    paired_ret = xvenue_paired_log_return(
        pl.col("binance_close"),
        pl.col("bybit_close"),
        pl.col("binance_funding_rate"),
        pl.col("bybit_funding_rate"),
    )
    panel = panel.with_columns(
        paired_ret.over("symbol").alias("xvenue_paired_log_return")
    )
    panel = panel.with_columns(
        pl.col("xvenue_paired_log_return")
        .shift(-1)
        .over("symbol")
        .alias("fwd_xvenue_paired_return_1")
    )
    return panel


# --------------------------------------------------------------------------
# Shared metrics (thin re-export so Phase 4/5 scripts import one module)
# --------------------------------------------------------------------------


def book_trade_frame(
    panel: pl.DataFrame, weights: pl.DataFrame, target_col: str, origin_offset: int = 0
) -> pl.DataFrame:
    trade_frame = research.portfolio_trade_frame(weights, panel, target_col=target_col)
    if origin_offset > 0:
        unique_times = trade_frame["datetime"].unique(maintain_order=True).sort()
        if origin_offset < len(unique_times):
            cutoff = unique_times[origin_offset]
            trade_frame = trade_frame.filter(pl.col("datetime") >= cutoff)
    return trade_frame


def book_metrics(
    costed_trade_frame: pl.DataFrame, annualized_rate: float, label: str = "book"
) -> dict[str, Any]:
    return bl18.book_metrics(costed_trade_frame, annualized_rate, label)


def ols_beta(returns: np.ndarray, benchmark: np.ndarray) -> float:
    return bl18.ols_beta(returns, benchmark)
