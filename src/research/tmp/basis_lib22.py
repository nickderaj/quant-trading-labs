"""Notebook 022 -- CEX/DEX funding-rate spread (Hyperliquid vs Binance).
NEXT_PROMPT.md Candidate 1.

Imports basis_lib20 (as bl20, which itself imports basis_lib18 as bl18) and
research, and reuses their primitives; it duplicates nothing it can call.
Never edits any of them. bl20's cross-venue machinery
(xvenue_paired_log_return, xvenue_carry_estimate, build_xvenue_book_weights,
book_trade_frame, book_metrics, ols_beta) is fully generic on column names
and is imported here unchanged -- this file only supplies what is genuinely
new: Hyperliquid-specific constants, its own cost model, and panel assembly
that reads src/research/cache/hyperliquid22/dev/ instead of bybit20/dev/.

Sign convention: throughout this file "A" = Hyperliquid, "B" = Binance, so
bl20.xvenue_paired_log_return(hl_close, binance_close, hl_funding,
binance_funding) = binance_leg - hl_leg + (hl_funding - binance_funding),
and carry = EWMA(hl_funding - binance_funding). carry > 0 means Hyperliquid
funding is the more expensive side -- the structural prior from
NEXT_PROMPT.md ("on-chain venues pay structurally higher funding") -- and
the w=+1 direction is short Hyperliquid's perp / long Binance's perp,
collecting the spread. This is a SIGNED-carry book exactly like Mechanism B
in bl20, not a fixed-direction one.

Data-loading fence, mirroring bl18/bl20's own: load_hlvenue_panel always
reads from hyperliquid22/dev and basis18/dev, and raises if asked to reach
past research.HOLDOUT_START. No other function in this file may ever be
handed a holdout-reaching date; that guard is not optional (the 2025-07-01+
Hyperliquid holdout is freely fetchable from the same public endpoint that
fed Phase 1b -- the fence here is a matter of discipline, not of what is on
disk, and this is the one place that discipline is enforced in code).
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
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import basis_lib18 as bl18
import basis_lib20 as bl20

import research

# --------------------------------------------------------------------------
# Inherited unchanged from 018/020 -- re-exported, not re-derived.
# --------------------------------------------------------------------------
PERP_TAKER_BP = bl18.PERP_TAKER_BP  # 5.0, Binance
SLIPPAGE_BP = bl18.SLIPPAGE_BP  # 1.0
LIQUIDITY_FLOOR_USD = bl18.LIQUIDITY_FLOOR_USD
LIQUIDITY_WINDOW_BARS = bl20.LIQUIDITY_WINDOW_BARS
CARRY_EWMA_HALF_LIFE = bl18.CARRY_EWMA_PERIODS  # 21
TARGET_HOLD_PERIODS = bl18.TARGET_HOLD_PERIODS  # 45 (8h periods, ~15 days)
TARGET_HOLD_SLOW = bl20.TARGET_HOLD_SLOW  # 90 (~30 days)
INTERVAL = bl18.INTERVAL

# --------------------------------------------------------------------------
# Hyperliquid-specific constants (NEXT_PROMPT.md Candidate 1, "Fees favour
# this pair"). 1.5bp maker / 4.5bp taker published; this book, like every
# other book in the repo, prices at TAKER on both legs -- conservative, and
# consistent with bl18/bl20.
# --------------------------------------------------------------------------
HL_TAKER_BP = 4.5
HL_MAKER_BP = 1.5  # unused; recorded for the pre-registration's cost table only

ROUND_TURN_BP_HL = 2 * ((PERP_TAKER_BP + SLIPPAGE_BP) + (HL_TAKER_BP + SLIPPAGE_BP))
# == 23.0bp: cheaper than 020's Bybit round turn (25.0bp) because Hyperliquid's
# taker fee (4.5bp) is cheaper than Bybit's (5.5bp); both legs still charged
# 1bp slippage each, same convention as bl20.ROUND_TURN_BP_XV. The 19bp
# figure in NEXT_PROMPT.md's planning survey is the bare-fee (no slippage)
# comparator quoted against 020's fully-costed 25bp; this constant uses the
# same fully-costed convention as every other book in the repo so headline
# numbers are apples-to-apples with 018/020, not with the planning survey.

# Three pre-registered configurations (NEXT_PROMPT.md's design implication:
# "always-on is a live candidate", plus a fast-timing falsification
# comparator and a middle ground):
#   HL_ALWAYSON  -- timed=False: every liquid symbol held every bar,
#                   direction=sign(carry). The pre-registered headline.
#   HL_TIMED_FAST -- bl20's own XV thetas (45-period ~15-day target hold).
#                    Falsification comparator: NEXT_PROMPT.md says needing
#                    this to clear the tradeability gate while ALWAYSON does
#                    not would itself be evidence against the structural story.
#   HL_TIMED_SLOW -- 90-period (~30-day) target hold, this book's own cost.
MAX_POSITIONS_HL = 10
N_MIN_HL = 3  # only binds when timed=True (build_xvenue_book_weights' own no-op
# rule at n_min<=1 / timed=False applies here unchanged, sec bl20 docstring)

THETA_IN_HL_FAST = ROUND_TURN_BP_HL / TARGET_HOLD_PERIODS * 1e-4
THETA_OUT_HL_FAST = THETA_IN_HL_FAST / 2.0
THETA_IN_HL_SLOW = ROUND_TURN_BP_HL / TARGET_HOLD_SLOW * 1e-4
THETA_OUT_HL_SLOW = THETA_IN_HL_SLOW / 2.0

HL_DEV_CACHE_DIR = "src/research/cache/hyperliquid22/dev"
PHASE1_MANIFEST_PATH = "scratch/022/phase1_manifest.json"
PHASE1A_PROBE_PATH = "scratch/022/phase1a_probe.json"

# Matches Probe P1/P2's own window (NEXT_PROMPT.md Candidate 1) -- shorter
# than 018/020's 2021-07-01 start because Hyperliquid's own BTC history only
# begins mid-2023.
DEV_START = datetime(2023, 7, 1, tzinfo=research.UTC)
DEV_END = datetime(2025, 6, 30, tzinfo=research.UTC)


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------


def load_hlvenue_universe() -> dict[str, str]:
    """Binance-symbol -> Hyperliquid-coin map for every symbol with a
    successfully-cached dev-window Hyperliquid fetch (Phase 1b manifest).
    """
    with open(PHASE1_MANIFEST_PATH) as f:
        manifest = json.load(f)
    return {
        sym: v["hl_coin"]
        for sym, v in manifest["symbols"].items()
        if isinstance(v, dict) and v.get("status") == "ok"
    }


FROZEN_FEED_EXCLUSIONS_PATH = "scratch/022/frozen_feed_exclusions.json"


def load_frozen_feed_screened_symbols() -> list[str]:
    """The mapped universe (load_hlvenue_universe) minus every symbol whose
    BINANCE perp leg is frozen-feed-flagged on more than 5% of its own
    dev-window bars (run_phase_2b_22_frozen_feed_screen.py's disclosed,
    mechanical rule -- found necessary after Phase 4's first run produced an
    implausible headline traced to FTMUSDT's frozen Binance feed during its
    Sonic migration). Every phase from Phase 2 onward uses THIS universe,
    not load_hlvenue_universe's raw mapping.
    """
    with open(FROZEN_FEED_EXCLUSIONS_PATH) as f:
        screen = json.load(f)
    return list(screen["kept_symbols"])


def _load_hl_dev_raw(hl_coin: str) -> dict[str, pl.DataFrame | None]:
    funding_path = (
        Path(HL_DEV_CACHE_DIR)
        / f"{hl_coin}-funding-{DEV_START:%Y-%m-%d}-{DEV_END:%Y-%m-%d}.parquet"
    )
    candle_path = (
        Path(HL_DEV_CACHE_DIR)
        / f"{hl_coin}-candles8h-{DEV_START:%Y-%m-%d}-{DEV_END:%Y-%m-%d}.parquet"
    )
    funding = pl.read_parquet(funding_path) if funding_path.exists() else None
    candle = pl.read_parquet(candle_path) if candle_path.exists() else None
    return {"funding": funding, "candle": candle}


HL_FUNDING_INTERVAL_MIN = 60  # native cadence, verified directly (module docstring
# of run_phase_1b_22_fetch_hyperliquid.py) -- uniform across every symbol, unlike
# Bybit's per-symbol interval in 020, so no probe/map is needed here.


def _hl_symbol_frame(binance_symbol: str, hl_coin: str) -> pl.DataFrame | None:
    """Joins HL's 8h candles to its own funding, resampled from native
    HOURLY cadence into 8h buckets via bl20.resample_funding_to_8h (summed,
    not averaged -- funding is a cash flow, and this symbol pays 8x per
    Binance settlement period). Without this resample, an exact-datetime
    join between 8h candles and raw hourly funding matches almost nothing
    (caught before shipping: joining the two directly left BTC's own panel
    at 12 rows instead of ~2,190 -- see phase_0_22_preregistration.json's
    bugs-found note).
    """
    raw = _load_hl_dev_raw(hl_coin)
    if raw["funding"] is None or raw["candle"] is None:
        return None
    if len(raw["funding"]) == 0 or len(raw["candle"]) == 0:
        return None
    funding_8h = bl20.resample_funding_to_8h(raw["funding"], HL_FUNDING_INTERVAL_MIN)
    joined = (
        raw["candle"]
        .select(
            "datetime",
            pl.col("close").alias("hl_close"),
            pl.col("dollar_volume").alias("hl_dollar_volume"),
        )
        .join(
            funding_8h.select(
                "datetime", pl.col("funding_rate").alias("hl_funding_rate")
            ),
            on="datetime",
            how="inner",
        )
    )
    if len(joined) == 0:
        return None
    return joined.with_columns(pl.lit(binance_symbol).alias("symbol"))


def load_hlvenue_panel(
    symbols: Sequence[str] | None = None,
    start_date: datetime = DEV_START,
    end_date: datetime = DEV_END,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Cross-venue panel: Binance perp close+funding (via bl20's own
    _binance_perp_funding_frame, unchanged, reading basis18/dev) joined to
    Hyperliquid perp close+funding (from hyperliquid22/dev). Reads
    hyperliquid22/dev and basis18/dev ONLY -- see module docstring.
    """
    if end_date > research.HOLDOUT_START:
        raise ValueError(
            f"end_date {end_date} reaches into the frozen holdout period "
            f"(>= {research.HOLDOUT_START:%Y-%m-%d}). load_hlvenue_panel never "
            "accepts allow_holdout -- see module docstring."
        )
    universe = load_hlvenue_universe()
    if symbols is None:
        symbols = sorted(universe)

    frames = []
    manifest: dict[str, str] = {}
    for symbol in symbols:
        hl_coin = universe.get(symbol)
        if hl_coin is None:
            manifest[symbol] = "not_in_hl_manifest"
            continue
        hl_frame = _hl_symbol_frame(symbol, hl_coin)
        if hl_frame is None:
            manifest[symbol] = "no_hl_data"
            continue
        binance_frame = bl20._binance_perp_funding_frame(symbol)
        if binance_frame is None:
            manifest[symbol] = "no_binance_perp_or_funding"
            continue

        joined = binance_frame.join(hl_frame, on="datetime", how="inner")
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
        raise ValueError("No symbols survived Binance/Hyperliquid cross-venue assembly")

    panel = pl.concat(frames, how="diagonal_relaxed").sort(["datetime", "symbol"])
    return panel, manifest


def hlvenue_liquidity_screen(
    panel: pl.DataFrame,
    floor_usd: float = LIQUIDITY_FLOOR_USD,
    window_bars: int = LIQUIDITY_WINDOW_BARS,
) -> pl.DataFrame:
    """Both venues independently clear the liquidity floor, mirroring
    bl20.xvenue_liquidity_screen exactly, on hl_dollar_volume instead of
    bybit_dollar_volume.
    """
    return (
        panel.sort(["symbol", "datetime"])
        .with_columns(
            [
                pl.col("binance_dollar_volume")
                .rolling_median(window_size=window_bars, min_samples=window_bars)
                .over("symbol")
                .alias("binance_dollar_volume_median"),
                pl.col("hl_dollar_volume")
                .rolling_median(window_size=window_bars, min_samples=window_bars)
                .over("symbol")
                .alias("hl_dollar_volume_median"),
            ]
        )
        .with_columns(
            (
                (pl.col("binance_dollar_volume_median") >= floor_usd)
                & (pl.col("hl_dollar_volume_median") >= floor_usd)
            ).alias("liquid")
        )
    )


def add_hlvenue_trade_features(
    panel: pl.DataFrame, *, half_life: int = CARRY_EWMA_HALF_LIFE
) -> pl.DataFrame:
    """hlvenue analogue of bl20.add_xvenue_trade_features. carry = causal
    EWMA of (hl_funding_rate - binance_funding_rate) -- see module docstring
    for the sign convention (A=Hyperliquid, B=Binance).
    """
    panel = hlvenue_liquidity_screen(panel)
    spread = pl.col("hl_funding_rate") - pl.col("binance_funding_rate")
    panel = panel.sort(["symbol", "datetime"]).with_columns(
        bl20.xvenue_carry_estimate(spread, periods=half_life)
        .over("symbol")
        .alias("carry")
    )
    paired_ret = bl20.xvenue_paired_log_return(
        pl.col("hl_close"),
        pl.col("binance_close"),
        pl.col("hl_funding_rate"),
        pl.col("binance_funding_rate"),
    )
    panel = panel.with_columns(
        paired_ret.over("symbol").alias("hlvenue_paired_log_return")
    )
    panel = panel.with_columns(
        pl.col("hlvenue_paired_log_return")
        .shift(-1)
        .over("symbol")
        .alias("fwd_hlvenue_paired_return_1")
    )
    return panel


def apply_hlvenue_costs(trade_frame: pl.DataFrame) -> pl.DataFrame:
    """cost_frac = (HL_TAKER_BP+SLIPPAGE_BP)*1e-4 + (PERP_TAKER_BP+SLIPPAGE_BP)*1e-4
    per unit of one-way turnover -- exactly half of ROUND_TURN_BP_HL,
    mirroring bl20.apply_xvenue_costs' structure.
    """
    hl_leg_cost = (HL_TAKER_BP + SLIPPAGE_BP) * 1e-4
    binance_leg_cost = (PERP_TAKER_BP + SLIPPAGE_BP) * 1e-4
    cost_frac = hl_leg_cost + binance_leg_cost
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


def apply_hlvenue_costs_at(
    trade_frame: pl.DataFrame, round_turn_bp: float
) -> pl.DataFrame:
    """As apply_hlvenue_costs, but at an arbitrary total round-turn bp
    (Phase 5's cost-sensitivity/stress ablations) instead of ROUND_TURN_BP_HL.
    """
    cost_frac = round_turn_bp / 2.0 * 1e-4
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
# Book construction -- thin wrapper naming build_xvenue_book_weights'
# parameters for each of the three pre-registered HL configurations.
# --------------------------------------------------------------------------


def build_hl_book_weights(panel: pl.DataFrame, config: str) -> pl.DataFrame:
    if config == "HL_ALWAYSON":
        return bl20.build_xvenue_book_weights(panel, timed=False)
    if config == "HL_TIMED_FAST":
        return bl20.build_xvenue_book_weights(
            panel,
            timed=True,
            n_min=N_MIN_HL,
            theta_in=bl20.THETA_IN_XV,
            theta_out=bl20.THETA_OUT_XV,
            max_positions=MAX_POSITIONS_HL,
        )
    if config == "HL_TIMED_SLOW":
        return bl20.build_xvenue_book_weights(
            panel,
            timed=True,
            n_min=N_MIN_HL,
            theta_in=THETA_IN_HL_SLOW,
            theta_out=THETA_OUT_HL_SLOW,
            max_positions=MAX_POSITIONS_HL,
        )
    raise ValueError(f"unknown config {config!r}")


# --------------------------------------------------------------------------
# Shared metrics (thin re-export so Phase 2/3/4 scripts import one module)
# --------------------------------------------------------------------------


def book_trade_frame(
    panel: pl.DataFrame, weights: pl.DataFrame, origin_offset: int = 0
) -> pl.DataFrame:
    return bl20.book_trade_frame(
        panel,
        weights,
        target_col="fwd_hlvenue_paired_return_1",
        origin_offset=origin_offset,
    )


def book_metrics(
    costed_trade_frame: pl.DataFrame, annualized_rate: float, label: str = "book"
) -> dict[str, Any]:
    return bl20.book_metrics(costed_trade_frame, annualized_rate, label)


def ols_beta(returns: np.ndarray, benchmark: np.ndarray) -> float:
    return bl20.ols_beta(returns, benchmark)


def mde_annualized_sharpe(
    n_obs: int, annualized_rate: float, power: float = 0.80, alpha: float = 0.05
) -> float:
    """Minimum detectable ANNUALIZED Sharpe ratio at the given power/alpha,
    for a per-period Sharpe estimated from n_obs i.i.d.-ish observations
    (HD-3's power gate). Standard result: SE(sharpe_hat) ~= 1/sqrt(n_obs)
    for a per-period Sharpe near 0 (the null this gate is testing against),
    so the minimum detectable per-period Sharpe is
    (z_{1-alpha/2} + z_power) / sqrt(n_obs), and annualizing multiplies by
    annualized_rate (research.sharpe_to_annualized_rate(INTERVAL) ==
    sqrt(periods_per_year)) -- the same sqrt(periods) scaling every Sharpe
    in this repo uses. This ignores the skew/kurtosis correction
    deflated_sharpe_prob applies; it is a feasibility bound computed BEFORE
    Phase 4 runs (021's lesson), not a substitute for the bootstrap CIs and
    DSR computed on the actual book afterwards.
    """
    z_alpha = st.norm.ppf(1 - alpha / 2)
    z_power = st.norm.ppf(power)
    mde_per_period = (z_alpha + z_power) / np.sqrt(n_obs)
    return float(mde_per_period * annualized_rate)
