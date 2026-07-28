"""Causal feature library for the cross-sectional panel.

Every raw feature at bar t is computable using only rows up to and including
t (no future data), verified per-feature by tests/test_features.py's
causality check: truncating a symbol's history at any point must leave every
earlier feature value unchanged.

Convention: raw features are per-symbol (rolling windows must never cross a
symbol boundary - see apply_per_symbol), while the "_cs_demean"/"_cs_z"
cross-sectional variants are computed per timestamp across the panel. That's
still causal: at time t, every symbol's bar t has already closed
simultaneously, so comparing symbols at bar t uses no information from
bar t+1 onward.

forward_return is the target, not a feature: it looks forward via
shift(-horizon) and must never appear on the feature side of a model.
"""

from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

# --------------------------------------------------------------------------
# Target (not a feature - uses shift(-horizon), i.e. future data)
# --------------------------------------------------------------------------


def forward_return(price_col: str = "close", horizon: int = 1) -> pl.Expr:
    """log(price_{t+horizon} / price_t): the return AFTER bar t.

    This is a label, computed with a negative (forward) shift. Never use it
    as a feature - only as the thing a model is trained to predict.
    """
    return (
        (pl.col(price_col).shift(-horizon) / pl.col(price_col))
        .log()
        .alias(f"fwd_return_{horizon}")
    )


# --------------------------------------------------------------------------
# Per-symbol application (rolling windows must not cross symbol boundaries)
# --------------------------------------------------------------------------


def apply_per_symbol(
    panel: pl.DataFrame, fn: Callable[[pl.DataFrame], pl.DataFrame]
) -> pl.DataFrame:
    """Run fn(sorted-by-datetime single-symbol frame) for every symbol in
    panel and concatenate the results back into one panel, re-sorted by
    (datetime, symbol).

    Rolling/lag features computed directly on the whole panel would leak
    across symbol boundaries (e.g. BTC's rolling vol window picking up ETH's
    trailing bars if the panel weren't grouped first); this guarantees each
    symbol only ever sees its own history.
    """
    parts = panel.partition_by("symbol", maintain_order=True)
    return pl.concat(
        [fn(p.sort("datetime")) for p in parts], how="diagonal_relaxed"
    ).sort(["datetime", "symbol"])


# --------------------------------------------------------------------------
# 1. Order flow - already in the cached klines, unused by prior notebooks
# --------------------------------------------------------------------------


def taker_buy_ratio() -> pl.Expr:
    """Fraction of a bar's volume that was taker-initiated buying, in [0, 1]."""
    return (pl.col("taker_buy_volume") / pl.col("volume")).alias("taker_buy_ratio")


def order_flow_imbalance() -> pl.Expr:
    """Signed taker buy/sell imbalance in [-1, 1]. +1 = all taker buys, -1 = all taker sells."""
    return (2.0 * (pl.col("taker_buy_volume") / pl.col("volume")) - 1.0).alias(
        "order_flow_imbalance"
    )


def avg_trade_size() -> pl.Expr:
    """Volume per trade in a bar - proxies whether a bar's flow was many small
    trades or a few large ones."""
    return (pl.col("volume") / pl.col("count")).alias("avg_trade_size")


def rolling_zscore(col: str, window: int) -> pl.Expr:
    """Causal rolling z-score of col: polars' rolling_mean/std are trailing
    (each row uses only itself and prior rows), so this never looks ahead.
    Nulls for the first (window - 1) rows of a symbol's history.
    """
    mean = pl.col(col).rolling_mean(window)
    std = pl.col(col).rolling_std(window)
    return ((pl.col(col) - mean) / std).alias(f"{col}_z{window}")


ORDER_FLOW_WINDOWS = (20, 60)


def add_order_flow_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(taker_buy_ratio(), order_flow_imbalance(), avg_trade_size())
    zscores = [
        rolling_zscore(col, w)
        for col in ("order_flow_imbalance", "avg_trade_size", "count")
        for w in ORDER_FLOW_WINDOWS
    ]
    return df.with_columns(zscores)


# --------------------------------------------------------------------------
# 2. Seasonality - hour-of-day / day-of-week, cyclically encoded
# --------------------------------------------------------------------------


def seasonality_exprs() -> list[pl.Expr]:
    """Cyclic (sin/cos) encodings of hour-of-day and day-of-week.

    Plain integer hour/day-of-week would tell a linear model that hour 23 and
    hour 0 are far apart when they're adjacent; sin/cos keeps the wraparound
    distance correct. Hour-of-day is degenerate at 1d bars (always 0) and
    coarse at 12h (only 0/12) - screening will show that directly rather than
    needing it hardcoded here.
    """
    hour = pl.col("datetime").dt.hour()
    dow = pl.col("datetime").dt.weekday()
    return [
        (2 * np.pi * hour / 24).sin().alias("hour_sin"),
        (2 * np.pi * hour / 24).cos().alias("hour_cos"),
        (2 * np.pi * dow / 7).sin().alias("dow_sin"),
        (2 * np.pi * dow / 7).cos().alias("dow_cos"),
    ]


def add_seasonality_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(seasonality_exprs())


# --------------------------------------------------------------------------
# 3. Realized vol - multi-window, vol-of-vol, vol regime
# --------------------------------------------------------------------------

VOL_WINDOWS = (8, 24, 96)


def log_return_expr(price_col: str = "close") -> pl.Expr:
    return (pl.col(price_col) / pl.col(price_col).shift(1)).log().alias("log_return")


def realized_vol(window: int) -> pl.Expr:
    """Trailing realized volatility: rolling std of log returns over window bars."""
    return pl.col("log_return").rolling_std(window).alias(f"realized_vol_{window}")


def add_vol_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(log_return_expr())
    df = df.with_columns([realized_vol(w) for w in VOL_WINDOWS])
    short, long = VOL_WINDOWS[0], VOL_WINDOWS[-1]
    df = df.with_columns(
        pl.col(f"realized_vol_{short}").rolling_std(long).alias(f"vol_of_vol_{long}"),
        # vol regime: current (short-window) vol relative to its own longer
        # trailing average - >1 means "vol is elevated vs its recent norm".
        (pl.col(f"realized_vol_{short}") / pl.col(f"realized_vol_{long}")).alias(
            "vol_regime"
        ),
    )
    return df


# --------------------------------------------------------------------------
# 4. Momentum / mean-reversion - the notebook-2 baseline to beat
# --------------------------------------------------------------------------

MOMENTUM_WINDOWS = (1, 4, 12)


def momentum(window: int) -> pl.Expr:
    """Cumulative log return over the trailing window bars, causal (uses shift only)."""
    return (
        (pl.col("close") / pl.col("close").shift(window))
        .log()
        .alias(f"momentum_{window}")
    )


def mean_reversion(window: int) -> pl.Expr:
    """Negative of momentum - a bet that recent moves partially revert."""
    return (-1.0 * (pl.col("close") / pl.col("close").shift(window)).log()).alias(
        f"mean_reversion_{window}"
    )


def add_momentum_features(df: pl.DataFrame) -> pl.DataFrame:
    exprs = [momentum(w) for w in MOMENTUM_WINDOWS] + [
        mean_reversion(w) for w in MOMENTUM_WINDOWS
    ]
    return df.with_columns(exprs)


# --------------------------------------------------------------------------
# 5. Funding rate (best-effort) - carry is the most robust crypto signal
# --------------------------------------------------------------------------


def add_funding_rate_feature(bars: pl.DataFrame, funding: pl.DataFrame) -> pl.DataFrame:
    """Join a symbol's funding rate history onto its bars.

    Backward asof join: each bar gets the most recently published funding
    rate as of its own timestamp, never a rate published after the bar
    closes, so this stays causal despite funding events (~every 8h) not
    lining up with arbitrary bar boundaries (4h/12h/1d).

    bars and funding must both be single-symbol frames (see
    apply_per_symbol) - call per-symbol with that symbol's own funding
    history, not the whole panel at once.
    """
    joined = bars.sort("datetime").join_asof(
        funding.sort("datetime"), on="datetime", strategy="backward"
    )
    return joined.with_columns(rolling_zscore("funding_rate", 20))


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

RAW_FEATURE_BUILDERS: tuple[Callable[[pl.DataFrame], pl.DataFrame], ...] = (
    add_order_flow_features,
    add_seasonality_features,
    add_vol_features,
    add_momentum_features,
)


def add_all_raw_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add every per-symbol causal feature to a single symbol's
    sorted-by-datetime OHLCV frame. Call via apply_per_symbol on a panel so
    rolling windows never cross a symbol boundary.
    """
    for builder in RAW_FEATURE_BUILDERS:
        df = builder(df)
    return df


# --------------------------------------------------------------------------
# Cross-sectional variants (per-bar, across symbols) - what the ranking model consumes
# --------------------------------------------------------------------------


def cross_sectional_demean(col: str) -> pl.Expr:
    """col minus its cross-sectional (same-timestamp, across symbols) mean."""
    return (pl.col(col) - pl.col(col).mean().over("datetime")).alias(f"{col}_cs_demean")


def cross_sectional_zscore(col: str) -> pl.Expr:
    """col standardized against its own timestamp's cross-section: what the
    pooled ranking model in Phase 5 actually consumes, so a fitted weight
    means the same thing for every symbol regardless of that symbol's own
    scale.
    """
    mean = pl.col(col).mean().over("datetime")
    std = pl.col(col).std().over("datetime")
    return ((pl.col(col) - mean) / std).alias(f"{col}_cs_z")


def add_cross_sectional_features(
    panel: pl.DataFrame, cols: Sequence[str]
) -> pl.DataFrame:
    """Add _cs_demean/_cs_z variants of cols to a full (multi-symbol) panel.

    Requires "datetime" and "symbol" columns; operates across the whole
    panel (not per-symbol), since the point is comparing symbols to each
    other at the same bar.
    """
    exprs = []
    for col in cols:
        exprs.append(cross_sectional_demean(col))
        exprs.append(cross_sectional_zscore(col))
    return panel.with_columns(exprs)


def build_feature_panel(
    panel: pl.DataFrame,
    cross_sectional_cols: Sequence[str] | None = None,
    funding_by_symbol: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """End-to-end: raw per-symbol features (via apply_per_symbol) -> forward
    return target -> cross-sectional demean/z-score of the requested columns.

    funding_by_symbol is optional (best-effort, see add_funding_rate_feature)
    - a dict of symbol -> that symbol's funding rate history. Symbols
    missing from the dict simply don't get a funding_rate column filled in
    (left null), so a partial funding dataset doesn't block the rest of the
    panel.

    cross_sectional_cols defaults to every raw feature column added by
    add_all_raw_features (i.e. everything except the original OHLCV columns
    and "log_return").
    """

    def _build_one(df: pl.DataFrame) -> pl.DataFrame:
        df = add_all_raw_features(df)
        if funding_by_symbol is not None:
            symbol = df["symbol"][0]
            funding = funding_by_symbol.get(symbol)
            if funding is not None:
                df = add_funding_rate_feature(df, funding)
        return df.with_columns(forward_return())

    featured = apply_per_symbol(panel, _build_one)

    if cross_sectional_cols is None:
        base_cols = set(panel.columns)
        cross_sectional_cols = [
            c
            for c in featured.columns
            if c not in base_cols and not c.startswith("fwd_return_")
        ]

    return add_cross_sectional_features(featured, cross_sectional_cols)
