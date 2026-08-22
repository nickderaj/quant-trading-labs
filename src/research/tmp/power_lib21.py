"""Notebook 021 library (NEXT_PROMPT.md sec 4 Phase 2): the liquidity-collapse
frozen-feed detector, the book-return exclusion-set alignment rule, and the
closed-form power/MDE helpers. Imports research, basis_lib18, basis_lib20;
edits none of them.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import polars as pl
import scipy.stats as st

_FROZEN_FEED_FLAG = (
    (pl.col("open") == pl.col("high"))
    & (pl.col("high") == pl.col("low"))
    & (pl.col("low") == pl.col("close"))
    & (pl.col("volume") == 0)
)


def flag_frozen_feed_bars(perp_glob: str) -> pl.DataFrame:
    """018's frozen-feed signature: open == high == low == close AND
    volume == 0 on the perp leg. Scans every file matching perp_glob (one
    per symbol, full-range). Returns columns (symbol, datetime), one row per
    flagged panel bar. No halo, no minimum run length, no widening.
    """
    frames = []
    for path in sorted(glob.glob(perp_glob)):
        symbol = os.path.basename(path).split("-perp-")[0]
        d = pl.read_parquet(
            path, columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        flagged = (
            d.filter(_FROZEN_FEED_FLAG)
            .select("datetime")
            .with_columns(pl.lit(symbol).alias("symbol"))
        )
        if len(flagged):
            frames.append(flagged)
    if not frames:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "datetime": pl.Datetime})
    return pl.concat(frames).select("symbol", "datetime").sort(["symbol", "datetime"])


def excluded_book_bars(catalogue: pl.DataFrame, a0_weights: pl.DataFrame) -> set:
    """The {T-1, T} book-return alignment rule (NEXT_PROMPT.md sec 3): a
    flagged panel bar (symbol, T) contaminates book returns recorded at
    t=T-1 and t=T, restricted to bars where the symbol carried non-zero
    weight in a0_weights at that same t. a0_weights must have columns
    (datetime, symbol, weight).

    Returns the set of excluded book-return datetimes.
    """
    held = a0_weights.filter(pl.col("weight") != 0).select("symbol", "datetime")
    held_set = set(
        zip(held["symbol"].to_list(), held["datetime"].to_list(), strict=True)
    )

    times = sorted(a0_weights["datetime"].unique().to_list())
    idx = {t: i for i, t in enumerate(times)}

    excluded: set = set()
    for symbol, big_t in zip(
        catalogue["symbol"].to_list(), catalogue["datetime"].to_list(), strict=True
    ):
        i = idx.get(big_t)
        if i is None:
            continue
        if (symbol, big_t) in held_set:
            excluded.add(big_t)
        if i > 0:
            t_prev = times[i - 1]
            if (symbol, t_prev) in held_set:
                excluded.add(t_prev)
    return excluded


def bootstrap_se_from_ci(ci_lo: float, ci_hi: float) -> float:
    """Inverts a 95% two-sided bootstrap CI back to the implied standard
    error, assuming the CI is (approximately) normal-quantile-based:
    se = (hi - lo) / (2 * z_0.975).
    """
    z = st.norm.ppf(0.975)
    return float((ci_hi - ci_lo) / (2 * z))


def mde(se: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable effect at the given power/alpha, closed-form from
    a standard error: mde = (z_{1-alpha/2} + z_power) * se.
    """
    z_alpha = st.norm.ppf(1 - alpha / 2)
    z_power = st.norm.ppf(power)
    return float((z_alpha + z_power) * se)


def n_required(n_obs: int, observed_mean: float, mde_value: float) -> float:
    """Sample size required for the observed mean to reach mde_value at the
    same power/alpha, using SE ~ 1/sqrt(n): n_required = n_obs * (mde /
    observed_mean) ** 2.
    """
    return float(n_obs * (mde_value / observed_mean) ** 2)


def placebo_mean_diffs(
    diff_frame: pl.DataFrame, n_excluded: int, n_draws: int = 200, seed: int = 0
) -> np.ndarray:
    """PW-4's placebo control: draw n_draws random subsets of size
    n_excluded from diff_frame's own bars (without replacement), exclude
    them, and return the array of resulting means. diff_frame must have a
    "diff" column. Compares means only, not CIs (sec 3/5's cost note).
    """
    values = diff_frame["diff"].to_numpy()
    n = len(values)
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for i in range(n_draws):
        drop_idx = rng.choice(n, size=n_excluded, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[drop_idx] = False
        out[i] = values[mask].mean()
    return out
