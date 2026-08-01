"""Notebook-7-local machinery: turnover reduction, risk-gating, carry, and
tail-factor construction, on top of notebook 3/6's already-built and tested
`research`/`features`/`dist_lib6` machinery.

Run as a script from the repo root (`sys.path.insert(0, "src/research/tmp")`
and `sys.path.insert(0, "src")`), and imported from the notebook the same
way (`sys.path.insert(0, "..")`, `sys.path.insert(0, "tmp")` from
`src/research/`).
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

import numpy as np
import polars as pl

import research


def hysteresis_weights(
    panel: pl.DataFrame,
    pred_col: str,
    band: float,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    top_frac: float = 0.2,
    size_col: str | None = None,
    gross_exposure: float = 1.0,
    max_position_per_symbol: float = 0.25,
) -> pl.DataFrame:
    """`research.dollar_neutral_weights`, but a symbol only ENTERS a leg when
    it ranks in the top/bottom `top_frac` of that bar's cross-section, and
    only EXITS once it has fallen outside the wider top/bottom
    `top_frac + band` band - between those two thresholds it holds its
    previous bar's leg membership rather than being re-picked from scratch.

    Uses rank counts (k_enter = floor(n*top_frac), k_exit = floor(n*(top_frac
    + band))), not percentile thresholds, specifically so that `band=0.0`
    collapses k_exit to k_enter and this function produces EXACTLY the same
    top-k/bottom-k membership every bar, with no persistence effect, as
    `research.dollar_neutral_weights` - the single most important
    correctness property of this function (see tests/test_alpha_lib7.py).

    Weighting within a leg (proportional to |size_col| or equal, normalized
    to +-gross_exposure/2 per leg, then clipped to max_position_per_symbol)
    is otherwise identical to `dollar_neutral_weights`.
    """
    if band < 0:
        raise ValueError(f"band must be >= 0, got {band}")

    cols = [datetime_col, symbol_col, pred_col] + ([size_col] if size_col else [])
    df = panel.select(cols).drop_nulls().sort(datetime_col)

    state: dict[str, str] = {}  # symbol -> "long" | "short" (absent = neutral)
    rows: list[dict[str, Any]] = []

    for key, group in df.group_by(datetime_col, maintain_order=True):
        n = len(group)
        k_enter = max(1, int(np.floor(n * top_frac)))
        k_exit = max(k_enter, int(np.floor(n * (top_frac + band))))

        preds = group[pred_col].to_numpy()
        symbols = group[symbol_col].to_list()
        size = np.abs(group[size_col].to_numpy()) if size_col else np.ones(n)
        order = np.argsort(preds)  # ascending: order[0] = lowest pred

        long_enter = set(np.array(symbols)[order[-k_enter:]])
        long_keep = set(np.array(symbols)[order[-k_exit:]])
        short_enter = set(np.array(symbols)[order[:k_enter]])
        short_keep = set(np.array(symbols)[order[:k_exit]])

        new_state: dict[str, str] = {}
        for sym in symbols:
            prev = state.get(sym)
            if prev == "long":
                if sym in long_keep:
                    new_state[sym] = "long"
                elif sym in short_enter:
                    new_state[sym] = "short"
            elif prev == "short":
                if sym in short_keep:
                    new_state[sym] = "short"
                elif sym in long_enter:
                    new_state[sym] = "long"
            else:
                if sym in long_enter:
                    new_state[sym] = "long"
                elif sym in short_enter:
                    new_state[sym] = "short"
        # symbols absent from this bar's group must not leak stale state
        # into a future bar where they reappear with unrelated ranking
        state = new_state

        weight = np.zeros(n)
        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        long_idx = np.array(
            [sym_to_idx[s] for s in symbols if new_state.get(s) == "long"], dtype=int
        )
        short_idx = np.array(
            [sym_to_idx[s] for s in symbols if new_state.get(s) == "short"], dtype=int
        )
        if len(long_idx) and size[long_idx].sum() > 0:
            weight[long_idx] = (
                (gross_exposure / 2) * size[long_idx] / size[long_idx].sum()
            )
        if len(short_idx) and size[short_idx].sum() > 0:
            weight[short_idx] = (
                -(gross_exposure / 2) * size[short_idx] / size[short_idx].sum()
            )
        weight = np.clip(weight, -max_position_per_symbol, max_position_per_symbol)

        for sym, w in zip(symbols, weight, strict=True):
            rows.append({datetime_col: key[0], symbol_col: sym, "weight": float(w)})

    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                datetime_col: df[datetime_col].dtype,
                symbol_col: pl.Utf8,
                "weight": pl.Float64,
            }
        )
    )


def quantize_weights(
    weights: pl.DataFrame, grid: float, weight_col: str = "weight"
) -> pl.DataFrame:
    """Round weight_col to the nearest multiple of `grid`, so a marginal
    rebalance smaller than the grid never triggers a trade at all.

    Rounding happens on the already-computed dollar-neutral weights (does
    not re-derive gross exposure or re-normalize legs), so the two legs can
    drift slightly off exact +-gross/2 after quantization - reported, not
    hidden: `abs(weights.sum())` should stay small (a few multiples of
    `grid` at most, not O(gross_exposure)), and this is checked explicitly
    wherever quantized weights are backtested (see tripwire in
    run_phase_a_turnover.py).
    """
    if grid <= 0:
        raise ValueError(f"grid must be > 0, got {grid}")
    return weights.with_columns(
        (pl.col(weight_col) / grid).round(0).mul(grid).alias(weight_col)
    )


def throttle_weights(
    weights: pl.DataFrame,
    k: int,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Rebalance only every k-th distinct timestamp in `weights`, holding
    each symbol's previous weight on the bars in between. k=1 is a no-op
    (rebalance every bar, identical to the input).

    A symbol whose very first appearance in the panel falls on a
    non-rebalance bar has no prior weight to hold and is treated as flat
    (0.0) until its first rebalance bar - a deliberate, stated convention
    for newly-listed symbols, not an attempt to backfill a position it
    never actually held.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k == 1:
        return weights

    times = weights[datetime_col].unique(maintain_order=False).sort()
    time_idx = pl.DataFrame(
        {datetime_col: times, "_tidx": np.arange(len(times))}
    )
    w = weights.join(time_idx, on=datetime_col, how="left").sort(
        [symbol_col, datetime_col]
    )
    is_rebalance = (pl.col("_tidx") % k) == 0
    w = w.with_columns(
        pl.when(is_rebalance).then(pl.col(weight_col)).otherwise(None).alias(weight_col)
    )
    w = w.with_columns(pl.col(weight_col).forward_fill().over(symbol_col))
    w = w.with_columns(pl.col(weight_col).fill_null(0.0))
    return w.drop("_tidx").sort([datetime_col, symbol_col])


def var_gate_standdown(
    var_forecast: pl.DataFrame,
    k: float,
    trailing_window: int = 250,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    var_col: str = "var_forecast",
) -> pl.DataFrame:
    """B1 "stand-down": cross-sectional median predicted 1% VaR magnitude,
    per bar, compared to its own trailing median (causal, shifted so bar t's
    gate never uses bar t's own value in its own reference level, same
    convention as run_phase6_application.py's `build_overlay_weight`).
    Returns [datetime_col, "book_scale"] with book_scale in {0.0, 1.0}: 0
    when the cross-sectional median |VaR| exceeds `k` times its own trailing
    median, else 1 - the whole book stands down together, not per-symbol.
    """
    per_bar = (
        var_forecast.select(datetime_col, symbol_col, var_col)
        .with_columns(pl.col(var_col).abs().alias("_abs_var"))
        .group_by(datetime_col)
        .agg(pl.col("_abs_var").median().alias("median_abs_var"))
        .sort(datetime_col)
    )
    trailing = (
        per_bar["median_abs_var"]
        .rolling_median(window_size=trailing_window, min_periods=trailing_window // 2)
        .shift(1)
    )
    per_bar = per_bar.with_columns(trailing.alias("trailing_median"))
    per_bar = per_bar.with_columns(
        pl.when(
            pl.col("trailing_median").is_not_null()
            & (pl.col("median_abs_var") > k * pl.col("trailing_median"))
        )
        .then(0.0)
        .otherwise(1.0)
        .alias("book_scale")
    )
    return per_bar.select(datetime_col, "book_scale")


def var_gate_tilt(
    var_forecast: pl.DataFrame,
    trailing_window: int = 250,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    var_col: str = "var_forecast",
) -> pl.DataFrame:
    """B2 "per-symbol tilt": reuses run_phase6_application.py's own
    `build_overlay_weight` logic per-symbol (full weight when |VaR_t| is
    within that symbol's own trailing median, shrunk proportionally when it
    exceeds it) - applied BEFORE dollar-neutralizing, so calm symbols get
    relatively more capital than turbulent ones within each leg. Returns
    [datetime_col, symbol_col, "tilt"] in [0, 1].
    """
    sys.path.insert(0, "src/research/tmp")
    from run_phase6_application import build_overlay_weight  # noqa: E402

    out = []
    for sym, g in var_forecast.sort(datetime_col).group_by(symbol_col, maintain_order=True):
        g = g.sort(datetime_col)
        tilt = build_overlay_weight(g[var_col].to_numpy())
        out.append(g.select(datetime_col, symbol_col).with_columns(pl.Series("tilt", tilt)))
    return pl.concat(out).sort([datetime_col, symbol_col])


def apply_book_scale(weights: pl.DataFrame, book_scale: pl.DataFrame) -> pl.DataFrame:
    """Multiply every symbol's weight that bar by the (bar-level) book_scale
    from `var_gate_standdown`."""
    return (
        weights.join(book_scale, on="datetime", how="left")
        .with_columns(pl.col("book_scale").fill_null(1.0))
        .with_columns((pl.col("weight") * pl.col("book_scale")).alias("weight"))
        .drop("book_scale")
    )


def forward_fill_shape_path(fits: list[dict], n: int, shape_idx: int) -> np.ndarray:
    """Forward-filled per-bar path of one component of a rolling zoo fit's
    `shape` tuple (e.g. Hansen skew-t's nu at shape_idx=0, lambda at
    shape_idx=1), using the exact same per-refit-segment convention as
    `dist_lib6.zoo_quantile_forecast` - each refit's shape value holds from
    its own `t` until the next refit's `t` (or the end of the series).
    Causal by construction: fits[i]["shape"] was only ever estimated from
    data strictly before fits[i]["t"] pushed forward, same guarantee
    dist_lib6.rolling_garch_forecast_zoo already provides.
    """
    out = np.full(n, np.nan)
    if not fits:
        return out
    for i, f in enumerate(fits):
        start = f["t"]
        end = fits[i + 1]["t"] if i + 1 < len(fits) else n
        out[start:end] = f["shape"][shape_idx]
    return out


def zoo_es_forecast(
    variance_forecast: np.ndarray, fits: list[dict], family_module, q: float
) -> np.ndarray:
    """sigma_t * family_module.es(q, shape_t): the actual-return-scale
    expected-shortfall analogue of dist_lib6.zoo_quantile_forecast (which
    does the same thing for the VaR/ppf). Not in dist_lib6.py itself because
    notebook 6 never needed a standalone ES *forecast* path (only
    Acerbi-Szekely's realized-vs-predicted ES test, which calls
    family_module.es directly per-refit) - added here as the direct analogue
    for Phase D's D3 "rank on predicted ES" factor.
    """
    n = len(variance_forecast)
    out = np.full(n, np.nan)
    if not fits:
        return out
    for i, f in enumerate(fits):
        start = f["t"]
        end = fits[i + 1]["t"] if i + 1 < len(fits) else n
        v = variance_forecast[start:end]
        mask = np.isfinite(v) & (v > 0)
        sigma = np.sqrt(v[mask])
        es_z = family_module.es(q, f["shape"])
        idx = np.arange(start, end)[mask]
        out[idx] = sigma * es_z
    return out


def apply_tilt(panel: pl.DataFrame, tilt: pl.DataFrame, pred_col: str) -> pl.DataFrame:
    """Shrink pred_col's SIZE input (not the ranking itself) by tilt,
    BEFORE `dollar_neutral_weights`/`hysteresis_weights` are called - the
    caller is responsible for passing the resulting `size_col` to those
    functions, per B2's own spec (shrink weight, not the rank)."""
    return panel.join(tilt, on=["datetime", "symbol"], how="left").with_columns(
        pl.col("tilt").fill_null(1.0)
    )
