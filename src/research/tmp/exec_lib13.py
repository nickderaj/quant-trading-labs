"""Notebook-13-local machinery: execution-first sizing (Design A), causal
monthly re-parameterization (Design C), and a causal correlation-graph
attention model (Design D), on top of `research`/`spread_lib11`/`alpha_lib7`.

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

from spread_lib11 import true_atr_series, sma_causal  # noqa: E402


# ---------------------------------------------------------------------------
# Design A -- trend/momentum state, sizing, exits, capacity
# ---------------------------------------------------------------------------


def trend_momentum_state(
    close: np.ndarray,
    fast: int = 32,
    slow: int = 96,
    roc_window: int = 48,
    smooth_span: int = 10,
    atr_window: int = 14,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> np.ndarray:
    """A single continuous regime state: ATR-normalized EMA(fast)-EMA(slow)
    log-price spread, plus the ATR-normalized rate-of-change over
    `roc_window`, averaged and then EWMA-smoothed over `smooth_span`.

    The smoothing is the mechanism under test (NEXT_PROMPT.md sec4.A) --
    a raw crossing signal whipsaws every bar it straddles the mean; the
    smoothed state changes sign only when the regime itself has actually
    turned. All internal series are shift(1) or built only from data
    through t-1, so `state[t]` never uses bar t's own close.
    """
    close = np.asarray(close, dtype=float)
    high = close if high is None else np.asarray(high, dtype=float)
    low = close if low is None else np.asarray(low, dtype=float)
    log_close = np.log(close)

    log_s = pl.Series(log_close)
    ema_fast = log_s.ewm_mean(span=fast).shift(1).to_numpy()
    ema_slow = log_s.ewm_mean(span=slow).shift(1).to_numpy()
    atr = true_atr_series(high, low, close, window=atr_window)
    atr_frac = np.where(atr > 1e-12, atr / np.where(close > 1e-12, close, np.nan), np.nan)

    trend_term = (ema_fast - ema_slow) / np.where(atr_frac > 1e-12, atr_frac, np.nan)

    roc = pl.Series(log_close).diff(roc_window).shift(1).to_numpy()
    momentum_term = roc / np.where(atr_frac > 1e-12, atr_frac, np.nan)

    raw_state = np.nanmean(np.vstack([trend_term, momentum_term]), axis=0)
    raw_state = np.where(np.isfinite(raw_state), raw_state, 0.0)
    state = pl.Series(raw_state).ewm_mean(span=smooth_span).to_numpy().copy()
    # state[t] is a function of raw_state[<=t], each already shift(1)'d
    # relative to price, so state itself carries no further lookahead.
    state[: max(fast, slow, roc_window)] = np.nan
    return state


def fractional_kelly_scalar(
    expected_edge: np.ndarray, vol: np.ndarray, kelly_fraction: float = 0.25
) -> np.ndarray:
    """`kelly_fraction * edge / vol**2`, clipped to +-1 -- the classic
    continuous-time Kelly fraction (edge/variance) shrunk by
    `kelly_fraction` for estimation error, as Design A specifies. `edge`
    and `vol` must already be trailing/causal quantities.
    """
    edge = np.asarray(expected_edge, dtype=float)
    v = np.asarray(vol, dtype=float)
    raw = kelly_fraction * edge / np.where(v > 1e-12, v**2, np.nan)
    return np.clip(np.nan_to_num(raw, nan=0.0), -1.0, 1.0)


def sqrt_impact_discount(
    intended_notional: np.ndarray,
    adv_notional: np.ndarray,
    impact_coefficient: float = 0.1,
) -> np.ndarray:
    """Square-root market-impact discount factor in [0, 1]:
    `1 / (1 + impact_coefficient * sqrt(|intended_notional| / adv_notional))`.
    Multiplies the pre-impact position size; larger intended trades relative
    to average daily (dollar) volume are discounted harder. `adv_notional`
    must be a trailing (causal) average.
    """
    notional = np.abs(np.asarray(intended_notional, dtype=float))
    adv = np.asarray(adv_notional, dtype=float)
    ratio = np.where(adv > 1e-12, notional / adv, np.inf)
    return 1.0 / (1.0 + impact_coefficient * np.sqrt(np.maximum(ratio, 0.0)))


def capacity_curve(
    base_notional: float,
    adv_notional: np.ndarray,
    base_sharpe: float,
    impact_coefficient: float = 0.1,
) -> dict[str, float]:
    """AUM at which the impact discount alone would degrade net Sharpe by
    25% and 50% relative to `base_sharpe`, holding position sizing rules
    fixed and scaling AUM (and therefore intended notional) up uniformly.
    Pure algebra on `sqrt_impact_discount`: Sharpe scales linearly with the
    discount factor since impact only shrinks realized position size, not
    its sign or timing.
    """
    adv = np.asarray(adv_notional, dtype=float)
    median_adv = float(np.nanmedian(adv))

    def sharpe_at_scale(scale: float) -> float:
        notional = base_notional * scale
        discount = sqrt_impact_discount(
            np.array([notional]), np.array([median_adv]), impact_coefficient
        )[0]
        return base_sharpe * discount

    scales = np.geomspace(1e-3, 1e6, 4000)
    sharpes = np.array([sharpe_at_scale(s) for s in scales])
    aum_at_scale = scales * base_notional

    def aum_for_degradation(frac: float) -> float | None:
        target = base_sharpe * (1 - frac)
        idx = np.where(sharpes <= target)[0]
        return float(aum_at_scale[idx[0]]) if len(idx) else None

    return {
        "aum_25pct_degradation": aum_for_degradation(0.25),
        "aum_50pct_degradation": aum_for_degradation(0.50),
    }


# ---------------------------------------------------------------------------
# Stop-fill convention -- shared by Designs A and C (NEXT_PROMPT.md sec7 trap1)
# ---------------------------------------------------------------------------


def trailing_atr_stop(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    position: np.ndarray,
    atr_mult: float = 3.0,
    atr_window: int = 14,
) -> np.ndarray:
    """Wilder-ATR trailing stop level, `S_t = max(S_{t-1}, P_t - alpha*ATR_t)`
    while long and mirrored while short, reset whenever `position` changes
    sign or goes flat. Returns the stop level series (NaN while flat).
    """
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    position = np.asarray(position, dtype=float)
    atr = true_atr_series(high, low, close, window=atr_window)

    stop = np.full(len(close), np.nan)
    prev_stop = np.nan
    prev_pos_sign = 0
    for t in range(len(close)):
        pos_sign = np.sign(position[t])
        if pos_sign == 0 or not np.isfinite(atr[t]):
            stop[t] = np.nan
            prev_stop = np.nan
            prev_pos_sign = 0
            continue
        if pos_sign != prev_pos_sign:
            prev_stop = np.nan
        if pos_sign > 0:
            candidate = close[t] - atr_mult * atr[t]
            prev_stop = candidate if not np.isfinite(prev_stop) else max(prev_stop, candidate)
        else:
            candidate = close[t] + atr_mult * atr[t]
            prev_stop = candidate if not np.isfinite(prev_stop) else min(prev_stop, candidate)
        stop[t] = prev_stop
        prev_pos_sign = pos_sign
    return stop


def apply_stop_fill(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    stop_level: np.ndarray,
    position: np.ndarray,
    optimistic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Required stop-fill convention (NEXT_PROMPT.md sec7 trap1): a long stop
    is hit if `low[t] <= stop_level[t]`; a short stop is hit if
    `high[t] >= stop_level[t]`. The fill price is the stop level UNLESS the
    bar's open already gapped through it, in which case the fill is the
    (worse) open price. `optimistic=True` reproduces the naive
    always-fill-at-stop-price convention, for the mandatory sensitivity row
    only -- never the headline.

    Returns `(exit_mask, fill_price)`: `exit_mask[t]` True where the stop
    fired on bar t, `fill_price[t]` the resulting exit price there.
    """
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    stop_level = np.asarray(stop_level, dtype=float)
    position = np.asarray(position, dtype=float)

    pos_sign = np.sign(position)
    long_hit = (pos_sign > 0) & np.isfinite(stop_level) & (low <= stop_level)
    short_hit = (pos_sign < 0) & np.isfinite(stop_level) & (high >= stop_level)
    exit_mask = long_hit | short_hit

    fill_price = np.full(len(open_), np.nan)
    if optimistic:
        fill_price = np.where(exit_mask, stop_level, fill_price)
        return exit_mask, fill_price

    long_gap = long_hit & (open_ <= stop_level)
    short_gap = short_hit & (open_ >= stop_level)
    fill_price = np.where(long_hit & ~long_gap, stop_level, fill_price)
    fill_price = np.where(short_hit & ~short_gap, stop_level, fill_price)
    fill_price = np.where(long_gap | short_gap, open_, fill_price)
    return exit_mask, fill_price


# ---------------------------------------------------------------------------
# Design C -- causal monthly grid search, quality/liquidity selection
# ---------------------------------------------------------------------------


def causal_monthly_regrid_search(
    dates: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    grid: dict[str, list[float]],
    score_fn,
) -> pl.DataFrame:
    """For each calendar month boundary after the first, fit `(L, theta,
    alpha)` on the PRECEDING calendar month only (grid search maximizing
    `score_fn`), then apply that fit forward through the NEXT calendar
    month. Returns one row per (fit_month, applied_month, L, theta, alpha).

    NEXT_PROMPT.md sec7 trap2: the panel passed in is sliced to
    `dates <= month_end` before any indicator is computed inside `score_fn`
    -- `score_fn` receives only that slice, never the full series, so no
    future bar can leak into a fitted parameter. This is asserted by
    `tests/test_exec_lib13.py::test_monthly_regrid_no_future_leakage`.
    """
    dates = np.asarray(dates)
    months = np.array([str(d)[:7] for d in dates])
    unique_months = sorted(set(months.tolist()))

    rows: list[dict[str, Any]] = []
    for i in range(1, len(unique_months)):
        fit_month = unique_months[i - 1]
        applied_month = unique_months[i]
        fit_mask = months == fit_month
        fit_idx = np.where(fit_mask)[0]
        if len(fit_idx) < 5:
            continue
        fit_slice = slice(fit_idx[0], fit_idx[-1] + 1)
        c_fit, h_fit, l_fit = close[fit_slice], high[fit_slice], low[fit_slice]

        best = None
        for L in grid["L"]:
            for theta in grid["theta"]:
                for alpha in grid["alpha"]:
                    score = score_fn(c_fit, h_fit, l_fit, L, theta, alpha)
                    if best is None or score > best[0]:
                        best = (score, L, theta, alpha)

        rows.append(
            {
                "fit_month": fit_month,
                "applied_month": applied_month,
                "score": best[0],
                "L": best[1],
                "theta": best[2],
                "alpha": best[3],
            }
        )
    return pl.DataFrame(rows)


def rate_of_change_entry(close: np.ndarray, lookback: int, theta: float) -> np.ndarray:
    """+1 / -1 / 0 entry signal: rate of change over `lookback` bars vs
    +-theta, shift(1) so signal[t] never uses close[t]."""
    close = np.asarray(close, dtype=float)
    roc = pl.Series(np.log(close)).diff(lookback).shift(1).to_numpy()
    sig = np.zeros(len(close))
    sig[roc > theta] = 1.0
    sig[roc < -theta] = -1.0
    return sig


def quality_liquidity_selection(
    trailing_sharpe: np.ndarray,
    trailing_dollar_volume: np.ndarray,
    long_quality_threshold: float,
    short_quality_threshold: float,
    liquidity_rank_frac: float,
    leg: str,
) -> np.ndarray:
    """Boolean eligibility mask for one leg ("long" or "short"): trailing
    one-month realized Sharpe must clear `long_quality_threshold` (longs) or
    `short_quality_threshold` (shorts, expected higher bar), AND trailing
    dollar volume must rank in the top `liquidity_rank_frac` of the
    cross-section that bar -- the market-cap substitute (NEXT_PROMPT.md
    sec4.C), disclosed here as a proxy, not silently swapped in.
    """
    sharpe = np.asarray(trailing_sharpe, dtype=float)
    dvol = np.asarray(trailing_dollar_volume, dtype=float)
    threshold = long_quality_threshold if leg == "long" else short_quality_threshold
    quality_ok = sharpe >= threshold if leg == "long" else sharpe <= -threshold
    rank = pl.Series(dvol).rank(method="average", descending=True) / len(dvol)
    liquidity_ok = (rank <= liquidity_rank_frac).to_numpy()
    return quality_ok & liquidity_ok & np.isfinite(sharpe) & np.isfinite(dvol)


# ---------------------------------------------------------------------------
# Design D -- causal correlation graph + graph attention model
# ---------------------------------------------------------------------------


def rolling_causal_corr_graph(
    returns: pl.DataFrame,
    lookback: int = 30,
    threshold: float = 0.3,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    return_col: str = "ret",
) -> dict[Any, np.ndarray]:
    """At every date, the adjacency is Pearson correlation over the
    TRAILING `lookback` bars up to and including t-1 (never t itself),
    thresholded at `|corr| >= threshold`. Returns `{date: adjacency_matrix}`
    with a fixed, alphabetically sorted symbol order recorded alongside.

    NEXT_PROMPT.md sec7 trap3: the graph is never built once over the full
    sample -- each date's adjacency is computed from a data slice that ends
    strictly before that date, asserted by
    `tests/test_exec_lib13.py::test_corr_graph_no_future_leakage`.
    """
    wide = returns.pivot(
        index=datetime_col, on=symbol_col, values=return_col
    ).sort(datetime_col)
    symbols = sorted(c for c in wide.columns if c != datetime_col)
    dates = wide[datetime_col].to_numpy()
    mat = wide.select(symbols).to_numpy()

    graphs: dict[Any, np.ndarray] = {"__symbols__": np.array(symbols)}
    for t in range(lookback, len(dates)):
        window = mat[t - lookback : t]  # strictly bars < t
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(window, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        adj = (np.abs(corr) >= threshold).astype(float)
        np.fill_diagonal(adj, 0.0)
        graphs[dates[t]] = adj
    return graphs


def node_degree_stats(graphs: dict[Any, np.ndarray]) -> dict[str, float]:
    """Mean/median node degree across all rebalance-date adjacency matrices
    in a `rolling_causal_corr_graph` output (NEXT_PROMPT.md sec4.D: report
    node degree so a sparser 30-symbol graph is judged on its own terms)."""
    degrees = []
    for key, adj in graphs.items():
        if key == "__symbols__":
            continue
        degrees.append(adj.sum(axis=1))
    if not degrees:
        return {"mean_degree": float("nan"), "median_degree": float("nan")}
    all_deg = np.concatenate(degrees)
    return {"mean_degree": float(np.mean(all_deg)), "median_degree": float(np.median(all_deg))}


try:
    import torch
    import torch.nn as nn

    class GraphAttentionPredictor(nn.Module):
        """Single-head additive graph attention over each node's neighbours
        (as given by a fixed adjacency mask per forward call), predicting
        next-bar return per node from that node's own current feature
        vector plus an attention-weighted aggregate of its neighbours'.

        `use_time_mixing=True` adds a small GRU branch over each node's
        trailing feature history before the attention step -- Design D's
        actual thesis is that this branch HURTS crypto IC
        (NEXT_PROMPT.md sec4.D); both variants share every other weight
        shape so the ablation isolates exactly that branch.
        """

        def __init__(
            self,
            n_features: int,
            hidden_dim: int = 16,
            use_time_mixing: bool = False,
            time_window: int = 5,
        ):
            super().__init__()
            self.use_time_mixing = use_time_mixing
            self.time_window = time_window
            in_dim = n_features
            if use_time_mixing:
                self.gru = nn.GRU(n_features, hidden_dim, batch_first=True)
                in_dim = hidden_dim
            self.node_proj = nn.Linear(in_dim, hidden_dim)
            self.attn_query = nn.Linear(hidden_dim, hidden_dim)
            self.attn_key = nn.Linear(hidden_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim * 2, 1)

        def encode_nodes(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (n_nodes, time_window, n_features) if use_time_mixing else
            # (n_nodes, n_features)
            if self.use_time_mixing:
                _, h_n = self.gru(x)
                x = h_n.squeeze(0)
            return torch.relu(self.node_proj(x))

        def forward(self, x: "torch.Tensor", adjacency: "torch.Tensor") -> "torch.Tensor":
            h = self.encode_nodes(x)  # (n_nodes, hidden_dim)
            q = self.attn_query(h)
            k = self.attn_key(h)
            scores = q @ k.T / (h.shape[-1] ** 0.5)
            mask = adjacency <= 0
            scores = scores.masked_fill(mask, float("-inf"))
            has_neighbour = (~mask).any(dim=1)
            weights = torch.softmax(scores, dim=1)
            weights = torch.nan_to_num(weights, nan=0.0)
            neighbour_agg = weights @ h
            neighbour_agg = torch.where(has_neighbour.unsqueeze(1), neighbour_agg, torch.zeros_like(h))
            combined = torch.cat([h, neighbour_agg], dim=1)
            return self.out(combined).squeeze(-1)

    class LSTMForecaster(nn.Module):
        """Plain LSTM over lagged per-instrument features, one scalar
        vol-normalized-return-forecast output per instrument. Design B's
        model 2 (NEXT_PROMPT.md sec4.B)."""

        def __init__(self, n_features: int, hidden_dim: int = 32, n_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=n_layers, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    class GatedLSTMForecaster(nn.Module):
        """LSTM with a learned per-feature gating layer applied before the
        recurrent stack, so the model chooses its own inputs per instrument
        -- Design B's model 3, its specific architectural claim
        (NEXT_PROMPT.md sec4.B)."""

        def __init__(self, n_features: int, hidden_dim: int = 32, n_layers: int = 1):
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(n_features, n_features), nn.Sigmoid())
            self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=n_layers, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            gated = x * self.gate(x)
            out, _ = self.lstm(gated)
            return self.head(out[:, -1, :]).squeeze(-1)

    def negative_sharpe_loss(
        position: "torch.Tensor", forward_return: "torch.Tensor", eps: float = 1e-6
    ) -> "torch.Tensor":
        """Design B's central methodological claim (NEXT_PROMPT.md sec4.B):
        train on negative realized Sharpe of the induced position rather
        than MSE. `pnl = position * forward_return`; loss is
        `-mean(pnl) / (std(pnl) + eps)` over the batch, so the model is
        rewarded for a high risk-adjusted book, not a low squared error."""
        pnl = position * forward_return
        return -pnl.mean() / (pnl.std(unbiased=False) + eps)

except ImportError:  # torch optional at import time for lighter test runs
    GraphAttentionPredictor = None
    LSTMForecaster = None
    GatedLSTMForecaster = None
    negative_sharpe_loss = None


# ---------------------------------------------------------------------------
# Shared -- breakeven cost sweep (Design B, adopted permanently per sec4.B)
# ---------------------------------------------------------------------------


def breakeven_cost_bps(
    gross_returns: np.ndarray,
    turnover: np.ndarray,
    cost_grid_bps: np.ndarray | None = None,
) -> float | None:
    """The per-side cost (in bps of notional traded) at which
    `mean(gross_returns - cost*turnover) / std(...)` (annualization cancels
    in the zero-crossing) first reaches zero. `turnover` is the per-bar
    fraction of notional traded (same convention as
    `research.portfolio_turnover`). Returns None if net Sharpe is already
    <=0 at zero cost, or never crosses zero within `cost_grid_bps`.
    """
    gross = np.asarray(gross_returns, dtype=float)
    turn = np.asarray(turnover, dtype=float)
    if cost_grid_bps is None:
        cost_grid_bps = np.linspace(0, 200, 401)

    def sharpe_at_cost(cost_bps: float) -> float:
        net = gross - (cost_bps / 1e4) * turn
        sd = np.nanstd(net)
        return float(np.nanmean(net) / sd) if sd > 1e-12 else float("nan")

    base = sharpe_at_cost(0.0)
    if not np.isfinite(base) or base <= 0:
        return None
    for cost_bps in cost_grid_bps:
        if sharpe_at_cost(float(cost_bps)) <= 0:
            return float(cost_bps)
    return None
