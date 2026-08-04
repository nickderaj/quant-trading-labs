import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "research" / "tmp")
)

from exec_lib13 import (
    apply_stop_fill,
    breakeven_cost_bps,
    capacity_curve,
    causal_monthly_regrid_search,
    fractional_kelly_scalar,
    node_degree_stats,
    quality_liquidity_selection,
    rate_of_change_entry,
    rolling_causal_corr_graph,
    sqrt_impact_discount,
    trailing_atr_stop,
    trend_momentum_state,
)


def _synthetic_ohlc(n=300, seed=0, trend=0.0005):
    rng = np.random.default_rng(seed)
    ret = rng.normal(trend, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return open_, high, low, close


def test_trend_momentum_state_no_lookahead():
    open_, high, low, close = _synthetic_ohlc()
    state_full = trend_momentum_state(close, high=high, low=low)
    # perturb only the last bar; every earlier state value must be identical.
    close_pert = close.copy()
    close_pert[-1] *= 1.5
    high_pert, low_pert = high.copy(), low.copy()
    high_pert[-1] *= 1.5
    low_pert[-1] *= 1.5
    state_pert = trend_momentum_state(close_pert, high=high_pert, low=low_pert)
    np.testing.assert_allclose(state_full[:-1], state_pert[:-1], equal_nan=True)


def test_fractional_kelly_scalar_bounds_and_sign():
    edge = np.array([0.02, -0.02, 0.0])
    vol = np.array([0.01, 0.01, 0.01])
    out = fractional_kelly_scalar(edge, vol, kelly_fraction=0.25)
    assert out[0] > 0
    assert out[1] < 0
    assert out[2] == 0
    assert np.all(np.abs(out) <= 1.0)


def test_sqrt_impact_discount_monotonic_in_size():
    adv = np.full(5, 1_000_000.0)
    notional = np.array([1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    discount = sqrt_impact_discount(notional, adv)
    assert np.all(np.diff(discount) < 0)
    assert np.all((discount > 0) & (discount <= 1))


def test_capacity_curve_degrades_with_aum():
    adv = np.full(50, 5_000_000.0)
    out = capacity_curve(base_notional=100_000, adv_notional=adv, base_sharpe=2.0)
    assert out["aum_25pct_degradation"] is not None
    assert out["aum_50pct_degradation"] is not None
    assert out["aum_50pct_degradation"] > out["aum_25pct_degradation"]


def test_trailing_atr_stop_only_tightens_favourably():
    open_, high, low, close = _synthetic_ohlc(trend=0.002)
    position = np.ones(len(close))
    stop = trailing_atr_stop(close, high, low, position, atr_mult=2.0)
    valid = np.isfinite(stop)
    diffs = np.diff(stop[valid])
    # a long trailing stop must never move down
    assert np.all(diffs >= -1e-9)


def test_apply_stop_fill_gap_uses_worse_price():
    # long position, bar gaps open below the stop level -> fill at open, not stop.
    open_ = np.array([90.0])
    high = np.array([91.0])
    low = np.array([89.0])
    stop_level = np.array([95.0])
    position = np.array([1.0])
    exit_mask, fill = apply_stop_fill(open_, high, low, stop_level, position)
    assert exit_mask[0]
    assert fill[0] == 90.0  # worse than the stop price of 95


def test_apply_stop_fill_no_gap_uses_stop_price():
    open_ = np.array([100.0])
    high = np.array([101.0])
    low = np.array([94.0])
    stop_level = np.array([95.0])
    position = np.array([1.0])
    exit_mask, fill = apply_stop_fill(open_, high, low, stop_level, position)
    assert exit_mask[0]
    assert fill[0] == 95.0


def test_apply_stop_fill_optimistic_convention_differs():
    open_ = np.array([90.0])
    high = np.array([91.0])
    low = np.array([89.0])
    stop_level = np.array([95.0])
    position = np.array([1.0])
    _, fill_required = apply_stop_fill(open_, high, low, stop_level, position, optimistic=False)
    _, fill_optimistic = apply_stop_fill(open_, high, low, stop_level, position, optimistic=True)
    assert fill_required[0] < fill_optimistic[0]


def test_rate_of_change_entry_shift_one():
    close = np.array([100.0] * 10 + [200.0])
    sig = rate_of_change_entry(close, lookback=1, theta=0.01)
    # the +100% jump happens on the last bar; because of shift(1) the
    # resulting long signal can only appear STARTING the bar after that.
    assert sig[-1] == 0.0 or np.isnan(sig[-1]) is False


def test_quality_liquidity_selection_long_short_asymmetry():
    sharpe = np.array([0.5, 0.5, -0.5, -0.5])
    dvol = np.array([100.0, 1.0, 100.0, 1.0])
    long_ok = quality_liquidity_selection(sharpe, dvol, 0.3, 0.3, 0.5, "long")
    short_ok = quality_liquidity_selection(sharpe, dvol, 0.3, 0.3, 0.5, "short")
    assert long_ok[0] and not long_ok[1]
    assert short_ok[2] and not short_ok[3]
    assert not long_ok[2] and not short_ok[0]


def test_monthly_regrid_no_future_leakage():
    dates = np.array(
        [np.datetime64("2022-01-01") + np.timedelta64(i, "D") for i in range(90)]
    )
    open_, high, low, close = _synthetic_ohlc(n=90, seed=1)

    def score_fn(c, h, l, L, theta, alpha):
        return float(np.nanmean(np.diff(np.log(c))))

    grid = {"L": [3, 5], "theta": [0.001, 0.01], "alpha": [1.0, 2.0]}
    fits_full = causal_monthly_regrid_search(dates, close, high, low, grid, score_fn)

    # perturb bars strictly AFTER the first fit month's boundary; the first
    # fitted row (fit_month == first month) must not change.
    close_pert = close.copy()
    boundary = np.where(np.array([str(d)[:7] for d in dates]) == "2022-02")[0][0]
    close_pert[boundary:] *= 1.7
    fits_pert = causal_monthly_regrid_search(dates, close_pert, high, low, grid, score_fn)

    first_full = fits_full.filter(pl.col("fit_month") == "2022-01").row(0, named=True)
    first_pert = fits_pert.filter(pl.col("fit_month") == "2022-01").row(0, named=True)
    assert first_full["L"] == first_pert["L"]
    assert first_full["theta"] == first_pert["theta"]
    assert first_full["alpha"] == first_pert["alpha"]
    assert first_full["score"] == first_pert["score"]


def _synthetic_returns_panel(n_times=60, symbols=("A", "B", "C", "D"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    base = datetime(2022, 1, 1)
    for t in range(n_times):
        dt = base + timedelta(days=t)
        for s in symbols:
            rows.append({"datetime": dt, "symbol": s, "ret": float(rng.normal(0, 0.02))})
    return pl.DataFrame(rows)


def test_corr_graph_no_future_leakage():
    panel = _synthetic_returns_panel()
    graphs_full = rolling_causal_corr_graph(panel, lookback=10, threshold=0.3)

    dates = sorted(d for d in graphs_full if d != "__symbols__")
    perturb_from = dates[20]

    rng = np.random.default_rng(99)
    noise = pl.Series(rng.normal(0, 0.1, panel.height))
    panel_pert = panel.with_columns(
        pl.when(pl.col("datetime") >= perturb_from)
        .then(pl.col("ret") + noise)
        .otherwise(pl.col("ret"))
        .alias("ret")
    )
    graphs_pert = rolling_causal_corr_graph(panel_pert, lookback=10, threshold=0.3)

    # the adjacency AT perturb_from is built from bars strictly before it,
    # none of which were touched -- it must be identical.
    np.testing.assert_allclose(graphs_full[perturb_from], graphs_pert[perturb_from])
    # an adjacency well AFTER the perturbation is free to differ (sanity
    # check that the perturbation actually does something, so the assertion
    # above isn't vacuously true).
    later = dates[40]
    assert not np.allclose(graphs_full[later], graphs_pert[later])


def test_node_degree_stats_reasonable():
    panel = _synthetic_returns_panel(n_times=80)
    graphs = rolling_causal_corr_graph(panel, lookback=10, threshold=0.1)
    stats = node_degree_stats(graphs)
    assert 0 <= stats["mean_degree"] <= 3  # 3 possible neighbours per node
    assert 0 <= stats["median_degree"] <= 3


def test_breakeven_cost_bps_zero_at_zero_edge():
    rng = np.random.default_rng(0)
    gross = rng.normal(0, 0.01, 500)  # no edge
    turnover = np.full(500, 0.5)
    out = breakeven_cost_bps(gross, turnover)
    assert out is None or out < 5  # near-zero edge breaks even almost immediately


def test_breakeven_cost_bps_positive_for_real_edge():
    rng = np.random.default_rng(0)
    gross = rng.normal(0.002, 0.01, 2000)
    turnover = np.full(2000, 0.2)
    out = breakeven_cost_bps(gross, turnover)
    assert out is not None
    assert out > 0


def test_graph_attention_time_mixing_shapes():
    torch = __import__("torch")
    from exec_lib13 import GraphAttentionPredictor

    n_nodes, n_features, window = 6, 4, 5
    adjacency = (torch.rand(n_nodes, n_nodes) > 0.5).float()
    adjacency.fill_diagonal_(0)

    model_no_mix = GraphAttentionPredictor(n_features, use_time_mixing=False)
    x_no_mix = torch.randn(n_nodes, n_features)
    out_no_mix = model_no_mix(x_no_mix, adjacency)
    assert out_no_mix.shape == (n_nodes,)

    model_mix = GraphAttentionPredictor(n_features, use_time_mixing=True, time_window=window)
    x_mix = torch.randn(n_nodes, window, n_features)
    out_mix = model_mix(x_mix, adjacency)
    assert out_mix.shape == (n_nodes,)


def test_negative_sharpe_loss_rewards_aligned_position():
    torch = __import__("torch")
    from exec_lib13 import negative_sharpe_loss

    fwd_return = torch.tensor([0.01, -0.01, 0.02, -0.02])
    aligned = torch.tensor([1.0, -1.0, 1.0, -1.0])
    misaligned = -aligned
    loss_aligned = negative_sharpe_loss(aligned, fwd_return)
    loss_misaligned = negative_sharpe_loss(misaligned, fwd_return)
    assert loss_aligned < loss_misaligned
