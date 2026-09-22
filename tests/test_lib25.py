import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "research" / "tmp")
)
import lib25

# --------------------------------------------------------------------------- #
# Closed form
# --------------------------------------------------------------------------- #


def test_black76_put_call_parity():
    F, K, T, sigma, df = 70.0, 65.0, 1.5, 0.35, 0.97
    call = lib25.black76(F, K, T, sigma, df, "call")
    put = lib25.black76(F, K, T, sigma, df, "put")
    assert (call - put) == pytest.approx(df * (F - K), abs=1e-10)


def test_black76_vs_gbm_monte_carlo():
    F, K, T, sigma, df = 70.0, 70.0, 1.0, 0.3, 1.0
    closed = lib25.black76(F, K, T, sigma, df, "call")
    paths = lib25.simulate_gbm(F, sigma, T, n_steps=252, n_paths=200_000, seed=1)
    payoffs = np.maximum(paths[:, -1] - K, 0.0) * df
    mc_price = payoffs.mean()
    se = payoffs.std(ddof=1) / np.sqrt(len(payoffs))
    assert abs(mc_price - closed) < 3 * se


def test_black76_at_expiry_is_intrinsic():
    assert lib25.black76(75.0, 70.0, 0.0, 0.3, 1.0, "call") == pytest.approx(5.0)
    assert lib25.black76(65.0, 70.0, 0.0, 0.3, 1.0, "put") == pytest.approx(5.0)


def test_black76_greeks_delta_bounds():
    g_call = lib25.black76_greeks(70.0, 70.0, 1.0, 0.3, 0.98, "call")
    g_put = lib25.black76_greeks(70.0, 70.0, 1.0, 0.3, 0.98, "put")
    assert 0.0 < g_call["delta"] < 1.0
    assert -1.0 < g_put["delta"] < 0.0
    assert g_call["gamma"] > 0.0
    assert g_call["vega"] > 0.0


def test_bachelier_admits_negative_underlying_and_strike():
    price = lib25.bachelier(-2.67, -10.0, 0.05, 8.0, 1.0, "put")
    assert np.isfinite(price)
    assert price >= 0.0


def test_bachelier_put_call_parity():
    F, K, T, sigma_n, df = 40.0, 45.0, 0.5, 10.0, 0.99
    call = lib25.bachelier(F, K, T, sigma_n, df, "call")
    put = lib25.bachelier(F, K, T, sigma_n, df, "put")
    assert (call - put) == pytest.approx(df * (F - K), abs=1e-8)


def test_displaced_black_converges_to_black76_as_shift_shrinks():
    F, K, T, sigma, df = 70.0, 68.0, 1.0, 0.3, 0.99
    black = lib25.black76(F, K, T, sigma, df, "call")
    displaced = lib25.displaced_black(F, K, T, sigma, df, "call", shift=1e-6)
    assert displaced == pytest.approx(black, rel=1e-3)


def test_displaced_black_converges_to_bachelier_for_large_shift():
    F, K, T, sigma, df = 70.0, 68.0, 1.0, 0.25, 0.99
    shift = 5000.0
    displaced = lib25.displaced_black(F, K, T, sigma, df, "call", shift=shift)
    sigma_n = sigma * (F + shift)
    normal = lib25.bachelier(F, K, T, sigma_n, df, "call")
    assert displaced == pytest.approx(normal, rel=0.05)


def test_implied_vol_round_trips_black76():
    F, K, T, df = 70.0, 72.0, 0.75, 0.98
    true_sigma = 0.28
    price = lib25.black76(F, K, T, true_sigma, df, "call")
    recovered = lib25.implied_vol(price, F, K, T, df, "call", model="black76")
    assert recovered == pytest.approx(true_sigma, abs=1e-8)


def test_implied_vol_round_trips_bachelier():
    F, K, T, df = 40.0, 38.0, 0.5, 0.98
    true_sigma_n = 9.0
    price = lib25.bachelier(F, K, T, true_sigma_n, df, "put")
    recovered = lib25.implied_vol(price, F, K, T, df, "put", model="bachelier")
    assert recovered == pytest.approx(true_sigma_n, abs=1e-6)


# --------------------------------------------------------------------------- #
# Path engines
# --------------------------------------------------------------------------- #


def test_simulate_gbm_is_a_martingale():
    F0 = 70.0
    paths = lib25.simulate_gbm(F0, 0.3, 1.0, n_steps=100, n_paths=100_000, seed=2)
    terminal = paths[:, -1]
    se = terminal.std(ddof=1) / np.sqrt(len(terminal))
    assert abs(terminal.mean() - F0) < 4 * se


def test_simulate_bachelier_is_a_martingale_and_goes_negative():
    F0 = 5.0
    paths = lib25.simulate_bachelier(F0, 8.0, 1.0, n_steps=100, n_paths=50_000, seed=3)
    terminal = paths[:, -1]
    se = terminal.std(ddof=1) / np.sqrt(len(terminal))
    assert abs(terminal.mean() - F0) < 4 * se
    assert (terminal < 0).any()


def test_simulate_correlated_recovers_target_correlation():
    rho = 0.6
    paths = lib25.simulate_correlated(
        [70.0, 30.0], [0.3, 0.35], rho, T=1.0, n_steps=100, n_paths=50_000, seed=4
    )
    r1 = np.diff(np.log(paths[0]), axis=1).ravel()
    r2 = np.diff(np.log(paths[1]), axis=1).ravel()
    corr = np.corrcoef(r1, r2)[0, 1]
    assert corr == pytest.approx(rho, abs=0.03)


def test_sobol_normals_shape_and_moments():
    z = lib25.sobol_normals(n_paths=1024, n_steps=4, seed=5)
    assert z.shape == (1024, 4)
    assert abs(z.mean()) < 0.05
    assert z.std() == pytest.approx(1.0, abs=0.1)


def test_bootstrap_paths_never_chains_across_contracts_and_is_causal():
    paths_early = lib25.bootstrap_paths(
        "CL", T=0.25, n_paths=30, block_days=15, seed=6, asof=pd.Timestamp("2015-01-02")
    )
    paths_late = lib25.bootstrap_paths(
        "CL", T=0.25, n_paths=30, block_days=15, seed=6, asof=pd.Timestamp("2019-01-02")
    )
    # Causal: the starting level is the front-month close as of `asof`, so
    # cutting off the panel earlier must not see later history's starting price.
    assert paths_early[0, 0] != paths_late[0, 0]
    assert np.all(np.isfinite(paths_early))
    assert np.all(np.isfinite(paths_late))


# --------------------------------------------------------------------------- #
# Structures
# --------------------------------------------------------------------------- #


def test_collar_premium_equals_put_minus_call():
    out = lib25.collar(70.0, 65.0, 75.0, 1.0, 0.3, 0.98)
    put_price = lib25.black76(70.0, 65.0, 1.0, 0.3, 0.98, "put")
    call_price = lib25.black76(70.0, 75.0, 1.0, 0.3, 0.98, "call")
    assert out["premium"] == pytest.approx(put_price - call_price)


def test_zero_cost_collar_has_zero_net_premium():
    out = lib25.zero_cost_collar(70.0, 60.0, 1.0, 0.35, 0.97)
    assert out["premium"] == pytest.approx(0.0, abs=1e-6)
    assert out["K_call"] > 70.0


def test_participating_forward_runs_and_is_finite():
    out = lib25.participating_forward(70.0, 70.0, 0.5, 1.0, 0.3, 0.98)
    assert np.isfinite(out["premium"])
    assert out["participation"] == 0.5


def test_price_structure_sums_legs():
    legs = [
        {"type": "put", "strike": 65.0, "qty": 1, "price": 3.0},
        {"type": "call", "strike": 75.0, "qty": -1, "price": 1.5},
    ]
    out = lib25.price_structure(legs, inputs={})
    assert out["premium"] == pytest.approx(1.5)


def test_realised_payoff_zero_for_knocked_out_put():
    structure = {"legs": [{"type": "put", "strike": 70.0, "qty": 1}]}
    path = np.array([70.0, 65.0, 55.0, 60.0])  # breaches a down-and-out barrier at 60
    schedule = {"barrier": {"level": 60.0, "kind": "do"}}
    assert lib25.realised_payoff(structure, path, schedule) == 0.0


def test_realised_payoff_vanilla_for_non_breached_path():
    structure = {"legs": [{"type": "put", "strike": 70.0, "qty": 1}]}
    path = np.array([70.0, 68.0, 65.0])  # never touches barrier at 55
    schedule = {"barrier": {"level": 55.0, "kind": "do"}}
    expected = max(70.0 - 65.0, 0.0)
    assert lib25.realised_payoff(structure, path, schedule) == pytest.approx(expected)


def test_realised_payoff_knock_in_only_pays_if_breached():
    structure = {"legs": [{"type": "put", "strike": 70.0, "qty": 1}]}
    path_hit = np.array([70.0, 60.0, 65.0])
    path_miss = np.array([70.0, 69.0, 65.0])
    schedule = {"barrier": {"level": 61.0, "kind": "di"}}
    assert lib25.realised_payoff(structure, path_hit, schedule) == pytest.approx(5.0)
    assert lib25.realised_payoff(structure, path_miss, schedule) == 0.0


def test_realised_payoff_uses_averaging_schedule():
    structure = {"legs": [{"type": "call", "strike": 10.0, "qty": 1}]}
    path = np.array([10.0, 11.0, 12.0, 13.0])
    schedule = {"avg_idx": np.array([1, 2, 3])}
    expected = max(np.mean([11.0, 12.0, 13.0]) - 10.0, 0.0)
    assert lib25.realised_payoff(structure, path, schedule) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_kupiec_pof_does_not_reject_correctly_calibrated_series():
    rng = np.random.default_rng(7)
    exceed = rng.random(2000) < 0.01
    out = lib25.kupiec_pof(exceed, alpha=0.01)
    assert out["n"] == 2000
    assert out["pvalue"] > 0.01


def test_kupiec_pof_rejects_badly_miscalibrated_series():
    exceed = np.zeros(500, dtype=bool)
    exceed[:100] = True  # 20% exceedance rate against a 1% target
    out = lib25.kupiec_pof(exceed, alpha=0.01)
    assert out["reject_5pct"] is True


def test_qlike_zero_when_forecast_matches_realised():
    rv = np.array([0.02, 0.05, 0.1])
    assert lib25.qlike(rv, rv) == pytest.approx(0.0, abs=1e-10)


def test_qlike_positive_when_forecast_wrong():
    forecast = np.array([0.01, 0.01, 0.01])
    realised = np.array([0.02, 0.05, 0.1])
    assert lib25.qlike(forecast, realised) > 0.0


# --------------------------------------------------------------------------- #
# Market inputs (real data)
# --------------------------------------------------------------------------- #


def test_forward_curve_sorted_and_positive_tau():
    curve = lib25.forward_curve("CL", pd.Timestamp("2017-06-04"))
    assert len(curve) > 5
    assert (curve["tau"] > 0).all()
    assert curve["tau"].is_monotonic_increasing


def test_discount_curve_decreasing_in_maturity():
    df_fn = lib25.discount_curve(pd.Timestamp("2018-01-02"))
    assert df_fn(0.0) == pytest.approx(1.0)
    assert df_fn(5.0) <= df_fn(1.0) <= df_fn(0.1)


def test_inputs_as_of_returns_callable_bundle():
    inputs = lib25.inputs_as_of("CL", pd.Timestamp("2017-06-04"))
    assert inputs["state"] in ("contango", "backwardation", "unknown")
    assert np.isfinite(inputs["F"](1.0)).all()
    assert np.isfinite(inputs["sigma"](1.0)).all()
    assert inputs["sigma"](1.0)[0] > 0.0


def test_corr_rolling_bounded():
    corr = lib25.corr_rolling("CL", "HO", window=126).dropna()
    assert len(corr) > 0
    assert corr.between(-1.0001, 1.0001).all()
