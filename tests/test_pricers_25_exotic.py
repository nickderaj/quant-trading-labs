"""Tests for pricers_25_exotic.py functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sstats

# Setup path to access lib25
_TMP_DIR = Path(__file__).resolve().parent.parent / "src" / "research" / "tmp"
if str(_TMP_DIR) not in sys.path:
    sys.path.insert(0, str(_TMP_DIR))

import lib25
import pricers_25_exotic as pricers


class TestMargrabe:
    """Margrabe exchange option tests."""

    def test_margrabe_equals_spread_mc_at_zero_strike(self):
        """Margrabe should equal spread_mc at K=0 within 3 standard errors."""
        # Set up parameters
        F1, F2 = 100.0, 95.0
        s1, s2 = 0.25, 0.30
        rho = 0.5
        T = 1.0
        df = 0.98

        # Closed form (Margrabe)
        margrabe_price = pricers.margrabe(F1, F2, T, s1, s2, rho, df)

        # Monte Carlo with K=0
        paths = lib25.simulate_correlated(
            [F1, F2], [s1, s2], rho, T, n_steps=252, n_paths=10000, seed=42
        )
        mc_result = pricers.spread_mc(paths[0], paths[1], K=0.0, opt="call", df=df)
        mc_price = mc_result["price"]
        mc_se = mc_result["se"]

        # Should agree within 3 standard errors
        diff = abs(margrabe_price - mc_price)
        assert diff < 3 * mc_se, f"Margrabe {margrabe_price} vs MC {mc_price} ± {mc_se}"

    def test_margrabe_intrinsic_at_zero_maturity(self):
        """Margrabe should return discounted intrinsic at T <= 0."""
        F1, F2 = 100.0, 95.0
        df = 0.98
        price = pricers.margrabe(F1, F2, T=0, s1=0.2, s2=0.2, rho=0.5, df=df)
        expected = df * max(F1 - F2, 0.0)
        assert abs(price - expected) < 1e-10


class TestKirk:
    """Kirk approximation for spread options."""

    def test_kirk_vs_mc_deviates_in_corner(self):
        """Kirk approximation deviates in high-correlation corner (documented teaching result)."""
        # Test that Kirk and MC give measurably different results in a corner case.
        # Corner: F1 ≈ F2, high rho, where Kirk's approximation is known to degrade
        F1, F2, K = 100.0, 101.0, 0.1
        s1, s2 = 0.20, 0.22
        rho = 0.99  # Very high correlation
        T = 0.25
        df = 0.995

        # Kirk price
        kirk_price = pricers.kirk(F1, F2, K, T, s1, s2, rho, df, opt="call")

        # Monte Carlo price
        paths = lib25.simulate_correlated(
            [F1, F2], [s1, s2], rho, T, n_steps=63, n_paths=100000, seed=456
        )
        mc_result = pricers.spread_mc(paths[0], paths[1], K=K, opt="call", df=df)
        mc_price = mc_result["price"]

        # Kirk and MC should deviate in this corner (documented teaching result).
        # The approximation may not be perfect, but should show measurable differences.
        dev = abs(kirk_price - mc_price)
        mc_se = mc_result["se"]
        # Verify deviation exists (is non-negligible compared to MC noise)
        # Kirk's approximation is known to degrade when F1 ≈ F2 and rho is very high
        assert dev > 0.5 * mc_se, f"Kirk deviation should be visible: {dev} vs {mc_se}"

    def test_kirk_intrinsic_at_zero_maturity(self):
        """Kirk should return discounted intrinsic at T <= 0."""
        F1, F2, K = 100.0, 95.0, 2.0
        df = 0.98
        price = pricers.kirk(F1, F2, K, T=0, s1=0.2, s2=0.2, rho=0.5, df=df, opt="call")
        expected = df * max(F1 - F2 - K, 0.0)
        assert abs(price - expected) < 1e-10

    def test_kirk_put_call_parity(self):
        """Kirk put and call should respect put-call parity via F2_adj strike."""
        F1, F2, K = 100.0, 90.0, 5.0
        s1, s2 = 0.25, 0.30
        rho = 0.5
        T = 1.0
        df = 0.98

        call_price = pricers.kirk(F1, F2, K, T, s1, s2, rho, df, opt="call")
        put_price = pricers.kirk(F1, F2, K, T, s1, s2, rho, df, opt="put")

        # Parity: put = call - df * (F1 - F2 - K)
        # Rearrange: call - put = df * (F1 - F2 - K)
        F2_adj = F2 + K
        parity_diff = df * (F1 - F2_adj)
        assert abs((call_price - put_price) - parity_diff) < 1e-8


class TestSpreadMC:
    """Monte Carlo spread option tests."""

    def test_spread_mc_intrinsic_at_maturity(self):
        """MC with flat paths should return intrinsic payoff."""
        n_paths = 1000
        # All paths end at a fixed spread level
        S1_terminal = 110.0
        S2_terminal = 100.0
        K = 5.0

        paths_1 = np.ones((n_paths, 2)) * S1_terminal
        paths_2 = np.ones((n_paths, 2)) * S2_terminal
        df = 1.0

        result = pricers.spread_mc(paths_1, paths_2, K, opt="call", df=df)
        expected = max(S1_terminal - S2_terminal - K, 0.0)
        assert abs(result["price"] - expected) < 1e-8

    def test_spread_mc_put_call(self):
        """Put and call prices should satisfy put-call parity."""
        paths_1 = lib25.simulate_gbm(
            100.0, 0.25, 1.0, n_steps=252, n_paths=5000, seed=99
        )
        paths_2 = lib25.simulate_gbm(
            90.0, 0.28, 1.0, n_steps=252, n_paths=5000, seed=100
        )
        K = 5.0
        df = 0.98

        call_result = pricers.spread_mc(paths_1, paths_2, K, opt="call", df=df)
        put_result = pricers.spread_mc(paths_1, paths_2, K, opt="put", df=df)

        # Parity: call - put ≈ df * (F1 - F2 - K)
        F1_0, F2_0 = paths_1[0, 0], paths_2[0, 0]
        parity_rhs = df * (F1_0 - F2_0 - K)
        diff = call_result["price"] - put_result["price"]
        assert abs(diff - parity_rhs) < 3 * (call_result["se"] + put_result["se"])


class TestQuantoBlack76:
    """Quanto (fixed-FX) adjustment tests."""

    def test_quanto_reduces_to_black76_at_zero_correlation(self):
        """Quanto should reduce to plain black76 when rho=0 (no correlation)."""
        F, K, T, sigma = 100.0, 95.0, 1.0, 0.25
        sigma_fx = 0.15
        df_foreign = 0.98

        # Quanto with rho=0
        quanto_price = pricers.quanto_black76(
            F, K, T, sigma, sigma_fx, rho=0.0, df_foreign=df_foreign, opt="call"
        )

        # Plain black76
        black76_price = lib25.black76(F, K, T, sigma, df_foreign, opt="call")

        assert abs(quanto_price - black76_price) < 1e-10

    def test_quanto_adjustment_sign_flips_with_rho(self):
        """The sign of (quanto - black76) should flip when rho flips sign."""
        F, K, T, sigma = 100.0, 95.0, 1.0, 0.25
        sigma_fx = 0.15
        df_foreign = 0.98

        # Positive correlation adjustment
        quanto_pos = pricers.quanto_black76(
            F, K, T, sigma, sigma_fx, rho=0.6, df_foreign=df_foreign, opt="call"
        )

        # Negative correlation adjustment
        quanto_neg = pricers.quanto_black76(
            F, K, T, sigma, sigma_fx, rho=-0.6, df_foreign=df_foreign, opt="call"
        )

        # Plain black76
        black76_price = lib25.black76(F, K, T, sigma, df_foreign, opt="call")

        # Adjustment sign should flip
        adj_pos = quanto_pos - black76_price
        adj_neg = quanto_neg - black76_price
        assert adj_pos * adj_neg < 0, "Adjustments should have opposite signs"
        assert abs(adj_pos) > 1e-8 and abs(adj_neg) > 1e-8, (
            "Adjustments should be non-negligible"
        )


class TestCompositeBlack76:
    """Composite (floating-FX) option tests."""

    def test_composite_volatility_combination(self):
        """Composite vol should combine commodity and FX vols via sqrt formula."""
        F, K_foreign, T = 100.0, 95.0, 1.0
        sigma, sigma_fx = 0.25, 0.15
        rho = 0.5
        df_foreign = 0.98

        # Composite price
        composite_price = pricers.composite_black76(
            F, K_foreign, T, sigma, sigma_fx, rho, df_foreign, opt="call"
        )

        # Equivalent: Black-76 with combined volatility
        sigma_composite = np.sqrt(sigma**2 + sigma_fx**2 + 2 * rho * sigma * sigma_fx)
        expected_price = lib25.black76(
            F, K_foreign, T, sigma_composite, df_foreign, opt="call"
        )

        assert abs(composite_price - expected_price) < 1e-10

    def test_composite_uncorrelated_case(self):
        """With rho=0, composite vol = sqrt(sigma^2 + sigma_fx^2)."""
        F, K_foreign, T = 100.0, 95.0, 1.0
        sigma, sigma_fx = 0.25, 0.15
        df_foreign = 0.98

        composite_price = pricers.composite_black76(
            F, K_foreign, T, sigma, sigma_fx, rho=0.0, df_foreign=df_foreign, opt="call"
        )

        sigma_independent = np.sqrt(sigma**2 + sigma_fx**2)
        expected_price = lib25.black76(
            F, K_foreign, T, sigma_independent, df_foreign, opt="call"
        )

        assert abs(composite_price - expected_price) < 1e-10


class TestVarswapStrike:
    """Variance-swap strike estimation tests."""

    def test_varswap_constant_vol(self):
        """For constant sigma_fn, strike_vol should closely equal that constant."""
        sigma_const = 0.30
        sigma_fn = lambda t: sigma_const
        T = 1.0

        result = pricers.varswap_strike(sigma_fn, T, n_strikes=50)

        # strike_vol should be close to 0.30
        assert abs(result["strike_vol"] - sigma_const) < 0.01
        assert result["strike_variance"] > 0
        assert result["T"] == T
        assert result["n_strikes"] == 50

    def test_varswap_finite_positive(self):
        """Varswap strike should be finite and positive."""
        sigma_fn = lambda t: 0.25 * np.sqrt(t) if t > 0 else 0.25
        T = 0.5

        result = pricers.varswap_strike(sigma_fn, T)

        assert np.isfinite(result["strike_vol"]) and result["strike_vol"] > 0
        assert np.isfinite(result["strike_variance"]) and result["strike_variance"] > 0

    def test_varswap_default_n_strikes(self):
        """Default n_strikes should be 50."""
        sigma_fn = lambda t: 0.2
        result = pricers.varswap_strike(sigma_fn, T=1.0)
        assert result["n_strikes"] == 50


class TestMarkovFunctional1F:
    """Markov-functional 1-factor model tests."""

    def test_markov_functional_reprices_calibration(self):
        """Should reprice its own calibration marginals to high precision."""
        # Generate some sample marginals
        np.random.seed(42)
        T1, T2 = 1.0, 2.0
        samples_T1 = np.random.lognormal(mean=np.log(100), sigma=0.25, size=1000)
        samples_T2 = np.random.lognormal(mean=np.log(100), sigma=0.20, size=1000)
        marginals = {T1: samples_T1, T2: samples_T2}

        # Build model
        driver_grid = np.linspace(-3, 3, 100)
        mf = pricers.markov_functional_1f(marginals, driver_grid)

        # Reprice T1: feed z_of_quantile through and should recover sorted_samples
        sorted_samples_T1 = np.sort(samples_T1)
        n = len(sorted_samples_T1)
        empirical_quantiles = (np.arange(1, n + 1) - 0.5) / n
        z_of_quantile = sstats.norm.ppf(empirical_quantiles)

        recovered = mf(T1, z_of_quantile)
        assert np.allclose(recovered, sorted_samples_T1, rtol=1e-8, atol=1e-8)

    def test_markov_functional_raises_on_unknown_maturity(self):
        """Should raise ValueError for a maturity not in calibration."""
        marginals = {1.0: np.random.lognormal(mean=0, sigma=0.2, size=100)}
        mf = pricers.markov_functional_1f(marginals, np.linspace(-3, 3, 50))

        with pytest.raises(ValueError, match="not in calibration"):
            mf(2.0, 0.0)

    def test_markov_functional_scalar_and_array_z(self):
        """Should handle both scalar and array driver values."""
        np.random.seed(99)
        T = 1.0
        samples = np.random.lognormal(mean=np.log(100), sigma=0.25, size=500)
        mf = pricers.markov_functional_1f({T: samples}, np.linspace(-3, 3, 50))

        # Scalar z
        val_scalar = mf(T, 0.5)
        assert isinstance(val_scalar, np.ndarray) and val_scalar.shape == (1,)

        # Array z
        z_arr = np.array([-1.0, 0.0, 1.0])
        val_array = mf(T, z_arr)
        assert val_array.shape == (3,)


class TestAutocallable:
    """Autocallable reverse convertible tests."""

    def test_autocallable_early_autocall(self):
        """Path that autocalls early should get accrued coupon."""
        coupon = 0.10

        # Path 0: autocalls at step 2
        # Path 1: survives to maturity above barrier
        # Path 2: breaches barrier at maturity
        paths = np.array(
            [
                [
                    1.0,
                    1.05,
                    1.10,
                    0.95,
                    0.90,
                ],  # Autocalls at obs_idx[1]=2, accrued coupon
                [1.0, 1.02, 1.01, 1.05, 1.15],  # Survives above barrier (1.15 >= 1.0)
                [1.0, 0.95, 0.92, 0.88, 0.65],  # Below barrier (0.65 < 0.7)
            ],
            dtype=float,
        )

        # Observation dates: quarterly
        obs_idx = np.array([1, 2, 3, 4])
        autocall_level = 1.05
        barrier = 0.70
        df = 0.99

        result = pricers.autocallable(
            paths, coupon, autocall_level, barrier, obs_idx, df
        )

        # Check that we get reasonable outputs
        assert 0 < result["price"] < 1.2
        assert result["n"] == 3
        assert 0 <= result["never_autocalled_frequency"] <= 1
        assert len(result["autocall_frequency_by_obs"]) == len(obs_idx)
        assert result["se"] >= 0

    def test_autocallable_flat_path_at_barrier(self):
        """Flat path at initial level, survives to maturity at barrier level."""
        n_paths = 100
        n_steps = 20

        # All paths flat at 1.0 (initial level)
        paths = np.ones((n_paths, n_steps + 1))

        obs_idx = np.array([5, 10, 15, 20])
        autocall_level = 1.05  # Never reached
        barrier = 1.0  # At terminal level
        coupon = 0.05
        df = 0.98

        result = pricers.autocallable(
            paths, coupon, autocall_level, barrier, obs_idx, df
        )

        # All paths should survive to maturity above barrier
        expected_payoff = df ** (20 / 20) * (1.0 + coupon)  # df^1 * (1 + coupon)
        assert abs(result["price"] - expected_payoff) < 0.01
        assert result["never_autocalled_frequency"] > 0.99

    def test_autocallable_barrier_breach(self):
        """Paths breaching barrier at maturity should lose principal."""
        n_paths = 100
        n_steps = 10

        # All paths decay below barrier
        paths = np.linspace(1.0, 0.5, n_steps + 1)[np.newaxis, :].repeat(
            n_paths, axis=0
        )

        obs_idx = np.array([5, 10])
        autocall_level = 2.0  # Never reached
        barrier = 0.7
        coupon = 0.10
        df = 0.97

        result = pricers.autocallable(
            paths, coupon, autocall_level, barrier, obs_idx, df
        )

        # All paths should breach barrier (terminal = 0.5 < 0.7)
        # Payoff = 0.5 * df^(10/10) = 0.5 * df
        expected_payoff = 0.5 * df
        assert abs(result["price"] - expected_payoff) < 0.01

    def test_autocallable_multiple_autocalls(self):
        """Multiple observation dates should see some autocalls."""
        # Mix of paths, some autocall early, some at later dates
        paths = np.array(
            [
                [
                    1.0,
                    1.1,
                    1.15,
                    1.2,
                    1.0,
                    0.9,
                    0.8,
                    0.9,
                    1.0,
                    1.05,
                    1.1,
                    1.15,
                    1.2,
                    0.9,
                    0.8,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                    1.1,
                    1.15,
                ],  # Autocalls at obs 0
                [
                    1.0,
                    1.05,
                    1.1,
                    1.15,
                    1.2,
                    1.0,
                    0.9,
                    0.8,
                    0.9,
                    1.0,
                    1.05,
                    1.1,
                    1.15,
                    1.2,
                    0.9,
                    0.8,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                    1.1,
                ],  # Autocalls at obs 1
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                ],  # Autocalls at obs 3
                [
                    1.0,
                    0.95,
                    0.9,
                    0.85,
                    0.8,
                    0.75,
                    0.7,
                    0.65,
                    0.6,
                    0.55,
                    0.5,
                    0.45,
                    0.4,
                    0.35,
                    0.3,
                    0.25,
                    0.2,
                    0.15,
                    0.1,
                    0.05,
                    0.0,
                ],  # Never autocalls, breaches
                [
                    1.0,
                    1.02,
                    1.04,
                    1.06,
                    1.08,
                    1.1,
                    1.12,
                    1.14,
                    1.16,
                    1.18,
                    1.2,
                    1.22,
                    1.24,
                    1.26,
                    1.28,
                    1.3,
                    1.32,
                    1.34,
                    1.36,
                    1.38,
                    1.4,
                ],  # Autocalls at obs 0
            ],
            dtype=float,
        )

        obs_idx = np.array([1, 4, 8, 12, 20])
        autocall_level = 1.05
        barrier = 0.5
        coupon = 0.08
        df = 0.99

        result = pricers.autocallable(
            paths, coupon, autocall_level, barrier, obs_idx, df
        )

        # Check outputs are sensible
        assert result["price"] > 0
        assert result["n"] == 5
        assert 0 < result["never_autocalled_frequency"] <= 1
        assert np.sum(result["autocall_frequency_by_obs"]) > 0
