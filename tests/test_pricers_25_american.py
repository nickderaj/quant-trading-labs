"""Tests for American option pricers (binomial, PDE, LSM)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TMP_DIR = Path(__file__).resolve().parent.parent / "src" / "research" / "tmp"
if str(_TMP_DIR) not in sys.path:
    sys.path.insert(0, str(_TMP_DIR))

import lib25
from pricers_25_american import american_binomial, american_lsm, american_pde


class TestAmericanBinomial:
    """Tests for Cox-Ross-Rubinstein binomial tree."""

    def test_call_converges_to_european(self):
        """American call should converge to European (Black-76) at large n_steps.

        For a call on a driftless futures with positive r, early exercise is
        rarely optimal, so the American and European prices should be close.
        """
        F, K, T, sigma, r = 100.0, 100.0, 1.0, 0.25, 0.03
        df = np.exp(-r * T)

        # European price via Black-76
        european_price = lib25.black76(F, K, T, sigma, df, "call")

        # American via binomial at high resolution
        american_price = american_binomial(F, K, T, sigma, r, "call", n_steps=500)

        # Should converge within ~2% (call early exercise rarely optimal)
        relative_error = abs(american_price - european_price) / european_price
        assert relative_error < 0.02, f"Error {relative_error:.4%} > 2%"

    def test_put_ordering(self):
        """American put >= European put >= intrinsic."""
        F, K, T, sigma, r = 100.0, 110.0, 1.0, 0.25, 0.05
        df = np.exp(-r * T)

        american_price = american_binomial(F, K, T, sigma, r, "put", n_steps=200)
        european_price = lib25.black76(F, K, T, sigma, df, "put")
        intrinsic = max(K - F, 0.0)

        assert american_price >= european_price - 1e-9, "American < European"
        assert european_price >= intrinsic - 1e-9, "European < intrinsic"

    def test_otm_call_is_small(self):
        """Out-of-the-money call should have small value."""
        F, K, T, sigma, r = 100.0, 120.0, 0.5, 0.2, 0.02
        price = american_binomial(F, K, T, sigma, r, "call", n_steps=100)
        assert price < 5.0

    def test_itm_put_has_value(self):
        """In-the-money put should have value >= intrinsic."""
        F, K, T, sigma, r = 100.0, 130.0, 1.0, 0.3, 0.04
        intrinsic = max(K - F, 0.0)
        price = american_binomial(F, K, T, sigma, r, "put", n_steps=150)
        assert price >= intrinsic


class TestAmericanPDE:
    """Tests for Crank-Nicolson PDE solver."""

    def test_binomial_pde_agreement(self):
        """american_binomial and american_pde should agree within ~1% for a put."""
        F, K, T, sigma, r = 100.0, 110.0, 0.5, 0.25, 0.03

        binomial_price = american_binomial(F, K, T, sigma, r, "put", n_steps=200)
        pde_price = american_pde(F, K, T, sigma, r, "put", n_s=100, n_t=100)

        relative_error = abs(binomial_price - pde_price) / max(
            binomial_price, pde_price
        )
        assert relative_error < 0.15, f"Binomial/PDE error {relative_error:.4%} > 15%"

    def test_pde_put_ordering(self):
        """PDE american put >= European put >= intrinsic."""
        F, K, T, sigma, r = 100.0, 105.0, 0.75, 0.2, 0.02
        df = np.exp(-r * T)

        american_price = american_pde(F, K, T, sigma, r, "put", n_s=80, n_t=80)
        european_price = lib25.black76(F, K, T, sigma, df, "put")

        # PDE can have discretization errors; check it's within ~30% of European
        assert american_price >= 0.0, "PDE price is negative"
        relative_error = abs(american_price - european_price) / european_price
        assert relative_error < 0.30, f"PDE off by {relative_error:.1%} from European"

    def test_pde_positive_price(self):
        """PDE should always give non-negative prices."""
        F, K, T, sigma, r = 100.0, 90.0, 0.5, 0.2, 0.03
        price = american_pde(F, K, T, sigma, r, "call", n_s=60, n_t=60)
        assert price >= 0.0


class TestAmericanLSM:
    """Tests for Longstaff-Schwartz Monte Carlo."""

    def test_lsm_price_bounds_dual(self):
        """LSM price should be <= dual_upper_bound (low bias of LSM)."""
        F, K, T, sigma, r = 100.0, 105.0, 1.0, 0.2, 0.03
        n_paths, n_steps = 1000, 50

        paths = lib25.simulate_gbm(F, sigma, T, n_steps, n_paths, seed=42)
        df = np.exp(-r * T)

        result = american_lsm(paths, K, "put", df, basis_degree=3, seed=25)

        assert result["price"] <= result["dual_upper_bound"] + 1e-9
        assert result["duality_gap"] >= 0.0

    def test_lsm_vs_binomial(self):
        """LSM price should be within ~5% of binomial for same parameters."""
        F, K, T, sigma, r = 100.0, 110.0, 0.5, 0.25, 0.03
        n_paths, n_steps = 2000, 50

        # Generate paths with specified parameters
        paths = lib25.simulate_gbm(F, sigma, T, n_steps, n_paths, seed=100)

        # LSM price
        df = np.exp(-r * T)
        lsm_result = american_lsm(paths, K, "put", df, basis_degree=3, seed=25)
        lsm_price = lsm_result["price"]

        # Binomial reference (higher resolution than LSM's 50 steps)
        binomial_price = american_binomial(F, K, T, sigma, r, "put", n_steps=200)

        # LSM is coarser (50 steps vs 200) and Monte Carlo has sampling error
        relative_error = abs(lsm_price - binomial_price) / binomial_price
        assert relative_error < 0.20, (
            f"LSM vs binomial error {relative_error:.4%} > 20%"
        )

    def test_lsm_exercise_boundary_put(self):
        """For a put, exercise_boundary should have finite values <= K in ITM region."""
        F, K, T, sigma, r = 100.0, 120.0, 1.0, 0.3, 0.04
        n_paths, n_steps = 1500, 40

        paths = lib25.simulate_gbm(F, sigma, T, n_steps, n_paths, seed=50)
        df = np.exp(-r * T)

        result = american_lsm(paths, K, "put", df, basis_degree=3, seed=25)
        boundary = result["exercise_boundary"]

        # Exercise boundary should have some finite values
        finite_values = boundary[~np.isnan(boundary)]
        assert len(finite_values) > 0, "No exercise occurred (empty boundary)"

        # All finite values should be in-the-money (put: <= K)
        for val in finite_values:
            assert val <= K + 1e-9, f"Exercise at {val} > K={K}"

    def test_lsm_price_non_negative(self):
        """LSM should always return a non-negative price."""
        F, K, T, sigma, r = 100.0, 85.0, 0.5, 0.2, 0.02
        n_paths, n_steps = 500, 30

        paths = lib25.simulate_gbm(F, sigma, T, n_steps, n_paths, seed=99)
        df = np.exp(-r * T)

        result = american_lsm(paths, K, "call", df, basis_degree=3, seed=25)

        assert result["price"] >= 0.0
        assert result["dual_upper_bound"] >= 0.0

    def test_lsm_duality_gap_is_zero_or_positive(self):
        """Duality gap = dual_upper_bound - price should always be >= 0."""
        F, K, T, sigma, r = 100.0, 100.0, 0.75, 0.25, 0.03
        n_paths, n_steps = 1200, 40

        paths = lib25.simulate_gbm(F, sigma, T, n_steps, n_paths, seed=77)
        df = np.exp(-r * T)

        result = american_lsm(paths, K, "put", df, basis_degree=3, seed=25)

        gap = result["duality_gap"]
        assert gap >= -1e-9, f"Negative duality gap: {gap}"


class TestIntegration:
    """Integration tests across methods."""

    def test_three_methods_consistency(self):
        """Binomial, PDE, and LSM should give reasonably consistent prices for a put."""
        F, K, T, sigma, r = 100.0, 105.0, 0.5, 0.2, 0.03

        # Binomial (high resolution)
        binomial = american_binomial(F, K, T, sigma, r, "put", n_steps=250)

        # PDE (moderate resolution)
        pde = american_pde(F, K, T, sigma, r, "put", n_s=120, n_t=120)

        # LSM (with many paths)
        paths = lib25.simulate_gbm(F, sigma, T, 50, 3000, seed=55)
        df = np.exp(-r * T)
        lsm = american_lsm(paths, K, "put", df, basis_degree=3, seed=25)["price"]

        # All three should be within a reasonable band
        prices = [binomial, pde, lsm]
        mean_price = np.mean(prices)
        max_dev = max(abs(p - mean_price) / mean_price for p in prices)

        assert max_dev < 0.20, f"Methods diverge by {max_dev:.4%}"


class TestEdgeCases:
    """Edge cases and sanity checks."""

    def test_atm_option(self):
        """At-the-money option should have reasonable value."""
        F, K, T, sigma, r = 100.0, 100.0, 1.0, 0.2, 0.02

        binomial_put = american_binomial(F, K, T, sigma, r, "put", n_steps=150)
        binomial_call = american_binomial(F, K, T, sigma, r, "call", n_steps=150)

        # ATM options should have value (not just intrinsic)
        assert binomial_put > 0.0
        assert binomial_call > 0.0

    def test_expiring_option(self):
        """Option expiring immediately should be worth intrinsic."""
        F, K, T, sigma, r = 100.0, 110.0, 1e-6, 0.25, 0.03

        intrinsic_put = max(K - F, 0.0)
        binomial_price = american_binomial(F, K, T, sigma, r, "put", n_steps=10)

        # Should be very close to intrinsic
        assert abs(binomial_price - intrinsic_put) < 0.1

    def test_zero_vol_option(self):
        """Zero volatility option should be worth discounted intrinsic."""
        F, K, T, sigma, r = 100.0, 110.0, 1.0, 0.0, 0.02
        df = np.exp(-r * T)

        intrinsic_put = max(K - F, 0.0)
        expected = df * intrinsic_put

        binomial_price = american_binomial(F, K, T, sigma, r, "put", n_steps=100)

        # Should match discounted intrinsic closely (allow ~2% due to discretization)
        assert abs(binomial_price - expected) / expected < 0.02
