"""Tests for Asian option pricing (geometric and arithmetic averages).

Six required test scenarios:
1. asian_geometric vs asian_mc agree within 3 standard errors
2. Arithmetic >= Geometric Asian prices
3. Asian <= European prices
4. averaging_schedule returns only business days in the month
5. Single fixing converges to European (as n_fix->1, avg_start->T)
6. Control variate reduces variance
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TMP_DIR = Path(__file__).resolve().parent.parent / "src" / "research" / "tmp"
sys.path.insert(0, str(_TMP_DIR))

import lib25
import pricers_25_asian as asian


class TestAsianGeometricVsMC:
    """Test 1: asian_geometric vs asian_mc (geometric payoff) agree within 3se."""

    def test_geometric_asian_vs_mc_large_sample(self):
        """Geometric average payoff from MC should match closed form within 3 SE."""
        F, K, T = 100.0, 100.0, 0.25
        sigma, df = 0.30, 0.995
        opt = "call"
        avg_start, n_fix = 0.0, 5

        # Closed-form geometric Asian price
        closed_price = asian.asian_geometric(F, K, T, sigma, df, opt, avg_start, n_fix)

        # MC simulation with geometric average
        n_paths = 100_000
        paths = lib25.simulate_gbm(F, sigma, T, n_steps=50, n_paths=n_paths, seed=42)

        # Fixing times for this averaging schedule
        t = avg_start + np.arange(n_fix) * (T - avg_start) / (n_fix - 1)
        # Convert to step indices: step i corresponds to time i*T/n_steps
        avg_idx = np.round(t / T * 50).astype(int)
        avg_idx = np.clip(avg_idx, 0, 50)

        # MC payoff: geometric average (no control variate for clean comparison)
        geo_avg = np.exp(np.log(paths[:, avg_idx]).mean(axis=1))
        if opt == "call":
            payoff = np.maximum(geo_avg - K, 0.0)
        else:
            payoff = np.maximum(K - geo_avg, 0.0)
        mc_price = df * np.mean(payoff)
        mc_se = df * np.std(payoff, ddof=1) / np.sqrt(n_paths)

        # Should agree within 3 standard errors
        diff = abs(closed_price - mc_price)
        assert diff <= 3 * mc_se, (
            f"Closed form {closed_price:.6f} vs MC {mc_price:.6f} (se={mc_se:.6f}), diff={diff:.6f}"
        )


class TestAsianOrdering:
    """Test 2: Arithmetic Asian >= Geometric Asian (same params)."""

    def test_arithmetic_geq_geometric(self):
        """For any set of params, arithmetic average price >= geometric."""
        F, K, T = 100.0, 100.0, 0.25
        sigma, df = 0.30, 0.995
        opt = "call"
        avg_start, n_fix = 0.0, 5

        arith_price = asian.asian_turnbull_wakeman(
            F, K, T, sigma, df, opt, avg_start, n_fix
        )
        geom_price = asian.asian_geometric(F, K, T, sigma, df, opt, avg_start, n_fix)

        # Arithmetic should be >= geometric (in value, not necessarily premium)
        # For calls, the effective forward of arithmetic is >= geometric
        assert arith_price >= geom_price - 1e-10, (
            f"Arithmetic {arith_price:.6f} < Geometric {geom_price:.6f}"
        )

    def test_arithmetic_ordering_both_opts(self):
        """For calls, arith >= geom; for puts, arith <= geom (arithmetic mean >= geometric mean)."""
        F, K, T = 100.0, 105.0, 0.25
        sigma, df = 0.25, 0.99
        avg_start, n_fix = 0.0, 3

        # Call: arithmetic forward >= geometric forward => call premium >= geometric call premium
        arith_call = asian.asian_turnbull_wakeman(
            F, K, T, sigma, df, "call", avg_start, n_fix
        )
        geom_call = asian.asian_geometric(F, K, T, sigma, df, "call", avg_start, n_fix)
        assert arith_call >= geom_call - 1e-10, (
            f"Call: Arithmetic {arith_call:.6f} < Geometric {geom_call:.6f}"
        )

        # Put: arithmetic forward >= geometric forward => put premium <= geometric put premium
        arith_put = asian.asian_turnbull_wakeman(
            F, K, T, sigma, df, "put", avg_start, n_fix
        )
        geom_put = asian.asian_geometric(F, K, T, sigma, df, "put", avg_start, n_fix)
        assert arith_put <= geom_put + 1e-10, (
            f"Put: Arithmetic {arith_put:.6f} > Geometric {geom_put:.6f}"
        )


class TestAsianVsEuropean:
    """Test 3: Asian price <= European price (averaging reduces effective vol)."""

    def test_asian_call_leq_european(self):
        """Asian call price <= European at same strike/expiry/vol."""
        F, K, T = 100.0, 100.0, 0.25
        sigma, df = 0.30, 0.995
        avg_start, n_fix = 0.0, 5

        asian_price = asian.asian_geometric(
            F, K, T, sigma, df, "call", avg_start, n_fix
        )
        european_price = lib25.black76(F, K, T, sigma, df, "call")

        assert asian_price <= european_price + 1e-10, (
            f"Asian {asian_price:.6f} > European {european_price:.6f}"
        )

    def test_asian_put_leq_european(self):
        """Asian put price <= European."""
        F, K, T = 100.0, 110.0, 0.25
        sigma, df = 0.30, 0.995
        avg_start, n_fix = 0.0, 3

        asian_price = asian.asian_geometric(F, K, T, sigma, df, "put", avg_start, n_fix)
        european_price = lib25.black76(F, K, T, sigma, df, "put")

        assert asian_price <= european_price + 1e-10, (
            f"Asian {asian_price:.6f} > European {european_price:.6f} for put"
        )


class TestAveragingSchedule:
    """Test 4: averaging_schedule returns only business days in the month."""

    def test_schedule_only_business_days(self):
        """Should return only Mon-Fri (no weekends)."""
        dates = asian.averaging_schedule("CL", "2020-01")

        # Convert to pandas for day-of-week check
        pd_dates = pd.DatetimeIndex(dates)
        day_of_week = pd_dates.dayofweek  # 0=Mon, 4=Fri, 5=Sat, 6=Sun

        # All should be Mon-Fri
        assert np.all(day_of_week < 5), (
            f"Found weekend dates: {dates[day_of_week >= 5]}"
        )

    def test_schedule_within_month(self):
        """All dates should be within the specified calendar month."""
        month = "2020-06"
        dates = asian.averaging_schedule("CL", month)

        pd_dates = pd.DatetimeIndex(dates)
        month_ts = pd.Timestamp(month)
        month_end = month_ts + pd.offsets.MonthEnd(0)

        assert np.all(pd_dates >= month_ts), (
            f"Found dates before month start: {dates[pd_dates < month_ts]}"
        )
        assert np.all(pd_dates <= month_end), (
            f"Found dates after month end: {dates[pd_dates > month_end]}"
        )

    def test_schedule_february_leap_year(self):
        """February in leap year has 29 days."""
        dates = asian.averaging_schedule("CL", "2020-02")
        pd_dates = pd.DatetimeIndex(dates)

        # 2020 is a leap year; Feb has 29 days, which includes 21-22 business days
        assert len(dates) > 18, f"February 2020 should have ~22 bdays, got {len(dates)}"
        assert np.all(pd_dates.month == 2), "All dates should be in February"


class TestSingleFixingConvergence:
    """Test 5: As n_fix->1 (single fixing at t=T), Asian converges to European."""

    def test_geometric_single_fix_to_european(self):
        """Single fixing at t=T should match European."""
        F, K, T = 100.0, 100.0, 0.25
        sigma, df = 0.30, 0.995

        # Single fixing at T (avg_start=T, n_fix=1)
        asian_price = asian.asian_geometric(
            F, K, T, sigma, df, "call", avg_start=T, n_fix=1
        )
        european_price = lib25.black76(F, K, T, sigma, df, "call")

        # Should be very close
        assert abs(asian_price - european_price) < 1e-8, (
            f"Asian single fix {asian_price:.10f} != European {european_price:.10f}"
        )

    def test_arithmetic_single_fix_to_european(self):
        """Single fixing at t=T should match European for arithmetic too."""
        F, K, T = 100.0, 95.0, 0.25
        sigma, df = 0.25, 0.99

        asian_price = asian.asian_turnbull_wakeman(
            F, K, T, sigma, df, "put", avg_start=T, n_fix=1
        )
        european_price = lib25.black76(F, K, T, sigma, df, "put")

        assert abs(asian_price - european_price) < 1e-8, (
            f"Arithmetic single fix {asian_price:.10f} != European {european_price:.10f}"
        )

    def test_both_converge_same_value(self):
        """Geometric and arithmetic should both converge to European."""
        F, K, T = 100.0, 100.0, 0.5
        sigma, df = 0.20, 0.98

        geo_single = asian.asian_geometric(
            F, K, T, sigma, df, "call", avg_start=T, n_fix=1
        )
        arith_single = asian.asian_turnbull_wakeman(
            F, K, T, sigma, df, "call", avg_start=T, n_fix=1
        )
        european = lib25.black76(F, K, T, sigma, df, "call")

        # All three should match
        assert abs(geo_single - european) < 1e-8
        assert abs(arith_single - european) < 1e-8
        assert abs(geo_single - arith_single) < 1e-10


class TestControlVariate:
    """Test 6: Control variate reduces variance."""

    def test_control_variate_reduces_se(self):
        """SE with control variate should be smaller than without."""
        F, K, T = 100.0, 100.0, 0.25
        sigma, df = 0.30, 0.995

        # Generate paths
        n_paths = 10_000
        paths = lib25.simulate_gbm(F, sigma, T, n_steps=50, n_paths=n_paths, seed=42)

        # Averaging at 5 equally-spaced times
        n_fix = 5
        t = 0.0 + np.arange(n_fix) * (T - 0.0) / (n_fix - 1)
        avg_idx = np.round(t / T * 50).astype(int)
        avg_idx = np.clip(avg_idx, 0, 50)

        # MC with and without control variate
        result_with_cv = asian.asian_mc(
            paths, K, "call", df, avg_idx, control_variate=True
        )
        result_no_cv = asian.asian_mc(
            paths, K, "call", df, avg_idx, control_variate=False
        )

        se_with = result_with_cv["se"]
        se_no = result_no_cv["se"]

        # SE should be reduced (or at least not significantly worse)
        # Allow small tolerance for randomness in covariance estimation
        assert se_with <= se_no * 1.01, f"CV SE {se_with:.6f} > No-CV SE {se_no:.6f}"

    def test_control_variate_same_path_sampling(self):
        """Both methods should use same paths and fixing indices."""
        F, K, T = 100.0, 95.0, 0.25
        sigma, df = 0.25, 0.99

        n_paths = 5000
        paths = lib25.simulate_gbm(F, sigma, T, n_steps=30, n_paths=n_paths, seed=123)

        n_fix = 3
        t = 0.05 + np.arange(n_fix) * (T - 0.05) / (n_fix - 1)
        avg_idx = np.round(t / T * 30).astype(int)
        avg_idx = np.clip(avg_idx, 0, 30)

        result_with = asian.asian_mc(paths, K, "put", df, avg_idx, control_variate=True)
        result_no = asian.asian_mc(paths, K, "put", df, avg_idx, control_variate=False)

        # Both should use same number of paths
        assert result_with["n"] == result_no["n"] == n_paths

        # Prices should be similar (within a few SE of each other)
        # The CV version may be more or less efficient, but should be in ballpark
        price_diff = abs(result_with["price"] - result_no["price"])
        # Allow up to 2 standard errors of difference
        max_se = max(result_with["se"], result_no["se"])
        assert price_diff <= 2 * max_se, (
            f"Prices differ by {price_diff:.6f}, max SE {max_se:.6f}"
        )


class TestEdgeCases:
    """Additional edge case tests for robustness."""

    def test_zero_vol_fallback(self):
        """With sigma=0, should return discounted intrinsic."""
        F, K, T = 100.0, 100.0, 0.25
        df = 0.99

        geo_price = asian.asian_geometric(F, K, T, 0.0, df, "call", 0.0, 3)
        intrinsic = max(F - K, 0.0) * df
        assert abs(geo_price - intrinsic) < 1e-10

    def test_zero_time_fallback(self):
        """With T=0, should return discounted intrinsic."""
        F, K = 100.0, 95.0
        df = 0.99

        put_price = asian.asian_turnbull_wakeman(F, K, 0.0, 0.30, df, "put", 0.0, 1)
        intrinsic = max(K - F, 0.0) * df
        assert abs(put_price - intrinsic) < 1e-10

    def test_large_strike_call(self):
        """Deep OTM call should be cheap."""
        F, K, T = 100.0, 150.0, 0.25
        sigma, df = 0.30, 0.995

        price = asian.asian_geometric(F, K, T, sigma, df, "call", 0.0, 3)
        # Should be positive but small
        assert 0.0 <= price < 1.0

    def test_deep_itm_put(self):
        """Deep ITM put should be close to intrinsic."""
        F, K, T = 100.0, 150.0, 0.25
        sigma, df = 0.30, 0.995

        price = asian.asian_geometric(F, K, T, sigma, df, "put", 0.0, 3)
        intrinsic = (K - F) * df
        # Price should be close to intrinsic (slightly above due to vol)
        assert intrinsic <= price <= intrinsic * 1.05
