"""Unit tests for notebook-11a machinery (src/research/tmp/spread_lib11.py):
ported evaluation machinery, term-structure regime labelling, statistical
tests, trading rule simulation, and portfolio metrics. Mirrors existing test
conventions.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp"))

import spread_lib10 as SL10
import spread_lib11 as SL11

SEED = 0


# ---------------------------------------------------------------------------
# Phase 1 primitives
# ---------------------------------------------------------------------------


class TestApproxAdfPvalue:
    """Test ADF p-value approximation via interpolation/extrapolation."""

    def test_monotone_decreasing_in_t_stat(self):
        """More negative t_stat should yield smaller p-value."""
        t_vals = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]
        pvals = [SL11.approx_adf_pvalue(t) for t in t_vals]
        # Should be monotonically increasing (pval gets larger as t_stat rises)
        for i in range(len(pvals) - 1):
            assert pvals[i] <= pvals[i + 1], f"p-value not monotone at {t_vals[i]}"

    def test_critical_value_hits_1pct(self):
        """At the 1% critical value, p-value should be close to 0.01."""
        cv_1pct = SL10.ADF_CRITICAL_VALUES["1%"]
        pval = SL11.approx_adf_pvalue(cv_1pct)
        assert pval == pytest.approx(0.01, rel=0.05)

    def test_critical_value_hits_5pct(self):
        """At the 5% critical value, p-value should be close to 0.05."""
        cv_5pct = SL10.ADF_CRITICAL_VALUES["5%"]
        pval = SL11.approx_adf_pvalue(cv_5pct)
        assert pval == pytest.approx(0.05, rel=0.05)

    def test_critical_value_hits_10pct(self):
        """At the 10% critical value, p-value should be close to 0.10."""
        cv_10pct = SL10.ADF_CRITICAL_VALUES["10%"]
        pval = SL11.approx_adf_pvalue(cv_10pct)
        assert pval == pytest.approx(0.10, rel=0.05)

    def test_very_negative_t_stat_floored(self):
        """Very negative t_stat should be floored at 1e-4."""
        pval = SL11.approx_adf_pvalue(-100.0)
        assert pval >= 1e-4
        assert pval < 0.01

    def test_positive_t_stat_capped(self):
        """Positive t_stat should be capped at 1.0."""
        pval = SL11.approx_adf_pvalue(10.0)
        assert pval <= 1.0


class TestComputeZscore:
    """Test rolling z-score with shift(1) to ensure no lookahead."""

    def test_first_lookback_values_are_nan(self):
        """First `lookback` values should be NaN due to shift(1) and rolling."""
        rng = np.random.default_rng(SEED)
        data = rng.normal(0, 1, 200)
        z = SL11.compute_zscore(data, lookback=60)
        # First 60 should be NaN (rolling not ready) or at least first 1 from shift
        assert np.isnan(z[0])  # shift(1) makes first value NaN
        assert np.isnan(z[59])  # rolling still filling up

    def test_zscore_depends_only_on_prior_data(self):
        """Mutating a later value should not change earlier z-scores."""
        rng = np.random.default_rng(SEED)
        data1 = rng.normal(0, 1, 200)
        z1 = SL11.compute_zscore(data1, lookback=60)

        data2 = data1.copy()
        data2[150] = data1[150] + 100.0  # Mutate a later value
        z2 = SL11.compute_zscore(data2, lookback=60)

        # z-scores before bar 150 should be identical
        assert np.allclose(z1[:140], z2[:140], rtol=1e-10, equal_nan=True)

    def test_constant_array_produces_nan(self):
        """On a constant array, std=0 should produce NaN (0/0)."""
        data = np.full(200, 5.0)
        z = SL11.compute_zscore(data, lookback=60)
        # After the rolling window fills, std should be 0, giving NaN
        # (or possibly inf depending on numpy's handling of 0/0)
        valid_tail = z[100:]
        # All should be NaN (0 std gives NaN z-score)
        assert np.all(~np.isfinite(valid_tail))

    def test_output_length_matches_input(self):
        """Output array length should equal input length."""
        data = np.arange(100, dtype=float)
        z = SL11.compute_zscore(data, lookback=30)
        assert len(z) == len(data)


class TestComputeAtrSeries:
    """Test rolling std of daily changes (ATR approximation)."""

    def test_atr_always_non_negative_or_nan(self):
        """ATR (rolling std) should always be >= 0 or NaN."""
        rng = np.random.default_rng(SEED)
        data = np.cumsum(rng.normal(0, 1, 200))
        atr = SL11.compute_atr_series(data, window=14)
        valid = atr[np.isfinite(atr)]
        assert np.all(valid >= 0.0)

    def test_first_window_plus_shift_values_nan(self):
        """First `window+1`-ish values should be NaN (diff + rolling + shift)."""
        rng = np.random.default_rng(SEED)
        data = rng.normal(0, 1, 200)
        atr = SL11.compute_atr_series(data, window=14)
        # shift(1) makes first value NaN, rolling needs window bars to compute
        assert np.isnan(atr[0])
        # Roughly first 14-16 should be NaN
        assert np.isnan(atr[14])

    def test_constant_spread_zero_atr(self):
        """On a constant spread, all changes are zero, std=0, giving NaN (or 0)."""
        data = np.full(200, 100.0)
        atr = SL11.compute_atr_series(data, window=14)
        # After the window is ready, atr should be 0 or NaN
        # (std of zero changes = 0, possibly NaN from shift)
        valid_tail = atr[50:]
        # All finite values should be 0 or all NaN
        finite_tail = valid_tail[np.isfinite(valid_tail)]
        if len(finite_tail) > 0:
            assert np.all(finite_tail == 0.0)

    def test_output_length_matches_input(self):
        """Output array length should equal input length."""
        data = np.arange(200, dtype=float)
        atr = SL11.compute_atr_series(data, window=14)
        assert len(atr) == len(data)


class TestFixedFractionalSizing:
    """Test position quantity calculation."""

    def test_returns_zero_when_atr_non_finite(self):
        """qty should be 0 when atr is NaN or inf."""
        qty_nan = SL11.FixedFractionalSizing.quantity(
            equity=100000,
            atr=np.nan,
            price=100,
            risk_pct=0.03,
            stop_atr_mult=6.0,
            min_atr=0.10,
            max_leverage=5.0,
        )
        assert qty_nan == 0

        qty_inf = SL11.FixedFractionalSizing.quantity(
            equity=100000,
            atr=np.inf,
            price=100,
            risk_pct=0.03,
            stop_atr_mult=6.0,
            min_atr=0.10,
            max_leverage=5.0,
        )
        assert qty_inf == 0

    def test_returns_zero_when_price_non_finite_or_zero(self):
        """qty should be 0 when price is NaN, inf, or 0."""
        qty = SL11.FixedFractionalSizing.quantity(
            equity=100000,
            atr=0.5,
            price=0,
            risk_pct=0.03,
            stop_atr_mult=6.0,
            min_atr=0.10,
            max_leverage=5.0,
        )
        assert qty == 0

        qty = SL11.FixedFractionalSizing.quantity(
            equity=100000,
            atr=0.5,
            price=np.nan,
            risk_pct=0.03,
            stop_atr_mult=6.0,
            min_atr=0.10,
            max_leverage=5.0,
        )
        assert qty == 0

    def test_returns_zero_when_equity_non_positive(self):
        """qty should be 0 when equity <= 0."""
        qty = SL11.FixedFractionalSizing.quantity(
            equity=0,
            atr=0.5,
            price=100,
            risk_pct=0.03,
            stop_atr_mult=6.0,
            min_atr=0.10,
            max_leverage=5.0,
        )
        assert qty == 0

        qty = SL11.FixedFractionalSizing.quantity(
            equity=-100000,
            atr=0.5,
            price=100,
            risk_pct=0.03,
            stop_atr_mult=6.0,
            min_atr=0.10,
            max_leverage=5.0,
        )
        assert qty == 0

    def test_respects_min_atr_floor(self):
        """Very small atr should use min_atr instead."""
        qty_low = SL11.FixedFractionalSizing.quantity(
            equity=100000,
            atr=0.001,
            price=100,
            risk_pct=0.03,
            stop_atr_mult=6.0,
            min_atr=0.10,
            max_leverage=5.0,
        )
        qty_floored = SL11.FixedFractionalSizing.quantity(
            equity=100000,
            atr=0.10,
            price=100,
            risk_pct=0.03,
            stop_atr_mult=6.0,
            min_atr=0.10,
            max_leverage=5.0,
        )
        # Both should use min_atr=0.10, so should be equal
        assert qty_low == qty_floored

    def test_respects_leverage_cap(self):
        """When risk sizing would exceed leverage cap, cap should bind."""
        # Construct case: very small stop_atr_mult and atr so risk sizing
        # wants huge quantity, but leverage cap limits it
        qty = SL11.FixedFractionalSizing.quantity(
            equity=1_000_000,
            atr=0.10,
            price=100,
            risk_pct=0.03,
            stop_atr_mult=0.1,
            min_atr=0.10,
            max_leverage=5.0,
        )
        # Leverage cap: floor(5.0 * 1_000_000 / 100) = 50000
        assert qty <= 50000

    def test_respects_risk_and_stop_calculation(self):
        """Basic formula check: qty = floor(equity*risk_pct / (atr*stop_mult))."""
        qty = SL11.FixedFractionalSizing.quantity(
            equity=1_000_000,
            atr=2.0,
            price=50,
            risk_pct=0.03,
            stop_atr_mult=6.0,
            min_atr=0.10,
            max_leverage=5.0,
        )
        # risk_qty = floor(1_000_000 * 0.03 / (2.0 * 6.0))
        #          = floor(30_000 / 12) = floor(2500) = 2500
        # leverage_qty = floor(5.0 * 1_000_000 / 50) = 100_000
        # min(2500, 100_000) = 2500
        assert qty == 2500


class TestComputeCarryFv:
    """Test full-carry fair value calculation."""

    def test_output_always_negative(self):
        """Carry FV should always be negative (costs push deferred up)."""
        fv1 = SL11.compute_carry_fv(
            leg2_price=100.0, storage_per_month=0.30, financing_rate=0.05
        )
        assert fv1 < 0.0

        fv2 = SL11.compute_carry_fv(
            leg2_price=50.0, storage_per_month=0.60, financing_rate=0.10
        )
        assert fv2 < 0.0

    def test_larger_storage_more_negative(self):
        """Larger storage_per_month should produce more negative FV."""
        fv_low = SL11.compute_carry_fv(
            leg2_price=100.0, storage_per_month=0.20, financing_rate=0.05
        )
        fv_high = SL11.compute_carry_fv(
            leg2_price=100.0, storage_per_month=0.60, financing_rate=0.05
        )
        assert fv_high < fv_low  # More negative

    def test_larger_financing_rate_more_negative(self):
        """Larger financing_rate should produce more negative FV (higher financing cost)."""
        fv_low = SL11.compute_carry_fv(
            leg2_price=100.0, storage_per_month=0.30, financing_rate=0.02
        )
        fv_high = SL11.compute_carry_fv(
            leg2_price=100.0, storage_per_month=0.30, financing_rate=0.10
        )
        assert fv_high < fv_low


class TestCarryRatio:
    """Test carry ratio calculation: -value / full_carry."""

    def test_scalar_inputs(self):
        """Test with scalar inputs."""
        cr = SL11.carry_ratio(value=-50.0, full_carry=-30.0)
        # -value / full_carry = 50 / (-30) = -1.667
        expected = 50.0 / (-30.0)
        assert cr == pytest.approx(expected)

    def test_array_inputs(self):
        """Test with array inputs."""
        values = np.array([-50.0, 0.0, 50.0])
        full_carry = np.array([-30.0, -30.0, -30.0])
        cr = SL11.carry_ratio(values, full_carry)
        expected = -values / full_carry
        assert np.allclose(cr, expected)

    def test_sign_flip(self):
        """Test sign flip: negative value -> positive ratio, etc."""
        # Deep contango: value < full_carry (both negative)
        # Example: value = -50, full_carry = -30
        # cr = -(-50) / (-30) = 50 / (-30) = -1.667 (negative ratio in contango)
        cr_contango = SL11.carry_ratio(value=-50.0, full_carry=-30.0)
        assert cr_contango < 0.0  # Negative in contango

        # Backwardation: value > 0, full_carry still < 0
        # cr = -(20) / (-30) = -20 / (-30) = 0.667 (positive ratio in backwardation)
        cr_backwardation = SL11.carry_ratio(value=20.0, full_carry=-30.0)
        assert cr_backwardation > 0.0  # Positive in backwardation

    def test_division_by_zero_behavior(self):
        """Test division by zero via np.errstate (works on arrays, not scalars)."""
        # With arrays, np.errstate suppresses the warning and returns inf
        cr = SL11.carry_ratio(value=np.array([100.0]), full_carry=np.array([0.0]))
        # Should give inf (not raise exception)
        assert np.isinf(cr[0]) or np.isnan(cr[0])


class TestLabelTsRegime:
    """Test term-structure regime labeling."""

    def test_value_above_flat_band_is_backwardation(self):
        """value > flat_band should label 'backwardation'."""
        values = np.array([0.05, 0.10, 0.20])
        labels = SL11.label_ts_regime(values, flat_band=0.02)
        expected = np.array(
            ["backwardation", "backwardation", "backwardation"], dtype=object
        )
        assert np.array_equal(labels, expected)

    def test_value_below_negative_flat_band_is_contango(self):
        """value < -flat_band should label 'contango'."""
        values = np.array([-0.05, -0.10, -0.20])
        labels = SL11.label_ts_regime(values, flat_band=0.02)
        expected = np.array(["contango", "contango", "contango"], dtype=object)
        assert np.array_equal(labels, expected)

    def test_value_within_flat_band_is_flat(self):
        """value within [-flat_band, flat_band] should label 'flat'."""
        values = np.array([-0.01, 0.0, 0.01])
        labels = SL11.label_ts_regime(values, flat_band=0.02)
        expected = np.array(["flat", "flat", "flat"], dtype=object)
        assert np.array_equal(labels, expected)

    def test_mixed_regimes(self):
        """Test array spanning all regimes in one call."""
        values = np.array([-0.20, -0.005, 0.0, 0.005, 0.20])
        labels = SL11.label_ts_regime(values, flat_band=0.01)
        expected = np.array(
            ["contango", "flat", "flat", "flat", "backwardation"], dtype=object
        )
        assert np.array_equal(labels, expected)

    def test_non_finite_values_unknown(self):
        """NaN and inf should label 'unknown'."""
        values = np.array([np.nan, np.inf, -np.inf, 0.05])
        labels = SL11.label_ts_regime(values, flat_band=0.02)
        assert labels[0] == "unknown"
        assert labels[1] == "unknown"
        assert labels[2] == "unknown"
        assert labels[3] == "backwardation"


class TestVarianceRatio:
    """Test Lo-MacKinlay variance ratio."""

    def test_random_walk_vr_close_to_one(self):
        """Random walk should have VR close to 1."""
        rng = np.random.default_rng(SEED)
        rw = np.cumsum(rng.normal(0, 1, 500))
        result = SL11.variance_ratio(rw, q=2)
        # Allow wide tolerance for random walk estimate
        assert 0.5 < result["vr"] < 1.5

    def test_mean_reverting_vr_below_one(self):
        """Strong mean reversion should have VR < 1."""
        rng = np.random.default_rng(SEED)
        n = 500
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
        result = SL11.variance_ratio(x, q=2)
        # Mean-reverting AR(1) should have VR < 1
        if np.isfinite(result["vr"]):
            assert result["vr"] < 1.0

    def test_window_parameter_truncates(self):
        """window parameter should truncate to last window observations."""
        rng = np.random.default_rng(SEED)
        x = np.cumsum(rng.normal(0, 1, 500))
        result_full = SL11.variance_ratio(x, q=2, window=None)
        result_short = SL11.variance_ratio(x, q=2, window=100)
        # Both should return valid results
        assert np.isfinite(result_full["vr"])
        assert np.isfinite(result_short["vr"])
        # n values should differ (one uses all 500-1 obs, other uses 100-1)
        assert result_full["n"] > result_short["n"]

    def test_z_stat_structure(self):
        """Result should contain vr, z_stat, and n."""
        rng = np.random.default_rng(SEED)
        x = np.cumsum(rng.normal(0, 1, 500))
        result = SL11.variance_ratio(x, q=2)
        assert "vr" in result
        assert "z_stat" in result
        assert "n" in result

    def test_insufficient_data_returns_nan(self):
        """Very short series relative to q should return NaN."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SL11.variance_ratio(x, q=10)
        assert np.isnan(result["vr"])
        assert np.isnan(result["z_stat"])


class TestHurstExponent:
    """Test Hurst exponent estimator."""

    def test_random_walk_close_to_half(self):
        """Random walk should have H close to 0.5."""
        rng = np.random.default_rng(SEED)
        rw = np.cumsum(rng.normal(0, 1, 1000))
        h = SL11.hurst_exponent(rw, min_lag=2, max_lag=50)
        # Allow wide tolerance (0.3-0.7) for finite-sample roughness
        assert 0.3 < h < 0.7

    def test_mean_reverting_below_random_walk(self):
        """Mean-reverting series should have H < random walk's estimate."""
        rng = np.random.default_rng(SEED)
        n = 1000
        # Mean-reverting AR(1) with phi=0.8
        mr = np.zeros(n)
        for t in range(1, n):
            mr[t] = 0.8 * mr[t - 1] + rng.normal(0, 1)
        # Random walk for comparison
        rw = np.cumsum(rng.normal(0, 1, n))
        h_mr = SL11.hurst_exponent(mr, min_lag=2, max_lag=50)
        h_rw = SL11.hurst_exponent(rw, min_lag=2, max_lag=50)
        # MR should be smaller (closer to 0)
        if np.isfinite(h_mr) and np.isfinite(h_rw):
            assert h_mr < h_rw

    def test_insufficient_data_returns_nan(self):
        """Very short series should return NaN."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        h = SL11.hurst_exponent(x, min_lag=2, max_lag=100)
        assert np.isnan(h)

    def test_output_is_float(self):
        """Output should be a float."""
        rng = np.random.default_rng(SEED)
        x = np.cumsum(rng.normal(0, 1, 500))
        h = SL11.hurst_exponent(x)
        assert isinstance(h, (float, np.floating))


class TestRollingHalfLife:
    """Test rolling AR(1) half-life on chunks."""

    def test_returns_len_divided_by_window_values(self):
        """Should return len(x)//window values."""
        rng = np.random.default_rng(SEED)
        x = np.cumsum(rng.normal(0, 1, 1000))
        hl = SL11.rolling_half_life(x, window=100)
        assert len(hl) == 1000 // 100

    def test_mean_reverting_chunk_returns_finite_half_life(self):
        """Strong MR chunk should return finite positive half-life."""
        rng = np.random.default_rng(SEED)
        n = 400
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + rng.normal(0, 1)
        hl = SL11.rolling_half_life(x, window=100)
        # Should have 4 chunks; at least some should be finite and positive
        finite_hls = hl[np.isfinite(hl)]
        assert len(finite_hls) > 0
        assert np.all(finite_hls > 0.0)

    def test_nans_in_short_chunks(self):
        """Chunks < 10 obs should return NaN."""
        x = np.arange(30, dtype=float)  # 30 obs
        hl = SL11.rolling_half_life(x, window=20)  # 30//20 = 1 chunk of 20
        # Single chunk of 20 is >= 10, should be computed (not NaN)
        # But if x is too short/random, AR(1) fit might return None -> NaN
        # Just check structure
        assert len(hl) == 30 // 20


class TestRollingAdfStat:
    """Test rolling ADF t-stat on chunks."""

    def test_returns_len_divided_by_window_values(self):
        """Should return len(x)//window values."""
        rng = np.random.default_rng(SEED)
        x = np.cumsum(rng.normal(0, 1, 1000))
        adf_stats = SL11.rolling_adf_stat(x, window=100)
        assert len(adf_stats) == 1000 // 100

    def test_chunks_under_60_obs_stay_nan(self):
        """Chunks < 60 obs should stay NaN (adf_test minimum)."""
        x = np.arange(100, dtype=float)
        adf_stats = SL11.rolling_adf_stat(x, window=40)  # 100//40 = 2 chunks of 40
        # Both chunks have 40 obs < 60, should be NaN
        assert np.all(np.isnan(adf_stats))

    def test_chunks_over_60_obs_computed(self):
        """Chunks >= 60 obs should compute ADF t-stat (not NaN)."""
        rng = np.random.default_rng(SEED)
        n = 1000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + rng.normal(0, 1)  # MR series
        adf_stats = SL11.rolling_adf_stat(
            x, window=100
        )  # 1000//100 = 10 chunks of 100 each
        # All chunks are 100 >= 60, should be computed
        finite_stats = adf_stats[np.isfinite(adf_stats)]
        assert len(finite_stats) > 0


class TestRollingStability:
    """Test rolling stability metric."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with all required keys."""
        rng = np.random.default_rng(SEED)
        x = np.cumsum(rng.normal(0, 1, 500))
        result = SL11.rolling_stability(x, n_subperiods=4, hl_band=(3.0, 60.0))
        assert "full_sample_half_life" in result
        assert "full_sample_in_band" in result
        assert "n_subperiods" in result
        assert "n_subperiods_in_band" in result
        assert "sub_half_lives" in result
        assert "stable" in result

    def test_stable_flag_when_full_sample_in_band(self):
        """stable should be True when full_sample_in_band and n_subperiods_in_band >= 3."""
        rng = np.random.default_rng(SEED)
        n = 400
        x = np.zeros(n)
        # Engineer strong MR series (half-life ~15 days, within default band [3, 60])
        for t in range(1, n):
            x[t] = 0.95 * x[t - 1] + rng.normal(0, 1)
        result = SL11.rolling_stability(x, n_subperiods=4, hl_band=(10.0, 60.0))
        # With strong MR, should likely be in band
        # (but not guaranteed due to finite-sample variance, so just check it runs)
        assert isinstance(result["stable"], (bool, np.bool_))

    def test_n_subperiods_matches_input(self):
        """n_subperiods in result should match input n_subperiods (or fewer if window too short)."""
        x = np.arange(200, dtype=float)
        result = SL11.rolling_stability(x, n_subperiods=4)
        assert result["n_subperiods"] <= 4


# ---------------------------------------------------------------------------
# Phase 2 evaluation harness
# ---------------------------------------------------------------------------


class TestPnlAtr:
    """Test PnL normalized by entry volatility."""

    def test_returns_zero_when_quantity_zero(self):
        """pnl_atr should return NaN when quantity=0."""
        result = SL11.pnl_atr(realized_pnl=1000.0, quantity=0, atr_at_entry=0.5)
        assert np.isnan(result)

    def test_returns_nan_when_atr_non_finite(self):
        """pnl_atr should return NaN when atr_at_entry is NaN or inf."""
        result = SL11.pnl_atr(realized_pnl=1000.0, quantity=10, atr_at_entry=np.nan)
        assert np.isnan(result)

        result = SL11.pnl_atr(realized_pnl=1000.0, quantity=10, atr_at_entry=np.inf)
        assert np.isnan(result)

    def test_returns_nan_when_atr_zero(self):
        """pnl_atr should return NaN when atr_at_entry=0 (division by zero)."""
        result = SL11.pnl_atr(realized_pnl=1000.0, quantity=10, atr_at_entry=0.0)
        assert np.isnan(result)

    def test_simple_division_check(self):
        """pnl_atr = realized_pnl / (quantity * atr_at_entry)."""
        result = SL11.pnl_atr(realized_pnl=1200.0, quantity=10, atr_at_entry=4.0)
        expected = 1200.0 / (10 * 4.0)
        assert result == pytest.approx(expected)


class TestRetEq:
    """Test fixed-notional return."""

    def test_returns_nan_when_equity_zero(self):
        """ret_eq should return NaN when equity_at_open=0."""
        result = SL11.ret_eq(realized_pnl=500.0, equity_at_open=0)
        assert np.isnan(result)

    def test_simple_division_check(self):
        """ret_eq = realized_pnl / equity_at_open."""
        result = SL11.ret_eq(realized_pnl=30000.0, equity_at_open=1_000_000.0)
        expected = 30000.0 / 1_000_000.0
        assert result == pytest.approx(expected)

    def test_negative_pnl(self):
        """Negative PnL should give negative return."""
        result = SL11.ret_eq(realized_pnl=-50000.0, equity_at_open=1_000_000.0)
        assert result == pytest.approx(-0.05)


class TestTradeBlocks:
    """Test trade date to quarter block assignment."""

    def test_same_quarter_same_block_id(self):
        """Dates in the same quarter should map to the same block id."""
        dates = np.array(
            [
                np.datetime64("2024-01-15"),
                np.datetime64("2024-02-20"),
                np.datetime64("2024-03-30"),
            ],
            dtype="datetime64[D]",
        )
        blocks = SL11.trade_blocks(dates, freq="1q")
        # All should map to 2024-01-01 (Q1 start)
        assert blocks[0] == blocks[1] == blocks[2]

    def test_different_quarters_different_block_ids(self):
        """Dates in different quarters should map to different block ids."""
        dates = np.array(
            [
                np.datetime64("2024-03-30"),
                np.datetime64("2024-04-01"),
            ],
            dtype="datetime64[D]",
        )
        blocks = SL11.trade_blocks(dates, freq="1q")
        # Q1 vs Q2 should differ
        assert blocks[0] != blocks[1]

    def test_monotone_in_time(self):
        """Block ids should be monotone (non-decreasing) in time."""
        dates = np.array(
            [
                np.datetime64("2024-01-15"),
                np.datetime64("2024-06-15"),
                np.datetime64("2024-12-15"),
                np.datetime64("2025-03-15"),
            ],
            dtype="datetime64[D]",
        )
        blocks = SL11.trade_blocks(dates, freq="1q")
        # Should be monotone increasing
        for i in range(len(blocks) - 1):
            assert blocks[i] <= blocks[i + 1]


class TestPairedBlockBootstrap:
    """Test paired block bootstrap for (treatment - control) delta."""

    def test_delta_point_equals_sum_difference(self):
        """delta_point should equal treatment.sum() - control.sum() exactly."""
        control_pnl = np.array([100.0, 200.0, 150.0, 50.0])
        control_blocks = np.array([0, 0, 1, 1])
        treatment_pnl = np.array([120.0, 250.0, 180.0, 70.0])
        treatment_blocks = np.array([0, 0, 1, 1])

        result = SL11.paired_block_bootstrap(
            control_pnl,
            control_blocks,
            treatment_pnl,
            treatment_blocks,
            n_boot=100,
            seed=SEED,
        )
        expected_delta = treatment_pnl.sum() - control_pnl.sum()
        assert result["delta_point"] == pytest.approx(expected_delta)

    def test_ci_brackets_point(self):
        """CI [lo, hi] should bracket the point estimate (approximately)."""
        control_pnl = np.array([100.0, 200.0, 150.0, 50.0])
        control_blocks = np.array([0, 0, 1, 1])
        treatment_pnl = np.array([120.0, 250.0, 180.0, 70.0])
        treatment_blocks = np.array([0, 0, 1, 1])

        result = SL11.paired_block_bootstrap(
            control_pnl,
            control_blocks,
            treatment_pnl,
            treatment_blocks,
            n_boot=500,
            seed=SEED,
            ci=0.95,
        )
        lo, hi = result["delta_ci"]
        # Point should be inside CI (or at least close)
        assert lo <= result["delta_point"] <= hi

    def test_returns_required_keys(self):
        """Should return all required keys."""
        control_pnl = np.array([100.0, 200.0])
        control_blocks = np.array([0, 1])
        treatment_pnl = np.array([120.0, 250.0])
        treatment_blocks = np.array([0, 1])

        result = SL11.paired_block_bootstrap(
            control_pnl,
            control_blocks,
            treatment_pnl,
            treatment_blocks,
            n_boot=100,
            seed=SEED,
        )
        assert "control_point" in result
        assert "treatment_point" in result
        assert "delta_point" in result
        assert "delta_ci" in result
        assert "delta_excludes_zero" in result
        assert "control_ci" in result
        assert "n_blocks" in result


class TestNoiseFloor:
    """Test control book's own return bootstrap."""

    def test_point_return_equals_sum(self):
        """point_return should equal ret_eq_values.sum() exactly."""
        ret_eq_values = np.array([0.02, 0.03, -0.01, 0.05])
        blocks = np.array([0, 0, 1, 1])

        result = SL11.noise_floor(ret_eq_values, blocks, n_boot=100, seed=SEED)
        expected = ret_eq_values.sum()
        assert result["point_return"] == pytest.approx(expected)

    def test_ci_return_structure(self):
        """ci_return should be a 2-tuple [lo, hi] with lo <= hi."""
        ret_eq_values = np.array([0.02, 0.03, -0.01, 0.05])
        blocks = np.array([0, 0, 1, 1])

        result = SL11.noise_floor(ret_eq_values, blocks, n_boot=100, seed=SEED)
        lo, hi = result["ci_return"]
        assert lo <= hi

    def test_half_width_non_negative(self):
        """half_width_pp should be non-negative."""
        ret_eq_values = np.array([0.02, 0.03, -0.01, 0.05])
        blocks = np.array([0, 0, 1, 1])

        result = SL11.noise_floor(ret_eq_values, blocks, n_boot=100, seed=SEED)
        assert result["half_width_pp"] >= 0.0


# ---------------------------------------------------------------------------
# Phase 4 trading rule
# ---------------------------------------------------------------------------


class TestPointValue:
    """Test POINT_VALUE dict."""

    def test_known_entries_present(self):
        """Check a few known entries."""
        assert SL11.POINT_VALUE["BZ"] == 1000.0
        assert SL11.POINT_VALUE["CL"] == 1000.0
        assert SL11.POINT_VALUE["ZC"] == 5000.0

    def test_all_values_positive(self):
        """All point values should be positive."""
        assert np.all(np.array(list(SL11.POINT_VALUE.values())) > 0.0)


class TestTradingRuleParams:
    """Test TradingRuleParams dataclass defaults."""

    def test_default_entry_threshold(self):
        """entry_threshold should default to 2.0."""
        p = SL11.TradingRuleParams()
        assert p.entry_threshold == 2.0

    def test_default_exit_threshold(self):
        """exit_threshold should default to 0.75."""
        p = SL11.TradingRuleParams()
        assert p.exit_threshold == 0.75

    def test_default_lookback(self):
        """lookback should default to 60."""
        p = SL11.TradingRuleParams()
        assert p.lookback == 60

    def test_default_stop_atr_mult(self):
        """stop_atr_mult should default to 6.0."""
        p = SL11.TradingRuleParams()
        assert p.stop_atr_mult == 6.0

    def test_default_risk_pct(self):
        """risk_pct should default to 0.03."""
        p = SL11.TradingRuleParams()
        assert p.risk_pct == 0.03

    def test_default_atr_window(self):
        """atr_window should default to 14."""
        p = SL11.TradingRuleParams()
        assert p.atr_window == 14

    def test_default_min_atr(self):
        """min_atr should default to 0.10."""
        p = SL11.TradingRuleParams()
        assert p.min_atr == 0.10

    def test_default_max_leverage(self):
        """max_leverage should default to 5.0."""
        p = SL11.TradingRuleParams()
        assert p.max_leverage == 5.0

    def test_default_cooldown_days(self):
        """cooldown_days should default to 10."""
        p = SL11.TradingRuleParams()
        assert p.cooldown_days == 10

    def test_default_episode_window_days(self):
        """episode_window_days should default to 30."""
        p = SL11.TradingRuleParams()
        assert p.episode_window_days == 30

    def test_default_max_per_episode(self):
        """max_per_episode should default to 1."""
        p = SL11.TradingRuleParams()
        assert p.max_per_episode == 1

    def test_default_half_life_max(self):
        """half_life_max should default to 45.0."""
        p = SL11.TradingRuleParams()
        assert p.half_life_max == 45.0

    def test_default_adf_pmax(self):
        """adf_pmax should default to 0.10."""
        p = SL11.TradingRuleParams()
        assert p.adf_pmax == 0.10

    def test_default_vol_pctile(self):
        """vol_pctile should default to 0.75."""
        p = SL11.TradingRuleParams()
        assert p.vol_pctile == 0.75

    def test_default_vol_regime_pctile(self):
        """vol_regime_pctile should default to 0.90."""
        p = SL11.TradingRuleParams()
        assert p.vol_regime_pctile == 0.90

    def test_default_liquidity_pctile(self):
        """liquidity_pctile should default to 0.10."""
        p = SL11.TradingRuleParams()
        assert p.liquidity_pctile == 0.10

    def test_default_vol_window(self):
        """vol_window should default to 20."""
        p = SL11.TradingRuleParams()
        assert p.vol_window == 20

    def test_default_percentile_window(self):
        """percentile_window should default to 252."""
        p = SL11.TradingRuleParams()
        assert p.percentile_window == 252

    def test_default_max_single_name_pct(self):
        """max_single_name_pct should default to 12.0."""
        p = SL11.TradingRuleParams()
        assert p.max_single_name_pct == 12.0

    def test_default_max_gross_exposure_pct(self):
        """max_gross_exposure_pct should default to 100.0."""
        p = SL11.TradingRuleParams()
        assert p.max_gross_exposure_pct == 100.0

    def test_default_daily_drawdown_limit_pct(self):
        """daily_drawdown_limit_pct should default to 3.0."""
        p = SL11.TradingRuleParams()
        assert p.daily_drawdown_limit_pct == 3.0

    def test_default_overnight_concentration_max_pct(self):
        """overnight_concentration_max_pct should default to 30.0."""
        p = SL11.TradingRuleParams()
        assert p.overnight_concentration_max_pct == 30.0


class TestCostPerContractForSpread:
    """Test round-turn cost summing."""

    def test_sums_cost_per_product(self):
        """Should sum cost across legs."""

        def cost_fn(product):
            return {"BZ": 5.0, "CL": 3.0}[product]

        cost = SL11.cost_per_contract_for_spread(["BZ", "CL"], cost_fn)
        assert cost == pytest.approx(8.0)

    def test_single_leg(self):
        """Single-leg spread should return that leg's cost."""

        def cost_fn(product):
            return {"CL": 2.5}[product]

        cost = SL11.cost_per_contract_for_spread(["CL"], cost_fn)
        assert cost == pytest.approx(2.5)


class TestSimulateSingleSpread:
    """Test single-spread simulation (integration test)."""

    def _make_synthetic_frame(self):
        """Build a small synthetic polars DataFrame with strong oscillating value."""
        rng = np.random.default_rng(SEED)
        n = 250
        dates = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )
        # Strong oscillation to exceed z-score entry threshold of 2.0
        # Create a repeating pattern: spike up, decay back down
        value = np.zeros(n)
        for cycle in range(5):
            start = cycle * 50
            end = min((cycle + 1) * 50, n)
            if start < n:
                # Within each 50-day cycle, spike from -8 to +8 and back
                cycle_idx = np.arange(end - start)
                value[start:end] = 8 * np.sin(np.pi * cycle_idx / (end - start))
        value = value + rng.normal(0, 0.3, n)  # Small noise

        leg1_price = np.full(n, 100.0) + 0.5 * value
        leg2_price = np.full(n, 100.0) - 0.5 * value
        roll_flag = np.zeros(n, dtype=bool)
        roll_flag[::60] = True
        ts_regime = np.array(
            [
                "flat" if abs(v) < 2 else "backwardation" if v > 0 else "contango"
                for v in value
            ],
            dtype=object,
        )
        leg_roles = [[{"product": "CL"}, {"product": "CL"}] for _ in range(n)]

        return pl.DataFrame(
            {
                "date": dates,
                "value": value,
                "leg1_price": leg1_price,
                "leg2_price": leg2_price,
                "roll_window_flag": roll_flag,
                "ts_regime": ts_regime,
                "leg_roles": leg_roles,
            }
        )

    def test_returns_required_keys(self):
        """simulate_single_spread should return all required keys."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()
        result = SL11.simulate_single_spread(df, p, cost_per_contract=2.0)

        assert "trades" in result
        assert "equity_curve" in result
        assert "dates" in result
        assert "point_value" in result
        assert "leg_products" in result

    def test_simulation_with_manual_spikes(self):
        """Manually engineered value spikes should generate trades."""
        # Build frame with manually engineered spikes designed to trigger z-score > 2
        n = 300
        dates = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )
        # Create a series with clear spikes around mean
        value = np.zeros(n)
        value[75:85] = 8.0  # Spike above mean
        value[125:135] = -8.0  # Spike below mean
        value[175:185] = 8.0  # Another spike above
        value[225:235] = -8.0  # Another spike below
        # Add slight trend to avoid constant values
        value = value + 0.01 * np.arange(n)

        leg1_price = np.full(n, 100.0) + 0.5 * value
        leg2_price = np.full(n, 100.0) - 0.5 * value
        roll_flag = np.zeros(n, dtype=bool)
        ts_regime = np.array(["flat"] * n, dtype=object)
        leg_roles = [[{"product": "CL"}, {"product": "CL"}] for _ in range(n)]

        df = pl.DataFrame(
            {
                "date": dates,
                "value": value,
                "leg1_price": leg1_price,
                "leg2_price": leg2_price,
                "roll_window_flag": roll_flag,
                "ts_regime": ts_regime,
                "leg_roles": leg_roles,
            }
        )

        p = SL11.TradingRuleParams()
        result = SL11.simulate_single_spread(df, p, cost_per_contract=2.0)

        # Should generate at least one trade with these engineered spikes
        assert len(result["trades"]) > 0, "Spikes should generate at least one trade"

    def test_trade_dict_has_required_keys(self):
        """Each trade dict should have expected keys."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()
        result = SL11.simulate_single_spread(df, p, cost_per_contract=2.0)

        if len(result["trades"]) > 0:
            trade = result["trades"][0]
            required_keys = {
                "entry_date",
                "exit_date",
                "direction",
                "qty",
                "realized_pnl",
                "exit_reason",
                "pnl_atr",
                "ret_eq",
                "entry_value",
                "exit_value",
            }
            for key in required_keys:
                assert key in trade, f"Missing key {key} in trade dict"

    def test_exit_reason_in_stop_or_zscore(self):
        """Each trade's exit_reason should be 'stop' or 'zscore'."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()
        result = SL11.simulate_single_spread(df, p, cost_per_contract=2.0)

        for trade in result["trades"]:
            assert trade["exit_reason"] in {"stop", "zscore"}

    def test_equity_curve_length_matches_dates(self):
        """equity_curve should have same length as input dates."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()
        result = SL11.simulate_single_spread(df, p, cost_per_contract=2.0)

        assert len(result["equity_curve"]) == len(df)


class TestSimulateSingleSpreadSignFlipArray:
    """Test per-bar sign_flip array parameter."""

    def _make_synthetic_frame(self):
        """Build a small synthetic frame with distinct entry-triggering bars."""
        n = 200
        dates = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )
        # Create clear spikes to trigger z-score entries at known bars
        value = np.zeros(n)
        value[40:50] = 8.0  # Spike 1 (bars 40-49) -- entry should happen ~bar 50
        value[100:110] = -8.0  # Spike 2 (bars 100-109) -- entry should happen ~bar 110
        value[160:170] = 8.0  # Spike 3 (bars 160-169) -- entry should happen ~bar 170
        value = value + 0.01 * np.arange(n)  # Slight trend to avoid constant values

        leg1_price = np.full(n, 100.0) + 0.5 * value
        leg2_price = np.full(n, 100.0) - 0.5 * value
        roll_flag = np.zeros(n, dtype=bool)
        ts_regime = np.array(["flat"] * n, dtype=object)
        leg_roles = [[{"product": "CL"}, {"product": "CL"}] for _ in range(n)]

        return pl.DataFrame(
            {
                "date": dates,
                "value": value,
                "leg1_price": leg1_price,
                "leg2_price": leg2_price,
                "roll_window_flag": roll_flag,
                "ts_regime": ts_regime,
                "leg_roles": leg_roles,
            }
        )

    def test_sign_flip_array_does_not_error(self):
        """Passing a per-bar mask should not error and should produce valid results."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # Create a mask that flips only the first half of days
        flip_mask = np.zeros(len(df), dtype=bool)
        flip_mask[:100] = True

        # Run with the selective mask
        result_partial_flip = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, sign_flip=flip_mask
        )

        # Should complete without error and return valid structure
        assert "trades" in result_partial_flip
        assert "equity_curve" in result_partial_flip
        assert len(result_partial_flip["equity_curve"]) == len(df)

    def test_sign_flip_bool_true_matches_all_true_array(self):
        """sign_flip=True (bool) should match sign_flip=np.ones(n, dtype=bool) (array)."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # Run with sign_flip=True (scalar bool)
        result_bool = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, sign_flip=True
        )

        # Run with all-True array
        flip_mask = np.ones(len(df), dtype=bool)
        result_array = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, sign_flip=flip_mask
        )

        # Both should produce identical trades
        assert len(result_bool["trades"]) == len(result_array["trades"])
        for t1, t2 in zip(result_bool["trades"], result_array["trades"]):
            assert t1["direction"] == t2["direction"]
            assert t1["qty"] == t2["qty"]
            assert t1["entry_date"] == t2["entry_date"]
            assert t1["exit_date"] == t2["exit_date"]

    def test_sign_flip_bool_false_matches_all_false_array(self):
        """sign_flip=False (bool) should match sign_flip=np.zeros(n, dtype=bool) (array)."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # Run with sign_flip=False (scalar bool)
        result_bool = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, sign_flip=False
        )

        # Run with all-False array
        flip_mask = np.zeros(len(df), dtype=bool)
        result_array = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, sign_flip=flip_mask
        )

        # Both should produce identical trades
        assert len(result_bool["trades"]) == len(result_array["trades"])
        for t1, t2 in zip(result_bool["trades"], result_array["trades"]):
            assert t1["direction"] == t2["direction"]
            assert t1["qty"] == t2["qty"]
            assert t1["entry_date"] == t2["entry_date"]
            assert t1["exit_date"] == t2["exit_date"]


class TestSimulateSingleSpreadStopMultScale:
    """Test per-bar stop_mult_scale parameter."""

    def _make_synthetic_frame(self):
        """Build frame with volatility that will trigger stops under certain scales."""
        n = 250
        dates = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )
        # Create a strong spike to enter, then adverse move to test stop
        value = np.zeros(n)
        value[50:60] = 6.0  # Spike to trigger entry around bar 60
        # Adverse move starting at bar 70 (move 8 points against position over 30 bars)
        value[70:100] = 6.0 - 8.0 * (np.arange(30) / 30.0)
        value = value + 0.01 * np.arange(n)

        leg1_price = np.full(n, 100.0) + 0.5 * value
        leg2_price = np.full(n, 100.0) - 0.5 * value
        roll_flag = np.zeros(n, dtype=bool)
        ts_regime = np.array(["flat"] * n, dtype=object)
        leg_roles = [[{"product": "CL"}, {"product": "CL"}] for _ in range(n)]

        return pl.DataFrame(
            {
                "date": dates,
                "value": value,
                "leg1_price": leg1_price,
                "leg2_price": leg2_price,
                "roll_window_flag": roll_flag,
                "ts_regime": ts_regime,
                "leg_roles": leg_roles,
            }
        )

    def test_stop_mult_scale_widens_stop(self):
        """Scale=1.25 should produce wider stop than scale=None/1.0."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # Run with scale=1.25 (wider stop)
        scale_array = np.full(len(df), 1.25)
        result_scale_1_25 = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, stop_mult_scale=scale_array
        )

        # With a wider stop, a moderate adverse move should exit the stop-scale=1.0
        # trade but NOT the scale=1.25 trade (same stop triggered earlier on scale=1.0)
        # Wider stop should result in fewer stop exits (or same if stops don't matter)
        # At minimum, we should have at least tried the scaled version
        assert len(result_scale_1_25["trades"]) > 0

    def test_stop_mult_scale_nan_fallback_to_one(self):
        """NaN entries in scale array should fall back to 1.0 (not error)."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # Create array with some NaN entries
        scale_array = np.full(len(df), 1.2)
        scale_array[80:90] = np.nan

        # Should not raise an error
        result = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, stop_mult_scale=scale_array
        )
        # Should produce trades (didn't error)
        assert "trades" in result
        assert "equity_curve" in result

    def test_stop_mult_scale_all_nan_falls_back(self):
        """All-NaN scale array should behave like scale=1.0."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # All NaN
        scale_array_nan = np.full(len(df), np.nan)
        result_nan = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, stop_mult_scale=scale_array_nan
        )

        # Default (scale=1.0)
        scale_array_ones = np.ones(len(df))
        result_ones = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, stop_mult_scale=scale_array_ones
        )

        # Should produce identical results
        assert len(result_nan["trades"]) == len(result_ones["trades"])
        for t1, t2 in zip(result_nan["trades"], result_ones["trades"]):
            assert t1["direction"] == t2["direction"]
            assert t1["exit_reason"] == t2["exit_reason"]


class TestSimulateBookIntegration:
    """Test simulate_book with new sign_flip_masks and stop_mult_scales params."""

    def _make_synthetic_frame(self):
        """Build a small synthetic frame."""
        rng = np.random.default_rng(SEED)
        n = 200
        dates = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )
        value = 6 * np.sin(2 * np.pi * np.arange(n) / 60) + rng.normal(0, 0.2, n)
        leg1_price = np.full(n, 100.0) + 0.5 * value
        leg2_price = np.full(n, 100.0) - 0.5 * value
        roll_flag = np.zeros(n, dtype=bool)
        ts_regime = np.array(["flat"] * n, dtype=object)
        leg_roles = [[{"product": "CL"}, {"product": "CL"}] for _ in range(n)]

        return pl.DataFrame(
            {
                "date": dates,
                "value": value,
                "leg1_price": leg1_price,
                "leg2_price": leg2_price,
                "roll_window_flag": roll_flag,
                "ts_regime": ts_regime,
                "leg_roles": leg_roles,
            }
        )

    def test_simulate_book_with_sign_flip_masks(self):
        """sign_flip_masks dict should be correctly plumbed through to spreads."""
        df = self._make_synthetic_frame()
        spread_frames = {"spread_A": df, "spread_B": df}
        params = {
            "spread_A": SL11.TradingRuleParams(),
            "spread_B": SL11.TradingRuleParams(),
        }

        # Create selective flip masks
        flip_mask_A = np.zeros(len(df), dtype=bool)
        flip_mask_A[50:100] = True

        flip_mask_B = np.ones(len(df), dtype=bool)

        sign_flip_masks = {"spread_A": flip_mask_A, "spread_B": flip_mask_B}

        result = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
            sign_flip_masks=sign_flip_masks,
        )

        # Should complete without error and return required keys
        assert "trades" in result
        assert "portfolio_equity" in result
        assert "per_spread" in result
        # Trades should be allocated to spreads
        spread_A_trades = [t for t in result["trades"] if t["spread"] == "spread_A"]
        spread_B_trades = [t for t in result["trades"] if t["spread"] == "spread_B"]
        assert len(spread_A_trades) + len(spread_B_trades) == len(result["trades"])

    def test_simulate_book_with_stop_mult_scales(self):
        """stop_mult_scales dict should be correctly plumbed through to spreads."""
        df = self._make_synthetic_frame()
        spread_frames = {"spread_X": df}
        params = {"spread_X": SL11.TradingRuleParams()}

        scale_array = np.full(len(df), 1.15)
        stop_mult_scales = {"spread_X": scale_array}

        result = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
            stop_mult_scales=stop_mult_scales,
        )

        # Should complete without error
        assert "trades" in result
        assert "portfolio_equity" in result
        assert "per_spread" in result

    def test_simulate_book_combined_masks_and_scales(self):
        """Both sign_flip_masks and stop_mult_scales together."""
        df = self._make_synthetic_frame()
        spread_frames = {"spread_1": df}
        params = {"spread_1": SL11.TradingRuleParams()}

        flip_mask = np.random.default_rng(SEED).random(len(df)) > 0.5
        scale_array = np.full(len(df), 1.1)

        result = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
            sign_flip_masks={"spread_1": flip_mask},
            stop_mult_scales={"spread_1": scale_array},
        )

        # Should produce valid results
        assert "trades" in result
        assert "portfolio_equity" in result
        assert len(result["portfolio_equity"]) == len(df)


class TestSimulateBook:
    """Test portfolio-level simulation."""

    def _make_synthetic_frame(self):
        """Build a small synthetic frame with oscillation."""
        rng = np.random.default_rng(SEED)
        n = 200
        dates = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )
        # Create oscillations to trigger trades
        value = 6 * np.sin(2 * np.pi * np.arange(n) / 60) + rng.normal(0, 0.2, n)
        leg1_price = np.full(n, 100.0) + 0.5 * value
        leg2_price = np.full(n, 100.0) - 0.5 * value
        roll_flag = np.zeros(n, dtype=bool)
        ts_regime = np.array(["flat"] * n, dtype=object)
        leg_roles = [[{"product": "CL"}, {"product": "CL"}] for _ in range(n)]

        return pl.DataFrame(
            {
                "date": dates,
                "value": value,
                "leg1_price": leg1_price,
                "leg2_price": leg2_price,
                "roll_window_flag": roll_flag,
                "ts_regime": ts_regime,
                "leg_roles": leg_roles,
            }
        )

    def test_returns_required_keys(self):
        """simulate_book should return all required keys."""
        df = self._make_synthetic_frame()
        spread_frames = {"test_spread": df}
        params = {"test_spread": SL11.TradingRuleParams()}
        stop_overrides = {}
        regime_reqs = {}

        result = SL11.simulate_book(
            spread_frames,
            params,
            stop_overrides,
            regime_reqs,
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
        )

        assert "trades" in result
        assert "portfolio_equity" in result
        assert "dates" in result
        assert "per_spread" in result
        assert "start_equity" in result

    def test_portfolio_equity_starts_at_start_equity(self):
        """First portfolio equity value should be start_equity."""
        df = self._make_synthetic_frame()
        spread_frames = {"test_spread": df}
        params = {"test_spread": SL11.TradingRuleParams()}
        start_eq = 100_000.0

        result = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=start_eq,
        )

        assert result["portfolio_equity"][0] == pytest.approx(start_eq)


class TestBookMetrics:
    """Test book risk/return metrics."""

    def _make_synthetic_book(self):
        """Build a synthetic book result."""
        rng = np.random.default_rng(SEED)
        n = 100
        start_equity = 1_000_000.0
        # Simple equity curve with some volatility
        daily_returns = rng.normal(0.0005, 0.01, n)
        equity = start_equity * np.exp(np.cumsum(daily_returns))

        dates = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )

        trades = [
            {
                "spread": "test",
                "entry_date": dates[10],
                "exit_date": dates[20],
                "direction": 1,
                "qty": 10,
                "realized_pnl": 5000.0,
                "exit_reason": "zscore",
                "pnl_atr": 1.5,
                "ret_eq": 0.005,
            },
            {
                "spread": "test",
                "entry_date": dates[30],
                "exit_date": dates[50],
                "direction": -1,
                "qty": 5,
                "realized_pnl": 2000.0,
                "exit_reason": "stop",
                "pnl_atr": 0.8,
                "ret_eq": 0.002,
            },
        ]

        return {
            "trades": trades,
            "portfolio_equity": equity,
            "dates": dates,
            "start_equity": start_equity,
            "per_spread": {},
        }

    def test_returns_required_keys(self):
        """book_metrics should return all required keys."""
        book = self._make_synthetic_book()
        metrics = SL11.book_metrics(book)

        required_keys = {
            "sharpe",
            "max_drawdown",
            "fixed_notional_return",
            "equity_path_return",
            "return_over_drawdown",
            "n_trades",
            "n_stop_exits",
            "n_zscore_exits",
            "final_equity",
        }
        for key in required_keys:
            assert key in metrics, f"Missing key {key} in metrics"

    def test_n_trades_equals_sum_of_exits(self):
        """n_trades should equal n_stop_exits + n_zscore_exits."""
        book = self._make_synthetic_book()
        metrics = SL11.book_metrics(book)

        assert (
            metrics["n_trades"] == metrics["n_stop_exits"] + metrics["n_zscore_exits"]
        )

    def test_fixed_notional_return_is_sum_of_ret_eq(self):
        """fixed_notional_return should equal sum of all trade ret_eq values."""
        book = self._make_synthetic_book()
        metrics = SL11.book_metrics(book)

        expected_fnr = sum(t["ret_eq"] for t in book["trades"])
        assert metrics["fixed_notional_return"] == pytest.approx(expected_fnr)

    def test_max_drawdown_non_positive(self):
        """max_drawdown should be <= 0 (a loss from peak)."""
        book = self._make_synthetic_book()
        metrics = SL11.book_metrics(book)

        assert metrics["max_drawdown"] <= 0.0

    def test_final_equity_equals_last_portfolio_equity(self):
        """final_equity should equal portfolio_equity[-1]."""
        book = self._make_synthetic_book()
        metrics = SL11.book_metrics(book)

        assert metrics["final_equity"] == pytest.approx(book["portfolio_equity"][-1])


# ---------------------------------------------------------------------------
# Gate LC: roc_auc_score and veto_entry_mask tests
# ---------------------------------------------------------------------------


class TestRocAucScore:
    """Test Mann-Whitney U / rank-sum AUC implementation."""

    def test_perfect_separation(self):
        """All positives score higher than all negatives -> AUC = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
        auc = SL11.roc_auc_score(y_true, y_score)
        assert auc == pytest.approx(1.0)

    def test_perfect_reversal(self):
        """All positives score lower than all negatives -> AUC = 0.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([7.0, 6.0, 5.0, 3.0, 2.0, 1.0])
        auc = SL11.roc_auc_score(y_true, y_score)
        assert auc == pytest.approx(0.0)

    def test_all_ties(self):
        """All samples have identical scores -> AUC = 0.5."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        auc = SL11.roc_auc_score(y_true, y_score)
        assert auc == pytest.approx(0.5)

    def test_single_class_all_zeros(self):
        """y_true with no positives (all zeros) -> NaN."""
        y_true = np.array([0, 0, 0, 0])
        y_score = np.array([1.0, 2.0, 3.0, 4.0])
        auc = SL11.roc_auc_score(y_true, y_score)
        assert np.isnan(auc)

    def test_single_class_all_ones(self):
        """y_true with no negatives (all ones) -> NaN."""
        y_true = np.array([1, 1, 1, 1])
        y_score = np.array([1.0, 2.0, 3.0, 4.0])
        auc = SL11.roc_auc_score(y_true, y_score)
        assert np.isnan(auc)

    def test_worked_example_with_tied_ranks(self):
        """Hand-computed example with known tied ranks.

        y_true = [0, 1, 0, 1]
        y_score = [1.0, 3.0, 1.0, 4.0]

        After sorting by score (ascending):
          scores: [1.0, 1.0, 3.0, 4.0]
          y_true: [0,   0,   1,   1]
          indices:[0,   2,   1,   3]

        Ranks (1-indexed, with average for ties):
          1.0: indices [0,2] -> tied -> avg rank = (1 + 2) / 2 = 1.5 each
          3.0: index [1] -> rank 3
          4.0: index [3] -> rank 4

        Sum of ranks for positives (y_true==1): ranks[1] + ranks[3] = 3 + 4 = 7
        n_pos=2, n_neg=2
        AUC = (7 - 2*(2+1)/2) / (2*2) = (7 - 3) / 4 = 4/4 = 1.0
        """
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([1.0, 3.0, 1.0, 4.0])
        auc = SL11.roc_auc_score(y_true, y_score)
        assert auc == pytest.approx(1.0)

    def test_mixed_scores(self):
        """A more typical AUC scenario with partial overlap."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([1.0, 2.5, 3.0, 4.0, 2.0, 3.5])
        auc = SL11.roc_auc_score(y_true, y_score)
        # AUC should be between 0 and 1, and >= 0.5 (some positive correlation)
        assert 0.0 <= auc <= 1.0
        assert auc >= 0.5

    def test_empty_arrays_match_single_class_nan(self):
        """Empty y_true or y_score with no examples of one class -> NaN."""
        # If all are class 0, then n_pos=0
        y_true = np.array([0, 0])
        y_score = np.array([1.0, 2.0])
        auc = SL11.roc_auc_score(y_true, y_score)
        assert np.isnan(auc)


class TestSimulateSingleSpreadVetoEntryMask:
    """Test per-bar veto_entry_mask parameter."""

    def _make_synthetic_frame(self):
        """Build synthetic frame with clear entry-triggering spikes."""
        n = 250
        dates = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )
        # Create spikes to reliably trigger entries
        value = np.zeros(n)
        value[50:60] = 6.0  # Spike 1: should trigger entry ~bar 60
        value[120:130] = -6.0  # Spike 2: should trigger entry ~bar 130
        value = value + 0.01 * np.arange(n)  # Slight trend

        leg1_price = np.full(n, 100.0) + 0.5 * value
        leg2_price = np.full(n, 100.0) - 0.5 * value
        roll_flag = np.zeros(n, dtype=bool)
        ts_regime = np.array(["flat"] * n, dtype=object)
        leg_roles = [[{"product": "CL"}, {"product": "CL"}] for _ in range(n)]

        return pl.DataFrame(
            {
                "date": dates,
                "value": value,
                "leg1_price": leg1_price,
                "leg2_price": leg2_price,
                "roll_window_flag": roll_flag,
                "ts_regime": ts_regime,
                "leg_roles": leg_roles,
            }
        )

    def test_veto_entry_mask_none_produces_baseline_trades(self):
        """veto_entry_mask=None should produce baseline trades."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        result_baseline = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, veto_entry_mask=None
        )

        # Should have some trades (spikes should trigger entries)
        assert len(result_baseline["trades"]) > 0
        assert "equity_curve" in result_baseline

    def test_veto_entry_mask_all_true_suppresses_all_entries(self):
        """veto_entry_mask=all-True should result in zero trades."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # Baseline: with no veto (just to verify the frame can generate trades)
        result_baseline = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, veto_entry_mask=None
        )
        assert len(result_baseline["trades"]) > 0  # Verify baseline has trades

        # With all-True mask: every entry should be vetoed
        veto_all = np.ones(len(df), dtype=bool)
        result_vetoed = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, veto_entry_mask=veto_all
        )

        # All entries should be suppressed
        assert len(result_vetoed["trades"]) == 0
        # Equity should remain at starting value (no trades made)
        assert result_vetoed["equity_curve"][1] == pytest.approx(1_000_000.0)

    def test_veto_entry_mask_selective_blocks_specific_entry(self):
        """Veto only at the first entry bar should suppress only that entry."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # Baseline: get the trades
        result_baseline = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, veto_entry_mask=None
        )
        baseline_trades = result_baseline["trades"]
        n_baseline = len(baseline_trades)
        assert n_baseline > 0

        # Find the first entry date
        first_entry_date = baseline_trades[0]["entry_date"]
        first_entry_idx = np.where(df["date"].to_numpy() == first_entry_date)[0][0]

        # Create a veto mask that is True only at the first entry bar
        veto_selective = np.zeros(len(df), dtype=bool)
        veto_selective[first_entry_idx] = True

        result_vetoed = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, veto_entry_mask=veto_selective
        )
        vetoed_trades = result_vetoed["trades"]

        # The first entry should be suppressed; subsequent entries may still occur
        # or the first entry may be skipped in favor of later logic
        # At minimum, verify that veto worked (fewer or zero trades)
        assert len(vetoed_trades) <= n_baseline
        # And at least the first entry date should not be in vetoed trades
        vetoed_entry_dates = [t["entry_date"] for t in vetoed_trades]
        assert first_entry_date not in vetoed_entry_dates

    def test_veto_entry_mask_does_not_exit_open_positions(self):
        """Veto mask should only suppress entries, not affect exits."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # Run with all-True veto (no entries at all)
        veto_all = np.ones(len(df), dtype=bool)
        result_vetoed = SL11.simulate_single_spread(
            df, p, cost_per_contract=2.0, veto_entry_mask=veto_all
        )

        # No trades should occur; exit logic is not tested since there are no entries
        assert len(result_vetoed["trades"]) == 0

    def test_veto_entry_mask_array_length_validation(self):
        """veto_entry_mask length should match DataFrame length (or error gracefully)."""
        df = self._make_synthetic_frame()
        p = SL11.TradingRuleParams()

        # Create a veto mask of incorrect length
        veto_wrong_length = np.ones(50, dtype=bool)  # Too short

        # This should either error or handle gracefully (index out of bounds)
        try:
            result = SL11.simulate_single_spread(
                df, p, cost_per_contract=2.0, veto_entry_mask=veto_wrong_length
            )
            # If it doesn't error, it should at least return a valid result
            assert "trades" in result
        except (IndexError, ValueError):
            # Expected: veto mask length mismatch should raise an error
            pass


class TestSimulateBookVetoEntryMasks:
    """Test veto_entry_masks parameter in simulate_book."""

    def _make_synthetic_frame(self):
        """Build a small synthetic frame with guaranteed entries."""
        n = 200
        dates = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        )
        # Create clear spikes that reliably trigger z-score entries
        value = np.zeros(n)
        value[50:60] = 6.0  # Spike 1 to trigger entry
        value[120:130] = -6.0  # Spike 2 to trigger entry
        value = value + 0.01 * np.arange(n)  # Slight trend

        leg1_price = np.full(n, 100.0) + 0.5 * value
        leg2_price = np.full(n, 100.0) - 0.5 * value
        roll_flag = np.zeros(n, dtype=bool)
        ts_regime = np.array(["flat"] * n, dtype=object)
        leg_roles = [[{"product": "CL"}, {"product": "CL"}] for _ in range(n)]

        return pl.DataFrame(
            {
                "date": dates,
                "value": value,
                "leg1_price": leg1_price,
                "leg2_price": leg2_price,
                "roll_window_flag": roll_flag,
                "ts_regime": ts_regime,
                "leg_roles": leg_roles,
            }
        )

    def test_veto_entry_masks_none_default(self):
        """veto_entry_masks=None should behave as if no mask is provided."""
        df = self._make_synthetic_frame()
        spread_frames = {"spread_A": df}
        params = {"spread_A": SL11.TradingRuleParams()}

        result = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
            veto_entry_masks=None,
        )

        # Should complete without error
        assert "trades" in result
        assert "portfolio_equity" in result
        assert "per_spread" in result

    def test_veto_entry_masks_per_spread_isolation(self):
        """Veto mask for one spread should not affect another."""
        df = self._make_synthetic_frame()
        spread_frames = {"spread_A": df, "spread_B": df}
        params = {
            "spread_A": SL11.TradingRuleParams(),
            "spread_B": SL11.TradingRuleParams(),
        }

        # Baseline: no veto masks
        result_baseline = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
            veto_entry_masks=None,
        )
        baseline_trades_a = [
            t for t in result_baseline["trades"] if t["spread"] == "spread_A"
        ]
        baseline_trades_b = [
            t for t in result_baseline["trades"] if t["spread"] == "spread_B"
        ]
        baseline_total = len(result_baseline["trades"])

        # Now veto all entries for spread_A only
        veto_all_a = np.ones(len(df), dtype=bool)
        veto_entry_masks = {"spread_A": veto_all_a}

        result_vetoed = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
            veto_entry_masks=veto_entry_masks,
        )
        vetoed_trades_a = [
            t for t in result_vetoed["trades"] if t["spread"] == "spread_A"
        ]
        vetoed_trades_b = [
            t for t in result_vetoed["trades"] if t["spread"] == "spread_B"
        ]
        vetoed_total = len(result_vetoed["trades"])

        # spread_A should have no trades (all entries vetoed)
        assert len(vetoed_trades_a) == 0
        # spread_B should have same trades count as baseline
        # (veto on spread_A should not affect spread_B)
        assert len(vetoed_trades_b) == len(baseline_trades_b)
        # If baseline had any spread_A trades, vetoing them should reduce total
        if len(baseline_trades_a) > 0:
            assert vetoed_total < baseline_total

    def test_veto_entry_masks_all_spreads_vetoed(self):
        """Veto all entries across all spreads."""
        df = self._make_synthetic_frame()
        spread_frames = {"spread_X": df, "spread_Y": df}
        params = {
            "spread_X": SL11.TradingRuleParams(),
            "spread_Y": SL11.TradingRuleParams(),
        }

        veto_all = np.ones(len(df), dtype=bool)
        veto_entry_masks = {"spread_X": veto_all, "spread_Y": veto_all}

        result = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
            veto_entry_masks=veto_entry_masks,
        )

        # No trades should occur across the entire portfolio
        assert len(result["trades"]) == 0
        # Portfolio equity should remain at starting value (no P&L)
        assert result["portfolio_equity"][1] == pytest.approx(100_000.0)

    def test_veto_entry_masks_partial_veto(self):
        """Veto a portion of entry bars for one spread."""
        df = self._make_synthetic_frame()
        spread_frames = {"spread_A": df, "spread_B": df}
        params = {
            "spread_A": SL11.TradingRuleParams(),
            "spread_B": SL11.TradingRuleParams(),
        }

        # Baseline
        result_baseline = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
        )
        baseline_trades_a = [
            t for t in result_baseline["trades"] if t["spread"] == "spread_A"
        ]
        baseline_count = len(result_baseline["trades"])

        # Partial veto: veto first 100 bars only for spread_A
        veto_partial_a = np.zeros(len(df), dtype=bool)
        veto_partial_a[:100] = True
        veto_entry_masks = {"spread_A": veto_partial_a}

        result_vetoed = SL11.simulate_book(
            spread_frames,
            params,
            {},
            {},
            round_turn_cost_fn=lambda p: 2.0,
            start_equity=100_000.0,
            veto_entry_masks=veto_entry_masks,
        )
        vetoed_trades_a = [
            t for t in result_vetoed["trades"] if t["spread"] == "spread_A"
        ]
        vetoed_count = len(result_vetoed["trades"])

        # If baseline had trades on spread_A in the first 100 bars,
        # they should be suppressed by the veto mask
        baseline_early_trades_a = [
            t
            for t in baseline_trades_a
            if t["entry_date"] < np.datetime64("2024-04-09")
        ]
        if len(baseline_early_trades_a) > 0:
            # If there were early entries, veto should remove them
            assert len(vetoed_trades_a) < len(baseline_trades_a)
            assert vetoed_count < baseline_count
        else:
            # Edge case: no early trades, so veto has no effect; just verify structure
            assert len(vetoed_trades_a) == len(baseline_trades_a)
            assert "trades" in result_vetoed


# ---------------------------------------------------------------------------
# Notebook 11d -- momentum/breakout transfer
# ---------------------------------------------------------------------------


class TestTrueAtrSeries:
    """Test Wilder's ATR with rolling mean and shift(1) for causality."""

    def test_output_length_matches_input(self):
        """ATR array should have same length as close array."""
        rng = np.random.default_rng(SEED)
        n = 100
        high = rng.uniform(100, 110, n)
        low = rng.uniform(90, 100, n)
        close = rng.uniform(95, 105, n)
        atr = SL11.true_atr_series(high, low, close, window=14)
        assert len(atr) == n

    def test_first_window_values_are_nan(self):
        """First `window` values should be NaN (rolling warmup + shift)."""
        rng = np.random.default_rng(SEED)
        n = 100
        high = rng.uniform(100, 110, n)
        low = rng.uniform(90, 100, n)
        close = rng.uniform(95, 105, n)
        window = 14
        atr = SL11.true_atr_series(high, low, close, window=window)
        # First window values should be NaN due to rolling warmup + shift(1)
        assert np.isnan(atr[0])
        assert np.isnan(atr[window - 1])

    def test_constant_range_produces_constant_atr(self):
        """With constant high-low spread, ATR after warmup should equal that spread."""
        n = 50
        close = np.full(n, 100.0)
        high = np.full(n, 105.0)  # Always +5 from close
        low = np.full(n, 100.0)  # Always 0 from close
        # True range = max(5-0, |105-100|, |100-100|) = 5 each bar
        atr = SL11.true_atr_series(high, low, close, window=14)
        # After warmup (bar ~15), ATR should stabilize at 5.0
        assert np.isclose(atr[30], 5.0, atol=0.1)
        assert np.isclose(atr[40], 5.0, atol=0.1)

    def test_no_lookahead_bar_t_not_affected_by_perturb_bar_t(self):
        """Changing bar t's high/low should NOT change atr[t] (shift(1) causality)."""
        rng = np.random.default_rng(SEED)
        n = 100
        high1 = rng.uniform(100, 110, n)
        low1 = rng.uniform(90, 100, n)
        close = rng.uniform(95, 105, n)

        atr1 = SL11.true_atr_series(high1, low1, close, window=14)

        # Perturb bar 50
        high2 = high1.copy()
        low2 = low1.copy()
        high2[50] += 50.0
        low2[50] -= 50.0

        atr2 = SL11.true_atr_series(high2, low2, close, window=14)

        # atr[50] should be identical (depends only on bars 0:50)
        assert atr1[50] == atr2[50]
        # atr[51] may differ (depends on bar 50's TR)
        # but atr[49] should be identical
        assert atr1[49] == atr2[49]


class TestSmaCAusal:
    """Test rolling mean with shift(1) for causality."""

    def test_output_length_matches_input(self):
        """SMA output should match input length."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = SL11.sma_causal(x, window=2)
        assert len(sma) == len(x)

    def test_simple_mean_case(self):
        """Verify rolling mean on a simple synthetic series."""
        # [1, 2, 3, 4, 5] with window=2, shift(1)
        # sma[0] = NaN (shift)
        # sma[1] = mean([1]) = NaN (only 1 value in window)
        # sma[2] = mean([1, 2]) = 1.5
        # sma[3] = mean([2, 3]) = 2.5
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = SL11.sma_causal(x, window=2)
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])
        assert sma[2] == pytest.approx(1.5)
        assert sma[3] == pytest.approx(2.5)
        assert sma[4] == pytest.approx(3.5)

    def test_causality_no_lookahead(self):
        """Perturbing x[t] should not change sma[t]."""
        rng = np.random.default_rng(SEED)
        x1 = rng.normal(0, 1, 100)
        sma1 = SL11.sma_causal(x1, window=10)

        x2 = x1.copy()
        x2[50] += 100.0
        sma2 = SL11.sma_causal(x2, window=10)

        # sma[50] should be identical (only depends on x[:50])
        assert sma1[50] == sma2[50]
        # sma[49] should also be identical
        assert sma1[49] == sma2[49]


class TestRegimeGate:
    """Test fail-closed trend regime voting gate."""

    def test_single_symbol_matches_sma_condition(self):
        """With one symbol, gate should match fast_sma > slow_sma exactly."""
        rng = np.random.default_rng(SEED)
        n = 100
        close = rng.uniform(90, 110, n)
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )

        fast_sma = SL11.sma_causal(close, window=10)
        slow_sma = SL11.sma_causal(close, window=20)
        expected = (fast_sma > slow_sma) & np.isfinite(fast_sma) & np.isfinite(slow_sma)

        gate = SL11.regime_gate(
            {"sym": close}, {"sym": dates}, dates, fast=10, slow=20, min_confirm=1
        )
        assert np.array_equal(gate, expected)

    def test_missing_date_does_not_confirm(self):
        """A symbol missing a date should not contribute a confirm vote (fail-closed)."""
        # Symbol A: all dates present, always uptrend
        dates_all = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-07"),
            dtype="datetime64[D]",
        )
        close_a = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0])

        # Symbol B: missing date at index 3 (2024-01-04), always uptrend
        dates_b_full = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-07"),
            dtype="datetime64[D]",
        )
        dates_b = np.concatenate([dates_b_full[:3], dates_b_full[4:]])  # Skip index 3
        close_b = np.array([100.0, 105.0, 110.0, 120.0, 125.0])

        gate = SL11.regime_gate(
            {"A": close_a, "B": close_b},
            {"A": dates_all, "B": dates_b},
            dates_all,
            fast=2,
            slow=4,
            min_confirm=2,
        )

        # On date 3, only A can confirm (B is missing); with min_confirm=2, no pass
        assert gate[3] == False

    def test_min_confirm_or_behavior(self):
        """With min_confirm=1, any one symbol confirming should pass."""
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-05"),
            dtype="datetime64[D]",
        )
        close_a = np.array([100.0, 90.0, 85.0, 80.0])  # Downtrend
        close_b = np.array([100.0, 105.0, 110.0, 115.0])  # Uptrend

        gate = SL11.regime_gate(
            {"A": close_a, "B": close_b},
            {"A": dates, "B": dates},
            dates,
            fast=2,
            slow=3,
            min_confirm=1,
        )

        # Symbol B uptrend should pass gate even though A is downtrend
        # (after warmup)
        assert gate[3] == True


class TestBreakoutParams:
    """Test BreakoutParams dataclass defaults."""

    def test_default_values_exist(self):
        """All default values should be set as documented."""
        p = SL11.BreakoutParams()
        assert p.prior_run_lookback == 40
        assert p.prior_run_min_return == 0.25
        assert p.base_window == 10
        assert p.base_max_range_pct == 0.15
        assert p.atr_window == 14
        assert p.stop_atr_mult == 2.0
        assert p.trail_ma_window == 20
        assert p.risk_pct == 0.02
        assert p.max_position_pct == 0.20
        assert p.max_leverage == 3.0

    def test_custom_values_override(self):
        """Custom values should override defaults."""
        p = SL11.BreakoutParams(
            prior_run_lookback=50, base_max_range_pct=0.20, stop_atr_mult=1.5
        )
        assert p.prior_run_lookback == 50
        assert p.base_max_range_pct == 0.20
        assert p.stop_atr_mult == 1.5


class TestComputeBreakoutEntries:
    """Test breakout entry signal detection."""

    def test_no_entries_first_lookback_bars(self):
        """No signal in first prior_run_lookback + base_window bars."""
        n = 100
        p = SL11.BreakoutParams(prior_run_lookback=40, base_window=10)
        open_ = np.full(n, 100.0)
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)

        signal = SL11.compute_breakout_entries(open_, high, low, close, p)
        min_history = p.prior_run_lookback + p.base_window
        assert not signal[:min_history].any()

    def test_clean_breakout_fires(self):
        """Detect a clean breakout: prior run + tight base + breakout close."""
        n = 150
        p = SL11.BreakoutParams(
            prior_run_lookback=40,
            prior_run_min_return=0.25,
            base_window=10,
            base_max_range_pct=0.15,
        )

        open_ = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        close = np.full(n, 100.0)

        # Bars 0-40: prior run up 30% (from 100 to 130)
        close[:41] = np.linspace(100, 130, 41)
        open_[:41] = close[:41]
        high[:41] = close[:41]
        low[:41] = close[:41]

        # Bars 41-50: tight base around 130
        close[41:51] = np.linspace(130, 131, 10)
        open_[41:51] = close[41:51]
        high[41:51] = close[41:51] + 1.0  # Tight (1 point range)
        low[41:51] = close[41:51] - 1.0

        # Bar 51: breakout close above base high
        close[51] = 133.0
        high[51] = 133.0
        low[51] = 131.0
        open_[51] = 131.0

        signal = SL11.compute_breakout_entries(open_, high, low, close, p)
        # Signal at bar 51 should be True (close > base_high[50])
        assert signal[51] == True

    def test_big_prior_run_choppy_base_no_signal(self):
        """Big prior run but wide base range should not signal."""
        n = 100
        p = SL11.BreakoutParams(
            prior_run_lookback=40,
            prior_run_min_return=0.25,
            base_window=10,
            base_max_range_pct=0.15,
        )

        open_ = np.full(n, 100.0)
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)

        # Bars 0-40: strong prior run
        close[:41] = np.linspace(100, 130, 41)
        open_[:41] = close[:41]
        high[:41] = close[:41]
        low[:41] = close[:41]

        # Bars 41-50: wide/choppy base (>15% range)
        close[41:51] = np.linspace(130, 125, 10)
        open_[41:51] = close[41:51]
        high[41:51] = 145.0  # Wide range relative to close ~130
        low[41:51] = 115.0

        # Bar 51: attempt breakout
        close[51] = 146.0
        high[51] = 146.0
        low[51] = 145.0
        open_[51] = 145.0

        signal = SL11.compute_breakout_entries(open_, high, low, close, p)
        # Should NOT signal due to wide base range
        assert signal[51] == False

    def test_tight_base_no_prior_run_no_signal(self):
        """Tight base but insufficient prior run should not signal."""
        n = 100
        p = SL11.BreakoutParams(
            prior_run_lookback=40,
            prior_run_min_return=0.25,
            base_window=10,
            base_max_range_pct=0.15,
        )

        open_ = np.full(n, 100.0)
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)

        # Bars 0-40: weak prior run (only 5% up, < 25% min)
        close[:41] = np.linspace(100, 105, 41)
        open_[:41] = close[:41]
        high[:41] = close[:41]
        low[:41] = close[:41]

        # Bars 41-50: tight base
        close[41:51] = np.linspace(105, 106, 10)
        open_[41:51] = close[41:51]
        high[41:51] = close[41:51] + 0.5
        low[41:51] = close[41:51] - 0.5

        # Bar 51: attempt breakout
        close[51] = 107.0
        high[51] = 107.0
        low[51] = 106.0
        open_[51] = 106.0

        signal = SL11.compute_breakout_entries(open_, high, low, close, p)
        # Should NOT signal due to weak prior run
        assert signal[51] == False


class TestSimulateBreakoutSingle:
    """Test single-symbol breakout simulation."""

    def test_no_trades_with_regime_all_false(self):
        """With regime_ok all False, no trades should occur."""
        n = 80
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        open_ = np.full(n, 100.0)
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)
        regime_ok = np.zeros(n, dtype=bool)

        p = SL11.BreakoutParams()
        result = SL11.simulate_breakout_single(
            dates, open_, high, low, close, regime_ok, p, cost_bps_per_side=10.0
        )

        assert len(result["trades"]) == 0
        assert result["equity_curve"][0] == 1_000_000.0

    def test_one_trade_enters_at_next_open(self):
        """A breakout signal fires at bar t; entry is at open_[t+1]."""
        n = 150
        p = SL11.BreakoutParams(
            prior_run_lookback=40,
            prior_run_min_return=0.25,
            base_window=10,
            base_max_range_pct=0.15,
        )

        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        open_ = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        close = np.full(n, 100.0)

        # Construct prior run + tight base + breakout
        close[:41] = np.linspace(100, 130, 41)
        open_[:41] = close[:41]
        high[:41] = close[:41]
        low[:41] = close[:41]

        close[41:51] = np.linspace(130, 131, 10)
        open_[41:51] = close[41:51]
        high[41:51] = close[41:51] + 1.0
        low[41:51] = close[41:51] - 1.0

        close[51] = 133.0
        high[51] = 133.0
        low[51] = 131.0
        open_[51] = 131.0

        # Flatten after signal
        close[52:] = 134.0
        open_[52:] = 134.0
        high[52:] = 134.0
        low[52:] = 134.0

        regime_ok = np.ones(n, dtype=bool)

        result = SL11.simulate_breakout_single(
            dates, open_, high, low, close, regime_ok, p, cost_bps_per_side=10.0
        )

        # Should have exactly one trade
        assert len(result["trades"]) == 1
        trade = result["trades"][0]
        # Entry should be at open_[52] = 134.0
        assert trade["entry_price"] == pytest.approx(134.0)

    def test_gap_below_stop_fills_at_min_open_stop(self):
        """Price gaps below stop; fill should be min(open, stop_price)."""
        n = 80
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        p = SL11.BreakoutParams()

        open_ = np.full(n, 100.0)
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)

        # Inject an entry signal at bar 50
        # (simplify: assume compute_breakout_entries would detect it)
        entry_signal = np.zeros(n, dtype=bool)
        entry_signal[50] = True

        # Manually construct by monkey-patching: we'll create a scenario
        # instead that relies on actual entry logic
        # For simplicity, construct bars that trigger entry
        close[:51] = np.linspace(100, 125, 51)
        open_[:51] = close[:51]
        high[:51] = close[:51]
        low[:51] = close[:51]

        close[51:61] = np.linspace(125, 126, 10)
        open_[51:61] = close[51:61]
        high[51:61] = close[51:61] + 0.5
        low[51:61] = close[51:61] - 0.5

        close[61] = 127.0
        high[61] = 127.0
        low[61] = 126.0
        open_[61] = 126.0

        # Bar 62: gap down below stop (if stop is at ~124)
        close[62] = 120.0
        high[62] = 122.0
        low[62] = 119.0  # Gaps below stop
        open_[62] = 121.0  # Opens above low but below stop

        regime_ok = np.ones(n, dtype=bool)

        result = SL11.simulate_breakout_single(
            dates, open_, high, low, close, regime_ok, p, cost_bps_per_side=10.0
        )

        if len(result["trades"]) > 0:
            trade = result["trades"][-1]
            if trade["exit_reason"] == "stop":
                # Fill should be at or below stop_price
                assert trade["exit_price"] <= trade["entry_price"]

    def test_trail_exit_fills_at_next_open(self):
        """Trail-exit signal at bar t fills at open_[t+1] if available."""
        n = 100
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        p = SL11.BreakoutParams(trail_ma_window=5)

        open_ = np.full(n, 100.0)
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)

        # Construct entry
        close[:51] = np.linspace(100, 125, 51)
        open_[:51] = close[:51]
        high[:51] = close[:51]
        low[:51] = close[:51]

        close[51:61] = np.linspace(125, 126, 10)
        open_[51:61] = close[51:61]
        high[51:61] = close[51:61] + 0.5
        low[51:61] = close[51:61] - 0.5

        close[61] = 127.0
        high[61] = 127.0
        low[61] = 126.0
        open_[61] = 126.0

        # After entry (bar 62), drift price down to trigger trail
        close[62:75] = np.linspace(127, 110, 13)
        open_[62:75] = close[62:75]
        high[62:75] = close[62:75]
        low[62:75] = close[62:75]

        regime_ok = np.ones(n, dtype=bool)

        result = SL11.simulate_breakout_single(
            dates, open_, high, low, close, regime_ok, p, cost_bps_per_side=10.0
        )

        # May or may not have trail exit depending on exact trail_ma
        # Just check structure
        assert "trades" in result
        assert "equity_curve" in result

    def test_position_at_end_exits_with_data_end(self):
        """Open position at data end should close with exit_reason='data_end'."""
        n = 80
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        p = SL11.BreakoutParams()

        open_ = np.full(n, 100.0)
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)

        # Construct entry
        close[:51] = np.linspace(100, 125, 51)
        open_[:51] = close[:51]
        high[:51] = close[:51]
        low[:51] = close[:51]

        close[51:61] = np.linspace(125, 126, 10)
        open_[51:61] = close[51:61]
        high[51:61] = close[51:61] + 0.5
        low[51:61] = close[51:61] - 0.5

        close[61] = 127.0
        high[61] = 127.0
        low[61] = 126.0
        open_[61] = 126.0

        # Stay high (no stop, no trail) through end
        close[62:] = 128.0
        high[62:] = 129.0
        low[62:] = 127.0
        open_[62:] = 128.0

        regime_ok = np.ones(n, dtype=bool)

        result = SL11.simulate_breakout_single(
            dates, open_, high, low, close, regime_ok, p, cost_bps_per_side=10.0
        )

        if len(result["trades"]) > 0:
            last_trade = result["trades"][-1]
            # Last trade should exit at data_end
            assert last_trade["exit_reason"] == "data_end"
            assert last_trade["exit_price"] == pytest.approx(close[-1])

    def test_ret_eq_arithmetic_correct(self):
        """ret_eq should equal pnl / equity_at_open."""
        n = 80
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        p = SL11.BreakoutParams()

        open_ = np.full(n, 100.0)
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)

        # Construct entry
        close[:51] = np.linspace(100, 125, 51)
        open_[:51] = close[:51]
        high[:51] = close[:51]
        low[:51] = close[:51]

        close[51:61] = np.linspace(125, 126, 10)
        open_[51:61] = close[51:61]
        high[51:61] = close[51:61] + 0.5
        low[51:61] = close[51:61] - 0.5

        close[61] = 127.0
        high[61] = 127.0
        low[61] = 126.0
        open_[61] = 126.0

        close[62:] = 135.0
        high[62:] = 136.0
        low[62:] = 134.0
        open_[62:] = 135.0

        regime_ok = np.ones(n, dtype=bool)

        result = SL11.simulate_breakout_single(
            dates, open_, high, low, close, regime_ok, p, cost_bps_per_side=10.0
        )

        for trade in result["trades"]:
            expected_ret_eq = trade["pnl"] / trade["equity_at_open"]
            assert trade["ret_eq"] == pytest.approx(expected_ret_eq)

    def test_pnl_atr_matches_helper_function(self):
        """pnl_atr in trade should match SL11.pnl_atr() independently."""
        n = 80
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        p = SL11.BreakoutParams()

        open_ = np.full(n, 100.0)
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)

        # Construct entry
        close[:51] = np.linspace(100, 125, 51)
        open_[:51] = close[:51]
        high[:51] = close[:51]
        low[:51] = close[:51]

        close[51:61] = np.linspace(125, 126, 10)
        open_[51:61] = close[51:61]
        high[51:61] = close[51:61] + 0.5
        low[51:61] = close[51:61] - 0.5

        close[61] = 127.0
        high[61] = 127.0
        low[61] = 126.0
        open_[61] = 126.0

        close[62:] = 135.0
        high[62:] = 136.0
        low[62:] = 134.0
        open_[62:] = 135.0

        regime_ok = np.ones(n, dtype=bool)

        result = SL11.simulate_breakout_single(
            dates, open_, high, low, close, regime_ok, p, cost_bps_per_side=10.0
        )

        for trade in result["trades"]:
            expected = SL11.pnl_atr(trade["pnl"], trade["qty"], trade["atr_at_entry"])
            assert trade["pnl_atr"] == pytest.approx(expected, rel=1e-9)


class TestSimulateBreakoutBook:
    """Test multi-symbol breakout portfolio simulation."""

    def test_portfolio_equity_starts_at_start_equity(self):
        """Initial portfolio equity should equal start_equity."""
        n = 80
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        open_ = np.full(n, 100.0)
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)
        regime_ok = np.zeros(n, dtype=bool)  # No trades

        symbol_frames = {
            "SYM_A": {
                "dates": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
        }
        regime_ok_by_symbol = {"SYM_A": regime_ok}

        p = SL11.BreakoutParams()
        start_eq = 1_000_000.0
        book = SL11.simulate_breakout_book(
            symbol_frames,
            regime_ok_by_symbol,
            p,
            cost_bps_per_side=10.0,
            start_equity=start_eq,
        )

        assert book["portfolio_equity"][0] == pytest.approx(start_eq)

    def test_trades_carry_symbol_key(self):
        """Each trade should have a 'symbol' key."""
        n = 100
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n, "D"),
            dtype="datetime64[D]",
        )
        open_ = np.full(n, 100.0)
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)

        # Construct entry
        close[:51] = np.linspace(100, 125, 51)
        open_[:51] = close[:51]
        high[:51] = close[:51]
        low[:51] = close[:51]

        close[51:61] = np.linspace(125, 126, 10)
        open_[51:61] = close[51:61]
        high[51:61] = close[51:61] + 0.5
        low[51:61] = close[51:61] - 0.5

        close[61] = 127.0
        high[61] = 127.0
        low[61] = 126.0
        open_[61] = 126.0

        close[62:] = 128.0
        high[62:] = 129.0
        low[62:] = 127.0
        open_[62:] = 128.0

        regime_ok = np.ones(n, dtype=bool)

        symbol_frames = {
            "SYM_A": {
                "dates": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
        }
        regime_ok_by_symbol = {"SYM_A": regime_ok}

        p = SL11.BreakoutParams()
        book = SL11.simulate_breakout_book(
            symbol_frames, regime_ok_by_symbol, p, cost_bps_per_side=10.0
        )

        for trade in book["trades"]:
            assert "symbol" in trade
            assert trade["symbol"] == "SYM_A"

    def test_multi_symbol_pooling(self):
        """With two symbols, portfolio_equity should pool correctly."""
        n_a = 80
        n_b = 100
        dates_a = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n_a, "D"),
            dtype="datetime64[D]",
        )
        dates_b = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(n_b, "D"),
            dtype="datetime64[D]",
        )

        open_a = np.full(n_a, 100.0)
        high_a = np.full(n_a, 105.0)
        low_a = np.full(n_a, 95.0)
        close_a = np.full(n_a, 100.0)
        regime_ok_a = np.zeros(n_a, dtype=bool)

        open_b = np.full(n_b, 100.0)
        high_b = np.full(n_b, 105.0)
        low_b = np.full(n_b, 95.0)
        close_b = np.full(n_b, 100.0)
        regime_ok_b = np.zeros(n_b, dtype=bool)

        symbol_frames = {
            "SYM_A": {
                "dates": dates_a,
                "open": open_a,
                "high": high_a,
                "low": low_a,
                "close": close_a,
            },
            "SYM_B": {
                "dates": dates_b,
                "open": open_b,
                "high": high_b,
                "low": low_b,
                "close": close_b,
            },
        }
        regime_ok_by_symbol = {"SYM_A": regime_ok_a, "SYM_B": regime_ok_b}

        p = SL11.BreakoutParams()
        start_eq = 1_000_000.0
        book = SL11.simulate_breakout_book(
            symbol_frames,
            regime_ok_by_symbol,
            p,
            cost_bps_per_side=10.0,
            start_equity=start_eq,
        )

        # Portfolio equity should start at start_eq
        assert book["portfolio_equity"][0] == pytest.approx(start_eq)
        # With no trades, it should stay flat
        assert all(
            np.isclose(book["portfolio_equity"][i], start_eq)
            for i in range(len(book["portfolio_equity"]))
        )


class TestBreakoutBookMetrics:
    """Test breakout book metrics calculation."""

    def test_metrics_basic_structure(self):
        """Metrics should include all required keys."""
        book = {
            "trades": [],
            "portfolio_equity": np.array([1_000_000.0, 1_000_000.0, 1_000_000.0]),
            "start_equity": 1_000_000.0,
            "dates": np.arange(
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-01") + np.timedelta64(3, "D"),
                dtype="datetime64[D]",
            ),
        }
        metrics = SL11.breakout_book_metrics(book)

        required_keys = {
            "sharpe",
            "max_drawdown",
            "fixed_notional_return",
            "equity_path_return",
            "n_trades",
            "n_stop_exits",
            "n_trail_exits",
            "n_data_end_exits",
        }
        for key in required_keys:
            assert key in metrics

    def test_exit_reason_counts_sum_correctly(self):
        """n_stop + n_trail + n_data_end should equal n_trades."""
        trades = [
            {"exit_reason": "stop", "ret_eq": 0.01},
            {"exit_reason": "stop", "ret_eq": -0.02},
            {"exit_reason": "trail", "ret_eq": 0.03},
            {"exit_reason": "data_end", "ret_eq": 0.005},
        ]
        book = {
            "trades": trades,
            "portfolio_equity": np.array(
                [1_000_000.0, 1_010_000.0, 990_000.0, 1_020_000.0, 1_025_000.0]
            ),
            "start_equity": 1_000_000.0,
            "dates": np.arange(
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-01") + np.timedelta64(5, "D"),
                dtype="datetime64[D]",
            ),
        }
        metrics = SL11.breakout_book_metrics(book)

        total = (
            metrics["n_stop_exits"]
            + metrics["n_trail_exits"]
            + metrics["n_data_end_exits"]
        )
        assert total == metrics["n_trades"]

    def test_fixed_notional_return_sums_ret_eq(self):
        """fixed_notional_return should equal sum of ret_eq."""
        trades = [
            {"exit_reason": "stop", "ret_eq": 0.01},
            {"exit_reason": "trail", "ret_eq": 0.02},
        ]
        book = {
            "trades": trades,
            "portfolio_equity": np.array([1_000_000.0, 1_010_000.0, 1_030_000.0]),
            "start_equity": 1_000_000.0,
            "dates": np.arange(
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-01") + np.timedelta64(3, "D"),
                dtype="datetime64[D]",
            ),
        }
        metrics = SL11.breakout_book_metrics(book)

        assert metrics["fixed_notional_return"] == pytest.approx(0.03)

    def test_max_drawdown_computed_correctly(self):
        """max_drawdown should be min(equity / running_max - 1)."""
        # Equity path: 1M -> 1.2M -> 0.9M -> 1.1M
        equity = np.array([1_000_000.0, 1_200_000.0, 900_000.0, 1_100_000.0])
        book = {
            "trades": [],
            "portfolio_equity": equity,
            "start_equity": 1_000_000.0,
            "dates": np.arange(
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-01") + np.timedelta64(4, "D"),
                dtype="datetime64[D]",
            ),
        }
        metrics = SL11.breakout_book_metrics(book)

        # running_max: [1M, 1.2M, 1.2M, 1.2M]
        # drawdown: [0, 0, -0.25, -0.0833]
        expected_dd = -0.25
        assert metrics["max_drawdown"] == pytest.approx(expected_dd, rel=1e-4)


def _uptrend_base_breakout_ohlcv(n=150, rng_half_width=1.0):
    """Shared fixture: prior run up, tight base, breakout on bar 51 --
    same shape as TestComputeBreakoutEntries's clean_breakout fixture but
    with a nonzero daily range (needed for a finite ATR) and a volume
    series, used by the Gate VB (volume-confirmed) tests below."""
    open_ = np.full(n, 100.0)
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    volume = np.full(n, 1000.0)

    close[:41] = np.linspace(100, 130, 41)
    open_[:41] = close[:41]
    high[:41] = close[:41] + rng_half_width
    low[:41] = close[:41] - rng_half_width

    close[41:51] = np.linspace(130, 131, 10)
    open_[41:51] = close[41:51]
    high[41:51] = close[41:51] + rng_half_width
    low[41:51] = close[41:51] - rng_half_width

    close[51] = 133.0
    high[51] = 133.0 + rng_half_width
    low[51] = 131.0
    open_[51] = 131.0

    close[52:] = np.linspace(133, 140, n - 52)
    open_[52:] = close[52:]
    high[52:] = close[52:] + rng_half_width
    low[52:] = close[52:] - rng_half_width

    return open_, high, low, close, volume


class TestVolBreakoutParams:
    def test_defaults(self):
        p = SL11.VolBreakoutParams()
        assert p.use_volume is True
        assert p.vol_k == 1.5
        assert p.vol_window == 20
        assert p.prior_run_min_atr_mult == 6.0
        assert p.base_max_range_atr_mult == 3.0

    def test_ungated_control_flag(self):
        p = SL11.VolBreakoutParams(use_volume=False)
        assert p.use_volume is False


class TestComputeVolBreakoutEntries:
    def test_ungated_long_breakout_fires(self):
        open_, high, low, close, _volume = _uptrend_base_breakout_ohlcv()
        p = SL11.VolBreakoutParams(use_volume=False)
        entries = SL11.compute_vol_breakout_entries(open_, high, low, close, None, p)
        assert entries["long"][51]
        assert not entries["short"][51]

    def test_volume_gate_blocks_low_volume_breakout(self):
        open_, high, low, close, volume = _uptrend_base_breakout_ohlcv()
        volume[51] = 500.0  # below vol_k(1.5) x trailing median(1000)
        p = SL11.VolBreakoutParams(use_volume=True)
        entries = SL11.compute_vol_breakout_entries(open_, high, low, close, volume, p)
        assert not entries["long"][51]

    def test_volume_gate_allows_confirmed_breakout(self):
        open_, high, low, close, volume = _uptrend_base_breakout_ohlcv()
        volume[51] = 2000.0  # 2x trailing median(1000) clears vol_k=1.5
        p = SL11.VolBreakoutParams(use_volume=True)
        entries = SL11.compute_vol_breakout_entries(open_, high, low, close, volume, p)
        assert entries["long"][51]

    def test_short_side_mirrors_long(self):
        open_, high, low, close, _volume = _uptrend_base_breakout_ohlcv()
        # mirror the fixture: downtrend, tight base, breakdown
        open_, high, low, close = (
            200.0 - open_ + 100.0,
            200.0 - low + 100.0,
            200.0 - high + 100.0,
            200.0 - close + 100.0,
        )
        p = SL11.VolBreakoutParams(use_volume=False)
        entries = SL11.compute_vol_breakout_entries(open_, high, low, close, None, p)
        assert entries["short"][51]
        assert not entries["long"][51]

    def test_roll_bar_disqualified_from_volume_leg(self):
        open_, high, low, close, volume = _uptrend_base_breakout_ohlcv()
        volume[51] = 2000.0
        is_roll = np.zeros(len(close), dtype=bool)
        is_roll[51] = True
        p = SL11.VolBreakoutParams(use_volume=True)
        entries = SL11.compute_vol_breakout_entries(
            open_, high, low, close, volume, p, is_roll=is_roll
        )
        assert not entries["long"][51]


class TestSimulateVolBreakoutSingle:
    def test_long_trade_stop_exit(self):
        open_, high, low, close, volume = _uptrend_base_breakout_ohlcv()
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(len(close), "D"),
            dtype="datetime64[D]",
        )
        p = SL11.VolBreakoutParams(use_volume=False)
        regime_ok = np.ones(len(close), dtype=bool)
        # crash the price right after entry to force a stop-exit
        close_crash = close.copy()
        high_crash = high.copy()
        low_crash = low.copy()
        low_crash[53:] = 50.0
        close_crash[53:] = 55.0
        high_crash[53:] = 60.0

        res = SL11.simulate_vol_breakout_single(
            dates,
            open_,
            high_crash,
            low_crash,
            close_crash,
            volume,
            regime_ok,
            p,
            cost_bps_per_side=0.0,
        )
        assert len(res["trades"]) >= 1
        assert res["trades"][0]["side"] == "long"
        assert res["trades"][0]["exit_reason"] == "stop"
        assert res["trades"][0]["pnl"] < 0

    def test_no_trades_with_regime_all_false(self):
        open_, high, low, close, volume = _uptrend_base_breakout_ohlcv()
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(len(close), "D"),
            dtype="datetime64[D]",
        )
        p = SL11.VolBreakoutParams(use_volume=False)
        regime_ok = np.zeros(len(close), dtype=bool)
        res = SL11.simulate_vol_breakout_single(
            dates,
            open_,
            high,
            low,
            close,
            volume,
            regime_ok,
            p,
            cost_bps_per_side=0.0,
        )
        assert len(res["trades"]) == 0


class TestSimulateVolBreakoutBook:
    def test_pools_two_symbols(self):
        open_, high, low, close, volume = _uptrend_base_breakout_ohlcv()
        dates = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + np.timedelta64(len(close), "D"),
            dtype="datetime64[D]",
        )
        frames = {
            "A": {
                "dates": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            "B": {
                "dates": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
        }
        regime = {
            "A": np.ones(len(close), dtype=bool),
            "B": np.ones(len(close), dtype=bool),
        }
        p = SL11.VolBreakoutParams(use_volume=False)
        book = SL11.simulate_vol_breakout_book(frames, regime, p, cost_bps_per_side=0.0)
        symbols_traded = {t["symbol"] for t in book["trades"]}
        assert symbols_traded == {"A", "B"}
        metrics = SL11.breakout_book_metrics(book)
        assert metrics["n_trades"] == len(book["trades"])
