"""Unit tests for notebook-10a/10b machinery (src/research/tmp/spread_lib10.py):
term-structure regime definitions, spread taxonomy classification, ADF
cointegration testing, regime-conditional structure, and COT positioning.
Mirrors tests/test_research_lib9.py and tests/test_commod_lib8.py conventions.
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp"))

import spread_lib10 as SL10

SEED = 0


class TestClassifySpreadTaxonomy:
    def test_calendar_spread_same_product(self):
        """A spread with two legs of the same product (e.g., CL-F1 and CL-F2)
        should be classified as 'calendar'."""
        assert SL10.classify_spread_taxonomy(["CL", "CL"]) == "calendar"

    def test_calendar_spread_multiple_same_product(self):
        """A spread with three or more legs of the same product should be
        classified as 'calendar'."""
        assert SL10.classify_spread_taxonomy(["WTI", "WTI", "WTI"]) == "calendar"

    def test_inter_commodity_spread_two_products(self):
        """A spread with legs from two distinct products (e.g., crude/heating
        oil, or corn/wheat) should be classified as 'inter_commodity'."""
        assert SL10.classify_spread_taxonomy(["CL", "HO"]) == "inter_commodity"

    def test_inter_commodity_spread_three_products(self):
        """A spread with three or more distinct products should be classified
        as 'inter_commodity'."""
        assert SL10.classify_spread_taxonomy(["CL", "HO", "RB"]) == "inter_commodity"

    def test_raises_on_single_leg(self):
        """A spread must have at least 2 legs; passing a single-element list
        should raise ValueError."""
        with pytest.raises(ValueError, match="a spread needs at least 2 legs"):
            SL10.classify_spread_taxonomy(["CL"])

    def test_raises_on_empty_list(self):
        """An empty leg list should raise ValueError."""
        with pytest.raises(ValueError, match="a spread needs at least 2 legs"):
            SL10.classify_spread_taxonomy([])


class TestAdfTest:
    def test_strongly_mean_reverting_series_is_stationary(self):
        """A synthetic AR(1) series with strong mean reversion (e.g.,
        x[t] = 0.9*x[t-1] + noise) should be flagged as stationary at 5%
        with a large negative t_stat."""
        rng = np.random.default_rng(SEED)
        n = 2000
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = 0.9 * v[t - 1] + rng.normal(0, 1)
        result = SL10.adf_test(v)
        assert result["stationary_5pct"] is True
        assert result["t_stat"] < SL10.ADF_CRITICAL_VALUES["5%"]
        assert result["t_stat"] < -2.0  # clearly negative (strong reversion)
        assert result["n_obs"] >= n - 20  # most observations used

    def test_random_walk_not_stationary(self):
        """A random walk (cumulative sum of noise, equivalent to unit root)
        should NOT be flagged as stationary at 5%."""
        rng = np.random.default_rng(SEED)
        n = 2000
        v = np.cumsum(rng.normal(0, 1, n))
        result = SL10.adf_test(v)
        assert result["stationary_5pct"] is False
        # Random walk should have t_stat > critical value (or NaN)
        if np.isfinite(result["t_stat"]):
            assert result["t_stat"] > SL10.ADF_CRITICAL_VALUES["5%"]

    def test_short_series_returns_degenerate_dict(self):
        """A series with fewer than 30 observations should return the
        degenerate result dict without raising an exception."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SL10.adf_test(v)
        assert result["n_obs"] == 5
        assert result["stationary_5pct"] is False
        assert result["stationary_1pct"] is False
        assert np.isnan(result["t_stat"])
        assert result["n_lags"] == 0

    def test_series_with_nans_filtered(self):
        """NaN values in the input should be filtered out before testing."""
        rng = np.random.default_rng(SEED)
        v = np.zeros(100)
        for t in range(1, 100):
            v[t] = 0.9 * v[t - 1] + rng.normal(0, 1)
        # Insert NaNs
        v[10:15] = np.nan
        v[50:52] = np.nan
        result = SL10.adf_test(v)
        # Should have used approximately 100 - 5 - 2 = 93 observations
        assert result["n_obs"] < 100
        assert result["n_obs"] >= 30

    def test_adf_critical_values_structure(self):
        """The critical values dict should contain the expected percentiles."""
        assert "1%" in SL10.ADF_CRITICAL_VALUES
        assert "5%" in SL10.ADF_CRITICAL_VALUES
        assert "10%" in SL10.ADF_CRITICAL_VALUES
        # Critical values should be negative (left tail)
        assert SL10.ADF_CRITICAL_VALUES["1%"] < SL10.ADF_CRITICAL_VALUES["5%"]
        assert SL10.ADF_CRITICAL_VALUES["5%"] < SL10.ADF_CRITICAL_VALUES["10%"]


class TestRegimeDeadband:
    def test_flat_regime_below_deadband(self):
        """Roll slopes with magnitude below the deadband should be labelled
        'flat' regardless of sign."""
        slopes = pl.Series("slope", [0.001, -0.001, 0.0, 0.01, -0.01])
        deadband = 0.02
        result = SL10.regime_deadband(slopes, deadband=deadband)
        expected = ["flat", "flat", "flat", "flat", "flat"]
        assert result.to_list() == expected

    def test_contango_above_deadband(self):
        """Positive roll slopes above the deadband should be labelled
        'contango'."""
        slopes = pl.Series("slope", [0.03, 0.05, 0.5])
        result = SL10.regime_deadband(slopes, deadband=0.02)
        expected = ["contango", "contango", "contango"]
        assert result.to_list() == expected

    def test_backwardation_below_negative_deadband(self):
        """Negative roll slopes below the negative deadband threshold should
        be labelled 'backwardation' (following the sign convention: negative
        slope = backwardation)."""
        slopes = pl.Series("slope", [-0.03, -0.05, -0.5])
        result = SL10.regime_deadband(slopes, deadband=0.02)
        expected = ["backwardation", "backwardation", "backwardation"]
        assert result.to_list() == expected

    def test_mixed_regimes(self):
        """A series with values spanning all three regimes should label them
        correctly."""
        slopes = pl.Series("slope", [-0.05, -0.001, 0.0, 0.001, 0.05])
        result = SL10.regime_deadband(slopes, deadband=0.01)
        expected = ["backwardation", "flat", "flat", "flat", "contango"]
        assert result.to_list() == expected

    def test_null_values_propagate(self):
        """Null roll slopes should result in null regime labels."""
        slopes = pl.Series("slope", [0.05, None, -0.05])
        result = SL10.regime_deadband(slopes, deadband=0.02)
        result_list = result.to_list()
        assert result_list[0] == "contango"
        assert result_list[1] is None
        assert result_list[2] == "backwardation"


class TestRegimePersistent:
    def test_run_shorter_than_n_days_stays_unconfirmed(self):
        """A run of the same state that is shorter than n_days should remain
        labelled 'unconfirmed' throughout."""
        state = pl.Series("state", ["contango", "contango", "backwardation"])
        result = SL10.regime_persistent(state, n_days=5)
        expected = pl.Series("", ["unconfirmed", "unconfirmed", "unconfirmed"])
        assert (result == expected).all()

    def test_run_at_least_n_days_confirms_exactly_at_n_days(self):
        """A run of the same state that reaches n_days or more should flip to
        confirmed exactly at the n_days-th observation of that run, not
        before."""
        state = pl.Series("state", ["contango"] * 10)
        result = SL10.regime_persistent(state, n_days=5)
        # First 4 obs should be unconfirmed, 5th onwards should be confirmed
        expected = pl.Series(
            "",
            [
                "unconfirmed",
                "unconfirmed",
                "unconfirmed",
                "unconfirmed",
                "contango",
                "contango",
                "contango",
                "contango",
                "contango",
                "contango",
            ],
        )
        assert (result == expected).all()

    def test_state_transition_resets_run_counter(self):
        """A change in state should reset the run counter and start a new
        unconfirmed run."""
        state = pl.Series(
            "state", ["contango"] * 7 + ["backwardation"] * 5 + ["contango"] * 3
        )
        result = SL10.regime_persistent(state, n_days=5)
        # contango runs: 0-3 unconfirmed, 4-6 confirmed
        # backwardation runs: 7-10 unconfirmed (only 5 long, but starts at 7)
        # Actually wait: the backwardation run is indices 7-11 (5 days),
        # so 7-10 unconfirmed, 11 confirmed
        # contango again: 12-14 unconfirmed (only 3)
        expected = pl.Series(
            "",
            [
                "unconfirmed",  # 0
                "unconfirmed",  # 1
                "unconfirmed",  # 2
                "unconfirmed",  # 3
                "contango",  # 4
                "contango",  # 5
                "contango",  # 6
                "unconfirmed",  # 7 (new state)
                "unconfirmed",  # 8
                "unconfirmed",  # 9
                "unconfirmed",  # 10
                "backwardation",  # 11 (5th day of backwardation run)
                "unconfirmed",  # 12 (new state)
                "unconfirmed",  # 13
                "unconfirmed",  # 14
            ],
        )
        assert (result == expected).all()

    def test_null_states_treated_as_unconfirmed(self):
        """Null state values should keep the label as 'unconfirmed' and not
        affect the run tracking."""
        state = pl.Series("state", ["contango"] * 3 + [None] + ["contango"] * 3)
        result = SL10.regime_persistent(state, n_days=2)
        expected = pl.Series(
            "",
            [
                "unconfirmed",
                "contango",
                "contango",
                "unconfirmed",
                "unconfirmed",
                "contango",
                "contango",
            ],
        )
        assert (result == expected).all()


class TestRollingLegCorrelation:
    def test_perfectly_correlated_series(self):
        """Two perfectly correlated series (r2 = 2*r1) should yield a rolling
        correlation of approximately 1.0 in every valid window."""
        rng = np.random.default_rng(SEED)
        r1 = rng.normal(0, 1, 500)
        r2 = 2.0 * r1  # Perfect correlation
        result = SL10.rolling_leg_correlation(r1, r2, window=60)
        # The first window_size-1 values should be NaN
        # All remaining values should be ~1.0
        assert np.isnan(result[0])
        assert np.isnan(
            result[58]
        )  # First 59 (window_size) should have < window samples
        valid = result[60:]  # Window complete from index 60 onward
        # Check that valid values are very close to 1.0
        valid = valid[np.isfinite(valid)]
        assert np.all(valid > 0.99)

    def test_independent_random_series(self):
        """Two independent random series should have rolling correlations
        well below 1.0 in absolute value."""
        rng = np.random.default_rng(SEED)
        r1 = rng.normal(0, 1, 500)
        r2 = rng.normal(0, 1, 500)  # Independent
        result = SL10.rolling_leg_correlation(r1, r2, window=60)
        valid = result[np.isfinite(result)]
        # Correlations should be far from ±1.0 on average
        assert np.mean(np.abs(valid)) < 0.5

    def test_window_smaller_than_observations_returns_nan(self):
        """Windows with fewer than `window` finite observations should return
        NaN."""
        rng = np.random.default_rng(SEED)
        r1 = rng.normal(0, 1, 100)
        r2 = rng.normal(0, 1, 100)
        result = SL10.rolling_leg_correlation(r1, r2, window=60)
        # First window_size - 1 should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[58])

    def test_nans_in_input_handled(self):
        """NaN values in either leg should be treated as missing and excluded
        from the rolling correlation window."""
        rng = np.random.default_rng(SEED)
        r1 = rng.normal(0, 1, 200)
        r2 = rng.normal(0, 1, 200)
        r1[10:15] = np.nan
        r2[50:52] = np.nan
        result = SL10.rolling_leg_correlation(r1, r2, window=60)
        # Should still return valid results for windows unaffected by NaNs
        # (well after the NaN regions)
        valid_section = result[150:]
        valid = valid_section[np.isfinite(valid_section)]
        assert len(valid) > 0


class TestRegimeConditionalAr1:
    def test_two_regimes_with_sufficient_data(self):
        """Given a regime label array that is half 'a' and half 'b', both
        labels should appear as keys in the output plus a '_pooled' key."""
        rng = np.random.default_rng(SEED)
        n = 240  # 120 per regime, above the 60 observation minimum
        values = rng.normal(0, 1, n).cumsum()  # Random walk to ensure non-NaN
        labels = ["a"] * 120 + ["b"] * 120  # Regime "a" is first 120, "b" is last 120
        result = SL10.regime_conditional_ar1(values, labels)
        assert "a" in result
        assert "b" in result
        assert "_pooled" in result
        assert result["a"]["fit"] is not None
        assert result["b"]["fit"] is not None
        assert result["_pooled"]["fit"] is not None

    def test_regime_below_minimum_observations(self):
        """A regime with fewer than 60 observations should report fit as None
        rather than raising or fabricating a number."""
        rng = np.random.default_rng(SEED)
        n = 100
        values = rng.normal(0, 1, n)
        labels = ["a"] * 30 + ["b"] * 70  # Regime "a" has only 30 obs
        result = SL10.regime_conditional_ar1(values, labels)
        assert result["a"]["fit"] is None
        assert result["a"]["n"] == 30
        assert result["b"]["fit"] is not None
        assert result["b"]["n"] == 70

    def test_pooled_fit_computed(self):
        """The '_pooled' key should always be present with the unconditional
        fit across all regimes combined."""
        rng = np.random.default_rng(SEED)
        n = 200
        values = rng.normal(0, 1, n).cumsum()
        labels = ["a"] * 100 + ["b"] * 100
        result = SL10.regime_conditional_ar1(values, labels)
        assert "_pooled" in result
        assert result["_pooled"]["n"] == n
        assert result["_pooled"]["fit"] is not None

    def test_nans_filtered_before_regime_split(self):
        """NaN values in the values array should be filtered out before
        regime-conditional computation."""
        rng = np.random.default_rng(SEED)
        n = 200
        values = rng.normal(0, 1, n).cumsum()
        values[50:55] = np.nan
        labels = ["a"] * 100 + ["b"] * 100
        result = SL10.regime_conditional_ar1(values, labels)
        # Total observations should reflect NaNs filtered out
        assert result["_pooled"]["n"] == n - 5


class TestRegimeConditionalVol:
    def test_two_regimes_with_sufficient_data(self):
        """Given a regime label array that is half 'a' and half 'b', both
        labels should appear as keys in the output plus a '_pooled' key, with
        vol_annualized reported for all regimes with >= 20 observations."""
        rng = np.random.default_rng(SEED)
        n = 200
        returns = rng.normal(0, 0.01, n)  # Daily returns
        labels = ["a"] * 100 + ["b"] * 100
        result = SL10.regime_conditional_vol(returns, labels)
        assert "a" in result
        assert "b" in result
        assert "_pooled" in result
        assert result["a"]["vol_annualized"] is not None
        assert result["b"]["vol_annualized"] is not None
        assert result["_pooled"]["vol_annualized"] is not None

    def test_regime_below_minimum_observations_vol_is_none(self):
        """A regime with fewer than 20 observations should report
        vol_annualized as None."""
        rng = np.random.default_rng(SEED)
        n = 100
        returns = rng.normal(0, 0.01, n)
        labels = ["a"] * 15 + ["b"] * 85  # Regime "a" has only 15 obs
        result = SL10.regime_conditional_vol(returns, labels)
        assert result["a"]["vol_annualized"] is None
        assert result["a"]["n"] == 15
        assert result["b"]["vol_annualized"] is not None
        assert result["b"]["n"] == 85

    def test_vol_annualization_factor(self):
        """The vol should be annualized by sqrt(252) (trading days per year)."""
        n = 100
        # Create returns with known standard deviation: 1% daily returns
        # std([0.01, 0.01, ..., 0.01]) = 0, so use a series with known std
        returns = np.full(n, 0.01)  # All same value -> std=0 -> vol=0
        returns[0] = 0.02  # Perturb one value to create variance
        labels = ["a"] * n
        result = SL10.regime_conditional_vol(returns, labels)
        # Verify that annualization factor is applied: vol = std(returns) * sqrt(252)
        expected_std = np.std(returns)
        expected_vol = expected_std * np.sqrt(252)
        assert result["a"]["vol_annualized"] == pytest.approx(expected_vol, rel=1e-5)

    def test_nans_filtered_before_regime_split_vol(self):
        """NaN values should be filtered out before regime-conditional vol
        computation."""
        rng = np.random.default_rng(SEED)
        n = 200
        returns = rng.normal(0, 0.01, n)
        returns[50:55] = np.nan
        labels = ["a"] * 100 + ["b"] * 100
        result = SL10.regime_conditional_vol(returns, labels)
        assert result["_pooled"]["n"] == n - 5


class TestRegimeStatePersistence:
    def test_mean_run_length_computed(self):
        """Given a known, hand-constructed state sequence, the mean run
        lengths per state should match the expected values."""
        state = pl.Series("state", ["a", "a", "a", "b", "b", "a", "a", "a", "a", "a"])
        result = SL10.regime_state_persistence(state)
        # Runs: "a" appears in runs [0:3] (length 3), [5:10] (length 5)
        # "b" appears in run [3:5] (length 2)
        # Mean run lengths: "a" = (3+5)/2 = 4.0, "b" = 2.0
        assert result["mean_run_length"]["a"] == pytest.approx(4.0)
        assert result["mean_run_length"]["b"] == pytest.approx(2.0)

    def test_transition_matrix_rows_sum_to_one(self):
        """The transition matrix should have rows (from-states) that sum to
        1.0, and should skip any state that never appears as a from-state."""
        state = pl.Series("state", ["a", "a", "b", "b", "a", "a", "c", "c"])
        result = SL10.regime_state_persistence(state)
        trans = result["transition_matrix"]
        # States that appear as from-states: "a", "b", "c"
        # Transitions in sequence: a->a, a->b, b->b, b->a, a->a, a->c, c->c
        # "a" transitions: a->a (2 times at indices 0->1 and 4->5), a->b (1 time at 1->2), a->c (1 time at 5->6) -- total 4
        # "b" transitions: b->b (1 time at 2->3), b->a (1 time at 3->4) -- total 2
        # "c" transitions: c->c (1 time at 6->7) -- total 1
        assert trans["a"]["a"] == pytest.approx(0.5)
        assert trans["a"]["b"] == pytest.approx(0.25)
        assert trans["a"]["c"] == pytest.approx(0.25)
        assert trans["b"]["a"] == pytest.approx(0.5)
        assert trans["b"]["b"] == pytest.approx(0.5)
        assert trans["c"]["c"] == pytest.approx(1.0)
        # All rows should sum to 1.0
        for to_states in trans.values():
            row_sum = sum(to_states.values())
            assert row_sum == pytest.approx(1.0)

    def test_null_states_dropped(self):
        """Null values in the state series should be dropped before computing
        runs and transitions."""
        state = pl.Series("state", ["a", "a", None, "b", "b", "a"])
        result = SL10.regime_state_persistence(state)
        # After dropping nulls: ["a", "a", "b", "b", "a"]
        # Runs: "a" [0:2] (length 2), "b" [2:4] (length 2), "a" [4:5] (length 1)
        # Mean run lengths: "a" = (2+1)/2 = 1.5, "b" = 2.0
        assert result["mean_run_length"]["a"] == pytest.approx(1.5)
        assert result["mean_run_length"]["b"] == pytest.approx(2.0)

    def test_empty_state_series_returns_empty_dicts(self):
        """An empty state series (or all-null) should return empty dicts for
        mean_run_length and transition_matrix."""
        state = pl.Series("state", [], dtype=pl.Utf8)
        result = SL10.regime_state_persistence(state)
        assert result["mean_run_length"] == {}
        assert result["transition_matrix"] == {}


class TestCotNetNoncommFraction:
    def test_basic_fraction_calculation(self):
        """A tiny synthetic COT DataFrame should compute the net
        non-commercial fraction correctly: (long - short) / open_interest."""
        cot = pl.DataFrame(
            {
                "report_date": ["2024-01-01", "2024-01-08"],
                "noncomm_positions_long_all": [1000, 2000],
                "noncomm_positions_short_all": [400, 800],
                "open_interest_all": [10000, 20000],
            }
        )
        result = SL10.cot_net_noncomm_fraction(cot)
        # First row: (1000 - 400) / 10000 = 0.06
        # Second row: (2000 - 800) / 20000 = 0.06
        assert result["net_noncomm_frac"][0] == pytest.approx(0.06)
        assert result["net_noncomm_frac"][1] == pytest.approx(0.06)

    def test_public_date_is_report_date_plus_three_days(self):
        """The public_date column should be exactly report_date + 3 calendar
        days."""
        cot = pl.DataFrame(
            {
                "report_date": ["2024-01-01", "2024-01-08"],
                "noncomm_positions_long_all": [1000, 2000],
                "noncomm_positions_short_all": [400, 800],
                "open_interest_all": [10000, 20000],
            }
        )
        result = SL10.cot_net_noncomm_fraction(cot)
        # Check that public_date = report_date + 3 days
        result_dates = result["public_date"].to_list()
        assert result_dates[0] == datetime.date(2024, 1, 4)
        assert result_dates[1] == datetime.date(2024, 1, 11)

    def test_output_columns_present(self):
        """The output DataFrame should have exactly the columns:
        report_date, public_date, net_noncomm_frac, open_interest."""
        cot = pl.DataFrame(
            {
                "report_date": ["2024-01-01"],
                "noncomm_positions_long_all": [1000],
                "noncomm_positions_short_all": [400],
                "open_interest_all": [10000],
            }
        )
        result = SL10.cot_net_noncomm_fraction(cot)
        expected_cols = [
            "report_date",
            "public_date",
            "net_noncomm_frac",
            "open_interest",
        ]
        assert result.columns == expected_cols

    def test_zero_open_interest_clip(self):
        """When open_interest is 0, it should be clipped to 1 to avoid
        division by zero."""
        cot = pl.DataFrame(
            {
                "report_date": ["2024-01-01"],
                "noncomm_positions_long_all": [1000],
                "noncomm_positions_short_all": [400],
                "open_interest_all": [0],
            }
        )
        result = SL10.cot_net_noncomm_fraction(cot)
        # (1000 - 400) / 1 = 600
        assert result["net_noncomm_frac"][0] == pytest.approx(600.0)

    def test_sorted_by_report_date(self):
        """The output should be sorted by report_date."""
        cot = pl.DataFrame(
            {
                "report_date": ["2024-01-08", "2024-01-01", "2024-01-15"],
                "noncomm_positions_long_all": [2000, 1000, 3000],
                "noncomm_positions_short_all": [800, 400, 1200],
                "open_interest_all": [20000, 10000, 30000],
            }
        )
        result = SL10.cot_net_noncomm_fraction(cot)
        result_dates = result["report_date"].to_list()
        assert result_dates[0] == datetime.date(2024, 1, 1)
        assert result_dates[1] == datetime.date(2024, 1, 8)
        assert result_dates[2] == datetime.date(2024, 1, 15)
