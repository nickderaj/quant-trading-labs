"""Unit tests for the notebook-8-local machinery that was NOT promoted to
`src/risk/` (NEXT_PROMPT.md sec 3.6): cost arithmetic, stale-bar detection,
the colour palette, ACF/Ljung-Box/leverage/Samuelson/seasonality statistics,
GJR+zoo two-stage fitting, term-structure/seasonal state, and the futures
cost model applied to a weights panel. Mirrors tests/test_dist_lib5.py's
conventions.

The risk-engine tests that used to live here (roll schedule, continuous
series, hygiene filter, liquidity screen, numerical PIT, RiskModel,
ewma_vol, portfolio_risk, kupiec_by_state) moved to `tests/test_risk_hygiene.py`,
`tests/test_risk_model.py`, `tests/test_risk_portfolio.py`, and
`tests/test_risk_calibration.py` when their underlying functions were
promoted to `src/risk/`.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "research" / "tmp"))
sys.path.insert(0, str(_ROOT / "src"))

import commod_lib8 as C

SEED = 0


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------


class TestCostModel:
    def test_all_products_have_specs(self):
        for p in C.PRODUCTS:
            assert p in C.CONTRACT_SPECS, p

    def test_cl_round_turn_cost_matches_hand_calc(self):
        # $2.50 commission + 1 tick * $10/tick = $12.50
        cost = C.round_turn_cost_per_contract("CL", tick_multiplier=1.0)
        assert cost == pytest.approx(12.50)

    def test_thin_product_gets_wider_slippage(self):
        pl_cost_1x = C.round_turn_cost_per_contract("PL", tick_multiplier=1.0)
        # thin-product multiplier doubles the slippage leg vs a hypothetical
        # non-thin product with identical tick economics
        spec = C.CONTRACT_SPECS["PL"]
        base_slip = spec["tick_value"]
        assert pl_cost_1x == pytest.approx(
            spec["commission_per_contract"] + 2 * base_slip
        )

    def test_stress_and_optimistic_variants_order_correctly(self):
        opt = C.round_turn_cost_per_contract("CL", tick_multiplier=0.5)
        base = C.round_turn_cost_per_contract("CL", tick_multiplier=1.0)
        stress = C.round_turn_cost_per_contract("CL", tick_multiplier=2.0)
        assert opt < base < stress

    def test_cost_per_unit_notional_positive_and_finite(self):
        c = C.cost_per_unit_notional("CL", price=75.0)
        assert 0 < c < 1


# --------------------------------------------------------------------------
# Stale-bar audit
# --------------------------------------------------------------------------


class TestStaleBarRuns:
    def test_detects_a_run_of_identical_closes(self):
        close = np.array([10.0, 10.0, 10.0, 11.0, 12.0, 12.0])
        stats = C.stale_bar_runs(close)
        assert stats["n_runs"] == 2
        assert stats["max_run"] == 3
        assert (
            stats["n_stale_days"] == 3
        )  # 2 extra days in the 3-run + 1 extra in the 2-run

    def test_no_repeats_gives_zero_runs(self):
        close = np.array([10.0, 11.0, 12.0, 13.0])
        stats = C.stale_bar_runs(close)
        assert stats["n_runs"] == 0
        assert stats["max_run"] == 0


# --------------------------------------------------------------------------
# Product colour palette
# --------------------------------------------------------------------------


class TestPalette:
    def test_every_product_has_a_colour_and_sector(self):
        for p in C.PRODUCTS:
            assert p in C.SECTOR
            assert C.product_color(p).startswith("#")

    def test_colors_are_unique_within_a_sector(self):
        for sector in ("energy", "metals", "grains"):
            members = [p for p in C.PRODUCTS if C.SECTOR[p] == sector]
            colors = [C.product_color(p) for p in members]
            assert len(set(colors)) == len(colors)

    def test_es_is_flagged_as_control_not_a_commodity(self):
        assert C.SECTOR["ES"] == "control"


# --------------------------------------------------------------------------
# Phase 1 machinery: ACF/Ljung-Box, leverage effect, Samuelson, seasonality
# --------------------------------------------------------------------------


class TestAcfLjungBox:
    def test_acf_of_white_noise_is_small(self):
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(5000)
        rho = C.acf(x, 10)
        assert np.all(np.abs(rho) < 0.1)

    def test_acf_detects_ar1_persistence(self):
        rng = np.random.default_rng(SEED)
        n = 5000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + rng.standard_normal()
        rho = C.acf(x, 5)
        assert rho[0] == pytest.approx(0.7, abs=0.05)

    def test_ljung_box_rejects_ar1_series(self):
        rng = np.random.default_rng(SEED)
        n = 2000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.5 * x[i - 1] + rng.standard_normal()
        result = C.ljung_box_test(x, lags=10)
        assert result["p_value"] < 0.01

    def test_ljung_box_does_not_reject_white_noise(self):
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(2000)
        result = C.ljung_box_test(x, lags=10)
        assert result["p_value"] > 0.01


class TestLeverageCorrelation:
    def test_recovers_known_negative_correlation(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000)
        vol_next = -0.5 * r + rng.standard_normal(3000) * 0.3
        result = C.leverage_correlation(r, vol_next, n_boot=200)
        assert result["corr"] < -0.3
        assert result["ci_lo"] < result["corr"] < result["ci_hi"]

    def test_recovers_known_positive_correlation(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000)
        vol_next = 0.5 * r + rng.standard_normal(3000) * 0.3
        result = C.leverage_correlation(r, vol_next, n_boot=200)
        assert result["corr"] > 0.3


class TestSamuelsonEffect:
    def test_detects_rising_vol_near_expiry(self):
        rng = np.random.default_rng(SEED)
        dte = rng.integers(0, 200, 5000).astype(float)
        vol = 0.1 + 0.5 * np.exp(-dte / 20) + rng.standard_normal(5000) * 0.01
        result = C.samuelson_effect(vol, dte)
        buckets = result["buckets"]
        near = next(b for b in buckets if b["dte_lo"] == 0)
        far = next(b for b in buckets if b["dte_lo"] >= 90)
        assert near["mean_vol"] > far["mean_vol"]


class TestSeasonality:
    def test_month_of_year_stats_shape(self):
        from datetime import date as dt

        dates = [dt(2020, (i % 12) + 1, 15) for i in range(240)]
        rng = np.random.default_rng(SEED)
        returns = rng.standard_normal(240) * 0.01
        result = C.month_of_year_stats(dates, returns)
        assert len(result["months"]) == 12

    def test_day_of_week_stats_shape(self):
        from datetime import date, timedelta

        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(500)]
        rng = np.random.default_rng(SEED)
        returns = rng.standard_normal(500) * 0.01
        result = C.day_of_week_stats(dates, returns)
        assert len(result["weekdays"]) == 5


class TestNamedEvents:
    def test_negative_wti_event_found_for_cl(self):
        events = C.events_in_window("CL", "2020-04-01", "2020-05-01")
        names = [e["name"] for e in events]
        assert any("negative WTI" in n for n in names)

    def test_event_not_returned_for_unrelated_product(self):
        events = C.events_in_window("ZC", "2020-04-15", "2020-04-25")
        names = [e["name"] for e in events]
        assert not any("negative WTI" in n for n in names)


# --------------------------------------------------------------------------
# Phase 3: GJR + zoo two-stage fitting
# --------------------------------------------------------------------------


class TestGjrZooTwoStage:
    def _synthetic_returns(self, n=1500, seed=SEED):
        rng = np.random.default_rng(seed)
        omega, alpha, gamma, beta = 1e-6, 0.03, 0.05, 0.9
        sig2 = omega / (1 - alpha - gamma / 2 - beta)
        r = np.empty(n)
        for t in range(n):
            r[t] = np.sqrt(sig2) * rng.standard_normal()
            shock = r[t] ** 2
            lev = gamma * shock if r[t] < 0 else 0.0
            sig2 = omega + alpha * shock + lev + beta * sig2
        return r

    def test_fit_recovers_plausible_persistence(self):
        import sys as _sys

        _sys.path.insert(0, "src/research/tmp")
        import densities

        r = self._synthetic_returns()
        fit = C.fit_gjr_zoo_two_stage(r, densities.ged)
        assert fit is not None
        assert 0 <= fit["alpha"] + fit["gamma"] / 2 + fit["beta"] < 1
        assert fit["family"] == "ged"
        assert len(fit["shape"]) == densities.ged.N_SHAPE

    def test_rolling_forecast_produces_positive_variance(self):
        import sys as _sys

        _sys.path.insert(0, "src/research/tmp")
        import densities

        r = self._synthetic_returns(n=1200)
        forecast, fits = C.rolling_gjr_forecast_zoo(
            r, refit_every=200, min_train=300, family_module=densities.ged
        )
        assert len(fits) >= 2
        valid = np.isfinite(forecast)
        assert valid.sum() > 0
        assert np.all(forecast[valid] > 0)


# --------------------------------------------------------------------------
# Phase 4: term-structure / seasonal / macro conditioning
# --------------------------------------------------------------------------


class TestTermStructureState:
    def test_backwardation_and_contango_labeled_correctly(self):
        curve = pl.DataFrame(
            {
                "date": [date(2020, 1, 1), date(2020, 1, 2)],
                "close_f1": [100.0, 100.0],
                "dte_f1": [10, 10],
                "close_f2": [
                    95.0,
                    105.0,
                ],  # row0: F2<F1 -> backwardation; row1: F2>F1 -> contango
                "dte_f2": [40, 40],
            }
        )
        out = C.term_structure_state(curve)
        assert out["term_structure_state"].to_list() == ["backwardation", "contango"]
        assert out["roll_slope_annualized"][0] < 0
        assert out["roll_slope_annualized"][1] > 0


class TestSeasonalState:
    def test_ng_heating_season(self):
        dates = [date(2020, 1, 15), date(2020, 7, 15)]
        states = C.seasonal_state(dates, "NG")
        assert states == ["heating_season", "off_season"]

    def test_product_without_cycle_is_na(self):
        dates = [date(2020, 1, 15), date(2020, 7, 15)]
        states = C.seasonal_state(dates, "GC")
        assert states == ["na", "na"]

    def test_grain_harvest_window(self):
        states = C.seasonal_state([date(2020, 9, 15)], "ZC")
        assert states == ["harvest"]


# --------------------------------------------------------------------------
# Phase 5: futures cost model applied to a cross-sectional weights panel
# --------------------------------------------------------------------------


class TestPortfolioCostsFutures:
    def _setup(self):
        dts = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
        weights = pl.DataFrame(
            {
                "datetime": [dts[0], dts[1], dts[2], dts[0], dts[1], dts[2]],
                "symbol": ["CL", "CL", "CL", "GC", "GC", "GC"],
                "weight": [0.5, 0.5, -0.5, -0.5, 0.0, 0.5],
            }
        )
        prices = pl.DataFrame(
            {
                "datetime": [dts[0], dts[1], dts[2], dts[0], dts[1], dts[2]],
                "symbol": ["CL", "CL", "CL", "GC", "GC", "GC"],
                "close": [75.0, 76.0, 74.0, 1900.0, 1910.0, 1890.0],
            }
        )
        return weights, prices

    def test_costs_are_nonnegative_and_zero_when_no_turnover(self):
        weights, prices = self._setup()
        costs = C.portfolio_costs_futures(weights, prices)
        assert (costs["cost_frac"] >= 0).all()
        # day 2 (2020-01-02): CL weight unchanged (0.5->0.5), only GC moves
        day2 = costs.filter(pl.col("datetime") == date(2020, 1, 2))
        day1 = costs.filter(pl.col("datetime") == date(2020, 1, 1))
        assert day2["cost_frac"][0] > 0
        assert day1["cost_frac"][0] > 0  # first day: full entry weight on both

    def test_add_portfolio_costs_futures_net_le_gross(self):
        weights, prices = self._setup()
        costs = C.portfolio_costs_futures(weights, prices)
        trade_frame = pl.DataFrame(
            {
                "datetime": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
                "trade_log_return": [0.01, -0.005, 0.02],
            }
        )
        out = C.add_portfolio_costs_futures(trade_frame, costs)
        # tripwire from sec 7: net must never exceed gross
        assert (out["trade_log_return_net"] <= out["trade_log_return"] + 1e-12).all()

    def test_futures_portfolio_metrics_reports_gross_and_net(self):
        weights, prices = self._setup()
        costs = C.portfolio_costs_futures(weights, prices)
        trade_frame = pl.DataFrame(
            {
                "datetime": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
                "trade_log_return": [0.01, -0.005, 0.02],
                "turnover": [1.0, 0.5, 1.0],
            }
        )
        metrics = C.futures_portfolio_metrics(
            trade_frame, costs, annualized_rate=16.0, label="test"
        )
        assert "sharpe" in metrics
        assert "sharpe_net" in metrics
        assert metrics["sharpe_net"] <= metrics["sharpe"] + 1e-9
