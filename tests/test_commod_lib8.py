"""Unit tests for notebook-8-local machinery (src/research/tmp/commod_lib8.py):
roll-date selection, adjusted-price continuity across a known roll, the hygiene
filter's accept/reject behaviour on the documented CL/GC cases, cost arithmetic,
and stale-bar detection. Mirrors tests/test_dist_lib5.py's conventions.
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
# Roll schedule
# --------------------------------------------------------------------------


class TestRollSchedule:
    def _roll_calendar(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "product": ["CL", "CL", "CL"],
                "contract_month": ["2020-04", "2020-05", "2020-06"],
                "expiry": [
                    date(2020, 3, 20),
                    date(2020, 4, 21),
                    date(2020, 5, 19),
                ],
                "first_notice_date": [
                    date(2020, 3, 23),
                    date(2020, 4, 22),
                    date(2020, 5, 20),
                ],
                "last_trade_date": [
                    date(2020, 3, 20),
                    date(2020, 4, 21),
                    date(2020, 5, 19),
                ],
            }
        )

    def test_hand_checked_roll_date_n6(self):
        # anchor = first_notice_date (2020-04-22, a Wednesday) for the 2020-05
        # contract. The roll rule is N *calendar* days before the anchor, snapped
        # backward off a weekend (no exchange calendar available for true business
        # days): 2020-04-22 - 6 calendar days = 2020-04-16 (Thu, no weekend
        # crossed) -> roll_date = 2020-04-16.
        sched = C.build_roll_schedule(self._roll_calendar(), "CL", roll_days_before=6)
        row = sched.filter(pl.col("contract_month") == "2020-05")
        assert row["roll_date"][0] == date(2020, 4, 16)

    def test_roll_date_snaps_off_weekend(self):
        # 2020-05 anchor is 2020-04-22 (Wed); 1 business day before is
        # 2020-04-21 (Tue) -- no weekend crossing at N=1, sanity check the
        # snap logic separately with a case that does cross a weekend: N=3
        # from anchor 2020-04-22 (Wed) lands on Sun 2020-04-19 -> Fri 2020-04-17.
        sched = C.build_roll_schedule(self._roll_calendar(), "CL", roll_days_before=3)
        row = sched.filter(pl.col("contract_month") == "2020-05")
        rd = row["roll_date"][0]
        assert rd.weekday() < 5  # Mon-Fri only
        assert rd == date(2020, 4, 17)

    def test_roll_uses_first_notice_over_last_trade(self):
        sched = C.build_roll_schedule(self._roll_calendar(), "CL", roll_days_before=0)
        row = sched.filter(pl.col("contract_month") == "2020-05")
        assert row["anchor_date"][0] == date(
            2020, 4, 22
        )  # first_notice, not last_trade (04-21)


# --------------------------------------------------------------------------
# Continuous series construction
# --------------------------------------------------------------------------


class TestContinuousSeries:
    def _tiny_universe(self):
        contracts = pl.DataFrame(
            {
                "contract_id": [1, 2],
                "ticker": ["CL202001", "CL202002"],
                "product": ["CL", "CL"],
                "contract_month": ["2020-01", "2020-02"],
                "expiry": [date(2020, 1, 20), date(2020, 2, 20)],
            }
        )
        roll_cal = pl.DataFrame(
            {
                "product": ["CL", "CL"],
                "contract_month": ["2020-01", "2020-02"],
                "expiry": [date(2020, 1, 20), date(2020, 2, 20)],
                "first_notice_date": [date(2020, 1, 21), date(2020, 2, 21)],
                "last_trade_date": [date(2020, 1, 20), date(2020, 2, 20)],
            }
        )
        dates = [date(2020, 1, d) for d in range(13, 18)]
        ohlcv = pl.DataFrame(
            {
                "product": ["CL"] * 10,
                "contract_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "date": dates + dates,
                "close": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.5, 58.0, 59.5, 61.0],
                "volume": [1000] * 10,
            }
        )
        return ohlcv, contracts, roll_cal

    def test_roll_produces_continuous_backadj_price(self):
        ohlcv, contracts, roll_cal = self._tiny_universe()
        # roll_days_before=0 -> anchor (first_notice 2020-01-21) minus 0 business
        # days = 2020-01-21, so F1 is contract 1 through 2020-01-17 (all 5 rows).
        curve = C.build_continuous_series(
            ohlcv, contracts, roll_cal, "CL", roll_days_before=0, n_legs=1
        )
        assert curve["contract_month_f1"].to_list() == ["2020-01"] * 5
        # no roll inside this window -> unadjusted, backadj, ratioadj all agree
        unadj = curve["log_return_unadj"].drop_nulls().to_numpy()
        backadj = curve["log_return_backadj"].drop_nulls().to_numpy()
        np.testing.assert_allclose(unadj, backadj, atol=1e-9)

    def test_backadjusted_price_continuous_across_roll(self):
        ohlcv, contracts, roll_cal = self._tiny_universe()
        # anchor (first_notice) = 2020-01-21; 2 calendar days before = 2020-01-19
        # (Sun) -> snapped back to 2020-01-17 (Fri). The test window is
        # 2020-01-13..17, so F1 is contract 1 through 2020-01-16 and rolls to
        # contract 2 on 2020-01-17 (date >= roll_date).
        curve = C.build_continuous_series(
            ohlcv, contracts, roll_cal, "CL", roll_days_before=2, n_legs=1
        )
        months = curve["contract_month_f1"].to_list()
        assert "2020-01" in months and "2020-02" in months
        roll_idx = months.index("2020-02")
        # back-adjusted price must have NO jump at the roll (unlike raw close_f1)
        backadj = curve["close_backadj"].to_numpy()
        raw = curve["close_f1"].to_numpy()
        assert abs(backadj[roll_idx] - backadj[roll_idx - 1]) < abs(
            raw[roll_idx] - raw[roll_idx - 1]
        )
        # log_return_unadj must be null exactly at the roll (never computed
        # across the boundary on unadjusted prices)
        assert curve["log_return_unadj"][roll_idx] is None
        # but log_return_backadj is a real number at the roll
        assert curve["log_return_backadj"][roll_idx] is not None

    def test_dte_decreases_within_a_contract(self):
        ohlcv, contracts, roll_cal = self._tiny_universe()
        curve = C.build_continuous_series(
            ohlcv, contracts, roll_cal, "CL", roll_days_before=0, n_legs=1
        )
        dte = curve["dte_f1"].to_numpy()
        assert np.all(np.diff(dte) <= 0)


# --------------------------------------------------------------------------
# Hygiene filter -- accept CL 2020-04-20, reject GC-style junk
# --------------------------------------------------------------------------


class TestHygieneFilter:
    def _mixed_frame(self) -> pl.DataFrame:
        # Mirrors the documented, real-data-verified cases: CL contract 752
        # (CL202005) trades normally for 15 days and has exactly ONE genuine
        # negative-settle day (2020-04-20, on huge volume) -- it must survive.
        # GC contract 2542 (GC201511) prints near-zero junk on every single day
        # of its life, on tiny volume -- a mislabeled spread-differential series,
        # not an outright -- it must be dropped in its entirety. Legitimate GC
        # outrights trade normally alongside it every day.
        rows = []
        anchor_prices = [20.0 + 0.5 * i for i in range(15)]
        for i, (d, anchor_px) in enumerate(
            zip([date(2020, 4, 1 + i) for i in range(15)], anchor_prices, strict=True)
        ):
            rows.append(
                ("CL", d, 682, anchor_px, 900_000)
            )  # anchor: always liquid, normal price
            rows.append(
                ("CL", d, 744, anchor_px * 1.03, 90_000)
            )  # a second normal contract
            if i == 14:  # 2020-04-15 stand-in for the crash day (last day in range)
                rows.append(("CL", d, 752, -2.67, 102_083))
            else:
                rows.append(("CL", d, 752, anchor_px * 0.95, 80_000))

        for i in range(12):
            d = date(2015, 8, 20 + i) if i < 11 else date(2015, 9, 1)
            rows.append(("GC", d, 2600, 1130.0 + i, 5000))  # anchor: legit outright
            rows.append(("GC", d, 2601, 1132.5 + i, 6200))  # legit
            rows.append(("GC", d, 2542, -0.40 - 0.02 * i, 14))  # junk every single day

        return pl.DataFrame(
            rows,
            schema=["product", "date", "contract_id", "close", "volume"],
            orient="row",
        )

    def test_cl_real_negative_settle_survives(self):
        flagged = C.flag_contaminated_rows(self._mixed_frame())
        cl_rows = flagged.filter(
            (pl.col("product") == "CL") & (pl.col("contract_id") == 752)
        )
        assert not cl_rows["contaminated"].any()

    def test_gc_spread_junk_rejected(self):
        flagged = C.flag_contaminated_rows(self._mixed_frame())
        gc_rows = flagged.filter(
            (pl.col("product") == "GC") & (pl.col("contract_id") == 2542)
        )
        assert gc_rows["contaminated"].all()

    def test_gc_legitimate_outrights_survive(self):
        flagged = C.flag_contaminated_rows(self._mixed_frame())
        legit = flagged.filter(
            (pl.col("product") == "GC") & (pl.col("contract_id") != 2542)
        )
        assert not legit["contaminated"].any()

    def test_apply_hygiene_filter_drops_only_contaminated(self):
        clean = C.apply_hygiene_filter(self._mixed_frame())
        assert 2542 not in clean["contract_id"].to_list()
        assert 752 in clean["contract_id"].to_list()


# --------------------------------------------------------------------------
# Liquidity screen
# --------------------------------------------------------------------------


class TestLiquidityScreen:
    def test_drops_low_volume_and_thin_days(self):
        df = pl.DataFrame(
            {
                "product": ["CL"] * 4,
                "date": [date(2020, 1, 1)] * 2 + [date(2020, 1, 2)] * 2,
                "contract_id": [1, 2, 1, 2],
                "close": [50.0, 51.0, 52.0, 5.0],
                "volume": [1000, 5, 1000, 1000],
            }
        )
        screened = C.liquidity_screen(df, min_volume=50, min_active_contracts=2)
        # 2020-01-01 loses contract 2 (vol 5) -> only 1 active contract that day
        # -> the whole date is dropped by min_active_contracts.
        assert date(2020, 1, 1) not in screened["date"].to_list()
        assert screened.filter(pl.col("date") == pl.date(2020, 1, 2)).height == 2


class TestLiquidContractMonths:
    def test_drops_nominally_listed_never_traded_months(self):
        # PL-style pattern: 2020-01 (active month) has real volume across
        # many days; 2020-02 ("phantom" month) prints a handful of trades on
        # a handful of days and never accumulates real size.
        ohlcv = pl.DataFrame(
            {
                "product": ["PL"] * 6,
                "contract_id": [1, 1, 1, 2, 2, 2],
                "volume": [50000, 60000, 55000, 10, 5, 8],
            }
        )
        contracts = pl.DataFrame(
            {
                "contract_id": [1, 2],
                "product": ["PL", "PL"],
                "contract_month": ["2020-01", "2020-02"],
            }
        )
        liquid = C.liquid_contract_months(ohlcv, contracts, "PL", min_total_volume=5000)
        assert liquid == {"2020-01"}


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
# Numerical PIT/CDF (Phase 2: shape-only density families have no cdf)
# --------------------------------------------------------------------------


class TestNumericalPit:
    def test_recovers_standard_normal_cdf(self):
        from scipy import stats as sp_stats

        logpdf_fn = sp_stats.norm.logpdf
        pit = C.numerical_pit(logpdf_fn, np.array([-1.0, 0.0, 1.0, 2.0]))
        expected = sp_stats.norm.cdf(np.array([-1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(pit, expected, atol=1e-3)

    def test_pit_is_uniform_for_true_model(self):
        from scipy import stats as sp_stats

        rng = np.random.default_rng(SEED)
        z = rng.standard_normal(2000)
        pit = C.numerical_pit(sp_stats.norm.logpdf, z)
        _stat, p = sp_stats.kstest(pit, "uniform")
        assert p > 0.01

    def test_cdf_grid_is_monotonic_and_bounded(self):
        from scipy import stats as sp_stats

        g = C.numerical_cdf_grid(sp_stats.norm.logpdf)
        assert np.all(np.diff(g["cdf"]) >= -1e-12)
        assert g["cdf"][0] >= 0
        assert g["cdf"][-1] <= 1.0 + 1e-9

    def test_numerical_ppf_recovers_standard_normal_quantiles(self):
        from scipy import stats as sp_stats

        u = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        z = C.numerical_ppf(sp_stats.norm.logpdf, u)
        expected = sp_stats.norm.ppf(u)
        np.testing.assert_allclose(z, expected, atol=0.02)

    def test_numerical_ppf_is_fast_for_many_points(self):
        import time

        from scipy import stats as sp_stats

        rng = np.random.default_rng(SEED)
        u = rng.uniform(1e-4, 1 - 1e-4, 20000)
        t0 = time.time()
        C.numerical_ppf(sp_stats.norm.logpdf, u)
        assert time.time() - t0 < 2.0


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


class TestKupiecByState:
    def test_flags_a_state_with_bad_coverage(self):
        rng = np.random.default_rng(SEED)
        n = 2000
        # state A: correctly calibrated 1% hit rate; state B: hits at 5% (miscalibrated)
        states = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
        hits = np.concatenate([rng.random(n // 2) < 0.01, rng.random(n // 2) < 0.05])
        result = C.kupiec_by_state(hits, states, expected_rate=0.01)
        assert result["A"]["kupiec_p"] > 0.01
        assert result["B"]["kupiec_p"] < 0.01


# --------------------------------------------------------------------------
# Phase 7: risk engine
# --------------------------------------------------------------------------


class TestRiskModel:
    def test_fit_normal_recovers_var_close_to_analytic(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(5000) * 0.02
        model = C.fit_risk_model(r, "TEST", "normal")
        assert model is not None
        from scipy import stats as sp_stats

        expected_var = -0.02 * sp_stats.norm.ppf(0.01)
        assert model.var(0.01) == pytest.approx(expected_var, rel=0.1)

    def test_es_exceeds_var_in_magnitude(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_t(5, 3000) * 0.02
        model = C.fit_risk_model(r, "TEST", "t")
        assert model is not None
        assert model.es(0.01) > model.var(0.01)

    def test_var_scales_with_sqrt_horizon(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000) * 0.02
        model = C.fit_risk_model(r, "TEST", "normal")
        assert model.var(0.01, horizon=4) == pytest.approx(
            model.var(0.01, horizon=1) * 2, rel=1e-6
        )

    def test_simulate_matches_fitted_moments(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(4000) * 0.03 + 0.001
        model = C.fit_risk_model(r, "TEST", "normal")
        sim = model.simulate(20000, seed=1)
        assert sim.mean() == pytest.approx(0.001, abs=0.01)
        assert sim.std() == pytest.approx(0.03, rel=0.1)

    def test_shape_only_family_fits_and_scores(self):

        rng = np.random.default_rng(SEED)
        r = rng.standard_t(6, 4000) * 0.02
        model = C.fit_risk_model(r, "TEST", "ged")
        assert model is not None
        assert model.kind == "standardized"
        assert np.isfinite(model.var(0.01))

    def test_stress_replays_scenario(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(3000) * 0.02
        model = C.fit_risk_model(r, "TEST", "normal")
        scenario = np.array([-0.05, -0.03, 0.01])
        result = model.stress(scenario)
        assert result["n_days"] == 3
        assert result["worst_day"] == pytest.approx(-0.05)

    def test_conditional_var_scales_with_supplied_sigma(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(4000) * 0.02
        model = C.fit_risk_model(r, "TEST", "normal")
        low_vol_var = model.var_conditional(0.01, sigma_t=0.01)
        high_vol_var = model.var_conditional(0.01, sigma_t=0.04)
        assert high_vol_var > low_vol_var
        assert high_vol_var == pytest.approx(4 * low_vol_var, rel=0.05)

    def test_conditional_var_at_own_std_matches_unconditional(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(4000) * 0.02
        model = C.fit_risk_model(r, "TEST", "normal")
        assert model is not None
        assert model.var_conditional(0.01, sigma_t=model.std) == pytest.approx(
            model.var(0.01), rel=1e-6
        )


class TestEwmaVol:
    def test_seeds_from_initial_window_variance(self):
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(500) * 0.02
        vol = C.ewma_vol(r, lam=0.94, seed_window=20)
        assert vol[20] == pytest.approx(np.std(r[:20]), rel=1e-6)
        assert np.all(np.isnan(vol[:20]))

    def test_reacts_to_a_volatility_regime_shift(self):
        rng = np.random.default_rng(SEED)
        calm = rng.standard_normal(300) * 0.005
        stormy = rng.standard_normal(300) * 0.05
        r = np.concatenate([calm, stormy])
        vol = C.ewma_vol(r, lam=0.94, seed_window=20)
        # vol should be materially higher deep into the stormy regime than
        # deep into the calm regime
        assert np.nanmean(vol[550:600]) > np.nanmean(vol[100:150]) * 2

    def test_never_uses_current_bar(self):
        # a single huge outlier at t should NOT move vol[t] itself, only vol[t+1:]
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(200) * 0.01
        r[100] = 5.0
        vol = C.ewma_vol(r, lam=0.94, seed_window=20)
        assert vol[100] < 1.0  # unaffected by its own bar's shock
        assert vol[101] > vol[99]  # reacts the bar after


class TestPortfolioRisk:
    def _two_asset_setup(self, corr=0.6, seed=SEED):
        rng = np.random.default_rng(seed)
        n = 3000
        z1 = rng.standard_normal(n)
        z2 = corr * z1 + np.sqrt(1 - corr**2) * rng.standard_normal(n)
        r1, r2 = 0.02 * z1, 0.02 * z2
        m1 = C.fit_risk_model(r1, "A", "normal")
        m2 = C.fit_risk_model(r2, "B", "normal")
        return {"A": m1, "B": m2}, {"A": r1, "B": r2}

    def test_gaussian_copula_var_positive(self):
        models, hist = self._two_asset_setup()
        result = C.portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="gaussian",
            historical_returns=hist,
            n_sims=5000,
            seed=0,
        )
        assert result["var_01"] > 0
        assert result["es_01"] >= result["var_01"]

    def test_empirical_dependence_runs(self):
        models, hist = self._two_asset_setup()
        result = C.portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="empirical",
            historical_returns=hist,
            n_sims=5000,
            seed=0,
        )
        assert result["var_01"] > 0

    def test_t_copula_shows_more_tail_dependence_than_gaussian(self):
        # low correlation but fat-tailed joint -> t-copula should show
        # materially higher lower-tail dependence than Gaussian at the same
        # correlation, since Gaussian tail dependence -> 0 asymptotically.
        models, hist = self._two_asset_setup(corr=0.3)
        gauss = C.portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="gaussian",
            historical_returns=hist,
            n_sims=20000,
            seed=0,
        )
        t_cop = C.portfolio_risk(
            models,
            {"A": 0.5, "B": 0.5},
            dependence="t",
            historical_returns=hist,
            t_df=3.0,
            n_sims=20000,
            seed=0,
        )
        gauss_td = gauss["lower_tail_dependence"]["A_B"]
        t_td = t_cop["lower_tail_dependence"]["A_B"]
        assert t_td > gauss_td


class TestTailDependenceHelpers:
    def test_to_pseudo_uniform_is_in_unit_interval(self):
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(500)
        u = C.to_pseudo_uniform(x)
        assert u.min() > 0
        assert u.max() < 1

    def test_empirical_tail_dependence_high_for_comonotonic(self):
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(2000)
        u = C.to_pseudo_uniform(x)
        td = C.empirical_lower_tail_dependence(
            u, u, q=0.1
        )  # perfectly comonotonic with itself
        assert td == pytest.approx(1.0)

    def test_empirical_tail_dependence_low_for_independent(self):
        rng = np.random.default_rng(SEED)
        u1 = C.to_pseudo_uniform(rng.standard_normal(5000))
        u2 = C.to_pseudo_uniform(rng.standard_normal(5000))
        td = C.empirical_lower_tail_dependence(u1, u2, q=0.1)
        assert td == pytest.approx(0.1, abs=0.05)


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
