"""Unit tests for `src/risk/hygiene.py`: roll-date selection, adjusted-price
continuity across a known roll, the hygiene filter's accept/reject behaviour
on the documented CL/GC cases, the liquidity/liquid-contract-months screens,
and the new data-contract entry points (`build_risk_inputs`/
`assert_risk_inputs`, NEXT_PROMPT.md sec 4).

Split from `tests/test_commod_lib8.py` (NEXT_PROMPT.md sec 3.6) when the
underlying functions were promoted to `src/risk/hygiene.py`.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from risk import hygiene as H

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
        sched = H.build_roll_schedule(self._roll_calendar(), "CL", roll_days_before=6)
        row = sched.filter(pl.col("contract_month") == "2020-05")
        assert row["roll_date"][0] == date(2020, 4, 16)

    def test_roll_date_snaps_off_weekend(self):
        # 2020-05 anchor is 2020-04-22 (Wed); 1 business day before is
        # 2020-04-21 (Tue) -- no weekend crossing at N=1, sanity check the
        # snap logic separately with a case that does cross a weekend: N=3
        # from anchor 2020-04-22 (Wed) lands on Sun 2020-04-19 -> Fri 2020-04-17.
        sched = H.build_roll_schedule(self._roll_calendar(), "CL", roll_days_before=3)
        row = sched.filter(pl.col("contract_month") == "2020-05")
        rd = row["roll_date"][0]
        assert rd.weekday() < 5  # Mon-Fri only
        assert rd == date(2020, 4, 17)

    def test_roll_uses_first_notice_over_last_trade(self):
        sched = H.build_roll_schedule(self._roll_calendar(), "CL", roll_days_before=0)
        row = sched.filter(pl.col("contract_month") == "2020-05")
        assert row["anchor_date"][0] == date(
            2020, 4, 22
        )  # first_notice, not last_trade (04-21)


# --------------------------------------------------------------------------
# Continuous series construction
# --------------------------------------------------------------------------


def _tiny_universe():
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


class TestContinuousSeries:
    def test_roll_produces_continuous_backadj_price(self):
        ohlcv, contracts, roll_cal = _tiny_universe()
        # roll_days_before=0 -> anchor (first_notice 2020-01-21) minus 0 business
        # days = 2020-01-21, so F1 is contract 1 through 2020-01-17 (all 5 rows).
        curve = H.build_continuous_series(
            ohlcv, contracts, roll_cal, "CL", roll_days_before=0, n_legs=1
        )
        assert curve["contract_month_f1"].to_list() == ["2020-01"] * 5
        # no roll inside this window -> unadjusted, backadj, ratioadj all agree
        unadj = curve["log_return_unadj"].drop_nulls().to_numpy()
        backadj = curve["log_return_backadj"].drop_nulls().to_numpy()
        np.testing.assert_allclose(unadj, backadj, atol=1e-9)

    def test_backadjusted_price_continuous_across_roll(self):
        ohlcv, contracts, roll_cal = _tiny_universe()
        # anchor (first_notice) = 2020-01-21; 2 calendar days before = 2020-01-19
        # (Sun) -> snapped back to 2020-01-17 (Fri). The test window is
        # 2020-01-13..17, so F1 is contract 1 through 2020-01-16 and rolls to
        # contract 2 on 2020-01-17 (date >= roll_date).
        curve = H.build_continuous_series(
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
        ohlcv, contracts, roll_cal = _tiny_universe()
        curve = H.build_continuous_series(
            ohlcv, contracts, roll_cal, "CL", roll_days_before=0, n_legs=1
        )
        dte = curve["dte_f1"].to_numpy()
        assert np.all(np.diff(dte) <= 0)


class TestContinuousSeriesOhlcv:
    """`build_continuous_series_ohlcv` -- notebook 12's OHLCV extension,
    which `build_continuous_series` never provided (close-only)."""

    def _tiny_universe_ohlcv(self):
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
        close = [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.5, 58.0, 59.5, 61.0]
        ohlcv = pl.DataFrame(
            {
                "product": ["CL"] * 10,
                "contract_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "date": dates + dates,
                "open": [c - 0.5 for c in close],
                "high": [c + 1.0 for c in close],
                "low": [c - 1.0 for c in close],
                "close": close,
                # contract 2's volume spike sits on 2020-01-17 (its last row
                # here, since both contracts' rows cover the same 5 calendar
                # dates and the roll-date row is the one actually selected).
                "volume": [1000, 1100, 1200, 1300, 1400, 900, 950, 1000, 1100, 5000],
            }
        )
        return ohlcv, contracts, roll_cal

    def test_carries_ohlcv_and_flags_roll(self):
        ohlcv, contracts, roll_cal = self._tiny_universe_ohlcv()
        curve = H.build_continuous_series_ohlcv(
            ohlcv, contracts, roll_cal, "CL", roll_days_before=2
        )
        assert set(curve.columns) >= {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_roll",
            "open_backadj",
            "high_backadj",
            "low_backadj",
            "close_backadj",
        }
        # same roll_days_before=2 convention as the close-only test above:
        # rolls into contract 2 on 2020-01-17
        months = curve["contract_month"].to_list()
        assert "2020-01" in months and "2020-02" in months
        roll_idx = months.index("2020-02")
        assert bool(curve["is_roll"][roll_idx]) is True
        assert not any(curve["is_roll"].to_list()[:roll_idx])
        # raw volume is untouched, including the roll-date spike (5000)
        assert curve["volume"][roll_idx] == 5000
        assert roll_idx == len(months) - 1

    def test_backadj_ohlc_continuous_across_roll(self):
        ohlcv, contracts, roll_cal = self._tiny_universe_ohlcv()
        curve = H.build_continuous_series_ohlcv(
            ohlcv, contracts, roll_cal, "CL", roll_days_before=2
        )
        months = curve["contract_month"].to_list()
        roll_idx = months.index("2020-02")
        for col in ["open_backadj", "high_backadj", "low_backadj", "close_backadj"]:
            adj = curve[col].to_numpy()
            raw = curve[col.replace("_backadj", "")].to_numpy()
            assert abs(adj[roll_idx] - adj[roll_idx - 1]) < abs(
                raw[roll_idx] - raw[roll_idx - 1]
            )

    def test_no_roll_no_adjustment(self):
        ohlcv, contracts, roll_cal = self._tiny_universe_ohlcv()
        curve = H.build_continuous_series_ohlcv(
            ohlcv, contracts, roll_cal, "CL", roll_days_before=0
        )
        # roll_days_before=0 keeps the whole 5-day test window on contract 1
        assert not any(curve["is_roll"].to_list())
        np.testing.assert_allclose(
            curve["close_backadj"].to_numpy(), curve["close"].to_numpy()
        )


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
        flagged = H.flag_contaminated_rows(self._mixed_frame())
        cl_rows = flagged.filter(
            (pl.col("product") == "CL") & (pl.col("contract_id") == 752)
        )
        assert not cl_rows["contaminated"].any()

    def test_gc_spread_junk_rejected(self):
        flagged = H.flag_contaminated_rows(self._mixed_frame())
        gc_rows = flagged.filter(
            (pl.col("product") == "GC") & (pl.col("contract_id") == 2542)
        )
        assert gc_rows["contaminated"].all()

    def test_gc_legitimate_outrights_survive(self):
        flagged = H.flag_contaminated_rows(self._mixed_frame())
        legit = flagged.filter(
            (pl.col("product") == "GC") & (pl.col("contract_id") != 2542)
        )
        assert not legit["contaminated"].any()

    def test_apply_hygiene_filter_drops_only_contaminated(self):
        clean = H.apply_hygiene_filter(self._mixed_frame())
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
        screened = H.liquidity_screen(df, min_volume=50, min_active_contracts=2)
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
        liquid = H.liquid_contract_months(ohlcv, contracts, "PL", min_total_volume=5000)
        assert liquid == {"2020-01"}


# --------------------------------------------------------------------------
# The data contract: build_risk_inputs / assert_risk_inputs (NEXT_PROMPT.md
# sec 4, gate DC)
# --------------------------------------------------------------------------


def _clean_frame_for_product(ohlcv, contracts, roll_cal, product="CL"):
    return H.build_risk_inputs(
        ohlcv, contracts, roll_cal, product, roll_days_before=0, n_legs=1
    )


class TestBuildRiskInputs:
    def test_stamps_provenance_and_log_return_column(self):
        ohlcv, contracts, roll_cal = _tiny_universe()
        curve = _clean_frame_for_product(ohlcv, contracts, roll_cal)
        assert "log_return" in curve.columns
        assert getattr(curve, H.PROVENANCE_ATTR, None) == H.PROVENANCE_VALUE
        np.testing.assert_allclose(
            curve["log_return"].fill_null(0.0).to_numpy(),
            curve["log_return_ratioadj"].fill_null(0.0).to_numpy(),
        )


class TestAssertRiskInputs:
    def _long_clean_series(self, n=200, seed=SEED):
        rng = np.random.default_rng(seed)
        dates = [date(2020, 1, 1)]
        while len(dates) < n:
            nxt = dates[-1]
            from datetime import timedelta

            dates.append(nxt + timedelta(days=1))
        close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
        curve = pl.DataFrame(
            {
                "date": dates,
                "close_f1": close,
                "contract_month_f1": ["2020-01"] * n,
            }
        )
        log_ret = np.diff(np.log(close), prepend=np.nan)
        log_ret[0] = np.nan
        curve = curve.with_columns(pl.Series("log_return", log_ret))
        setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
        return curve

    def test_accepts_a_clean_frame(self):
        curve = self._long_clean_series()
        H.assert_risk_inputs(curve)  # must not raise

    def test_rejects_missing_provenance(self):
        curve = self._long_clean_series()
        curve = curve.clone()  # a fresh object without the stamped attribute
        with pytest.raises(H.RiskInputError, match="provenance"):
            H.assert_risk_inputs(curve)

    def test_rejects_too_few_observations(self):
        curve = self._long_clean_series(n=50)
        with pytest.raises(H.RiskInputError, match="observations"):
            H.assert_risk_inputs(curve)

    def test_rejects_large_unflagged_move(self):
        curve = self._long_clean_series()
        ret = curve["log_return"].to_numpy().copy()
        ret[50] = 0.65
        curve = curve.with_columns(pl.Series("log_return", ret))
        setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
        with pytest.raises(H.RiskInputError, match=r"\|log_return\|"):
            H.assert_risk_inputs(curve)

    def test_accepts_a_run_of_exactly_three_identical_closes(self):
        # 008's stale-bar audit found a worst-case run of 3 on real,
        # hygiene-passed data (CL's own clean curve has one) -- that is the
        # observed ceiling, not a rejection trigger.
        curve = self._long_clean_series()
        close = curve["close_f1"].to_numpy().copy()
        close[60:63] = close[60]
        curve = curve.with_columns(pl.Series("close_f1", close))
        setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
        H.assert_risk_inputs(curve)  # must not raise

    def test_rejects_a_run_of_four_identical_closes(self):
        curve = self._long_clean_series()
        close = curve["close_f1"].to_numpy().copy()
        close[60:64] = close[60]
        curve = curve.with_columns(pl.Series("close_f1", close))
        setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
        with pytest.raises(H.RiskInputError, match="consecutive identical"):
            H.assert_risk_inputs(curve)

    def test_accepts_a_single_zero_return_row(self):
        # A single day with no price change is normal, unremarkable market
        # data (real clean CL data has hundreds of these) and must not be
        # rejected -- only an extended frozen *window* is a red flag.
        curve = self._long_clean_series()
        ret = curve["log_return"].to_numpy().copy()
        ret[70] = 0.0
        curve = curve.with_columns(pl.Series("log_return", ret))
        setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
        H.assert_risk_inputs(curve)  # must not raise

    def test_rejects_a_frozen_realized_vol_window(self):
        curve = self._long_clean_series()
        ret = curve["log_return"].to_numpy().copy()
        ret[70 : 70 + H.REALIZED_VOL_WINDOW] = 0.0
        curve = curve.with_columns(pl.Series("log_return", ret))
        setattr(curve, H.PROVENANCE_ATTR, H.PROVENANCE_VALUE)
        with pytest.raises(H.RiskInputError, match="frozen"):
            H.assert_risk_inputs(curve)


# --------------------------------------------------------------------------
# The holdout-truncation guard on the fitting path (NEXT_PROMPT.md sec 2
# ground rule 1, sec 12) -- separate from assert_risk_inputs because
# risk.ingest.refresh calls assert_risk_inputs on current-dated data and
# must not be rejected by it (sec 7.4).
# --------------------------------------------------------------------------


_WELL_PAST_HOLDOUT = date(2026, 8, 12)  # fixed, not datetime.date.today() (DTZ011)


class TestAssertNotHoldout:
    def _frame_ending(self, end: date) -> pl.DataFrame:
        n = 200
        dates = [end - timedelta(days=n - 1 - i) for i in range(n)]
        return pl.DataFrame({"date": dates, "x": list(range(n))})

    def test_accepts_a_frame_that_ends_before_truncation(self):
        frame = self._frame_ending(H.TRUNCATION)
        H.assert_not_holdout(frame)  # must not raise

    def test_rejects_a_frame_that_extends_one_day_past_truncation(self):
        frame = self._frame_ending(H.TRUNCATION + timedelta(days=1))
        with pytest.raises(H.HoldoutLeakError, match="holdout"):
            H.assert_not_holdout(frame)

    def test_rejects_a_frame_that_extends_well_past_the_holdout(self):
        frame = self._frame_ending(_WELL_PAST_HOLDOUT)
        with pytest.raises(H.HoldoutLeakError):
            H.assert_not_holdout(frame)

    def test_ignores_a_frame_with_no_date_column(self):
        frame = pl.DataFrame({"x": [1, 2, 3]})
        H.assert_not_holdout(frame)  # must not raise -- nothing to check

    def test_ignores_a_frame_with_a_differently_named_date_column(self):
        frame = self._frame_ending(_WELL_PAST_HOLDOUT).rename({"date": "as_of"})
        H.assert_not_holdout(frame)  # must not raise under the default date_col

    def test_custom_date_col(self):
        frame = self._frame_ending(_WELL_PAST_HOLDOUT).rename({"date": "as_of"})
        with pytest.raises(H.HoldoutLeakError):
            H.assert_not_holdout(frame, date_col="as_of")
