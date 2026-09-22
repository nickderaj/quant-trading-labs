import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "research" / "tmp")
)
import lib24


def _synthetic_returns(n=2000, sigma_daily=0.02, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0, sigma_daily, size=n)


def test_mad_vol_recovers_sigma_on_gaussian_data():
    r = _synthetic_returns(n=5000, sigma_daily=0.02, seed=1)
    expected = 0.02 * np.sqrt(252)
    assert lib24.mad_vol(r) == pytest.approx(expected, rel=0.05)


def test_mad_vol_robust_to_single_outlier_std_is_not():
    r = _synthetic_returns(n=2000, sigma_daily=0.02, seed=2)
    clean_mad = lib24.mad_vol(r)
    clean_std = lib24.std_vol(r)
    r_contaminated = r.copy()
    r_contaminated[0] = 500 * 0.02  # a single ~500x outlier
    contaminated_mad = lib24.mad_vol(r_contaminated)
    contaminated_std = lib24.std_vol(r_contaminated)
    assert contaminated_mad == pytest.approx(clean_mad, rel=0.05)
    assert contaminated_std > clean_std * 5


def test_winsor_vol_between_mad_and_contaminated_std():
    r = _synthetic_returns(n=2000, sigma_daily=0.02, seed=3)
    r[0] = 500 * 0.02
    m = lib24.mad_vol(r)
    w = lib24.winsor_vol(r)
    s = lib24.std_vol(r)
    assert m < w < s or abs(w - m) < abs(s - m)


def test_returns_never_chained_across_contract():
    df = pd.DataFrame(
        {
            "contract_id": [1, 1, 2, 2],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]
            ),
            "close": [100.0, 110.0, 50.0, 55.0],
        }
    )
    df = df.sort_values(["contract_id", "date"]).reset_index(drop=True)
    df["log_close"] = np.log(df["close"])
    df["r"] = df.groupby("contract_id")["log_close"].diff()
    # contract 2's first row must not pick up contract 1's last close
    first_row_c2 = df[(df["contract_id"] == 2) & (df["date"] == "2020-01-01")]
    assert first_row_c2["r"].isna().all()


def test_usable_returns_drops_synthetic_40_day_gap():
    dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-10")]
    df = pd.DataFrame(
        {
            "contract_id": [1, 1],
            "date": dates,
            "close": [100.0, 105.0],
            "volume": [1000, 1000],
            "expiry": [pd.Timestamp("2020-06-01")] * 2,
        }
    )
    df = df.sort_values(["contract_id", "date"]).reset_index(drop=True)
    df["log_close"] = np.log(df["close"])
    df["r"] = df.groupby("contract_id")["log_close"].diff()
    df["gap"] = df.groupby("contract_id")["date"].diff().dt.days
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    out = lib24.usable_returns(df, max_gap=1)
    assert len(out) == 0


def test_samuelson_slope_recovers_known_negative_slope():
    rng = np.random.default_rng(24)
    rows = []
    contract_id = 0
    for dte_mid in [15, 45, 75, 150, 300, 450, 900, 1800, 3000]:
        true_sigma = 0.30 * (dte_mid / 15.0) ** (-0.3)  # power-law decay
        daily_sigma = true_sigma / np.sqrt(252)
        for c in range(6):
            contract_id += 1
            r = rng.normal(0, daily_sigma, size=80)
            rows.append(
                pd.DataFrame(
                    {
                        "contract_id": contract_id,
                        "dte": dte_mid,
                        "r": r,
                    }
                )
            )
    df = pd.concat(rows, ignore_index=True)
    result = lib24.samuelson_slope(
        df,
        lib24.mad_vol,
        n_boot=50,
        bins=[0, 30, 60, 100, 200, 400, 700, 1500, 3500],
        min_obs=10,
    )
    assert result["slope"] < 0
    assert result["slope"] == pytest.approx(-0.3, abs=0.15)


def test_samuelson_slope_bootstrap_ci_brackets_true_slope():
    rng = np.random.default_rng(24)
    rows = []
    contract_id = 0
    for dte_mid in [15, 45, 75, 150, 300, 450, 900, 1800, 3000]:
        true_sigma = 0.30 * (dte_mid / 15.0) ** (-0.3)
        daily_sigma = true_sigma / np.sqrt(252)
        for c in range(10):
            contract_id += 1
            r = rng.normal(0, daily_sigma, size=100)
            rows.append(
                pd.DataFrame({"contract_id": contract_id, "dte": dte_mid, "r": r})
            )
    df = pd.concat(rows, ignore_index=True)
    result = lib24.samuelson_slope(
        df,
        lib24.mad_vol,
        n_boot=200,
        bins=[0, 30, 60, 100, 200, 400, 700, 1500, 3500],
        min_obs=10,
    )
    assert result["ci_lo"] <= -0.3 <= result["ci_hi"]


def test_rolling_maturity_vol_is_causal():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "contract_id": np.tile([1, 2], 100),
            "dte": np.tile([30, 400], 100),
            "r": rng.normal(0, 0.02, size=200),
        }
    )
    out1 = lib24.rolling_maturity_vol(df, window=20, buckets=[(0, 60), (300, 500)])
    # perturb a return far in the future and confirm past values are unchanged
    df2 = df.copy()
    future_mask = (df2["date"] == dates[90]) & (df2["dte"] == 30)
    df2.loc[future_mask, "r"] = 5.0
    out2 = lib24.rolling_maturity_vol(df2, window=20, buckets=[(0, 60), (300, 500)])
    past_col = "vol_0_60"
    early = out1[out1["date"] <= dates[50]][past_col].to_numpy()
    early2 = out2[out2["date"] <= dates[50]][past_col].to_numpy()
    np.testing.assert_allclose(early, early2, equal_nan=True)


def test_log_moneyness_zero_for_front_contract():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")] * 3,
            "dte": [10, 100, 400],
            "close": [50.0, 52.0, 55.0],
        }
    )
    out = lib24.log_moneyness(df)
    front_row = out.loc[out["dte"].idxmin()]
    assert front_row["log_moneyness"] == pytest.approx(0.0)


def test_vol_by_bucket_drops_buckets_under_min_obs():
    df = pd.DataFrame(
        {
            "contract_id": [1] * 5 + [2] * 60,
            "dte": [10] * 5 + [400] * 60,
            "r": np.concatenate(
                [
                    np.random.default_rng(1).normal(0, 0.02, 5),
                    np.random.default_rng(2).normal(0, 0.01, 60),
                ]
            ),
        }
    )
    out = lib24.vol_by_bucket(df, lib24.mad_vol, bins=[0, 30, 500], min_obs=50)
    assert len(out) == 1
    assert out.iloc[0]["dte_mid"] == pytest.approx((30 + 500) / 2)


def test_usable_returns_applies_volume_floor():
    df = pd.DataFrame(
        {
            "contract_id": [1, 1, 1],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "close": [100.0, 101.0, 102.0],
            "volume": [50, 500, 5000],
            "expiry": [pd.Timestamp("2020-06-01")] * 3,
        }
    )
    df = df.sort_values(["contract_id", "date"]).reset_index(drop=True)
    df["log_close"] = np.log(df["close"])
    df["r"] = df.groupby("contract_id")["log_close"].diff()
    df["gap"] = df.groupby("contract_id")["date"].diff().dt.days
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    out = lib24.usable_returns(df, max_gap=1, min_volume=500)
    assert len(out) == 1
    assert out.iloc[0]["volume"] == 5000


def test_curve_state_classifies_backwardation_and_contango():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")] * 2 + [pd.Timestamp("2020-01-02")] * 2,
            "dte": [10, 100, 10, 100],
            "close": [60.0, 55.0, 50.0, 55.0],  # day1: backwardation, day2: contango
        }
    )
    out = lib24.curve_state(df).set_index("date")
    assert out.loc[pd.Timestamp("2020-01-01"), "curve_state"] == "backwardation"
    assert out.loc[pd.Timestamp("2020-01-02"), "curve_state"] == "contango"
