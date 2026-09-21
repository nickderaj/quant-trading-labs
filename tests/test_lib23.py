import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "research" / "tmp")
)
import lib23


def _synthetic_day(tau, S, L, beta, nu):
    lnF = lib23.model26_lnF(S, L, tau, beta, nu)
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2020-01-01"),
            "ticker": [f"C{i}" for i in range(len(tau))],
            "tau": tau,
            "close": np.exp(lnF),
            "volume": 1000,
            "log_close": lnF,
        }
    )


def test_spot_proxy_recovers_flat_curve():
    day = pd.DataFrame({"tau": [0.1, 0.2], "close": [60.0, 60.0]})
    assert lib23.spot_proxy_for_day(day) == pytest.approx(60.0)


def test_spot_proxy_extrapolates_linear_slope():
    # F1 at tau1, F2 at tau2, linear in tau -> extrapolate to tau=0
    day = pd.DataFrame({"tau": [0.1, 0.2], "close": [59.0, 58.0]})
    s = lib23.spot_proxy_for_day(day)
    assert s == pytest.approx(60.0)


def test_spot_proxy_none_on_single_row():
    day = pd.DataFrame({"tau": [0.1], "close": [60.0]})
    assert lib23.spot_proxy_for_day(day) is None


def test_model26_reduces_to_S_at_tau_zero():
    tau = np.array([0.0])
    lnF = lib23.model26_lnF(S=60.0, L=55.0, tau=tau, beta=0.5, nu=0.04)
    assert np.exp(lnF[0]) == pytest.approx(60.0)


def test_model26_reduces_to_L_at_long_tau():
    tau = np.array([100.0])
    lnF = lib23.model26_lnF(S=60.0, L=55.0, tau=tau, beta=0.5, nu=0.04)
    assert np.exp(lnF[0]) == pytest.approx(55.0, rel=1e-3)


def test_model26_B_bounds():
    tau = np.array([0.0, 1.0, 100.0])
    B = lib23.model26_B(tau, beta=0.3)
    assert B[0] == pytest.approx(1.0)
    assert 0 < B[1] < 1
    assert B[2] == pytest.approx(0.0, abs=1e-10)


def test_fit_L_joint_ls_recovers_known_params():
    # fit_L_joint_ls fits the (S,L)-blend weight only and ignores the model-26
    # convexity term A(tau) (nu is unknown at this stage -- it comes from rolling
    # realised vol downstream). With nu=0 the blend fit is exact; a nonzero nu
    # introduces a documented systematic bias in the recovered beta, so this test
    # uses nu=0 to isolate blend-weight recovery from that known limitation.
    tau = np.linspace(0.05, 6.0, 25)
    S, ell, beta, nu = 60.0, 50.0, 0.4, 0.0
    day = _synthetic_day(tau, S, ell, beta, nu)
    fit = lib23.fit_L_joint_ls(day)
    assert fit is not None
    beta_hat, L_hat, sse = fit
    assert beta_hat == pytest.approx(beta, abs=0.02)
    assert L_hat == pytest.approx(ell, rel=0.01)
    assert sse < 1e-6


def test_fit_beta_given_SL_recovers_known_beta():
    tau = np.linspace(0.05, 6.0, 25)
    S, ell, beta, nu = 60.0, 50.0, 0.7, 0.03
    day = _synthetic_day(tau, S, ell, beta, nu)
    b_hat, _sse = lib23.fit_beta_given_SL(day, S, ell)
    assert b_hat == pytest.approx(beta, abs=0.05)


def test_model28_reduces_to_model26_when_theta_zero():
    tau = np.linspace(0.1, 5.0, 10)
    a = lib23.model26_lnF(60.0, 55.0, tau, 0.4, 0.03)
    b = lib23.model28_lnF(60.0, 55.0, tau, 0.4, 0.03, theta=0.0, eta=0.6)
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_model28_handles_beta_eq_eta_without_nan():
    tau = np.linspace(0.1, 5.0, 10)
    out = lib23.model28_lnF(60.0, 55.0, tau, 0.4, 0.03, theta=0.1, eta=0.4)
    assert np.all(np.isfinite(out))


def test_vol_identity_matches_endpoints():
    tau = np.array([0.0, 1000.0])
    sig = lib23.vol_identity_sigma_F(tau, beta=0.5, sigma_S=0.4, sigma_L=0.15, rho=0.2)
    assert sig[0] == pytest.approx(0.4, abs=1e-6)
    assert sig[1] == pytest.approx(0.15, abs=1e-6)


def test_apply_hygiene_drops_negative_price_event_and_junk():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-04-20", "2020-04-20", "2021-01-01", "2021-01-01"]
            ),
            "ticker": ["CL202005", "CL202006", "CL203212", "CL202101"],
            "expiry": pd.to_datetime(
                ["2020-05-19", "2020-06-19", "2032-11-19", "2021-01-19"]
            ),
            "close": [-2.67, 20.0, 0.0, 45.0],
            "volume": [102083, 500, 5, 500],
        }
    )
    df["tau"] = (df["expiry"] - df["date"]).dt.days / 365.25
    clean, report = lib23.apply_hygiene(df, "all")
    assert report["dropped"]["negative_price_event_real"] == 1
    assert report["dropped"]["nonpositive_junk"] == 1
    assert (clean["ticker"] == "CL202005").sum() == 0
    assert (clean["ticker"] == "CL203212").sum() == 0


def test_apply_hygiene_min_contracts_per_day():
    dates = pd.to_datetime(["2021-01-01"] * 3)
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C"],
            "expiry": pd.to_datetime(["2021-02-01", "2021-03-01", "2021-04-01"]),
            "close": [10.0, 11.0, 12.0],
            "volume": [100, 100, 100],
        }
    )
    df["tau"] = (df["expiry"] - df["date"]).dt.days / 365.25
    clean, report = lib23.apply_hygiene(df, "all")
    assert len(clean) == 0
    assert report["skipped_days_thin_curve"] == 1


def test_liquidity_variant_filters_volume():
    dates = pd.to_datetime(["2021-01-01"] * 10)
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": [f"T{i}" for i in range(10)],
            "expiry": pd.date_range("2021-02-01", periods=10, freq="30D"),
            "close": np.linspace(10, 20, 10),
            "volume": [0, 50, 150, 200, 5, 300, 0, 400, 120, 60],
        }
    )
    df["tau"] = (df["expiry"] - df["date"]).dt.days / 365.25
    clean_all, _ = lib23.apply_hygiene(df.copy(), "all")
    clean_g0, _ = lib23.apply_hygiene(df.copy(), "vol_gt_0")
    clean_g100, _ = lib23.apply_hygiene(df.copy(), "vol_gt_100")
    assert len(clean_all) == 10
    assert len(clean_g0) == 8
    # vol_gt_100 leaves only 5 rows, below MIN_CONTRACTS_PER_DAY (8), so the
    # thin-curve rule drops the whole day -- exercises rule interaction, not a bug.
    assert len(clean_g100) == 0


def test_bench_flat_forward():
    day = pd.DataFrame({"tau": [0.1], "close": [60.0], "log_close": [np.log(60.0)]})
    out = lib23.bench_flat_forward(day, 60.0, np.array([1.0, 2.0]))
    np.testing.assert_allclose(out, [60.0, 60.0])


def test_bench_cubic_spline_interpolates_exactly_at_nodes():
    tau = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    close = np.array([60.0, 59.0, 58.5, 58.0, 57.5])
    day = pd.DataFrame({"tau": tau, "close": close, "log_close": np.log(close)})
    out = lib23.bench_cubic_spline(day, tau)
    np.testing.assert_allclose(out, close, rtol=1e-6)


def test_bonferroni_alpha():
    assert lib23.bonferroni_alpha(10, 0.05) == pytest.approx(0.005)
