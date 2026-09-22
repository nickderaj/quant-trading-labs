"""
Case studies E, F, G and variance-swap aside for notebook 025.

E: Producer and negative prices (April 2020)
F: European buyer (FX quanto/composite)
G: Structured note (autocallable reverse convertible)
Variance-swap aside: Fair strike and forecast quality
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sstats

_TMP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TMP_DIR.parents[2]
for _p in (str(_TMP_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lib24
import lib25
import pricers_25_exotic

DAYS_PER_YEAR = 365.25


def case_e_negative_prices() -> dict[str, Any]:
    """Case E: Producer and negative prices (April 2020).

    Price a $10 put on CL202005 as of 2020-04-06 using three models,
    then run Kupiec POF test on rolling 1% VaR forecasts.
    """
    valuation_date = pd.Timestamp("2020-04-06")

    # Load panel and get forward price and expiry for CL202005 on valuation date
    panel = lib24.load_panel("CL")
    row = panel[
        (panel["ticker"] == "CL202005") & (panel["date"] == valuation_date)
    ].iloc[0]
    F = float(row["close"])
    expiry = pd.Timestamp(row["expiry"])
    T = (expiry - valuation_date).days / DAYS_PER_YEAR

    # Get inputs as of that date for vol estimate
    inputs = lib25.inputs_as_of("CL", valuation_date)
    sigma = float(np.atleast_1d(inputs["sigma"](T))[0])  # lognormal vol
    df = float(np.atleast_1d(inputs["df"](T))[0])

    K = 10.0
    actual_settlement = -2.67

    # Price the put under three models
    black76_price = lib25.black76(F, K, T, sigma, df, "put")

    # Bachelier: convert lognormal vol to normal vol
    sigma_n = sigma * F
    bach_price = lib25.bachelier(F, K, T, sigma_n, df, "put")

    # Displaced Black with shift=15.0
    shift = 15.0
    disp_price = lib25.displaced_black(F, K, T, sigma, df, "put", shift)

    # Log-densities at actual settlement -2.67
    # Black-76: lognormal density has zero support at F_T <= 0
    black76_logpdf = float("-inf")

    # Bachelier: normal density
    # F_T ~ Normal(F, (sigma_n*sqrt(T))^2)
    scale = sigma_n * np.sqrt(T)
    bach_logpdf = sstats.norm.logpdf(actual_settlement, loc=F, scale=scale)

    # Displaced Black: lognormal in (F+shift), but we need logpdf of original variable
    # If F_T+shift ~ Lognormal, then F_T ~ Shifted Lognormal
    # logpdf(x) = logpdf_lognorm(x+shift, ...)
    # For the shifted variable: ln(F_T + shift) ~ Normal(ln(F+shift) - 0.5*sigma^2*T, sigma^2*T)
    Fs = F + shift
    x_shifted = actual_settlement + shift  # -2.67 + 15 = 12.33 > 0
    if x_shifted > 0:
        # Lognormal PDF of shifted variable
        disp_logpdf = sstats.lognorm.logpdf(
            x_shifted, s=sigma * np.sqrt(T), scale=Fs * np.exp(-0.5 * sigma**2 * T)
        )
    else:
        disp_logpdf = float("-inf")

    # Displacement needed for -2.67 to be in support: shift > 2.67
    min_shift_for_support = 2.67 + 0.1

    # $0-strike floor (put at K=0)
    # Black-76: put at K=0 is worthless since F_T > 0 always (lognormal)
    black76_floor = 0.0
    # Bachelier: put at K=0 is worth E[max(0, 0-F_T)] = E[max(0, -F_T)]
    bach_floor = lib25.bachelier(F, 0.0, T, sigma_n, df, "put")

    # Now: rolling 1% VaR test over full CL history
    panel_full = lib24.load_panel("CL")
    usable = lib24.usable_returns(panel_full)
    front = usable.loc[usable.groupby("date")["dte"].idxmin()].sort_values("date")

    # Build rolling 1% quantile forecasts (60-day window, lognormal returns)
    # Three models:
    # (1) Lognormal: rolling vol on log-returns, 1% quantile = mean - 2.326*vol
    # (2) Bachelier (level changes): rolling vol on level changes, 1% quantile = mean - 2.326*vol
    # (3) Displaced (we'll use level-change model same as Bachelier for simplicity)

    returns = np.asarray(front["r"].values, dtype=float)
    closes = np.asarray(front["close"].values, dtype=float)
    level_changes = np.diff(closes)

    window = 60
    alpha = 0.01
    z_quantile = sstats.norm.ppf(alpha)  # ~-2.326 for 1%

    # Model 1: Lognormal returns
    lognorm_forecasts = np.full(len(returns), np.nan)
    lognorm_realised = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        vol_rolling = lib24.mad_vol(returns[max(0, i - window) : i])
        # Next day's 1% quantile of log-return
        lognorm_forecasts[i] = z_quantile * vol_rolling
        # Realised next day's log-return
        lognorm_realised[i] = returns[i]

    # Model 2: Bachelier level changes
    # Level change is dF = F_t - F_{t-1}, so closing price differences
    bach_forecasts = np.full(len(level_changes) + 1, np.nan)
    bach_realised = np.full(len(level_changes) + 1, np.nan)
    for i in range(window, len(level_changes)):
        vol_level = np.std(level_changes[max(0, i - window) : i], ddof=1)
        # Next day's 1% quantile of level change
        bach_forecasts[i + 1] = z_quantile * vol_level
        # Realised next day's level change
        bach_realised[i + 1] = level_changes[i]

    # Build exceedance indicators
    lognorm_exceed = lognorm_realised < lognorm_forecasts
    valid_log = ~np.isnan(lognorm_exceed)
    lognorm_exceed = lognorm_exceed[valid_log]

    bach_exceed = bach_realised < bach_forecasts
    valid_bach = ~np.isnan(bach_exceed)
    bach_exceed = bach_exceed[valid_bach]

    # Model 3: same as Bachelier (displaced level changes)
    displaced_exceed = bach_exceed.copy()

    # Kupiec POF tests
    kupiec_lognorm = lib25.kupiec_pof(lognorm_exceed, alpha)
    kupiec_bach = lib25.kupiec_pof(bach_exceed, alpha)
    kupiec_displaced = lib25.kupiec_pof(displaced_exceed, alpha)

    return {
        "valuation_date": str(valuation_date.date()),
        "ticker": "CL202005",
        "expiry": str(expiry.date()),
        "F": F,
        "T": T,
        "sigma_lognormal": sigma,
        "vol_model": "rolling 756-day realised MAD vol to date",
        "strike": K,
        "models": {
            "black76": {
                "put_price": black76_price,
                "log_density_at_negative_2_67": black76_logpdf,
                "note": "Lognormal has zero support below 0; logpdf is -Infinity",
            },
            "bachelier": {
                "sigma_normal": sigma_n,
                "put_price": bach_price,
                "log_density_at_negative_2_67": float(bach_logpdf),
                "note": "Normal admits negative values; finite logpdf at -2.67",
            },
            "displaced_black": {
                "shift": shift,
                "put_price": disp_price,
                "log_density_at_negative_2_67": float(disp_logpdf),
                "note": f"-2.67+shift={x_shifted:.2f}>0, so finite logpdf exists",
            },
        },
        "displacement_analysis": {
            "actual_settlement": actual_settlement,
            "min_shift_for_support": min_shift_for_support,
            "note": f"Shift of {shift} comfortably admits -2.67; min would be ~{min_shift_for_support}",
        },
        "zero_strike_floor": {
            "black76_price": black76_floor,
            "bachelier_price": bach_floor,
            "note": "Black-76 assigns zero prob below 0, so K=0 put is ~0; Bachelier is positive",
        },
        "rolling_var_kupiec_pof": {
            "window_days": window,
            "alpha": alpha,
            "lognormal_returns": kupiec_lognorm,
            "bachelier_level_changes": kupiec_bach,
            "displaced_level_changes": kupiec_displaced,
        },
    }


def case_f_european_buyer() -> dict[str, Any]:
    """Case F: European buyer (FX quanto/composite).

    Long CL exposure, reports in EUR. Valuation on calm date.
    """
    # Load FX and correlation data from preregistration
    with open(_TMP_DIR / "phase_1_25_inputs.json") as f:
        phase1 = json.load(f)

    fx_stats = phase1["fx"]
    valuation_date = pd.Timestamp(
        phase1["products"]["CL"]["representative_dates"]["calm"]
    )

    # Get inputs
    inputs = lib25.inputs_as_of("CL", valuation_date)
    F = float(np.atleast_1d(inputs["F"](1.0))[0])
    sigma = float(np.atleast_1d(inputs["sigma"](1.0))[0])
    df = float(np.atleast_1d(inputs["df"](1.0))[0])

    T = 1.0
    K = F  # at-the-money call

    sigma_fx = fx_stats["annualised_mad_vol"]
    rho_mean = fx_stats["rolling_corr_with_CL"]["mean"]
    rho_std = fx_stats["rolling_corr_with_CL"]["std"]

    # Structures
    # (i) Unhedged FX: no premium, just describe exposure
    unhedged = {
        "structure": "unhedged FX",
        "premium": 0.0,
        "note": "Exposure = CL * EUR/USD (both random), not priced",
    }

    # (ii) FX forward: locks EUR/USD rate, no option
    fx_forward = {
        "structure": "FX forward",
        "premium": 0.0,
        "note": "Locks EUR/USD rate, converts the FX spot exposure to forward at zero cost",
    }

    # (iii) Quanto call (fixed FX)
    quanto_price = pricers_25_exotic.quanto_black76(
        F, K, T, sigma, sigma_fx, rho_mean, df, "call"
    )
    # Unadjusted commodity-only price
    vanilla_call = lib25.black76(F, K, T, sigma, df, "call")
    quanto_adjustment = quanto_price - vanilla_call

    # (iv) Composite call (floating FX)
    composite_price = pricers_25_exotic.composite_black76(
        F, K, T, sigma, sigma_fx, rho_mean, df, "call"
    )

    # Price sensitivity to rho variation
    rho_lo = np.clip(rho_mean - rho_std, -0.999, 0.999)
    rho_hi = np.clip(rho_mean + rho_std, -0.999, 0.999)

    quanto_price_lo = pricers_25_exotic.quanto_black76(
        F, K, T, sigma, sigma_fx, rho_lo, df, "call"
    )
    quanto_price_hi = pricers_25_exotic.quanto_black76(
        F, K, T, sigma, sigma_fx, rho_hi, df, "call"
    )

    quanto_range = {
        "rho_lo": rho_lo,
        "rho_mean": rho_mean,
        "rho_hi": rho_hi,
        "price_at_rho_lo": quanto_price_lo,
        "price_at_rho_mean": quanto_price,
        "price_at_rho_hi": quanto_price_hi,
        "price_range": abs(quanto_price_hi - quanto_price_lo),
    }

    # Compare adjustment magnitude to price range
    adjustment_magnitude = abs(quanto_adjustment)
    range_magnitude = quanto_range["price_range"]

    note = (
        f"Rho's std ({rho_std:.3f}) creates price range ${range_magnitude:.3f}, "
        f"which is {'LARGER' if range_magnitude > adjustment_magnitude else 'smaller'} "
        f"than adjustment magnitude ${adjustment_magnitude:.3f}"
    )

    return {
        "valuation_date": str(valuation_date.date()),
        "T": T,
        "F": F,
        "sigma": sigma,
        "df": df,
        "K": K,
        "sigma_fx": sigma_fx,
        "fx_rolling_corr_with_CL": {
            "mean": rho_mean,
            "std": rho_std,
            "window": fx_stats["rolling_corr_with_CL"]["window"],
        },
        "structures": {
            "unhedged_fx": unhedged,
            "fx_forward": fx_forward,
            "quanto_call": {
                "price": quanto_price,
                "unadjusted_vanilla_call": vanilla_call,
                "quanto_adjustment": quanto_adjustment,
            },
            "composite_call": {"price": composite_price},
        },
        "rho_sensitivity_quanto": quanto_range,
        "teaching_point": note,
    }


def case_g_structured_note() -> dict[str, Any]:
    """Case G: Structured note (autocallable reverse convertible on CL).

    One-year note with quarterly observations, 100k path simulation.
    """
    with open(_TMP_DIR / "phase_1_25_inputs.json") as f:
        phase1 = json.load(f)
    valuation_date = pd.Timestamp(
        phase1["products"]["CL"]["representative_dates"]["calm"]
    )

    inputs = lib25.inputs_as_of("CL", valuation_date)
    F = float(np.atleast_1d(inputs["F"](1.0))[0])
    sigma = float(np.atleast_1d(inputs["sigma"](1.0))[0])
    df = float(np.atleast_1d(inputs["df"](1.0))[0])
    r = float(np.atleast_1d(inputs["r"](1.0))[0])

    T = 1.0
    n_steps = 252
    n_paths = 100_000

    # Simulate paths
    print("  Simulating 100k GBM paths...")
    paths = lib25.simulate_gbm(F, sigma, T, n_steps, n_paths, seed=25)
    paths_norm = paths / F  # Normalize to start at 1.0

    # Quarterly observations: indices 63, 126, 189, 252
    obs_idx = np.array([63, 126, 189, 252])

    # Price the autocallable
    coupon = 0.08
    autocall_level = 1.0
    barrier = 0.70

    result = pricers_25_exotic.autocallable(
        paths_norm, coupon, autocall_level, barrier, obs_idx, df
    )

    # Decompose into legs
    note_face_value = 1.0
    note_fair_value = result["price"]
    implied_issuer_margin = note_face_value - note_fair_value

    # Historical resampling
    print("  Resampling historical paths...")
    panel = lib24.load_panel("CL")
    front = panel.loc[panel.groupby("date")["dte"].idxmin()].sort_values("date")

    # Downsample start dates every 20 trading days
    start_idx = np.arange(0, len(front) - n_steps, 20)

    historical_payoffs = []
    historical_autocalls = []

    for start in start_idx:
        # Extract n_steps+1 trading days from that start
        window = front.iloc[start : start + n_steps + 1]
        if len(window) < n_steps + 1:
            continue

        # Get levels and normalize
        levels = np.asarray(window["close"].values, dtype=float)
        level_0 = levels[0]
        levels_norm = levels / level_0

        # Resample onto fixed 253-point grid (0 to n_steps)
        # Create index for original window
        original_idx = np.arange(len(levels))
        resampled_idx = np.linspace(0, len(levels) - 1, n_steps + 1)
        levels_resampled = np.interp(resampled_idx, original_idx, levels_norm)

        # Run autocallable on this single path
        path_single = levels_resampled.reshape(1, -1)
        result_hist = pricers_25_exotic.autocallable(
            path_single, coupon, autocall_level, barrier, obs_idx, df
        )

        payoff = result_hist["price"]
        did_autocall = (
            result_hist["autocall_frequency_by_obs"].sum() > 0
        )  # Any obs triggered it

        historical_payoffs.append(payoff)
        historical_autocalls.append(1.0 if did_autocall else 0.0)

    historical_payoffs = np.array(historical_payoffs, dtype=float)  # type: ignore
    historical_autocalls = np.array(historical_autocalls, dtype=float)  # type: ignore

    # Statistics
    n_historical = len(historical_payoffs)
    realized_autocall_freq = (
        float(historical_autocalls.mean())  # type: ignore
        if n_historical > 0
        else 0.0
    )

    return {
        "valuation_date": str(valuation_date.date()),
        "T": T,
        "F": F,
        "sigma": sigma,
        "df": df,
        "r": r,
        "simulation": {
            "n_paths": n_paths,
            "n_steps": n_steps,
            "seed": 25,
            "antithetic": True,
        },
        "structure": {
            "coupon": coupon,
            "autocall_level": autocall_level,
            "barrier": barrier,
            "obs_indices": obs_idx.tolist(),
            "note": "Quarterly observations; early redemption if level >= 1.0",
        },
        "model_pricing": {
            "note_fair_value": float(note_fair_value),
            "note_face_value": note_face_value,
            "implied_issuer_margin": float(implied_issuer_margin),
            "price_se": float(result["se"]),
            "autocall_frequency_by_obs": result["autocall_frequency_by_obs"].tolist(),
            "never_autocalled_frequency": float(result["never_autocalled_frequency"]),
        },
        "historical_resampling": {
            "n_windows": n_historical,
            "downsample_every_n_days": 20,
            "payoff_distribution": {
                "n": n_historical,
                "mean": float(np.mean(historical_payoffs)),
                "std": float(np.std(historical_payoffs, ddof=1)),
                "min": float(np.min(historical_payoffs)),
                "p25": float(np.quantile(historical_payoffs, 0.25)),
                "median": float(np.median(historical_payoffs)),
                "p75": float(np.quantile(historical_payoffs, 0.75)),
                "max": float(np.max(historical_payoffs)),
            },
            "realized_autocall_frequency": realized_autocall_freq,
            "model_expected_autocall_frequency": float(
                result["autocall_frequency_by_obs"][:-1].sum() / len(obs_idx)
            ),
            "note": "Model uses risk-neutral GBM; historical has actual drift. Autocall rates compared for sanity check.",
        },
    }


def variance_swap_aside() -> dict[str, Any]:
    """Variance-swap aside: fair strike and forecast quality."""
    with open(_TMP_DIR / "phase_1_25_inputs.json") as f:
        phase1 = json.load(f)
    valuation_date = pd.Timestamp(
        phase1["products"]["CL"]["representative_dates"]["calm"]
    )

    inputs = lib25.inputs_as_of("CL", valuation_date)
    sigma_fn = inputs["sigma"]

    # Fair variance-swap strike
    print("  Computing variance-swap strike...")
    varswap_result = pricers_25_exotic.varswap_strike(sigma_fn, T=1.0, n_strikes=50)

    # Forecast quality via QLIKE
    panel = lib24.load_panel("CL")
    front = panel.loc[panel.groupby("date")["dte"].idxmin()].sort_values("date")

    # Rolling 21-day realised variance
    returns = np.asarray(front["r"].values, dtype=float)
    window = 21

    realised_var_list: list[float] = []
    dates_var: list[int] = []

    for i in range(window, len(returns)):
        ret_window = returns[max(0, i - window) : i]
        rv = float(np.sum(ret_window**2) * DAYS_PER_YEAR / window)
        realised_var_list.append(rv)
        dates_var.append(i)

    realised_var = np.array(realised_var_list, dtype=float)

    # Constant varswap strike forecast (broadcast)
    varswap_forecast = np.full_like(realised_var, varswap_result["strike_variance"])

    # Naive baseline: yesterday's realised variance as today's forecast
    naive_baseline = np.roll(realised_var, 1)
    naive_baseline[0] = realised_var[0]  # Fill first with same-day

    # QLIKE scores (scored against realised_var)
    # We need next-period realised, so shift realised_var forward
    next_realised = np.roll(realised_var, -1)
    next_realised[-1] = realised_var[-1]  # Fill last

    qlike_varswap = lib25.qlike(varswap_forecast, next_realised)
    qlike_naive = lib25.qlike(naive_baseline, next_realised)

    # Intraday vs close-to-close variance ratio
    print("  Computing intraday vs close-to-close variance ratio...")
    try:
        intraday_df = pd.read_parquet(
            f"{_REPO_ROOT / 'src/research/data/market/databento/intraday/CL.parquet'}"
        )

        # Filter to Jan-Jul 2026
        intraday_df["date"] = pd.to_datetime(intraday_df["timestamp"]).dt.date
        intraday_df = intraday_df[
            (intraday_df["date"] >= pd.Timestamp("2026-01-01").date())
            & (intraday_df["date"] <= pd.Timestamp("2026-07-31").date())
        ]

        # Compute 1-min returns and daily realized variance from intraday
        intraday_df["log_close"] = np.log(intraday_df["close"])
        intraday_df["r_1min"] = intraday_df.groupby("date")["log_close"].diff()

        intraday_var_by_day = intraday_df.groupby("date")["r_1min"].apply(
            lambda x: np.sum(x**2) * DAYS_PER_YEAR
        )

        # Compare with daily close-to-close on same dates
        close_close_df = front[
            (front["date"].dt.date >= pd.Timestamp("2026-01-01").date())
            & (front["date"].dt.date <= pd.Timestamp("2026-07-31").date())
        ].copy()
        close_close_df["date_only"] = close_close_df["date"].dt.date

        close_close_var_by_day = close_close_df.groupby("date_only")["r"].apply(
            lambda x: (x**2).sum() * DAYS_PER_YEAR
        )

        # Align and compute ratio
        common_dates_set = set(intraday_var_by_day.index) & set(
            close_close_var_by_day.index
        )
        if len(common_dates_set) > 0:
            common_dates_list = sorted(common_dates_set)
            intraday_subset = np.asarray(
                intraday_var_by_day.loc[common_dates_list].values, dtype=float
            )
            closeclose_subset = np.asarray(
                close_close_var_by_day.loc[common_dates_list].values, dtype=float
            )

            # Avoid division by zero
            valid = closeclose_subset > 0
            if valid.sum() > 0:
                ratio = float(
                    np.mean(intraday_subset[valid] / closeclose_subset[valid])
                )
            else:
                ratio = float("nan")

            intraday_ratio = {
                "n_common_dates": len(common_dates_list),
                "intraday_realized_var_mean": float(np.nanmean(intraday_subset)),
                "closeclose_realized_var_mean": float(np.nanmean(closeclose_subset)),
                "ratio_intraday_over_closeclose": ratio,
                "note": "Jan-Jul 2026 only; intraday captures moves daily close misses",
            }
        else:
            intraday_ratio = {
                "n_common_dates": 0,
                "note": "No common dates between intraday and close-to-close data",
            }
    except (FileNotFoundError, KeyError, ValueError) as e:
        intraday_ratio = {
            "error": str(e),
            "note": "Could not compute intraday ratio; file may not exist or data format issue",
        }

    return {
        "replication_note": "The replication leg is untestable here -- replicating a variance swap needs a strip of options, and there are none in this repo.",
        "valuation_date": str(valuation_date.date()),
        "varswap_strike": varswap_result,
        "forecast_quality_qlike": {
            "varswap_strike_forecast": qlike_varswap,
            "naive_trailing_baseline": qlike_naive,
            "note": "Lower QLIKE is better; varswap is a constant forecast, naive is yesterday's realized",
        },
        "intraday_vs_closeclose": intraday_ratio,
    }


def main() -> None:
    """Run all case studies and write output."""
    output = {}

    print("Case E: Negative prices...")
    output["E_negative_prices"] = case_e_negative_prices()

    print("Case F: European buyer...")
    output["F_european_buyer"] = case_f_european_buyer()

    print("Case G: Structured note...")
    output["G_structured_note"] = case_g_structured_note()

    print("Variance-swap aside...")
    output["variance_swap_aside"] = variance_swap_aside()

    # Write output
    output_path = _TMP_DIR / "phase_6_25_cases_efg.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, allow_nan=True)

    print(f"\nWrote {output_path}")
    print(f"Top-level keys: {list(output.keys())}")
    print(f"File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
