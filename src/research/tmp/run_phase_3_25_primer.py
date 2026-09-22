"""Phase 3 -- primer on payoff geometry, real paths, and barrier dynamics.

Pure payoff geometry plus real paths, no new pricing theory. Computes:
1. Payoff-at-expiry grids for vanilla and barrier instruments
2. Real-path barrier illustrations from historical CL data (COVID crash window)
3. Realised knock-out frequency across barrier distances, tenors, and regimes
4. Premium vs strike, barrier, maturity, volatility
5. Greeks (delta, gamma) vs underlying for vanilla and knock-out puts

This is a teaching notebook: every premium computed here is a *model* price
from realised volatility, never a market quote.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_TMP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TMP_DIR))

import lib24
import lib25
import pricers_25_barrier

TMP = _TMP_DIR


def load_calm_date() -> pd.Timestamp:
    """Load the calm representative date for CL from phase_1_25_inputs.json."""
    with open(TMP / "phase_1_25_inputs.json") as f:
        data = json.load(f)
    date_str = data["products"]["CL"]["representative_dates"]["calm"]
    return pd.Timestamp(date_str)


def section_1_payoff_grids(product: str, calm_date: pd.Timestamp) -> dict[str, Any]:
    """Payoff-at-expiry grids for various instruments."""
    print("Computing payoff grids...", end=" ", flush=True)
    inputs = lib25.inputs_as_of(product, calm_date)
    F = float(np.atleast_1d(inputs["F"](1.0))[0])

    S = np.linspace(0.5 * F, 1.5 * F, 200)
    K = F
    B = 0.85 * F

    payoff_grids: dict[str, Any] = {}

    # Future
    payoff_grids["future"] = {"S": S.tolist(), "payoff": (S - F).tolist()}

    # Call
    payoff_grids["call"] = {"S": S.tolist(), "payoff": np.maximum(S - K, 0.0).tolist()}

    # Put
    payoff_grids["put"] = {"S": S.tolist(), "payoff": np.maximum(K - S, 0.0).tolist()}

    # Collar (long put at 0.95*F, short call at 1.05*F)
    K_put = 0.95 * F
    K_call = 1.05 * F
    collar_payoff = np.maximum(K_put - S, 0.0) - np.maximum(S - K_call, 0.0)
    payoff_grids["collar"] = {"S": S.tolist(), "payoff": collar_payoff.tolist()}

    # Down-and-out put at K=F with barrier B=0.85*F
    # Pure payoff geometry: if S >= B, payoff is max(K-S, 0); else 0
    ko_payoff = np.where(S >= B, np.maximum(K - S, 0.0), 0.0)
    payoff_grids["ko_put"] = {"S": S.tolist(), "payoff": ko_payoff.tolist()}

    print("done")
    return payoff_grids


def section_2_real_path_barrier() -> dict[str, Any]:
    """Real-path barrier from CL front-month during COVID crash window."""
    print("Computing real-path barrier...", end=" ", flush=True)

    panel = lib24.load_panel("CL")

    # Get front-month prices (minimum dte per date)
    front = panel.loc[panel.groupby("date")["dte"].idxmin()].copy()
    front = front.sort_values("date")

    # Restrict to 2019-06-01 through 2020-09-30
    start_date = pd.Timestamp("2019-06-01")
    end_date = pd.Timestamp("2020-09-30")
    front = front[(front["date"] >= start_date) & (front["date"] <= end_date)]

    # Pick barrier = 0.7 * front-month close on 2020-01-02 (or nearest trading day)
    ref_date = pd.Timestamp("2020-01-02")
    ref_rows = front[
        (front["date"] >= ref_date) & (front["date"] <= ref_date + pd.Timedelta(days=5))
    ]
    if len(ref_rows) > 0:
        ref_price = ref_rows.iloc[0]["close"]
    else:
        # Fallback: use first available date after 2020-01-02
        ref_price = (
            front[front["date"] >= ref_date].iloc[0]["close"]
            if len(front[front["date"] >= ref_date]) > 0
            else front.iloc[0]["close"]
        )

    barrier_level = 0.7 * ref_price

    # Check if/when front-month closes go at-or-below the barrier
    breached_rows = front[front["close"] <= barrier_level]
    if len(breached_rows) > 0:
        breach_date = breached_rows.iloc[0]["date"]
        breach_index = int(np.where(front["date"] == breach_date)[0][0])
    else:
        breach_date = None
        breach_index = None

    result: dict[str, Any] = {
        "dates": front["date"].dt.strftime("%Y-%m-%d").tolist(),
        "closes": front["close"].tolist(),
        "barrier_level": float(barrier_level),
        "breach_date": str(breach_date.date()) if breach_date is not None else None,
        "breach_index": int(breach_index) if breach_index is not None else None,
    }

    print("done")
    return result


def section_3_ko_frequency() -> list[dict[str, Any]]:
    """Realised knock-out frequency over historical CL data."""
    print(
        "Computing knock-out frequency (this may take a minute)...", end=" ", flush=True
    )

    panel = lib24.load_panel("CL")
    panel = panel.sort_values(["contract_id", "date"]).reset_index(drop=True)

    # Get crisis windows
    crisis_windows = lib24.CRISIS_WINDOWS
    crisis_dates: set[pd.Timestamp] = set()
    for start, end in crisis_windows.values():
        crisis_dates.update(pd.date_range(start, end, freq="D"))
    crisis_dates = {pd.Timestamp(d) for d in crisis_dates}

    barrier_distances = [0.05, 0.10, 0.15, 0.20, 0.30]
    tenors_days = [30, 90, 180, 365]

    results: list[dict[str, Any]] = []

    # Downsample to every 5th trading day for speed
    unique_dates = sorted(panel["date"].unique())
    sampled_dates = unique_dates[::5]

    for barrier_dist in barrier_distances:
        for tenor in tenors_days:
            freq_overall = 0.0
            n_overall = 0
            freq_crisis = 0.0
            n_crisis = 0
            freq_calm = 0.0
            n_calm = 0

            for t0_date in sampled_dates:
                t0_date = pd.Timestamp(t0_date)

                # Get front-month contract at t0
                contracts_at_t0 = panel[panel["date"] == t0_date]
                if len(contracts_at_t0) == 0:
                    continue

                front_at_t0 = contracts_at_t0.loc[contracts_at_t0["dte"].idxmin()]
                contract_id = front_at_t0["contract_id"]
                close_t0 = front_at_t0["close"]
                dte_t0 = front_at_t0["dte"]

                # Skip if contract doesn't have enough life left
                if dte_t0 < tenor:
                    continue

                # Look forward tenor_days within the same contract
                t1_date = t0_date + pd.Timedelta(days=tenor)
                contract_data = panel[panel["contract_id"] == contract_id]
                forward_data = contract_data[
                    (contract_data["date"] >= t0_date)
                    & (contract_data["date"] <= t1_date)
                ]

                if len(forward_data) == 0:
                    continue

                # Set barrier and check for knock-out
                barrier = close_t0 * (1 - barrier_dist)
                knockout_occurred = (forward_data["close"] <= barrier).any()

                # Increment counters
                is_crisis = t0_date in crisis_dates
                if is_crisis:
                    n_crisis += 1
                    if knockout_occurred:
                        freq_crisis += 1
                else:
                    n_calm += 1
                    if knockout_occurred:
                        freq_calm += 1

                n_overall += 1
                if knockout_occurred:
                    freq_overall += 1

            # Compute frequencies
            freq_overall_val = (
                freq_overall / n_overall if n_overall > 0 else float("nan")
            )
            freq_crisis_val = freq_crisis / n_crisis if n_crisis > 0 else float("nan")
            freq_calm_val = freq_calm / n_calm if n_calm > 0 else float("nan")

            results.append(
                {
                    "barrier_distance_pct": float(barrier_dist),
                    "tenor_days": int(tenor),
                    "freq_overall": float(freq_overall_val),
                    "n_overall": int(n_overall),
                    "freq_crisis": float(freq_crisis_val),
                    "n_crisis": int(n_crisis),
                    "freq_calm": float(freq_calm_val),
                    "n_calm": int(n_calm),
                }
            )

    print("done")
    return results


def section_4_premium_vs_parameters(
    product: str, calm_date: pd.Timestamp
) -> dict[str, Any]:
    """Premium vs strike, barrier, maturity, volatility."""
    print("Computing premiums vs parameters...", end=" ", flush=True)

    inputs = lib25.inputs_as_of(product, calm_date)
    F1y = float(np.atleast_1d(inputs["F"](1.0))[0])
    sigma1y = float(np.atleast_1d(inputs["sigma"](1.0))[0])
    df1y = float(np.atleast_1d(inputs["df"](1.0))[0])

    result: dict[str, Any] = {}

    # Premium vs strike
    K_values = np.linspace(0.7 * F1y, 1.3 * F1y, 25)
    put_prices = []
    ko_put_prices = []
    for K in K_values:
        put_price = lib25.black76(F1y, K, 1.0, sigma1y, df1y, "put")
        put_prices.append(put_price)
        ko_put_price = pricers_25_barrier.barrier_analytic(
            F1y, K, 0.85 * F1y, 1.0, sigma1y, df1y, "put", "do"
        )
        ko_put_prices.append(ko_put_price)

    result["premium_vs_strike"] = {
        "K": K_values.tolist(),
        "put": put_prices,
        "ko_put": ko_put_prices,
    }

    # Premium vs barrier
    B_values = np.linspace(0.6 * F1y, 0.95 * F1y, 20)
    ko_put_prices_b = []
    for B in B_values:
        ko_put_price = pricers_25_barrier.barrier_analytic(
            F1y, F1y, B, 1.0, sigma1y, df1y, "put", "do"
        )
        ko_put_prices_b.append(ko_put_price)

    vanilla_put_ref = lib25.black76(F1y, F1y, 1.0, sigma1y, df1y, "put")

    result["premium_vs_barrier"] = {
        "B": B_values.tolist(),
        "ko_put": ko_put_prices_b,
        "vanilla_put_reference": float(vanilla_put_ref),
    }

    # Premium vs maturity
    T_values = [0.083, 0.25, 0.5, 1.0, 2.0]
    put_prices_t = []
    for T in T_values:
        F_t = float(np.atleast_1d(inputs["F"](T))[0])
        sigma_t = float(np.atleast_1d(inputs["sigma"](T))[0])
        df_t = float(np.atleast_1d(inputs["df"](T))[0])
        K_t = F_t
        put_price = lib25.black76(F_t, K_t, T, sigma_t, df_t, "put")
        put_prices_t.append(put_price)

    result["premium_vs_maturity"] = {"T": T_values, "put": put_prices_t}

    # Premium vs volatility
    sigma_values = np.linspace(0.10, 0.60, 20)
    put_prices_sigma = []
    for sigma in sigma_values:
        put_price = lib25.black76(F1y, F1y, 1.0, sigma, df1y, "put")
        put_prices_sigma.append(put_price)

    result["premium_vs_vol"] = {"sigma": sigma_values.tolist(), "put": put_prices_sigma}

    print("done")
    return result


def section_5_greeks_vs_underlying(
    product: str, calm_date: pd.Timestamp
) -> dict[str, Any]:
    """Greeks (delta, gamma) vs underlying for vanilla and KO puts."""
    print("Computing greeks...", end=" ", flush=True)

    inputs = lib25.inputs_as_of(product, calm_date)
    F1y = float(np.atleast_1d(inputs["F"](1.0))[0])
    sigma1y = float(np.atleast_1d(inputs["sigma"](1.0))[0])
    df1y = float(np.atleast_1d(inputs["df"](1.0))[0])
    K = F1y
    T = 1.0
    B = 0.85 * F1y

    S_values = np.linspace(0.7 * F1y, 1.3 * F1y, 40)

    vanilla_deltas = []
    vanilla_gammas = []
    ko_put_delta_fds = []

    eps = 0.01 * F1y

    for S in S_values:
        # Vanilla put greeks
        greeks = lib25.black76_greeks(S, K, T, sigma1y, df1y, "put")
        vanilla_deltas.append(greeks["delta"])
        vanilla_gammas.append(greeks["gamma"])

        # KO put finite-difference delta
        ko_up = pricers_25_barrier.barrier_analytic(
            S + eps, K, B, T, sigma1y, df1y, "put", "do"
        )
        ko_down = pricers_25_barrier.barrier_analytic(
            S - eps, K, B, T, sigma1y, df1y, "put", "do"
        )
        ko_delta_fd = (ko_up - ko_down) / (2 * eps)
        ko_put_delta_fds.append(ko_delta_fd)

    result: dict[str, Any] = {
        "S": S_values.tolist(),
        "vanilla_delta": vanilla_deltas,
        "vanilla_gamma": vanilla_gammas,
        "ko_put_delta_fd": ko_put_delta_fds,
    }

    print("done")
    return result


def main() -> None:
    """Assemble all sections into a single JSON output."""
    calm_date = load_calm_date()
    product = "CL"

    output: dict[str, Any] = {}

    output["payoff_grids"] = section_1_payoff_grids(product, calm_date)
    output["real_path_barrier"] = section_2_real_path_barrier()
    output["ko_frequency"] = section_3_ko_frequency()
    output["premium_vs_strike"] = section_4_premium_vs_parameters(product, calm_date)[
        "premium_vs_strike"
    ]

    premiums = section_4_premium_vs_parameters(product, calm_date)
    output["premium_vs_strike"] = premiums["premium_vs_strike"]
    output["premium_vs_barrier"] = premiums["premium_vs_barrier"]
    output["premium_vs_maturity"] = premiums["premium_vs_maturity"]
    output["premium_vs_vol"] = premiums["premium_vs_vol"]

    output["greeks"] = section_5_greeks_vs_underlying(product, calm_date)

    # Write JSON
    output_path = TMP / "phase_3_25_primer.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Verify it's valid JSON and report
    with open(output_path) as f:
        verified = json.load(f)

    top_keys = sorted(verified.keys())
    print(f"\nPhase 3 complete: {output_path}")
    print(f"Top-level keys: {top_keys}")


if __name__ == "__main__":
    main()
