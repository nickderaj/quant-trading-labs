#!/usr/bin/env python3
"""
Phase 5: Four case studies for bespoke commodity derivatives (notebook 025).

Structures:
  A. Refiner: 3:2:1 crack spread (CL/HO/RB)
  B. Farmer: ZW wheat with floor (THE BEST ONE)
  C. Airline: HO monthly average diesel buyer
  D. Gas utility: NG winter strip with seasonality

Every premium is a MODEL price from realised vol, never a market quote.
Realised payoffs are historical arithmetic payoffs, not backtests.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_TMP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TMP_DIR.parents[2]
for _p in (str(_TMP_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lib24
import lib25
import pricers_25_asian
import pricers_25_barrier
import pricers_25_exotic


def subsample_values(arr: np.ndarray, max_samples: int = 50) -> list:
    """Subsample array to at most max_samples for JSON display."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= max_samples:
        return arr.tolist()
    idx = np.arange(len(arr))[:: max(1, len(arr) // max_samples)]
    return arr[idx].tolist()


def summarise_payoffs(payoffs: Sequence | np.ndarray) -> dict:
    """Summary stats for realised payoff distribution."""
    payoffs = np.asarray(payoffs, dtype=float)
    payoffs = payoffs[np.isfinite(payoffs)]
    if len(payoffs) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "values": [],
        }
    return {
        "n": len(payoffs),
        "mean": float(np.mean(payoffs)),
        "std": float(np.std(payoffs, ddof=1) if len(payoffs) > 1 else 0),
        "p10": float(np.percentile(payoffs, 10)),
        "p50": float(np.percentile(payoffs, 50)),
        "p90": float(np.percentile(payoffs, 90)),
        "values": subsample_values(payoffs),
    }


# ========================================================================== #
# Case A: Refiner (3:2:1 Crack Spread)
# ========================================================================== #


def case_a_refiner() -> dict:
    """3:2:1 crack spread: long crude, short products."""
    print("Case A: Refiner (3:2:1 crack spread)...", end=" ", flush=True)

    # Find nearest date with CL data to 2017-06-04
    target_date = pd.Timestamp("2017-06-04")
    panel_probe = lib24.load_panel("CL")
    closest_date = panel_probe.loc[
        (panel_probe["date"] - target_date).abs().idxmin(), "date"
    ]
    valuation_date = pd.Timestamp(str(closest_date))
    T = 0.5

    # Get market inputs
    inputs_cl = lib25.inputs_as_of("CL", valuation_date)
    inputs_ho = lib25.inputs_as_of("HO", valuation_date)
    inputs_rb = lib25.inputs_as_of("RB", valuation_date)

    F_cl = float(np.asarray(inputs_cl["F"](T)).flat[0])
    F_ho = float(np.asarray(inputs_ho["F"](T)).flat[0])
    F_rb = float(np.asarray(inputs_rb["F"](T)).flat[0])

    sigma_cl = float(np.asarray(inputs_cl["sigma"](T)).item())
    sigma_ho = float(np.asarray(inputs_ho["sigma"](T)).item())
    sigma_rb = float(np.asarray(inputs_rb["sigma"](T)).item())

    df = float(np.asarray(inputs_cl["df"](T)).item())

    # 3:2:1 crack spread: per barrel of crude
    # 1 bbl crude -> 0.667 bbl gasoline + 0.333 bbl heating oil
    # crack = (2*F_rb*42 + 1*F_ho*42)/3 - F_cl  [per bbl crude, $/bbl]
    F_crack_leg = (2 * F_rb * 42 + 1 * F_ho * 42) / 3
    crack_fwd = F_crack_leg - F_cl

    # Correlation for spread pricer
    rho_crhb = lib25.corr_rolling("CL", "HO", window=126).iloc[-1]
    if not np.isfinite(rho_crhb):
        rho_crhb = 0.85
    rho_crrb = lib25.corr_rolling("CL", "RB", window=126).iloc[-1]
    if not np.isfinite(rho_crrb):
        rho_crrb = 0.85

    # Weighted correlation for the blended RB/HO leg
    rho_blended = (2 * rho_crrb + 1 * rho_crhb) / 3

    # Structure 1: Sell crack forward (premium 0)
    struct_1 = {
        "name": "Sell crack forward",
        "description": "Short the 3:2:1 crack spread at the current forward level",
        "premium": 0.0,
        "method": "forward",
        "strike": float(crack_fwd),
    }

    # Structure 2: Buy crack put (Kirk + spread MC)
    K_put = crack_fwd
    kirk_price = np.asarray(
        pricers_25_exotic.kirk(
            F_crack_leg,
            F_cl,
            K_put,
            T,
            sigma_ho * 0.67 + sigma_rb * 0.33,
            sigma_cl,
            rho_blended,
            df,
            "put",
        )
    ).item()

    # Spread MC: simulate correlated paths
    n_paths_mc = 5000
    seed_mc = 42
    paths_blend, paths_cl_mc = lib25.simulate_correlated(
        [F_crack_leg, F_cl],
        [sigma_ho * 0.67 + sigma_rb * 0.33, sigma_cl],
        rho_blended,
        T,
        int(np.ceil(T * 252)),
        n_paths_mc,
        seed_mc,
    )
    mc_result = pricers_25_exotic.spread_mc(paths_blend, paths_cl_mc, K_put, "put", df)

    struct_2 = {
        "name": "Buy crack put",
        "description": "Protection on crack spread collapse",
        "premium": float(kirk_price),
        "method": "Kirk closed form",
        "strike": float(K_put),
        "kirk_price": float(kirk_price),
        "mc_price": float(mc_result["price"]),
        "mc_se": float(mc_result["se"]),
    }

    # Structure 3: Crack collar (long put, short call)
    K_call = crack_fwd * 1.1  # OTM call
    collar_put_price = np.asarray(
        pricers_25_exotic.kirk(
            F_crack_leg,
            F_cl,
            K_put,
            T,
            sigma_ho * 0.67 + sigma_rb * 0.33,
            sigma_cl,
            rho_blended,
            df,
            "put",
        )
    ).item()
    collar_call_price = np.asarray(
        pricers_25_exotic.kirk(
            F_crack_leg,
            F_cl,
            K_call,
            T,
            sigma_ho * 0.67 + sigma_rb * 0.33,
            sigma_cl,
            rho_blended,
            df,
            "call",
        )
    ).item()

    struct_3 = {
        "name": "Crack collar",
        "description": f"Long put at K={K_put:.2f}, short call at K={K_call:.2f}",
        "premium": float(collar_put_price - collar_call_price),
        "method": "Kirk closed form",
        "put_strike": float(K_put),
        "call_strike": float(K_call),
    }

    # Realised payoffs: historical
    # Load panels for CL, HO, RB
    panel_cl = lib24.load_panel("CL")
    panel_ho = lib24.load_panel("HO")
    panel_rb = lib24.load_panel("RB")

    # Compute historical crack at each date, then payoff T days later
    dates = sorted(
        set(panel_cl["date"]) & set(panel_ho["date"]) & set(panel_rb["date"])
    )

    lookforward_days = int(np.ceil(T * 252))
    payoff_1_list, payoff_2_list, payoff_3_list = [], [], []

    for start_date in dates[:: max(1, len(dates) // 100)]:  # Downsample for speed
        # Get front-month settle at start_date
        cl_start = panel_cl[panel_cl["date"] == start_date].sort_values("dte").head(1)
        ho_start = panel_ho[panel_ho["date"] == start_date].sort_values("dte").head(1)
        rb_start = panel_rb[panel_rb["date"] == start_date].sort_values("dte").head(1)

        if len(cl_start) == 0 or len(ho_start) == 0 or len(rb_start) == 0:
            continue

        # Project forward to start_date + lookforward_days
        end_date = start_date + pd.Timedelta(days=lookforward_days)
        cl_end = (
            panel_cl[
                (panel_cl["date"] >= end_date)
                & (panel_cl["date"] <= end_date + pd.Timedelta(days=2))
            ]
            .sort_values("dte")
            .head(1)
        )
        ho_end = (
            panel_ho[
                (panel_ho["date"] >= end_date)
                & (panel_ho["date"] <= end_date + pd.Timedelta(days=2))
            ]
            .sort_values("dte")
            .head(1)
        )
        rb_end = (
            panel_rb[
                (panel_rb["date"] >= end_date)
                & (panel_rb["date"] <= end_date + pd.Timedelta(days=2))
            ]
            .sort_values("dte")
            .head(1)
        )

        if len(cl_end) == 0 or len(ho_end) == 0 or len(rb_end) == 0:
            continue

        F_cl_t = float(cl_start["close"].iloc[0])
        F_ho_t = float(ho_start["close"].iloc[0])
        F_rb_t = float(rb_start["close"].iloc[0])

        F_cl_T = float(cl_end["close"].iloc[0])
        F_ho_T = float(ho_end["close"].iloc[0])
        F_rb_T = float(rb_end["close"].iloc[0])

        crack_t = (2 * F_rb_t * 42 + 1 * F_ho_t * 42) / 3 - F_cl_t
        crack_T = (2 * F_rb_T * 42 + 1 * F_ho_T * 42) / 3 - F_cl_T

        # Payoffs
        payoff_1 = -(crack_T - crack_t)  # Short forward
        payoff_2 = max(-crack_T + crack_t, 0)  # Long put
        payoff_3 = max(-crack_T + crack_t, 0) - max(
            crack_T - crack_t - (K_call - crack_t), 0
        )  # Collar

        payoff_1_list.append(payoff_1)
        payoff_2_list.append(payoff_2)
        payoff_3_list.append(payoff_3)

    result = {
        "exposure": "long crude, short products -- the 3:2:1 crack spread margin, not the oil price",
        "valuation_date": str(valuation_date.date()),
        "structures": [struct_1, struct_2, struct_3],
        "realised_payoffs": {
            "Sell crack forward": summarise_payoffs(payoff_1_list),
            "Buy crack put": summarise_payoffs(payoff_2_list),
            "Crack collar": summarise_payoffs(payoff_3_list),
        },
        "trade_off_numbers": {
            "rho_crhb": float(rho_crhb),
            "rho_crrb": float(rho_crrb),
            "crack_forward": float(crack_fwd),
            "kirk_vs_mc_gap_pct": float(
                100 * abs(kirk_price - mc_result["price"]) / (abs(kirk_price) + 1e-6)
            ),
        },
    }

    print(f"done (3 structures, {len(payoff_1_list)} historical payoffs)")
    return result


# ========================================================================== #
# Case B: Wheat Farmer (THE BEST ONE)
# ========================================================================== #


def case_b_farmer() -> dict:
    """ZW wheat farmer: floor on harvest price. Knock-out analysis is the punchline."""
    print("Case B: Farmer (ZW wheat)...", end=" ", flush=True)

    # Find nearest date with ZW data to 2017-06-04
    target_date = pd.Timestamp("2017-06-04")
    panel_probe = lib24.load_panel("ZW")
    closest_date = panel_probe.loc[
        (panel_probe["date"] - target_date).abs().idxmin(), "date"
    ]
    valuation_date = pd.Timestamp(str(closest_date))
    T = 0.3  # Harvest horizon ~4 months

    inputs = lib25.inputs_as_of("ZW", valuation_date)
    F = np.asarray(inputs["F"](T)).item()
    sigma = np.asarray(inputs["sigma"](T)).item()
    df = np.asarray(inputs["df"](T)).item()

    # Structure 1: Plain put at K=F
    put_price = lib25.black76(F, F, T, sigma, df, "put")
    struct_1 = {
        "name": "Plain put",
        "description": f"Floor at forward level K={F:.2f}",
        "premium": float(put_price),
        "method": "Black-76",
        "strike": float(F),
    }

    # Structure 2: Zero-cost collar
    collar_result = lib25.zero_cost_collar(F, 0.95 * F, T, sigma, df)
    struct_2 = {
        "name": "Zero-cost collar",
        "description": f"Long put at K={0.95 * F:.2f}, short call at K={collar_result['K_call']:.2f}",
        "premium": float(collar_result["premium"]),
        "method": "Black-76 (solved call strike)",
        "put_strike": float(0.95 * F),
        "call_strike": float(collar_result["K_call"]),
    }

    # Structure 3: Knock-out put
    B_ko = 0.85 * F
    ko_put_price = pricers_25_barrier.barrier_analytic(
        F, F, B_ko, T, sigma, df, "put", "do"
    )
    struct_3 = {
        "name": "Knock-out put",
        "description": f"Floor at K={F:.2f}, down-and-out barrier at B={B_ko:.2f}",
        "premium": float(ko_put_price),
        "method": "Reiner-Rubinstein barrier analytic",
        "strike": float(F),
        "barrier": float(B_ko),
    }

    # Realised payoffs: historical, with knock-out frequency analysis
    panel = lib24.load_panel("ZW")
    panel = panel.sort_values("date").reset_index(drop=True)

    barrier_level = 0.85
    lookforward_days = round(T * 365.25)

    payoff_1_list, payoff_2_list, payoff_3_list = [], [], []
    ko_knockout_count, ko_total_count = 0, 0
    ko_costly_count = 0  # knock-outs where the plain put would have paid > 0
    ko_shortfalls = []
    # The zero-cost collar's call strike as a ratio of the valuation-date
    # forward, so the historical loop can re-strike it at each date's own F_t.
    call_strike_ratio = float(collar_result["K_call"]) / float(F)

    for i, start_date in enumerate(panel["date"].unique()):
        if i % max(1, len(panel["date"].unique()) // 50) != 0:
            continue  # Downsample

        # Pick the contract with dte closest to the target horizon (NOT the
        # front month, whose dte is usually too short to look forward a full
        # ~4-month harvest horizon within the same contract without chaining
        # across a roll -- using the front month here left almost no valid
        # starting points).
        day_rows = panel[
            (panel["date"] == start_date) & (panel["dte"] >= lookforward_days)
        ]
        if len(day_rows) == 0:
            continue
        start_row = day_rows.iloc[[(day_rows["dte"] - lookforward_days).abs().argmin()]]

        start_cid = start_row["contract_id"].iloc[0]
        F_t = float(start_row["close"].iloc[0])

        # Find end date within same contract (don't chain across rolls)
        end_date = start_date + pd.Timedelta(days=lookforward_days)
        end_rows = panel[
            (panel["date"] >= end_date)
            & (panel["date"] <= end_date + pd.Timedelta(days=2))
            & (panel["contract_id"] == start_cid)
        ]

        if len(end_rows) == 0:
            continue

        end_row = end_rows.sort_values("dte").head(1)
        F_T = float(end_row["close"].iloc[0])

        # Payoffs (long the crop, so put protects downside).
        #
        # Every structure is RE-STRUCK at each historical start date's own
        # forward F_t, exactly as the knock-out barrier below is. Striking
        # them at the single 2017 valuation-date forward F instead would
        # compare a 2017 strike against sixteen years of wheat prices, which
        # says more about where wheat traded in 2017 than about the hedge.
        # Case A re-strikes the same way (crack_t, not a frozen crack_0).
        K_put_t = F_t
        K_collar_put_t = 0.95 * F_t
        K_collar_call_t = F_t * call_strike_ratio
        payoff_1 = max(K_put_t - F_T, 0)  # Long put at K=F_t
        payoff_2 = max(K_collar_put_t - F_T, 0) - max(
            F_T - K_collar_call_t, 0
        )  # Collar

        # For knock-out: monitor the path from start to end within the contract
        path_rows = panel[
            (panel["date"] >= start_date)
            & (panel["date"] <= end_date + pd.Timedelta(days=2))
            & (panel["contract_id"] == start_cid)
        ].sort_values("date")

        if len(path_rows) == 0:
            continue

        path_prices = path_rows["close"].to_numpy()
        barrier = F_t * barrier_level
        hit_barrier = np.any(path_prices <= barrier)
        ko_total_count += 1

        if hit_barrier:
            ko_knockout_count += 1
            # What the plain put would have paid, in ZW quote units
            # (cents per bushel -- ZW is quoted in cents, not dollars).
            shortfall = max(K_put_t - F_T, 0)
            ko_shortfalls.append(shortfall)
            if shortfall > 0:
                ko_costly_count += 1
            payoff_3 = 0.0  # Knocked out
        else:
            payoff_3 = max(K_put_t - F_T, 0)  # Still alive

        payoff_1_list.append(payoff_1)
        payoff_2_list.append(payoff_2)
        payoff_3_list.append(payoff_3)

    ko_freq = float(ko_knockout_count / max(1, ko_total_count))

    # Cross-check with KE
    panel_ke = lib24.load_panel("KE")
    panel_ke = panel_ke.sort_values("date").reset_index(drop=True)
    ko_ko_count_ke, ko_total_count_ke = 0, 0
    for i, start_date in enumerate(panel_ke["date"].unique()):
        if i % max(1, len(panel_ke["date"].unique()) // 50) != 0:
            continue
        day_rows_ke = panel_ke[
            (panel_ke["date"] == start_date) & (panel_ke["dte"] >= lookforward_days)
        ]
        if len(day_rows_ke) == 0:
            continue
        start_row = day_rows_ke.iloc[
            [(day_rows_ke["dte"] - lookforward_days).abs().argmin()]
        ]
        start_cid = start_row["contract_id"].iloc[0]
        F_t_ke = float(start_row["close"].iloc[0])
        end_date = start_date + pd.Timedelta(days=lookforward_days)
        end_rows = panel_ke[
            (panel_ke["date"] >= end_date)
            & (panel_ke["date"] <= end_date + pd.Timedelta(days=2))
            & (panel_ke["contract_id"] == start_cid)
        ]
        if len(end_rows) == 0:
            continue
        path_rows = panel_ke[
            (panel_ke["date"] >= start_date)
            & (panel_ke["date"] <= end_date + pd.Timedelta(days=2))
            & (panel_ke["contract_id"] == start_cid)
        ].sort_values("date")
        if len(path_rows) == 0:
            continue
        path_prices = path_rows["close"].to_numpy()
        barrier = F_t_ke * barrier_level
        hit_barrier = np.any(path_prices <= barrier)
        ko_total_count_ke += 1
        if hit_barrier:
            ko_ko_count_ke += 1

    ko_freq_ke = float(ko_ko_count_ke / max(1, ko_total_count_ke))

    result = {
        "exposure": "short the harvest price on ZW (cross-check KE); wants a floor, balks at the premium",
        "valuation_date": str(valuation_date.date()),
        "structures": [struct_1, struct_2, struct_3],
        "realised_payoffs": {
            "Plain put": summarise_payoffs(payoff_1_list),
            "Zero-cost collar": summarise_payoffs(payoff_2_list),
            "Knock-out put": summarise_payoffs(payoff_3_list),
        },
        "trade_off_numbers": {
            "quote_units": "cents per bushel (ZW and KE are quoted in cents)",
            "ko_barrier_pct_of_forward": float(barrier_level),
            "ko_knockout_frequency_zw": float(ko_freq),
            "ko_knockout_frequency_ke": float(ko_freq_ke),
            # Mean over ALL knock-outs, including the ones where the put would
            # have expired worthless anyway and the knock-out therefore cost
            # the farmer nothing. In cents/bushel, not dollars.
            "mean_shortfall_when_ko_cents": float(np.mean(ko_shortfalls))
            if ko_shortfalls
            else 0.0,
            # The number that actually matters: knock-outs that cost something,
            # and the mean shortfall over just those. A knock-out on a year the
            # price recovered above the strike is free.
            "n_knockouts": int(ko_knockout_count),
            "n_costly_knockouts": int(ko_costly_count),
            "costly_knockout_frequency_zw": float(
                ko_costly_count / max(1, ko_total_count)
            ),
            "mean_shortfall_when_costly_cents": float(
                np.mean([x for x in ko_shortfalls if x > 0])
            )
            if ko_costly_count
            else 0.0,
            "n_observations": len(payoff_1_list),
        },
    }

    print(
        f"done (3 structures, KO freq ZW={ko_freq:.2%} KE={ko_freq_ke:.2%}, n={len(payoff_1_list)})"
    )
    return result


# ========================================================================== #
# Case C: Airline (HO monthly average)
# ========================================================================== #


def case_c_airline() -> dict:
    """HO monthly average buyer: Asian pricing and vol reduction."""
    print("Case C: Airline (HO monthly average)...", end=" ", flush=True)

    # Find nearest date with HO data to 2017-06-04
    target_date = pd.Timestamp("2017-06-04")
    panel_probe = lib24.load_panel("HO")
    closest_date = panel_probe.loc[
        (panel_probe["date"] - target_date).abs().idxmin(), "date"
    ]
    valuation_date = pd.Timestamp(str(closest_date))
    T = 0.083  # ~1 month
    month_str = "2017-06"  # A representative month with full history

    inputs = lib25.inputs_as_of("HO", valuation_date)
    F = float(np.asarray(inputs["F"](T)).flat[0])
    sigma = float(np.asarray(inputs["sigma"](T)).item())
    df = float(np.asarray(inputs["df"](T)).item())

    # Asian averaging schedule
    avg_sched = pricers_25_asian.averaging_schedule("HO", month_str)
    avg_dates = pd.to_datetime(avg_sched)
    # Only fixings strictly after the valuation date are still unknown. An
    # averaging window that starts before today is an already-started Asian,
    # whose observed fixings are no longer random -- pricing one as if the
    # whole window were still ahead overstates its optionality. Keep only the
    # forward part of the schedule.
    avg_dates = avg_dates[avg_dates > pd.Timestamp(valuation_date)]
    avg_start = (avg_dates[0] - pd.Timestamp(valuation_date)).days / 365.25
    n_fix = len(avg_dates)

    # Structure 1: Strip of futures (premium 0)
    struct_1 = {
        "name": "Strip of futures",
        "description": "Buy front-month HO futures",
        "premium": 0.0,
        "method": "forward",
        "strike": float(F),
    }

    # Structure 2: Asian call via geometric and Turnbull-Wakeman
    asian_geo_price = pricers_25_asian.asian_geometric(
        F, F, T, sigma, df, "call", avg_start, n_fix
    )
    asian_tw_price = pricers_25_asian.asian_turnbull_wakeman(
        F, F, T, sigma, df, "call", avg_start, n_fix
    )

    # The like-for-like European: same strike, same expiry, single fixing.
    # The Asian must price below it, because averaging reduces the variance
    # of the quantity being optioned.
    european_price = float(np.asarray(lib25.black76(F, F, T, sigma, df, "call")).item())

    # Wrong textbook approximation: sigma / sqrt(3)
    sigma_wrong = sigma / np.sqrt(3)
    wrong_price = lib25.black76(F, F, T, sigma_wrong, df, "call")

    struct_2 = {
        "name": "Asian call",
        "description": f"Average-price call over {n_fix} fixing dates in {month_str}",
        "premium": float(asian_tw_price),
        "method": "Turnbull-Wakeman (arithmetic average)",
        "geometric_price": float(asian_geo_price),
        "wrong_sigma_sqrt3_price": float(wrong_price),
        "wrong_vs_correct_pct": float(
            100 * (wrong_price - asian_tw_price) / max(abs(asian_tw_price), 1e-6)
        ),
    }

    # Structure 3: Asian collar.
    #
    # The airline BUYS diesel, so it is short the commodity and wants a CAP.
    # Its collar is therefore long the call (the cap it wants) and short the
    # put (the floor it sells to pay for it) -- the mirror image of the wheat
    # farmer's collar in case B, who is long the crop and wants a floor.
    # Pricing this the farmer's way round would hedge the wrong direction.
    K_cap = F * 1.05
    K_floor = F * 0.95
    asian_call_long = pricers_25_asian.asian_turnbull_wakeman(
        F, K_cap, T, sigma, df, "call", avg_start, n_fix
    )
    asian_put_short = pricers_25_asian.asian_turnbull_wakeman(
        F, K_floor, T, sigma, df, "put", avg_start, n_fix
    )

    struct_3 = {
        "name": "Asian collar",
        "description": f"Long Asian call (cap) at K={K_cap:.2f}, short Asian put (floor) at K={K_floor:.2f}",
        "premium": float(asian_call_long - asian_put_short),
        "method": "Turnbull-Wakeman for both legs",
        "cap_strike": float(K_cap),
        "floor_strike": float(K_floor),
    }

    # Realised payoffs: historical monthly averages
    panel = lib24.load_panel("HO")
    lookforward_days = int(np.ceil(T * 252))

    payoff_1_list, payoff_2_list, payoff_3_list = [], [], []

    for start_date in panel["date"].unique()[
        :: max(1, len(panel["date"].unique()) // 50)
    ]:
        start_row = panel[panel["date"] == start_date].sort_values("dte").head(1)
        if len(start_row) == 0:
            continue

        F_t = float(start_row["close"].iloc[0])

        # Find data T months later
        end_date = start_date + pd.Timedelta(days=lookforward_days)
        end_rows = (
            panel[
                (panel["date"] >= end_date)
                & (panel["date"] <= end_date + pd.Timedelta(days=2))
            ]
            .sort_values("dte")
            .head(1)
        )

        if len(end_rows) == 0:
            continue

        F_T = float(end_rows["close"].iloc[0])

        # Payoffs. The airline is a BUYER, so it hedges by going LONG
        # futures: the hedge pays +(F_T - F_t) when diesel rallies, which is
        # what offsets its higher physical bill. (Case A's refiner is short
        # the crack and correctly carries the opposite sign; copying that sign
        # here would report every hedge backwards.)
        payoff_1 = F_T - F_t  # Long futures
        payoff_2 = max(F_T - F_t, 0)  # Long call (the cap)
        payoff_3 = max(F_T - F_t * 1.05, 0) - max(
            F_t * 0.95 - F_T, 0
        )  # Collar: long cap, short floor -- the same legs priced above

        payoff_1_list.append(payoff_1)
        payoff_2_list.append(payoff_2)
        payoff_3_list.append(payoff_3)

    # Vol comparison: the vol of the monthly AVERAGE vs the vol of a single
    # settlement price. This is the whole point of an Asian option, so it has
    # to be computed on the right quantity:
    #
    #   * both legs stay INSIDE one contract -- a ratio taken across two
    #     different contracts is a roll, not a return;
    #   * the Asian quantity is the mean of the month's closes, not the
    #     month's last close over its first;
    #   * month-over-month observations annualise with sqrt(12), not the
    #     sqrt(252) that lib24.mad_vol assumes for daily data.
    #
    # Getting any of the three wrong inflates the ratio above 1 and inverts
    # the finding, since averaging can only ever reduce variance.
    panel_clean = lib24.usable_returns(panel)
    daily_vol = lib24.mad_vol(panel_clean["r"].to_numpy())

    panel = panel.copy()
    panel["year_month"] = panel["date"].dt.to_period("M")

    # Monthly average price per (contract, month), then month-over-month log
    # changes within each contract.
    avg_by_contract_month = (
        panel.groupby(["contract_id", "year_month"])["close"].mean().reset_index()
    )
    avg_returns, last_returns = [], []
    for _, g in avg_by_contract_month.groupby("contract_id"):
        g = g.sort_values("year_month")
        if len(g) > 1:
            avg_returns.extend(np.diff(np.log(g["close"].to_numpy())).tolist())

    # Same schedule, but the single month-end close instead of the average --
    # the like-for-like European comparison.
    last_by_contract_month = (
        panel.sort_values("date")
        .groupby(["contract_id", "year_month"])["close"]
        .last()
        .reset_index()
    )
    for _, g in last_by_contract_month.groupby("contract_id"):
        g = g.sort_values("year_month")
        if len(g) > 1:
            last_returns.extend(np.diff(np.log(g["close"].to_numpy())).tolist())

    monthly_vol = (
        lib24.mad_vol(np.array(avg_returns), periods=12)
        if avg_returns
        else float("nan")
    )
    monthly_single_vol = (
        lib24.mad_vol(np.array(last_returns), periods=12)
        if last_returns
        else float("nan")
    )

    result = {
        "exposure": "buys physical diesel at a monthly average price on HO",
        "valuation_date": str(valuation_date.date()),
        "structures": [struct_1, struct_2, struct_3],
        "realised_payoffs": {
            "Strip of futures": summarise_payoffs(payoff_1_list),
            "Asian call": summarise_payoffs(payoff_2_list),
            "Asian collar": summarise_payoffs(payoff_3_list),
        },
        "trade_off_numbers": {
            "daily_vol_realized": float(daily_vol),
            # Like-for-like, both annualised from monthly observations inside
            # one contract: the vol of the month's AVERAGE vs the vol of the
            # month's single closing price.
            "monthly_avg_vol_realized": float(monthly_vol),
            "monthly_single_date_vol_realized": float(monthly_single_vol),
            "vol_reduction_ratio": float(monthly_vol / monthly_single_vol)
            if monthly_single_vol > 0
            else float("nan"),
            "vol_reduction_ratio_note": (
                "vol of the monthly average / vol of a single month-end close, "
                "both from month-over-month changes within a contract. Must be "
                "< 1: averaging cannot add variance."
            ),
            "asian_vs_european_premium_pct": float(
                100 * (asian_tw_price - european_price) / max(abs(european_price), 1e-6)
            ),
            "european_price": float(european_price),
            "asian_fixing_count": int(n_fix),
            "wrong_vs_correct_pitfall_pct": float(
                100 * (wrong_price - asian_tw_price) / max(abs(asian_tw_price), 1e-6)
            ),
        },
    }

    print(
        f"done (3 structures, {n_fix} fixings, vol ratio "
        f"{monthly_vol / monthly_single_vol:.3f}, n={len(payoff_1_list)})"
    )
    return result


# ========================================================================== #
# Case D: Gas Utility (NG winter strip with seasonality)
# ========================================================================== #


def case_d_gas_utility() -> dict:
    """NG winter (Dec-Mar): strip with caps and seasonal vol effect."""
    print("Case D: Gas utility (NG winter)...", end=" ", flush=True)

    # Find nearest date with NG data to 2017-06-04
    target_date = pd.Timestamp("2017-06-04")
    panel_probe = lib24.load_panel("NG")
    closest_date = panel_probe.loc[
        (panel_probe["date"] - target_date).abs().idxmin(), "date"
    ]
    valuation_date = pd.Timestamp(str(closest_date))
    T = 0.5  # Next winter, ~6 months out

    inputs = lib25.inputs_as_of("NG", valuation_date)

    # Winter strip: Dec (T~0.5), Jan (T~0.67), Feb (T~0.83), Mar (T~1.0)
    winter_months = [
        ("Dec", 0.5),
        ("Jan", 0.67),
        ("Feb", 0.83),
        ("Mar", 1.0),
    ]
    WINTER_DELIVERY = (12, 1, 2, 3)
    SUMMER_DELIVERY = (4, 5, 6, 7, 8, 9, 10, 11)

    # Three vol inputs, so the two effects that get conflated stay separate:
    #
    #   1. FLAT       sigma(T_Dec) for every month -- the naive desk that
    #                 quotes the whole strip off one front-month number.
    #   2. MATURITY   sigma(T_i) per month from the unconditional Samuelson
    #                 power law. Vol falls with tenor, so the later winter
    #                 months price below the flat number.
    #   3. SEASONAL   sigma_winter(T_i), the same fit restricted to contracts
    #                 whose DELIVERY month is Dec-Mar (024 Phase 5's split).
    #                 This is the only one of the three that is actually about
    #                 seasonality; (2) is purely about time to expiry.
    #
    # The original version of this case study called (2) "seasonal" and
    # compared it to (1). That gap is real but it is the Samuelson effect
    # wearing a seasonal label -- inputs["sigma"] has no month-of-year term
    # in it at all. Measuring (3) separately is what makes the claim testable.
    sigma_flat = float(np.asarray(inputs["sigma"](T)).item())

    vol_winter = lib25.vol_term_structure(
        "NG", valuation_date, delivery_months=WINTER_DELIVERY
    )
    vol_summer = lib25.vol_term_structure(
        "NG", valuation_date, delivery_months=SUMMER_DELIVERY
    )

    # Keep the raw bucket-by-bucket winter/summer comparison the fits are
    # built from, so the seasonality claim can be read off the data rather
    # than taken on trust from two power-law coefficients. 024 Phase 5 finds
    # the winter premium concentrated at the front end; this records whether
    # it survives at the 6-12 month tenors this hedge actually spans.
    _ng_panel = lib24.load_panel("NG")
    _train = _ng_panel[
        (_ng_panel["date"] < valuation_date)
        & (_ng_panel["date"] >= valuation_date - pd.Timedelta(days=756))
    ]
    _train = lib24.usable_returns(_train)
    _months = lib25.delivery_month(_train)
    _bw = lib24.vol_by_bucket(
        _train[_months.isin([float(m) for m in WINTER_DELIVERY])], lib24.mad_vol
    )
    _bs = lib24.vol_by_bucket(
        _train[_months.isin([float(m) for m in SUMMER_DELIVERY])], lib24.mad_vol
    )
    _merged = _bw.merge(_bs, on="dte_mid", suffixes=("_winter", "_summer"))
    seasonal_buckets = [
        {
            "dte_mid": float(row["dte_mid"]),
            "vol_winter": float(row["vol_winter"]),
            "vol_summer": float(row["vol_summer"]),
            "ratio": float(row["vol_winter"] / row["vol_summer"]),
            "n_obs_winter": int(row["n_obs_winter"]),
            "n_obs_summer": int(row["n_obs_summer"]),
        }
        for _, row in _merged.iterrows()
    ]

    def _sig(fn, tau: float) -> float:
        return float(np.asarray(fn(tau)).flat[0])

    flat_cap_premium = 0.0
    maturity_cap_premium = 0.0
    seasonal_cap_premium = 0.0
    per_month = []
    for month, T_i in winter_months:
        F_i = float(np.asarray(inputs["F"](T_i)).flat[0])
        df_i = float(np.asarray(inputs["df"](T_i)).item())
        sigma_mat_i = _sig(inputs["sigma"], T_i)
        sigma_sea_i = _sig(vol_winter["fn"], T_i)

        p_flat = float(
            np.asarray(lib25.black76(F_i, F_i, T_i, sigma_flat, df_i, "call")).item()
        )
        p_mat = float(
            np.asarray(lib25.black76(F_i, F_i, T_i, sigma_mat_i, df_i, "call")).item()
        )
        p_sea = float(
            np.asarray(lib25.black76(F_i, F_i, T_i, sigma_sea_i, df_i, "call")).item()
        )
        flat_cap_premium += p_flat
        maturity_cap_premium += p_mat
        seasonal_cap_premium += p_sea
        per_month.append(
            {
                "month": month,
                "T": T_i,
                "F": F_i,
                "sigma_flat": sigma_flat,
                "sigma_maturity": sigma_mat_i,
                "sigma_seasonal_winter": sigma_sea_i,
                "premium_flat": p_flat,
                "premium_maturity": p_mat,
                "premium_seasonal": p_sea,
            }
        )

    # Structure 1: Winter strip (premium 0)
    struct_1 = {
        "name": "Winter strip of futures",
        "description": "Long futures for Dec, Jan, Feb, Mar delivery",
        "premium": 0.0,
        "method": "forward strip",
        "months": winter_months,
    }

    # Structure 2: Strip of caps, priced on the seasonal (delivery-month
    # conditional) vol, with the other two vol inputs carried alongside.
    struct_2 = {
        "name": "Strip of caps (calls)",
        "description": "Long call on each of 4 winter months, struck at forward",
        "premium": float(seasonal_cap_premium),
        "method": "Black-76 per month (winter-delivery seasonal vol)",
        "flat_vol_premium": float(flat_cap_premium),
        "maturity_vol_premium": float(maturity_cap_premium),
        "per_month": per_month,
    }

    # Structure 3: Cheaper cap with a knock-out.
    #
    # A down-and-out call struck at the forward can only knock out in states
    # where the call is already far out of the money, so the barrier has to be
    # close to the strike before it removes any value worth paying for. At
    # B = 0.7 * F the saving is ~1e-7 of premium: the structure is a no-op
    # dressed up as a cheaper hedge. Sweep the barrier instead of asserting a
    # level, so the reader can see where it starts to bite -- and price the
    # headline structure at a barrier that actually does something.
    ko_ladder = []
    for b_pct in (0.70, 0.80, 0.85, 0.90, 0.95):
        total = 0.0
        for month, T_i in winter_months:
            F_i = float(np.asarray(inputs["F"](T_i)).flat[0])
            df_i = float(np.asarray(inputs["df"](T_i)).item())
            sigma_i = _sig(vol_winter["fn"], T_i)
            ko_call = pricers_25_barrier.barrier_analytic(
                F_i, F_i, b_pct * F_i, T_i, sigma_i, df_i, "call", "do"
            )
            total += float(np.asarray(ko_call).item())
        ko_ladder.append(
            {
                "barrier_pct_of_forward": b_pct,
                "premium": total,
                "savings_vs_plain_cap": float(seasonal_cap_premium - total),
                "savings_pct": float(
                    100
                    * (seasonal_cap_premium - total)
                    / max(abs(seasonal_cap_premium), 1e-12)
                ),
            }
        )

    ko_headline_pct = 0.90
    ko_cap_premium = next(
        row["premium"]
        for row in ko_ladder
        if row["barrier_pct_of_forward"] == ko_headline_pct
    )

    struct_3 = {
        "name": "Cheaper cap with knock-out",
        "description": (
            f"Down-and-out call for each month, barrier at "
            f"{ko_headline_pct:.0%} of that month's forward"
        ),
        "premium": float(ko_cap_premium),
        "method": "Reiner-Rubinstein barrier analytic",
        "barrier_pct_of_forward": ko_headline_pct,
        "savings_vs_cap": float(seasonal_cap_premium - ko_cap_premium),
        "savings_pct": float(
            100
            * (seasonal_cap_premium - ko_cap_premium)
            / max(abs(seasonal_cap_premium), 1e-12)
        ),
        "barrier_ladder": ko_ladder,
    }

    # Realised payoffs: historical
    panel = lib24.load_panel("NG")
    lookforward_days = int(np.ceil(T * 252))

    payoff_1_list, payoff_2_list, payoff_3_list = [], [], []
    ko_hit_count, ko_window_count = 0, 0

    for start_date in panel["date"].unique()[
        :: max(1, len(panel["date"].unique()) // 50)
    ]:
        start_row = panel[panel["date"] == start_date].sort_values("dte").head(1)
        if len(start_row) == 0:
            continue

        F_t = float(start_row["close"].iloc[0])

        end_date = start_date + pd.Timedelta(days=lookforward_days)
        end_rows = (
            panel[
                (panel["date"] >= end_date)
                & (panel["date"] <= end_date + pd.Timedelta(days=2))
            ]
            .sort_values("dte")
            .head(1)
        )

        if len(end_rows) == 0:
            continue

        F_T = float(end_rows["close"].iloc[0])

        # The utility BUYS gas, so it hedges by going LONG the winter strip:
        # the hedge pays +(F_T - F_t) when gas rallies, offsetting its higher
        # fuel bill. (Case A's refiner is short the crack and correctly
        # carries the opposite sign.)
        payoff_1 = F_T - F_t  # Long futures
        payoff_2 = max(F_T - F_t, 0)  # Long call (the cap)

        # The knock-out has to be monitored along the PATH, not tested once at
        # expiry: a barrier that is only checked against F_T is not a barrier.
        # Testing F_T alone also made this payoff identical to the plain cap by
        # construction, since a call that finishes in the money has F_T > F_t
        # and so can never fail an F_T >= 0.7 * F_t test.
        path_rows = panel[
            (panel["date"] >= start_date)
            & (panel["date"] <= end_date + pd.Timedelta(days=2))
            & (panel["contract_id"] == start_row["contract_id"].iloc[0])
        ]
        path_prices = path_rows.sort_values("date")["close"].to_numpy()
        knocked_out = bool(
            len(path_prices) and np.any(path_prices <= ko_headline_pct * F_t)
        )
        payoff_3 = 0.0 if knocked_out else max(F_T - F_t, 0)  # KO call
        if knocked_out:
            ko_hit_count += 1
        ko_window_count += 1

        payoff_1_list.append(payoff_1)
        payoff_2_list.append(payoff_2)
        payoff_3_list.append(payoff_3)

    result = {
        "exposure": "winter NG demand, seasonal vol (024 Phase 5)",
        "valuation_date": str(valuation_date.date()),
        "structures": [struct_1, struct_2, struct_3],
        "realised_payoffs": {
            "Winter strip": summarise_payoffs(payoff_1_list),
            "Strip of caps": summarise_payoffs(payoff_2_list),
            "Cheaper cap with knock-out": summarise_payoffs(payoff_3_list),
        },
        "trade_off_numbers": {
            "flat_vol_cap_premium": float(flat_cap_premium),
            "maturity_vol_cap_premium": float(maturity_cap_premium),
            "seasonal_cap_premium": float(seasonal_cap_premium),
            # Effect 1: pricing the whole strip off one front-month vol vs
            # letting vol decay with tenor. This is the Samuelson effect.
            "flat_overprices_vs_maturity_pct": float(
                100
                * (flat_cap_premium - maturity_cap_premium)
                / max(abs(maturity_cap_premium), 1e-12)
            ),
            # Effect 2: what conditioning on winter DELIVERY adds on top --
            # the only part of this case study that is about seasonality.
            "seasonal_vs_maturity_pct": float(
                100
                * (seasonal_cap_premium - maturity_cap_premium)
                / max(abs(maturity_cap_premium), 1e-12)
            ),
            "sigma_winter_at_half_year": _sig(vol_winter["fn"], 0.5),
            "sigma_summer_at_half_year": _sig(vol_summer["fn"], 0.5),
            "sigma_winter_over_summer": float(
                _sig(vol_winter["fn"], 0.5) / max(_sig(vol_summer["fn"], 0.5), 1e-12)
            ),
            "vol_winter_fit": {
                "sigma_1": vol_winter["sigma_1"],
                "k": vol_winter["k"],
                "r2": vol_winter["r2"],
            },
            "vol_summer_fit": {
                "sigma_1": vol_summer["sigma_1"],
                "k": vol_summer["k"],
                "r2": vol_summer["r2"],
            },
            "seasonal_buckets": seasonal_buckets,
            "seasonal_buckets_note": (
                "Winter/summer realised-vol ratio by dte bucket in the causal "
                "756-day window. The winter premium is visible at the front "
                "(dte < 60) but changes sign repeatedly at the 180-365 day "
                "tenors this strip spans, which is why the fitted seasonal "
                "and unconditional term structures are indistinguishable."
            ),
            "ko_headline_barrier_pct": ko_headline_pct,
            "ko_savings": float(seasonal_cap_premium - ko_cap_premium),
            "ko_savings_pct": float(
                100
                * (seasonal_cap_premium - ko_cap_premium)
                / max(abs(seasonal_cap_premium), 1e-12)
            ),
            "ko_realised_hit_frequency": float(ko_hit_count / max(1, ko_window_count)),
            "ko_barrier_ladder": ko_ladder,
        },
    }

    flat_gap = (
        100
        * (flat_cap_premium - maturity_cap_premium)
        / max(abs(maturity_cap_premium), 1e-12)
    )
    sea_gap = (
        100
        * (seasonal_cap_premium - maturity_cap_premium)
        / max(abs(maturity_cap_premium), 1e-12)
    )
    print(
        f"done (3 structures, flat vs maturity {flat_gap:+.1f}%, "
        f"seasonal vs maturity {sea_gap:+.1f}%, n={len(payoff_1_list)})"
    )
    return result


# ========================================================================== #
# Main
# ========================================================================== #


def main():
    """Generate all four case studies."""
    print("\n=== Phase 5: Case Studies A, B, C, D ===\n")
    start_time = time.time()

    results = {
        "A_refiner": case_a_refiner(),
        "B_farmer": case_b_farmer(),
        "C_airline": case_c_airline(),
        "D_gas_utility": case_d_gas_utility(),
    }

    # Write JSON
    output_file = _TMP_DIR / "phase_5_25_cases_abcd.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    file_size_mb = output_file.stat().st_size / 1e6

    print(f"\n✓ Wrote {output_file}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Wall clock: {elapsed:.1f}s")
    print(f"  Top-level keys: {list(results.keys())}")
    for k, v in results.items():
        n_struct = len(v["structures"])
        n_payoff = len(v["realised_payoffs"])
        print(f"    {k}: {n_struct} structures, {n_payoff} realised-payoff entries")


if __name__ == "__main__":
    main()
