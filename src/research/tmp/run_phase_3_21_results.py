"""Notebook 021, Phase 3 (NEXT_PROMPT.md sec 4): diff CIs with/without the
liquidity-collapse exclusion, MDE/n_required/years_required (both series),
and PW-4's placebo. Reuses 020's stored A0/A3 returns parquets for the diff
series (no book rebuild) and rebuilds only A0's/A3's *weights* (needed for
the holdings intersection that defines the exclusion set and, for A3, a
diagnostic comparison of the floor's own immunity).

Refuses to start if phase_1_21_catalogue.json is not on disk -- the one
guard kept from 020's runner architecture (sec 5). PW-1's coverage check
(ICPUSDT/MATICUSDT) is scored here, not in the detector -- the grep
discipline in sec 4/6 applies only to run_phase_1_21_catalogue.py.

Usage: uv run python src/research/tmp/run_phase_3_21_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib20 as bl20
import power_lib21 as pw21

import research

CATALOGUE_JSON = REPO_ROOT / "src" / "research" / "tmp" / "phase_1_21_catalogue.json"
OUT_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_3_21_results.json"

PERP_GLOB = str(
    REPO_ROOT
    / "src"
    / "research"
    / "cache"
    / "basis18"
    / "dev"
    / "*-perp-8h-2021-07-01-2025-06-30.parquet"
)
SCRATCH_020 = REPO_ROOT / "scratch" / "020"
A0_RETURNS = SCRATCH_020 / "cells" / "phase4" / "A0-0.returns.parquet"
A3_RETURNS = SCRATCH_020 / "cells" / "phase4" / "A3-0.returns.parquet"

PW4_CAP_FRAC = 0.05
N_DRAWS = 200
SEED = 0
BARS_PER_YEAR = 1095.75


def _guard() -> None:
    if not CATALOGUE_JSON.exists():
        raise SystemExit(
            f"{CATALOGUE_JSON} missing -- Phase 3 must not start before Phase 1's "
            "catalogue is committed (sec 4/5's ordering guard)"
        )


def _load_diff_frame() -> pl.DataFrame:
    a3 = pl.read_parquet(A3_RETURNS).rename({"trade_log_return_net": "refined_net"})
    a0 = pl.read_parquet(A0_RETURNS).rename({"trade_log_return_net": "baseline_net"})
    joined = a3.select("datetime", "refined_net").join(
        a0.select("datetime", "baseline_net"), on="datetime", how="inner"
    )
    return (
        joined.with_columns(
            (pl.col("refined_net") - pl.col("baseline_net")).alias("diff")
        )
        .select("datetime", "diff")
        .drop_nulls()
    )


def _a0_weights() -> pl.DataFrame:
    panel = pl.read_parquet(
        SCRATCH_020 / f"panel_dev_binance_hl{bl20.CARRY_EWMA_HALF_LIFE}.parquet"
    )
    return bl20.build_book_weights_v2(
        panel, timed=True, n_min=1, theta_in=bl20.THETA_IN, theta_out=bl20.THETA_OUT
    )


def _a3_weights() -> pl.DataFrame:
    panel = pl.read_parquet(
        SCRATCH_020 / f"panel_dev_binance_hl{bl20.SLOW_CARRY_HALF_LIFE}.parquet"
    )
    return bl20.build_book_weights_v2(
        panel,
        timed=True,
        n_min=bl20.N_MIN,
        theta_in=bl20.THETA_IN_SLOW,
        theta_out=bl20.THETA_OUT_SLOW,
    )


def _pw1_check(catalogue: pl.DataFrame) -> dict:
    icp = catalogue.filter(
        (pl.col("symbol") == "ICPUSDT")
        & (pl.col("datetime") >= pl.datetime(2022, 6, 20))
        & (pl.col("datetime") <= pl.datetime(2022, 6, 30))
    )
    matic = catalogue.filter(
        (pl.col("symbol") == "MATICUSDT")
        & (pl.col("datetime") >= pl.datetime(2024, 9, 1))
        & (pl.col("datetime") <= pl.datetime(2024, 9, 30))
    )
    icp_ok = len(icp) >= 1
    matic_ok = len(matic) >= 1
    return {
        "icp_flagged_in_window": icp_ok,
        "icp_n_flagged_in_window": len(icp),
        "matic_flagged_in_window": matic_ok,
        "matic_n_flagged_in_window": len(matic),
        "fires": bool(icp_ok and matic_ok),
    }


def main() -> None:
    _guard()

    catalogue = pw21.flag_frozen_feed_bars(PERP_GLOB)
    pw1 = _pw1_check(catalogue)

    a0_weights = _a0_weights()
    a3_weights = _a3_weights()

    excluded = pw21.excluded_book_bars(catalogue, a0_weights)
    a3_excluded_diag = pw21.excluded_book_bars(catalogue, a3_weights)

    diff_frame = _load_diff_frame()
    diff_dts = diff_frame["datetime"].to_list()
    n_total = len(diff_frame)
    diff_vals_all = diff_frame["diff"].to_numpy()

    n_excluded_book_bars = sum(1 for dt in diff_dts if dt in excluded)
    frac_excluded = n_excluded_book_bars / n_total
    n_a3_excluded_book_bars_diag = sum(1 for dt in diff_dts if dt in a3_excluded_diag)

    diff_vals_excl = diff_frame.filter(~pl.col("datetime").is_in(list(excluded)))[
        "diff"
    ].to_numpy()

    ci_without = research.block_bootstrap_ci(diff_vals_all)
    p_without = research.block_bootstrap_pvalue(diff_vals_all)
    ci_with = research.block_bootstrap_ci(diff_vals_excl)
    p_with = research.block_bootstrap_pvalue(diff_vals_excl)

    mean_without = float(np.mean(diff_vals_all))
    mean_with = float(np.mean(diff_vals_excl))

    pw2_fires = bool(ci_with[0] > 0)

    se_without = pw21.bootstrap_se_from_ci(*ci_without)
    mde_without = pw21.mde(se_without)
    n_required_without = pw21.n_required(n_total, mean_without, mde_without)
    years_without = n_required_without / BARS_PER_YEAR

    se_with = pw21.bootstrap_se_from_ci(*ci_with)
    mde_with = pw21.mde(se_with)
    n_required_with = pw21.n_required(len(diff_vals_excl), mean_with, mde_with)
    years_with = n_required_with / BARS_PER_YEAR

    pw3_fires = bool(mean_without >= mde_without)

    placebo_means = pw21.placebo_mean_diffs(
        diff_frame, n_excluded=n_excluded_book_bars, n_draws=N_DRAWS, seed=SEED
    )
    placebo_p95 = float(np.percentile(placebo_means, 95))
    pw4_cap_ok = bool(frac_excluded < PW4_CAP_FRAC)
    pw4_placebo_ok = bool(mean_with > placebo_p95)
    pw4_fires = bool(pw4_cap_ok and pw4_placebo_ok)

    if not pw1["fires"]:
        branch = (
            "(c) PW-1 does not fire -- detector wrong; PW-2/PW-3/PW-4 are void, stop."
        )
    elif pw2_fires and pw4_fires:
        branch = "(a) PW-2 and PW-4 both fire -- data-quality-corrected RC-3."
    else:
        branch = "(b) data quality is not the binding constraint -- power is."

    result = {
        "n_trials_used": 6,
        "guard": {"phase_1_catalogue_present": True},
        "gate_PW1": pw1,
        "exclusion": {
            "total_flagged_symbol_bars": len(catalogue),
            "n_symbols_flagged": (
                catalogue["symbol"].n_unique() if len(catalogue) else 0
            ),
            "n_excluded_book_return_bars": n_excluded_book_bars,
            "n_total_book_return_bars": n_total,
            "frac_excluded": frac_excluded,
            "pw4_cap_frac": PW4_CAP_FRAC,
        },
        "a3_immunity_diagnostic": {
            "note": (
                "not used for any exclusion -- A0's holdings define the exclusion set "
                "exclusively, per pre-registration. Reported only to show whether A3's "
                "diversification floor reduces exposure to the same flagged bars."
            ),
            "n_book_return_bars_that_would_be_excluded_under_a3_holdings": (
                n_a3_excluded_book_bars_diag
            ),
        },
        "diff_ci_without_exclusion": {
            "n_obs": n_total,
            "mean": mean_without,
            "bootstrap_ci_95": [float(ci_without[0]), float(ci_without[1])],
            "bootstrap_pvalue": float(p_without),
            "ci_excludes_zero": bool(ci_without[0] > 0),
        },
        "diff_ci_with_exclusion": {
            "n_obs": len(diff_vals_excl),
            "mean": mean_with,
            "bootstrap_ci_95": [float(ci_with[0]), float(ci_with[1])],
            "bootstrap_pvalue": float(p_with),
            "ci_excludes_zero": pw2_fires,
        },
        "gate_PW2": {"fires": pw2_fires},
        "power": {
            "without_exclusion": {
                "se": se_without,
                "mde": mde_without,
                "n_required": n_required_without,
                "years_required": years_without,
            },
            "with_exclusion": {
                "se": se_with,
                "mde": mde_with,
                "n_required": n_required_with,
                "years_required": years_with,
            },
        },
        "gate_PW3": {
            "observed_mean_diff": mean_without,
            "mde": mde_without,
            "fires": pw3_fires,
            "note": (
                "scored on the without-exclusion (original RC-3) series -- see "
                "pre-registration implementation_notes"
            ),
        },
        "placebo": {
            "n_draws": N_DRAWS,
            "seed": SEED,
            "n_excluded_drawn": n_excluded_book_bars,
            "flagged_exclusion_mean_diff": mean_with,
            "placebo_mean_diffs_p95": placebo_p95,
            "placebo_mean_diffs_min": float(placebo_means.min()),
            "placebo_mean_diffs_max": float(placebo_means.max()),
        },
        "gate_PW4": {
            "cap_ok": pw4_cap_ok,
            "placebo_ok": pw4_placebo_ok,
            "fires": pw4_fires,
        },
        "branch": branch,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"PW-1={pw1['fires']} PW-2={pw2_fires} PW-3={pw3_fires} PW-4={pw4_fires}")
    print(f"branch: {branch}")


if __name__ == "__main__":
    main()
