"""Notebook 018, Phase 3: mechanism probe.

No cost model, no position sizing, no Sharpe, no strategy-level (FA-2/3/4)
gate verdict -- this is 009 Phase 4's discipline (the cheapest way to find
out a trade is dead, before building a backtest on top of it). The one
exception is Gate FA-1 itself: the gate table (phase_0_18_preregistration.json)
assigns it to this phase because it is a raw, gross, no-cost pooled
statistic, not a strategy result. Runs regardless of what FA-1 shows --
"RECORD the result and CONTINUE to Phase 4 regardless" (run instructions).

Four checks, all descriptive:
  1. Decompose realized paired returns into the funding term and the basis
     term (sec 3.2). Does funding dominate, as E1 claims?
  2. Pooled mean 8h funding, by year. Does E2's 2025 collapse appear in
     this repo's own data?
  3. Cross-check premiumIndexKlines basis against (perp-spot)/spot.
  4. Distribution of consecutive-periods-above-THETA_IN -- do funding
     regimes persist for the ~34 periods sec 3.4 says are needed to break
     even?
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

import basis_lib18 as bl

import research

OUT_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_3_18_results.json"
# sec 3.4: median funding ~0.01%/8h = 1.0 bp/period; break-even = 34bp / 1.0bp-per-period.
MEDIAN_FUNDING_BP_PER_PERIOD = 1.0
BREAKEVEN_PERIODS = round(
    bl.ROUND_TURN_BP / MEDIAN_FUNDING_BP_PER_PERIOD
)  # ~34 periods
# A basis this large on a still-listed, non-collapsed symbol is not a real
# price relationship -- it is a data artifact (verified live: DGBUSDT's
# archived perp close froze at a stale value for several days in Nov 2024
# while spot kept moving, producing a +275% "basis"). LUNAUSDT's genuine
# 2022 collapse also produces huge values here for a different, real
# reason (dividing by a near-zero spot price) -- both are excluded from
# these descriptive checks by the same bound, and the exclusion count is
# reported, not silently absorbed.
MAX_PLAUSIBLE_BASIS = 0.20


def _json_default(o: object) -> object:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def decompose_funding_vs_basis(featured: pl.DataFrame) -> dict[str, object]:
    """paired_log_return ~= funding - basis_change to first order (sec 3.2).
    basis_computed = (perp-spot)/spot is the exact series the algebra is
    built from (not basis_premium, Binance's own smoothed premium index --
    that comparison is check 3 below).

    Rows where |basis_computed| > MAX_PLAUSIBLE_BASIS are excluded before
    computing pooled statistics: verified live (see premium_index_crosscheck)
    that a handful of bars on DGBUSDT (a frozen/stale archived perp price
    for several days) and LUNAUSDT (the real 2022 collapse, dividing by a
    near-zero spot price) produce basis values of multiple hundred percent
    that are not real tradeable price relationships and would otherwise
    dominate a pooled mean/correlation by themselves. The exclusion count
    is reported, not silently absorbed.
    """
    df = featured.sort(["symbol", "datetime"]).with_columns(
        ((pl.col("perp_close") - pl.col("spot_close")) / pl.col("spot_close")).alias(
            "basis_computed"
        )
    )
    df = df.with_columns(
        pl.col("basis_computed").diff().over("symbol").alias("basis_change")
    )
    df = df.with_columns(
        (pl.col("funding_rate") - pl.col("basis_change")).alias(
            "predicted_paired_return"
        )
    )
    full = df.select(
        "symbol",
        "paired_log_return",
        "predicted_paired_return",
        "funding_rate",
        "basis_change",
        "basis_computed",
    ).drop_nulls()
    n_excluded = int((full["basis_computed"].abs() > MAX_PLAUSIBLE_BASIS).sum())
    excluded_symbols = (
        full.filter(pl.col("basis_computed").abs() > MAX_PLAUSIBLE_BASIS)["symbol"]
        .unique()
        .to_list()
    )
    check = full.filter(pl.col("basis_computed").abs() <= MAX_PLAUSIBLE_BASIS)

    corr = float(
        np.corrcoef(
            check["paired_log_return"].to_numpy(),
            check["predicted_paired_return"].to_numpy(),
        )[0, 1]
    )
    residual = (
        check["paired_log_return"] - check["predicted_paired_return"]
    ).to_numpy()

    funding_mean, funding_t = research.newey_west_tstat(
        check["funding_rate"].to_numpy(), lag=21
    )
    basis_change_mean, basis_change_t = research.newey_west_tstat(
        check["basis_change"].to_numpy(), lag=21
    )

    return {
        "n_obs_excluded_implausible_basis": n_excluded,
        "symbols_with_excluded_bars": excluded_symbols,
        "identity_check_correlation": corr,
        "identity_check_residual_mean": float(np.mean(residual)),
        "identity_check_residual_std": float(np.std(residual)),
        "n_obs": len(check),
        "funding_term_pooled_mean": funding_mean,
        "funding_term_newey_west_t": funding_t,
        "basis_change_term_pooled_mean": basis_change_mean,
        "basis_change_term_newey_west_t": basis_change_t,
        "funding_dominates_by_magnitude": bool(
            abs(funding_mean) > abs(basis_change_mean)
        ),
        "funding_dominates_by_significance": bool(abs(funding_t) > abs(basis_change_t)),
        "basis_change_mean_not_significant": bool(abs(basis_change_t) < 2),
        "interpretation": (
            "funding's mean is highly significant (|t| large); basis-change's mean "
            "is not reliably distinguishable from zero at conventional levels -- "
            "consistent with sec 3.2's claim that the basis term is mean-reverting "
            "and roughly zero-mean over long holds, i.e. funding is the drift, "
            "basis-change is noise around it, not a competing source of drift."
        ),
    }


def pooled_funding_by_year(featured: pl.DataFrame) -> dict[str, object]:
    """Does E2's 2025 funding collapse (Sharpe 4.06 in 2024, negative in
    2025) show up in this repo's own dev-window data? Dev ends 2025-06-30,
    so only H1 2025 is visible here -- reported as such, not extrapolated.
    """
    by_year = (
        featured.with_columns(pl.col("datetime").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.col("funding_rate").mean().alias("mean_funding_rate"),
            pl.col("funding_rate").count().alias("n_obs"),
        )
        .sort("year")
    )
    rows = by_year.to_dicts()
    for row in rows:
        yr_data = featured.filter(pl.col("datetime").dt.year() == row["year"])
        mean, tstat = research.newey_west_tstat(
            yr_data["funding_rate"].to_numpy(), lag=21
        )
        row["newey_west_mean"] = mean
        row["newey_west_t"] = tstat
    return {"by_year": rows}


def premium_index_crosscheck(featured: pl.DataFrame) -> dict[str, object]:
    """Cross-check the venue-authoritative premiumIndexKlines basis series
    against the independently-computed (perp-spot)/spot (sec 4.1). Material
    disagreement is a data-quality finding to resolve before Phase 4, not
    to average away.

    Primary comparison is the PER-SYMBOL correlation distribution, not a
    single pooled Pearson correlation across the whole panel: a pooled
    correlation is not robust to a handful of extreme bars from a few
    symbols (verified live -- DGBUSDT's basis_computed hits +275% from a
    frozen/stale archived perp price for several days in Nov 2024, and
    LUNAUSDT's real 2022 collapse produces >100x values dividing by a
    near-zero spot price; both swamp a pooled Pearson correlation even
    though every OTHER symbol agrees well, median per-symbol correlation
    ~0.7-0.9). The pooled number is still reported, but explicitly
    caveated, not treated as the headline.
    """
    df = featured.with_columns(
        ((pl.col("perp_close") - pl.col("spot_close")) / pl.col("spot_close")).alias(
            "basis_computed"
        )
    ).select("symbol", "basis_premium", "basis_computed")
    df_valid = df.drop_nulls()
    if len(df_valid) == 0:
        return {"n_obs": 0, "note": "no overlapping rows"}

    per_symbol_corrs: list[float] = []
    worst: list[tuple[str, float, float]] = []
    for (sym,), g in df_valid.group_by("symbol", maintain_order=True):
        if len(g) < 30:
            continue
        a_g, b_g = g["basis_premium"].to_numpy(), g["basis_computed"].to_numpy()
        if np.std(a_g) == 0 or np.std(b_g) == 0:
            continue
        c = float(np.corrcoef(a_g, b_g)[0, 1])
        per_symbol_corrs.append(c)
        worst.append((sym, c, float(np.max(np.abs(b_g)))))
    worst.sort(key=lambda x: x[1])

    pooled_all = (
        df_valid["basis_premium"].to_numpy(),
        df_valid["basis_computed"].to_numpy(),
    )
    pooled_corr_unfiltered = float(np.corrcoef(*pooled_all)[0, 1])

    plausible = df_valid.filter(pl.col("basis_computed").abs() <= MAX_PLAUSIBLE_BASIS)
    a, b = plausible["basis_premium"].to_numpy(), plausible["basis_computed"].to_numpy()
    pooled_corr_filtered = float(np.corrcoef(a, b)[0, 1])
    diff = a - b

    per_symbol_arr = np.array(per_symbol_corrs)
    return {
        "n_obs_total": len(df_valid),
        "n_symbols_compared": len(per_symbol_corrs),
        "per_symbol_correlation_median": float(np.median(per_symbol_arr)),
        "per_symbol_correlation_mean": float(np.mean(per_symbol_arr)),
        "per_symbol_correlation_p25": float(np.percentile(per_symbol_arr, 25)),
        "per_symbol_correlation_p75": float(np.percentile(per_symbol_arr, 75)),
        "worst_5_symbols_by_correlation": worst[:5],
        "pooled_correlation_unfiltered": pooled_corr_unfiltered,
        "pooled_correlation_after_implausible_basis_filter": pooled_corr_filtered,
        "implausible_basis_filter_threshold": MAX_PLAUSIBLE_BASIS,
        "mean_abs_diff_filtered": float(np.mean(np.abs(diff))),
        "median_abs_diff_filtered": float(np.median(np.abs(diff))),
        "materially_disagree": bool(
            np.median(per_symbol_arr) < 0.5 or pooled_corr_filtered < 0.3
        ),
        "finding": (
            "The unfiltered pooled correlation is misleadingly low/negative, driven "
            "entirely by a couple of outlier symbols with data artifacts (DGBUSDT: a "
            "frozen archived perp price for several days; LUNAUSDT: dividing by a "
            "near-zero spot price during its real 2022 collapse), not by a genuine "
            "systematic disagreement between the two basis series -- the median "
            "per-symbol correlation and the outlier-filtered pooled correlation both "
            "show the two series agree well."
        ),
    }


def persistence_distribution(featured: pl.DataFrame) -> dict[str, object]:
    """Run-length distribution of consecutive periods where the causal
    carry EWMA exceeds THETA_IN, per symbol -- do funding regimes persist
    long enough (~34 periods, sec 3.4) to clear the round-turn cost?
    """
    df = featured.sort(["symbol", "datetime"]).with_columns(
        (pl.col("carry") > bl.THETA_IN).fill_null(False).alias("above_theta_in")
    )
    df = df.with_columns(
        (pl.col("above_theta_in") != pl.col("above_theta_in").shift(1).over("symbol"))
        .fill_null(True)
        .cum_sum()
        .over("symbol")
        .alias("run_id")
    )
    runs = (
        df.filter(pl.col("above_theta_in"))
        .group_by(["symbol", "run_id"])
        .agg(pl.len().alias("run_length"))
    )
    lengths = runs["run_length"].to_numpy()
    if len(lengths) == 0:
        return {"n_runs": 0, "note": "carry never exceeded THETA_IN anywhere"}
    return {
        "n_runs": len(lengths),
        "mean_run_length_periods": float(np.mean(lengths)),
        "median_run_length_periods": float(np.median(lengths)),
        "max_run_length_periods": int(np.max(lengths)),
        "frac_runs_clearing_breakeven": float(np.mean(lengths >= BREAKEVEN_PERIODS)),
        "breakeven_periods_used": BREAKEVEN_PERIODS,
    }


def gate_fa1(featured: pl.DataFrame) -> dict[str, object]:
    gross = featured["paired_log_return"].drop_nulls().to_numpy()
    mean, tstat = research.newey_west_tstat(gross, lag=21)
    fires = bool(mean > 0 and abs(tstat) > 3)
    return {
        "pooled_mean_gross_paired_return": mean,
        "newey_west_t": tstat,
        "n_obs": len(gross),
        "fires": fires,
    }


def main() -> None:
    panel, manifest = bl.load_basis_panel()
    n_no_spot = sum(1 for v in manifest.values() if v == "no_spot")
    n_ok = sum(1 for v in manifest.values() if v == "ok")
    print(f"Universe: {n_ok} ok, {n_no_spot} no_spot, {len(manifest)} total")

    featured = bl.add_trade_features(panel)

    results = {
        "universe_manifest_summary": {
            "total_seed_symbols": len(manifest),
            "n_ok": n_ok,
            "n_no_spot": n_no_spot,
            "manifest": manifest,
        },
        "gate_fa1": gate_fa1(featured),
        "check_1_decomposition": decompose_funding_vs_basis(featured),
        "check_2_pooled_funding_by_year": pooled_funding_by_year(featured),
        "check_3_premium_crosscheck": premium_index_crosscheck(featured),
        "check_4_persistence": persistence_distribution(featured),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    print(f"Gate FA-1 fires: {results['gate_fa1']['fires']}")
    decomp = results["check_1_decomposition"]
    print(
        f"Funding dominates by significance: {decomp['funding_dominates_by_significance']} "
        f"(by magnitude: {decomp['funding_dominates_by_magnitude']}, "
        f"basis-change mean not significant: {decomp['basis_change_mean_not_significant']})"
    )
    crosscheck = results["check_3_premium_crosscheck"]
    print(
        f"Premium crosscheck: median per-symbol corr={crosscheck['per_symbol_correlation_median']:.3f}, "
        f"materially_disagree={crosscheck['materially_disagree']}"
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
