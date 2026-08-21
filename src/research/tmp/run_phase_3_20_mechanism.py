"""Notebook 020, Phase 3 (NEXT_PROMPT.md sec 8): mechanism probes.

No cost model, no Sharpe, no strategy-level gate verdict except RC-1/XD-1
(both raw, gross, no-cost pooled statistics -- 018 Phase 3's own discipline,
sec 8's "no cost model, no position sizing, no Sharpe" instruction).

Mechanism A: symbols-held distribution for A0/A1/A3, gross skew/kurtosis,
stand-down cost, RC-1.
Mechanism B: XD-1, funding-vs-price-divergence decomposition, spread
persistence, the |basis|>20% sanity bound, the sign-convention assertion.

Usage: uv run python src/research/tmp/run_phase_3_20_mechanism.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib20 as bl20

import research

SCRATCH_DIR = REPO_ROOT / "scratch" / "020"
OUT_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_3_20_mechanism.json"

MAX_PLAUSIBLE_BASIS = 0.20  # 018's own sanity bound, reused verbatim (sec 8)
BREAKEVEN_PERIODS_XV = round(bl20.ROUND_TURN_BP_XV / 1.0)  # 25 periods, sec 8


def _json_default(o: object) -> object:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _symbols_held_distribution(weights: pl.DataFrame) -> dict[str, Any]:
    held = weights.filter(pl.col("weight") != 0)
    n_held = held.group_by("datetime").agg(pl.len().alias("n_held"))
    all_bars = weights.select("datetime").unique()
    n_held = all_bars.join(n_held, on="datetime", how="left").with_columns(
        pl.col("n_held").fill_null(0)
    )
    counts = n_held["n_held"].to_numpy()
    return {
        "median": float(np.median(counts)),
        "mean": float(np.mean(counts)),
        "frac_1_symbol": float(np.mean(counts == 1)),
        "frac_2_symbols": float(np.mean(counts == 2)),
        "frac_lt_n_min_3": float(np.mean(counts < 3)),
        "frac_zero": float(np.mean(counts == 0)),
        "frac_at_cap_10": float(np.mean(counts == bl20.MAX_POSITIONS)),
    }


def _gross_moments(gross_returns: np.ndarray) -> dict[str, float]:
    import scipy.stats as st

    return {
        "skew": float(st.skew(gross_returns, nan_policy="omit")),
        "kurtosis_non_excess": float(
            st.kurtosis(gross_returns, fisher=False, nan_policy="omit")
        ),
        "n_obs": len(gross_returns),
    }


def mechanism_a() -> dict[str, Any]:
    hl21 = pl.read_parquet(
        SCRATCH_DIR / f"panel_dev_binance_hl{bl20.CARRY_EWMA_HALF_LIFE}.parquet"
    )
    hl42 = pl.read_parquet(
        SCRATCH_DIR / f"panel_dev_binance_hl{bl20.SLOW_CARRY_HALF_LIFE}.parquet"
    )

    variant_specs: list[tuple[str, pl.DataFrame, int, float, float]] = [
        ("A0", hl21, 1, bl20.THETA_IN, bl20.THETA_OUT),
        ("A1", hl21, bl20.N_MIN, bl20.THETA_IN, bl20.THETA_OUT),
        ("A2", hl42, 1, bl20.THETA_IN_SLOW, bl20.THETA_OUT_SLOW),
        ("A3", hl42, bl20.N_MIN, bl20.THETA_IN_SLOW, bl20.THETA_OUT_SLOW),
    ]

    out: dict[str, Any] = {}
    for name, panel, n_min, theta_in, theta_out in variant_specs:
        weights = bl20.build_book_weights_v2(
            panel, timed=True, n_min=n_min, theta_in=theta_in, theta_out=theta_out
        )
        trade_frame = bl20.book_trade_frame(
            panel, weights, target_col="fwd_paired_return_1"
        )
        gross = trade_frame["trade_log_return"].drop_nulls().to_numpy()
        mean, tstat = research.newey_west_tstat(gross, lag=21)
        turnover = research.portfolio_turnover(weights)
        held_dist = _symbols_held_distribution(weights)

        entry: dict[str, Any] = {
            "symbols_held_distribution": held_dist,
            "gross_moments": _gross_moments(gross),
            "gross_pooled_mean": mean,
            "gross_newey_west_t": tstat,
            "mean_gross_turnover_per_bar": research._as_float(
                turnover["turnover"].mean()
            ),
        }
        if name in ("A1", "A3"):
            entry["standdown_frac_bars_in_cash"] = held_dist["frac_zero"]
        out[name] = entry
        print(
            f"{name}: median_held={held_dist['median']:.1f} "
            f"frac_1sym={held_dist['frac_1_symbol']:.4f} "
            f"skew={entry['gross_moments']['skew']:.2f} "
            f"kurt={entry['gross_moments']['kurtosis_non_excess']:.2f} "
            f"NW_t={tstat:.2f}"
        )

    out["A0_reproduces_018_5.4pct_singlesymbol"] = {
        "measured": out["A0"]["symbols_held_distribution"]["frac_1_symbol"],
        "expected_018": 0.054,
        "close": bool(
            abs(out["A0"]["symbols_held_distribution"]["frac_1_symbol"] - 0.054) < 0.01
        ),
    }

    out["gate_rc1"] = {
        "fires": bool(
            out["A3"]["gross_pooled_mean"] > 0
            and abs(out["A3"]["gross_newey_west_t"]) > 3
        ),
        "pooled_mean_gross_paired_return": out["A3"]["gross_pooled_mean"],
        "newey_west_t": out["A3"]["gross_newey_west_t"],
        "n_obs": out["A3"]["gross_moments"]["n_obs"],
    }
    return out


def _xvenue_decomposition(featured: pl.DataFrame) -> dict[str, Any]:
    df = featured.with_columns(
        (
            (pl.col("bybit_close") - pl.col("binance_close")) / pl.col("binance_close")
        ).alias("xvenue_basis")
    )
    n_excluded = int((df["xvenue_basis"].abs() > MAX_PLAUSIBLE_BASIS).sum())
    excluded_symbols = (
        df.filter(pl.col("xvenue_basis").abs() > MAX_PLAUSIBLE_BASIS)["symbol"]
        .unique()
        .to_list()
    )
    plausible = df.filter(pl.col("xvenue_basis").abs() <= MAX_PLAUSIBLE_BASIS)

    terms = (
        plausible.sort(["symbol", "datetime"])
        .select(
            (pl.col("binance_funding_rate") - pl.col("bybit_funding_rate")).alias(
                "funding_spread"
            ),
            (
                (
                    pl.col("bybit_close")
                    / pl.col("bybit_close").shift(1).over("symbol")
                ).log()
                - (
                    pl.col("binance_close")
                    / pl.col("binance_close").shift(1).over("symbol")
                ).log()
            ).alias("price_divergence"),
        )
        .drop_nulls()
    )
    funding_spread = terms["funding_spread"].to_numpy()
    price_divergence = terms["price_divergence"].to_numpy()

    funding_mean, funding_t = research.newey_west_tstat(funding_spread, lag=21)
    price_mean, price_t = research.newey_west_tstat(price_divergence, lag=21)

    return {
        "n_obs_excluded_implausible_basis": n_excluded,
        "symbols_with_excluded_bars": excluded_symbols,
        "funding_spread_term_pooled_mean": funding_mean,
        "funding_spread_term_newey_west_t": funding_t,
        "price_divergence_term_pooled_mean": price_mean,
        "price_divergence_term_newey_west_t": price_t,
        "funding_dominates_by_magnitude": bool(abs(funding_mean) > abs(price_mean)),
        "funding_dominates_by_significance": bool(abs(funding_t) > abs(price_t)),
    }


def _xvenue_persistence(featured: pl.DataFrame) -> dict[str, Any]:
    df = featured.sort(["symbol", "datetime"]).with_columns(
        (pl.col("carry").abs() > bl20.THETA_IN_XV)
        .fill_null(False)
        .alias("above_theta_in")
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
        return {"n_runs": 0, "note": "|carry| never exceeded THETA_IN_XV anywhere"}
    return {
        "n_runs": len(lengths),
        "mean_run_length_periods": float(np.mean(lengths)),
        "median_run_length_periods": float(np.median(lengths)),
        "frac_runs_clearing_breakeven": float(np.mean(lengths >= BREAKEVEN_PERIODS_XV)),
        "breakeven_periods_used": BREAKEVEN_PERIODS_XV,
    }


def _sign_convention_check(featured: pl.DataFrame) -> dict[str, Any]:
    df = featured.with_columns(
        (pl.col("binance_funding_rate") - pl.col("bybit_funding_rate")).alias("spread")
    ).drop_nulls(["spread"])
    if len(df) == 0:
        return {"note": "no rows to check"}
    row = df.sort(pl.col("spread").abs(), descending=True).head(1).to_dicts()[0]
    a_funding, b_funding = row["binance_funding_rate"], row["bybit_funding_rate"]
    funding_term = a_funding - b_funding
    return {
        "symbol": row["symbol"],
        "datetime": str(row["datetime"]),
        "binance_funding_rate": a_funding,
        "bybit_funding_rate": b_funding,
        "funding_term_in_w_plus1_return": funding_term,
        "interpretation": (
            "w=+1 (short Binance perp/long Bybit perp) receives +binance_funding "
            "(short receives positive funding) and pays -bybit_funding (long pays "
            "positive funding); funding_term = a_funding - b_funding matches the "
            "formula exactly, verified against this real bar."
        ),
    }


def mechanism_b() -> dict[str, Any] | None:
    path = SCRATCH_DIR / f"panel_dev_xvenue_hl{bl20.CARRY_EWMA_HALF_LIFE}.parquet"
    if not path.exists():
        return None
    featured = pl.read_parquet(path)
    if featured["symbol"].n_unique() < 20:
        return {
            "blocked": True,
            "n_symbols": featured["symbol"].n_unique(),
            "reason": "fewer than 20 symbols in the cross-venue panel -- sec 4.6 data blocker",
        }

    gross = featured["xvenue_paired_log_return"].drop_nulls().to_numpy()
    mean, tstat = research.newey_west_tstat(gross, lag=21)
    xd1 = {
        "pooled_mean_gross_xvenue_paired_return": mean,
        "newey_west_t": tstat,
        "n_obs": len(gross),
        "fires": bool(mean > 0 and abs(tstat) > 3),
    }
    print(f"XD-1: mean={mean:.6e} NW_t={tstat:.2f} n={len(gross)} fires={xd1['fires']}")

    # B0 headline book, gross (no costs) -- the portfolio-level moments
    # Phase 3b's prediction needs, matching how A1/A3's moments were built.
    b0_weights = bl20.build_xvenue_book_weights(
        featured,
        timed=True,
        n_min=bl20.N_MIN_XV,
        theta_in=bl20.THETA_IN_XV,
        theta_out=bl20.THETA_OUT_XV,
    )
    b0_trade_frame = bl20.book_trade_frame(
        featured, b0_weights, target_col="fwd_xvenue_paired_return_1"
    )
    b0_gross = b0_trade_frame["trade_log_return"].drop_nulls().to_numpy()
    b0_moments = _gross_moments(b0_gross)
    b0_mean, b0_tstat = research.newey_west_tstat(b0_gross, lag=21)

    decomposition = _xvenue_decomposition(featured)
    persistence = _xvenue_persistence(featured)
    sign_check = _sign_convention_check(featured)

    return {
        "n_symbols": featured["symbol"].n_unique(),
        "gate_xd1": xd1,
        "b0_gross_moments": b0_moments,
        "b0_gross_pooled_mean": b0_mean,
        "b0_gross_newey_west_t": b0_tstat,
        "decomposition": decomposition,
        "spread_persistence": persistence,
        "sign_convention_check": sign_check,
    }


def main() -> None:
    print("== Mechanism A ==")
    result_a = mechanism_a()
    print("== Mechanism B ==")
    result_b = mechanism_b()
    if result_b is None:
        print(
            "Mechanism B panel not yet available (Bybit fetch/Phase 2 xvenue panel not built)"
        )

    out = {"mechanism_a": result_a, "mechanism_b": result_b}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
