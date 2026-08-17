"""Notebook 018, Phase 4: the backtest -- timed / always-on / cash books,
origin offsets [0,1,2,3], sec 3.4 costs. Computes Gates FA-2, FA-3, FA-4,
and FUND.

n_trials for DSR is the pre-registered 18-baseline (phase_0_18_preregistration.json
sec "n_trials"), counted honestly upward if Phase 5 adds configurations --
never downward, and declared before this script was written, not chosen
after seeing a Sharpe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.stats as st

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib18 as bl

import research

OUT_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_4_18_results.json"
ORIGIN_OFFSETS = [0, 1, 2, 3]
N_TRIALS = (
    18  # phase_0_18_preregistration.json, revised upward only if Phase 5 adds more
)
BENCHMARK_BETA_BOUND = 0.10
FA2_SHARPE_BOUND = 0.5
DSR_BOUND = 0.95


def _json_default(o: object) -> object:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def trim_to_offset(
    frame: pl.DataFrame, all_times: pl.Series, offset: int
) -> pl.DataFrame:
    if offset == 0 or offset >= len(all_times):
        return frame
    cutoff = all_times[offset]
    return frame.filter(pl.col("datetime") >= cutoff)


def concentration_diagnostic(
    weights: pl.DataFrame, costed: pl.DataFrame
) -> dict[str, Any]:
    """How many symbols the timed book actually holds per bar, and which
    symbols were behind the worst single-bar net returns.

    Investigated because the timed book's DSR came back with extreme
    sample skew/kurtosis; tracing it found the book is sometimes 100% in
    ONE symbol (equal weight among however many currently qualify, with no
    diversification floor), and the worst bars (ICPUSDT 2022-06-25,
    MATICUSDT 2024-09-04..09) coincide with the short perp leg's own
    volume dropping to exactly zero for several consecutive days (verified
    directly against the cached kline data: open=high=low=close, volume=0)
    -- a real perp-market liquidity collapse the 30-day trailing MEDIAN
    liquidity screen is slow to catch (it does catch it, one bar after the
    worst loss in ICPUSDT's case), during which the short leg cannot
    actually be hedged because there is no real perp price discovery. This
    is a structural property of a median-based liquidity screen and an
    equal-weight-with-no-floor book, not a code bug -- reported here, not
    silently patched by changing the frozen liquidity floor or hysteresis
    parameters (sec 12).
    """
    held = weights.filter(pl.col("weight") > 0)
    n_held = held.group_by("datetime").agg(pl.len().alias("n_symbols_held"))
    all_bars = weights.select("datetime").unique()
    n_held = all_bars.join(n_held, on="datetime", how="left").with_columns(
        pl.col("n_symbols_held").fill_null(0)
    )
    counts = n_held["n_symbols_held"].to_numpy()

    worst = costed.sort("trade_log_return_net").head(5)
    worst_detail = []
    for row in worst.iter_rows(named=True):
        held_syms = weights.filter(
            (pl.col("datetime") == row["datetime"]) & (pl.col("weight") > 0)
        )["symbol"].to_list()
        worst_detail.append(
            {
                "datetime": str(row["datetime"]),
                "trade_log_return_net": row["trade_log_return_net"],
                "symbols_held_at_decision": held_syms,
            }
        )

    return {
        "n_symbols_held_mean": float(np.mean(counts)),
        "n_symbols_held_median": float(np.median(counts)),
        "frac_bars_single_symbol": float(np.mean(counts == 1)),
        "frac_bars_zero_symbols": float(np.mean(counts == 0)),
        "frac_bars_at_cap": float(np.mean(counts == bl.MAX_POSITIONS)),
        "worst_5_bars": worst_detail,
    }


def main() -> None:
    panel, manifest = bl.load_basis_panel()
    n_ok = sum(1 for v in manifest.values() if v == "ok")
    print(f"Universe: {n_ok}/{len(manifest)} symbols ok")

    featured = bl.add_trade_features(panel)
    annualized_rate = research.sharpe_to_annualized_rate("8h")
    periods_per_year = annualized_rate**2

    timed_w = bl.build_book_weights(featured, timed=True)
    always_w = bl.build_book_weights(featured, timed=False)

    all_times = featured["datetime"].unique(maintain_order=True).sort()
    cash_frame_full = pl.DataFrame(
        {
            "datetime": all_times,
            "trade_log_return": [0.0] * len(all_times),
            "turnover": [0.0] * len(all_times),
        }
    )

    basket = research.equal_weight_basket_returns(
        featured, target_col="fwd_perp_return_1"
    ).rename({"trade_log_return": "trade_log_return_basket"})
    btc = featured.filter(pl.col("symbol") == "BTCUSDT").select(
        "datetime", pl.col("fwd_perp_return_1").alias("btc_return")
    )

    by_offset: dict[int, dict[str, Any]] = {}
    timed_sharpes: list[float] = []
    timed_returns_all_offsets: list[np.ndarray] = []

    for offset in ORIGIN_OFFSETS:
        timed_tf = bl.book_trade_frame(featured, timed_w, offset)
        always_tf = bl.book_trade_frame(featured, always_w, offset)
        cash_tf = trim_to_offset(cash_frame_full, all_times, offset)

        timed_costed = bl.apply_two_leg_costs(timed_tf)
        always_costed = bl.apply_two_leg_costs(always_tf)

        timed_metrics = bl.book_metrics(timed_costed, annualized_rate, "timed")
        always_metrics = bl.book_metrics(always_costed, annualized_rate, "always_on")
        cash_metrics = research._series_metrics(
            cash_tf["trade_log_return"], annualized_rate, "cash"
        )

        if offset == 0:
            concentration = concentration_diagnostic(timed_w, timed_costed)

        timed_net = timed_costed["trade_log_return_net"].to_numpy()
        timed_ci_lo, timed_ci_hi = research.block_bootstrap_ci(timed_net)
        fa2_ci_excludes_zero = bool(
            timed_ci_lo > 0
        )  # cash is always 0, so "excess over cash" == the net return itself

        joined = timed_costed.select(
            "datetime", pl.col("trade_log_return_net").alias("timed_net")
        ).join(
            always_costed.select(
                "datetime", pl.col("trade_log_return_net").alias("always_net")
            ),
            on="datetime",
            how="inner",
        )
        diff = (joined["timed_net"] - joined["always_net"]).to_numpy()
        diff_ci_lo, diff_ci_hi = research.block_bootstrap_ci(diff)
        fa3_this_offset = bool(diff_ci_lo > 0)  # excludes zero, in favour of timed

        beta_frame = (
            timed_costed.select("datetime", "trade_log_return_net")
            .join(basket, on="datetime", how="inner")
            .join(btc, on="datetime", how="inner")
        )
        beta_basket = bl.ols_beta(
            beta_frame["trade_log_return_net"].to_numpy(),
            beta_frame["trade_log_return_basket"].to_numpy(),
        )
        beta_btc = bl.ols_beta(
            beta_frame["trade_log_return_net"].to_numpy(),
            beta_frame["btc_return"].to_numpy(),
        )
        fa4_this_offset = bool(
            np.isfinite(beta_basket)
            and np.isfinite(beta_btc)
            and abs(beta_basket) < BENCHMARK_BETA_BOUND
            and abs(beta_btc) < BENCHMARK_BETA_BOUND
        )

        timed_sharpes.append(float(timed_metrics["sharpe_net"]))
        timed_returns_all_offsets.append(timed_net)

        by_offset[offset] = {
            "timed": timed_metrics,
            "always_on": always_metrics,
            "cash": cash_metrics,
            "timed_net_bootstrap_ci_95": [timed_ci_lo, timed_ci_hi],
            "fa2_ci_excludes_zero_this_offset": fa2_ci_excludes_zero,
            "timed_minus_always_on_bootstrap_ci_95": [diff_ci_lo, diff_ci_hi],
            "fa3_fires_this_offset": fa3_this_offset,
            "beta_to_crypto_basket": beta_basket,
            "beta_to_btc": beta_btc,
            "fa4_fires_this_offset": fa4_this_offset,
        }
        print(
            f"offset={offset}: timed net Sharpe={timed_metrics['sharpe_net']:.3f} "
            f"always-on net Sharpe={always_metrics['sharpe_net']:.3f} "
            f"beta_basket={beta_basket:.4f} beta_btc={beta_btc:.4f}"
        )

    best_idx = int(np.argmax(timed_sharpes))
    best_offset = ORIGIN_OFFSETS[best_idx]
    best_returns = timed_returns_all_offsets[best_idx]
    best_sharpe_annualized = timed_sharpes[best_idx]
    best_sharpe_per_period = best_sharpe_annualized / annualized_rate
    sample_skew = float(st.skew(best_returns, nan_policy="omit"))
    sample_kurtosis = float(st.kurtosis(best_returns, fisher=False, nan_policy="omit"))
    dsr = research.deflated_sharpe_prob(
        best_sharpe_per_period,
        n_trials=N_TRIALS,
        n_obs=len(best_returns),
        skew=sample_skew,
        kurtosis=sample_kurtosis,
    )

    fa2_sharpe_leg = all(s > FA2_SHARPE_BOUND for s in timed_sharpes)
    fa2_ci_leg = all(
        by_offset[o]["fa2_ci_excludes_zero_this_offset"] for o in ORIGIN_OFFSETS
    )
    fa2_dsr_leg = bool(dsr > DSR_BOUND)
    fa2_fires = bool(fa2_sharpe_leg and fa2_ci_leg and fa2_dsr_leg)

    fa3_fires = all(by_offset[o]["fa3_fires_this_offset"] for o in ORIGIN_OFFSETS)
    fa4_fires = all(by_offset[o]["fa4_fires_this_offset"] for o in ORIGIN_OFFSETS)

    max_drawdowns_net = [
        by_offset[o]["timed"]["max_drawdown_net"] for o in ORIGIN_OFFSETS
    ]
    fund_fires = bool(
        fa2_sharpe_leg and fa2_dsr_leg
    )  # sec 8: Sharpe>0.5 every offset AND DSR>0.95 AND a stated bounded MDD (reported, not gated further)
    holdout_access_granted = bool(fa2_fires and fa3_fires)

    results = {
        "annualized_rate": annualized_rate,
        "periods_per_year": periods_per_year,
        "n_trials_used": N_TRIALS,
        "by_offset": by_offset,
        "concentration_diagnostic": concentration,
        "dsr": {
            "best_offset": best_offset,
            "best_sharpe_annualized": best_sharpe_annualized,
            "best_sharpe_per_period": best_sharpe_per_period,
            "sample_skew": sample_skew,
            "sample_kurtosis": sample_kurtosis,
            "n_obs": len(best_returns),
            "deflated_sharpe_prob": dsr,
            "known_caveat": (
                "research.deflated_sharpe_prob is documented (NEXT_PROMPT sec 11.4) as "
                "likely over-rejecting for a trial family of near-identical origin offsets. "
                "Used unmodified per pre-registration; fixing it is notebook 017."
            ),
        },
        "gates": {
            "FA-2": {
                "sharpe_leg_fires": fa2_sharpe_leg,
                "bootstrap_ci_leg_fires": fa2_ci_leg,
                "dsr_leg_fires": fa2_dsr_leg,
                "fires": fa2_fires,
                "fires_except_dsr_leg": bool(
                    fa2_sharpe_leg and fa2_ci_leg and not fa2_dsr_leg
                ),
            },
            "FA-3": {"fires": fa3_fires},
            "FA-4": {"fires": fa4_fires},
            "FUND": {
                "fires": fund_fires,
                "max_drawdowns_net_by_offset": dict(
                    zip(ORIGIN_OFFSETS, max_drawdowns_net, strict=True)
                ),
            },
        },
        "holdout_access": {
            "rule": "requires FA-2 AND FA-3",
            "fa2_fires": fa2_fires,
            "fa3_fires": fa3_fires,
            "access_granted": holdout_access_granted,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    print(
        f"FA-2 fires: {fa2_fires} (sharpe={fa2_sharpe_leg} ci={fa2_ci_leg} dsr={fa2_dsr_leg}, dsr_value={dsr:.4f})"
    )
    print(f"FA-3 fires: {fa3_fires}")
    print(f"FA-4 fires: {fa4_fires}")
    print(f"Holdout access granted: {holdout_access_granted}")
    print(
        f"Concentration: median symbols held={concentration['n_symbols_held_median']}, "
        f"frac single-symbol bars={concentration['frac_bars_single_symbol']:.3f}"
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
