"""Notebook 018, Phase 6: the holdout run -- GATED.

The ONLY file in this repo that names src/research/cache/basis18/holdout/
or reads data past research.HOLDOUT_START (2025-07-01) for this notebook
(grep for "basis18/holdout" or "HOLDOUT_START" to confirm). Runs the
frozen configuration ONCE, if and only if the orchestrator has already
verified Gate FA-2 AND Gate FA-3 both fired on development
(phase_4_18_results.json's own "holdout_access" block, computed and
committed BEFORE this script is ever invoked). No re-tuning, no second
look, no "let me just check one variant" -- this script does not accept
any parameter that could change the strategy definition.

If invoked when access was not granted, it refuses and exits nonzero
without reading anything under basis18/holdout/ -- the check happens
before any holdout path is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats as st

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib18 as bl

import research

PHASE4_RESULTS_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_4_18_results.json"
OUT_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_6_18_results.json"

HOLDOUT_START = research.HOLDOUT_START
HOLDOUT_END = __import__("datetime").datetime(
    2026, 8, 1, tzinfo=__import__("datetime").UTC
)
HOLDOUT_CACHE_DIR = "src/research/cache/basis18/holdout"
HOLDOUT_DOWNLOAD_DIR = "src/research/tmp_dl/basis18/holdout"

N_TRIALS = 18


def _json_default(o: object) -> object:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def check_access() -> bool:
    if not PHASE4_RESULTS_PATH.exists():
        print("REFUSED: phase_4_18_results.json does not exist -- Phase 4 has not run.")
        return False
    with open(PHASE4_RESULTS_PATH) as f:
        phase4 = json.load(f)
    access = phase4.get("holdout_access", {})
    granted = bool(access.get("access_granted", False))
    print(f"Phase 4 holdout_access block: {access}")
    if not granted:
        print(
            "REFUSED: Gate FA-2 and Gate FA-3 did not both fire on development. "
            "Per NEXT_PROMPT sec 8/9.3, this script will not read "
            f"{HOLDOUT_CACHE_DIR} under this condition."
        )
    return granted


def main() -> None:
    if not check_access():
        sys.exit(1)

    symbols = bl.load_universe_seed()
    panel, manifest = bl._load_basis_panel(
        HOLDOUT_CACHE_DIR,
        HOLDOUT_DOWNLOAD_DIR,
        symbols,
        HOLDOUT_START,
        HOLDOUT_END,
        funding_loader=lambda sym: _load_holdout_funding(sym),
    )
    n_ok = sum(1 for v in manifest.values() if v == "ok")
    print(f"Holdout universe: {n_ok}/{len(manifest)} symbols ok")

    featured = bl.add_trade_features(panel)
    annualized_rate = research.sharpe_to_annualized_rate("8h")

    timed_w = bl.build_book_weights(featured, timed=True)
    always_w = bl.build_book_weights(featured, timed=False)

    timed_tf = bl.book_trade_frame(featured, timed_w, origin_offset=0)
    always_tf = bl.book_trade_frame(featured, always_w, origin_offset=0)

    timed_costed = bl.apply_two_leg_costs(timed_tf)
    always_costed = bl.apply_two_leg_costs(always_tf)

    timed_metrics = bl.book_metrics(timed_costed, annualized_rate, "holdout_timed")
    always_metrics = bl.book_metrics(
        always_costed, annualized_rate, "holdout_always_on"
    )

    timed_net = timed_costed["trade_log_return_net"].to_numpy()
    ci_lo, ci_hi = research.block_bootstrap_ci(timed_net)

    basket = research.equal_weight_basket_returns(
        featured, target_col="fwd_perp_return_1"
    ).rename({"trade_log_return": "trade_log_return_basket"})
    btc = featured.filter(pl.col("symbol") == "BTCUSDT").select(
        "datetime", pl.col("fwd_perp_return_1").alias("btc_return")
    )
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

    sharpe_per_period = float(timed_metrics["sharpe_net"]) / annualized_rate
    dsr = research.deflated_sharpe_prob(
        sharpe_per_period,
        n_trials=N_TRIALS,
        n_obs=len(timed_net),
        skew=float(st.skew(timed_net, nan_policy="omit")),
        kurtosis=float(st.kurtosis(timed_net, fisher=False, nan_policy="omit")),
    )

    diff = timed_costed.select(
        "datetime", pl.col("trade_log_return_net").alias("timed_net")
    ).join(
        always_costed.select(
            "datetime", pl.col("trade_log_return_net").alias("always_net")
        ),
        on="datetime",
        how="inner",
    )
    diff_arr = (diff["timed_net"] - diff["always_net"]).to_numpy()
    diff_ci_lo, diff_ci_hi = research.block_bootstrap_ci(diff_arr)

    results = {
        "holdout_window": ["2025-07-01", HOLDOUT_END.strftime("%Y-%m-%d")],
        "universe_ok": n_ok,
        "universe_total": len(manifest),
        "timed": timed_metrics,
        "always_on": always_metrics,
        "timed_net_bootstrap_ci_95": [ci_lo, ci_hi],
        "beta_to_crypto_basket": beta_basket,
        "beta_to_btc": beta_btc,
        "deflated_sharpe_prob": dsr,
        "timed_minus_always_on_bootstrap_ci_95": [diff_ci_lo, diff_ci_hi],
        "spent": True,
        "spent_note": "This holdout is now spent by this run, regardless of what the result shows.",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"Holdout timed net Sharpe: {timed_metrics['sharpe_net']:.3f}")
    print(f"Wrote {OUT_PATH}")


def _load_holdout_funding(symbol: str) -> pl.DataFrame | None:
    path = (
        Path(HOLDOUT_CACHE_DIR)
        / f"{symbol}-funding-{HOLDOUT_START:%Y-%m-%d}-{HOLDOUT_END:%Y-%m-%d}.parquet"
    )
    if not path.exists():
        return None
    return pl.read_parquet(path)


if __name__ == "__main__":
    main()
