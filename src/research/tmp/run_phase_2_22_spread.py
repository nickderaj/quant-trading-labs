"""Notebook 022, Phase 2: build the Hyperliquid/Binance cross-venue panel,
score HD-1 (gross spread exists) and HD-3 (adequately powered), and
reproduce the planning-time probes (P1 per-symbol spread stats, P2's BTC
price-divergence stats) on the full pre-registered dev window and mapped
universe -- not just the two symbols the probe looked at.

Usage: uv run python src/research/tmp/run_phase_2_22_spread.py
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

import basis_lib22 as bl22

import research

TMP = REPO_ROOT / "src" / "research" / "tmp"
OUT_PATH = TMP / "phase_2_22_results.json"


def main() -> None:
    symbols = bl22.load_frozen_feed_screened_symbols()
    panel, manifest = bl22.load_hlvenue_panel(symbols=symbols)
    panel = bl22.add_hlvenue_trade_features(panel)

    n_ok = sum(1 for v in manifest.values() if v == "ok")
    print(f"panel assembled: {n_ok}/{len(manifest)} symbols joined ok")

    liquid = panel.filter(pl.col("liquid"))
    gross = liquid["hlvenue_paired_log_return"].drop_nulls().to_numpy()
    mean, tstat = research.newey_west_tstat(gross, lag=bl22.CARRY_EWMA_HALF_LIFE)
    ann_rate = research.sharpe_to_annualized_rate(bl22.INTERVAL)
    gross_mean_ann = mean * (ann_rate**2)  # periods_per_year == ann_rate**2
    gross_sharpe_ann = (
        (mean / gross.std(ddof=1)) * ann_rate if gross.std(ddof=1) > 0 else float("nan")
    )
    hd1_fires = bool(mean > 0 and abs(tstat) > 3)

    # HD-3: power, computed on this same pooled gross series, before Phase 4 runs.
    n_obs = len(gross)
    mde_ann_sharpe = bl22.mde_annualized_sharpe(n_obs, ann_rate)
    hd3_fires = bool(gross_sharpe_ann > mde_ann_sharpe)

    # Per-symbol spread table (reproduces Probe P1's shape on the full universe).
    per_symbol = {}
    for (sym,), grp in liquid.group_by("symbol", maintain_order=True):
        r = grp["hlvenue_paired_log_return"].drop_nulls().to_numpy()
        if len(r) < 30:
            continue
        m, t = research.newey_west_tstat(r, lag=bl22.CARRY_EWMA_HALF_LIFE)
        per_symbol[sym] = {
            "n_bars": len(r),
            "mean_ann": float(m * (ann_rate**2)),
            "median_ann": float(np.median(r) * (ann_rate**2)),
            "frac_positive": float((r > 0).mean()),
            "newey_west_t": float(t),
        }

    # BTC price-divergence stats (reproduces Probe P2 on the full-window join).
    btc = panel.filter(pl.col("symbol") == "BTCUSDT").sort("datetime")
    price_div = None
    if len(btc) > 30:
        log_gap = (btc["hl_close"] / btc["binance_close"]).log().to_numpy() * 1e4  # bp
        change = np.diff(log_gap)
        price_div = {
            "n_bars": len(btc),
            "mean_log_gap_bp": float(np.mean(log_gap)),
            "sd_log_gap_bp": float(np.std(log_gap, ddof=1)),
            "sd_of_change_bp": float(np.std(change, ddof=1)),
            "sd_of_change_ann_pct": float(
                np.std(change, ddof=1) * 1e-4 * ann_rate * 100
            ),
            "p99_abs_change_bp": float(np.percentile(np.abs(change), 99)),
            "implied_gross_sharpe": float("nan"),
        }
        # implied always-on gross Sharpe on BTC alone: the symbol's own gross
        # hlvenue_paired_log_return series, not the price-gap change series
        # directly (the gap's own drift is ~0 by construction -- the tradeable
        # quantity is the funding-carried spread return computed above).
        btc_r = btc["hlvenue_paired_log_return"].drop_nulls().to_numpy()
        price_div["implied_gross_sharpe"] = float(
            (btc_r.mean() / btc_r.std(ddof=1)) * ann_rate
            if btc_r.std(ddof=1) > 0
            else np.nan
        )

    results = {
        "n_symbols_joined": n_ok,
        "n_symbols_liquid_scored": len(per_symbol),
        "join_manifest": manifest,
        "gate_HD1": {
            "fires": hd1_fires,
            "pooled_mean_gross_per_bar": float(mean),
            "pooled_mean_gross_annualized": float(gross_mean_ann),
            "pooled_gross_sharpe_annualized": float(gross_sharpe_ann),
            "newey_west_t": float(tstat),
            "n_bars": n_obs,
        },
        "gate_HD3": {
            "fires": hd3_fires,
            "n_obs": n_obs,
            "mde_annualized_sharpe": mde_ann_sharpe,
            "observed_gross_sharpe_annualized": gross_sharpe_ann,
        },
        "per_symbol_spread": per_symbol,
        "btc_price_divergence": price_div,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(
        f"HD-1 fires={hd1_fires} (t={tstat:.2f}); HD-3 fires={hd3_fires} "
        f"(sharpe={gross_sharpe_ann:.2f} vs mde={mde_ann_sharpe:.2f})"
    )
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
