"""Notebook 022, Phase 2b (unplanned -- added after the tripwire below).

Phase 4's first run of HL_ALWAYSON produced an implausible headline: net
Sharpe 4.29 on a 2-year book, kurtosis (non-excess) 80.4, skew 2.08. Per the
repo's standing tripwire discipline ("an implausible number gets
investigated, not reported and not silently patched"), the worst/best bars
were traced to a single week (2025-01-06..11) dominated by FTMUSDT, whose
BINANCE perp feed turned out to be frozen (Fantom's Sonic migration) for
524/2191 dev-window bars (24%) while Hyperliquid's own FTM kept trading --
so the "spread" on those bars was a real, moving HL price against a stale,
non-updating Binance mark, not a captured funding spread. This is exactly
018/021's own documented frozen-feed signature (open==high==low==close AND
volume==0 on the perp leg), reused here unmodified via power_lib21
(import-only, per scope discipline).

Checking every one of the 50 mapped symbols, not just FTM, found the same
problem is far more widespread than in 018/020/021's own books: 19/50
symbols have SOME frozen-feed contamination on their Binance leg within
this notebook's dev window, and all but one (MATICUSDT, 0.96%) exceed 10% --
several (RAYUSDT, SCUSDT, FTTUSDT) are 100% frozen for the entire window,
i.e. their Binance futures market was effectively dead throughout.

018/020's own threshold-gated, low-turnover books are naturally less
exposed to this: a frozen, unchanging funding rate rarely crosses their
entry threshold, so a dead symbol mostly just sits out. HL_ALWAYSON has no
such filter by design (every liquid symbol is held every bar) -- so it is
maximally exposed to exactly this failure mode. That is a genuine,
reportable characteristic of the always-on construction, not an artefact of
this screen.

The fix, mechanical and disclosed, not tuned to the outcome: any symbol
whose Binance perp leg is frozen-feed-flagged on MORE than 5% of its own
dev-window bars is excluded from the universe passed to load_hlvenue_panel
for every remaining phase. 5% is a round-number bright line comfortably
above MATIC's 0.96% (kept) and comfortably below every other flagged
symbol's contamination (18 excluded, all >10%). This was decided before
looking at whether exclusion helps or hurts the headline -- see
phase_2b_22_frozen_feed_results.json for the exact list and both the
contaminated and corrected book numbers, reported side by side.

Usage: uv run python src/research/tmp/run_phase_2b_22_frozen_feed_screen.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib22 as bl22
import power_lib21 as pw21

OUT_PATH = REPO_ROOT / "scratch" / "022" / "frozen_feed_exclusions.json"
PERP_GLOB = str(
    REPO_ROOT
    / "src"
    / "research"
    / "cache"
    / "basis18"
    / "dev"
    / "*-perp-8h-2023-07-01-2025-06-30.parquet"
)
FROZEN_FRACTION_THRESHOLD = 0.05
DEV_BARS = 2191  # 2023-07-01 00:00 .. 2025-06-30 00:00, 8h grid, inclusive


def main() -> None:
    universe = sorted(bl22.load_hlvenue_universe())
    catalogue = pw21.flag_frozen_feed_bars(PERP_GLOB)
    counts = (
        catalogue.filter(pl.col("symbol").is_in(universe))
        .group_by("symbol")
        .agg(pl.len().alias("n_flagged"))
        .with_columns((pl.col("n_flagged") / DEV_BARS).alias("frozen_fraction"))
        .sort("frozen_fraction", descending=True)
    )
    excluded = counts.filter(pl.col("frozen_fraction") > FROZEN_FRACTION_THRESHOLD)[
        "symbol"
    ].to_list()
    kept = [s for s in universe if s not in excluded]

    out = {
        "threshold": FROZEN_FRACTION_THRESHOLD,
        "n_mapped_universe": len(universe),
        "n_excluded": len(excluded),
        "n_kept": len(kept),
        "excluded_symbols": excluded,
        "kept_symbols": kept,
        "per_symbol_frozen_fraction": {
            row["symbol"]: row["frozen_fraction"]
            for row in counts.iter_rows(named=True)
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"excluded {len(excluded)}/{len(universe)} symbols (>{FROZEN_FRACTION_THRESHOLD:.0%} frozen)"
    )
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
