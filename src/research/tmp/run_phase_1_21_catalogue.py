"""Notebook 021, Phase 1 (NEXT_PROMPT.md sec 4): the mechanical
liquidity-collapse (frozen-feed) catalogue, over all 128 cached full-range
perp files. No symbol names appear anywhere in this file, by construction --
that is PW-1's own discipline (see phase_0_21_preregistration.json's gate
PW-1 note), enforced mechanically in Phase 4 by a grep over this file's own
path that must return nothing.

Writes phase_1_21_catalogue.json (committed -- the full flagged list, as
contiguous per-symbol runs, plus summary counts) and
phase_1_21_catalogue.parquet (gitignored, regenerable in ~0.34s, the exact
per-bar frame Phase 3 needs for the book-return exclusion).

Usage: uv run python src/research/tmp/run_phase_1_21_catalogue.py
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import power_lib21 as pw21

PERP_GLOB = str(
    REPO_ROOT
    / "src"
    / "research"
    / "cache"
    / "basis18"
    / "dev"
    / "*-perp-8h-2021-07-01-2025-06-30.parquet"
)
OUT_JSON = REPO_ROOT / "src" / "research" / "tmp" / "phase_1_21_catalogue.json"
OUT_PARQUET = REPO_ROOT / "src" / "research" / "tmp" / "phase_1_21_catalogue.parquet"
BAR_STEP = timedelta(hours=8)


def _runs(datetimes: list) -> list[dict]:
    """Collapse a sorted list of 8h-spaced datetimes into contiguous runs
    (start, end, count) -- a compact, lossless disclosure format matching
    the mostly-contiguous dead-feed-tail shape sec 1 describes.
    """
    if not datetimes:
        return []
    runs = []
    run_start = datetimes[0]
    prev = datetimes[0]
    count = 1
    for dt in datetimes[1:]:
        if dt - prev == BAR_STEP:
            count += 1
        else:
            runs.append({"start": str(run_start), "end": str(prev), "count": count})
            run_start = dt
            count = 1
        prev = dt
    runs.append({"start": str(run_start), "end": str(prev), "count": count})
    return runs


def main() -> None:
    n_files = len(glob.glob(PERP_GLOB))
    catalogue = pw21.flag_frozen_feed_bars(PERP_GLOB)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    catalogue.write_parquet(OUT_PARQUET)

    per_symbol: dict[str, dict] = {}
    for symbol in catalogue["symbol"].unique().sort().to_list():
        dts = sorted(catalogue.filter(pl.col("symbol") == symbol)["datetime"].to_list())
        per_symbol[symbol] = {"count": len(dts), "runs": _runs(dts)}

    result = {
        "perp_glob": PERP_GLOB,
        "n_files_scanned": n_files,
        "total_flagged_symbol_bars": len(catalogue),
        "n_symbols_flagged": catalogue["symbol"].n_unique() if len(catalogue) else 0,
        "per_symbol": per_symbol,
        "book_return_bar_count_note": (
            "the count that matters for the diff series -- bars where A0 was actually "
            "holding a flagged symbol -- requires A0's own weight frame and is computed "
            "and disclosed in Phase 3 (phase_3_21_results.json), before the "
            "excluded-diff CI is computed there."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"scanned {n_files} files; flagged {result['total_flagged_symbol_bars']} "
        f"symbol-bars across {result['n_symbols_flagged']} symbols"
    )


if __name__ == "__main__":
    main()
