"""Notebook 020, Phase 4 grid builder (NEXT_PROMPT.md sec 7/sec 8).

Writes scratch/020/phase4_cells.txt (one "<variant> <offset>" line per cell)
and scratch/020/phase4_manifest.json. The 17 cells here are the exact
itemisation of sec 7's "Mechanism A, Phase 4 (9)" and "Mechanism B, Phase 4
(8)" rows -- the wall-clock table's "14 book builds" (sec 3) was a rough
pre-run estimate; sec 7's itemised count is the load-bearing one used for
every DSR in this notebook and is what this grid reproduces exactly.

Usage: uv run python src/research/tmp/build_phase4_grid_20.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH_DIR = REPO_ROOT / "scratch" / "020"
CELLS_PATH = SCRATCH_DIR / "phase4_cells.txt"
MANIFEST_PATH = SCRATCH_DIR / "phase4_manifest.json"

# (variant, offsets) -- sec 7's itemisation.
CELLS: list[tuple[str, int]] = [
    ("A0", 0),
    ("A1", 0),
    ("A2", 0),
    ("A3", 0),
    ("A3", 1),
    ("A3", 2),
    ("A3", 3),
    ("A_alwayson", 0),
    ("A_cash", 0),
    ("B0", 0),
    ("B0", 1),
    ("B0", 2),
    ("B0", 3),
    ("B1", 0),
    ("B_single", 0),
    ("B_alwayson", 0),
    ("B_cash", 0),
]


def main() -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(CELLS_PATH, "w") as f:
        f.writelines(f"{variant} {offset}\n" for variant, offset in CELLS)

    manifest = {
        "n_cells": len(CELLS),
        "cells": [{"variant": v, "offset": o} for v, o in CELLS],
        "n_trials_itemisation_check": {
            "mechanism_a_count": sum(1 for v, _ in CELLS if v.startswith("A")),
            "mechanism_b_count": sum(1 for v, _ in CELLS if v.startswith("B")),
            "expected_a": 9,
            "expected_b": 8,
        },
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {CELLS_PATH} ({len(CELLS)} cells) and {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
