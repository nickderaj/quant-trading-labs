"""Notebook 020, Phase 6 (NEXT_PROMPT.md sec 9): the holdout gate.

Mechanical fencing, not discipline (sec 9 point 6): this is the ONLY file in
the repo permitted to name `src/research/cache/basis18/holdout` or
`src/research/cache/bybit20/holdout`, or pass an end_date past
`research.HOLDOUT_START`. It reads phase_4_20_results.json's
"holdout_access" block and exits 1 WITHOUT constructing any path into a
holdout directory if neither mechanism qualifies (sec 9 point 4).

Mechanism A unlocks iff RC-2 AND RC-3 both fire.
Mechanism B unlocks iff XD-2 AND XD-3 both fire.
No partial credit (sec 9 point 2).

If both qualify, this runs a single non-iterative pass over both headline
books (A3, B0) and reports both -- n_trials=32 already counts this pass
(sec 9 point 4). After a holdout run the notebook closes: no dev-side
re-runs, no parameter changes (sec 9 point 5).

Usage: uv run python src/research/tmp/run_phase_6_20_holdout.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

PHASE4_RESULTS_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_4_20_results.json"
OUT_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_6_20_holdout.json"

# The only two literal holdout directory names in this notebook's entire
# codebase live HERE, and nowhere else (verify with the sec 9 point 6 grep).
BASIS18_HOLDOUT_DIR = "src/research/cache/basis18/holdout"
BYBIT20_HOLDOUT_DIR = "src/research/cache/bybit20/holdout"


def main() -> None:
    with open(PHASE4_RESULTS_PATH) as f:
        results = json.load(f)

    access = results["holdout_access"]
    a_unlocked = bool(access["mechanism_a_unlocked"])
    b_unlocked = bool(access["mechanism_b_unlocked"])

    print(f"holdout_access from phase_4_20_results.json: {access}")

    if not a_unlocked and not b_unlocked:
        doc = {
            "ran": False,
            "reason": (
                "Neither mechanism qualified: Mechanism A needs RC-2 AND RC-3 "
                "(RC-2 fired, RC-3 did not -- the paired diff vs the 018-baseline "
                "reproduction did not clear its bootstrap CI). Mechanism B needs "
                "XD-2 AND XD-3 (neither fired -- H-B was not supported). "
                "No path into a holdout directory was constructed."
            ),
            "holdout_access_read": access,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(doc, f, indent=2)
        print(
            "REFUSED: no mechanism unlocked the holdout. Exiting 1 without touching any holdout path."
        )
        print(f"Wrote {OUT_PATH}")
        sys.exit(1)

    # Not reached in this run (neither mechanism qualified) and deliberately
    # NOT implemented ahead of that fact: sec 9 point 5 forbids "one more
    # variant" after a holdout run, and pre-writing an untested holdout path
    # against data this notebook has never touched is exactly the kind of
    # speculative code this repo's tripwire discipline warns against. If a
    # future run of this script ever reaches here (RC-2+RC-3, or XD-2+XD-3,
    # both fire), implement the single real pass THEN, against
    # BASIS18_HOLDOUT_DIR / BYBIT20_HOLDOUT_DIR, and nowhere else.
    raise NotImplementedError(
        f"Holdout unlocked (mechanism_a={a_unlocked}, mechanism_b={b_unlocked}) "
        "but the real holdout pass is not yet implemented -- implement it now, "
        "against BASIS18_HOLDOUT_DIR / BYBIT20_HOLDOUT_DIR only, before proceeding."
    )


if __name__ == "__main__":
    main()
