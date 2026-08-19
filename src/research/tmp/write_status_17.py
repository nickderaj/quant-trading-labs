"""Notebook 017 Phase 3 (NEXT_PROMPT.md sec 9.1 item 3): heartbeat writer
for scratch/017/status.json, independent of the workers -- called by the
runner's background heartbeat loop, not by the cell workers themselves.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

STATUS_PATH = "scratch/017/status.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--done", type=int, default=0)
    parser.add_argument("--total", type=int, default=0)
    parser.add_argument("--failed", type=int, default=0)
    parser.add_argument("--state", required=True, choices=["running", "done", "failed"])
    args = parser.parse_args()

    now = datetime.now(UTC)
    remaining = max(args.total - args.done, 0)
    eta_seconds = 0
    if args.state == "running" and args.done > 0 and remaining > 0:
        # crude linear ETA from a fixed per-cell estimate isn't available
        # here (workers don't report timing); left at 0 and the heartbeat
        # cadence itself (every 30s) is the actual progress signal.
        eta_seconds = 0

    doc = {
        "phase": args.phase,
        "state": args.state,
        "done": args.done,
        "total": args.total,
        "failed": args.failed,
        "eta_seconds": eta_seconds,
        "last_heartbeat": now.isoformat(),
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(doc, f, indent=2)


if __name__ == "__main__":
    main()
