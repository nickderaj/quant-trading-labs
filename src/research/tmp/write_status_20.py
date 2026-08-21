"""Notebook 020 (NEXT_PROMPT.md sec 3 rule 6): heartbeat writer for
scratch/020/status.json, independent of the workers.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

STATUS_PATH = "scratch/020/status.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--done", type=int, default=0)
    parser.add_argument("--total", type=int, default=0)
    parser.add_argument("--failed", type=int, default=0)
    parser.add_argument("--state", required=True, choices=["running", "done", "failed"])
    args = parser.parse_args()

    now = datetime.now(UTC)
    doc = {
        "phase": args.phase,
        "state": args.state,
        "done": args.done,
        "total": args.total,
        "failed": args.failed,
        "last_heartbeat": now.isoformat(),
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(doc, f, indent=2)


if __name__ == "__main__":
    main()
