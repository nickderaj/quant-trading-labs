"""Regenerates the deployable risk-engine dashboard at `index.html` (repo
root, so Vercel can deploy the whole repo with zero-config static hosting --
see `vercel.json`) from the current `src/risk/data/` ingest output.

Not a gate script -- this is the operator convenience for keeping the
publicly-deployed dashboard (README link, Vercel) in sync with a fresh
`risk.ingest.refresh()` run. Run `risk.ingest.refresh()` first if the
ingested data is stale; this script only renders, it does not ingest.

Usage: `uv run python src/research/tmp/render_risk_dashboard.py [as_of]`
`as_of` defaults to today (UTC date), display-only (NEXT_PROMPT.md sec 7.4).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "src")

from risk import serve

OUT_PATH = Path(__file__).resolve().parents[3] / "index.html"


def main() -> None:
    as_of = sys.argv[1] if len(sys.argv) > 1 else datetime.now(UTC).date().isoformat()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serve.render_dashboard(as_of=as_of, out_path=OUT_PATH)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH} ({size_kb:.0f} KiB, as_of={as_of})")


if __name__ == "__main__":
    main()
