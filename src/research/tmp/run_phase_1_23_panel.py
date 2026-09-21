"""Phase 1 -- build the curve panel for each liquidity variant."""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib23

TMP = Path(__file__).resolve().parent

raw = lib23.load_raw_panel("CL")
report: dict[str, Any] = {"raw_rows": len(raw)}

for variant in ["all", "vol_gt_0", "vol_gt_100"]:
    clean, rep = lib23.apply_hygiene(raw.copy(), variant)
    clean.to_parquet(TMP / f"phase_1_23_panel_{variant}.parquet")
    report[variant] = rep

out = TMP / "phase_1_23_panel_report.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {out}")
for v in ["all", "vol_gt_0", "vol_gt_100"]:
    r = report[v]
    print(
        v,
        "days_retained=",
        r["days_retained"],
        "mean_contracts/day=",
        round(r["mean_contracts_per_day"], 2),
    )
