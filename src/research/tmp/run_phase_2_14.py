"""Notebook 014 Phase 2: descriptive check -- does each sector x dimension
behave like a regime model at all? (NEXT_PROMPT.md sec6 Phase 2)

Reads the Phase 1 panel parquet, computes transition matrix, regime
durations, time_in_regime, label_stability (flip rate, avg spell,
coverage), and expected_remaining_duration per sector x dimension. Flags
the three failure modes named in advance in NEXT_PROMPT.md sec6 Phase 2:
>90% single-label occupancy, flip rate high enough hysteresis isn't
binding, or a dimension NaN for most of the sample.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

import polars as pl

from regime.evaluation import label_stability
from regime.transitions import (
    expected_remaining_duration,
    regime_durations,
    time_in_regime,
    transition_matrix,
)

TMP = "src/research/tmp"
PANEL_PATH = "src/research/data/market/research/regime_panel.parquet"

# Failure-mode thresholds, named in advance (NEXT_PROMPT.md sec6 Phase 2).
MAX_SINGLE_LABEL_OCCUPANCY = 0.90
MAX_FLIP_RATE = 0.10  # more than one flip per 10 bars
MIN_LABELLED_COVERAGE = 0.10  # a dimension NaN >90% of the time is a coverage failure


def main() -> None:
    panel = pl.read_parquet(PANEL_PATH)

    # The panel (Phase 1) drops rows where both score and label are NaN, so
    # a naive pct-labelled computed on the panel alone is 100% by
    # construction and can never catch the "NaN for most of the sample"
    # failure mode (term_structure/carry before a curve's 2018 start, per
    # NEXT_PROMPT.md sec3.4). Use each sector's full score_history row count
    # from Phase 1 (which spans the sector's whole available history,
    # 2000-era start) as the coverage denominator instead.
    with open(f"{TMP}/phase_1_14_results.json") as f:
        phase1 = json.load(f)
    sector_total_rows = {s["name"]: s["rows"] for s in phase1["sectors"]}

    rows = []
    disqualified: list[dict] = []
    for (sector, dimension), group in panel.group_by(
        ["sector", "dimension"], maintain_order=True
    ):
        labels = (
            group.sort("date")
            .select("date", "label")
            .to_pandas()
            .set_index("date")["label"]
            .astype("string")
            .rename(None)  # regime_durations does a named agg(label=...) that
            # collides if the input Series is itself named "label"
        )
        stability = label_stability(labels)
        occupancy = time_in_regime(labels)
        matrix = transition_matrix(labels)
        durations = regime_durations(labels)
        remaining = expected_remaining_duration(matrix) if len(matrix) else None

        total_rows = sector_total_rows[sector]
        pct_of_sector_history = len(labels) / total_rows if total_rows else float("nan")
        max_occupancy = float(occupancy.max()) if len(occupancy) else float("nan")
        reasons = []
        if max_occupancy > MAX_SINGLE_LABEL_OCCUPANCY:
            reasons.append(
                f"single-label occupancy {max_occupancy:.2%} > {MAX_SINGLE_LABEL_OCCUPANCY:.0%}"
            )
        if stability["flip_rate"] > MAX_FLIP_RATE:
            reasons.append(f"flip_rate {stability['flip_rate']:.3f} > {MAX_FLIP_RATE}")
        if pct_of_sector_history < MIN_LABELLED_COVERAGE:
            reasons.append(
                f"labelled coverage {pct_of_sector_history:.2%} of sector history "
                f"< {MIN_LABELLED_COVERAGE:.0%}"
            )

        entry = {
            "sector": sector,
            "dimension": dimension,
            "n_bars": len(labels),
            "n_labelled": int(labels.notna().sum()),
            "sector_total_rows": total_rows,
            "flip_rate": stability["flip_rate"],
            "avg_duration": stability["avg_duration"],
            "pct_time_labeled_within_available": stability["pct_time_labeled"],
            "pct_of_sector_history_covered": pct_of_sector_history,
            "max_single_label_occupancy": max_occupancy,
            "occupancy_by_label": occupancy.to_dict(),
            "durations_by_label": durations.to_dict(orient="index"),
            "expected_remaining_duration": remaining.to_dict()
            if remaining is not None
            else {},
            "disqualified": bool(reasons),
            "disqualification_reasons": reasons,
        }
        rows.append(entry)
        if reasons:
            disqualified.append(
                {"sector": sector, "dimension": dimension, "reasons": reasons}
            )

    with open(f"{TMP}/phase_2_14_results.json", "w") as f:
        json.dump(
            {"per_sector_dimension": rows, "disqualified": disqualified},
            f,
            indent=2,
            default=str,
        )

    print(f"{len(rows)} sector x dimension pairs evaluated")
    print(f"{len(disqualified)} disqualified from Phase 3 scoring:")
    for item in disqualified:
        print(f"  {item['sector']} / {item['dimension']}: {item['reasons']}")


if __name__ == "__main__":
    main()
