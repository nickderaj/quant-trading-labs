"""Notebook 014 Phase 1: build the full historical regime panel over every
sector's whole available history (NEXT_PROMPT.md sec6 Phase 1).

Persists the panel to
src/research/data/market/research/regime_panel.parquet (long format:
sector, dimension, date, score, label) so later phases and future
notebooks load it rather than recompute it. Renders the snapshot heatmap,
6-month history small multiples, and the full-history label ribbon
(with the Phase-0-frozen episode table overlaid).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import polars as pl
import yaml

from regime.builder import build_regime_report
from regime.charts import (
    EpisodeSpan,
    history_figure,
    ribbon_figure,
    snapshot_figure,
    to_png,
)
from regime.universe import load_regime_universe

TMP = "src/research/tmp"
CHART_DIR = f"{TMP}/phase_1_14_charts"
PANEL_PATH = "src/research/data/market/research/regime_panel.parquet"


def _panel_frame(report) -> pl.DataFrame:
    rows = []
    for sector in report.sectors:
        for dimension in sector.score_history.columns:
            scores = sector.score_history[dimension]
            labels = (
                sector.label_history[dimension]
                if dimension in sector.label_history
                else pd.Series(index=scores.index, dtype="string")
            )
            for date, score, label in zip(scores.index, scores, labels, strict=True):
                if pd.isna(score) and pd.isna(label):
                    continue
                rows.append(
                    {
                        "sector": sector.name,
                        "kind": sector.kind,
                        "dimension": dimension,
                        "date": date,
                        "score": None if pd.isna(score) else float(score),
                        "label": None if pd.isna(label) else str(label),
                    }
                )
    return pl.DataFrame(rows)


def main() -> None:
    import os

    os.makedirs(CHART_DIR, exist_ok=True)

    with open("src/regime/configs/universe.yaml") as f:
        raw = yaml.safe_load(f)
    universe = load_regime_universe(raw)

    print("Building full-history report over", len(universe.baskets) + 1, "sectors...")
    report = build_regime_report(universe, as_of=None, include_oil_products_cot=False)
    if report.errors:
        print("ERRORS:", report.errors)

    panel = _panel_frame(report)
    panel.write_parquet(PANEL_PATH)
    print(
        f"Wrote {PANEL_PATH}: {panel.height} rows, {panel['sector'].n_unique()} sectors"
    )

    with open(f"{TMP}/phase_0_14_preregistration.json") as f:
        prereg = json.load(f)
    episodes = [
        EpisodeSpan(e["start"], e["end"], e["episode"]) for e in prereg["episode_table"]
    ]

    snap_path = f"{CHART_DIR}/snapshot.png"
    with open(snap_path, "wb") as f:
        f.write(to_png(snapshot_figure(report)))

    hist_fig = history_figure(report)
    hist_path = None
    if hist_fig is not None:
        hist_path = f"{CHART_DIR}/history.png"
        with open(hist_path, "wb") as f:
            f.write(to_png(hist_fig))

    ribbon_path = f"{CHART_DIR}/ribbon_full.png"
    with open(ribbon_path, "wb") as f:
        f.write(to_png(ribbon_figure(report, episodes=episodes)))

    # A second ribbon scoped to just the sectors the episode table actually
    # names, so the chart Phase 3 leans on stays legible.
    episode_sectors = {"Macro", "Commodities", "oil products"}
    ribbon_episodes_path = f"{CHART_DIR}/ribbon_episode_sectors.png"
    with open(ribbon_episodes_path, "wb") as f:
        f.write(
            to_png(
                ribbon_figure(
                    report, episodes=episodes, sectors=sorted(episode_sectors)
                )
            )
        )

    results = {
        "errors": report.errors,
        "panel_rows": panel.height,
        "panel_path": PANEL_PATH,
        "sectors": [
            {
                "name": sector.name,
                "kind": sector.kind,
                "dimensions": list(sector.score_history.columns),
                "symbols_used": sector.symbols_used,
                "symbols_skipped": sector.symbols_skipped,
                "rows": len(sector.score_history),
                "first": str(sector.score_history.index.min())
                if len(sector.score_history)
                else None,
                "last": str(sector.score_history.index.max())
                if len(sector.score_history)
                else None,
            }
            for sector in report.sectors
        ],
        "charts": {
            "snapshot": snap_path.replace(TMP, "tmp"),
            "history": hist_path.replace(TMP, "tmp") if hist_path else None,
            "ribbon_full": ribbon_path.replace(TMP, "tmp"),
            "ribbon_episode_sectors": ribbon_episodes_path.replace(TMP, "tmp"),
        },
    }
    with open(f"{TMP}/phase_1_14_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Wrote phase_1_14_results.json")


if __name__ == "__main__":
    main()
