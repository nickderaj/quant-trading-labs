from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime.align import align_frame_to_daily, align_to_daily


def test_cot_publication_lag_prevents_lookahead() -> None:
    target = pd.date_range("2024-01-02", "2024-01-12", freq="B")
    tuesday = pd.Series([42.0], index=pd.DatetimeIndex(["2024-01-02"]), name="cot")

    aligned = align_to_daily(target, tuesday, publication_lag_days=3)

    assert aligned.loc[:"2024-01-04"].isna().all()
    assert aligned.loc["2024-01-05"] == 42.0


def test_alignment_staleness_and_per_column_lags() -> None:
    target = pd.date_range("2024-01-02", "2024-01-12", freq="B")
    source = pd.DataFrame(
        {"daily": [1.0], "weekly": [2.0]}, index=pd.DatetimeIndex(["2024-01-02"])
    )

    aligned = align_frame_to_daily(target, source, {"daily": 0, "weekly": 3}, 3)

    assert aligned.loc["2024-01-02", "daily"] == 1.0
    assert pd.isna(aligned.loc["2024-01-04", "weekly"])
    assert aligned.loc["2024-01-05", "weekly"] == 2.0
    assert pd.isna(aligned.loc["2024-01-08", "daily"])


def test_alignment_handles_market_close_timestamps() -> None:
    source = pd.Series([10.0], index=pd.DatetimeIndex(["2024-01-02"]), name="macro")
    target = pd.DatetimeIndex(["2024-01-02 05:00:00"])

    result = align_to_daily(target, source)

    assert result.iloc[0] == 10.0
