"""Walk-forward-safe alignment of lower-frequency data to market bars.

Ported verbatim from ``ultron_finance.regime.align``
(``../ultron/libs/finance/src/ultron_finance/regime/align.py``).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def align_to_daily(
    target_index: pd.DatetimeIndex,
    s: pd.Series,
    publication_lag_days: int = 0,
    max_staleness_days: int | None = None,
) -> pd.Series:
    """Shift observations by business-day publication lag and forward-fill safely."""
    if publication_lag_days < 0:
        raise ValueError("publication_lag_days must be non-negative")
    if max_staleness_days is not None and max_staleness_days < 0:
        raise ValueError("max_staleness_days must be non-negative")
    target = pd.DatetimeIndex(pd.to_datetime(target_index))
    source = s.copy()
    source.index = pd.DatetimeIndex(pd.to_datetime(source.index))
    source = source.sort_index()
    published = source.copy()
    published.index = published.index + pd.offsets.BDay(publication_lag_days)
    published = published.groupby(level=0).last()
    # Preserve source timestamps while filling. Reindexing directly to the
    # target discards midnight observations when target bars carry a market
    # close time (e.g. 05:00 UTC), leaving nothing to forward-fill.
    combined_index = published.index.union(target).sort_values()
    result = published.reindex(combined_index).ffill().reindex(target)
    if max_staleness_days is None:
        result.name = s.name
        return result
    publication_dates = (
        pd.Series(published.index, index=published.index)
        .reindex(combined_index)
        .ffill()
        .reindex(target)
    )
    age = pd.Series(target, index=target) - publication_dates
    aligned = result.where(age <= pd.Timedelta(days=max_staleness_days))
    aligned.name = s.name
    return aligned


def align_frame_to_daily(
    target_index: pd.DatetimeIndex,
    df: pd.DataFrame,
    lags: Mapping[str, int] | int = 0,
    max_staleness_days: int | None = None,
) -> pd.DataFrame:
    """Align each column of a lower-frequency frame with its publication lag."""
    if isinstance(lags, int):
        return pd.DataFrame(
            {
                column: align_to_daily(
                    target_index, df[column], lags, max_staleness_days
                )
                for column in df
            },
            index=target_index,
        )
    unknown = set(lags).difference(map(str, df.columns))
    if unknown:
        raise ValueError(f"lags contains unknown columns: {sorted(unknown)}")
    return pd.DataFrame(
        {
            column: align_to_daily(
                target_index, df[column], lags.get(str(column), 0), max_staleness_days
            )
            for column in df
        },
        index=target_index,
    )
