"""Tests for notebook 021's power_lib21.py (NEXT_PROMPT.md sec 4 Phase 2).
Network-free: every test builds small synthetic parquet files / polars
frames directly, no cached-cache dependency.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.stats as st

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "research" / "tmp")
)

import power_lib21 as pw21


def _dt(i: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=8 * i)


# --------------------------------------------------------------------------
# flag_frozen_feed_bars
# --------------------------------------------------------------------------


def test_flag_frozen_feed_bars_detects_only_true_frozen(tmp_path) -> None:
    times = [_dt(i) for i in range(4)]
    syma = pl.DataFrame(
        {
            "datetime": times,
            "open": [1.0, 100.0, 50.0, 1.0],
            "high": [1.1, 100.0, 50.0, 1.05],
            "low": [0.9, 100.0, 50.0, 0.95],
            "close": [1.05, 100.0, 50.0, 1.02],
            "volume": [100.0, 0.0, 10.0, 0.0],
        }
    )
    symb = pl.DataFrame(
        {
            "datetime": times,
            "open": [2.0, 2.1, 2.2, 2.3],
            "high": [2.2, 2.3, 2.4, 2.5],
            "low": [1.9, 2.0, 2.1, 2.2],
            "close": [2.1, 2.2, 2.3, 2.4],
            "volume": [50.0, 60.0, 70.0, 80.0],
        }
    )
    syma.write_parquet(tmp_path / "SYMAUSDT-perp-8h-2021-07-01-2025-06-30.parquet")
    symb.write_parquet(tmp_path / "SYMBUSDT-perp-8h-2021-07-01-2025-06-30.parquet")

    out = pw21.flag_frozen_feed_bars(
        str(tmp_path / "*-perp-8h-2021-07-01-2025-06-30.parquet")
    )

    assert len(out) == 1
    assert out["symbol"].to_list() == ["SYMAUSDT"]
    assert out["datetime"].to_list()[0] == times[1]


def test_flag_frozen_feed_bars_empty_when_nothing_matches(tmp_path) -> None:
    times = [_dt(i) for i in range(2)]
    symb = pl.DataFrame(
        {
            "datetime": times,
            "open": [2.0, 2.1],
            "high": [2.2, 2.3],
            "low": [1.9, 2.0],
            "close": [2.1, 2.2],
            "volume": [50.0, 60.0],
        }
    )
    symb.write_parquet(tmp_path / "SYMBUSDT-perp-8h-2021-07-01-2025-06-30.parquet")

    out = pw21.flag_frozen_feed_bars(
        str(tmp_path / "*-perp-8h-2021-07-01-2025-06-30.parquet")
    )
    assert len(out) == 0


# --------------------------------------------------------------------------
# excluded_book_bars -- hand-built 5-bar synthetic frame
# --------------------------------------------------------------------------


def test_excluded_book_bars_hand_built_5_bar_frame() -> None:
    times = [_dt(i) for i in range(5)]  # t0..t4

    # Catalogue: symbol X flagged at t0 (no prev, held=False at t0 itself),
    # t2, t3, t4 (held=False at t4 itself, prev t3 already covered).
    catalogue = pl.DataFrame(
        {
            "symbol": ["X", "X", "X", "X"],
            "datetime": [times[0], times[2], times[3], times[4]],
        }
    )

    # X held (weight != 0) at t1, t2, t3 only. Y held at every bar, never
    # flagged -- must never appear in the excluded set.
    x_weights = [0.0, 0.5, 0.5, 0.5, 0.0]
    y_weights = [0.5, 0.5, 0.5, 0.5, 0.5]
    a0_weights = pl.DataFrame(
        {
            "datetime": times + times,
            "symbol": ["X"] * 5 + ["Y"] * 5,
            "weight": x_weights + y_weights,
        }
    )

    excluded = pw21.excluded_book_bars(catalogue, a0_weights)

    # (X,t0): itself not held (i=0, no prev) -> nothing added.
    # (X,t2): itself held -> t2 added; prev t1 held -> t1 added.
    # (X,t3): itself held -> t3 added; prev t2 held -> t2 added (dup).
    # (X,t4): itself not held -> nothing; prev t3 held -> t3 added (dup).
    assert excluded == {times[1], times[2], times[3]}


def test_excluded_book_bars_empty_catalogue_gives_empty_set() -> None:
    times = [_dt(i) for i in range(3)]
    catalogue = pl.DataFrame({"symbol": [], "datetime": []}).cast(
        {"symbol": pl.Utf8, "datetime": pl.Datetime}
    )
    a0_weights = pl.DataFrame(
        {"datetime": times, "symbol": ["X"] * 3, "weight": [0.5, 0.5, 0.5]}
    )
    assert pw21.excluded_book_bars(catalogue, a0_weights) == set()


# --------------------------------------------------------------------------
# closed-form power helpers
# --------------------------------------------------------------------------


def test_bootstrap_se_from_ci_inverts_known_normal_case() -> None:
    se = 2e-5
    mean = 1e-5
    z = st.norm.ppf(0.975)
    ci_lo, ci_hi = mean - z * se, mean + z * se
    recovered = pw21.bootstrap_se_from_ci(ci_lo, ci_hi)
    assert recovered == pytest.approx(se, rel=1e-9)


def test_mde_matches_hand_derivation() -> None:
    se = 1.0
    expected = (st.norm.ppf(0.975) + st.norm.ppf(0.80)) * se
    assert pw21.mde(se) == pytest.approx(expected, rel=1e-9)
    assert pw21.mde(se) == pytest.approx(2.801585, rel=1e-5)


def test_n_required_scaling() -> None:
    assert pw21.n_required(
        n_obs=100, observed_mean=2.0, mde_value=2.0
    ) == pytest.approx(100.0)
    assert pw21.n_required(
        n_obs=100, observed_mean=1.0, mde_value=2.0
    ) == pytest.approx(400.0)


# --------------------------------------------------------------------------
# placebo_mean_diffs
# --------------------------------------------------------------------------


def test_placebo_mean_diffs_reproducible_with_seed() -> None:
    diff_frame = pl.DataFrame({"diff": np.arange(20, dtype=float)})
    out1 = pw21.placebo_mean_diffs(diff_frame, n_excluded=5, n_draws=10, seed=0)
    out2 = pw21.placebo_mean_diffs(diff_frame, n_excluded=5, n_draws=10, seed=0)
    assert out1.shape == (10,)
    np.testing.assert_array_equal(out1, out2)


def test_placebo_mean_diffs_different_seed_differs() -> None:
    diff_frame = pl.DataFrame({"diff": np.arange(20, dtype=float)})
    out1 = pw21.placebo_mean_diffs(diff_frame, n_excluded=5, n_draws=10, seed=0)
    out2 = pw21.placebo_mean_diffs(diff_frame, n_excluded=5, n_draws=10, seed=1)
    assert not np.array_equal(out1, out2)
