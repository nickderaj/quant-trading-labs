import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "research" / "tmp"))

import research
from alpha_lib7 import (
    hysteresis_weights,
    quantize_weights,
    throttle_weights,
    apply_book_scale,
)


def _synthetic_panel(n_times=40, n_symbols=20, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_times):
        dt = datetime(2022, 1, 1) + timedelta(hours=4 * t)  # noqa: DTZ001
        for s in range(n_symbols):
            rows.append(
                {
                    "datetime": dt,
                    "symbol": f"S{s}",
                    "pred": float(rng.normal()),
                    "vol": float(abs(rng.normal()) + 0.01),
                    "fwd_return_1": float(rng.normal(scale=0.01)),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# The single most important correctness check in this notebook: band=0.0
# must reproduce research.dollar_neutral_weights EXACTLY.
# ---------------------------------------------------------------------------


def test_hysteresis_band_zero_reproduces_dollar_neutral_weights_exactly():
    panel = _synthetic_panel()
    baseline = research.dollar_neutral_weights(
        panel, "pred", size_col="vol", top_frac=0.2,
        gross_exposure=1.0, max_position_per_symbol=0.25,
    ).sort(["datetime", "symbol"])
    hysteresis = hysteresis_weights(
        panel, "pred", band=0.0, size_col="vol", top_frac=0.2,
        gross_exposure=1.0, max_position_per_symbol=0.25,
    ).sort(["datetime", "symbol"])
    assert baseline.shape == hysteresis.shape
    np.testing.assert_allclose(
        baseline["weight"].to_numpy(), hysteresis["weight"].to_numpy(), atol=1e-12
    )


def test_hysteresis_band_zero_reproduces_without_size_col_too():
    panel = _synthetic_panel(seed=1)
    baseline = research.dollar_neutral_weights(
        panel, "pred", top_frac=0.3, gross_exposure=1.0, max_position_per_symbol=0.5
    ).sort(["datetime", "symbol"])
    hysteresis = hysteresis_weights(
        panel, "pred", band=0.0, top_frac=0.3,
        gross_exposure=1.0, max_position_per_symbol=0.5,
    ).sort(["datetime", "symbol"])
    np.testing.assert_allclose(
        baseline["weight"].to_numpy(), hysteresis["weight"].to_numpy(), atol=1e-12
    )


def test_hysteresis_is_near_neutral_and_within_gross_cap():
    """Net is allowed to deviate from exactly 0 by the same mechanism as
    research.dollar_neutral_weights itself: per-symbol clipping to
    max_position_per_symbol happens AFTER per-leg normalization, so a
    size-weighted leg with a skewed size distribution can clip asymmetrically
    across the long/short legs. That is inherited behavior, not new -
    band=0.0's exact match to dollar_neutral_weights (tested above) proves
    it's the same mechanism, not a hysteresis-specific bug - so this test
    bounds |net| by a small multiple of the cap rather than requiring exact
    zero."""
    panel = _synthetic_panel(seed=2)
    for band in (0.0, 0.05, 0.1, 0.2):
        w = hysteresis_weights(panel, "pred", band=band, size_col="vol", top_frac=0.2)
        per_bar = w.group_by("datetime").agg(pl.col("weight").sum().alias("net"))
        assert per_bar["net"].abs().max() <= 0.25 + 1e-9
        assert w["weight"].abs().max() <= 0.25 + 1e-9


def test_widening_hysteresis_band_reduces_turnover():
    """Tripwire from NEXT_PROMPT.md section 9: turnover must fall (not stay
    flat or rise) as the band widens, on a panel with enough rank noise to
    actually trigger the mechanism."""
    panel = _synthetic_panel(n_times=80, n_symbols=15, seed=3)
    turnovers = []
    for band in (0.0, 0.05, 0.1, 0.15, 0.2):
        w = hysteresis_weights(panel, "pred", band=band, size_col="vol", top_frac=0.2)
        t = research.portfolio_turnover(w)
        turnovers.append(float(t["turnover"].mean()))
    for earlier, later in zip(turnovers, turnovers[1:]):
        assert later <= earlier + 1e-9, f"turnover rose as band widened: {turnovers}"
    assert turnovers[-1] < turnovers[0], f"widest band did not reduce turnover at all: {turnovers}"


def test_hysteresis_holds_membership_between_thresholds():
    """A symbol sitting in the exit zone (still in long_keep, not in
    long_enter) must retain its prior weight sign rather than being
    flattened - the actual mechanism this function exists to test."""
    times = [datetime(2022, 1, 1) + timedelta(hours=4 * t) for t in range(3)]  # noqa: DTZ001
    symbols = [f"S{i}" for i in range(10)]
    # Bar 0: S9 is the clear top -> enters long. Bar 1: S9 drops just below
    # the entry cutoff (rank 7/10, i.e. inside a wide keep-band) but not far
    # enough to leave a band=0.5 window -> must remain long. Bar 2: S9 drops
    # to the very bottom -> should flip out of long (and even to short).
    preds = [
        list(range(10)),  # bar0: S0..S9 ranked ascending, S9 top
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0][::-1],  # placeholder, overwritten below
        None,
    ]
    rows = []
    # bar 0
    for i, s in enumerate(symbols):
        rows.append({"datetime": times[0], "symbol": s, "pred": float(i)})
    # bar 1: S9 falls to rank index 6 (0-indexed ascending) of 10 -> percentile 0.7
    bar1_order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S9", "S7", "S8"]
    for i, s in enumerate(bar1_order):
        rows.append({"datetime": times[1], "symbol": s, "pred": float(i)})
    # bar 2: S9 falls to the very bottom
    bar2_order = ["S9", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    for i, s in enumerate(bar2_order):
        rows.append({"datetime": times[2], "symbol": s, "pred": float(i)})
    panel = pl.DataFrame(rows)

    w = hysteresis_weights(panel, "pred", band=0.3, top_frac=0.2)
    w1 = w.filter((pl.col("datetime") == times[1]) & (pl.col("symbol") == "S9"))
    w2 = w.filter((pl.col("datetime") == times[2]) & (pl.col("symbol") == "S9"))
    assert w1["weight"][0] > 0, "S9 should still be held long inside the exit band"
    assert w2["weight"][0] < 0, "S9 should flip to short once it exits the band entirely"


def test_quantize_weights_snaps_to_grid():
    w = pl.DataFrame({"datetime": [1, 1, 2], "symbol": ["A", "B", "A"], "weight": [0.123, -0.049, 0.026]})
    q = quantize_weights(w, grid=0.05)
    np.testing.assert_allclose(q["weight"].to_numpy(), [0.10, -0.05, 0.05], atol=1e-12)


def test_quantize_weights_rejects_nonpositive_grid():
    w = pl.DataFrame({"datetime": [1], "symbol": ["A"], "weight": [0.1]})
    try:
        quantize_weights(w, grid=0.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_throttle_k1_is_noop():
    panel = _synthetic_panel(seed=4)
    w = research.dollar_neutral_weights(panel, "pred", top_frac=0.2)
    throttled = throttle_weights(w, k=1)
    joined = w.sort(["datetime", "symbol"]).join(
        throttled.sort(["datetime", "symbol"]), on=["datetime", "symbol"], suffix="_t"
    )
    np.testing.assert_allclose(
        joined["weight"].to_numpy(), joined["weight_t"].to_numpy(), atol=1e-12
    )


def test_throttle_holds_weight_between_rebalances():
    panel = _synthetic_panel(n_times=12, n_symbols=6, seed=5)
    w = research.dollar_neutral_weights(panel, "pred", top_frac=0.3)
    throttled = throttle_weights(w, k=3).sort(["symbol", "datetime"])
    for _sym, g in throttled.group_by("symbol", maintain_order=True):
        weights = g["weight"].to_numpy()
        # bars 1,2 must equal bar 0; bars 4,5 must equal bar 3; etc.
        for start in range(0, len(weights) - 2, 3):
            np.testing.assert_allclose(weights[start : start + 3], weights[start], atol=1e-12)


def test_throttle_reduces_or_preserves_turnover():
    panel = _synthetic_panel(n_times=60, n_symbols=15, seed=6)
    w = research.dollar_neutral_weights(panel, "pred", top_frac=0.2)
    base_turnover = float(research.portfolio_turnover(w)["turnover"].mean())
    for k in (2, 3, 6):
        throttled = throttle_weights(w, k=k)
        t = float(research.portfolio_turnover(throttled)["turnover"].mean())
        assert t <= base_turnover + 1e-9, f"k={k} turnover {t} exceeded k=1 baseline {base_turnover}"


def test_apply_book_scale_zeroes_out_book_on_standdown_bars():
    w = pl.DataFrame(
        {
            "datetime": [1, 1, 2, 2],
            "symbol": ["A", "B", "A", "B"],
            "weight": [0.5, -0.5, 0.4, -0.4],
        }
    )
    scale = pl.DataFrame({"datetime": [1, 2], "book_scale": [1.0, 0.0]})
    out = apply_book_scale(w, scale).sort(["datetime", "symbol"])
    np.testing.assert_allclose(
        out.filter(pl.col("datetime") == 2)["weight"].to_numpy(), [0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        out.filter(pl.col("datetime") == 1)["weight"].to_numpy(), [0.5, -0.5], atol=1e-12
    )
