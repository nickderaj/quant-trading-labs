"""The six tests NEXT_PROMPT.md sec 7.3 requires for notebook 018's library.

Network-free: every test builds small synthetic polars frames directly, no
Binance fetch involved.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "research" / "tmp")
)

import basis_lib18 as bl

import research


def _dt_col(n: int, start: datetime | None = None) -> list[datetime]:
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    return [start + timedelta(hours=8 * i) for i in range(n)]


# --------------------------------------------------------------------------
# 1. carry_estimate is strictly causal
# --------------------------------------------------------------------------


def test_carry_estimate_is_causal() -> None:
    """Perturbing funding at time t must not change any carry value at
    s < t -- the leakage test 013's own traps require and this whole
    strategy's entry/exit signal depends on.
    """
    rng = np.random.default_rng(0)
    n = 200
    base_funding = rng.normal(0.0001, 0.00005, n)

    df_base = pl.DataFrame({"funding": base_funding})
    carry_base = df_base.select(bl.carry_estimate(pl.col("funding")).alias("carry"))[
        "carry"
    ].to_numpy()

    perturb_idx = 150
    perturbed_funding = base_funding.copy()
    perturbed_funding[perturb_idx] += 0.05  # a large, obvious shock

    df_pert = pl.DataFrame({"funding": perturbed_funding})
    carry_pert = df_pert.select(bl.carry_estimate(pl.col("funding")).alias("carry"))[
        "carry"
    ].to_numpy()

    # Every carry value strictly before the perturbation must be unchanged.
    np.testing.assert_array_equal(carry_base[:perturb_idx], carry_pert[:perturb_idx])
    # And the perturbation must actually have moved something at/after it,
    # so this test isn't vacuously passing on a no-op estimator.
    assert not np.allclose(carry_base[perturb_idx:], carry_pert[perturb_idx:])


# --------------------------------------------------------------------------
# 2. paired return is delta-neutral
# --------------------------------------------------------------------------


def test_paired_return_is_delta_neutral() -> None:
    """Spot and perp moving identically with zero funding must produce
    exactly zero paired return -- the whole point of the trade.
    """
    n = 50
    price_path = 100.0 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, n)))
    df = pl.DataFrame(
        {
            "spot_close": price_path,
            "perp_close": price_path,
            "funding": np.zeros(n),
        }
    )
    result = df.select(
        bl.paired_log_return(
            pl.col("spot_close"), pl.col("perp_close"), pl.col("funding")
        ).alias("paired_return")
    )["paired_return"].to_numpy()

    assert np.allclose(result[1:], 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# 3. positive funding pays the short (sign convention)
# --------------------------------------------------------------------------


def test_positive_funding_pays_the_short() -> None:
    """Flat prices (no basis mark-to-market term), positive funding: the
    long-spot/short-perp book must show a POSITIVE paired return, since
    Binance pays shorts when funding is positive. Getting this backwards
    inverts the entire strategy (007's own sec 185-190 bug, new costume).
    """
    n = 10
    flat = np.full(n, 100.0)
    df = pl.DataFrame(
        {
            "spot_close": flat,
            "perp_close": flat,
            "funding": np.full(n, 0.0005),
        }
    )
    result = df.select(
        bl.paired_log_return(
            pl.col("spot_close"), pl.col("perp_close"), pl.col("funding")
        ).alias("paired_return")
    )["paired_return"].to_numpy()

    assert np.all(result[1:] > 0)
    np.testing.assert_allclose(result[1:], 0.0005, atol=1e-12)


# --------------------------------------------------------------------------
# 4. hysteresis holds through the band
# --------------------------------------------------------------------------


def test_hysteresis_holds_through_the_band() -> None:
    """Carry dipping between THETA_OUT and THETA_IN must NOT close an open
    position, and must NOT open a closed one.
    """
    mid_band = (bl.THETA_IN + bl.THETA_OUT) / 2.0
    assert bl.THETA_OUT < mid_band < bl.THETA_IN

    df = pl.DataFrame(
        {
            "carry": [mid_band, mid_band],
            "liquid": [True, True],
            "held": [True, False],
        }
    )
    result = df.select(
        bl.qualifies(pl.col("carry"), pl.col("liquid"), pl.col("held")).alias("q")
    )["q"].to_list()

    assert result[0] is True, "already-held position must stay open inside the band"
    assert result[1] is False, "flat position must not open inside the band"

    # Sanity checks at the edges, so the band test isn't accidentally vacuous.
    above_in = bl.THETA_IN + 1e-6
    below_out = bl.THETA_OUT - 1e-6
    df_edges = pl.DataFrame(
        {
            "carry": [above_in, above_in, below_out, below_out],
            "liquid": [True, True, True, True],
            "held": [True, False, True, False],
        }
    )
    edges = df_edges.select(
        bl.qualifies(pl.col("carry"), pl.col("liquid"), pl.col("held")).alias("q")
    )["q"].to_list()
    assert edges == [True, True, False, False]


# --------------------------------------------------------------------------
# 5. fills are next-bar-open (no same-bar fills)
# --------------------------------------------------------------------------


def test_fills_are_next_bar_open() -> None:
    """A decision at bar t must never transact against bar t's own return --
    only the label the position construction path aligns to.

    Uses `add_trade_features`' own "fwd_paired_return_1" column: its value
    at row t must equal "paired_log_return" at row t+1, i.e. the return
    materializing strictly AFTER the decision bar, never the same bar's own
    just-realized return.
    """
    n = 20
    rng = np.random.default_rng(2)
    dt = _dt_col(n)
    spot = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    perp = spot * (1 + rng.normal(0, 0.001, n))
    funding = rng.normal(0.0001, 0.00002, n)

    panel = pl.DataFrame(
        {
            "datetime": dt,
            "symbol": ["BTCUSDT"] * n,
            "spot_close": spot,
            "perp_close": perp,
            "spot_dollar_volume": np.full(n, 1e8),
            "perp_dollar_volume": np.full(n, 1e8),
            "funding_rate": funding,
            "basis_premium": np.zeros(n),
        }
    )
    featured = bl.add_trade_features(panel)

    paired = featured["paired_log_return"].to_numpy()
    fwd = featured["fwd_paired_return_1"].to_numpy()

    # fwd[t] must equal paired[t+1] (the NEXT bar's own realized return),
    # and must NOT equal paired[t] (the current/decision bar's return).
    np.testing.assert_allclose(fwd[:-1], paired[1:], equal_nan=True)
    same_bar = np.isclose(fwd[:-2], paired[:-2])
    assert not same_bar.all(), (
        "fwd_paired_return_1 must not collapse to the same-bar return"
    )


# --------------------------------------------------------------------------
# 6. round-turn cost pins the sec 3.4 arithmetic
# --------------------------------------------------------------------------


def test_round_turn_bp_is_34() -> None:
    """Pins the derived constants so a later edit to a fee constant can't
    silently move theta_in/theta_out without a test failing.
    """
    assert bl.ROUND_TURN_BP == pytest.approx(34.0)
    assert bl.THETA_IN == pytest.approx(7.555555555555556e-05, rel=1e-9)
    assert bl.THETA_OUT == pytest.approx(bl.THETA_IN / 2.0, rel=1e-12)
    assert bl.THETA_OUT == pytest.approx(3.7777777777777777e-05, rel=1e-9)


# --------------------------------------------------------------------------
# Extra: the two-leg cost accounting matches the blended
# research.add_portfolio_costs cross-check (sec 7.3's "one genuine gap").
# --------------------------------------------------------------------------


def test_two_leg_costs_match_blended_add_portfolio_costs_crosscheck() -> None:
    n = 10
    trade_frame = pl.DataFrame(
        {
            "datetime": _dt_col(n),
            "trade_log_return": np.random.default_rng(3).normal(0, 0.001, n),
            "turnover": np.random.default_rng(4).uniform(0, 1, n),
        }
    )
    explicit = bl.apply_two_leg_costs(trade_frame)
    blended_taker_fee = (bl.SPOT_TAKER_BP + bl.PERP_TAKER_BP) * 1e-4
    blended_slippage = 2 * bl.SLIPPAGE_BP * 1e-4
    crosscheck = research.add_portfolio_costs(
        trade_frame, taker_fee=blended_taker_fee, slippage=blended_slippage
    )
    np.testing.assert_allclose(
        explicit["trade_log_return_net"].to_numpy(),
        crosscheck["trade_log_return_net"].to_numpy(),
    )
