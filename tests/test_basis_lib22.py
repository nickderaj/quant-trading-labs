"""Network-free pinned tests for notebook 022's library
(phase_0_22_preregistration.json's implementation_notes). Every test builds
small synthetic polars frames directly, no Hyperliquid/Binance fetch
involved.
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

import basis_lib20 as bl20
import basis_lib22 as bl22

import research


def _dt_col(
    n: int, start: datetime | None = None, step_hours: int = 8
) -> list[datetime]:
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    return [start + timedelta(hours=step_hours * i) for i in range(n)]


# --------------------------------------------------------------------------
# 1. ROUND_TURN_BP_HL == 23.0 and its derivation
# --------------------------------------------------------------------------


def test_round_turn_bp_hl_is_23() -> None:
    assert bl22.ROUND_TURN_BP_HL == pytest.approx(23.0)
    expected = 2 * (
        (bl22.PERP_TAKER_BP + bl22.SLIPPAGE_BP) + (bl22.HL_TAKER_BP + bl22.SLIPPAGE_BP)
    )
    assert bl22.ROUND_TURN_BP_HL == pytest.approx(expected)
    # cheaper than 020's Bybit round turn -- HL's taker fee (4.5bp) beats
    # Bybit's (5.5bp), same fully-costed convention.
    assert bl22.ROUND_TURN_BP_HL < bl20.ROUND_TURN_BP_XV


# --------------------------------------------------------------------------
# 2. theta derivations
# --------------------------------------------------------------------------


def test_theta_derivations() -> None:
    assert bl22.THETA_IN_HL_FAST == pytest.approx(23.0 / 45 * 1e-4, rel=1e-12)
    assert bl22.THETA_IN_HL_SLOW == pytest.approx(23.0 / 90 * 1e-4, rel=1e-12)
    for theta_in, theta_out in [
        (bl22.THETA_IN_HL_FAST, bl22.THETA_OUT_HL_FAST),
        (bl22.THETA_IN_HL_SLOW, bl22.THETA_OUT_HL_SLOW),
    ]:
        assert theta_out == pytest.approx(theta_in / 2.0, rel=1e-12)


# --------------------------------------------------------------------------
# 3. sign convention: A=Hyperliquid, B=Binance, matches bl20's own generic
#    identity test but with this file's own naming
# --------------------------------------------------------------------------


def test_hlvenue_paired_return_sign_convention() -> None:
    n = 30
    rng = np.random.default_rng(8)
    price_path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    hl_funding = rng.normal(0.0001, 0.00003, n)
    binance_funding = rng.normal(0.00005, 0.00003, n)
    df = pl.DataFrame(
        {
            "hl_close": price_path,
            "binance_close": price_path,  # identical price path on both venues
            "hl_funding_rate": hl_funding,
            "binance_funding_rate": binance_funding,
        }
    )
    result = df.select(
        bl20.xvenue_paired_log_return(
            pl.col("hl_close"),
            pl.col("binance_close"),
            pl.col("hl_funding_rate"),
            pl.col("binance_funding_rate"),
        ).alias("r")
    )["r"].to_numpy()
    # with identical price paths, the paired return is exactly the funding
    # spread short-HL/long-Binance collects: hl_funding - binance_funding
    expected = hl_funding - binance_funding
    np.testing.assert_allclose(result[1:], expected[1:], atol=1e-12)


# --------------------------------------------------------------------------
# 4. HL_ALWAYSON holds every liquid symbol every bar, direction=sign(carry),
#    no floor, no position cap (build_xvenue_book_weights' timed=False path)
# --------------------------------------------------------------------------


def test_hl_alwayson_holds_every_liquid_symbol() -> None:
    dts = _dt_col(3)
    rows = []
    for i, dt in enumerate(dts):
        for sym, sign in [("AAA", 1.0), ("BBB", -1.0), ("CCC", 1.0)]:
            rows.append(
                {"datetime": dt, "symbol": sym, "carry": sign * 1e-6, "liquid": True}
            )
    panel = pl.DataFrame(rows)
    weights = bl22.build_hl_book_weights(panel, "HL_ALWAYSON")

    for dt in dts:
        bar = weights.filter(pl.col("datetime") == dt).sort("symbol")
        w = dict(zip(bar["symbol"].to_list(), bar["weight"].to_list(), strict=True))
        assert w["AAA"] == pytest.approx(1 / 3)
        assert w["BBB"] == pytest.approx(-1 / 3)
        assert w["CCC"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# 5. cost frac matches ROUND_TURN_BP_HL, and apply_hlvenue_costs_at
#    reproduces apply_hlvenue_costs at round_turn_bp=ROUND_TURN_BP_HL
# --------------------------------------------------------------------------


def test_hlvenue_cost_frac_matches_round_turn() -> None:
    n = 10
    trade_frame = pl.DataFrame(
        {
            "datetime": _dt_col(n),
            "trade_log_return": np.random.default_rng(11).normal(0, 0.001, n),
            "turnover": np.random.default_rng(12).uniform(0, 1, n),
        }
    )
    explicit = bl22.apply_hlvenue_costs(trade_frame)

    blended_taker_fee = (bl22.HL_TAKER_BP + bl22.PERP_TAKER_BP) * 1e-4
    blended_slippage = 2 * bl22.SLIPPAGE_BP * 1e-4
    crosscheck = research.add_portfolio_costs(
        trade_frame, taker_fee=blended_taker_fee, slippage=blended_slippage
    )
    np.testing.assert_allclose(
        explicit["trade_log_return_net"].to_numpy(),
        crosscheck["trade_log_return_net"].to_numpy(),
    )
    cost_frac_used = blended_taker_fee + blended_slippage
    assert cost_frac_used == pytest.approx(bl22.ROUND_TURN_BP_HL / 2 * 1e-4, rel=1e-9)

    at_headline = bl22.apply_hlvenue_costs_at(trade_frame, bl22.ROUND_TURN_BP_HL)
    np.testing.assert_allclose(
        at_headline["trade_log_return_net"].to_numpy(),
        explicit["trade_log_return_net"].to_numpy(),
    )


# --------------------------------------------------------------------------
# 6. load_hlvenue_panel refuses the holdout
# --------------------------------------------------------------------------


def test_load_hlvenue_panel_refuses_holdout() -> None:
    with pytest.raises(ValueError, match="holdout"):
        bl22.load_hlvenue_panel(
            symbols=["BTCUSDT"], end_date=datetime(2025, 8, 1, tzinfo=UTC)
        )


# --------------------------------------------------------------------------
# 7. MDE closed form reproduces the planning-time estimate
#    (NEXT_PROMPT.md Candidate 1: MDE annualized Sharpe ~= 1.98 at
#    n_obs=2190, INTERVAL="8h")
# --------------------------------------------------------------------------


def test_mde_annualized_sharpe_matches_planning_estimate() -> None:
    rate = research.sharpe_to_annualized_rate("8h")
    mde = bl22.mde_annualized_sharpe(2190, rate)
    assert mde == pytest.approx(1.98, abs=0.01)
    # more observations -> a smaller (easier to clear) MDE
    assert bl22.mde_annualized_sharpe(4380, rate) < mde
