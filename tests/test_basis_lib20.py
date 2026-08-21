"""The twelve tests NEXT_PROMPT.md sec 5.7 requires for notebook 020's
library. Network-free: every test builds small synthetic polars frames
directly, no Binance/Bybit fetch involved.
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

import basis_lib18 as bl18
import basis_lib20 as bl20

import research


def _dt_col(
    n: int, start: datetime | None = None, step_hours: int = 8
) -> list[datetime]:
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    return [start + timedelta(hours=step_hours * i) for i in range(n)]


# --------------------------------------------------------------------------
# 1. ROUND_TURN_BP_XV == 25.0 and its derivation
# --------------------------------------------------------------------------


def test_round_turn_bp_xv_is_25() -> None:
    assert bl20.ROUND_TURN_BP_XV == pytest.approx(25.0)
    expected = 2 * (
        (bl20.PERP_TAKER_BP + bl20.SLIPPAGE_BP)
        + (bl20.BYBIT_TAKER_BP + bl20.SLIPPAGE_BP)
    )
    assert bl20.ROUND_TURN_BP_XV == pytest.approx(expected)


# --------------------------------------------------------------------------
# 2. theta derivations
# --------------------------------------------------------------------------


def test_theta_derivations() -> None:
    assert bl20.THETA_IN_SLOW == pytest.approx(34 / 90 * 1e-4, rel=1e-12)
    assert bl20.THETA_IN_XV == pytest.approx(25 / 45 * 1e-4, rel=1e-12)
    for theta_in, theta_out in [
        (bl20.THETA_IN, bl20.THETA_OUT),
        (bl20.THETA_IN_SLOW, bl20.THETA_OUT_SLOW),
        (bl20.THETA_IN_XV, bl20.THETA_OUT_XV),
        (bl20.THETA_IN_XV_SLOW, bl20.THETA_OUT_XV_SLOW),
    ]:
        assert theta_out == pytest.approx(theta_in / 2.0, rel=1e-12)
    # sec 5.3's expected coincidence: THETA_IN_SLOW == 018's THETA_OUT exactly.
    assert bl20.THETA_IN_SLOW == pytest.approx(bl18.THETA_OUT, rel=1e-12)


# --------------------------------------------------------------------------
# 3. slow carry is causal
# --------------------------------------------------------------------------


def test_slow_carry_is_causal() -> None:
    rng = np.random.default_rng(0)
    n = 200
    base_funding = rng.normal(0.0001, 0.00005, n)

    df_base = pl.DataFrame({"funding": base_funding})
    carry_base = df_base.select(
        bl18.carry_estimate(pl.col("funding"), periods=bl20.SLOW_CARRY_HALF_LIFE).alias(
            "carry"
        )
    )["carry"].to_numpy()

    perturb_idx = 150
    perturbed = base_funding.copy()
    perturbed[perturb_idx] += 0.05

    df_pert = pl.DataFrame({"funding": perturbed})
    carry_pert = df_pert.select(
        bl18.carry_estimate(pl.col("funding"), periods=bl20.SLOW_CARRY_HALF_LIFE).alias(
            "carry"
        )
    )["carry"].to_numpy()

    np.testing.assert_array_equal(carry_base[:perturb_idx], carry_pert[:perturb_idx])
    assert not np.allclose(carry_base[perturb_idx:], carry_pert[perturb_idx:])


# --------------------------------------------------------------------------
# 4. diversification floor stands down
# --------------------------------------------------------------------------


def _floor_panel() -> pl.DataFrame:
    """One bar with 2 qualifying symbols (A, B above theta_in; C below),
    one bar with 3 qualifying (A, B, C all above theta_in).
    """
    dts = _dt_col(2)
    rows = []
    above = bl20.THETA_IN + 1e-5
    below = bl20.THETA_IN - 1e-5
    # bar 0: only A, B qualify (2 < N_MIN=3)
    rows += [
        {"datetime": dts[0], "symbol": "A", "carry": above, "liquid": True},
        {"datetime": dts[0], "symbol": "B", "carry": above, "liquid": True},
        {"datetime": dts[0], "symbol": "C", "carry": below, "liquid": True},
    ]
    # bar 1: A, B, C all qualify (3 >= N_MIN=3)
    rows += [
        {"datetime": dts[1], "symbol": "A", "carry": above, "liquid": True},
        {"datetime": dts[1], "symbol": "B", "carry": above, "liquid": True},
        {"datetime": dts[1], "symbol": "C", "carry": above, "liquid": True},
    ]
    return pl.DataFrame(rows)


def test_diversification_floor_stands_down() -> None:
    dts = _dt_col(2)
    panel = _floor_panel()
    weights = bl20.build_book_weights_v2(panel, timed=True, n_min=3)

    bar0 = weights.filter(pl.col("datetime") == dts[0])
    assert set(bar0["weight"].to_list()) == {0.0}, (
        "2 qualifying < N_MIN=3 must stand down entirely"
    )

    bar1 = weights.filter(pl.col("datetime") == dts[1])
    bar1_weights = dict(
        zip(bar1["symbol"].to_list(), bar1["weight"].to_list(), strict=True)
    )
    assert bar1_weights == {
        "A": pytest.approx(1 / 3),
        "B": pytest.approx(1 / 3),
        "C": pytest.approx(1 / 3),
    }


# --------------------------------------------------------------------------
# 5. floor does not fill up
# --------------------------------------------------------------------------


def test_floor_does_not_fill_up() -> None:
    dt = _dt_col(1)[0]
    above = bl20.THETA_IN + 1e-5
    panel = pl.DataFrame(
        [
            {"datetime": dt, "symbol": s, "carry": above, "liquid": True}
            for s in ["A", "B", "C", "D", "E"]
        ]
    )
    weights = bl20.build_book_weights_v2(panel, timed=True, n_min=3)
    assert len(weights) == 5
    for w in weights["weight"].to_list():
        assert w == pytest.approx(1 / 5)


# --------------------------------------------------------------------------
# 6. stand-down resets held and charges turnover; re-entry uses theta_in
# --------------------------------------------------------------------------


def test_floor_stand_down_resets_held_and_charges_turnover() -> None:
    dts = _dt_col(3)
    above_in = bl20.THETA_IN + 1e-5
    mid_band = (bl20.THETA_IN + bl20.THETA_OUT) / 2.0  # qualifies to HOLD, not to ENTER
    rows = []
    # bar0: A,B,C all above theta_in -> all held, n=3
    for s in ["A", "B", "C"]:
        rows.append(
            {"datetime": dts[0], "symbol": s, "carry": above_in, "liquid": True}
        )
    # bar1: only A stays in the hold band (mid_band would hold if already held,
    # but NOT open a flat position) -- B,C drop to a value that doesn't qualify
    # to hold either, so only A would normally hold... to isolate the floor,
    # make B, C drop below theta_out entirely (so pre-floor candidates = {A}),
    # which is < N_MIN=3 -> the whole book (including A) stands down.
    rows.append({"datetime": dts[1], "symbol": "A", "carry": mid_band, "liquid": True})
    rows.append({"datetime": dts[1], "symbol": "B", "carry": 0.0, "liquid": True})
    rows.append({"datetime": dts[1], "symbol": "C", "carry": 0.0, "liquid": True})
    # bar2: all three back above theta_in -> re-entry
    for s in ["A", "B", "C"]:
        rows.append(
            {"datetime": dts[2], "symbol": s, "carry": above_in, "liquid": True}
        )
    panel = pl.DataFrame(rows)

    weights = bl20.build_book_weights_v2(panel, timed=True, n_min=3)
    bar1 = weights.filter(pl.col("datetime") == dts[1])
    assert set(bar1["weight"].to_list()) == {0.0}, (
        "candidates={A} < N_MIN=3 must stand down A too"
    )

    bar2 = weights.filter(pl.col("datetime") == dts[2])
    assert all(w == pytest.approx(1 / 3) for w in bar2["weight"].to_list())

    # turnover: bar0->bar1 every symbol closes (1/3 -> 0), bar1->bar2 every
    # symbol re-enters (0 -> 1/3) at theta_in (all bar2 carries are above_in,
    # not just above theta_out, so this is consistent with re-entry needing
    # the IN threshold, not the lower OUT threshold).
    turnover = research.portfolio_turnover(weights)
    t1 = turnover.filter(pl.col("datetime") == dts[1])["turnover"][0]
    t2 = turnover.filter(pl.col("datetime") == dts[2])["turnover"][0]
    assert t1 == pytest.approx(1.0)  # 3 symbols x |1/3 - 0|
    assert t2 == pytest.approx(1.0)  # 3 symbols x |0 - 1/3|


# --------------------------------------------------------------------------
# 7. n_min=1 reproduces 018 weights exactly (load-bearing)
# --------------------------------------------------------------------------


def test_n_min_1_reproduces_018_weights() -> None:
    rng = np.random.default_rng(7)
    dts = _dt_col(40)
    symbols = ["A", "B", "C", "D", "E"]
    rows = []
    for dt in dts:
        for sym in symbols:
            carry = float(rng.normal(5e-5, 4e-5))
            liquid = bool(rng.uniform() > 0.1)
            rows.append(
                {"datetime": dt, "symbol": sym, "carry": carry, "liquid": liquid}
            )
    panel = pl.DataFrame(rows)

    v2 = bl20.build_book_weights_v2(panel, timed=True, n_min=1)
    v18 = bl18.build_book_weights(panel, timed=True)

    v2_sorted = v2.sort(["datetime", "symbol"])
    v18_sorted = v18.sort(["datetime", "symbol"])
    assert v2_sorted["symbol"].to_list() == v18_sorted["symbol"].to_list()
    np.testing.assert_array_equal(
        v2_sorted["weight"].to_numpy(), v18_sorted["weight"].to_numpy()
    )

    # also true for the always-on (timed=False) path
    v2_always = bl20.build_book_weights_v2(panel, timed=False, n_min=1).sort(
        ["datetime", "symbol"]
    )
    v18_always = bl18.build_book_weights(panel, timed=False).sort(
        ["datetime", "symbol"]
    )
    np.testing.assert_array_equal(
        v2_always["weight"].to_numpy(), v18_always["weight"].to_numpy()
    )


# --------------------------------------------------------------------------
# 8. xvenue paired return identity
# --------------------------------------------------------------------------


def test_xvenue_paired_return_identity() -> None:
    n = 30
    rng = np.random.default_rng(8)
    price_path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    a_funding = rng.normal(0.0001, 0.00003, n)
    b_funding = rng.normal(0.00005, 0.00003, n)
    df = pl.DataFrame(
        {
            "a_close": price_path,
            "b_close": price_path,  # identical price path on both venues
            "a_funding": a_funding,
            "b_funding": b_funding,
        }
    )
    result = df.select(
        bl20.xvenue_paired_log_return(
            pl.col("a_close"),
            pl.col("b_close"),
            pl.col("a_funding"),
            pl.col("b_funding"),
        ).alias("r")
    )["r"].to_numpy()
    expected = a_funding - b_funding
    np.testing.assert_allclose(result[1:], expected[1:], atol=1e-12)


# --------------------------------------------------------------------------
# 9. sign flip is an exit+re-entry, not a free flip
# --------------------------------------------------------------------------


def test_xvenue_sign_flip() -> None:
    dts = _dt_col(4)
    theta_in = bl20.THETA_IN_XV
    strong_pos = theta_in + 2e-5
    strong_neg = -(theta_in + 2e-5)
    rows = [
        {"datetime": dts[0], "symbol": "X", "carry": strong_pos, "liquid": True},
        {"datetime": dts[1], "symbol": "X", "carry": strong_pos, "liquid": True},
        {"datetime": dts[2], "symbol": "X", "carry": strong_neg, "liquid": True},
        {"datetime": dts[3], "symbol": "X", "carry": strong_neg, "liquid": True},
    ]
    panel = pl.DataFrame(rows)
    weights = bl20.build_xvenue_book_weights(panel, timed=True, n_min=1).sort(
        "datetime"
    )
    w = weights["weight"].to_list()

    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(1.0)
    assert w[2] == pytest.approx(-1.0), (
        "sign flip must flip the weight's sign, not zero it out"
    )
    assert w[3] == pytest.approx(-1.0)

    # the flip bar's turnover is 2x a single round-turn (close 1.0 + open 1.0
    # in one step), not a free flip and not two separate charged bars.
    turnover = research.portfolio_turnover(weights)
    t_flip = turnover.filter(pl.col("datetime") == dts[2])["turnover"][0]
    assert t_flip == pytest.approx(2.0)
    t_hold = turnover.filter(pl.col("datetime") == dts[1])["turnover"][0]
    assert t_hold == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 10. Bybit funding resample to 8h
# --------------------------------------------------------------------------


def test_bybit_funding_resample_to_8h() -> None:
    # A Binance-style 8h stamp grid.
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    # 1h series: 8 payments of 0.0001 each should sum to 0.0008 in the
    # bucket ending at t0+8h.
    hourly_dts = [t0 + timedelta(hours=h) for h in range(1, 9)]
    hourly = pl.DataFrame(
        {
            "datetime": [d.replace(tzinfo=None) for d in hourly_dts],
            "funding_rate": [0.0001] * 8,
        }
    )
    resampled_1h = bl20.resample_funding_to_8h(hourly, funding_interval_min=60)
    bucket = resampled_1h.filter(
        pl.col("datetime") == t0.replace(tzinfo=None) + timedelta(hours=8)
    )
    assert len(bucket) == 1
    assert bucket["funding_rate"][0] == pytest.approx(0.0008)
    assert bucket["n_payments"][0] == 8

    # 4h series: 2 payments should sum into the same 8h bucket.
    four_hourly_dts = [t0 + timedelta(hours=4), t0 + timedelta(hours=8)]
    four_hourly = pl.DataFrame(
        {
            "datetime": [d.replace(tzinfo=None) for d in four_hourly_dts],
            "funding_rate": [0.0002, 0.0003],
        }
    )
    resampled_4h = bl20.resample_funding_to_8h(four_hourly, funding_interval_min=240)
    bucket4 = resampled_4h.filter(
        pl.col("datetime") == t0.replace(tzinfo=None) + timedelta(hours=8)
    )
    assert len(bucket4) == 1
    assert bucket4["funding_rate"][0] == pytest.approx(0.0005)
    assert bucket4["n_payments"][0] == 2

    # 6h series (non-divisor of 480 minutes) must be rejected, not interpolated.
    six_hourly = pl.DataFrame(
        {
            "datetime": [t0.replace(tzinfo=None) + timedelta(hours=6)],
            "funding_rate": [0.0001],
        }
    )
    with pytest.raises(ValueError, match="does not divide"):
        bl20.resample_funding_to_8h(six_hourly, funding_interval_min=360)


# --------------------------------------------------------------------------
# 11. xvenue cost fraction matches ROUND_TURN_BP_XV/2, cross-checked
# --------------------------------------------------------------------------


def test_xvenue_cost_frac_matches_round_turn() -> None:
    n = 10
    trade_frame = pl.DataFrame(
        {
            "datetime": _dt_col(n),
            "trade_log_return": np.random.default_rng(11).normal(0, 0.001, n),
            "turnover": np.random.default_rng(12).uniform(0, 1, n),
        }
    )
    explicit = bl20.apply_xvenue_costs(trade_frame)

    blended_taker_fee = (bl20.PERP_TAKER_BP + bl20.BYBIT_TAKER_BP) * 1e-4
    blended_slippage = 2 * bl20.SLIPPAGE_BP * 1e-4
    crosscheck = research.add_portfolio_costs(
        trade_frame, taker_fee=blended_taker_fee, slippage=blended_slippage
    )
    np.testing.assert_allclose(
        explicit["trade_log_return_net"].to_numpy(),
        crosscheck["trade_log_return_net"].to_numpy(),
    )

    cost_frac_used = blended_taker_fee + blended_slippage
    assert cost_frac_used == pytest.approx(bl20.ROUND_TURN_BP_XV / 2 * 1e-4, rel=1e-9)


# --------------------------------------------------------------------------
# 12. load_xvenue_panel refuses the holdout
# --------------------------------------------------------------------------


def test_load_xvenue_panel_refuses_holdout() -> None:
    with pytest.raises(ValueError, match="holdout"):
        bl20.load_xvenue_panel(
            symbols=["BTCUSDT"], end_date=datetime(2025, 8, 1, tzinfo=UTC)
        )
