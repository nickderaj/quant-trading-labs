"""Notebook 11d Phase 2 -- Gate MB-E: the same breakout construction as
Gate MB, on our 69-ticker commodity-equity universe (NEXT_PROMPT.md sec 7).

fires_if (phase_6_11a_results.json gates["MB-E"], transcribed verbatim):
"same construction as MB; NOT eligible for the institutionally-fundable
flag (survivorship-unknown universe cannot support an absolute-performance
claim)". dsr_counts["MB-E"] = 4 (offsets only, no cost-multiplier sweep) --
so unlike Gate MB, the gate criterion here is evaluated at the single 1x
cost level across the 4 offsets; 2x/3x are still reported for the same
robustness framing sec 7 asks for, but do not enter this gate's n_trials.

Survivorship caveat, mandatory on every number this notebook reports: our
69 yfinance daily tickers are current-listing commodity-sector equities,
ETFs and futures, not a survivor-safe universe (no delisted names are
present because none could be, by construction of how the tickers were
chosen) -- exactly the same caveat NEXT_PROMPT.md sec 7 attaches to the
external programme's own 39-symbol legacy proxy registry.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import run_phase_0_11d_data_and_signals as P0
import run_phase_1_11d_gate_mb as P1
import spread_lib11 as S11

import research

research.set_seed(0)

ORIGIN_OFFSETS = [0, 7, 14, 21]
COST_MULTIPLIERS = [1, 2, 3]
N_TRIALS_MB_E = 4  # 4 offsets, equity universe, no cost-multiplier sweep
ANNUALIZED_RATE = float(np.sqrt(252))


def assert_matches_preregistration() -> None:
    prereg = json.loads(Path("src/research/tmp/phase_6_11a_results.json").read_text())
    assert prereg["dsr_counts"]["MB-E"]["n_trials"] == N_TRIALS_MB_E, (
        "Gate MB-E n_trials drifted from 11a's pre-registration"
    )


def main() -> None:
    assert_matches_preregistration()
    equity = P0.build_equity_universe()
    _, regime = P0.build_regime_series(equity, P0.EQUITY_REGIME_REF, min_confirm=2)

    by_offset_by_cost = {}
    for offset in ORIGIN_OFFSETS:
        by_cost = {}
        for mult in COST_MULTIPLIERS:
            book = P1.build_book(equity, regime, offset, mult)
            by_cost[f"cost_{mult}x"] = S11.breakout_book_metrics(book)
            if mult == 1:
                by_cost["_book_1x"] = book
        by_offset_by_cost[f"offset_{offset}"] = by_cost

    sharpes_1x = [
        by_offset_by_cost[f"offset_{o}"]["cost_1x"]["sharpe"] for o in ORIGIN_OFFSETS
    ]
    positive_every_offset = all(s > 0 for s in sharpes_1x)

    book0_1x = by_offset_by_cost["offset_0"]["_book_1x"]
    trades0 = book0_1x["trades"]
    ret_eq0 = np.array([t["ret_eq"] for t in trades0])
    blocks0 = S11.trade_blocks(np.array([t["exit_date"] for t in trades0]))
    floor0 = S11.noise_floor(ret_eq0, blocks0)
    ci_excludes_zero = bool(floor0["ci_return"][0] > 0 or floor0["ci_return"][1] < 0)

    n_trades0 = len(trades0)
    dsr = (
        research.deflated_sharpe_prob(
            sharpes_1x[0] / ANNUALIZED_RATE, n_trials=N_TRIALS_MB_E, n_obs=n_trades0
        )
        if n_trades0 > 1
        else float("nan")
    )
    dsr_fires = bool(dsr > 0.95) if np.isfinite(dsr) else False
    fires = positive_every_offset and ci_excludes_zero and dsr_fires

    out = {
        "gate": "MB-E",
        "n_trials": N_TRIALS_MB_E,
        "origin_offsets": ORIGIN_OFFSETS,
        "cost_multipliers_reported": COST_MULTIPLIERS,
        "by_offset": {
            off: {
                cost: m
                for cost, m in by_offset_by_cost[off].items()
                if not cost.startswith("_")
            }
            for off in by_offset_by_cost
        },
        "sharpes_1x_by_offset": dict(
            zip([f"offset_{o}" for o in ORIGIN_OFFSETS], sharpes_1x, strict=True)
        ),
        "positive_every_offset": positive_every_offset,
        "noise_floor_offset0_1x": floor0,
        "ci_excludes_zero": ci_excludes_zero,
        "deflated_sharpe_prob": dsr,
        "dsr_fires": dsr_fires,
        "gate_fires": fires,
        "fundable_flag_eligible": False,
        "fundable_flag_reason": "survivorship-unknown universe cannot support an absolute-performance claim",
        "n_trades_offset0_1x": n_trades0,
        "survivorship_caveat": (
            "current-listing yfinance universe; no delisted names present by "
            "construction; every number here inherits the same "
            "survivorship-unknown caveat as the external programme's own "
            "39-symbol legacy proxy registry"
        ),
    }
    Path("src/research/tmp/phase_2_11d_results.json").write_text(
        json.dumps(out, indent=2)
    )
    print(
        f"Gate MB-E: fires={fires} (fundable_flag_eligible=False, always) "
        f"sharpes_1x={[round(s, 3) for s in sharpes_1x]} "
        f"noise_floor=[{floor0['ci_return'][0]:.3f},{floor0['ci_return'][1]:.3f}] "
        f"dsr={dsr:.3f} n_trades={n_trades0} "
        f"maxDD_1x_offset0={by_offset_by_cost['offset_0']['cost_1x']['max_drawdown']:.4f}"
    )


if __name__ == "__main__":
    main()
