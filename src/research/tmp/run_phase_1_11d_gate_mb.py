"""Notebook 11d Phase 1 -- Gate MB: the breakout setup on our 30 crypto
perpetuals (NEXT_PROMPT.md sec 7). Full cost model, 1x/2x/3x cost stress,
4 origin offsets, three-way risk gate, DSR, includes delisted pairs.

fires_if (phase_6_11a_results.json gates.MB, transcribed verbatim): "net
Sharpe > 0 at every offset AND DSR > 0.95, full cost model, 1x/2x/3x cost
stress, walk-forward, three-way risk gate, includes delisted pairs" -- the
1x-cost book is the primary book the Sharpe/CI/DSR criteria are evaluated
against; 2x/3x are reported alongside as the cost-stress robustness check
NEXT_PROMPT.md sec 7 asks for. Unlike Gate TS/BF/SCR/VA, there is no
external-programme comparator book to pair a bootstrap difference against
here -- their own Phase -1 cost study reports trade-count/mean-R, not an
equity path -- so "CI excludes zero" is operationalized as `noise_floor`'s
own-book bootstrap CI (same primitive 11a Phase 2 built for exactly this:
"bootstrap the control alone to get its own interval"), and the 1x-vs-3x
cost-stress delta is reported separately via the paired block bootstrap.
This substitution is a judgement call, disclosed here rather than silently
assumed.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import run_phase_0_11d_data_and_signals as P0
import spread_lib11 as S11

import research

research.set_seed(0)

ORIGIN_OFFSETS = [0, 7, 14, 21]
COST_MULTIPLIERS = [1, 2, 3]
BASE_COST_BPS_PER_SIDE = 0.0004 + 0.0001  # taker_fee + slippage, repo convention
N_TRIALS_MB = 12  # 3 cost multipliers x 4 offsets
ANNUALIZED_RATE = float(np.sqrt(252))


def trim(f: dict, offset: int) -> dict:
    if offset == 0:
        return f
    return {k: v[offset:] for k, v in f.items()}


def build_book(
    universe: dict[str, dict],
    regime: dict[str, np.ndarray],
    offset: int,
    cost_mult: int,
) -> dict:
    trimmed = {s: trim(f, offset) for s, f in universe.items()}
    trimmed_regime = {s: r[offset:] if offset else r for s, r in regime.items()}
    p = S11.BreakoutParams()
    return S11.simulate_breakout_book(
        trimmed, trimmed_regime, p, BASE_COST_BPS_PER_SIDE * cost_mult
    )


def assert_matches_preregistration() -> None:
    prereg = json.loads(Path("src/research/tmp/phase_6_11a_results.json").read_text())
    assert prereg["dsr_counts"]["MB"]["n_trials"] == N_TRIALS_MB, (
        "Gate MB n_trials drifted from 11a's pre-registration"
    )


def main() -> None:
    assert_matches_preregistration()
    crypto = P0.build_crypto_universe()
    _, regime = P0.build_regime_series(crypto, P0.CRYPTO_REGIME_REF, min_confirm=1)

    by_offset_by_cost = {}
    for offset in ORIGIN_OFFSETS:
        by_cost = {}
        for mult in COST_MULTIPLIERS:
            book = build_book(crypto, regime, offset, mult)
            metrics = S11.breakout_book_metrics(book)
            by_cost[f"cost_{mult}x"] = metrics
            if mult == 1:
                by_cost["_book_1x"] = book  # kept only in-memory, not serialized
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

    book0_3x = build_book(crypto, regime, 0, 3)
    trades0_3x = book0_3x["trades"]
    cost_stress = S11.paired_block_bootstrap(
        ret_eq0,
        blocks0,
        np.array([t["ret_eq"] for t in trades0_3x]),
        S11.trade_blocks(np.array([t["exit_date"] for t in trades0_3x])),
    )

    n_trades0 = len(trades0)
    dsr = (
        research.deflated_sharpe_prob(
            sharpes_1x[0] / ANNUALIZED_RATE, n_trials=N_TRIALS_MB, n_obs=n_trades0
        )
        if n_trades0 > 1
        else float("nan")
    )
    dsr_fires = bool(dsr > 0.95) if np.isfinite(dsr) else False

    fires = positive_every_offset and ci_excludes_zero and dsr_fires
    fundable = (
        all(s > 0.5 for s in sharpes_1x)
        and dsr_fires
        and abs(by_offset_by_cost["offset_0"]["cost_1x"]["max_drawdown"]) <= 0.25
    )

    delisted_trades = [t for t in trades0 if t["symbol"] in P0.DELISTED_CRYPTO]

    out = {
        "gate": "MB",
        "n_trials": N_TRIALS_MB,
        "origin_offsets": ORIGIN_OFFSETS,
        "cost_multipliers": COST_MULTIPLIERS,
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
        "cost_stress_1x_vs_3x_offset0": cost_stress,
        "deflated_sharpe_prob": dsr,
        "dsr_fires": dsr_fires,
        "gate_fires": fires,
        "fundable_flag": fundable,
        "n_trades_offset0_1x": n_trades0,
        "n_delisted_trades_offset0_1x": len(delisted_trades),
        "delisted_symbols_traded": sorted({t["symbol"] for t in delisted_trades}),
        "delisted_pnl_sum": float(sum(t["pnl"] for t in delisted_trades)),
    }
    Path("src/research/tmp/phase_1_11d_results.json").write_text(
        json.dumps(out, indent=2)
    )
    print(
        f"Gate MB: fires={fires} fundable={fundable} sharpes_1x={[round(s, 3) for s in sharpes_1x]} "
        f"noise_floor=[{floor0['ci_return'][0]:.3f},{floor0['ci_return'][1]:.3f}] "
        f"dsr={dsr:.3f} n_trades={n_trades0} delisted_trades={len(delisted_trades)} "
        f"maxDD_1x_offset0={by_offset_by_cost['offset_0']['cost_1x']['max_drawdown']:.4f}"
    )


if __name__ == "__main__":
    main()
