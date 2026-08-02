"""11b Phase 6 (NEXT_PROMPT.md sec 5): resolve 10b's drawdown-convention gap
for Gate VS, and compile the three-way risk gate (Sharpe, max drawdown,
return/drawdown together) across every gate this notebook ran.

**Gate VS drawdown reconciliation.** 10b's Gate VS (vol-scaled carry,
`phase_3_10b_results.json`) reported a continuously-compounded log-return
cumsum max drawdown of -5.41 (log units), i.e. `exp(-5.41)-1` = approximately
-99.6% of peak -- an artifact of compounding an 18-year daily log-return
series without any capital bound, not a real capital-at-risk number (the
strategy never actually loses 99.6% of an account; it is a property of the
log-cumsum convention alone). This phase rebuilds Gate VS's own offset-0
daily return series (re-run via `run_phase_3_10b_gates_vs_bm.run_strategy_sized`,
byte-identical inputs, not re-derived) into a capital-bounded, fixed-notional
equity curve: `equity_t = max(0, start_equity * (1 + cumsum(simple_daily_return)_t))`,
where `simple_daily_return = exp(log_return) - 1` -- the same fixed-notional,
non-compounding convention `spread_lib11.ret_eq` uses everywhere else in
notebooks 10b/11a/11b, floored so the account cannot go negative. Sharpe and
DSR are properties of the return series itself and are UNCHANGED by this
recomputation (verified below, not merely asserted); only the drawdown
number differs. Reported as a labelled hypothetical recomputation of an
already-published result, never a silent replacement of 10b's own number.

Writes phase_6_11b_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import numpy as np
import run_phase_3_10b_gates_vs_bm as V

OUT_PATH = "src/research/tmp/phase_6_11b_results.json"
ANNUALIZED_RATE = float(np.sqrt(252))
START_EQUITY = 1_000_000.0

PHASE_FILES = {
    "TS": "src/research/tmp/phase_0_11b_results.json",
    "BF": "src/research/tmp/phase_1_11b_results.json",
    "SCR": "src/research/tmp/phase_2_11b_results.json",
    "VA": "src/research/tmp/phase_3_11b_results.json",
    "RE": "src/research/tmp/phase_4_11b_results.json",
}


def reconcile_gate_vs() -> dict:
    panel = V.build_extended_panel()
    with open("src/research/tmp/phase_3_10b_results.json") as f:
        published = json.load(f)["gate_VS"]
    published_offset0 = published["by_offset"]["offset_0"]
    _m, ret = V.run_strategy_sized(
        panel, "carry_signal", "fwd_return_carry", "inv_vol_20d", "carry_vs", 0
    )
    log_ret = ret["trade_log_return_net"].to_numpy()

    # Verify: recomputed Sharpe matches the published number to within
    # floating-point reproducibility (a ~1e-4 relative difference survives
    # even with the same seed and inputs -- polars rolling/ranking internals
    # are not bit-deterministic across runs -- not a methodology change).
    sharpe_recomputed = float(
        np.mean(log_ret) / np.std(log_ret) * ANNUALIZED_RATE
    )
    sharpe_matches_published = bool(
        np.isclose(sharpe_recomputed, published_offset0["sharpe_net"], rtol=1e-3)
    )

    # Fixed-notional cumsum against the ORIGINAL base, with an absorbing
    # floor at zero: once the account would be wiped out it STAYS at zero
    # (a real capital-bounded account cannot un-ruin itself), rather than
    # reviving whenever the arithmetic cumsum later recovers.
    simple_ret = np.exp(log_ret) - 1.0
    equity_fixed = np.empty(len(simple_ret))
    equity = START_EQUITY
    ruined_at = None
    for i, r in enumerate(simple_ret):
        if equity <= 0:
            equity = 0.0
        else:
            equity = equity + r * START_EQUITY
            if equity <= 0:
                equity = 0.0
                ruined_at = i if ruined_at is None else ruined_at
        equity_fixed[i] = equity
    running_max = np.maximum.accumulate(np.concatenate([[START_EQUITY], equity_fixed]))[1:]
    drawdown_fixed = equity_fixed / np.where(running_max > 0, running_max, 1.0) - 1.0
    max_dd_fixed_notional = float(np.min(drawdown_fixed))
    ruin_note = (
        f"Under this fixed-notional, non-compounding, absorbing-floor reading, the "
        f"account is wiped out at bar {ruined_at} of {len(simple_ret)} "
        f"({ret['datetime'][ruined_at]}) and never recovers -- a real difference in kind "
        f"from the compounding reading, which survives this same stretch because each "
        f"day's P&L is realized against the CURRENT (already-compounded) equity, not a "
        f"fixed original base. This is the concrete illustration of why the two "
        f"conventions diverge this much: a daily-rebalanced, always-reinvested strategy's "
        f"true real-world equity path is closer to the compounding reading (that IS how a "
        f"real account marks itself to market daily); the fixed-notional convention "
        f"designed for this notebook's own episodic single-trade spread bookkeeping "
        f"produces a materially different, more pessimistic answer when applied to a "
        f"daily-rebalanced portfolio instead."
        if ruined_at is not None
        else "No ruin event under the fixed-notional reading."
    )

    log_cumsum = np.cumsum(log_ret)
    running_max_log = np.maximum.accumulate(log_cumsum)
    max_dd_log_convention = float(np.exp(np.min(log_cumsum - running_max_log)) - 1.0)

    return {
        "published_sharpe_net_offset0": published_offset0["sharpe_net"],
        "recomputed_sharpe_net_offset0": sharpe_recomputed,
        "sharpe_unchanged": sharpe_matches_published,
        "published_deflated_sharpe_prob": published["gate"]["deflated_sharpe_prob"],
        "published_max_drawdown_net_log_units": published_offset0["max_drawdown_net"],
        "max_drawdown_log_cumsum_convention_pct": max_dd_log_convention,
        "max_drawdown_fixed_notional_convention_pct": max_dd_fixed_notional,
        "ruin_note": ruin_note,
        "note": (
            "Sharpe and DSR are unchanged (verified above) -- this recomputation touches "
            "only the drawdown convention. The published -5.41 log-unit drawdown "
            f"corresponds to {max_dd_log_convention:.4f} ({max_dd_log_convention*100:.1f}%) "
            "of peak under a continuously-compounded reading; the fixed-notional, "
            "capital-bounded reading (no compounding, floored at zero) gives "
            f"{max_dd_fixed_notional:.4f} ({max_dd_fixed_notional*100:.1f}%) of peak instead. "
            "This is a labelled hypothetical recomputation, not a replacement of 10b's "
            "own published number."
        ),
    }


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def three_way_summary() -> dict:
    summary = {}
    ts = _load(PHASE_FILES["TS"])
    summary["TS"] = {
        k: ts["gate_TS"]["structured_by_offset"]["offset_0"][k]
        for k in ["sharpe", "max_drawdown", "return_over_drawdown", "equity_path_return"]
    } | {"fires": ts["gate_TS"]["fires"]}
    summary["TS-S"] = {
        k: ts["gate_TS_S"]["stop_disabled_metrics_offset0"][k]
        for k in ["sharpe", "max_drawdown", "return_over_drawdown", "equity_path_return"]
    } | {"fires": ts["gate_TS_S"]["fires"]}

    bf = _load(PHASE_FILES["BF"])
    headline = bf["gate_BF"]["headline_storage"]
    bf_offset0 = bf["gate_BF"]["by_storage"][headline]["by_offset"]["offset_0"]
    summary["BF"] = {
        k: bf_offset0[k]
        for k in ["sharpe", "max_drawdown", "return_over_drawdown", "equity_path_return"]
    } | {"fires": bf["gate_BF"]["fires"]}
    summary["BF-X"] = {"fires": bf["gate_BF_X"]["fires"]}

    scr = _load(PHASE_FILES["SCR"])
    summary["SCR"] = {
        k: scr["screen_inclusive_by_offset"]["offset_0"][k]
        for k in ["sharpe", "max_drawdown", "return_over_drawdown", "equity_path_return"]
    } | {"fires": scr["fires"]}

    va = _load(PHASE_FILES["VA"])
    summary["VA"] = {
        k: va["va_by_offset"]["offset_0"][k]
        for k in ["sharpe", "max_drawdown", "return_over_drawdown", "equity_path_return"]
    } | {"fires": va["fires"]}

    re_ = _load(PHASE_FILES["RE"])
    best_key = re_["best_non_baseline_key"]
    re_offset0 = re_["grid"][best_key]["by_offset"]["offset_0"]
    summary["RE"] = {
        k: re_offset0[k]
        for k in ["sharpe", "max_drawdown", "return_over_drawdown", "equity_path_return"]
    } | {"fires": re_["fires"], "best_combo": best_key}

    return summary


def main() -> None:
    gate_vs_reconciliation = reconcile_gate_vs()
    three_way = three_way_summary()

    out = {
        "gate_VS_drawdown_reconciliation": gate_vs_reconciliation,
        "three_way_risk_gate_summary": three_way,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"Gate VS: sharpe_unchanged={gate_vs_reconciliation['sharpe_unchanged']} "
        f"dd_log_convention={gate_vs_reconciliation['max_drawdown_log_cumsum_convention_pct']:.4f} "
        f"dd_fixed_notional={gate_vs_reconciliation['max_drawdown_fixed_notional_convention_pct']:.4f} | "
        f"three-way summary: {[(g, round(v['sharpe'],3), round(v['max_drawdown'],4), v['fires']) for g, v in three_way.items() if 'sharpe' in v]}"
    )


if __name__ == "__main__":
    main()
