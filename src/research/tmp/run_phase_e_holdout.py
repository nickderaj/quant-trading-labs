"""Phase E (checkpoint 5, GATED): the holdout runs only if Gate TC, RG, CY,
or TF fired (NEXT_PROMPT.md section 3's Gate H). This script is the
orchestrator's own paper trail for that decision - it reads the four gate
verdicts back out of the already-committed Phase A-D result JSONs and
decides programmatically, never by re-deriving a fresh number and never by
a subagent (section 1: "Never let a subagent decide a gate").

It does NOT import research.load_universe_panel with allow_holdout=True, or
touch any data past research.HOLDOUT_START, anywhere in this file - the
frozen holdout stays frozen. If this script ever finds a gate fired, the
correct action is to STOP and write the actual holdout driver by hand (per
section 4 Phase E's spec: single best config, completely unchanged, one run)
rather than silently extending this file to run it.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")


def _load(path):
    with open(path) as f:
        return json.load(f)


PHASE_A = _load("src/research/tmp/phase_a_turnover_results.json")
PHASE_B = _load("src/research/tmp/phase_b_risk_gated_results.json")
PHASE_C = _load("src/research/tmp/phase_c_carry_results.json")
PHASE_D = _load("src/research/tmp/phase_d_tail_factor_results.json")


def gate_tc_verdict() -> tuple[bool, str]:
    """Fires if a turnover-qualifying (>=30% turnover fall) variant has a
    bootstrap CI on excess return excluding zero, net Sharpe > 0, at >= 2/4
    origin offsets."""
    baseline_to = {
        off: v["A0_baseline_band0"]["turnover_per_year"]
        for off, v in PHASE_A["by_offset"].items()
    }
    names = {
        n for v in PHASE_A["by_offset"].values() for n in v if n != "A0_baseline_band0"
    }
    for name in names:
        n_pass = 0
        for off, variants in PHASE_A["by_offset"].items():
            v = variants[name]
            turnover_fall = 1 - v["turnover_per_year"] / baseline_to[off]
            lo, hi = v["bootstrap_ci_excess_return"]
            ci_excludes_zero = lo > 0 or hi < 0
            if turnover_fall >= 0.30 and v["sharpe_net"] > 0 and ci_excludes_zero:
                n_pass += 1
        if n_pass >= 2:
            return True, f"{name} passes at {n_pass}/4 offsets"
    return (
        False,
        "no turnover-qualifying variant clears CI-excludes-zero at >=2/4 offsets",
    )


def gate_rg_verdict() -> tuple[bool, str]:
    """Fires if a gated variant improves net Sharpe by >= 0.20 over the
    identical ungated book, on the same folds, at >= 3/4 offsets."""
    names = {
        n
        for v in PHASE_B["by_offset"].values()
        for n in v
        if n != "ungated_throttle_k6"
    }
    for name in names:
        n_pass = 0
        for variants in PHASE_B["by_offset"].values():
            base = variants["ungated_throttle_k6"]["sharpe_net"]
            delta = variants[name]["sharpe_net"] - base
            if delta >= 0.20:
                n_pass += 1
        if n_pass >= 3:
            return True, f"{name} improves net Sharpe by >=0.20 at {n_pass}/4 offsets"
    return (
        False,
        "no gated variant improves net Sharpe by >=0.20 over ungated at >=3/4 offsets",
    )


def gate_cy_verdict() -> tuple[bool, str]:
    """Fires if a carry variant has net Sharpe > 0, bootstrap CI excluding
    zero, at >= 3/4 offsets, AND deflated Sharpe prob > 50%."""
    for interval, idata in PHASE_C["intervals"].items():
        for pred_kind, offs in idata["by_pred"].items():
            for variant_key in ("base", "throttled_k6"):
                n_pass = 0
                for v in offs.values():
                    cell = v[variant_key]
                    lo, hi = cell["bootstrap_ci_excess_return"]
                    ci_excludes_zero = lo > 0 or hi < 0
                    if (
                        cell["sharpe_net"] > 0
                        and ci_excludes_zero
                        and cell["deflated_sharpe_prob"] > 0.50
                    ):
                        n_pass += 1
                if n_pass >= 3:
                    return (
                        True,
                        f"{interval}/{pred_kind}/{variant_key} passes at {n_pass}/4 offsets",
                    )
    return (
        False,
        "no carry variant clears net>0 + CI-excludes-zero + DSR>50% at >=3/4 offsets",
    )


def gate_tf_verdict() -> tuple[bool, str]:
    """Fires if a factor's cross-sectional IC is significant (|NW t|>2) AND
    its portfolio is net-Sharpe-positive at >=3/4 offsets - WITHOUT being a
    single-symbol artifact (see run_phase_d_tail_factor.py's own
    exclusion check, added after D2's numbers were investigated)."""
    for factor_name, fr in PHASE_D["factors"].items():
        if fr.get("portfolio_skipped_ic_not_significant"):
            continue
        ic_significant = abs(fr["ic_stats"]["nw_tstat"]) > 2
        if not ic_significant:
            continue
        n_pass = 0
        single_symbol_artifact = False
        for cell in fr["by_offset"].values():
            if cell["sharpe_net"] > 0:
                n_pass += 1
                if cell.get("sign_flips_excl_top_symbol"):
                    single_symbol_artifact = True
        if n_pass >= 3 and not single_symbol_artifact:
            return True, f"{factor_name} passes at {n_pass}/4 offsets"
        if n_pass >= 3 and single_symbol_artifact:
            return False, (
                f"{factor_name} technically clears the raw numeric bar at {n_pass}/4 offsets "
                f"but its sign flips when the dominant single symbol is excluded - "
                f"not counted as a genuine cross-sectional finding"
            )
    return (
        False,
        "no factor clears significant-IC + net-Sharpe-positive-at->=3/4-offsets without a single-symbol artifact",
    )


def main():
    verdicts = {
        "TC": gate_tc_verdict(),
        "RG": gate_rg_verdict(),
        "CY": gate_cy_verdict(),
        "TF": gate_tf_verdict(),
    }
    any_fired = False
    for gate, (fired, reason) in verdicts.items():
        print(f"Gate {gate}: {'FIRES' if fired else 'does not fire'} - {reason}")
        any_fired = any_fired or fired

    result = {
        "gate_verdicts": {
            g: {"fired": f, "reason": r} for g, (f, r) in verdicts.items()
        },
        "any_gate_fired": any_fired,
        "holdout_run": False,
    }

    if any_fired:
        print(
            "\nAt least one gate fired. Per Gate H, the holdout should now be run ONCE "
            "on the single best-performing config, unchanged. This script deliberately "
            "does not do that automatically - STOP and write the holdout driver by hand, "
            "selecting the config explicitly rather than letting this script pick one."
        )
    else:
        print(
            "\nNo gate fired (four-way null, per NEXT_PROMPT.md section 5's own written-"
            "in-advance expected outcome). The holdout stays frozen - not run, not touched, "
            "per Gate H and this repo's 'spent by use, not by notebook' discipline."
        )

    with open("src/research/tmp/phase_e_holdout_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nwritten phase_e_holdout_results.json")


if __name__ == "__main__":
    main()
