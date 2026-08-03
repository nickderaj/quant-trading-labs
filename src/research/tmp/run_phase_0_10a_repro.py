"""Phase 0: reproduction check for notebook 10a (Phase 0 cheap spread mean-reversion gate).

Re-derives load-bearing numbers from notebook 9 Phase 4 (spread mean-reversion probe)
and notebook 8 Phase 5 (carry/momentum gates) before any new gating logic is
overlaid. Per this repo's own house style, an AssertionError here IS the
stop-and-report signal - not a number to patch.

Reproduces:
  1. From phase_4_spread_probe_results.json (notebook 9 Phase 4): exactly 5 of 6
     spreads have ar1_mean_reversion.mean_reverting == True; exactly 4 of 6 have
     both zscore_5d_forward_ic.p_value < 0.05 AND zscore_5d_forward_ic.ic < 0.
     This is the "5 of 6 spreads mean-reverting, 4 of 6 with significant negative IC"
     figure cited in src/results/009_external_research_review.md.
  2. From phase_5_results.json (notebook 8 Phase 5): Gate AC (carry) does not fire
     (fires == False), deflated Sharpe prob approx 0.997, excess-return CI includes
     zero, net Sharpe 0.90-0.95 at every offset. Gate AM (momentum) does not fire,
     deflated Sharpe prob approx 0.098.
"""

import json

OUT_PATH = "src/research/tmp/phase_0_10a_repro_results.json"

# Track pass/fail for each assertion
checks: dict = {}


def load(name):
    with open(f"src/research/tmp/{name}") as f:
        return json.load(f)


def run_check(check_name, assertion, description):
    """Run an assertion and record pass/fail."""
    try:
        assert assertion, description
        checks[check_name] = "PASS"
        print(f"  [{check_name}] PASS")
    except AssertionError as e:
        checks[check_name] = f"FAIL: {e}"
        print(f"  [{check_name}] FAIL: {e}")


def main():
    results = {}

    # ---- 1. Phase 4 spread probe: mean reversion and IC checks ----
    print("\n=== Phase 4 Spread Probe Reproduction ===")
    spread_data = load("phase_4_spread_probe_results.json")
    per_spread = spread_data["per_spread"]
    spreads = list(per_spread.keys())
    print(f"Spreads: {spreads}")
    print(f"Total: {len(spreads)}")

    # Count mean-reverting spreads
    mean_reverting_spreads = [
        s for s in spreads
        if per_spread[s]["ar1_mean_reversion"]["mean_reverting"]
    ]
    run_check(
        "phase4_mean_reverting_count",
        len(mean_reverting_spreads) == 5,
        f"Expected 5 mean-reverting spreads, got {len(mean_reverting_spreads)}: {mean_reverting_spreads}"
    )
    print(f"  Mean-reverting: {mean_reverting_spreads}")

    # Count spreads with significant negative IC (p_value < 0.05 AND ic < 0)
    sig_neg_ic_spreads = [
        s for s in spreads
        if per_spread[s]["zscore_5d_forward_ic"]["p_value"] < 0.05
        and per_spread[s]["zscore_5d_forward_ic"]["ic"] < 0
    ]
    run_check(
        "phase4_sig_negative_ic_count",
        len(sig_neg_ic_spreads) == 4,
        f"Expected 4 spreads with sig negative IC, got {len(sig_neg_ic_spreads)}: {sig_neg_ic_spreads}"
    )
    print(f"  Significant negative IC: {sig_neg_ic_spreads}")

    # Detail each spread's IC status
    print("\n  Per-spread IC status:")
    for spread in spreads:
        ic_data = per_spread[spread]["zscore_5d_forward_ic"]
        p = ic_data["p_value"]
        ic = ic_data["ic"]
        sig_neg = p < 0.05 and ic < 0
        print(
            f"    {spread}: p_value={p:.6e}, ic={ic:.6f}, "
            f"sig_negative_ic={sig_neg}"
        )

    results["phase_4_spread_probe"] = {
        "mean_reverting_count": len(mean_reverting_spreads),
        "mean_reverting_spreads": mean_reverting_spreads,
        "sig_negative_ic_count": len(sig_neg_ic_spreads),
        "sig_negative_ic_spreads": sig_neg_ic_spreads,
    }

    # ---- 2. Phase 5 carry gate (Gate AC) ----
    print("\n=== Phase 5 Strategy A (Carry) Gate AC Reproduction ===")
    alpha = load("phase_5_results.json")
    carry = alpha["strategy_A_carry"]
    gate_ac = carry["gate_AC"]
    print(f"Gate AC: {json.dumps(gate_ac, indent=2)}")

    # Gate AC does not fire
    run_check(
        "gate_ac_fires",
        gate_ac["fires"] is False,
        f"Gate AC fires={gate_ac['fires']}, expected False"
    )

    # Deflated Sharpe prob approx 0.997
    dsr = gate_ac["deflated_sharpe_prob"]
    run_check(
        "gate_ac_dsr",
        abs(dsr - 0.997) < 0.002,
        f"Gate AC DSR {dsr} not approx 0.997 (within 0.002)"
    )

    # Excess return CI includes zero
    ci_lo, ci_hi = gate_ac["excess_return_ci"]
    run_check(
        "gate_ac_ci_includes_zero",
        ci_lo <= 0 <= ci_hi,
        f"Gate AC excess-return CI {gate_ac['excess_return_ci']} does not include zero"
    )

    # Net Sharpe 0.90-0.95 at every offset
    sharpes = gate_ac["sharpes_net_by_offset"]
    all_in_range = all(0.90 <= v <= 0.95 for v in sharpes.values())
    run_check(
        "gate_ac_sharpes_in_range",
        all_in_range,
        f"carry net Sharpes {sharpes} outside 0.90-0.95 range"
    )
    print(f"  Net Sharpes by offset: {sharpes}")

    results["gate_AC_repro"] = gate_ac

    # ---- 3. Phase 5 momentum gate (Gate AM) ----
    print("\n=== Phase 5 Strategy B (Momentum) Gate AM Reproduction ===")
    momentum = alpha["strategy_B_momentum"]
    gate_am = momentum["gate_AM"]
    print(f"Gate AM: {json.dumps(gate_am, indent=2)}")

    # Gate AM does not fire
    run_check(
        "gate_am_fires",
        gate_am["fires"] is False,
        f"Gate AM fires={gate_am['fires']}, expected False"
    )

    # Deflated Sharpe prob approx 0.098
    dsr_am = gate_am["deflated_sharpe_prob"]
    run_check(
        "gate_am_dsr",
        abs(dsr_am - 0.098) < 0.02,
        f"Gate AM DSR {dsr_am} not approx 0.098 (within 0.02)"
    )

    results["gate_AM_repro"] = gate_am

    # ---- Summary ----
    print("\n=== Check Summary ===")
    passed = sum(1 for v in checks.values() if v == "PASS")
    total = len(checks)
    print(f"Passed: {passed}/{total}")

    for check_name, result in checks.items():
        status = "PASS" if result == "PASS" else "FAIL"
        print(f"  {check_name}: {status}")
        if result != "PASS":
            print(f"    {result}")

    if passed == total:
        results["_verdict"] = (
            "All reproduction checks pass. Phase 4 confirms that 5 of 6 spreads "
            "are mean-reverting and 4 of 6 have significant negative forward IC, "
            "matching the figures in src/results/009_external_research_review.md. "
            "Phase 5 confirms that both Gate AC (carry, DSR 0.997, fires=False) "
            "and Gate AM (momentum, DSR 0.098, fires=False) are correctly recorded "
            "as null gates, which are the baseline conditions for notebook 10a's "
            "own Phase 1 gating investigation."
        )
        exit_code = 0
    else:
        results["_verdict"] = (
            f"FAILURE: {total - passed} check(s) failed. "
            "Do not proceed until all numbers match."
        )
        exit_code = 1

    print(f"\n{results['_verdict']}")

    # Write results JSON
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten {OUT_PATH}")

    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
