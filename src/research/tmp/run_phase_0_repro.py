"""Phase 0: reproduction check for notebook 9 (NEXT_PROMPT.md Phase 0).

Cheap re-grounding in the notebook 7/8 numbers this survey's diagnosis leans on
before any external research is gathered. Per this repo's own house style, an
AssertionError here IS the stop-and-report signal - not a number to patch.

Reproduces:
  1. Gate CE (notebook 8, phase_3b_gate_ce_results.json): 15/16 products reject
     normal-innovation 1% ES calibration, BH-adjusted, both tails.
  2. Gate RE (notebook 8, phase_7_results.json): 15/16 products pass 1% VaR
     coverage OOS.
  3. Gate AC (notebook 8, phase_5_results.json): commodity carry net Sharpe
     0.90-0.95 at every origin offset, deflated Sharpe prob 0.997, excess-return
     CI vs. basket includes zero -> does not fire. This is the specific result
     hypothesis (c) turns on.

Writes phase_0_repro9_results.json (NOT phase_0_results.json - that filename is
already owned by notebook 8's own Phase 0 hygiene script,
`run_phase_0_hygiene.py`, and must never be overwritten by this one).
"""

import json

OUT_PATH = "src/research/tmp/phase_0_repro9_results.json"


def load(name):
    with open(f"src/research/tmp/{name}") as f:
        return json.load(f)


def main():
    results = {}

    # ---- 1. Gate CE: 15/16 rejection count ----
    ce = load("phase_3b_gate_ce_results.json")
    gate_ce = ce["gate_CE"]
    print("Gate CE:", json.dumps(gate_ce, indent=2))
    assert gate_ce["n_products_rejecting_1pct_both_tails_bh"] == 15, (
        f"Gate CE rejection count {gate_ce['n_products_rejecting_1pct_both_tails_bh']} != 15"
    )
    assert gate_ce["n_products_total"] == 16
    assert gate_ce["fires"] is True
    results["gate_CE_repro"] = gate_ce

    # ---- 2. Gate RE: 15/16 pass count ----
    re_ = load("phase_7_results.json")
    gate_re = re_["gate_RE"]
    print("\nGate RE:", json.dumps(gate_re, indent=2))
    assert gate_re["n_products_passing_1pct_coverage"] == 15, (
        f"Gate RE pass count {gate_re['n_products_passing_1pct_coverage']} != 15"
    )
    assert gate_re["n_products_total"] == 16
    assert gate_re["fires"] is True
    results["gate_RE_repro"] = gate_re

    # ---- 3. Gate AC: carry Sharpe-by-offset, excess CI, DSR ----
    alpha = load("phase_5_results.json")
    carry = alpha["strategy_A_carry"]
    gate_ac = carry["gate_AC"]
    print("\nGate AC:", json.dumps(gate_ac, indent=2))
    sharpes = gate_ac["sharpes_net_by_offset"]
    assert all(0.90 <= v <= 0.95 for v in sharpes.values()), (
        f"carry net Sharpes {sharpes} outside the write-up's 0.90-0.95 range"
    )
    assert gate_ac["net_sharpe_positive_at_every_offset"] is True
    assert abs(gate_ac["deflated_sharpe_prob"] - 0.997) < 0.002, (
        f"carry DSR {gate_ac['deflated_sharpe_prob']} != write-up's 0.997"
    )
    ci_lo, ci_hi = gate_ac["excess_return_ci"]
    assert ci_lo < 0 < ci_hi, (
        f"carry excess-return CI {gate_ac['excess_return_ci']} does not include zero"
    )
    assert gate_ac["excess_ci_excludes_zero"] is False
    assert gate_ac["fires"] is False, (
        "Gate AC must be recorded as a null - hypothesis (c) turns on this exact fact"
    )
    results["gate_AC_repro"] = gate_ac

    results["_verdict"] = (
        "All three reproduction checks pass. Gate CE (15/16) and Gate RE (15/16) "
        "confirm notebook 8's risk-side headline results. Gate AC confirms the "
        "specific near-miss (net Sharpe 0.90-0.95, DSR 0.997, excess CI includes "
        "zero, does not fire) that hypothesis (c) in this notebook is built to "
        "investigate."
    )

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten {OUT_PATH}")
    print(results["_verdict"])


if __name__ == "__main__":
    main()
