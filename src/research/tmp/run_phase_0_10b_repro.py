"""Phase 0: reproduction check for notebook 10b (Phase 0 FA-data-conditioned signal gate).

Re-derives load-bearing numbers from notebook 10a Phase 2 (spread selection),
notebook 10a Phase 5 (regime definitions and DSR config counts), and notebook 10b
Phase 4 (FA-data availability check) before any new gating logic is overlaid.
Per this repo's own house style, an AssertionError here IS the stop-and-report
signal - not a number to patch.

Reproduces:
  1. From phase_2_10a_results.json (notebook 10a Phase 2): exactly 30 spreads
     total, 11 inter-commodity, 19 calendar, 23 pass ADF at 5%. Also verifies
     that gold_silver and platinum_palladium are excluded from 10b
     (include_in_10b == False) due to unresolved AR1-vs-IC disagreement.
  2. From phase_5_10a_results.json (notebook 10a Phase 5 / 10b pre-registration):
     regime_definitions.primary == "deadband", and DSR config counts are
     SP=8, SPR=12, SPR-BW=1, VS=8, BM=20. Also verifies all six gate names
     {SP, SPR, "SPR-BW", VS, BM, "FA-data"} are present in gate_table.
  3. From phase_4_10b_results.json (notebook 10b Phase 4 FA-data check):
     resolved == "FALSE" and fa_data_available == False.
"""

import json

OUT_PATH = "src/research/tmp/phase_0_10b_repro_results.json"

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

    # ---- 1. Phase 2 spread selection (notebook 10a) ----
    print("\n=== Phase 2 Spread Selection Reproduction (Notebook 10a) ===")
    phase2_data = load("phase_2_10a_results.json")
    summary = phase2_data["summary"]
    per_spread = phase2_data["per_spread"]

    run_check(
        "phase2_n_spreads",
        summary["n_spreads"] == 30,
        f"Expected 30 spreads, got {summary['n_spreads']}"
    )

    run_check(
        "phase2_n_inter_commodity",
        summary["n_inter_commodity"] == 11,
        f"Expected 11 inter-commodity spreads, got {summary['n_inter_commodity']}"
    )

    run_check(
        "phase2_n_calendar",
        summary["n_calendar"] == 19,
        f"Expected 19 calendar spreads, got {summary['n_calendar']}"
    )

    run_check(
        "phase2_n_pass_adf_5pct",
        summary["n_pass_adf_5pct"] == 23,
        f"Expected 23 spreads passing ADF at 5%, got {summary['n_pass_adf_5pct']}"
    )

    # Check that gold_silver is excluded from 10b
    run_check(
        "gold_silver_exclude_10b",
        per_spread["gold_silver"]["include_in_10b"] is False,
        f"gold_silver include_in_10b={per_spread['gold_silver']['include_in_10b']}, expected False"
    )

    # Check that platinum_palladium is excluded from 10b
    run_check(
        "platinum_palladium_exclude_10b",
        per_spread["platinum_palladium"]["include_in_10b"] is False,
        f"platinum_palladium include_in_10b={per_spread['platinum_palladium']['include_in_10b']}, expected False"
    )

    results["phase_2_spread_selection"] = {
        "n_spreads": summary["n_spreads"],
        "n_inter_commodity": summary["n_inter_commodity"],
        "n_calendar": summary["n_calendar"],
        "n_pass_adf_5pct": summary["n_pass_adf_5pct"],
        "gold_silver_exclude_10b": per_spread["gold_silver"]["include_in_10b"],
        "platinum_palladium_exclude_10b": per_spread["platinum_palladium"]["include_in_10b"],
    }

    # ---- 2. Phase 5 regime definitions and DSR config (notebook 10a) ----
    print("\n=== Phase 5 Regime Definitions & DSR Config Reproduction (Notebook 10a) ===")
    phase5_data = load("phase_5_10a_results.json")
    regime_definitions = phase5_data["regime_definitions"]
    dsr_config_counts = phase5_data["dsr_config_counts"]
    gate_table = phase5_data["gate_table"]

    run_check(
        "regime_definitions_primary",
        regime_definitions["primary"] == "deadband",
        f"Expected regime_definitions.primary='deadband', got '{regime_definitions['primary']}'"
    )

    run_check(
        "dsr_config_sp",
        dsr_config_counts["SP"]["n_trials"] == 8,
        f"Expected SP n_trials=8, got {dsr_config_counts['SP']['n_trials']}"
    )

    run_check(
        "dsr_config_spr",
        dsr_config_counts["SPR"]["n_trials"] == 12,
        f"Expected SPR n_trials=12, got {dsr_config_counts['SPR']['n_trials']}"
    )

    run_check(
        "dsr_config_spr_bw",
        dsr_config_counts["SPR-BW"]["n_trials"] == 1,
        f"Expected SPR-BW n_trials=1, got {dsr_config_counts['SPR-BW']['n_trials']}"
    )

    run_check(
        "dsr_config_vs",
        dsr_config_counts["VS"]["n_trials"] == 8,
        f"Expected VS n_trials=8, got {dsr_config_counts['VS']['n_trials']}"
    )

    run_check(
        "dsr_config_bm",
        dsr_config_counts["BM"]["n_trials"] == 20,
        f"Expected BM n_trials=20, got {dsr_config_counts['BM']['n_trials']}"
    )

    # Check that all six gate names are present in gate_table
    gate_names = {"SP", "SPR", "SPR-BW", "VS", "BM", "FA-data"}
    gate_table_keys = set(gate_table.keys())
    all_gates_present = gate_names.issubset(gate_table_keys)
    run_check(
        "gate_table_all_gates_present",
        all_gates_present,
        f"Expected gates {gate_names} all present in gate_table, got {gate_table_keys}"
    )
    print(f"  Gate table keys: {sorted(gate_table_keys)}")

    results["phase_5_regime_and_dsr"] = {
        "regime_definitions_primary": regime_definitions["primary"],
        "dsr_config_counts": {
            k: v["n_trials"] for k, v in dsr_config_counts.items()
            if k != "_transparency_log"
        },
        "gate_table_keys": sorted(gate_table_keys),
    }

    # ---- 3. Phase 4 FA-data check (notebook 10b) ----
    print("\n=== Phase 4 FA-data Check Reproduction (Notebook 10b) ===")
    phase4_10b_data = load("phase_4_10b_results.json")

    run_check(
        "phase4_10b_resolved",
        phase4_10b_data["resolved"] == "FALSE",
        f"Expected resolved='FALSE', got '{phase4_10b_data['resolved']}'"
    )

    run_check(
        "phase4_10b_fa_data_available",
        phase4_10b_data["fa_data_available"] is False,
        f"Expected fa_data_available=False, got {phase4_10b_data['fa_data_available']}"
    )

    results["phase_4_10b_fa_data"] = {
        "resolved": phase4_10b_data["resolved"],
        "fa_data_available": phase4_10b_data["fa_data_available"],
    }

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
            "All reproduction checks pass. Phase 2 confirms that 30 spreads are "
            "selected for 10b analysis, with gold_silver and platinum_palladium "
            "correctly excluded due to unresolved AR1-vs-IC disagreement. Phase 5 "
            "confirms regime_definitions.primary='deadband' and all DSR config "
            "counts match expectations (SP=8, SPR=12, SPR-BW=1, VS=8, BM=20), "
            "with all six gate names present in gate_table. Phase 4 confirms "
            "FA-data is unavailable (resolved=FALSE, fa_data_available=False), "
            "which is the baseline condition for notebook 10b's own gating "
            "investigation."
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
