"""Notebook 13, Phase L -- the look-ahead audit NEXT_PROMPT.md sec3/sec7
requires before any design's net Sharpe >= 1.5 is written into the results
file as a result rather than a suspected defect.

Reads every phase_{A,B,C,D}_13_results.json already produced and, for each
design whose headline dev-window net Sharpe clears 1.5, records which of
the audit legs apply and what they already show (each design's own script
computes several of these inline -- stop-fill sensitivity, offset checks,
survivorship -- so this phase's job is to assemble the verdict, not
recompute from scratch, and to say plainly when a design never triggers
the audit at all).

Writes phase_L_13_results.json.
"""

import json
from typing import Any

THRESHOLD = 1.5

PATHS = {
    "A": "src/research/tmp/phase_A_13_results.json",
    "B": "src/research/tmp/phase_B_13_results.json",
    "C": "src/research/tmp/phase_C_13_results.json",
    "D": "src/research/tmp/phase_D_13_results.json",
}


def headline_net_sharpe(design: str, data: dict) -> float | None:
    if design == "A":
        return data.get("trials", {}).get("base", {}).get("net", {}).get("sharpe")
    if design == "B":
        best = data.get("best_architecture")
        return (
            data.get("results", {}).get(best, {}).get("median_sharpe") if best else None
        )
    if design == "C":
        return data.get("trials", {}).get("adaptive", {}).get("net", {}).get("sharpe")
    if design == "D":
        return (
            data.get("books", {}).get("dollar_neutral", {}).get("net", {}).get("sharpe")
        )
    return None


def audit_design(design: str, data: dict, sharpe: float) -> dict:
    audit: dict[str, Any] = {
        "design": design,
        "headline_net_sharpe": sharpe,
        "triggered": True,
        "legs": {},
    }

    if design == "A":
        audit["legs"]["stop_fill_sensitivity"] = data.get(
            "stop_fill_sensitivity_optimistic"
        )
        audit["legs"]["arithmetic_red_flag"] = data.get("arithmetic_red_flag_check")
        audit["legs"]["offset_vacuity"] = (
            data.get("trials", {}).get("base", {}).get("by_offset")
        )
        audit["legs"]["benchmark_neutrality"] = data.get(
            "benchmark_neutrality_vs_spot_GCF"
        )
    elif design == "B":
        audit["legs"]["seed_spread"] = {
            k: v.get("seed_sharpes")
            for k, v in data.get("results", {}).items()
            if isinstance(v, dict) and "seed_sharpes" in v
        }
        audit["legs"]["breakeven_cost"] = {
            k: v.get("breakeven_cost_bps")
            for k, v in data.get("results", {}).items()
            if isinstance(v, dict)
        }
    elif design == "C":
        audit["legs"]["survivorship_with_without_delisted"] = {
            "with": data.get("trials", {}).get("adaptive"),
            "without": data.get("trials", {}).get("adaptive_excl_delisted"),
        }
        audit["legs"]["adaptive_vs_frozen"] = {
            "ci": data.get("adaptive_vs_frozen_paired_ci"),
            "beats_frozen": data.get("adaptive_beats_frozen"),
        }
        audit["legs"]["ablation_directions"] = data.get("ablation_predicted_direction")
        audit["legs"]["offset_check"] = data.get("by_offset")
    elif design == "D":
        audit["legs"]["ic_sanity_vs_003"] = data.get("ic_sanity_check")
        audit["legs"]["node_degree"] = data.get("node_degree")
        audit["legs"]["time_mixing_gate"] = data.get("gate_TM")

    return audit


def main():
    results = {}
    for design, path in PATHS.items():
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            results[design] = {"status": "not yet run"}
            continue
        sharpe = headline_net_sharpe(design, data)
        if sharpe is None:
            results[design] = {
                "status": "no headline sharpe found",
                "raw_keys": list(data.keys()),
            }
            continue
        if sharpe >= THRESHOLD:
            results[design] = audit_design(design, data, sharpe)
        else:
            results[design] = {
                "design": design,
                "headline_net_sharpe": sharpe,
                "triggered": False,
                "note": f"below the {THRESHOLD} audit trigger -- reported as-is, no audit required",
            }

    out = {"threshold": THRESHOLD, "designs": results}
    with open("src/research/tmp/phase_L_13_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
