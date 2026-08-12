"""Consolidates every risk-engine gate's result JSON into one document
(NEXT_PROMPT.md sec 11): `src/research/tmp/risk_engine_results.json`.

Purely mechanical -- reads the already-written per-gate result files (each
produced and reviewed by its own `run_risk_0N_*.py` script) and republishes
their gate verdicts side by side, so a reader (or `docs/10-risk-engine.md`)
has one place to see every gate's pass/fail without opening six files.
"""

from __future__ import annotations

import json

PATHS = {
    "DC": "src/research/tmp/run_risk_01_data_contract_results.json",
    "FS": "src/research/tmp/run_risk_02_family_policy_results.json",
    "PR_PH_DT": "src/research/tmp/run_risk_03_reproduction_results.json",
    "MB": "src/research/tmp/run_risk_04_monitor_results.json",
    "NL": "src/research/tmp/run_risk_05_lookahead_results.json",
}
OUT_PATH = "src/research/tmp/risk_engine_results.json"


def main() -> None:
    raw = {}
    for key, path in PATHS.items():
        with open(path) as f:
            raw[key] = json.load(f)

    gates = {
        "DC": raw["DC"]["gate_DC"],
        "FS": {
            "pass_counts": raw["FS"]["pass_counts"],
            "winner": raw["FS"]["winner"],
            "shipping_rule": raw["FS"]["shipping_rule"],
        },
        "PR": raw["PR_PH_DT"]["gate_PR"],
        "PH": raw["PR_PH_DT"]["gate_PH"],
        "DT": raw["PR_PH_DT"]["gate_DT"],
        "MB": raw["MB"]["gate_MB"],
        "NL": raw["NL"]["gate_NL"],
    }

    all_hard_gates_fire = all(
        [
            gates["DC"]["fires"],
            gates["PR"]["pass"],
            gates["PH"]["pass"],
            gates["DT"]["pass"],
            gates["NL"]["fires"],
        ]
    )

    summary = {
        "gates": gates,
        "all_hard_gates_fire": all_hard_gates_fire,
        "no_discovery_gates": True,
        "note": (
            "None of these is a discovery gate -- there is no outcome of "
            "this project that authorises a trade (NEXT_PROMPT.md sec 10). "
            "PR, PH, DC, DT and NL are hard gates: they determine whether "
            "anything else here means anything. FS and MB are reported "
            "either way, pre-committed before their numbers were known."
        ),
        "source_files": PATHS,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"written {OUT_PATH}")
    print(json.dumps({k: v for k, v in gates.items()}, indent=2, default=str))
    print(f"\nall_hard_gates_fire: {all_hard_gates_fire}")


if __name__ == "__main__":
    main()
