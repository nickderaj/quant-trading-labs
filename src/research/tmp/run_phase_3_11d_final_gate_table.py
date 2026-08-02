"""Notebook 11d Phase 3 -- assemble Gate MB and Gate MB-E into one final gate
table, cross-checked against `phase_6_11a_results.json`'s pre-registration
(same discipline as 11b's Phase 7 and 11c's own programmatic cross-check).
"""

import json
from pathlib import Path

TMP = Path("src/research/tmp")


def main() -> None:
    prereg = json.loads((TMP / "phase_6_11a_results.json").read_text())
    mb = json.loads((TMP / "phase_1_11d_results.json").read_text())
    mbe = json.loads((TMP / "phase_2_11d_results.json").read_text())

    assert prereg["dsr_counts"]["MB"]["n_trials"] == mb["n_trials"]
    assert prereg["dsr_counts"]["MB-E"]["n_trials"] == mbe["n_trials"]
    assert prereg["gates"]["MB"]["claim"] and prereg["gates"]["MB-E"]["claim"]

    table = {
        "MB": {
            "claim": prereg["gates"]["MB"]["claim"],
            "fires_if": prereg["gates"]["MB"]["fires_if"],
            "n_trials": mb["n_trials"],
            "gate_fires": mb["gate_fires"],
            "fundable_flag": mb["fundable_flag"],
            "deflated_sharpe_prob": mb["deflated_sharpe_prob"],
            "sharpes_1x_by_offset": mb["sharpes_1x_by_offset"],
            "noise_floor_ci": mb["noise_floor_offset0_1x"]["ci_return"],
        },
        "MB-E": {
            "claim": prereg["gates"]["MB-E"]["claim"],
            "fires_if": prereg["gates"]["MB-E"]["fires_if"],
            "n_trials": mbe["n_trials"],
            "gate_fires": mbe["gate_fires"],
            "fundable_flag": mbe["fundable_flag_eligible"],
            "deflated_sharpe_prob": mbe["deflated_sharpe_prob"],
            "sharpes_1x_by_offset": mbe["sharpes_1x_by_offset"],
            "noise_floor_ci": mbe["noise_floor_offset0_1x"]["ci_return"],
        },
        "total_dsr_trials": mb["n_trials"] + mbe["n_trials"],
        "n_gates_fired": int(mb["gate_fires"]) + int(mbe["gate_fires"]),
    }
    (TMP / "phase_3_11d_results.json").write_text(json.dumps(table, indent=2))
    print(
        f"Final gate table 11d: MB fires={mb['gate_fires']} MB-E fires={mbe['gate_fires']} "
        f"total_dsr_trials={table['total_dsr_trials']} n_gates_fired={table['n_gates_fired']}"
    )


if __name__ == "__main__":
    main()
