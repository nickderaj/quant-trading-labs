"""Notebook 12 Phase 3 -- the final gate table, cross-checked
programmatically against `phase_1_12_preregistration.json` (never edited
after Phase 1) and `phase_0_12_results.json`'s frozen thresholds (never
edited after Phase 0). No new backtest runs here.
"""

import json
from pathlib import Path


def main() -> None:
    prereg = json.loads(
        Path("src/research/tmp/phase_1_12_preregistration.json").read_text()
    )["gates"]["VB"]
    calib = json.loads(Path("src/research/tmp/phase_0_12_results.json").read_text())
    result = json.loads(Path("src/research/tmp/phase_2_12_results.json").read_text())

    assert result["n_trials"] == prereg["n_trials"]
    assert result["origin_offsets"] == prereg["origin_offsets"]
    assert result["cost_multipliers"] == prereg["cost_multipliers"]
    assert (
        result["thresholds"]["base_max_range_atr_mult"]
        == calib["thresholds"]["base_max_range_atr_mult"]
    )
    assert (
        result["thresholds"]["prior_run_min_atr_mult"]
        == calib["thresholds"]["prior_run_min_atr_mult"]
    )
    assert result["thresholds"]["vol_k"] == calib["thresholds"]["vol_k"]

    legs = {
        "positive_every_offset": result["positive_every_offset"],
        "gated_vs_ungated_ci_excludes_zero": result["ci_excludes_zero"],
        "dsr_fires": result["dsr_fires"],
        "cost_stress_correctly_signed": result["cost_stress_correctly_signed"],
    }
    all_legs_pass = all(legs.values())
    assert all_legs_pass == result["gate_fires"], (
        "Phase 2's own AND does not match Phase 3's recomputation"
    )

    out = {
        "gate": "VB",
        "fires": result["gate_fires"],
        "fundable": result["fundable_flag"],
        "legs": legs,
        "n_trials": result["n_trials"],
        "deflated_sharpe_prob": result["deflated_sharpe_prob"],
        "thresholds": result["thresholds"],
        "n_instruments": calib["n_instruments"],
        "total_instruments": calib["total_instruments"],
        "trade_counts_by_asset_class_gated": result[
            "trade_counts_by_asset_class_gated"
        ],
        "trade_counts_by_asset_class_ungated": result[
            "trade_counts_by_asset_class_ungated"
        ],
        "sharpes_gated_1x_by_offset": result["sharpes_gated_1x_by_offset"],
        "gated_vs_ungated_delta_ci": result["gated_vs_ungated_bootstrap_offset0_1x"][
            "delta_ci"
        ],
        "max_drawdown_offset0_1x_gated": result["max_drawdown_offset0_1x_gated"],
    }
    Path("src/research/tmp/phase_3_12_results.json").write_text(
        json.dumps(out, indent=2)
    )
    print(f"Gate VB final: fires={out['fires']} fundable={out['fundable']} legs={legs}")


if __name__ == "__main__":
    main()
