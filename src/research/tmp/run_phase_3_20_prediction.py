"""Notebook 020, Phase 3b (NEXT_PROMPT.md sec 8): the written-down
prediction, made BEFORE any Phase 4 cell runs (019's device, reused).

Converts this notebook's central hypothesis (H-A: better moments from the
diversification floor mechanically raise the DSR) into something falsifiable
before the answer exists. scripts/run_020_books.sh refuses to start Phase 4
if this file is missing.

Usage: uv run python src/research/tmp/run_phase_3_20_prediction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "research" / "tmp"))

import basis_lib18 as bl18
import basis_lib20 as bl20

import research

MECHANISM_JSON = REPO_ROOT / "src" / "research" / "tmp" / "phase_3_20_mechanism.json"
OUT_PATH = REPO_ROOT / "src" / "research" / "tmp" / "phase_3_20_prediction.json"

EXPECTED_018_NET_SHARPE = 0.5766182328943011
DSR_BOUND = 0.95
N_TRIALS = 32
ANNUALIZED_RATE = research.sharpe_to_annualized_rate("8h")


def _counterfactual_dsr(
    sharpe_annualized: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    return research.deflated_sharpe_prob(
        sharpe_annualized / ANNUALIZED_RATE,
        n_trials=N_TRIALS,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
    )


def _b_single_sharpe(xvenue_symbols: list[str]) -> dict[str, Any]:
    """018-style single-venue comparator, restricted to Mechanism B's own
    intersected universe (needed by XD-3 and, here, as the Phase 3b
    stand-in Sharpe for B0's counterfactual DSR).
    """
    panel, manifest = bl18.load_basis_panel(
        symbols=xvenue_symbols, start_date=bl18.DEV_START, end_date=bl18.DEV_END
    )
    featured = bl18.add_trade_features(panel)
    weights = bl18.build_book_weights(featured, timed=True)
    trade_frame = bl18.book_trade_frame(featured, weights, origin_offset=0)
    costed = bl18.apply_two_leg_costs(trade_frame)
    metrics = bl18.book_metrics(costed, ANNUALIZED_RATE, "b_single")
    n_ok = sum(1 for v in manifest.values() if v == "ok")
    return {
        "n_symbols_ok": n_ok,
        "n_symbols_requested": len(xvenue_symbols),
        "net_sharpe": float(metrics["sharpe_net"]),
        "n_obs": len(costed),
    }


def main() -> None:
    with open(MECHANISM_JSON) as f:
        mech = json.load(f)

    mech_a = mech["mechanism_a"]
    a1_moments = mech_a["A1"]["gross_moments"]
    a3_moments = mech_a["A3"]["gross_moments"]
    a3_n_obs = a3_moments["n_obs"]

    prediction: dict[str, Any] = {
        "predicted_moments": {
            "A1": {
                "skew": a1_moments["skew"],
                "kurtosis_non_excess": a1_moments["kurtosis_non_excess"],
            },
            "A3": {
                "skew": a3_moments["skew"],
                "kurtosis_non_excess": a3_moments["kurtosis_non_excess"],
            },
        }
    }

    counterfactual_dsr_a3 = _counterfactual_dsr(
        EXPECTED_018_NET_SHARPE,
        a3_n_obs,
        a3_moments["skew"],
        a3_moments["kurtosis_non_excess"],
    )
    prediction["counterfactual_dsr_A3"] = {
        "definition": "if A3 merely matched 018's net Sharpe (0.5766) but carried A3's own predicted moments, would RC-2's DSR leg clear 0.95?",
        "sharpe_used": EXPECTED_018_NET_SHARPE,
        "n_obs": a3_n_obs,
        "skew_used": a3_moments["skew"],
        "kurtosis_used": a3_moments["kurtosis_non_excess"],
        "dsr": counterfactual_dsr_a3,
        "clears_0.95": bool(counterfactual_dsr_a3 > DSR_BOUND),
    }
    prediction["rc2_dsr_leg_prediction"] = {
        "predicted_verdict": "fires"
        if counterfactual_dsr_a3 > DSR_BOUND
        else "does_not_fire",
        "reason": (
            f"counterfactual DSR at 018's Sharpe with A3's predicted moments = "
            f"{counterfactual_dsr_a3:.4f}, {'above' if counterfactual_dsr_a3 > DSR_BOUND else 'below'} "
            f"the 0.95 bar -- {'A3 need not even beat 018 on Sharpe for the DSR leg to clear' if counterfactual_dsr_a3 > DSR_BOUND else 'A3 would need a higher Sharpe than 018, not just better moments, for the DSR leg to clear'}."
        ),
    }

    mech_b = mech.get("mechanism_b")
    if mech_b is None or mech_b.get("blocked"):
        prediction["mechanism_b"] = {
            "status": "not_available_at_prediction_time",
            "detail": mech_b,
        }
        print("Mechanism B not available -- prediction limited to Mechanism A")
    else:
        xvenue_symbols = bl20.load_xvenue_universe()
        print(f"Computing B_single on {len(xvenue_symbols)} symbols...")
        b_single = _b_single_sharpe(xvenue_symbols)

        b0_moments = mech_b["b0_gross_moments"]
        counterfactual_dsr_b0 = _counterfactual_dsr(
            b_single["net_sharpe"],
            b0_moments["n_obs"],
            b0_moments["skew"],
            b0_moments["kurtosis_non_excess"],
        )
        prediction["b_single"] = b_single
        prediction["counterfactual_dsr_B0"] = {
            "definition": "if B0 merely matched B_single's net Sharpe but carried B0's own predicted moments, would XD-2's DSR leg clear 0.95?",
            "sharpe_used": b_single["net_sharpe"],
            "n_obs": b0_moments["n_obs"],
            "skew_used": b0_moments["skew"],
            "kurtosis_used": b0_moments["kurtosis_non_excess"],
            "dsr": counterfactual_dsr_b0,
            "clears_0.95": bool(counterfactual_dsr_b0 > DSR_BOUND),
        }
        prediction["xd2_dsr_leg_prediction"] = {
            "predicted_verdict": "fires"
            if counterfactual_dsr_b0 > DSR_BOUND
            else "does_not_fire",
            "reason": (
                f"counterfactual DSR at B_single's Sharpe with B0's predicted moments = "
                f"{counterfactual_dsr_b0:.4f}, "
                f"{'above' if counterfactual_dsr_b0 > DSR_BOUND else 'below'} the 0.95 bar."
            ),
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(prediction, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(
        f"RC-2 DSR-leg prediction: {prediction['rc2_dsr_leg_prediction']['predicted_verdict']}"
    )
    if "xd2_dsr_leg_prediction" in prediction:
        print(
            f"XD-2 DSR-leg prediction: {prediction['xd2_dsr_leg_prediction']['predicted_verdict']}"
        )


if __name__ == "__main__":
    main()
