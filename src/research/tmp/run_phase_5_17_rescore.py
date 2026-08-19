"""Notebook 017 Phase 5 (NEXT_PROMPT.md sec 4, sec 5.4, sec 10): hash-gated
re-score of every stored DSR value in the sec 5.5 inventory (measured count
-- see Phase 0's discrepancy_note).

FIRST ACTION, before anything else: recompute both estimator source hashes
and refuse to run if either differs from what Phase 4 re-stamped into
phase_3_17_calibration.json after patching research.py. This is the ONLY
consumer of that hash gate (sec 10) -- editing the estimator after
re-scoring, or re-scoring against an uncertified estimator, is a mechanical
impossibility, not a promise.

Inputs (besides the certificate): src/research/tmp/dsr_inputs_17.json, a
per-row mapping of (file, json_pointer) -> the exact sharpe/n_trials/n_obs
/skew/kurtosis (and trial_sharpes where the family is recoverable) that
produced each stored value, built once by tracing each value back to its
originating script and verifying it reproduces the stored value bit for
bit. Rows not in that mapping, or explicitly marked unrecoverable, are
reported `not_rescorable` -- sec 5.4: no backtest is ever re-run to
recover a family, and guessing is the failure mode this whole notebook
exists to avoid.

Usage: uv run python src/research/tmp/run_phase_5_17_rescore.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dsr_lib17 as L

import research

CALIBRATION_PATH = "src/research/tmp/phase_3_17_calibration.json"
ADOPTION_PATH = "src/research/tmp/phase_4_17_adoption.json"
PREREG_PATH = "src/research/tmp/phase_0_17_preregistration.json"
INPUTS_PATH = "src/research/tmp/dsr_inputs_17.json"
OUT_PATH = "src/research/tmp/phase_5_17_rescore.json"

GATE_THRESHOLD = 0.95


def check_hash_gate(cert: dict) -> None:
    current = {
        "research.deflated_sharpe_prob": hashlib.sha256(
            inspect.getsource(research.deflated_sharpe_prob).encode()
        ).hexdigest(),
        "dsr_lib17.dsr_variant": hashlib.sha256(
            inspect.getsource(L.dsr_variant).encode()
        ).hexdigest(),
    }
    stamped = cert["estimator_source_sha256"]
    for name, current_hash in current.items():
        if stamped.get(name) != current_hash:
            print(
                f"HASH MISMATCH on {name}: certificate has {stamped.get(name)}, "
                f"current source hashes to {current_hash}. The estimator was "
                "edited after the certificate was stamped (or the certificate "
                "was never re-stamped through Phase 4). Refusing to run. "
                "Re-stamp via Phase 4 (sec 10) before re-scoring.",
                file=sys.stderr,
            )
            sys.exit(1)
    print("hash gate OK: certificate matches current estimator source.")


def rescore_row(row: dict, inputs: dict | None, adopted_variant: str | None) -> dict:
    stored_value = row["stored_value"]
    out = {
        "file": row["file"],
        "json_pointer": row["json_pointer"],
        "stored_value": stored_value,
        "stored_fires_at_0.95": stored_value > GATE_THRESHOLD,
    }

    if inputs is None or inputs.get("unrecoverable"):
        out["not_rescorable"] = True
        out["reason"] = (
            inputs.get("reason", "inputs not recoverable from stored artifacts")
            if inputs
            else "row not present in dsr_inputs_17.json"
        )
        return out

    sharpe = inputs["sharpe_per_period"]
    n_obs = inputs["n_obs"]
    skew = inputs.get("skew", 0.0)
    kurtosis = inputs.get("kurtosis", 3.0)
    n_trials = inputs["n_trials"]

    upper_bound = L.psr_upper_bound(sharpe, n_obs, skew, kurtosis)
    out["rho_to_1_upper_bound"] = upper_bound
    out["inputs"] = {
        "sharpe_per_period": sharpe,
        "n_trials": n_trials,
        "n_obs": n_obs,
        "skew": skew,
        "kurtosis": kurtosis,
        "source": inputs.get("source"),
    }

    if upper_bound < GATE_THRESHOLD:
        out["corrected_value"] = None
        out["not_rescorable"] = False
        out["verdict_change"] = False
        out["note"] = (
            f"upper bound {upper_bound:.4f} < 0.95: this verdict provably "
            "cannot flip under any dispersion-based repair (sec 5.4 item 2); "
            "no further work needed on this row."
        )
        return out

    trial_sharpes = inputs.get("trial_sharpes")
    if adopted_variant is None:
        out["not_rescorable"] = True
        out["reason"] = "no variant adopted in Phase 4"
        return out
    if adopted_variant.startswith("v1") and not trial_sharpes:
        out["not_rescorable"] = True
        out["reason"] = (
            f"upper bound >= 0.95 and adopted variant {adopted_variant} needs trial_sharpes, not recoverable"
        )
        return out
    if adopted_variant == "v2" and inputs.get("mean_pairwise_corr") is None:
        out["not_rescorable"] = True
        out["reason"] = (
            "upper bound >= 0.95 and adopted variant v2 needs mean_pairwise_corr, not recoverable"
        )
        return out

    kwargs: dict = {}
    base_variant = adopted_variant
    if adopted_variant.startswith("v1b"):
        base_variant = "v1b"
        kwargs["shrinkage_c"] = float(adopted_variant.split("v1b_c")[1])
    if base_variant in ("v1", "v1b"):
        kwargs["trial_sharpes"] = trial_sharpes
    elif base_variant == "v2":
        kwargs["mean_pairwise_corr"] = inputs["mean_pairwise_corr"]

    result = L.dsr_variant(
        sharpe, n_trials, n_obs, skew, kurtosis, variant=base_variant, **kwargs
    )
    corrected = float(result["probability"])  # type: ignore[arg-type]
    out["not_rescorable"] = False
    out["corrected_value"] = corrected
    out["trial_family_used"] = trial_sharpes if trial_sharpes else None
    out["family_mismatch"] = result.get("family_mismatch", False)
    out["verdict_change"] = (corrected > GATE_THRESHOLD) != (
        stored_value > GATE_THRESHOLD
    )
    return out


def main() -> None:
    with open(CALIBRATION_PATH) as f:
        cert = json.load(f)
    check_hash_gate(cert)

    with open(ADOPTION_PATH) as f:
        adoption = json.load(f)
    adopted_variant = adoption["adopted_variant"]

    with open(PREREG_PATH) as f:
        prereg = json.load(f)
    inventory_rows = prereg["inventory"]["rows"]

    inputs_by_key: dict[tuple[str, str], dict] = {}
    if Path(INPUTS_PATH).exists():
        with open(INPUTS_PATH) as f:
            for row in json.load(f):
                inputs_by_key[(row["file"], row["json_pointer"])] = row

    rows_out = []
    for row in inventory_rows:
        key = (row["file"], row["json_pointer"])
        rows_out.append(rescore_row(row, inputs_by_key.get(key), adopted_variant))

    ds4_fires = len(rows_out) == len(inventory_rows) and all(
        "not_rescorable" in r for r in rows_out
    )

    doc = {
        "notebook": "017_deflated_sharpe_correction",
        "phase": 5,
        "adopted_variant": adopted_variant,
        "n_rows": len(rows_out),
        "n_not_rescorable": sum(1 for r in rows_out if r.get("not_rescorable")),
        "n_verdict_change": sum(1 for r in rows_out if r.get("verdict_change")),
        "rows": rows_out,
        "gate_DS4": {"fires": ds4_fires},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(
        f"written {OUT_PATH}: {len(rows_out)} rows, {doc['n_not_rescorable']} not_rescorable, {doc['n_verdict_change']} verdict changes"
    )


if __name__ == "__main__":
    main()
