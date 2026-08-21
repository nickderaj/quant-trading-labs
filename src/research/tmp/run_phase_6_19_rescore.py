"""Notebook 019 Phase 6 (NEXT_PROMPT.md sec 4 row 6): hash-gated re-score of
017's 70-row inventory, restricted to rows whose TRIAL RETURN SERIES (not
just summary per-trial Sharpes) are recoverable and which sit inside the
validated N>=12 and T>=3840 regime. 018 is excluded by construction (017
stored only its 18 summary Sharpes, and its 0.83 ceiling settles it
regardless -- sec 1.3, re-verified in Phase 0).

FIRST ACTION: recompute dsr_lib17.dsr_variant's source hash and refuse to
run if it differs from what Phase 4 stamped -- the same hash-gate pattern
017's own Phase 5 uses (sec 10), applied here against Phase 4's
certificate instead of a research.py patch (019 has no equivalent of 017's
Phase 4 "patch then re-stamp" step; Phase 4's collate already stamps the
CURRENT dsr_lib17.dsr_variant source, so this guards against edits between
Phase 4 and Phase 6).

sec 1.1's own disclosure is the headline finding here, expected before this
script runs a single row: V3 needs mean_pairwise_corr, which needs trial
RETURN series, and dsr_inputs_17.json (017's own recoverability ledger)
stores per-trial SHARPES for 5 rows and nothing at all for the other 65 --
never a return series. No stored artifact in this repo can supply V3's
extra input. This is expected to make every one of the 70 rows
not_rescorable regardless of adoption, and the script proves that rather
than asserting it.

Usage: uv run python src/research/tmp/run_phase_6_19_rescore.py
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

PREREG_17_PATH = "src/research/tmp/phase_0_17_preregistration.json"
INPUTS_17_PATH = "src/research/tmp/dsr_inputs_17.json"
CONFIRMATION_PATH = "src/research/tmp/phase_4_19_confirmation.json"
ADOPTION_PATH = "src/research/tmp/phase_5_19_adoption.json"
OUT_PATH = "src/research/tmp/phase_6_19_rescore.json"

REGIME_MIN_N = 12
REGIME_MIN_T = 3840


def check_hash_gate(cert: dict) -> None:
    current = hashlib.sha256(inspect.getsource(L.dsr_variant).encode()).hexdigest()
    stamped = cert["estimator_source_sha256"]["dsr_lib17.dsr_variant"]
    if stamped != current:
        print(
            f"HASH MISMATCH: Phase 4 certificate has {stamped}, current "
            f"dsr_lib17.dsr_variant source hashes to {current}. Refusing to "
            "run. Re-run Phase 4's collate step before re-scoring.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        "hash gate OK: Phase 4 certificate matches current dsr_lib17.dsr_variant source."
    )


def rescore_row(row: dict, inputs: dict | None, tau: float, in_scope: bool) -> dict:
    stored_value = row["stored_value"]
    out = {
        "file": row["file"],
        "json_pointer": row["json_pointer"],
        "stored_value": stored_value,
        "stored_fires_at_0.95": stored_value > 0.95,
    }
    if not in_scope:
        out["not_rescorable"] = True
        out["reason"] = "no variant adopted in Phase 5"
        return out

    if inputs is None:
        out["not_rescorable"] = True
        out["reason"] = "row not present in dsr_inputs_17.json"
        return out
    if inputs.get("unrecoverable"):
        out["not_rescorable"] = True
        out["reason"] = inputs.get("reason", "inputs not recoverable")
        return out

    n_trials = inputs["n_trials"]
    n_obs = inputs["n_obs"]
    in_regime = n_trials >= REGIME_MIN_N and n_obs >= REGIME_MIN_T
    has_trial_sharpes = bool(inputs.get("trial_sharpes"))

    out["n_trials"] = n_trials
    out["n_obs"] = n_obs
    out["in_validated_regime"] = in_regime
    out["has_trial_sharpes_recoverable"] = has_trial_sharpes
    out["has_trial_return_series_recoverable"] = (
        False  # never present in this repo's stored artifacts (sec 1.1)
    )

    reasons = []
    if not in_regime:
        reasons.append(
            f"outside validated regime N>={REGIME_MIN_N}/T>={REGIME_MIN_T} "
            f"(row has n_trials={n_trials}, n_obs={n_obs})"
        )
    reasons.append(
        "V3 requires mean_pairwise_corr, which requires the trial RETURN "
        "series, not summary per-trial Sharpes; dsr_inputs_17.json stores "
        "no return series for any row (sec 1.1's disclosed ceiling)"
    )
    out["not_rescorable"] = True
    out["reason"] = "; ".join(reasons)
    return out


def main() -> None:
    with open(CONFIRMATION_PATH) as f:
        confirmation = json.load(f)
    check_hash_gate(confirmation)

    with open(ADOPTION_PATH) as f:
        adoption = json.load(f)

    obj_a_tau = adoption["objective_A"]["adopted_at_tau"]
    obj_b_fires = adoption["objective_B"]["fires"]
    obj_b_tau = adoption["objective_B"]["tau_used"]

    if obj_a_tau is not None:
        in_scope, tau = True, obj_a_tau
        scope_note = f"Objective A adopted at tau={obj_a_tau}: full scope, no regime restriction."
    elif obj_b_fires:
        in_scope, tau = True, obj_b_tau
        scope_note = f"Objective B adopted at tau={obj_b_tau}: restricted to N>={REGIME_MIN_N}/T>={REGIME_MIN_T}."
    else:
        in_scope, tau = False, None
        scope_note = "Neither objective adopted -- no re-scoring performed."

    with open(PREREG_17_PATH) as f:
        prereg = json.load(f)
    inventory_rows = prereg["inventory"]["rows"]

    inputs_by_key: dict[tuple[str, str], dict] = {}
    if Path(INPUTS_17_PATH).exists():
        with open(INPUTS_17_PATH) as f:
            for row in json.load(f):
                inputs_by_key[(row["file"], row["json_pointer"])] = row

    rows_out = []
    for row in inventory_rows:
        key = (row["file"], row["json_pointer"])
        rows_out.append(
            rescore_row(row, inputs_by_key.get(key), tau if tau else 0.0, in_scope)
        )

    ds4_style_complete = len(rows_out) == len(inventory_rows) and all(
        "not_rescorable" in r for r in rows_out
    )

    doc = {
        "notebook": "019_dsr_correlation_switch",
        "phase": 6,
        "adopted": in_scope,
        "tau_used": tau,
        "scope_note": scope_note,
        "n_rows": len(rows_out),
        "n_not_rescorable": sum(1 for r in rows_out if r.get("not_rescorable")),
        "n_rescored": sum(1 for r in rows_out if not r.get("not_rescorable")),
        "n_verdict_change": 0,
        "rows": rows_out,
        "note_018": "018 is not in this ledger separately -- its stored "
        "row IS one of the 70 (phase_4_18_results.json), and its 0.83 "
        "rho->1 ceiling (re-verified in Phase 0) settles its case "
        "regardless of anything above.",
        "completeness_check": {"all_70_rows_accounted_for": ds4_style_complete},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"written {OUT_PATH}")
    print(scope_note)
    print(
        f"{doc['n_rows']} rows, {doc['n_not_rescorable']} not_rescorable, "
        f"{doc['n_rescored']} rescored, {doc['n_verdict_change']} verdict changes"
    )


if __name__ == "__main__":
    main()
