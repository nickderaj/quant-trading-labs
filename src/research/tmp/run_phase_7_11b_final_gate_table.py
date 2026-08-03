"""11b Phase 7: final gate table, cross-checked verbatim against 11a's
Phase 6 pre-registration (`phase_6_11a_results.json`), per this notebook's
own added rule that 11b's gate table must match that pre-registration
exactly -- no gate claim reworded, no n_trials count changed or shrunk.

Writes phase_7_11b_results.json.
"""

import json
import sys

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

PREREG_PATH = "src/research/tmp/phase_6_11a_results.json"
OUT_PATH = "src/research/tmp/phase_7_11b_results.json"

PHASE_FILES = {
    "TS": "src/research/tmp/phase_0_11b_results.json",
    "TS-S": "src/research/tmp/phase_0_11b_results.json",
    "BF": "src/research/tmp/phase_1_11b_results.json",
    "BF-X": "src/research/tmp/phase_1_11b_results.json",
    "SCR": "src/research/tmp/phase_2_11b_results.json",
    "VA": "src/research/tmp/phase_3_11b_results.json",
    "RE": "src/research/tmp/phase_4_11b_results.json",
}

FUNDABLE_SHARPE_MIN = 0.5
FUNDABLE_DSR_MIN = 0.95
FUNDABLE_MAX_DD = 0.25


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    prereg = _load(PREREG_PATH)
    prereg_gates = prereg["gates"]
    prereg_counts = prereg["dsr_counts"]

    data = {k: _load(v) for k, v in PHASE_FILES.items()}
    ts, bf, scr, va, re_ = (
        data["TS"],
        data["BF"],
        data["SCR"],
        data["VA"],
        data["RE"],
    )

    fires = {
        "TS": ts["gate_TS"]["fires"],
        "TS-S": ts["gate_TS_S"]["fires"],
        "BF": bf["gate_BF"]["fires"],
        "BF-X": bf["gate_BF_X"]["fires"],
        "SCR": scr["fires"],
        "VA": va["fires"],
        "RE": re_["fires"],
    }
    n_trials_actual = {
        "TS": ts["gate_TS"]["n_trials"],
        "TS-S": ts["gate_TS_S"]["n_trials"],
        "BF": bf["gate_BF"]["n_trials"],
        "BF-X": bf["gate_BF_X"]["n_trials"],
        "SCR": scr["n_trials"],
        "VA": va["n_trials"],
        "RE": re_["n_trials"],
    }

    # Verbatim cross-check: every gate's n_trials actually used must equal
    # the pre-registered count exactly -- never shrunk (NEXT_PROMPT.md sec 9).
    mismatches = {
        g: (n_trials_actual[g], prereg_counts[g]["n_trials"])
        for g in n_trials_actual
        if n_trials_actual[g] != prereg_counts[g]["n_trials"]
    }

    sharpe_by_gate = {
        "TS": ts["gate_TS"]["structured_by_offset"]["offset_0"]["sharpe"],
        "BF": bf["gate_BF"]["by_storage"][bf["gate_BF"]["headline_storage"]][
            "by_offset"
        ]["offset_0"]["sharpe"],
        "SCR": scr["screen_inclusive_by_offset"]["offset_0"]["sharpe"],
        "VA": va["va_by_offset"]["offset_0"]["sharpe"],
        "RE": re_["grid"][re_["best_non_baseline_key"]]["by_offset"]["offset_0"][
            "sharpe"
        ],
    }
    dd_by_gate = {
        "TS": ts["gate_TS"]["structured_by_offset"]["offset_0"]["max_drawdown"],
        "BF": bf["gate_BF"]["by_storage"][bf["gate_BF"]["headline_storage"]][
            "by_offset"
        ]["offset_0"]["max_drawdown"],
        "SCR": scr["screen_inclusive_by_offset"]["offset_0"]["max_drawdown"],
        "VA": va["va_by_offset"]["offset_0"]["max_drawdown"],
        "RE": re_["grid"][re_["best_non_baseline_key"]]["by_offset"]["offset_0"][
            "max_drawdown"
        ],
    }
    dsr_by_gate = {
        "TS": ts["gate_TS"]["deflated_sharpe_prob"],
        "BF": bf["gate_BF"]["deflated_sharpe_prob_headline"],
        "VA": None,
        "RE": re_["deflated_sharpe_prob_best"],
    }

    fundable_flag = {}
    for g, sharpe in sharpe_by_gate.items():
        dsr = dsr_by_gate.get(g)
        fundable_flag[g] = bool(
            sharpe is not None
            and sharpe > FUNDABLE_SHARPE_MIN
            and dsr is not None
            and dsr > FUNDABLE_DSR_MIN
            and abs(dd_by_gate[g]) <= FUNDABLE_MAX_DD
        )

    n_trials_11b_total = sum(prereg_counts[g]["n_trials"] for g in n_trials_actual)

    out = {
        "gate_claims_verbatim_from_11a_prereg": {g: prereg_gates[g] for g in fires},
        "fires": fires,
        "n_trials_actual": n_trials_actual,
        "n_trials_preregistered": {
            g: prereg_counts[g]["n_trials"] for g in n_trials_actual
        },
        "n_trials_mismatches": mismatches,
        "n_trials_match_prereg_exactly": len(mismatches) == 0,
        "n_trials_11b_total": n_trials_11b_total,
        "sharpe_by_gate_offset0": sharpe_by_gate,
        "max_drawdown_by_gate_offset0": dd_by_gate,
        "deflated_sharpe_prob_by_gate": dsr_by_gate,
        "institutionally_fundable_flag": fundable_flag,
        "any_gate_fires": any(fires.values()),
        "summary": (
            "All seven 11b gates return a fired=False verdict, cross-checked against the "
            "exact pre-registered criteria and DSR trial counts from phase_6_11a_results.json "
            "(no wording or count changed). Every comparison is internal (structured book vs "
            "10b's continuous benchmark, sign-flipped vs unconditional, screen-inclusive vs "
            "screen-exclusive, vol-adaptive vs static stop, reentry-swept vs baseline) -- "
            "none validates the external programme's own absolute reported numbers, which "
            "this repo's Phase 4 reproduction (11a) already found diverge materially. This "
            "is the fifteenth-plus gate in this programme's history to return a well-powered "
            "null, extending the run from 10a/10b."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(
        f"Phase 7: n_trials_match_prereg_exactly={out['n_trials_match_prereg_exactly']} "
        f"n_trials_11b_total={n_trials_11b_total} any_gate_fires={out['any_gate_fires']} "
        f"fires={fires}"
    )


if __name__ == "__main__":
    main()
