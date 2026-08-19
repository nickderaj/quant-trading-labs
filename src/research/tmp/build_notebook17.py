import json
import uuid


def cid():
    return uuid.uuid4().hex[:8]


def md(src):
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid(),
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(
    md("""\
# Notebook 017 — The Deflated Sharpe estimator, diagnosed and repaired

`research.deflated_sharpe_prob` scales its deflation benchmark by the sampling standard error of a
single Sharpe estimate instead of the cross-sectional dispersion of the trial family's Sharpes
actually observed — correct for genuinely independent trials, wrong (in one direction or the
other) whenever a trial family is correlated, which is this repo's own standard robustness-check
pattern. This notebook establishes by Monte Carlo whether that divergence is real, repairs the
estimator if so, calibrates the repair across a grid of trial counts/sample lengths/return
moments/inter-trial correlation, and re-scores every stored DSR value this repository has on disk.

018's Gate FA-2 failed on its DSR leg (0.186 against a 0.95 bar) with a `known_caveat` pointing at
this notebook by name — the reserved slot and the five mechanical guards against motivated
reasoning are in NEXT_PROMPT.md sec 0.1 and not repeated here. Full narrative and numbers:
`src/results/017_deflated_sharpe_correction.md`.
""")
)

cells.append(
    code("""\
import json
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")
import dsr_lib17 as L

import research

TMP = "tmp"


def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


prereg = load("phase_0_17_preregistration.json")
print("Candidate variants:", list(prereg["candidate_variants"].keys()))
print("Adoption order:", ["V1", "V1b(0.25)", "V1b(0.5)", "V2"])
print()
print("Inventory: measured", prereg["inventory"]["measured"]["count"], "values across",
      prereg["inventory"]["measured"]["files"], "files (documented in NEXT_PROMPT.md sec 5.5:",
      f"{prereg['inventory']['documented_in_next_prompt']['count']}/"
      f"{prereg['inventory']['documented_in_next_prompt']['files']})")
""")
)

cells.append(
    md("""\
## Phase 0 — Pre-registration

Froze the three candidate repairs, the adoption rule, all four gates' thresholds, this notebook's
own 5-variant trial ledger, the verdict-change policy, and all three sec 14.2 branch texts for the
018 amendment — before any Monte Carlo ran. Also reproduced the sec 5.5 inventory mechanically
and verified 018's `fires_except_dsr_leg`, `bootstrap_ci_leg_fires`, and `holdout_access` fields
straight from the stored JSON.
""")
)

cells.append(
    code("""\
v018 = prereg["notebook_018_verification"]
print("018 gate FA-2:", v018["gate_FA2"])
print("018 holdout_access:", v018["holdout_access"])
print()
print(v018["conclusion"])
""")
)

cells.append(
    md("""\
**Inventory discrepancy.** Sec 5.5 documents "73 stored values across 17 files." An exact
`deflated_sharpe_prob`-keyed sweep (the literal reading of that phrase) finds 70 across 14. A
secondary fuzzy sweep (key-name variants like `deflated_sharpe_prob_headline`,
`published_deflated_sharpe_prob`) finds more, concentrated in notebook 011b's own multi-phase result
files — which appear to record the same figures redundantly under inconsistent key names rather
than being values the exact sweep missed. Per sec 5.5's own instruction, this notebook uses the
**measured** 70/14 for gate DS-4.
""")
)

cells.append(
    code("""\
inv = prereg["inventory"]
print("measured:", inv["measured"])
print()
print("fuzzy secondary check (disclosed only, not used for DS-4):")
print(inv["secondary_fuzzy_check_disclosed_not_used_for_DS4"])
""")
)

cells.append(
    md("""\
## Phase 1 — `dsr_lib17.py`

`expected_max_sharpe`, `dsr_variant` (all variants, returns the working inputs for auditability),
`psr_upper_bound`, `mc_cell` (one grid cell, all 5 variants, vectorized). Seven tests, including the
executable backward-compat proof: `dsr_variant(variant="v0")` reproduces
`research.deflated_sharpe_prob`'s two published constants bit for bit.
""")
)

cells.append(
    code("""\
import numpy as np

annualized_rate = float(np.sqrt(252))
out_ac = L.dsr_variant(0.9042246451305482 / annualized_rate, n_trials=4, n_obs=4507, variant="v0")
ref_ac = research.deflated_sharpe_prob(0.9042246451305482 / annualized_rate, n_trials=4, n_obs=4507)
print("gate AC: dsr_lib17 v0 =", out_ac["probability"], " research.py =", ref_ac,
      " match:", out_ac["probability"] == ref_ac)

out_18 = L.dsr_variant(
    0.01743215308672331, n_trials=18, n_obs=3837,
    skew=-11.516325584172863, kurtosis=816.8538707698766, variant="v0",
)
ref_18 = research.deflated_sharpe_prob(
    0.01743215308672331, n_trials=18, n_obs=3837,
    skew=-11.516325584172863, kurtosis=816.8538707698766,
)
print("018:     dsr_lib17 v0 =", out_18["probability"], " research.py =", ref_18,
      " match:", out_18["probability"] == ref_18)
""")
)

cells.append(
    md("""\
## Phase 2 — DS-1, the kill switch

Reproduces the sec 3 disclosed pilot's regime (N=18, T=3840 — 018's own trial count and bar
count, Gaussian) across the full ρ axis at honest M=20000 (vs. the pilot's M=400), plus V2
(untested by the pilot).
""")
)

cells.append(
    code("""\
phase2 = load("phase_2_17_results.json")
print(f"{'rho':>5}  {'V0 FPR':>10}  {'V0 MC SE':>10}  {'V1 FPR':>8}  {'V2 FPR':>8}")
for c in phase2["cells"]:
    print(f"{c['rho']:>5.2f}  {c['rate']['v0']:>10.4f}  {c['mc_se']['v0']:>10.4f}  "
          f"{c['rate']['v1']:>8.4f}  {c['rate']['v2']:>8.4f}")
print()
print("DS-1 fires:", phase2["gate_DS1"]["fires"])
print("  clause 1 (FPR <= 0.005 at rho>=0.9):", phase2["gate_DS1"]["clause_1_high_rho_fpr_le_0.005"]["fires"])
print("  clause 2 (non-increasing in rho):", phase2["gate_DS1"]["clause_2_fpr_non_increasing_in_rho"]["fires"])
""")
)

cells.append(
    md("""\
**DS-1 fires.** The defect is real: V0's false-positive rate falls from ~0.375% at ρ=0 to
0.10%/0.02% at ρ=0.9/0.99 — well below the 0.5% ceiling, and monotonically non-increasing
in ρ. Note that at the pilot's M=400 these high-ρ cells read as exactly 0.0000 ("below
resolution"); at M=20000 they resolve to small but nonzero values, which is a *more precise*
measurement, not a contradiction — both are consistent with DS-1's threshold and the pilot's own
stated limitations. Phase 3, the full calibration and power grid, proceeds.
""")
)

cells.append(
    md("""\
## Phase 3 — Full calibration + power grid

7x3x6x3 = 378 null cells x {null, injected-edge} = 756 cells, M=20000. Two disclosed deviations from
`distributions.frozen_dist`: the moderate-moments target (skew=-1.5, kurtosis=6.0) isn't reachable by
`jf_skew_t` at that kurtosis (closest fit lands at skew=-1.21); the extreme-moments target (skew=-11.5,
kurtosis=817, 018's own measured values) isn't reachable by `jf_skew_t` at all, so a method-of-moments
two-point jump mixture is used instead, solved to hit the target exactly. Both documented in
`dsr_lib17.py`'s module docstring.
""")
)

cells.append(
    code("""\
cert = load("phase_3_17_calibration.json")
print(f"n_cells: {cert['n_cells']} (expect 756)")
print("estimator_source_sha256:", cert["estimator_source_sha256"])
""")
)

cells.append(
    md("""\
## Phase 4 — Adoption

DS-2 (calibration) and DS-3 (power) applied to every candidate variant.
""")
)

cells.append(
    code("""\
adoption = load("phase_4_17_adoption.json")
for v, e in adoption["evaluations"].items():
    print(f"{v:12} DS-2a={e['DS2a']['fires']!s:6} DS-2b={e['DS2b']['fires']!s:6} "
          f"DS-3={e['DS3']['fires']!s:6} passes_both={e['passes_both_DS2_and_DS3']}")
print()
print("adopted_variant:", adoption["adopted_variant"])
print(adoption["verdict"])
""")
)

cells.append(
    md("""\
**None of the four candidates is adopted.** V1 and both V1b settings pass calibration (DS-2) cleanly
but fail power (DS-3) on its two-sided check: insufficient power gain over V0 at high correlation in
20 cells, AND a power *loss* versus V0 at rho=0 beyond the allowed margin in another 20 cells — the
"no free lunch" clause working as designed. V2 passes DS-3 but fails DS-2a badly (246/378 null cells
exceed the anti-conservatism ceiling). Per the pre-registered adoption rule, `research.py` is left
unmodified — confirmed by `git diff`, not merely asserted.
""")
)

cells.append(
    md("""\
## Phase 5 — Hash-gated re-score

The hash gate (sec 10) passed automatically: `research.py` was never touched in Phase 4, so its
source hash still matches the certificate Phase 3 stamped.
""")
)

cells.append(
    code("""\
rescore = load("phase_5_17_rescore.json")
print("gate_DS4 fires:", rescore["gate_DS4"]["fires"])
print(f"n_rows={rescore['n_rows']}  n_not_rescorable={rescore['n_not_rescorable']}  "
      f"n_verdict_change={rescore['n_verdict_change']}")
print()
for r in rescore["rows"]:
    if r.get("not_rescorable"):
        print(f"  not_rescorable: {r['file'].split('/')[-1]} {r['json_pointer']}  "
              f"stored={r['stored_value']:.4f}  reason={r['reason']}")
""")
)

cells.append(
    md("""\
65 of 70 rows have a rho->1 upper bound below 0.95 and provably cannot flip under any dispersion-based
repair, regardless of what Phase 4 decided. The remaining 5 (gate AC x3, notebook 10b's "calendar" and
"gate_VS") have an upper bound >=0.95 and would need a corrected value from an adopted variant — since
none was adopted, all 5 are `not_rescorable`, disclosed with an explicit reason rather than silently
skipped. Zero verdict changes overall.
""")
)

cells.append(
    md("""\
## Phase 6b — The 018 amendment

None of the three sec 14.2 branch texts frozen in Phase 0 literally applies: Branch A's premise
("DS-1 does not fire") is false — it fired. Branches B and C's premise ("repair adopted") is also
false — none was. A new, honest characterization is written instead
(`phase_7_18_dsr_addendum.json`), holding every sec 14.3 rule the three frozen branches share.
""")
)

cells.append(
    code("""\
addendum = load("phase_7_18_dsr_addendum.json")
print(addendum["sec_14_2_branch_applicability"]["conclusion"])
print()
row = addendum["018_own_row_in_phase_5_rescore"]
print("018's own rho->1 upper bound:", row["upper_bound"])
print()
m = addendum["what_this_means_for_018"]
print("research.py modified:", m["research_py_modified"])
print("018 DSR unchanged at:", m["018_dsr_value_unchanged"])
print("FA-2 fires:", m["gate_FA2_fires"], " FA-3 fires:", m["gate_FA3_fires"],
      " FUND fires:", m["gate_FUND_fires"], " holdout granted:", m["holdout_access_granted"])
""")
)

cells.append(
    md("""\
**018's case is settled more decisively than "no variant adopted" alone would suggest.** 018's own
stored inputs (sample skew -11.5, sample kurtosis 817 — 018's own book is exactly the extreme regime
Phase 0's "018_measured" moment axis exists to probe) put a rho->1 upper bound of **0.83** on its DSR —
below the 0.95 bar, and that bound holds for every dispersion-based variant this notebook evaluated,
adopted or not (Test 7, sec 7.3). So 018's DSR leg was never actually contingent on Phase 4's adoption
decision: even in the counterfactual where V1 had cleared both DS-2 and DS-3, 018's corrected DSR
could not have exceeded 0.83. FA-2, FA-3, and FUND stand exactly as 018 recorded them; the holdout
stays unspent. Full detail: `src/results/018_funding_basis_trade.md`'s appended Addendum section, and
the appended markdown cell in `018_funding_basis_trade.ipynb` (018's notebook was not re-executed).
""")
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("017_deflated_sharpe_correction.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("written")
