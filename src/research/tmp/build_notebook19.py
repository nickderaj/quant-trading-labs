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
# Notebook 019 — What a correlation-thresholded Deflated Sharpe switch can and cannot fix

017 diagnosed a real defect in `research.deflated_sharpe_prob` and found two repairs (V1, V1b) that
fix calibration but lose real power at ρ=0, and one (V2) that has power but is badly miscalibrated.
None was adopted. This notebook tests **V3**: a switch that reads V1's cross-sectional dispersion
only when a cheap correlation estimate says the trial family actually is correlated, falling back to
V0 otherwise. Full narrative and numbers: `src/results/019_dsr_correlation_switch.md`.
""")
)

cells.append(
    code("""\
import json
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")
import dsr_lib17 as L

TMP = "tmp"


def load(name):
    with open(f"{TMP}/{name}") as f:
        return json.load(f)


prereg = load("phase_0_19_preregistration.json")
print("Candidate:", prereg["candidate"]["name"])
print("Tau candidates:", prereg["tau_candidates"])
print()
print("018 ceiling re-verification:", prereg["notebook_018_ceiling_reverification"]["conclusion"])
""")
)

cells.append(
    md("""\
## Sec 0 disclosure — what was already visible before any new Monte Carlo ran

An inspection pass over 017's frozen 756-cell certificate, plus a preflight probe of the switch's own
sampling distribution, disclosed in full in the pre-registration below. Five facts constrain what this
notebook may honestly claim: the switch is deterministic almost everywhere on 017's grid (§0.1); no τ<0.9
can ever pass DS-3's high-ρ clause, provably (§0.2); τ=0.15 and τ=0.30 are gate-equivalent (§0.3); there
is a real in-sample improvement restricted to N≥12 ∧ T≥3840 (§0.4) — but it was found by looking at
already-visible results and needs out-of-sample confirmation; and 017's own write-up under-reports one
DS-3 count (20 vs the true 78) due to a list-truncation artifact (§0.5), corrected as an erratum in 017's
write-up, not by editing 017's frozen numbers.
""")
)

cells.append(
    code("""\
preflight = prereg["preflight_disclosure"]
switch = preflight["switch_activation_probe"]
print("rho=0 false-trigger rate (M=200 preflight):", switch["rho0_false_trigger"])
print("max sd of estimate at true rho=0:", switch["max_sd_of_estimate_at_true_rho0"])
print()
cert = preflight["from_017_certificate"]
print("017 write-up DS-3 count correction:", cert["ds3_count_correction"])
""")
)

cells.append(
    md("""\
## Phase 1 — `dsr_lib17.py` v3 extension (DS-5)

A `v3` branch in `dsr_variant`: `mean_pairwise_corr <= tau` delegates to V0, otherwise to V1 — a genuine
recursive delegation, so its output is bit-for-bit whatever V0/V1 alone produce. The boundary resolves
to V0. Three load-bearing tests in `tests/test_dsr_lib17.py`.
""")
)

cells.append(
    code("""\
sharpe, n_trials, n_obs, tau = 0.05, 12, 1000, 0.15
trial_sharpes = [0.04, 0.05, 0.06, 0.045, 0.055, 0.05, 0.048, 0.052, 0.049, 0.051, 0.047, 0.053]

below = L.dsr_variant(sharpe, n_trials, n_obs, variant="v3", tau=tau,
                       mean_pairwise_corr=tau - 1e-6, trial_sharpes=trial_sharpes)
above = L.dsr_variant(sharpe, n_trials, n_obs, variant="v3", tau=tau,
                       mean_pairwise_corr=tau + 1e-6, trial_sharpes=trial_sharpes)
boundary = L.dsr_variant(sharpe, n_trials, n_obs, variant="v3", tau=tau,
                          mean_pairwise_corr=tau, trial_sharpes=trial_sharpes)
print("just below tau -> branch:", below["branch_used"], " probability:", below["probability"])
print("just above tau -> branch:", above["branch_used"], " probability:", above["probability"])
print("exactly at tau -> branch:", boundary["branch_used"], "(boundary resolves to V0)")
""")
)

cells.append(
    md("""\
## Phase 2 — the switch-activation profile (DS-6 first clause)

`P(mean_pairwise_corr_estimate >= tau)` at all 378 design points x both modes, M=2000.
""")
)

cells.append(
    code("""\
profile = load("phase_2_19_switch_profile.json")
print("n_cells:", profile["n_cells"])
print("rho=0 false-trigger rate:", profile["rho0_false_trigger"])
print("max sd of estimate at true rho=0:", profile["max_sd_of_estimate_at_true_rho0"])
print("DS-6 first clause:", profile["ds6_first_clause"])
""")
)

cells.append(
    md("""\
## Phase 3 — the prediction, written before Phase 4 ran

Mixture-predicts every one of 017's 756 cells (`p*rate_v1 + (1-p)*rate_v0`, `p` from Phase 2's real
measurement), run through 017's own `evaluate_variant`, unmodified.
""")
)

cells.append(
    code("""\
prediction = load("phase_3_19_prediction.json")
print("Objective A predicted verdict:", prediction["objective_A_predicted_verdict"])
print()
for tau in ("0.15", "0.3"):
    u = prediction["by_tau"][tau]["uncapped_violation_counts"]
    print(f"tau={tau} uncapped violations:", u)
    widest = [(r["min_n_trials"], r["min_n_obs"])
              for r in prediction["by_tau"][tau]["restricted_regime_scan"] if r["passes"]]
    print(f"tau={tau} predicted widest passing boxes:", widest)
""")
)

cells.append(
    md("""\
## Phase 4 — the confirmation grid (the only phase that ran new Monte Carlo)

142 cells: C1 (96, points 017 never ran), C2 (22, ambiguous points), C3 (24, a deterministic control
sample). All seeds namespaced apart from 017's own `seed_for_cell` -- genuinely independent
replications, not replays. Single-core by design (017's own OOM lesson).
""")
)

cells.append(
    code("""\
confirmation = load("phase_4_19_confirmation.json")
print("n_cells by subset:", confirmation["n_cells_by_subset"])
with open("../../scratch/019/phase4_manifest.json") as f:
    manifest = json.load(f)
print("C2 disclosure:", manifest["c2_disclosure"])
""")
)

cells.append(
    md("""\
## Phase 5 — adoption (DS-5 through DS-8)
""")
)

cells.append(
    code("""\
adoption = load("phase_5_19_adoption.json")
print("DS-5:", adoption["DS5"]["fires"])
print("DS-6:", {k: v["fires"] for k, v in adoption["DS6_by_tau"].items()})
print("DS-7:", {k: v["fires"] for k, v in adoption["DS7_by_tau"].items()})
print("DS-8:", {k: v["fires"] for k, v in adoption["DS8_by_tau"].items()})
print()
print("Objective A adopted at tau:", adoption["objective_A"]["adopted_at_tau"])
print("Objective B fires:", adoption["objective_B"]["fires"], "at tau=", adoption["objective_B"]["tau_used"])
print()
print(adoption["final_verdict"])
""")
)

cells.append(
    code("""\
ds8 = adoption["DS8_by_tau"]["0.15"]["gate_evaluation"]
print("DS-3 high-rho violations on C1 (the only violations found):")
for v in ds8["DS3"]["high_rho_violations"]:
    print(" ", v)

ds7 = adoption["DS7_by_tau"]["0.15"]
print()
print(f"DS-7: {ds7['n_comparisons']} comparisons, "
      f"{ds7['frac_within_3_combined_se']:.0%} within 3 combined SE, "
      f"worst disagreement {ds7['max_se_multiple']:.2f} SE")
""")
)

cells.append(
    md("""\
**DS-8 fails by exactly one cell out of 96**: `(N=12, T=5000, rho=0.9, 018_measured moments)`, where
V3's power gain over V0 is 8.93pp against the 10pp bar required — a ~2.57 combined-SE shortfall. Every
other C1 cell passes cleanly, including N=14 in the *same* hardest regime (11.43pp, itself only ~0.3
SE past the bar). Per this notebook's own frozen rule, this is reported as "the N≥12 boundary did not
replicate," not narrowed to N≥14 after the fact.
""")
)

cells.append(
    md("""\
## Phase 6 — rescore
""")
)

cells.append(
    code("""\
rescore = load("phase_6_19_rescore.json")
print(rescore["scope_note"])
print(f"{rescore['n_rows']} rows, {rescore['n_not_rescorable']} not_rescorable, "
      f"{rescore['n_rescored']} rescored, {rescore['n_verdict_change']} verdict changes")
print()
print("018:", rescore["note_018"])
""")
)

cells.append(
    md("""\
## Bottom line

**Objective A fails, exactly as pre-registered and analytically proven** — V3 is bit-for-bit V1 at
rho>=0.9 for any tau<0.9, and V1 already fails DS-3's high-rho clause there. **Objective B fails too**,
but narrowly and informatively: the switch mechanism itself is exactly correct (DS-5, DS-6 both fire
cleanly), the cheap mixture-prediction machinery this notebook built is independently validated
(DS-7 fires decisively, 100% of 46 comparisons within 3 combined SE), and 95 of 96 out-of-sample C1
cells pass every gate clause -- but the claimed N>=12 boundary misses at its own edge value, in the
hardest moments regime, by a modest ~2.6 SE. `research.py` stays exactly as 017 left it. Full detail:
`src/results/019_dsr_correlation_switch.md`.
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

with open("019_dsr_correlation_switch.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("written")
