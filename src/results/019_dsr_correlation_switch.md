# Notebook 019 — What a correlation-thresholded Deflated Sharpe switch can and cannot fix

## What

Notebook 017 diagnosed a real defect in `research.deflated_sharpe_prob` (it over-penalizes correlated
trial families) and found two repairs (V1, V1b) that fix calibration but lose real detection power in
the ordinary independent-trials case at ρ=0 — and one repair (V2) that has power but is badly
miscalibrated. None was adopted. This notebook tests the natural next candidate 017's own "what to
test next" named: **V3**, a switch that reads V1's cross-sectional dispersion only when a cheap
estimate of the trial family's inter-trial correlation says the family actually is correlated, and
falls back to V0 (the current, correctly-calibrated-at-ρ=0 estimator) otherwise. The question is
whether that switch can separate V1's two failure modes — fix the ρ=0 power loss, which turns out to
be pure estimation noise, without pretending to fix the high-ρ power shortfall, which is not.

## Why

017's own "what to test next" (item 1) asked whether a repair restricted to larger trial families and
longer samples might pass where the general-purpose version doesn't. An inspection pass over 017's
frozen certificate, disclosed in full in this notebook's pre-registration before any new Monte Carlo
ran, found the answer was more specific than that: V1's ρ=0 power loss is concentrated exactly where a
correlation estimate should read close to zero, and routing those cells to V0 removes it completely,
*in sample*. The same inspection also proved, analytically, that no threshold could ever fix V1's
high-ρ shortfall (V3 **is** V1, bit for bit, in every cell that clause reads, for any τ<0.9). This
notebook exists to test the one real, falsifiable question left: does the in-sample ρ=0 fix survive
contact with grid points 017 never ran?

## How

**Phase 0 — pre-registration**, after a disclosed inspection pass (`scratch/019/preflight_from_017_certificate.py`,
`scratch/019/preflight_switch_probe.py`) that this document's own §0 discusses at length: the switch's
predicted outcome on 017's original 756-cell grid is deterministic almost everywhere (it never once
selects the V1 branch at true ρ=0 across 63 measured design points, and is essentially always in the
V1 branch by ρ≥0.5), so running the full 756-cell V3 grid would spend hours reproducing an answer
already visible for free. Both preflight scripts were re-run fresh and folded into
`phase_0_19_preregistration.json`; 018's ρ→1 upper bound (0.8316743360550332, sealing 018's case
regardless of anything below) was re-verified from 017's stored addendum, not recomputed.

**Phase 1 — `dsr_lib17.py` v3 extension.** A `v3` branch in `dsr_variant` (kwargs `tau`,
`mean_pairwise_corr`): `mean_pairwise_corr <= tau` delegates to the V0 branch, otherwise to V1 —
implemented as a genuine delegation (a recursive `dsr_variant` call), so its output is bit-for-bit
whatever V0/V1 alone produce, not a re-derivation that could drift. The boundary resolves to V0
(`<=`, not the stricter `<` the sec 1.1 pseudocode literally showed — resolved in favor of the
executable DS-5 test, which is the authoritative spec). `VARIANT_SPECS` gained `v3_tau0.15` and
`v3_tau0.30`; the vectorized `mc_cell` hot loop selects per-replication with `np.where` on the
`rho_bar` it already computes for the mean-pairwise-correlation estimator — no second correlation
pass. Three load-bearing tests (DS-5, `tests/test_dsr_lib17.py`), all passing.

**Phase 2 — the switch-activation profile.** `P(mean_pairwise_corr_estimate >= tau)` at all 378
(N, T, ρ, moments) design points × both modes (null, edge), M=2000 — completing the M=200 preflight
probe at the grid's real scale. **DS-6's first clause fires cleanly**: the measured false-trigger rate
at true ρ=0 is exactly 0.0 at both τ, for every one of the 63 (N, T, moments) points checked (well
under the 0.005 bar), with the estimate's own standard deviation topping out at 0.024 — τ=0.15 sits
more than 6 SD away even at its worst point.

**Phase 3 — the prediction, written before Phase 4 ran.** Every one of 017's 756 cells got a
mixture-predicted V3 rate (`p·rate_v1 + (1−p)·rate_v0`, `p` from Phase 2's real measurement, not the
preflight's deterministic approximation), run through 017's own `evaluate_variant` unmodified.
**Objective A's predicted verdict: FAILS at both τ**, exactly as pre-registered — DS-3's high-ρ clause
is inherited from V1 verbatim (0 DS-2a/2b violations, 20 uncapped high-ρ violations, 0 rho0
violations, at both τ). The predicted in-sample restricted-regime scan reproduced §0.4's finding on
real Phase-2-measured probabilities: V3(τ=0.15) and V3(τ=0.30) both pass every gate clause down to
**N≥12 ∧ T≥3840**, where V1 needed N≥95 — still only a prediction, not evidence, per this notebook's
own discipline.

**Phase 4 — the confirmation grid, the only phase that ran new Monte Carlo.** 142 cells across three
pre-declared subsets: **C1** (96 cells, N∈{12,14,24,50} × T=5000 × ρ∈{0, 0.5, 0.9, 0.95} × 3 moment
regimes × {null, edge} — points 017 never ran, at axis values (T=5000, ρ=0.95) that don't exist in
017's grid); **C2** (22 cells — Phase 2 found 11 ambiguous design points, all in the `018_measured`
extreme-moments regime, mostly at ρ=0.25, well under the 60-point cap, no truncation needed); **C3**
(24 cells, a deterministic control sample drawn from 017's own 756 with `numpy.random.default_rng(19)`,
excluding N≥95 ∧ T=3840). All cells used seeds explicitly namespaced apart from 017's own
`seed_for_cell` — genuinely independent replications, not replays. Single-core by design (`CONCURRENCY=1`
default, 017's own OOM lesson taken seriously); predicted cost 2.36h, actual wall time close to that,
well under the 6h refusal threshold.

**Phase 5 — adoption.** DS-5 fires (all three unit tests pass). DS-6 fires in full at both τ (first
clause as above; second clause — zero ρ=0 detection-rate shortfall against V0 — checked across all 18
confirmed ρ=0 edge cells spanning C1/C2/C3, zero shortfalls). **DS-7 fires decisively**: 100% of 46
predictable comparisons (C2 ∪ C3) agree with Phase 3's prediction within 3 combined MC standard
errors, worst disagreement 2.35 SE — well inside both the 95%-within-3SE and no-disagreement-over-5SE
bars. The cheap mixture prediction was trustworthy. **DS-8 does not fire.** Across all 96 C1 cells —
every one of them inside the claimed N≥12 ∧ T≥3840 regime by construction — exactly **one** violates a
gate clause: `(N=12, T=5000, ρ=0.9, 018_measured moments)`, where V3's power gain over V0 is 8.93pp
against the required 10pp margin, a shortfall of 1.07pp (≈2.57 combined MC SE, i.e. a real but modest
miss, not a blowout). Every other C1 cell — all three moment regimes at N∈{14, 24, 50}, and both
Gaussian and moderate-non-Gaussian moments even at N=12 — clears the same clause comfortably (the next
point up, N=14 in the *same* hardest regime, clears it at 11.43pp, itself only 0.3 combined SE past the
bar). DS-2a/2b: zero violations anywhere in C1.

**Phase 6 — rescore.** Hash-gated against Phase 4's stamped `dsr_lib17.dsr_variant` source (verified
matching, not merely asserted). Since neither objective was adopted, no re-scoring was performed —
but the deeper finding, checked regardless of adoption: of 017's 70-row inventory, only 5 rows have
*any* per-trial data recoverable (`dsr_inputs_17.json`), and all 5 store per-trial **Sharpes**, never
the trial **return series** V3's `mean_pairwise_corr` needs. Even in the counterfactual where Objective
B had fired, **zero of the 70 stored rows could have been re-scored** — the honest ceiling sec 1.1
flagged before any Monte Carlo ran. All 5 recoverable rows also have n_trials∈{4,8}, below the
regime's own N≥12 floor, a second, independent exclusion reason. 018 is not a separate row in this
ledger — it is one of the 70 (`phase_4_18_results.json`), and its re-verified 0.83 ceiling settles its
case regardless of anything above.

## Results

**Objective A (full-scope adoption): fails, exactly as pre-registered and analytically proven.** For
any τ<0.9, V3 is bit-for-bit V1 in every cell DS-3's high-ρ clause reads (measured: 100% branch
selection at ρ≥0.9 across the whole grid), and V1 already fails that clause. This is not contingent
on the prediction mechanism's accuracy — it is a property of the code, confirmed by DS-5's tests and
Phase 2's branch-probability measurement. `research.py` is untouched.

**Objective B (validated-regime adoption): fails, but not by much, and not everywhere.** The in-sample
finding that started this notebook — V3 widens V1's usable regime from N≥95 down to N≥12 — **does not
survive out-of-sample confirmation exactly as claimed.** It survives almost everywhere: 95 of 96 C1
cells pass every clause cleanly, including the hardest regime (`018_measured` moments) at every N from
14 up. It fails at exactly the regime's own boundary value, N=12, in exactly the hardest moment regime,
by a modest margin (≈2.6 combined SE). Per this notebook's own frozen rule (sec 3.6), that is not
license to narrow the boundary to N≥14 after the fact — the honest report is that **the N≥12 boundary
as pre-registered did not replicate**, full stop. `research.py` stays exactly as 017 left it; no
verdict changes anywhere in the 70-row inventory (Phase 6: 0 rescored, since neither objective fired
*and* no stored row has a recoverable trial return series regardless).

**The two-part diagnosis 017 could not reach alone did land, though.** V1's two failure modes really
are separable, and one of them really is a fixable estimation-noise artifact: the correlation-switch
mechanism is exactly correct (DS-5, DS-6 both fire cleanly, with zero false triggers and zero ρ=0
shortfalls measured at fresh grid points), and the mixture-prediction mechanism this notebook built to
avoid re-running the full grid is itself validated (DS-7 fires decisively — 100% of predictable cells
agree with measurement within 3 combined SE). The high-ρ shortfall is not an artifact of any threshold
choice; it survives every τ this notebook is allowed to consider, exactly as proven before Phase 4 ran.

## Gate verdicts — the full table

| gate | claim | fires? | number behind it |
|---|---|:---:|---|
| **DS-5** (switch reduces correctly) | unit-test gate on the boundary | **YES** | all 3 tests pass (`tests/test_dsr_lib17.py`) |
| **DS-6** (switch engages correctly) | false-trigger rate ≤0.5% at ρ=0; zero ρ=0 shortfall vs V0 | **YES** (both τ) | measured false-trigger 0.0/2000 at every design point; 0/18 confirmed ρ=0 edge cells show any shortfall |
| **DS-7** (prediction is trustworthy) | predicted vs measured agree within 3 combined SE in ≥95% of cells, none over 5 SE | **YES** (both τ) | 46/46 comparisons (100%) within 3 SE; worst disagreement 2.35 SE |
| **DS-8** (regime survives out of sample) | zero gate-clause violations on C1 (points 017 never ran, all inside N≥12∧T≥3840) | **NO** (both τ) | 1/96 C1 cells violates: N=12, T=5000, ρ=0.9, `018_measured` moments — 8.93pp power gain vs 10pp required (2.57 combined SE short); every other cell passes |
| **Objective A** (full-scope adoption) | adopt at first τ passing DS-2/DS-3/DS-5/DS-6 | **NO** | proven analytically (sec 0.2) and confirmed by prediction: DS-3 high-ρ clause fails at both τ, 20 uncapped violations each |
| **Objective B** (validated-regime adoption) | adopt restricted to N≥12∧T≥3840 iff DS-5/6/7/8 all fire | **NO** | DS-8 fails by one cell at the regime's own boundary |
| — | **adopted variant** | **NONE** | `research.py` confirmed unmodified |
| — | **rescore (Phase 6)** | 0/70 rescored | neither objective adopted; independently, 0/70 rows have a recoverable trial *return series* (only 5 have per-trial Sharpes, none ≥N=12) |

## What to test next

- **A narrower boundary, pre-registered fresh.** N=14 already clears DS-3's high-ρ margin comfortably
  in the exact regime (extreme moments) where N=12 misses it, by a wide-enough margin (11.43pp vs a
  10pp bar, itself only 0.3 combined SE past the line) that N≥14 ∧ T≥3840 looks like a plausible
  candidate boundary. This notebook's own rule (sec 3.6) forbids adopting that retroactively here —
  it would need its own pre-registration and its own out-of-sample confirmation grid, exactly this
  notebook's own methodology applied one boundary value tighter.
- **Trial return series as a storage convention.** Every notebook whose trial family might ever be
  worth re-scoring under a correlation-aware repair should store the trial return series, not just
  summary Sharpes — 017 flagged this (sec 5.4) and this notebook's Phase 6 makes the cost of not doing
  it concrete: literally zero of 70 historical values could be re-scored under V3, adopted or not,
  for exactly this reason.
- **The sequential/nested-search independence failure**, named again because it is untouched by
  anything in this notebook or 017: correlation is not the only way a trial family violates the
  "independent draws" assumption a DSR calculation makes, and no dispersion or correlation estimate
  detects search-path dependence.
- **018's own case is not reopened by anything here** — its 0.83 ρ→1 ceiling, re-verified in Phase 0,
  settles it independently of any DSR variant, adopted or not.

---

*Co-produced with Claude Sonnet 5. Notebook: `src/research/019_dsr_correlation_switch.ipynb`.*
