# Notebook 017 — The Deflated Sharpe estimator, diagnosed and repaired: Results Summary

## What

`research.deflated_sharpe_prob` implements Bailey & López de Prado's Deflated Sharpe Ratio, and it
scales its deflation benchmark by the sampling standard error of a single Sharpe estimate instead of
the cross-sectional dispersion of the trial family's Sharpes actually observed — correct for genuinely
independent trials, wrong (in one direction or the other, sec 1.3) whenever a trial family is
correlated, which is this repo's own standard robustness-check pattern (near-identical origin
offsets). This notebook establishes by Monte Carlo whether that divergence is real, repairs the
estimator if so, calibrates the repair across a grid of trial counts/sample lengths/return
moments/inter-trial correlation, and re-scores every stored DSR value this repository has on disk.

## Why

018's Gate FA-2 failed on its DSR leg (0.186 against a 0.95 bar) while passing its Sharpe leg cleanly,
with a `known_caveat` field pointing at this notebook by name. The reserved slot and five mechanical
guards against motivated reasoning are documented in NEXT_PROMPT.md sec 0.1 and are not repeated here.

## How

**Phase 0 — pre-registration.** Froze the three candidate repairs (V1: cross-sectional std per the
source paper; V2: effective-trials via inter-trial correlation; V1b: V1 with a shrinkage floor, the
pre-declared fallback for V1's small-N hazard), the adoption rule (simplest of V1 < V1b < V2 to pass
both DS-2 and DS-3), all four gates' thresholds, this notebook's own 5-variant trial ledger, the sec
5.4 verdict-change policy, and all three sec 14.2 branch texts for the 018 amendment — before any
Monte Carlo ran.

Reproduced the sec 5.5 inventory mechanically (an exact `deflated_sharpe_prob`-keyed sweep of
`src/research/tmp/*.json`) rather than re-typing the documented 73/17: **measured 70 stored values
across 14 files**, not 73/17. Disclosed rather than patched — see "Inventory discrepancy" below.

Verified from stored JSON (asserted, not eyeballed): 018's `gates.FA-2.fires_except_dsr_leg=False`
and `bootstrap_ci_leg_fires=False` (FA-2 fails independently of any DSR leg — it cannot flip under any
outcome of this notebook), and `holdout_access.access_granted=False`, `rule="requires FA-2 AND FA-3"`.

**Phase 1 — `dsr_lib17.py`.** `expected_max_sharpe`, `dsr_variant` (all variants, returns the working
inputs for auditability), `psr_upper_bound`, `mc_cell` (one grid cell, all 5 variants, vectorized).
Seven tests (sec 7.3), including the executable backward-compat proof: `dsr_variant(variant="v0")`
reproduces `research.deflated_sharpe_prob`'s two published constants (gate AC 0.997, 018's 0.18591)
bit for bit. The 0.997 pin moved out of `run_phase_0_repro.py` into `tests/test_research.py`.

**Phase 2 — DS-1 (the kill switch).** Reproduced the sec 3 disclosed pilot's regime (N=18, T=3840,
018's own trial count and bar count) across the full ρ axis at honest M=20000. **DS-1 fires**: V0's
FPR at ρ∈{0.9, 0.99} is 0.0010/0.0002 (≤0.005) and non-increasing in ρ from the ρ=0 baseline of
0.00375 within 2 MC SE at every intermediate point. The defect is real. Phase 3 proceeds.

**Phase 3 — full calibration + power grid.** 7×3×6×3 = 378 null cells × {null, injected-edge} = 756
cells, M=20000, `scripts/run_dsr_calibration.sh`. Completed after several interruptions unrelated to
the science: a machine-level reboot mid-run (the script is per-cell resumable, so this cost zero
finished cells) and a diagnosed-and-fixed memory/concurrency issue (4-way parallel workers on the
largest cells, N∈{95,122} at T=3840, peaked near 9GB combined on this 15GB machine — fixed by dropping
to sequential execution) plus a real performance bug found along the way: `scipy.stats.norm.cdf`/`.ppf`
measured 3-4x slower than the equivalent `scipy.special.ndtr`/`ndtri` on large arrays, hit only by the
two non-Gaussian moment paths. Fixed in `dsr_lib17.py`.

Two disclosed deviations from the literal `distributions.frozen_dist` instruction, both because the
target moments are outside what `jf_skew_t` can produce: the moderate-moments regime's target
(skew=-1.5, total kurtosis=6.0) is not reachable at any finite kurtosis close to 6 — a boundary search
showed the closest fit at kurtosis≈6 lands at skew≈-1.21, not -1.5, and the achieved-vs-target values
are recorded per cell. The extreme-moments regime's target (skew=-11.5, kurtosis=817, 018's own
measured values) is not reachable by `jf_skew_t` **at all** — |skew| saturates near 5.1-5.2 as
kurtosis→∞ in that family — so a method-of-moments two-point jump mixture (rare deterministic downward
jump + Gaussian bulk) is used instead, solved to hit the target exactly.

**Phase 4 — adoption.** Applies sec 2.3's rule to Phase 3's DS-2 (calibration) and DS-3 (power)
results. **None of the four candidates is adopted** — see Results below. `research.py` is confirmed
unmodified (`git diff` empty before and after this phase).

**Phase 5 — re-score.** Hash-gated (sec 10) re-score of the measured 70-value inventory. The gate
passed automatically since `research.py`'s source hash never changed. **DS-4 fires**: all 70 rows
present, 65 with a ρ→1 upper bound below 0.95 (provably cannot flip, sec 5.4 item 2), 5 marked
`not_rescorable` (upper bound ≥0.95, but no adopted variant to compute a corrected value with). Zero
verdict changes.

**Phase 6b — the 018 amendment.** None of the three sec 14.2 branch texts frozen in Phase 0 literally
applies to this outcome (DS-1 fired — the defect is real — but no repair was adopted). Branch A's
premise ("DS-1 does not fire") is false; Branches B and C's premise ("repair adopted") is also false.
Pasting the nearest-sounding branch (A, since the mechanical consequence — `research.py` unmodified —
matches) would misstate the reason: Branch A says the concern didn't reproduce, and it did. Phase 6b
therefore writes a new, honest characterization instead (`phase_7_18_dsr_addendum.json`), holding
every sec 14.3 rule the three frozen branches share. See "018's amendment" below.

## Results

**The defect is real (DS-1 fires), but no repair is adopted.** V0's false-positive rate collapses
toward zero as inter-trial correlation rises — the over-rejection pattern this programme's own
methodology notes had flagged as a suspicion is confirmed as fact. Of the four candidate repairs:

- **V1** (the source paper's own cross-sectional-std repair) and **V1b** at both shrinkage settings
  pass calibration (DS-2) cleanly — zero violations across all 378 null cells. But they fail power
  (DS-3), and in the way its two-sided design exists to catch: at high correlation (ρ≥0.9) their power
  gain over V0 falls short of the required 10-percentage-point margin in 20 of the relevant cells
  (mostly smaller trial counts, N=12, at shorter sample lengths); and — the "no free lunch" clause —
  at ρ=0, where trials really are independent and V0 is correctly calibrated, V1/V1b's detection rate
  is actually *below* V0's by more than the allowed 2-point margin in another 20 cells (concentrated
  at long samples, T=3840, moderate trial counts). A repair that buys correctness at high correlation
  by giving up real power in the ordinary case does not pass, exactly as designed.
- **V2** (effective-trials via inter-trial correlation) passes DS-3 cleanly but fails DS-2a badly: 246
  of 378 null cells (65%) exceed the 0.075 anti-conservatism ceiling — it cries wolf far too often to
  be usable.

Per the pre-registered adoption rule (sec 2.3), **none of the four candidates is adopted.**
`research.py` is left completely unmodified — confirmed by `git diff` before and after Phase 4, not
merely asserted. This was an explicitly anticipated possible outcome (sec 2.1: "the honest answer is
that the estimator is not repairable at this scope") and is reported as such, not retuned.

**Phase 5's re-score is a near-total non-event, as expected when no estimator code changes.** All 70
stored DSR values are unchanged in value; DS-4 fires (all 70 accounted for, each either provably
incapable of flipping or explicitly marked `not_rescorable`). 65 rows have a ρ→1 upper bound below
0.95 and could never have flipped under any dispersion-based repair regardless of Phase 4's outcome.
The remaining 5 (gate AC, stored three times across three files; notebook 10b's "calendar" and
"gate_VS") have an upper bound ≥0.95 and would have needed a corrected value from an adopted variant —
but none exists, so all 5 are `not_rescorable`, explicitly reasoned rather than silently skipped.

**018's own case is settled more decisively than "no variant adopted" alone would suggest.** 018's
stored DSR *is* one of the 70 inventory rows (`phase_4_18_results.json` → `dsr.deflated_sharpe_prob`),
and its ρ→1 upper bound — computed from 018's own extreme sample skew (−11.5) and kurtosis (817),
which is exactly the regime Phase 0's "018_measured" moment axis exists to probe — is **0.83**, below
the 0.95 bar. That ceiling applies to every dispersion-based variant this notebook evaluated (Test 7,
sec 7.3: `psr_upper_bound` dominates every variant's output), adopted or not. So 018's DSR leg was
never actually contingent on Phase 4's adoption decision: even in the counterfactual where V1 had
cleared both DS-2 and DS-3, 018's corrected DSR could not have exceeded 0.83. FA-2, FA-3, and FUND
stand exactly as 018 recorded them; the holdout stays unspent. Full detail in
`src/research/tmp/phase_7_18_dsr_addendum.json` and the appended addendum sections in 018's own
notebook and write-up.

## Inventory discrepancy (Phase 0)

Sec 5.5 documents "73 stored values across 17 files," found by "a `deflated_sharpe_prob`-keyed sweep."
An exact-key sweep (the literal reading of that phrase) finds **70 values across 14 files**. A
secondary, disclosed-only fuzzy sweep (matching key-name *variants* like
`deflated_sharpe_prob_headline`, `deflated_sharpe_prob_best`, `published_deflated_sharpe_prob`,
`deflated_sharpe_prob_by_gate`) finds more, concentrated in notebook 011b's `phase_0/1/4/6/7`
result files — which appear to record the *same* underlying figures redundantly under inconsistent
key names across 011b's own pipeline stages, rather than being 011b-external values the exact sweep
missed. Per sec 5.5's own instruction ("must reproduce that count and fail loudly if it differs"),
this notebook uses the **measured** 70/14 for gate DS-4 rather than forcing agreement with 73/17. This
is itself a small record-keeping finding, alongside sec 5.4's existing recommendation that notebooks
store their trial Sharpe vectors: notebooks should also use one consistent key name for a stored DSR
value.

## Gate verdicts — the full table

| gate | claim | fires? | number behind it |
|---|---|:---:|---|
| **DS-1** (defect is real) | V0 over-rejects as ρ→1 | **YES** | FPR at ρ=0.9/0.99: 0.0010/0.0002 (≤0.005); non-increasing in ρ from 0.00375 baseline |
| **DS-2** (repair calibrated) | candidate's FPR stays in [0.010, 0.075] appropriately | **V1/V1b: yes. V2: no.** | V1, V1b(0.25), V1b(0.5): 0 violations of DS-2a/b across 378 null cells. V2: 246/378 cells exceed the 0.075 DS-2a ceiling; V1b(0.5) alone also has 38 DS-2b violations. |
| **DS-3** (repair has power) | candidate detects a true edge ≥10pp better than V0 at high ρ, without losing >2pp at ρ=0 | **V1/V1b: no. V2: yes.** | V1/V1b: 20 cells fall short of the high-ρ power margin, 20 cells lose more than 2pp at ρ=0. V2: passes both clauses. |
| — | **adopted variant** | **NONE** | No candidate passes both DS-2 and DS-3; `research.py` unmodified per the pre-registered adoption rule. |
| **DS-4** (ledger complete) | all 70 measured stored values re-scored or marked not_rescorable | **YES** | 70/70 rows present: 65 provably cannot flip (upper bound <0.95), 5 `not_rescorable` (upper bound ≥0.95, no adopted variant). 0 verdict changes. |

## What to test next

- **A fifth repair variant, or a narrower validated regime.** V1/V1b's power shortfall is concentrated
  at moderate trial counts (N≈12) and shorter samples; V2's miscalibration is broad. A repair
  restricted to, say, N≥18 and T≥1000 — the regime most of this repo's actual trial families sit in —
  might pass both gates where the general-purpose version doesn't. This would need its own
  pre-registration stating the validated range explicitly, not a retroactive narrowing of this one's
  gates (sec 12.3).
- **The sequential/nested-search independence failure this notebook explicitly does not address**
  (sec 12.5, sec 13.5): correlation is not the only way a trial family violates the "independent
  draws" assumption a DSR calculation makes. A search where the next config tried depends on what
  already worked inflates the effective search space in a way no dispersion estimate can detect. Every
  DSR figure in this repo, corrected or not, still carries that limitation.
- **Consistent key naming for stored DSR values**, per the inventory discrepancy above — future
  notebooks should use the literal key `deflated_sharpe_prob`, not a pipeline-stage-specific variant,
  so a future audit's exact-key sweep finds everything on the first pass.
- **018's own construction is not reopened by anything here** (sec 14.3) — its binding constraint
  remains the bootstrap-CI leg on net returns, a data-and-construction question 018's own
  what-to-test-next already names (lower-turnover carry, a diversification floor), not an estimator
  question.

---

## Erratum (2026-08-20, found by notebook 019)

`run_phase_4_17_adoption.py`'s `evaluate_variant` stores each gate clause's violation list truncated
to `violations[:20]`, and this write-up's DS-3 row above quotes the length of that truncated list. The
high-rho clause's count is therefore reported as 20 when the uncapped count is **78**; the rho=0 count,
also reported as 20, happens to be exact. This changes no verdict: V1/V1b already failed DS-3 either
way (20 or 78 both exceed zero), every downstream conclusion in this document stands unedited, and
neither `phase_3_17_calibration.json` nor `phase_4_17_adoption.json` was touched to produce this note —
per this document's own "Inventory discrepancy" precedent, the fact is disclosed here rather than the
frozen numbers above being patched in place. Full detail: `scratch/019/preflight.json` → `ds3_count_correction`, and
`src/results/019_dsr_correlation_switch.md`.

---

*Co-produced with Claude Sonnet 5. Notebook: `src/research/017_deflated_sharpe_correction.ipynb`.*
