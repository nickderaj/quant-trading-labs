# Notebook 017 — The Deflated Sharpe estimator, diagnosed and repaired: Results Summary

**DRAFT — Phases 0-2 complete, Phases 3-6b in progress. This file is a working skeleton, not a
finished write-up. Sections marked TODO are written after their phase's results land.**

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

**Phase 3 — full calibration + power grid.** TODO. 7×3×6×3 = 378 null cells × {null, injected-edge} =
756 cells, M=20000, `scripts/run_dsr_calibration.sh`. *Note for the write-up:* the moderate-moments
regime's target (skew=-1.5, total kurtosis=6.0) is not reachable by `distributions.frozen_dist`'s
`jf_skew_t` family at any finite kurtosis close to 6 — a boundary search showed the closest fit at
kurtosis≈6 lands at skew≈-1.21, not -1.5, and the achieved-vs-target values are recorded per cell. The
extreme-moments regime's target (skew=-11.5, kurtosis=817, 018's own measured values) is not reachable
by `jf_skew_t` **at all** — |skew| saturates near 5.1-5.2 as kurtosis→∞ in that family — so a
method-of-moments two-point jump mixture (rare deterministic downward jump + Gaussian bulk) is used
instead, solved to hit the target exactly. Both deviations from the literal `frozen_dist` instruction
are disclosed in `dsr_lib17.py`'s module docstring and reproduced here for visibility.

**Phase 4 — adoption.** TODO. Applies sec 2.3's rule to Phase 3's DS-2/DS-3 results.

**Phase 5 — re-score.** TODO. Hash-gated (sec 10) re-score of the measured inventory.

**Phase 6b — the 018 amendment.** TODO. Selects one of the three frozen branch texts.

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

TODO (after Phase 5).

| gate | claim | fires? | number behind it |
|---|---|:---:|---|
| **DS-1** (defect is real) | V0 over-rejects as ρ→1 | **YES** | FPR at ρ=0.9/0.99: 0.0010/0.0002 (≤0.005); non-increasing in ρ from 0.00375 baseline |
| **DS-2** (repair calibrated) | adopted variant's FPR stays in [0.010, 0.075] appropriately | TODO | |
| **DS-3** (repair has power) | adopted variant detects a true edge ≥10pp better than V0 at high ρ | TODO | |
| **DS-4** (ledger complete) | all 70 measured stored values re-scored or marked not_rescorable | TODO | |

## What to test next

TODO.

---

*Co-produced with Claude Sonnet 5. Notebook: `src/research/017_deflated_sharpe_correction.ipynb`.*
