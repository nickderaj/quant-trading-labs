# 017 — Diagnosing and Trying to Repair the Deflated Sharpe Estimator

## The problem

Every notebook in this programme uses a deflated Sharpe calculation to correct for how many
configurations were tried before reporting the best one. That correction needs a benchmark: how good
would the best of N trials look if none of them had any edge at all?

**This repo's implementation builds that benchmark from the sampling standard error of a single
Sharpe estimate, rather than from the observed spread of the trial family's Sharpes.**

That is correct when trials are genuinely independent. It is wrong whenever a trial family is
correlated — which is this repo's own standard robustness pattern, since near-identical origin
offsets produce near-identical Sharpes.

The immediate trigger: notebook 018's funding-basis test failed on its deflation leg (0.186 against a
0.95 bar) while passing its Sharpe leg cleanly, with a caveat field pointing directly at this issue.

## What this notebook does

1. **Establish by simulation whether the divergence is real** — a kill switch. If the current
   estimator behaves correctly under correlation, everything downstream is unnecessary.
2. **Repair it** if so, with three candidate fixes frozen before any simulation ran.
3. **Calibrate the repair** across a grid of trial counts, sample lengths, return moments and
   inter-trial correlation.
4. **Re-score every stored deflated Sharpe value in the repository**, to see whether any recorded
   verdict changes.

The three candidates, and the rule for choosing between them, were all fixed in advance:

- **Cross-sectional spread** — use the actual observed spread of the trial family's Sharpes, as the
  source paper specifies.
- **Effective trial count** — estimate how many *independent* trials the correlated family is
  equivalent to.
- **Cross-sectional spread with a shrinkage floor** — the pre-declared fallback for the first
  candidate's known small-sample hazard.

Adoption rule: take the simplest candidate that passes **both** calibration and power.

## The answer

**The defect is real. No repair is adopted.**

| Check | Question | Result |
|---|---|:---:|
| **Is the defect real?** | Does the current estimator misbehave as correlation rises? | **Yes** |
| **Is a repair correctly calibrated?** | Does it hold its false-positive rate in the acceptable band? | **The spread-based ones, yes. The effective-trials one, no** |
| **Does a repair have power?** | Does it detect a true edge better under correlation, without losing power when trials really are independent? | **The spread-based ones, no. The effective-trials one, yes** |
| **Is every stored value accounted for?** | All 70 re-scored or explicitly marked | **Yes — zero verdict changes** |

**No candidate passes both calibration and power**, so `research.py` is left completely unmodified —
confirmed by diffing before and after, not merely asserted. This was an explicitly anticipated
possible outcome: the honest answer may be that the estimator is not repairable at this scope. It is
reported as such rather than retuned until something passes.

---

## Establishing that the defect is real

Reproducing the regime of the notebook that triggered this — 18 trials over 3,840 bars — across the
full correlation axis at 20,000 Monte Carlo replications per cell.

**The current estimator's false-positive rate collapses toward zero as correlation rises**: 0.0010 at
ρ = 0.9 and 0.0002 at ρ = 0.99, against a baseline of 0.00375 at zero correlation, and non-increasing
in correlation at every intermediate point within simulation error.

The suspicion this programme's own methodology notes had flagged is confirmed as fact. **The
estimator becomes drastically over-conservative on exactly the trial families this repo produces most
often.**

## The calibration and power grid

7 correlation levels × 3 trial counts × 6 sample lengths × 3 moment regimes = 378 null cells, each
also run with an injected true edge, for **756 cells at 20,000 replications each**.

Two disclosed deviations from the planned return-generating distribution, both because the target
moments lie outside what that family can produce:

- The **moderate-moments** target (skew −1.5, kurtosis 6.0) is not reachable at any kurtosis near 6 —
  a boundary search shows the closest fit lands at skew ≈ −1.21. Achieved-versus-target values are
  recorded per cell.
- The **extreme-moments** target (skew −11.5, kurtosis 817 — the values actually measured in notebook
  018) is not reachable by that family **at all**; its skew saturates near 5.1–5.2 as kurtosis grows.
  A two-point jump mixture — a rare deterministic downward jump plus a Gaussian bulk — is used instead,
  solved to hit the target exactly.

Along the way, a real performance bug: the standard normal distribution functions measured 3–4×
slower than their lower-level equivalents on large arrays, hit only by the two non-Gaussian paths.
Fixed.

*(The run survived a machine reboot at no cost, since it is resumable per cell, and a diagnosed
memory issue where four parallel workers on the largest cells peaked near 9GB on a 15GB machine —
fixed by dropping to sequential execution.)*

## Why every repair fails

**The spread-based repairs pass calibration cleanly** — zero violations across all 378 null cells, at
both shrinkage settings.

**And they fail power, in exactly the way the two-sided test exists to catch.**

At high correlation, their power gain over the current estimator falls short of the required
10-percentage-point margin in a substantial number of cells, concentrated at smaller trial counts and
shorter samples.

And — the "no free lunch" clause — **at zero correlation, where trials really are independent and the
current estimator is correctly calibrated, the spread-based repairs' detection rate is actually
*below* it** by more than the allowed 2-point margin in another set of cells, concentrated at long
samples and moderate trial counts.

**A repair that buys correctness under correlation by giving up real power in the ordinary case does
not pass.** That is by design.

**The effective-trials repair passes power cleanly and fails calibration badly**: 246 of 378 null
cells (65%) exceed the anti-conservatism ceiling. **It cries wolf far too often to be usable.**

---

## Re-scoring every stored value

A mechanical sweep of the stored results found **70 deflated Sharpe values across 14 files.**

All 70 are unchanged in value, as expected when no estimator code changes. Every one is accounted
for:

- **65 have a theoretical upper bound below 0.95** and therefore could never have flipped under *any*
  spread-based repair, regardless of the adoption decision.
- **5 have an upper bound at or above 0.95** and would have needed a corrected value from an adopted
  variant. Since none exists, they are explicitly marked as not re-scorable — reasoned rather than
  silently skipped.

**Zero verdict changes.**

### An inventory discrepancy, disclosed

The plan documented "73 stored values across 17 files". An exact-key sweep — the literal reading of
that phrase — finds **70 across 14**.

A secondary fuzzy sweep, matching key-name *variants*, finds more — but those are concentrated in one
notebook's pipeline-stage files and appear to record the **same underlying figures redundantly under
inconsistent key names**, rather than being values the exact sweep missed.

The **measured** count is used rather than forcing agreement with the documented one. That is itself a
small record-keeping finding: notebooks should use one consistent key name for a stored deflation
value, so a future audit's exact sweep finds everything on the first pass.

---

## Notebook 018's case is settled more decisively than expected

Its stored deflation value is one of the 70. Its theoretical upper bound — computed from its own
extreme sample skew (−11.5) and kurtosis (817), which is exactly the regime the extreme-moments axis
exists to probe — is **0.83**, below the 0.95 bar.

**That ceiling applies to every spread-based variant evaluated here, adopted or not.**

So notebook 018's deflation leg was never actually contingent on the adoption decision. Even in the
counterfactual where a repair had passed both checks, its corrected value could not have exceeded
0.83.

Its verdicts stand exactly as recorded, and the holdout stays unspent.

---

## The amendment to notebook 018

Three possible amendment texts were frozen in advance, and **none of them literally applies to this
outcome.** One assumed the defect wouldn't reproduce — it did. The other two assumed a repair would be
adopted — none was.

Pasting the nearest-sounding one would misstate the reason: it says the concern didn't reproduce,
and the concern *did* reproduce. So a new, honest characterisation was written instead, holding to
every rule the three frozen texts share.

---

## What to test next

- **A narrower validated regime.** The spread-based repairs' power shortfall is concentrated at
  moderate trial counts and shorter samples; the effective-trials repair's miscalibration is broad. A
  repair restricted to the regime most of this repo's actual trial families sit in — say 18 or more
  trials and 1,000 or more observations — might pass both checks where a general-purpose version
  doesn't. That would need its own pre-registration stating the validated range explicitly, not a
  retroactive narrowing of this one.
- **The sequential-search independence failure this notebook explicitly does not address.**
  Correlation is not the only way a trial family violates the independent-draws assumption. A search
  where the *next* configuration tried depends on what already worked inflates the effective search
  space in a way no spread estimate can detect. **Every deflation figure in this repo, corrected or
  not, still carries that limitation.**
- **Consistent key naming for stored values**, per the discrepancy above.
- **Notebook 018's construction is not reopened by anything here.** Its binding constraint remains the
  bootstrap interval on net returns — a data and construction question, not an estimator question.

---

## Erratum (found later, by notebook 019)

The adoption script stores each violation list truncated to the first 20 entries, and this write-up
originally quoted the length of that truncated list. The high-correlation clause's count is therefore
reported as 20 where the uncapped count is **78**. The zero-correlation count, also reported as 20,
happens to be exact.

**This changes no verdict** — the spread-based repairs already failed on power either way, since both
20 and 78 exceed zero, and every downstream conclusion stands unedited. Following the same precedent
as the inventory discrepancy above, the fact is disclosed here rather than the frozen numbers being
patched in place.

*Notebook: `src/research/017_deflated_sharpe_correction.ipynb`.*
