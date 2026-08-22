# 019 — What a Correlation-Triggered Deflation Switch Can and Cannot Fix

## The idea

Notebook 017 found a real defect in this repo's deflated Sharpe calculation — it over-penalises
correlated trial families — and tested three repairs. Two of them fixed calibration but **lost real
detection power in the ordinary independent-trials case.** The third had power but was badly
miscalibrated. None was adopted.

That failure had a specific shape worth attacking. The spread-based repairs only *needed* to apply
when trials are actually correlated. When they aren't, the existing estimator is already correct.

So: **a switch.** Use the spread-based repair only when a cheap estimate of the trial family's
inter-trial correlation says the family really is correlated, and fall back to the existing estimator
otherwise.

The question is whether that switch separates the repair's two failure modes — fixing the
independent-case power loss, which turns out to be pure estimation noise, without pretending to fix
the high-correlation shortfall, which is not.

## What was known before any new simulation ran

An inspection pass over notebook 017's frozen results, disclosed in full in the pre-registration
**before** any new simulation, found two things:

**The independent-case power loss is concentrated exactly where a correlation estimate should read
near zero**, and routing those cells to the existing estimator removes it completely — *in sample*.

**No threshold could ever fix the high-correlation shortfall.** This is provable analytically: for any
threshold below 0.9, the switch **is** the spread-based repair, bit for bit, in every cell that clause
reads. And that repair already fails there.

That leaves exactly one real, falsifiable question: **does the in-sample fix survive contact with grid
points notebook 017 never ran?**

## Results

| Check | Question | Result |
|---|---|:---:|
| **Does the switch reduce correctly?** | Unit tests on the boundary behaviour | **Passes** |
| **Does the switch engage correctly?** | Near-zero false triggers when trials are independent; no power loss there | **Passes** at both thresholds |
| **Is the cheap prediction trustworthy?** | Do predicted and measured rates agree? | **Passes decisively** — 46 of 46 within 3 combined standard errors |
| **Does the claimed regime survive out of sample?** | Zero violations at grid points never previously run | **Fails** — by exactly one cell |
| **Full-scope adoption?** | **No** — analytically proven and confirmed |
| **Restricted-regime adoption?** | **No** — misses at the regime's own boundary |

**No variant is adopted. `research.py` is confirmed unmodified.**

## How the switch was built

The switch delegates rather than re-derives: below the correlation threshold it calls the existing
estimator's branch, above it calls the spread-based repair's branch — as a genuine recursive call, so
its output is **bit-for-bit** whatever those produce alone, with no possibility of drift.

The boundary itself resolves to the existing estimator (inclusive at the threshold), following the
executable test rather than the plan's pseudocode, which showed the stricter comparison. The test is
the authoritative specification.

Two thresholds were evaluated, 0.15 and 0.30. The simulation loop selects per replication from the
correlation estimate it already computes — **no second correlation pass.**

## Does the switch engage correctly?

Measured across all 378 design points in both modes:

**The false-trigger rate when trials are genuinely independent is exactly 0.0** — at both thresholds,
at every one of the 63 design points checked, against a bar of 0.005. The estimate's own standard
deviation tops out at 0.024, so the lower threshold sits more than **6 standard deviations away even
at its worst point.**

And across all 18 confirmed independent-case cells with a true edge injected, there is **zero
detection-rate shortfall** against the existing estimator.

**The switch mechanism is exactly correct.**

## Predicting the answer before measuring it

Running the full 756-cell grid again would spend hours reproducing an answer largely visible for free:
the switch's branch selection is deterministic almost everywhere. It never once selects the repair
branch at true zero correlation across 63 measured design points, and is essentially always in the
repair branch by correlation 0.5.

So every cell got a **mixture-predicted** rate — the repair's rate weighted by the measured branch
probability, plus the existing estimator's rate weighted by the complement — run through notebook
017's own evaluation code unmodified. This prediction was written down **before** the confirmation
grid ran.

The predicted verdict: full-scope adoption **fails at both thresholds**, exactly as pre-registered.
The high-correlation clause is inherited from the repair verbatim.

The predicted restricted-regime finding: both thresholds pass every clause down to **12 trials and
3,840 observations**, where the repair alone needed 95 trials. Still a prediction, not evidence.

## The confirmation grid

142 cells across three pre-declared subsets, all with seeds explicitly namespaced apart from notebook
017's — **genuinely independent replications, not replays.** Run single-core by design, taking the
earlier out-of-memory lesson seriously.

- **96 cells at points never previously run**, at axis values that don't exist in the original grid,
  every one inside the claimed regime by construction.
- **22 cells at the ambiguous design points** the branch-probability measurement identified, all in the
  extreme-moments regime.
- **24 cells drawn deterministically from the original grid** as a control sample.

### The prediction held up

**46 of 46 predictable comparisons (100%) agree with measurement within 3 combined standard errors**,
with the worst disagreement at 2.35 — well inside both the 95%-within-3 and no-disagreement-over-5
bars.

**The cheap mixture prediction was trustworthy.** That is a methodological result in its own right:
it means an expensive grid can be predicted rather than re-run, when branch selection is nearly
deterministic.

### And the regime claim did not

Across all 96 out-of-sample cells, **exactly one violates a clause**: 12 trials, 5,000 observations,
correlation 0.9, in the extreme-moments regime. The power gain over the existing estimator is
**8.93 percentage points against a required 10** — a shortfall of 1.07 points, about 2.57 combined
standard errors. **A real but modest miss, not a blowout.**

Every other cell clears comfortably — all three moment regimes at 14, 24 and 50 trials, and both the
Gaussian and moderately non-Gaussian regimes even at 12 trials. The next point up, 14 trials in the
*same* hardest regime, clears at 11.43 points.

Zero calibration violations anywhere.

## Verdict

**Full-scope adoption fails, exactly as pre-registered and analytically proven.** For any threshold
below 0.9, the switch is bit-for-bit the spread-based repair in every cell the high-correlation clause
reads — measured at 100% branch selection above correlation 0.9 across the whole grid — and that
repair already fails there. **This is not contingent on the prediction's accuracy; it is a property of
the code.**

**Restricted-regime adoption fails, but not by much, and not everywhere.** The in-sample finding that
started this notebook — that the switch widens the usable regime from 95 trials down to 12 — **does not
survive out-of-sample confirmation exactly as claimed.** It survives almost everywhere: 95 of 96 cells
pass cleanly, including the hardest regime at every trial count from 14 up. It fails at exactly the
regime's own boundary value, in exactly the hardest moment regime, by a modest margin.

**Per the frozen rule, that is not licence to narrow the boundary to 14 after the fact.** The honest
report is that **the 12-trial boundary as pre-registered did not replicate.** Full stop.

**But the two-part diagnosis notebook 017 could not reach alone did land.** The repair's two failure
modes really are separable, and one of them really is a fixable estimation-noise artefact. The switch
mechanism is exactly correct, with zero false triggers and zero independent-case shortfalls measured
at fresh grid points. And the high-correlation shortfall is **not** an artefact of any threshold
choice — it survives every threshold considered, exactly as proven before the confirmation grid ran.

## A separate finding about record-keeping

Even setting adoption aside, the re-scoring pass found something worth stating.

Of notebook 017's 70-row inventory, **only 5 rows have any per-trial data recoverable at all — and all
5 store per-trial *Sharpes*, never the trial *return series* the correlation estimate needs.**

So even in the counterfactual where restricted adoption had succeeded, **zero of the 70 stored values
could have been re-scored.** All 5 recoverable rows also have trial counts of 4 or 8, below the
regime's own floor — a second, independent exclusion.

That is the honest ceiling, flagged before any simulation ran, and now made concrete.

## What to test next

- **A narrower boundary, pre-registered fresh.** 14 trials already clears the high-correlation margin
  comfortably in the exact regime where 12 misses it, at 11.43 points against a 10-point bar. That
  looks like a plausible candidate boundary — but adopting it retroactively here is forbidden. It would
  need its own pre-registration and its own out-of-sample confirmation grid: exactly this notebook's
  methodology applied one boundary value tighter.
- **Store trial return series as a convention.** Every notebook whose trial family might ever be worth
  re-scoring under a correlation-aware repair should store the return series, not just summary
  Sharpes. Notebook 017 flagged this; this notebook makes the cost concrete — literally zero of 70
  historical values could be re-scored, for exactly this reason.
- **The sequential-search independence failure**, named again because nothing here touches it.
  Correlation is not the only way a trial family violates the independent-draws assumption, and no
  spread or correlation estimate detects search-path dependence.
- **Notebook 018's case is not reopened by anything here.** Its ceiling of 0.83, re-verified before
  anything else ran, settles it independently of any variant, adopted or not.

*Notebook: `src/research/019_dsr_correlation_switch.ipynb`.*
