# 010a — Term-Structure Regimes and the Spread Taxonomy

## Purpose

Notebook 009's cheap probe found mean reversion in five of six commodity spreads. Before spending a
full costed backtest on that, three things need doing properly:

1. **Classify the spread universe.** Not all 30 pre-built spread series are the same kind of
   object, and treating them as one group would be a real error.
2. **Apply the cointegration precondition** that probe skipped, which should resolve the two
   spreads whose test results disagreed.
3. **Lock in the regime definitions and trading rule in advance**, so the eventual backtest can't
   be accused of picking its rules after seeing results.

**This notebook is descriptive only — no Sharpe ratios, no cost model, no verdicts.** Its actual
deliverable is the pre-registration for notebook 010b, committed here and unalterable once that
notebook starts running.

Development window only (2010-06-06 to 2024-12-31). The holdout period is never read here, even
though nothing in this notebook could spend it.

## Three findings that carry forward

**The cointegration check cuts the tradeable universe and resolves an open question.** Applying it
for the first time in this programme drops the inter-commodity backtest universe from 11 spreads to
7, and settles notebook 009's flagged disagreement: gold–silver and platinum–palladium both fail
cointegration outright (test statistics of −1.76 and −1.41 against a 5% critical value of −2.86).
They are not actually cointegrated pairs, which is consistent both with their weak mean-reversion
significance and their insignificant correlation. Both are excluded on this mechanical,
pre-declared criterion.

**The raw-sign regime definition is too weak to test the hypothesis it's meant to test.** The raw
sign of the curve slope is defined on essentially every trading day — only a slope of exactly zero
is undefined. Gating a book on it barely differs from trading unconditionally, so it cannot test the
actual claim, which is about trading only in a *definite* regime state. A **deadband** definition,
with a real "flat, no-trade" zone, is therefore promoted to primary, with raw sign and a persistence
requirement kept as secondary robustness variants.

This is a structural argument available at design time, not a result-driven change. All three
variants are computed and reported below regardless of which becomes primary, so nothing was chosen
by looking at which performed best.

**Brent–WTI's regime effect depends materially on which leg's curve defines the regime** — weak or
reversed under one leg alone, strong under a both-legs-agree definition. Left as a live, open
question for the backtest to settle rather than smoothed into either "confirmed" or "refuted".

## First: does the prior work reproduce?

Eight assertions re-derived from committed results before anything was built on top of them:
notebook 009's probe (5 of 6 spreads mean-reverting, 4 of 6 with a significant negative correlation)
and notebook 008's carry and momentum headline numbers (both failing; carry net Sharpe 0.9042–0.9459
across four offsets, deflated probability 0.9972, excess-return interval [−0.0028, +0.0094]
including zero; momentum deflated probability 0.098). All passed on the first run.

---

## The term-structure regime atlas

Annualised front-to-second-month roll slope, state label, state persistence and month-of-year
pattern, for all 16 products.

**Backwardation frequency varies enormously by sector, and even within it.**

Energy is the most backwardation-prone sector but far from uniform: gasoline spends 67.5% of days
backwardated, Brent 60.7%, heating oil 41.8%, crude 38.4% — but natural gas only 20.5%, since
heating-season contango dominates its curve most of the year.

Metals sit at the other extreme, in near-permanent contango exactly as cost-of-carry theory predicts
for a low-storage-cost, low-convenience-yield group: gold 19.9%, silver 14.8%, platinum 13.0%,
palladium 32.2%.

Grains are the most internally mixed: wheat just 5.7% backwardated, soybeans 38.6%, soy meal 52.3%.

**This dispersion is itself the reason a single repo-wide regime rule would be a mistake.** Treating
"commodities" as one regime-homogeneous group would average away exactly the product-specific
structure the atlas exists to surface.

**State persistence is long enough for a regime-gated strategy to be plausible, without being so
long that the regime carries no information.** Mean run lengths range from a few days for thin,
choppy products up to 50–95 days (crude's contango runs about 58 days; corn's about 95). That is
comparable to or longer than the 46–85-day mean-reversion half-lives notebook 009 measured, so a
regime label is unlikely to flip mid-trade on most positions.

Curve snapshots were captured for crude, natural gas and corn — the deepest observed contango and
deepest observed backwardation day for each, with the full front three months — making the concept
concrete rather than a slope number.

---

## Taxonomy and cointegration

All 30 spreads classified from their own metadata rather than a hardcoded name list: **11
inter-commodity** (two distinct underlyings) and **19 calendar** spreads (the term structure itself,
both legs the same product).

**This distinction matters and conflating it would be a real error.** The regime hypothesis is only
meaningful for the inter-commodity group. Gating a *calendar* spread on contango versus
backwardation is close to conditioning a signal on its own sign. No calendar spread is used as
evidence for the regime hypothesis anywhere in this notebook.

**Cointegration: 23 of 30 spreads pass at 5%.** Failures split unevenly by type — **4 of 11
inter-commodity** (gold–silver, platinum–palladium, the heating-oil crack, and the KC–Chicago wheat
spread) against **3 of 19 calendar** spreads. Notably, two of the three calendar failures are gold's
own nearest calendar spreads, echoing gold's weak inter-commodity result.

Failures are excluded from the backtest universe, leaving **7 of 11 inter-commodity and 16 of 19
calendar spreads** eligible.

**The mean-reversion probe extends cleanly from 6 spreads to all 30: 27 of 30 mean-revert on the
regression test, and 16 of 30 show a significant negative forward correlation.**

The 11 spreads where the two descriptive tests disagree — the regression says mean-reverting, the
correlation doesn't confirm — include the bean–corn and gas–heat spreads and eight calendar spreads.
That is a genuinely wider disagreement than notebook 009's two-spread version, reported in full
rather than folded into the "27 of 30" headline.

**Three-way agreement across the regression test, the correlation test and cointegration is what
decides eligibility**, not any single test in isolation. That is exactly why the cointegration check
notebook 009 never ran had to be added.

---

## Does a spread actually mean-revert harder in one regime state?

Inter-commodity spreads only, conditioned on the first leg's own curve — a fixed rule applied
identically to every spread — with all three regime definitions computed and reported.

**Under raw sign, 8 of 11 inter-commodity spreads show nominally stronger mean reversion in
backwardation than contango.** Reported for completeness, but it should not be over-read: raw sign
restricts almost nothing, since backwardation and contango together cover essentially all days. A
deadband-gated comparison is a considerably stronger test.

**Brent–WTI tells a genuinely two-sided story, and it's the most important finding here.**

Under the primary rule — Brent's own curve alone — mean reversion is *nominally stronger in
contango* (coefficient −0.035) than in backwardation (−0.010). On its face, the opposite sign from
the hypothesis.

But under the both-legs-agree variant, where Brent's and WTI's curves must label the same state, the
picture flips and sharpens considerably:

| State | Half-life |
|---|---|
| Both legs agree on backwardation | **9.5 days** |
| Both legs agree on contango | 18.3 days |
| The two legs disagree | 47.5 days |
| Unconditional | 79.3 days |

Both legs agreeing on backwardation is by a wide margin the fastest-reverting state found anywhere
in this phase — and the legs agree on 66.2% of trading days, so it isn't a thin-sample artefact.

**Reported as a live, unresolved tension for the backtest to settle empirically**: a single-leg
regime definition does not show the effect for this spread, while a both-legs-agree definition shows
it strongly. That exact comparison is carried forward as a declared secondary check.

Calendar spreads received the same machinery purely as a **labelled circularity diagnostic**, never
used as evidence for the regime hypothesis.

---

## Positioning data — one product only

The positioning cache holds exactly one series: light sweet crude. A single-product check, never
extrapolated into a panel claim.

**Positioning corroborates the regime label for crude.** Mean net non-commercial share of open
interest is 18.8% in backwardation against 15.9% in contango, and the correlation between roll slope
and net non-commercial share is −0.073 — the theory-consistent sign, since a negative roll slope *is*
backwardation, so speculators run more net-long exactly when the market is backwardated, as both
normal-backwardation theory and the inventory mechanism predict.

The p-value on the difference is astronomically small, but with roughly 3,600 joined days that
reflects sample size as much as effect size, and the correlation's magnitude is modest. This is
corroborating, directionally consistent evidence for one product — not an independently decisive test.

---

## The pre-registration

Written and committed here, before any backtest exists.

**Regime definitions.** Deadband (a ±2% annualised slope threshold — a pre-declared round convention,
not swept) is **primary**. Raw sign and a 5-day persistence requirement are secondary robustness
variants. The primary regime leg is the first leg, fixed and identical across every spread, with a
both-legs-agree variant running only for Brent–WTI.

**Trading rule.** A 60-day rolling z-score signal, reusing the same window as the descriptive probe
rather than re-tuned. Position equals the negative of the z-score, clipped to ±2 and halved. Daily
rebalance. **Two round-turn costs per unit of weight change** — one per leg. Roll-window rows
excluded from both signal and P&L. Four origin offsets: 0, 7, 14 and 21 days.

**Excluded universe.** Cointegration failures are out, as applied above.

**Trial counts for the deflation**, cumulative and per test:

| Test | Trials | Breakdown |
|---|---|---|
| Base spread mean reversion | 8 | 2 taxonomy groups × 4 origin offsets |
| Regime-conditional | 12 | 3 regime definitions × 4 offsets, inter-commodity only |
| Brent–WTI both-legs variant | +1 | Run once as a diagnostic, not itself a search |
| Volatility-scaled carry | 8 | 4 already-logged carry configurations carried forward + 4 new |
| Blended momentum | 20 | 16 already-logged momentum configurations + 4 new |

**One ambiguity, flagged rather than silently resolved.** The plan described the momentum test's
carried-forward count as 4 already-logged single-lookback configurations. The number that actually
fed notebook 008's own deflation was **16** — 4 lookbacks × 4 origin offsets, verified directly. The
larger, technically accurate historical figure is used, which makes the bar **harder** to clear, not
easier.

Also logged for transparency, though not counted toward any test's trial count since neither is a
performance-driven search: the 30 spreads mechanically screened by the cointegration criterion, and
the 33 regime-conditional descriptive runs, all reported regardless of outcome.

---

## Bugs found

One, caught by unit testing before it reached any reported number. The deadband regime function
initially returned an unevaluated query expression instead of a materialised series. The underlying
logic was correct throughout; only the return type was wrong, and it was wrong in a way that raises
a hard error on first real use rather than silently producing a bad number.

## Bottom line

No strategy verdict belongs here, by design. Three findings carry directly forward:

1. The cointegration precondition cuts the inter-commodity universe from 11 spreads to 7 and
   resolves notebook 009's flagged disagreement outright — neither of those two pairs is actually
   cointegrated.
2. The raw-sign regime definition is structurally too close to regime-blind to test the hypothesis,
   which is why the deadband definition is declared primary before any backtest exists.
3. Brent–WTI's regime effect depends materially on which leg's curve defines the regime — weak or
   reversed under one leg, strong (9.5-day half-life in backwardation) under both — an open,
   pre-registered question for a real costed backtest to settle rather than a descriptive
   correlation.

*Notebook: `src/research/010a_term_structure_regimes_and_spreads.ipynb`.*
