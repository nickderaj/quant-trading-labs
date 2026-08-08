# Notebook 10a — Term-Structure Regimes and the Spread Taxonomy: Results Summary

## What

This notebook builds the descriptive groundwork for a future gated backtest of the commodity spread mean-reversion candidate identified in notebook 9: a term-structure regime atlas across all 16 products, a taxonomy classifying all 30 pre-built spread series as inter-commodity or calendar spreads, a cointegration precondition check on those spreads, an initial look at whether spreads mean-revert harder in specific term-structure regimes, and a check of CFTC positioning data against the regime label. It deliberately produces no Sharpe ratios, cost model, or gate verdicts — its actual deliverable is pre-registering notebook 10b's complete gate table, regime definitions, and trading rule before any backtest is run.

## Why

Notebook 9's cheap first-look probe found a promising mean-reversion signal in 5 of 6 commodity spreads but skipped a cointegration precondition and left two spreads (gold_silver, platinum_palladium) with disagreeing test results unresolved. Before committing to a full, costed backtest in notebook 10b, the programme needed to properly classify the spread universe, apply the missing cointegration check, and lock in regime definitions and trading rules in advance — preventing the kind of after-the-fact rule selection that would undermine the eventual gate's credibility.

## How

Using term-structure state machinery already built in notebook 8 (`commod_lib8.term_structure_state`) plus new machinery (`spread_lib10.py`) for ADF cointegration testing, three regime definitions (raw sign, deadband, persistence), and regime-conditional statistics, the notebook computed a term-structure regime atlas for all 16 products (Phase 1), classified and cointegration-tested all 30 spreads (Phase 2), tested whether inter-commodity spreads mean-revert differently by regime (Phase 3), cross-checked CFTC positioning data for crude oil against the regime label (Phase 4), and wrote a formal pre-registration of notebook 10b's gates, regime definitions, and trading rule (Phase 5).

## Results

23 of 30 spreads passed the cointegration test at 5% significance, resolving notebook 9's disagreement: gold_silver and platinum_palladium both failed cointegration outright and are excluded from the future backtest universe. The deadband regime definition was promoted to primary (over raw sign, which restricts almost no trading days and barely tests the intended hypothesis) for structural reasons decided before any backtest result was seen. Backwardation frequency varied enormously by sector and even within it, supporting a per-spread rather than repo-wide regime rule. Brent-WTI showed a genuinely two-sided result depending on which leg's curve defines the regime — weak-to-reversed under one leg alone, strongly regime-dependent (9.5-day half-life in backwardation) when both legs agree — left as an open, pre-registered question for notebook 10b. CFTC positioning data for crude oil corroborated the regime label directionally, though with modest effect size.

**This notebook is descriptive only — no Sharpe ratios, no cost model, no gate verdicts
(NEXT_PROMPT.md sec 1 rule 1).** Its purpose is to build the term-structure regime atlas,
classify all 30 pre-built spread series, apply the cointegration precondition notebook 9's
probe skipped, and — its actual deliverable — pre-register 10b's full gate table, regime
definitions, trading rule, and cumulative DSR configuration count **before any backtest
exists**. That pre-registration (Phase 5, `phase_5_10a_results.json`) is committed as part
of this notebook and may not be edited once 10b starts running.

**One correction made at pre-registration time, not after seeing a backtest result.** The
raw sign of `commod_lib8.term_structure_state` is defined on essentially every trading day
(only a slope of exactly zero is null) — gating a book by raw sign alone barely differs
from trading it unconditionally, and cannot test the operator's actual claim ("only in a
*definite* state," NEXT_PROMPT.md sec 0). Phase 5 therefore promotes the **deadband**
regime definition (a real "flat, no-trade" zone) to primary for Gate SPR, with raw sign
and a persistence requirement kept as secondary robustness variants. This is a structural
argument available at design time, not a result-driven change — Phase 3 already computed
and reports all three variants regardless of which one becomes primary, so nothing here
was chosen by looking at which definition performed best.

**The taxonomy matters, and conflating it would have been a real error.** 11 of the 30
spreads are inter-commodity (two distinct underlyings); 19 are calendar spreads (the
term structure *itself*, both legs the same product). Per sec 4.2, the regime hypothesis
is only meaningful for the inter-commodity group — gating a calendar spread on
contango/backwardation is close to conditioning a signal on its own sign, and this
notebook never uses a calendar spread as evidence for the regime hypothesis.

**The cointegration precondition, applied for the first time in this programme, resolves
notebook 9's own flagged disagreement.** Notebook 9's Phase 4 probe found gold_silver and
platinum_palladium disagreeing between the AR(1) mean-reversion test and the z-score IC
test. Both now fail the ADF cointegration test outright (t = −1.76 and −1.41 vs. the 5%
critical value of −2.86) — they are not actually cointegrated pairs, which is consistent
with both their weak AR(1) significance and their insignificant IC. Both are excluded from
10b's backtest universe on this pre-declared, mechanical criterion.

Machinery: `src/research/tmp/run_phase_{0..5}_10a_*.py` (Phase 0 reproduction; Phase 1
regime atlas; Phase 2 taxonomy/cointegration; Phase 3 regime-conditional structure; Phase
4 COT positioning; Phase 5 pre-registration), `src/research/tmp/spread_lib10.py` (the new
computation this notebook needed — ADF cointegration test, three regime definitions,
taxonomy classification, rolling leg correlation, regime-conditional mean-reversion/vol —
41 unit tests in `tests/test_spread_lib10.py`). Development window only throughout
(2010-06-06 to 2024-12-31, ES from 2018-01-01, KE from 2013-12-16, matching notebook 8's
own convention) — **the 2025-01-01 → 2026-07-28 holdout is never read in this notebook**,
even though it is purely descriptive and produces no strategy verdict.

---

## Phase 0 — Reproduction check

Two already-committed results re-derived and asserted to match before anything was built
on top of them (`run_phase_0_10a_repro.py`): notebook 9's Phase 4 spread probe (5 of 6
spreads mean-reverting, 4 of 6 with a significant negative IC — crack_321, gold_silver,
brent_wti, corn_wheat, platinum_palladium, crush_soy) and notebook 8's Gate AC/AM headline
numbers (`fires: false` for both; carry net Sharpe 0.9042–0.9459 across all four origin
offsets, deflated Sharpe probability 0.9972, excess-vs-basket CI [−0.0028, +0.0094]
including zero; momentum deflated Sharpe probability 0.098). All eight assertions passed
on the first run.

---

## Phase 1 — The term-structure regime atlas

Annualised F1→F2 roll slope, state label, state persistence, and month-of-year pattern for
all 16 products, `commod_lib8.term_structure_state` unmodified.

**Backwardation frequency varies enormously by sector, and even within it.** Energy is the
most backwardation-prone sector but far from uniform: RB (gasoline) spends 67.5% of days
backwardated, BZ 60.7%, HO 41.8%, CL 38.4%, but NG only 20.5% (heating-season contango
dominates NG's own curve most of the year). Metals sit at the other extreme — near-
permanent contango, exactly as docs/09's own cost-of-carry worked example predicts for a
low-storage-cost, low-convenience-yield group: GC 19.9%, SI 14.8%, PL 13.0%, PA 32.2%.
Grains are the most internally mixed sector: ZW just 5.7% backwardated, ZS 38.6%, ZM
52.3%. **This dispersion is itself the reason a single, repo-wide regime rule would be a
mistake** — treating "commodities" as one regime-homogeneous group would average away the
real, product-specific structure the atlas exists to surface.

**State persistence is long enough to make a regime-gated strategy's turnover plausible,
not so long that the regime carries no information.** Mean run length ranges from a few
days (thin, choppy products) to 50–95 days (CL contango runs ~58 days, ZC contango runs
~95 days) — comparable to or longer than the 46–85-day half-lives notebook 9's probe found
for spread mean-reversion, meaning a regime label is unlikely to flip mid-trade on most
positions.

Curve snapshots captured for CL, NG, and ZC (deepest observed contango and deepest
observed backwardation day each, full F1/F2/F3 term structure) — the figure sec 7 calls
"the single most explanatory chart," making the concept concrete rather than a slope
number.

---

## Phase 2 — Spread taxonomy, cointegration, and the extended mean-reversion probe

All 30 spreads, classified from their own `leg_roles` metadata (not a hardcoded name
list): **11 inter-commodity, 19 calendar** — matching NEXT_PROMPT.md sec 4.2's own rough
count.

**Cointegration (ADF, constant-only case, MacKinnon asymptotic critical values, BIC-
selected augmentation lags — `spread_lib10.adf_test`): 23 of 30 spreads pass at 5%.**
Failures split unevenly by taxonomy: **4 of 11 inter-commodity spreads fail**
(gold_silver, platinum_palladium, heating_oil_crack, kc_chicago_wheat) against **3 of 19
calendar spreads** (es_calendar, gc_cal_m1m2, gc_cal_m2m3 — notably, gold's own two nearest
calendar spreads, echoing gold's already-weak inter-commodity cointegration result above).
**Per sec 4.3's decision, made here before any 10b backtest: ADF failures are excluded
from Gate SP/SPR's backtest universe**, leaving **7 of 11 inter-commodity spreads and 16
of 19 calendar spreads** eligible.

**The AR(1)/IC probe extends cleanly from notebook 9's 6 spreads to all 30: 27/30 mean-
reverting on AR(1) (|t|>2), 16/30 with a significant (p<0.05) negative 5-day-forward
z-score IC.** The 11 spreads where the two descriptive tests disagree (AR(1) says
mean-reverting, IC does not confirm) include bean_corn, gasheat_rbho, and eight calendar
spreads (cl_cal_m2m3, ho_cal_m2m3, ng_cal_m1m2, ng_cal_m2m3, ng_calendar, rb_cal_m2m3,
wti_calendar, zc_cal_m1m2) — a genuinely wider disagreement than notebook 9's 2-spread
version, reported here in full rather than smoothed into the "27/30 mean-reverting"
headline. The three-way agreement across AR(1), IC, and now ADF is what actually decides
10b eligibility, not any single test in isolation — exactly why sec 4.3 required adding
the cointegration check notebook 9 never ran.

---

## Phase 3 — Regime-conditional structure: does the spread actually mean-revert harder in one state?

Inter-commodity spreads only (per sec 4.2), conditioned on **leg1's own curve** (a fixed,
pre-declared rule applied identically to every spread — for brent_wti, BZ), all three
regime definitions computed and reported.

**Under raw sign — the definition Phase 5 explicitly does NOT use as Gate SPR's headline,
for the structural reason above — 8 of 11 inter-commodity spreads show nominally stronger
AR(1) mean reversion in backwardation than in contango.** This number is reported for
completeness but should not be over-read: raw sign restricts almost nothing (backwardation
and contango together cover ~100% of days), so "stronger in one raw-sign bucket than the
other" is a considerably weaker test of the operator's hypothesis than a deadband-gated
comparison will be in 10b.

**brent_wti specifically tells a genuinely two-sided story depending on which leg's curve
defines the regime — the single most important finding in this phase.** Under the primary
rule (BZ's own curve alone), mean reversion is *nominally stronger in contango*
(β = −0.035, half-life ≈ shorter) *than in backwardation* (β = −0.010) — on its face, the
opposite sign from the operator's prior. But under the secondary "both legs agree"
variant (BZ and CL curves both labelling the same state, sec 4.1's own named robustness
check), the picture flips and sharpens considerably: half-life is **9.5 days in
backwardation** versus **18.3 days in contango**, versus **47.5 days on days the two legs'
curves disagree**, versus **79.3 days pooled/unconditional**. Both legs agreeing on
backwardation is the single fastest-reverting state found anywhere in this phase for
brent_wti, by a wide margin — and the two legs actually agree on 66.2% of trading days,
so this is not a thin-sample artifact. **This is reported as a live, unresolved tension
for 10b to settle empirically, not smoothed into either "confirmed" or "refuted"**: a
single-leg regime definition does not show the operator's effect for brent_wti; a
both-legs-agree definition shows it strongly. 10b's Gate SPR-BW carries this exact
comparison forward as its own declared secondary check.

Calendar spreads received the same machinery as a **labelled circularity diagnostic only**
(`calendar_spread_circularity_diagnostic` in the Phase 3 JSON) — never used as evidence for
the regime hypothesis, per sec 4.2's explicit requirement.

---

## Phase 4 — Inventory positioning (CL only)

This repo's `data/market/cot/` cache holds exactly one CFTC series (067651, light sweet
crude, NYMEX) — a single-product check, never extrapolated into a panel claim, per
docs/09's own documented pitfall on this exact dataset.

**Net non-commercial positioning corroborates the regime label for CL.** Mean net
non-commercial fraction of open interest is 18.8% in backwardation versus 15.9% in
contango (Welch t-test, p ≈ 8×10⁻⁷³ — a huge sample, ~3,600 joined days, so this p-value
reflects sample size as much as effect size), and corr(roll slope, net non-commercial
fraction) = −0.073, the theory-consistent sign (recall: negative roll slope IS
backwardation, so speculators run *more* net-long exactly when the market is
backwardated, as Keynes' normal-backwardation theory and the inventory-theory mechanism
behind the regime hypothesis both predict). The correlation's magnitude is modest — this
is corroborating, directionally-consistent evidence for one product, not a strong or
independently decisive test of the regime hypothesis on its own.

---

## Phase 5 — Pre-registration for 10b

Written and committed here, before any 10b backtest (`phase_5_10a_results.json`),
restating NEXT_PROMPT.md sec 4's gate table verbatim (fire conditions unedited, per sec 1
rule 2) plus:

- **Regime definitions**: deadband (±2%/year annualised slope threshold — a pre-declared,
  round convention, not swept) is **primary** for Gate SPR; raw sign and a 5-day
  persistence requirement are secondary/robustness variants. Primary regime leg = leg1
  (BZ for brent_wti), fixed and identical across every spread; a both-legs-agree variant
  runs only for brent_wti, per sec 4.1.
- **Trading rule**: 60-day rolling z-score signal (reusing the same window as the Phase 2
  IC probe, not re-tuned), position = −clip(z,−2,2)/2, daily rebalance, **two round-turn
  costs per unit weight change** (one per leg — NEXT_PROMPT.md's own explicit warning),
  roll-window rows excluded from both signal and P&L, 4 origin offsets {0,7,14,21}.
- **Sec 4.3 decision**: ADF-failing spreads excluded from 10b's backtest universe (already
  applied in Phase 2, restated here).
- **DSR configuration counts, cumulative and per-gate** (full reasoning and worked
  breakdown in `phase_5_10a_results.json`'s `DSR_CONFIG_COUNTS`):

| gate | n_trials | breakdown |
|---|---|---|
| SP | 8 | 2 taxonomy groups × 4 origin offsets |
| SPR | 12 | 3 regime definitions × 4 origin offsets, inter-commodity only |
| SPR-BW | +1 | brent_wti both-legs-agree variant, run once (diagnostic, not itself a search) |
| VS | 8 | 4 already-logged notebook-8 carry configs (forwarded, not reset to 1) + 4 new vol-scaled configs |
| BM | 20 | 16 already-logged notebook-8 momentum configs (the literal historical n_trials — see note below) + 4 new blend configs |

**One explicit, resolved ambiguity, flagged rather than silently picked:** NEXT_PROMPT.md
sec 4's own prose describes BM's forwarded count as "4 already-logged single-lookback
configs," but the number that actually fed notebook 8's own `deflated_sharpe_prob` call
for Gate AM was **16** (4 lookbacks × 4 origin offsets — verified directly in
`phase_5_results.json`). Per sec 1 rule 3's binding "do not shrink the count" instruction,
the larger, technically accurate historical figure (16, not 4) is used, making Gate BM's
bar harder to clear, not easier. Full reasoning in `phase_5_10a_results.json`.

Also logged for transparency (not counted toward any gate's own n_trials, since neither is
a performance-driven search — see the JSON's own reasoning): 30 spreads mechanically
screened by the ADF criterion in Phase 2, and 11×3 = 33 regime-conditional descriptive
runs in Phase 3, all reported regardless of outcome.

---

## Bugs found

**One bug caught by unit testing before it reached any reported number.**
`spread_lib10.regime_deadband` initially returned an unevaluated Polars expression instead
of a materialised `Series` (a `pl.when/.then` chain with no `.select()` to force
evaluation) — caught by `tests/test_spread_lib10.py` during construction and fixed to
evaluate inside a one-column `DataFrame.select(...)` before Phase 3 or Phase 5 was
finalized. The underlying deadband/contango/backwardation *logic* was correct throughout;
only the return type was wrong, and it was wrong in a way that would have raised a hard
`TypeError` on first real use (as it did when Phase 3 was re-run against the fixed
signature) rather than silently producing a bad number — caught, not smoothed over.

## Bottom line

No strategy verdict belongs here, by design (sec 1 rule 1) — but three findings carry
directly into 10b. First, the cointegration precondition notebook 9 never applied cuts the
inter-commodity backtest universe from 11 to 7 spreads and resolves notebook 9's own
flagged gold_silver/platinum_palladium disagreement outright (neither pair is actually
cointegrated). Second, the raw-sign regime definition is structurally too close to
"regime-blind" to test the operator's own hypothesis, which is why deadband — not raw sign
— is declared primary for Gate SPR here, before any backtest exists. Third, brent_wti's
own regime effect depends materially on which leg's curve defines the regime: weak-to-
reversed under BZ alone, strong (9.5-day half-life in backwardation) under a both-legs-
agree definition — an open, pre-registered question for Gate SPR-BW to settle with a real,
costed backtest rather than a descriptive correlation.
