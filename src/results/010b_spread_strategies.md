# Notebook 10b — Spread Strategies: Results Summary

**The headline: five gates, five nulls, and the two most informative near-misses in this
programme's history.** Gate SP (unconditional spread mean-reversion) shows a real,
positive, cost-surviving Sharpe on both taxonomy groups but falls well short of the
deflated-Sharpe bar once the cumulative configuration count is honestly applied. Gate SPR
(regime-gating) is directionally consistent with the operator's prior — the gated book
beats the unconditional book at every origin offset — but the margin is tiny and neither
bootstrap CI nor DSR clears its bar. Gate SPR-BW delivers this notebook's single most
interesting result: **brent_wti's own regime effect depends on which leg's curve defines
the regime** — absent under the pre-declared primary (BZ-only) definition, present under
the secondary (both-legs-agree) definition — reported as exactly that tension, not forced
into a verdict either direction. Gate VS (vol-scaled carry) delivers the strongest
absolute-performance number in this notebook (net Sharpe 1.16–1.23, DSR 0.9997) and still
does not close Gate AC's excess-vs-basket gap — and fails the new §3 fundable-absolute-
performance flag on drawdown alone, a finding this notebook would have missed had it
stopped at Sharpe and DSR. Gate BM (blended momentum) is an unambiguous, small-margin
null. **No gate fires. No gate clears the §3 fundable flag either — the closest, Gate VS,
fails specifically and only on the drawdown bound.** A null reported this rigorously,
across five genuinely different constructions, is this programme's expected outcome, not
a disappointment (docs/08's own "a ninth and tenth null is the expected outcome" standard,
now extended to an eleventh through fifteenth).

Machinery: `src/research/tmp/run_phase_{0..4}_10b_*.py`, reusing
`commod_lib8.portfolio_costs_futures`/`round_turn_cost_per_contract` (futures cost model,
one round turn per leg — every spread trade here pays for *every* leg it holds, not one),
`research.block_bootstrap_ci`/`deflated_sharpe_prob` unmodified, and
`src/research/tmp/run_phase_5_alpha.py`'s own carry/momentum panel for Gates VS/BM
unmodified except the position-sizing/aggregation rule. Development window only; **the
2025-01-01 → 2026-07-28 holdout is untouched** — no gate here is a certified winner, so
there is no legitimate reason to spend it.

---

## Gate verdicts — the full table

| gate | claim | fires? | §3 fundable flag | number behind it |
|---|---|---|---|---|
| **SP** (inter-commodity) | unconditional mean-reversion survives cost | **NO** | **NO** | net Sharpe 0.42–0.42 across offsets, DSR 0.562 (n_trials=8), bootstrap CI on net return [−2.2e-5, +2.0e-4] does NOT exclude zero |
| **SP** (calendar) | unconditional mean-reversion survives cost | **NO** | **NO** | net Sharpe 0.50–0.51, DSR 0.680, CI [+1.9e-7, +6.8e-5] DOES exclude zero, but DSR alone kills it |
| **SPR** (deadband, primary) | regime-gating improves it | **NO** | **NO** | gated Sharpe exceeds unconditional at every offset (0.426 vs 0.423 at offset 0) but by a margin too small to matter: DSR 0.484 (n_trials=12), gated-vs-zero CI does not exclude zero, gated-minus-unconditional CI [−2.3e-5, +2.0e-5] does not exclude zero |
| **SPR-BW** | not a brent_wti artifact | **NO** | inherits SPR's NO | brent_wti itself does NOT show gated > unconditional under the primary (BZ-leg) definition (0.604 vs 0.614); 3 of 6 other eligible inter-commodity spreads do (crack_321, gasheat_rbho, gasoline_crack) — the "≥3 others" bar is cleared, but the binding brent_wti-must-show-it clause is not |
| **VS** | vol-scaled carry closes Gate AC's gap | **NO** | **NO — fails on drawdown alone** | net Sharpe 1.16–1.23 at every offset (up sharply from Gate AC's 0.90–0.95), DSR 0.9997 (n_trials=8) — both individually clear the fundable-flag bar — but the excess-vs-basket CI [−0.0022, +0.0098] still includes zero (fails the tradeable-alpha gate, same shape as Gate AC) AND cumulative log-drawdown corresponds to ≈99.6% of peak equity, nowhere near the 25%-of-peak bound (fails the fundable flag on its own, independent criterion) |
| **BM** | blended momentum is sign-consistent and survives cost | **NO** | **NO** | net Sharpe **negative** at every offset (−0.015 to −0.033) — sign-consistent, but consistently negative, not positive; DSR 0.025 (n_trials=20) |
| **FA-data** | this repo caches a crypto spot series distinct from perpetuals | **resolved FALSE** | n/a | every Binance URL `src/data.py` calls is a USDS-M perpetual-futures endpoint; every cached klines/ohlc symbol has a matching funding file (funding only exists for perpetuals) — no proxy built, Gate FA deferred with this note |

---

## The regime hypothesis — plain-English verdict, brent_wti and cross-spread stated separately

**Cross-spread: regime-gating does not survive as tradeable alpha, but the direction is
genuinely, consistently supportive, not noise.** Across all four origin offsets, the
deadband-gated inter-commodity book's net Sharpe exceeds the unconditional book's — a
small but perfectly consistent margin (0.426–0.427 vs. 0.423–0.424). That consistency is
worth something: it is not what four independent coin flips would produce. But the
absolute margin is too small for either bootstrap CI (gated-vs-zero, or gated-minus-
unconditional) to clear zero, and the DSR at the honestly-counted 12-configuration bar
(three regime definitions × four offsets) comes in at 0.484 — essentially "no better than
random search would produce this often." **Verdict: a real but too-small-to-trade
directional signal, not a tradeable improvement.**

**brent_wti-specific: genuinely unresolved, and reported as exactly that — not smoothed
into either "confirmed" or "refuted."** Under the pre-declared PRIMARY regime definition
(BZ's own curve alone, decided in 10a before any 10b backtest existed), brent_wti's own
gated Sharpe (0.604) is *lower* than its unconditional Sharpe (0.614) — the operator's
prior does not show up for brent_wti under the definition this notebook committed to in
advance. Under the SECONDARY "both legs must agree" definition (also pre-declared, as
sec 4.1's own named robustness check, run once, n_trials=1), the picture flips: gated
Sharpe 0.694 clears unconditional Sharpe 0.614 by a real margin, echoing 10a Phase 3's own
descriptive finding that brent_wti's half-life drops from 79 days (pooled) to 9.5 days
specifically when both BZ's and CL's curves agree on backwardation. **The honest reading:
brent_wti's regime effect, if it exists, needs *both* legs to confirm the state — a single
leg's curve is not enough — and Gate SPR-BW's own binding criterion (evaluated on the
pre-declared primary definition, as it must be to mean anything) correctly does not fire
on this basis.** A future notebook with its own fresh pre-registration is the legitimate
way to test the both-legs-agree definition as primary, not a retroactive edit here.

---

## Phase 0 — Reproduction check

Fifteen assertions against 10a's own committed JSON (spread counts, taxonomy split,
ADF-exclusion of gold_silver/platinum_palladium, the deadband-primary declaration, all
five gates' exact DSR n_trials, and the FA-data resolution) — all passed before this
notebook's own backtests ran (`run_phase_0_10b_repro.py`).

---

## Phase 1 — Gate SP

Trading rule (declared in 10a's pre-registration, not re-derived here): 60-day rolling
z-score of the spread's own value, position = −clip(z,−2,2)/2, one round-turn cost per
leg (summed across legs — crack_321 and crush_soy pay three, not one), roll-window rows
excluded from both signal and P&L, equal-weighted book across each taxonomy group's
eligible (ADF-cointegrated) spreads: 7 inter-commodity, 16 calendar.

Both books show a real, positive, cost-surviving net Sharpe at every origin offset
(inter-commodity 0.42–0.42, calendar 0.50–0.51) — genuinely better than a coin flip, and
directionally exactly what notebook 9's cheap first-look probe predicted. **Neither clears
DSR at the honestly-counted n_trials=8** (2 taxonomy groups × 4 offsets): 0.562 for
inter-commodity, 0.680 for calendar. The calendar book's bootstrap CI on net return does
exclude zero on its own, but DSR alone is enough to keep Gate SP from firing on either
group — a clean illustration of why the deflated-Sharpe bar exists: a Sharpe this size,
found after screening this many configurations, is not yet distinguishable from what an
unlucky multiple-testing search would produce by chance.

---

## Phase 2 — Gates SPR and SPR-BW

Same trading rule, same universe, with position zeroed on any day the term-structure
regime is not a "definite" state. Primary definition = deadband (10a Phase 5's own
correction from raw sign — see `src/results/010a_term_structure_regimes_and_spreads.md`).
Full detail — including the two secondary definitions (raw sign, persistence), both run
and both counted in the n_trials=12 total — in `phase_2_10b_results.json`.

Both secondary definitions independently corroborate the same overall picture: raw sign
shows the gated book *not* exceeding the unconditional book at every offset at all (raw
sign restricts almost no days, exactly the structural weakness Phase 5 flagged in advance
as the reason it was demoted from primary); persistence shows gated exceeding
unconditional at every offset, similar in spirit to deadband but with no DSR computed for
it (only the primary definition's DSR feeds the fire condition, per the pre-registration).

---

## Phase 3 — Gates VS and BM

**Gate VS is this notebook's most consequential result precisely because it does NOT
simply fire or not-fire — it fires on two of three fundable-flag criteria and fails hard
on the third.** Inverse-20-day-realized-vol position sizing (replacing notebook 8's
equal-weight-within-leg carry book, the only change from Gate AC) lifts net Sharpe from
0.90–0.95 to **1.16–1.23** and deflated Sharpe probability from 0.997 to **0.9997** — both
comfortably inside the §3 fundable-flag's own Sharpe>0.5 and DSR>0.95 bars. But the
excess-vs-basket bootstrap CI is essentially unchanged in shape ([−0.0022, +0.0098], still
including zero) — vol-scaling reshapes risk *within* the carry book, it does not change
whether carry beats a passive commodity basket, which is a question about the factor's
own exposure to broad commodity beta, not about how that exposure is risk-weighted
internally. And separately, the strategy's own cumulative log-drawdown corresponds to
roughly 99.6% of peak (log-return-equity-curve terms — see the note below), nowhere near
the 25%-of-peak bound sec 3 declares in advance — **failing the fundable flag on a
completely independent criterion from the one that fails the tradeable-alpha gate.**
Reported exactly per sec 3's own required framing: *fundable-looking on absolute Sharpe
and DSR, not shown to beat passive exposure to the same asset class, and not
institutionally fundable either once drawdown is actually checked* — never rounded up to
"essentially a pass."

*Drawdown note:* this repo's own `futures_portfolio_metrics` reports drawdown as a
cumulative-log-return quantity (matching every other notebook's own convention, e.g.
Gate AC's own −7.52), not a literal percentage. Converted to a percentage of peak equity
(`1 − exp(log_drawdown)`) for the first time in this programme specifically because sec 3's
fundable flag requires a literal bound, Gate VS's −5.41 log-drawdown is ≈99.6% of peak —
an artifact of an 18-year, high-turnover, unconstrained-compounding backtest convention
that was never previously asked to answer a literal percentage-drawdown question. This is
reported honestly rather than converted to a friendlier number, and is itself a finding
about how future notebooks should construct a backtest if the §3 flag is meant to be
checked routinely: a capital-bounded (not continuously-reinvested) equity curve would be
needed to make this comparison fair to the strategy.

**Gate BM is an unambiguous null.** The equal-weighted blend of notebook 8's four momentum
lookbacks is net-**negative** at every origin offset (−0.015 to −0.033) — sign-consistent,
but consistently the wrong sign, diluting 1-month and 12-month's weak positive Sharpes
with 3-month and 6-month's larger negative ones exactly as notebook 8's own per-lookback
breakdown predicted it would. DSR 0.025 at the honestly-counted n_trials=20 (16 already-
logged notebook-8 configurations forwarded + 4 new blend configurations) confirms there is
no signal being missed here, not a near-miss.

---

## Phase 4 — FA-data check

Resolved **FALSE**, without a proxy. `src/data.py`'s only Binance download paths are
`data.binance.vision/data/futures/um/...` (bulk trades/klines) and
`fapi.binance.com/fapi/v1/fundingRate` — both USDS-M perpetual-futures endpoints. No spot
host (`data.binance.vision/data/spot/...` or `api.binance.com`) appears anywhere in this
repo's downloader, and every cached klines/ohlc symbol has a matching funding file — a
genuine spot series would have none. Gate FA is deferred with this data-acquisition note;
building a proxy (e.g. treating the perpetual's own mark price as a spot stand-in) would
manufacture a cash-and-carry spread mechanically guaranteed to look small, not measure the
real opportunity, exactly what NEXT_PROMPT.md sec 2 warns against.

---

## Bugs found

None new in this notebook's own machinery (10a's single caught bug, `regime_deadband`'s
unevaluated-expression issue, was fixed before 10b began and is reported in 10a's own
results MD). One discipline note worth recording in the same spirit: Gate VS's raw
log-drawdown number, taken at face value without converting to a percentage, would have
silently passed an eyeball "does this look risky" check (−5.41 reads like a moderate
number next to Gate AC's own −7.52) — only converting it explicitly (`1 − exp(x)`) for the
first time this programme has needed a literal percentage bound revealed the true ≈99.6%
figure. Caught by taking sec 3's own bound literally rather than reading the existing
metric's sign and magnitude as "probably fine because it's smaller than last time."

## Bottom line

Five gates, five honest nulls, delivered with more texture than a flat "nothing works."
Gate SP shows the mean-reversion mechanism genuinely survives cost at a positive Sharpe on
both taxonomy groups — it simply hasn't cleared the bar this programme's own multiple-
testing discipline requires, and the honest configuration count (not a shrunk one) is why.
Gate SPR shows the operator's regime prior pointing the right direction at every offset,
too faintly to trade. Gate SPR-BW's brent_wti-specific result is this notebook's genuinely
open question: the effect is leg-definition-sensitive in a way that is itself informative
(both legs need to agree), not a simple "artifact" the way notebook 7's Gate TF single-
symbol result was. Gate VS is the most important negative result in this notebook because
it dissociates three previously-conflated ideas — absolute Sharpe, deflated Sharpe, and a
literal drawdown bound — and shows a strategy can clear the first two decisively while
failing the third just as decisively, exactly the discrimination sec 3's two-flag
reporting standard was built to make possible. Gate BM closes cleanly, no ambiguity. No
gate reaches the holdout; nothing here is a certified winner.
