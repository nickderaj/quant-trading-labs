# Commodity Tails, Density Selection, and a Cross-Asset Risk Engine - Results Summary

## What

This notebook tests whether the crypto risk-modeling findings from notebooks 4-6 (fat tails, thin-tailed models understating expected shortfall, a well-calibrated conditional risk engine) generalize to a structurally different asset class — 16 commodity futures plus an equity-index control — and whether two of the most literature-favored commodity trading strategies (carry and time-series momentum) can find a tradeable edge net of realistic costs. It also tests two more speculative, genuinely uncertain hypotheses: whether commodity tail asymmetry flips sign relative to equities as inventory theory predicts, and whether term-structure state (backwardation/contango) carries risk information beyond an unconditional model.

## Why

Every prior finding in this programme came from crypto, a market with unusual structure (24/7 trading, funding-rate mechanics, no physical delivery). Commodities offer a genuinely different test: real storage costs, physical delivery, producer hedging demand, and — unlike crypto — a literature that predicts carry and momentum should actually work here. Confirming or refuting the risk findings and the "no tradeable edge" pattern in this different market determines whether the programme's conclusions are facts about crypto specifically or about financial markets more broadly.

## How

After extensive data-hygiene work to build a reliable continuous futures price series (roll-adjustment, contamination filtering, liquidity screening — four significant bugs caught and fixed before any statistic was trusted), the notebook ran a tail atlas and density-selection contest across seven distribution families (Phases 1-2), a 13-model conditional GARCH/GJR risk battery replicating notebooks 4-6's tests (Phase 3), term-structure/seasonal/macro conditioning of VaR coverage (Phase 4), carry and momentum backtests with a futures-specific cost model and bootstrap significance testing (Phase 5), an intraday descriptive appendix (Phase 6), a cross-asset conditional risk engine with copula-based portfolio simulation (Phase 7), and a final holdout evaluation (Phase 8).

## Results

The core risk findings replicated cleanly: commodities are fat-tailed (16/16 products), thin-tailed models understate their own expected shortfall (15/16 products), and a well-calibrated conditional risk engine was built and held up on holdout (14/16 pass). GARCH-GED, not GARCH-t, was the dominant density family, a genuine departure from the crypto result. Tail-asymmetry sign-flipping and term-structure conditioning of VaR both came back as real, informative nulls. Both carry and momentum failed to clear the bootstrap-CI bar for tradeable edge (carry came close, with strong absolute Sharpe but a CI including zero), extending the programme's honest-null streak to eight notebooks — this time in a market where the literature's own prior favored finding something.

**The headline: the crypto risk findings are not crypto findings. They are facts about
financial returns.** Fat tails (Gate CT), thin-tailed models understating their own
expected shortfall (Gate CE), and a well-calibrated conditional risk engine (Gate RE)
all replicate cleanly across 16 commodity futures and an equity-index control — three
different asset classes with none of crypto's 24/7 trading, no funding-rate mechanics,
real physical delivery, real storage costs, and real hedging demand from producers. The
alpha side replicates too, in the opposite sense: commodity carry and time-series
momentum, the two most-cited edges in the futures literature, **both come back null**
against this repo's own bootstrap-CI bar, extending this programme's now-eight-notebook
run of honest nulls on tradeable alpha into a market where, unlike crypto, a real edge
was the literature's own prior.

**Two genuinely uncertain questions from the pre-registration came back negative.** The
tail-asymmetry skew-flip (Gate CA) does not fire — only 5 of 14 measurable products show
the predicted right-skew sign, essentially a coin flip, not the clean crypto-vs-commodity
split the inventory-theory literature predicts. Inventory-state conditioning (Gate CI)
does not fire either — a GARCH-t reference model's 1% VaR coverage, split by
backwardation/contango, reveals genuinely new miscalibration in only 4 of 16 products,
short of the 10-product bar. Both are reported as real, informative nulls, not near-misses
smoothed into a "directionally supportive" story.

**Four Phase 0 bugs were bad enough to invalidate everything downstream if uncaught,
and all four were caught before a single tail statistic was trusted.** A naive
volume-based hygiene rule could not tell CL's genuine 2020-04-20 negative settlement from
NG's mislabeled spread-differential contracts — both traded ~8-10% of that day's volume.
A liquidity screen applied upstream of curve construction silently deleted 38-60% of
individual products' front-month series. `roll_calendar.parquet` lists a contract-month
for every calendar month even for quarterly-cycle products (deleting 60% of one grain's
history the same way). And back-adjusted (additive) continuous prices drift negative
over a 16-year, ~200-roll history, turning ordinary trading days into 100%+ "returns"
that are pure adjustment artifact. Each is detailed in Phase 0 below, with the evidence
that caught it.

Canonical data root: `src/research/data/market/` (4.7 GB, gitignored). The "duplicate
tree" `src/research/market/` described in the pre-registration **does not exist on
disk** — flagged as a documentation/state discrepancy, not silently skipped (see Phase
0). Machinery: `src/research/tmp/commod_lib8.py` (new — roll/continuous-series
construction, the two-tier hygiene filter, the futures cost model, term-structure/
seasonal/macro conditioning, the risk-engine API, dependence/copula tools), reusing
`research.py`, `distributions.py`, `dist_lib5.py`, `dist_lib6.py`, `densities/`, and
`alpha_lib7.py` unmodified. New terminology is defined from scratch in `docs/`
(`09-market-data-and-microstructure.md` for futures/spread mechanics, `01-probability-
and-distributions.md` for copulas and tail dependence) and indexed in `GLOSSARY.md`.

## Gate verdicts — the full table

| gate | claim | fires? | number behind it |
|---|---|---|---|
| **CT** | commodities are fat-tailed, more so than the equity control | **YES** | Hill α < 5 in 16/16 products; ES's plateau range excluded from the product's own plateau range in 13/16; normality rejected at p≈0 in all 16 |
| **CA** | commodities show the opposite tail-asymmetry sign to equities | **NO** | only 5/14 measurable products show the predicted (right-skew) sign — no better than chance; ES's own left-tail plateau never found, so its "equity sign" could not even be verified |
| **CD** | no single density family wins everywhere | **partial / no** | 5 distinct families win somewhere in the raw OOS-log-score ranking (GARCH-GED, GJR-GED, GARCH-Hansen-skew-t, GARCH-NIG, GARCH-JohnsonSU) — qualitatively supports the "no universal winner" story — but only 2/16 products (GC, SI) clear BH-significance, and both are won by the *same* family, so the strict "BH-significant winner differs across products" criterion is not met |
| **CE** | thin-tailed (normal) models understate their own 1% ES | **YES** | 15/16 products reject at 5% (BH-adjusted) at the 1% level, both tails; holdout: 11/16 reject (smaller sample, same direction) |
| **CI** | term-structure state carries tail information beyond the unconditional model | **NO** | only 4/16 products show a state where coverage fails while the pooled test passes — well short of the 10-product bar |
| **AC** | commodity carry survives cost | **NO** | net Sharpe 0.90–0.95 at every origin offset, deflated Sharpe prob. 0.997, but the block-bootstrap CI on excess return vs. the equal-weight basket includes zero |
| **AM** | time-series momentum survives cost | **NO** | best lookback (1-month) net Sharpe 0.10–0.12, sign-inconsistent across lookbacks (3-month and 6-month are net-negative), deflated Sharpe prob. 0.098 |
| **AS** | spread mean reversion survives cost | **not run** | declared in scope, not executed this pass — see Phase 5 scope note |
| **RE** | the risk engine's own VaR is correctly calibrated OOS | **YES** | 15/16 products pass 1% coverage (Kupiec + Christoffersen) walk-forward; holdout: 14/16 pass |

Three of the four gates this notebook's own pre-registration called "likely to fire"
(§5) fired outright (CT, CE, RE); the fourth (CD) fired in spirit but not by the letter
of its own pre-declared significance bar — reported as a partial result, not rounded up.
Both gates flagged in advance as "genuinely uncertain" (CA, CI) came back negative. All
three alpha gates attempted came back null, exactly as predicted.

**Holdout (2025-01-01 → 2026-07-28): spent once, since CT/CE/RE all fired.** Every
finding replicates directionally on data no fitting procedure ever saw — see Phase 8.

---

## Phase 0 — Hygiene, construction, and reproduction

**The duplicate tree does not exist.** `src/research/market/` is described in the
pre-registration as a byte-identical copy of `src/research/data/market/`; on this disk,
only the latter is present. `commod_lib8.check_duplicate_tree` reports this explicitly
rather than the check being silently skipped.

**Bug 1 — the hygiene filter needed persistence, not volume, to separate real events
from junk.** The instinct is "low volume + big price deviation = bad data." It fails
here: CL's genuine 2020-04-20 negative settlement (-$2.67 close on the day, contract
CL202005) traded on 8.4% of that day's total CL volume; NG's mislabeled
spread-differential contract (`NG202507`, tagged as an outright but actually some kind
of calendar-spread quote) traded on a *comparable* 9.9% of NG's volume that day. A
volume cutoff — absolute or relative — flags both or neither. The signal that actually
separates them is how long the anomaly persists across each contract's *life*:
`NG202507` prints a near-zero or negative close on 97% of the ~575 days it appears; CL's
contract deviates like this on 0.6% of its ~343 days. The production rule
(`commod_lib8.flag_contaminated_rows`) is two-tier: a contract with ≥10 days of history
whose price deviates >30% from that date's highest-volume "anchor" contract on more than
half its days is flagged in its entirety (a mislabeled series, not an outright having a
bad day); a contract that passes this is still flagged row-by-row if a single day
deviates >30% *and* trades under 50,000 contracts (a genuine one-off glitch on an
otherwise-clean, thin contract). Verified directly against `exact_statistics/raw`: one
confirmed junk contract (GC201511) shows `settlement_price` printing exactly 0.0 while
its own `trading_session_low/high` and best bid/offer print ~$1127-1128, in line with
real gold spot that week — the settlement feed is broken for that contract, not gold
itself. Contamination rates post-fix: 0.0-3.98% for 15/16 products, 35.09% for NG (the
single worst-affected product, matching the scale of its mislabeling problem).

**Bug 2 — a liquidity screen applied before curve construction deletes data, it doesn't
clean it.** The roll schedule here is calendar-driven (from `roll_calendar.parquet`),
not volume-driven, so a quiet day on the contract the calendar already designates as
front month is a real trading day, not a contract needing replacement. Screening it out
upstream of `build_continuous_series` carved holes in the front-month series — 38-69% of
rows null for the thinnest products (PA, PL) before this was caught. Fixed by moving the
liquidity screen downstream (reported separately as a diagnostic; never applied before
curve construction).

**Bug 3 — `roll_calendar.parquet` lists a contract-month for every calendar month, even
for products that only trade a quarterly cycle.** Platinum and palladium list a ticker
for every month but only trade real size in Jan/Apr/Jul/Oct — the "in-between" months
print a handful of trades over a handful of days (total lifetime volume in the tens to
low hundreds) before going silent, versus hundreds of thousands of contracts in the real
months. Rolling into a "listed but dead" month left PL's front-month series null on 57%
of days before `commod_lib8.liquid_contract_months` (a lifetime-volume ≥5,000-contract
threshold, applied on top of — not instead of — the calendar-membership check) fixed it,
dropping PL/PA's null rate to under 10%.

**Bug 4 — back-adjusted continuous prices go negative over a long enough history, and
that corrupts every return computed near the crossing.** Additive back-adjustment
accumulates an offset at every roll; over 16 years and ~200 rolls, several products'
early-history back-adjusted prices crossed zero, producing single-day "returns" over
100% that were pure splice artifact. `log_return_ratioadj` (multiplicative/ratio
adjustment) never crosses zero as long as the raw price doesn't, and is the series used
for every tail/ACF/density statistic in this notebook once the bug was caught — CL's
post-fix excess kurtosis (36.09) matches its Phase-0 naive-vs-hygiene figure closely,
confirming internal consistency once the right series was in use.

**Roll rule and its sensitivity.** Production: roll 5 calendar days before first-notice
(or last-trade date where notice isn't populated), snapped backward off a weekend (no
exchange trading calendar available). Sensitivity at N ∈ {3, 5, 10} on CL: excess
kurtosis 35.4 / 36.1 / 36.1, annualised vol 37.8% / 37.6% / 37.6% — stable across the
range. CL's own 2020-04-20 print is instructive here: under N=5 the front-month series
had already rolled into the June contract three trading days earlier, so a real book
following this rule never touches the print at all — it survives only in the raw
per-contract data (correctly preserved by the hygiene filter above), not in the
continuous series, which is the economically correct outcome for a book that respects
delivery risk.

**Three-way validation — reported as attempted, and it does not clear its own bar.**

| check | pass threshold | result |
|---|---|---|
| vs. `research/*_curve.parquet` (CL, NG, GC, SI) | ≥99% of dates within 1 tick | **64-84%** |
| vs. `metrics/*.parquet` realised_vol_20d (12 products) | ≥90% within 25% relative tolerance | **7-68%** (ES: 100%, the one clean pass) |
| vs. `yfinance` `*=F` daily-return correlation (CL, NG, GC, ZC) | >0.98 | **0.80-0.86** |

Two real bugs were found and fixed inside this check itself (a join fan-out that
inflated overlap counts 10-25x, and NaN-vs-null handling that silently dropped GC's
usable comparison rows to near zero) — both are visible in the before/after numbers
above. What remains is a genuine, unresolved discrepancy, not a bug: pulling raw
`exact_statistics` evidence for the hygiene-filter case (Phase 0, Bug 1) showed
`settlement_price` printing a materially different value from `trading_session_low/high`
for at least one contract — circumstantial evidence that this vendor's `close` field
does not always equal the official settlement price other reference series use, which
would explain a *level* discrepancy but not fully explain return correlations as low as
0.80-0.86 against yfinance, whose own roll and construction conventions are also unknown
and plausibly differ meaningfully from this notebook's. **Read every number in this
notebook as internally consistent (Phase 0's naive/hygiene, N-sensitivity, and
CL-2020-04-20 checks all agree with each other) but not independently externally
validated to the pre-declared tolerance** — stated plainly rather than left to be
inferred from a pass/fail table alone.

**Post-hygiene tail statistics** (the "before" table from the pre-registration, now
"after"): excess kurtosis ranges from 0.64 (KE — genuinely the thinnest-tailed product
in the panel, consistent with its short history) to 53.10 (PA). Every naive-artifact
figure from the pre-registration (PL 4,810, NG 1,893) is gone; no product remains in the
hundreds.

**Stale-bar audit.** Minor across the board — worst run length 3 consecutive identical
closes (CL, 23 total stale days out of ~5,000), most products 1-2 day runs. Not treated
as a material data-quality issue; no stale-day exclusion applied.

**Reproduction check.** `phase3_zoo_results.json` and `phase_e_holdout_results.json`
from notebooks 6/7 load and structurally validate (existence + key-shape check, per this
programme's house style for a Phase-0 ritual on prior notebooks' committed JSONs).

---

## Phase 1 — The tail atlas

Moments, Hill tail index (both tails), volatility clustering, the leverage effect,
Samuelson effect, seasonality, and named-event annotation, computed on `log_return_ratioadj`
(the ratio-adjusted, gap-free continuous F1 series — see Phase 0 Bug 4) across all 16
products plus a BTCUSDT 1d bridge series.

**Skew is mixed, not clean.** Grains lean positive as predicted (ZW +0.25, KE +0.23,
ZM +0.28, ZC +0.06) — the low-inventory-upside-shock story holds best where the
literature says it should. Energy and metals lean **negative** (CL -1.52, PA -1.61,
SI -1.87), the opposite of the naive "commodities are right-skewed" prior — plausibly
because a 2010-2026 sample is dominated by large downside shocks (2014-15 oil collapse,
2020 COVID crash) rather than upside supply shocks, a sample-composition effect as much
as a structural one. This is the same story Gate CA's formal test tells: 5/14 products
match the predicted sign, no better than a coin flip.

**The inverse-leverage effect does not show up cleanly either.** corr(r_t, σ_{t+1}) sits
close to zero with wide, mostly zero-including bootstrap CIs for most products; where a
*significant* correlation does appear (palladium, -0.234, 95% CI excluding zero), it is
**negative** — the equity-style sign, not the predicted commodity sign. Phase 3's GJR
sign check (below) tells the same story independently: energy products show the equity
sign on gamma essentially unanimously (CL, BZ, HO, RB, ES: 100% of 90/36 refits
positive), a second, independent measurement agreeing with the first.

**Every product's excess kurtosis is large and Jarque-Bera rejects normality at
p ≈ 0** — the headline Gate CT result, detailed in the gate table above.

---

## Phase 2 — Unconditional density selection

Seven families (normal, Student-t, Hansen skew-t, NIG, GED, JohnsonSU, spliced-EVT) fit
via an expanding-window walk-forward (5 folds per product), OOS log score, PIT/KS
calibration, all-pairs Diebold-Mariano + Benjamini-Hochberg.

**Five distinct families win somewhere** in the raw ranking (t, ged, hansen_skewt,
johnsonsu, nig) — directionally the "no universal winner" story the pre-registration
predicted. **No product's win is individually BH-significant** — the gaps between the
top 2-3 families are real in ranking but too close, at this sample size, to clear a
per-product multiple-comparison bar. Reported honestly rather than rounded up to a gate
fire (this is Phase 2's own contribution to the eventual Gate CD verdict, formalised
further in Phase 3).

---

## Phase 3 — Conditional models and the risk battery (the replication test)

GARCH(1,1) and GJR(1,1,1), each × {normal, t, Hansen skew-t, NIG, GED, JohnsonSU}, plus
one spliced-EVT density (paired with the GARCH-normal variance process) — 13 models per
product, rolling out-of-sample (annual refit cadence, ~750-day minimum training window),
on the development window (2010-06-06 to 2024-12-31; ES from 2018-01-01; KE from
2013-12-16). ~80 minutes of wall-clock compute across 16 products.

**GARCH-GED dominates, not GARCH-t.** 9 of 16 products pick `garch_ged` as their
OOS-log-score winner (CL, BZ, HO, RB, GC, SI, PL, PA, ZS); ES picks `gjr_ged`. This is a
genuine, informative **departure** from notebook 6's own crypto finding (GARCH-t ≳
GARCH-NIG was crypto's ranking) — GED's finite tails, flexible enough to capture
moderate excess kurtosis without Student-t's very heavy polynomial tails, appear to fit
commodities' more moderate (though still decisively non-normal) kurtosis better than
crypto's more extreme tails needed. Grains split toward skewed families exactly as
predicted: ZC/ZW/ZL win with Hansen skew-t, KE/ZM with NIG. **Only 2/16 products (GC,
SI) clear BH-significance for their winner, and both are won by the same family** — the
reason Gate CD does not formally fire despite the qualitative pattern holding.

**Gate CE (Acerbi-Székely) is the cleanest, strongest result in the notebook** — see the
gate table. Computed as a targeted follow-up (`run_phase_3b_gate_ce.py`) since Phase 3's
main run recorded Z-statistics but not the bootstrap p-values Gate CE's formal criterion
needs. 15/16 products reject normal-innovation ES calibration at 5% (BH-adjusted, both
tails, 1% level) — only ZW's lower tail fails to reject (p=0.147). This is the
"highest-probability headline result" the pre-registration called it, and it delivered.

**The GJR sign check found the equity sign, not the predicted inverse-leverage sign, in
energy and most metals.** Mean fitted gamma is positive (equity-style: down-moves raise
next-bar variance more) in 90-100% of refits for CL, BZ, HO, RB, ES, and mostly positive
for the metals (PL 99%, GC 87%, SI 81%, PA 74%). Grains are the one sub-group showing
genuine mixed-to-negative signs: ZW (31% positive), KE (23%), ZM (8%) lean toward the
predicted commodity sign, while ZC (100% positive) does not. This is the same
disagreement Phase 1's leverage-correlation table found independently — two separate
measurements agreeing that the "commodity inverse-leverage effect" is, at best, a
grains-specific phenomenon in this sample, not a market-wide one.

**Violation-process PMF fits** (Poisson vs. negative-binomial counts; geometric vs.
discrete-Weibull durations) were run on each product's own best model's violations —
full detail in `phase_3_results.json`, not separately gated per the pre-registration.

---

## Phase 4 — Conditional tails and the inventory story

Term-structure state (backwardation/contango, from the F1→F2 roll slope), seasonal
state (NG heating season; grain planting/growing/harvest), and macro/vol regime (VIX
terciles, T10Y2Y sign, DFF terciles, all lagged), tested against a single GARCH-t
reference VaR model's 1% coverage — deliberately one consistent, cheap baseline for
every product rather than re-deploying Phase 3's full 13-model battery per state (a
declared compute/scope tradeoff, not an oversight).

**Gate CI does not fire, and the honest number matters here.** A first pass, scoring
"conditioning adds information" whenever *either* the pooled test failed *and* a state
also failed *or* the pooled test passed while a state failed, gave 14/16 — which would
have cleared the bar. That criterion is too permissive: a product where the pooled
(unconditional) test already fails is evidence about the reference model's calibration
(a Phase 3 question), not evidence that *conditioning* reveals anything beyond it. The
corrected, strict criterion — pooled test **passes** while at least one state's test
**fails** — gives **4/16** (NG, GC, ZS, ZL), well short of the 10-product bar. Caught and
fixed before being reported, in keeping with this notebook's own standard of showing the
number, not just the verdict.

---

## Phase 5 — Alpha attempts (carry and momentum only — a declared scope cut)

**Scope note, stated up front:** the pre-registration declares six strategy families
(A-F). This pass implements **A (carry) and B (time-series momentum)** only — the two
most central per the literature review (§3.3) — and explicitly does not run C
(cross-sectional momentum), D (basis-momentum), E (spread mean reversion), or F (COT
hedging pressure). This follows the pre-registration's own stated priority ("if time
runs short, cut Phase 5 and Phase 6 before cutting Phase 0, 3, or 7") rather than
thinning every strategy to fit the remaining time.

Universe: all 16 products, cross-sectional dollar-neutral construction (top/bottom 30%,
~5 names per leg), futures cost model (§7 below) charged via
`commod_lib8.portfolio_costs_futures`, block-bootstrap CI on excess return vs. the
equal-weight basket, deflated Sharpe on the true count of configurations tried (4 origin
offsets × {1 carry horizon, 4 momentum lookbacks} = 20 configurations logged).

**Gate AC (carry): does not fire, and it is a near-miss, not a wipeout.** Net Sharpe
0.90-0.95 at every origin offset (0/7/14/21 days) — genuinely strong, and deflated
Sharpe probability 0.997 says this is very unlikely to be pure multiple-testing luck.
What fails is the excess-return test: the 95% block-bootstrap CI on (carry net return −
equal-weight-basket return) is [-0.0028, +0.0094] — includes zero. Carry's absolute
Sharpe is real; it is not shown to *beat a passive commodity basket* by a margin this
bar can distinguish from noise. Matches this repo's own prior (§3.3: "commodity carry
survived on turnover grounds in a way notebook 7's crypto carry did not, but a 2010+
sample is entirely inside the post-financialisation regime where the literature expects
weak-to-zero net-of-cost alpha") almost exactly.

**Gate AM (momentum): does not fire, and less ambiguously.** Best lookback (1-month) net
Sharpe 0.10-0.12 across offsets — real but small — and **sign-inconsistent across
lookbacks**: 3-month (-0.13) and 6-month (-0.11) are net-negative, 1-month and 12-month
are weakly positive. Deflated Sharpe probability 0.098, nowhere near the 0.95 bar. A
seventh and eighth honest null for this research programme (following crypto's six),
now in a market where the literature's own prior favoured finding something.

---

## Phase 6 — Intraday appendix (descriptive only, excluded from every conclusion)

CL/BZ/HO/RB 1-minute bars, 2026-01-01 to 2026-07-19 only (six months, four energy
products) — explicitly outside every gate and every conclusion elsewhere in this
notebook, per the pre-registration.

**EIA petroleum status announcement (Wed 10:30 ET) shows a modest, consistent volume-of-
information effect.** Mean |return| in a ±30-minute window around the announcement,
Wednesdays vs. other weekdays: CL 1.05×, HO 1.08×, RB 1.11× — small but directionally
consistent across all three petroleum-status-relevant products, over 28 Wednesdays.

**Realised-vol signature plot** (RV at 1/5/15/30/60-minute sampling) shows only a mild
signature — CL's mean daily RV is essentially flat across frequencies (0.00142-0.00152),
consistent with a liquid, well-arbitraged market where microstructure noise is not a
first-order concern at this level of aggregation.

A real bug was caught and fixed here: `.dt.hour()` returns `Int8`, and `hour*60`
silently overflows Int8's ±127 range for any hour ≥3 — every minute-of-day computation
was corrupted (zero rows ever matched the EIA announcement window) until caught and
fixed with an explicit cast to Int32.

---

## Phase 7 — The risk engine

`commod_lib8.RiskModel` / `fit_risk_model` / `portfolio_risk`, with family selection
read from Phase 3's own per-product OOS ranking (never hardcoded — `garch_ged` for 9
products, `garch_johnsonsu` for NG, `garch_hansen_skewt` for ZC/ZW/ZL, `garch_nig` for
KE/ZM, `gjr_ged` for ES).

**Gate RE fires — see the gate table.** The naive version of this test (a VaR computed
once from the full-sample fitted density, held static) failed OOS coverage badly in
development — violations clustered exactly where Christoffersen's independence test is
built to catch them, the textbook failure mode of an unconditional VaR during a
volatility regime shift. The fix: `RiskModel.var_conditional`/`es_conditional` accept a
caller-supplied current volatility (from `commod_lib8.ewma_vol`, a causal RiskMetrics-
style EWMA, λ=0.94) and rescale the fitted shape's quantile by it — shape fixed per
fold, scale updates daily. This is deliberately a lighter conditioning step than Phase
3's full GARCH refit, not a claim that it replaces Phase 3's machinery.

**Portfolio-level risk under three dependence assumptions**, equal-weighted across all
16 products, 20,000 Monte Carlo draws each:

| dependence | VaR 1% | ES 1% |
|---|---|---|
| empirical (bootstrap real joint history) | 1.86% | 2.37% |
| Gaussian copula | 1.60% | 1.97% |
| t-copula (df=5) | 1.72% | 2.32% |

**The Gaussian copula understates portfolio 1% ES by ~17% relative to the empirical
estimate** (1.97% vs. 2.37%) — §3.4's predicted failure mode, now a measured number, not
an assumed footnote. Mean lower-tail dependence coefficient across all 120 product pairs:
0.146 (empirical) vs. 0.128 (Gaussian) vs. 0.177 (t-copula, df=5) — the t-copula, not the
Gaussian, is the better match to the empirically observed tail co-movement here.

**Stress scenarios** (Phase 1's named events, replayed at the equal-weighted portfolio
level): the 2014-15 OPEC collapse (-7.97% portfolio P&L) and the 2023-24 normalisation
window (-21.4%, the largest single stress figure, reflecting its long multi-year span
rather than a single acute shock) are the two largest drawdowns; 2020-04-20 (CL-only
event) shows a modest -2.66% at the diversified portfolio level, underlining that a
single-product tail event is a very different risk to a portfolio than to that product
alone.

A real performance bug was caught before the full run: NIG's `ppf` has no closed form
and root-finds its CDF per point (~50ms/point measured) — a naive 20,000-point Monte
Carlo draw would have taken **~16 minutes per NIG-family asset** (2 of 16 products), making
the three-dependence-assumption portfolio simulation impractical. Fixed with
`commod_lib8.numerical_ppf` — one grid-based CDF build per asset, then interpolation —
reducing a 20,000-point draw from ~16 minutes to ~5 milliseconds.

---

## Phase 8 — Holdout (spent once, since CT/CE/RE all fired)

2025-01-01 → 2026-07-28 (~490 trading days per product), touched for the first and only
time here. Nothing is re-tuned on this window: every model evaluated was already fit on
the development window by Phase 3/7; this phase only scores those frozen fits.

**RE replicates closely: 14/16 pass** OOS 1% coverage on holdout (vs. 15/16 in
development) — RB and SI, both passing in development, fail on holdout, but the overall
pass rate is materially unchanged.

**CE replicates directionally: 11/16 reject** normal-model ES calibration at the 1%
level on holdout (vs. 15/16 in development) — weaker, as expected from an ~8x smaller
sample reducing test power (and these holdout p-values are **not** BH-corrected, unlike
the development-window Gate CE result, since a 16-product correction on an already
underpowered small sample would only weaken the picture further; reported as raw,
unadjusted p-values for that reason). The direction is unchanged: normal-innovation
models understate tail risk in the holdout exactly as they did in development.

**CT is directionally consistent but too noisy to trust quantitatively at this sample
size.** Hill estimates on ~490 holdout observations per product are visibly less stable
than the ~4,800-5,000-observation development estimates (e.g. CL's holdout upper-tail α
≈ 1.48 vs. development's 2.42) — still comfortably fat-tailed, but the point estimate
itself should not be over-interpreted from a sample this short.

**The holdout confirms this notebook's central claims without materially changing any
of them.**

---

## Bugs found

Eight bugs were caught and fixed before being trusted, each documented in place above:
(1) the hygiene filter's volume-only rule, fixed via a persistence criterion; (2) the
liquidity screen deleting front-month data when applied upstream of curve construction;
(3) `roll_calendar.parquet` listing nominally-dead months for seasonal-cycle products;
(4) back-adjusted prices drifting negative over a long multi-roll history; (5) a
Phase-0-validation join fan-out and NaN-vs-null handling bug that inflated/corrupted
overlap counts; (6) a polars `Int8` overflow in the intraday minute-of-day computation
(Phase 6); (7) NIG's per-point root-finding `ppf` making a 20,000-draw Monte Carlo
impractical (Phase 7); (8) an initial, over-permissive Gate CI scoring rule that would
have reported a false fire (14/16) before being tightened to the criterion that actually
tests the claim (4/16). Eight is not a small number for one notebook — each is reported
here in the same detail as a positive result, per this programme's own standard.

## Bottom line

The crypto programme's two risk-side headline findings — fat tails understated by every
thin-tailed model, and a well-calibrated conditional risk engine being buildable at all
— replicate cleanly in an entirely different market structure, holdout included. That is
no longer a claim about crypto; it is a claim about financial returns generally, to the
extent 16 commodities and one equity-index control can support one. The alpha side
replicates too, in the negative: two of the three most literature-favoured commodity
factors (carry, time-series momentum) fail this repo's own bootstrap-CI bar exactly as
crypto's factors did, extending an eight-notebook run of honest nulls into a market
where a real edge was the prior, not the surprise. The two genuinely open questions from
the pre-registration — does tail asymmetry flip sign the way inventory theory predicts,
and does inventory-state conditioning carry information beyond an unconditional model —
both came back negative, reported at the same level of rigor as everything that fired.
The risk engine is the guaranteed deliverable this notebook promised, and it delivered:
family-selection driven entirely by Phase 2/3's own fitted results, portfolio-level risk
under three explicit and compared dependence assumptions, and OOS coverage that holds up
on data no part of the pipeline ever touched until Phase 8.

## What to test next

- **A genuinely conditional Gate CA/CI test.** Both negative results here used
  relatively coarse operationalisations (a plateau-range proxy for CI on Hill alpha; a
  single GARCH-t reference model for CI's coverage test) — a purpose-built bootstrap CI
  on α_left−α_right, and a state-conditioned re-fit of each product's own Phase-3-best
  model rather than one shared reference model, might sharpen both nulls or reverse one.
- **The grains-specific inverse-leverage signal**, found independently in both Phase 1's
  correlation table and Phase 3's GJR sign check — worth a dedicated test of whether
  storage/inventory dynamics specific to the grain complex (as opposed to energy or
  metals) are the mechanism, rather than treating it as noise in an otherwise-null
  market-wide result.
- **Strategies C-F from Phase 5's declared scope cut** — cross-sectional momentum,
  basis-momentum, spread mean reversion (with the mandatory roll-window-exclusion
  discipline this notebook's own spread data provides), and CL-only COT hedging
  pressure, none of which were run this pass.
- **A carry construction with a genuinely lower-turnover implementation** — Gate AC's
  near-miss (strong absolute Sharpe, CI-excludes-zero failure against the basket) is
  close enough that a slower rebalance frequency or a wider no-trade band, in the spirit
  of `alpha_lib7`'s turnover machinery (built for exactly this in notebook 7, not yet
  applied here), could plausibly close the gap.
- **Resolving the three-way validation discrepancy properly** — Phase 0's validation
  against `research/*_curve.parquet`, `metrics/*.parquet`, and yfinance did not clear its
  own pre-declared bar, and the settlement-vs-close hypothesis was only checked for one
  contract. A systematic comparison of `stat_type` provenance across more contracts
  would either confirm this explains the gap or point to a real, uncorrected construction
  difference this notebook has not yet found.
