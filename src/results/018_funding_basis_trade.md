# Notebook 018 — The Crypto Perpetual Funding Basis Trade (Gate FA): Results Summary

## What

This notebook tests the one candidate from notebook 009's own Phase 3 shortlist that had never been
run: **Gate FA, the crypto perpetual funding basis trade** — long spot, short the perpetual on the
same asset, delta-neutral by construction, collecting the 8-hourly funding payment as the return. It
is the only structurally *non-directional* trade this research programme has attempted; every prior
gate, in one way or another, bet that a feature predicted the direction of a price or a spread. This
one bets that a cash flow is positive more often than it costs to collect.

## Why

009 shortlisted Gate FA with Tier-1 evidence (by analogy to the ~$4tn Treasury cash-and-carry basis
trade) and parked it "not confirmed — needs a spot price series this repo's cached data was not
verified to include." That verification is done here (§Phase 1): Binance's spot, futures/um, and
`premiumIndexKlines` archives all serve native 8h data going back to the start of this repo's crypto
sample, including a still-unspent holdout window. A recent, superficially attractive paper
(AdaptiveTrend, arXiv:2602.11708) was found and explicitly *not* pursued — it is notebook 013's
Design C, already rebuilt twice and already a stronger null than a third attempt could produce.

## How

Pre-registered every gate, constant, and objection answer (`phase_0_18_preregistration.json`) before
any data was fetched. Verified the 009 data blocker live, fetched spot/perp/premium-index klines for
a 128-symbol universe seed (013's own list) across the full 2021-07-01→2025-06-30 development window
plus the 2025-07-01+ holdout (fetched but mechanically fenced off — no Phase 3/4/5 loader can reach
it). Built the trade mechanics (`basis_lib18.py`: causal EWMA carry, per-symbol hysteresis, explicit
two-leg cost accounting) with six required unit tests, ran a no-cost mechanism probe before any
backtest, then the full timed/always-on/cash backtest at four origin offsets, then six pre-registered
ablations.

## Results

**Gate FA-1 fires** (the mechanism exists: pooled mean gross paired return > 0, Newey-West |t| =
3.27) but **Gates FA-2 and FA-3 do not**, so the holdout was never touched. The timed book clears net
Sharpe > 0.5 at every origin offset (+0.577, identical to three decimals — the offsets are vacuous
for this fixed-parameter, non-refit design, the same pattern 012 and Design A found) but fails the
bootstrap-CI and deflated-Sharpe legs of FA-2. Timing clearly helps on a point-estimate basis (timed
net Sharpe +0.577 vs. always-on −0.415) but the paired bootstrap CI on the difference still includes
zero, so FA-3 does not fire either. Gate FA-4 fires cleanly: beta to the equal-weight crypto basket
is 0.0005 and to BTC is 0.0016, both far inside the ±0.10 bound — the hedge genuinely removes crypto
beta, confirmed independently by the perp-leg-only ablation, whose beta is −0.90 to −1.10 without it.
Two real bugs were found and fixed while building Phase 3 (a units error in the break-even-periods
constant, and a small number of price-feed-artifact bars distorting a pooled correlation check), both
disclosed below. A genuine, disclosed capacity finding traces the DSR's harsh verdict to a real
liquidity-screen blind spot, not a code defect.

## Gate verdicts — the full table

| gate | claim | fires? | number behind it |
|---|---|:---:|---|
| **FA-1** (mechanism) | Positive funding carry exists net of basis drift, before costs | **YES** | pooled mean gross paired return 4.30e-05/period, Newey-West \|t\|=3.27 (>3) |
| **FA-2** (tradeable) | Timed book is cost-surviving | **No** | net Sharpe +0.577 at every offset (clears >0.5) **but** bootstrap 95% CI on net return is [-1.3e-05, +7.0e-05] (includes zero) **and** DSR=0.186 (needs >0.95) |
| **FA-3** (timing adds value) | Timing beats always-on | **No** | (timed − always-on) net return 95% CI is [-1.0e-05, +1.05e-04] — point estimate favours timed, CI includes zero |
| **FA-4** (genuinely neutral) | Not a disguised long | **YES** | \|beta\| to crypto basket 0.0005, to BTC 0.0016, both < 0.10 at every offset |
| **FUND** (009 flag) | Institutionally fundable absolute performance | **No** | Sharpe leg passes, DSR leg fails (0.186 < 0.95) |
| **Holdout access** | requires FA-2 AND FA-3 | **Not granted** | neither fired — holdout never read (verified: `run_phase_6_18_holdout.py` refuses, exit 1) |

---

## Phase 0 — Pre-registration

`phase_0_18_preregistration.json`, committed before Phase 1 finished, not edited since. Freezes:
`K`=34bp round-turn cost, `H`=45 periods (15 days) target hold, `θ_in`=7.556e-05, `θ_out`=3.778e-05
(half of `θ_in`, sec 5.3's hysteresis band), `N`=21-period EWMA half-life (7 days), `N_max`=10
simultaneous positions, $5M/day liquidity floor on both legs independently, and the five gates
(FA-1..4, FUND) with their exact fire conditions. `n_trials`=18 (3 books × 4 offsets = 12, + 6 Phase 5
ablations), never revised — every configuration actually run was one of the 18 declared upfront.

Both pre-registration objections are answered in the JSON before any result existed:

- **"007 Phase C already tested funding carry and found it null"** — 007 tested a cross-sectional,
  perps-only, dollar-neutral book betting *funding predicts price direction*; its failure mode
  (rank-churn, ~674-681 round-trips/yr) does not apply to a per-symbol, spot+perp, delta-neutral
  position with no cross-section to rank against. What *does* transfer from 007 is its own prescribed
  fix — a no-trade band tuned to funding's own 8h cadence — which is exactly `qualifies()`'s
  hysteresis, built without calling `alpha_lib7.hysteresis_weights` (cross-sectional/rank-based,
  would reimport 007's own failure mode).
- **"Isn't this just crypto beta with extra steps?"** — Gate FA-4 is the check, and it fires cleanly
  (below).

---

## Phase 1 — Data (spot + perp + premium index, 8h)

The 009 blocker is resolved: spot, futures/um, and `premiumIndexKlines` 8h monthly archives all serve
live data back to 2021-07 (re-verified live in Phase 1, not just from the cached snapshot in
NEXT_PROMPT.md). `src/data.py`'s `download_and_unzip_klines` gained a `market` parameter (own commit,
own test — `tests/test_data.py`) to fetch the spot leg; `premiumIndexKlines` needed a small dedicated
fetcher in `basis_lib18.py` since its URL path is a different endpoint family, not just a different
market.

126 of 128 universe-seed symbols have a usable spot leg (`1000SHIBUSDT`, `1000XECUSDT` do not — a
capacity finding, not an error, counted per sec 4.3). The dev-window fetch (128 symbols × 3 series ×
48 months) hit a live transient DNS failure partway through (116/256 symbol-windows failed with
`NameResolutionError`); because every fetch is idempotent (cached per symbol/series/month), a resumed
run recovered all of it from cache plus the network in under 10 minutes once connectivity returned. A
retry-with-backoff wrapper was added to `run_phase_1_18_fetch.py` afterward so a future transient blip
doesn't need a manual resume. The holdout window (2025-07-01 onward, including funding rate history,
which had to be freshly fetched since the existing repo-wide funding cache stops at 2025-06-30) was
fetched in the same pass into a **separate cache directory** (`basis18/holdout/`) that no Phase 3/4/5
loader can read — `basis_lib18.load_basis_panel` structurally guards past `research.HOLDOUT_START`,
and only `run_phase_6_18_holdout.py` ever names the holdout path.

---

## Phase 2 — Library (`basis_lib18.py`)

`carry_estimate` (causal EWMA, `half_life=21` — a half-life, not a span, per polars' own ambiguity),
`paired_log_return` (explicit two-leg difference, not the sec 3.2 basis approximation, so the
approximation can be *checked* against it rather than assumed), `qualifies` (per-symbol absolute-
threshold hysteresis), and `apply_two_leg_costs` (explicit accounting for the 10bp spot / 5bp perp fee
split, cross-checked in a test against the blended-rate call to `research.add_portfolio_costs`). Six
required tests plus the cross-check, all green; `build_book_weights` does the sequential, causal,
per-symbol book construction (hysteresis-gated for "timed," liquidity-gated only for "always-on" —
disclosed as an implementation reading of an already-frozen book definition, not a swept parameter).

---

## Phase 3 — Mechanism probe (no cost model, no Sharpe, no strategy verdict except FA-1)

**Two real bugs found and fixed here, both disclosed rather than quietly patched:**

1. **A units error in the break-even-periods constant.** The first implementation computed
   `34bp / (0.01 * 3)` intending "3bp/day," landing on **1133 periods** instead of the sec 3.4 target
   of **34**. Caught because the persistence check's `frac_runs_clearing_breakeven` came back at
   0.1% — implausibly low against the sec 3.4 prior. Fixed to `34bp / 1.0bp-per-period = 34`.
2. **A handful of implausible-basis bars distorting two pooled statistics.** The premium-index
   cross-check's pooled Pearson correlation came back **−0.08** (materially disagreeing, by the
   pre-declared bar) even though the per-symbol distribution is healthy (median 0.744). Traced to two
   symbols: `DGBUSDT`'s perp close froze at a stale value for several days in Nov 2024 (volume=0,
   verified directly against the cached kline data) while spot kept moving, producing a computed basis
   of **+275%**; `LUNAUSDT`'s real 2022 collapse divides by a near-zero spot price, producing values
   over 100×. Both are excluded from the descriptive checks in this phase (not from the backtest
   universe) by a stated `|basis| > 20%` sanity bound, with the exclusion count reported
   (5,066 of 461,298 obs, 14 symbols touched). After exclusion: premium-index correlation is healthy
   (median per-symbol 0.744, pooled-after-filter recovers to a sane positive value), and the
   funding/basis-change identity check's correlation rises from 0.48 to **0.994**.

With both fixed, the substantive findings:

- **Gate FA-1 fires.** Pooled mean gross paired return 4.30e-05/period, Newey-West |t|=3.27 (>3),
  n=461,298 (post-filter).
- **Funding dominates the basis-change term, by both magnitude and significance.** Funding's pooled
  mean is highly significant (t=20.5); basis-change's mean is **not** distinguishable from zero
  (t=0.33) — exactly E1's claim and sec 3.2's own prediction that the basis term is mean-reverting
  and roughly zero-mean, i.e. noise around the funding drift, not a competing source of return.
- **E2's 2025 funding decay does not (yet) show up in this repo's dev-window data** — dev only
  reaches 2025-06-30, so only H1 2025 is visible, and pooled funding by year is actually higher in
  2024 (mean 1.10e-04, t=45.6) than 2021 (1.41e-04, t=25.8) or 2023 (3.60e-05, t=6.1); 2022 is
  slightly negative (−4.22e-05, t=−11.7). This is reported as a genuine limit on this notebook's own
  replication of E2's claim, not smoothed over: the interesting comparison (2024 vs. 2025) sits inside
  the holdout, which was never spent.
- **Funding regimes persist long enough to matter, but not by a wide margin.** 44% of carry-above-
  θ_in runs last ≥34 periods (the break-even hold) — a real, exploitable persistence, not
  overwhelming, matching sec 3.4's own "feasible but not by a wide margin" framing exactly.

---

## Phase 4 — The backtest (timed / always-on / cash, 4 origin offsets)

| book | gross Sharpe | net Sharpe | net max drawdown | annualized turnover |
|---|---:|---:|---:|---:|
| **timed** (headline) | 2.66 | **+0.577** | −8.6% | 56.9/yr |
| always-on | 0.21 | **−0.415** | −25.0% | 15.2/yr |
| cash | 0.0 | 0.0 | 0.0% | 0 |

(All four origin offsets — 0/1/2/3 periods — agree to 3+ decimals; vacuous for this fixed-parameter,
non-refit design, the same pattern 012 and Design A found, disclosed rather than presented as
robustness.)

**FA-2 fails on the bootstrap-CI and DSR legs, not the Sharpe leg.** Net Sharpe (+0.577) clears the
0.5 bar at every offset. The 95% block-bootstrap CI on net return is `[-1.31e-05, +7.02e-05]` —
includes zero. DSR is 0.186, far under 0.95, computed honestly at the pre-declared `n_trials=18`
— **and per sec 11.4's pre-registered caveat, `research.deflated_sharpe_prob` is a known-harsh
estimator for this specific kind of trial family (near-identical origin offsets), not fixed here** (that
is notebook 017). The sample's extreme skew (−11.5) and kurtosis (817) driving that DSR down were
investigated, not accepted at face value (see Concentration finding below) — real, not a stray input
error, but their *cause* is disclosed and explained, not just reported as a number.

**FA-3 fails despite a large point-estimate gap.** Timed net Sharpe (+0.577) vs. always-on
(−0.415) is a 0.99 Sharpe-point spread, and the underlying mechanism makes sense: timed only holds
positions when carry clears θ_in, so its *gross* Sharpe (2.66) is 12× always-on's (0.21) — timing
selectively captures the best funding periods. But turnover is *higher* for timed (56.9/yr) than
always-on (15.2/yr, since always-on barely changes membership), so the improvement has to clear a
real cost hurdle, and the paired bootstrap CI on (timed − always-on) net return,
`[-1.03e-05, +1.05e-04]`, still includes zero. Read plainly: timing looks like it helps a great deal,
and the data cannot yet rule out that it doesn't.

**FA-4 fires cleanly.** Beta to the equal-weight crypto basket is 0.0005, to BTC 0.0016 — both two
orders of magnitude inside the ±0.10 bound, at every offset. The paired construction is genuinely
delta-neutral, not luck; confirmed independently in Phase 5's perp-leg-only ablation.

**Concentration finding (why the DSR's skew/kurtosis are so extreme, investigated rather than
reported blind).** The timed book holds a median of 10 symbols (the cap) and is at the cap 54% of
bars, but is down to a **single symbol 5.4% of bars** — the equal-weight-among-qualifiers
construction has no diversification floor below `N_max`. The five worst single-bar net returns are
dominated by exactly this: `ICPUSDT` alone on 2022-06-25 (−5.7%) and `MATICUSDT` alone or nearly alone
across four bars in early Sept 2024 (−0.6% to −2.0%). Both trace to a **real perp-market liquidity
collapse**, verified directly against the cached kline data (open=high=low=close, volume=0 for
several consecutive days) — Binance's own MATIC→POL rebrand transition (already documented by 013 as
ending that symbol's clean feed after 2024-10) and, for ICP, a genuine multi-day zero-volume stretch.
During a zero-volume stretch the short perp leg cannot actually be hedged (there is no real price
discovery to hedge against), and the pre-registered 30-day trailing-**median** liquidity screen is
slow to catch a sudden collapse — it does catch ICPUSDT's, one bar after the worst loss. This is a
structural property of a median-based screen and an uncapped-below-`N_max` equal-weight book, not a
code defect, and it is **not** fixed here by changing the frozen liquidity floor, lookback window, or
book-construction rule (sec 12 forbids exactly that kind of after-the-fact tuning) — it is reported as
a capacity/robustness finding, in 009's own idiom.

**Holdout access: not granted.** FA-2 and FA-3 both need to fire; neither did.
`run_phase_6_18_holdout.py`, invoked directly to confirm the gate itself works, refuses (exit code 1)
without reading `src/research/cache/basis18/holdout/` — verified, not just asserted.

---

## Phase 5 — Controls and ablations (6 exhibits, `n_trials` 12→18, exactly as pre-declared)

| exhibit | headline number | finding |
|---|---|---|
| **no-hysteresis** (θ_out=θ_in) | net Sharpe +0.577 → **−0.726**, turnover 56.9→95.4/yr | the hysteresis band is genuinely load-bearing — 007's own prescribed fix matters here exactly as hypothesized |
| **perp-leg-only** (no spot hedge) | beta_basket −0.897, beta_btc −1.103 (hedged: 0.0005 / 0.0016) | confirms the spot hedge, not luck, is what removes beta — max drawdown on this unhedged variant is −99.6%, a near-total wipeout that also shows why the hedge matters practically, not just statistically |
| **excluding LUNA/FTT** | net Sharpe +0.577 → **+0.562** | the headline does not depend on either collapse |
| **cost sensitivity** (0/17/34/51bp round turn) | Sharpe 2.66 / 1.62 / 0.577 / −0.450 | crosses zero between the actual 34bp and 51bp — linear interpolation puts the break-even round-turn cost at roughly **43-44bp**, about 1.3× today's retail rate; a lower (VIP/institutional) fee tier would clear it by a wider margin |
| **levered 3×/5×** (with sec 6.3 liquidation analysis) | Sharpe 0.52 / 0.45, but 1% ES on the perp leg's own 8h return is **13.9%** | at 3×/5× that is 42%/70% of deployed capital in a single bad 8h period — the levered Sharpes should not be read as investable without this attached; cited to notebook 006 (crypto tail-shape families), not 008 (commodity futures, does not transfer) |
| **by-year decomposition** | net Sharpe 2021(H2) +7.56, 2022 −0.19, 2023 +1.72, 2024 +0.67 | 2021 H2's number is a small sample (~547 bars, half a year) during crypto's highest-funding era and should not be over-read; 2022-2024 show a real, if noisy, decline consistent with (but not a full replication of, since 2025 sits in the untouched holdout) E2's decay claim |

---

## Phase 6 — Holdout: **not spent**

Per the pre-registration, Phase 6 runs only if Gate FA-2 **and** Gate FA-3 both fire on development.
Neither did. `run_phase_6_18_holdout.py` — the only file in this repo that names
`src/research/cache/basis18/holdout/` or reads past `research.HOLDOUT_START` for this notebook
(`grep -rn "basis18/holdout\|HOLDOUT_START" src/research/tmp/*18*.py` confirms) — was invoked once,
directly, specifically to verify it refuses correctly: it printed the `holdout_access` block read back
from `phase_4_18_results.json`, declined, and exited 1 without ever calling
`bl._load_basis_panel` against the holdout directory. The 2025-07-01+ window remains exactly as spent
(not at all) as it was before this notebook ran — data for it sits pre-fetched in
`basis18/holdout/` for whichever future notebook next has a fired gate to spend it on, per sec 9.3's
"fetching it early costs nothing" rationale.

---

## Bugs found

Two real, in-flight bugs, both caught by suspicious numbers before being trusted (this repo's own
standing tripwire discipline — an implausible result gets investigated, not reported):

1. **Break-even-periods units error** (Phase 3) — `34bp / (0.01*3)` computed 1133 instead of 34;
   caught because the persistence check's clearing fraction was implausibly low (0.1%) against the
   sec 3.4 prior. Fixed; clearing fraction is now 44%, matching the prior.
2. **Pooled statistics distorted by two symbols' price-feed artifacts** (Phase 3) — DGBUSDT's frozen
   perp price (verified: volume=0 for multiple consecutive days) and LUNAUSDT's real 2022 collapse
   both produce basis values in the hundreds-of-percent range, which dominated a naive pooled Pearson
   correlation even though 124 of 126 symbols individually agree well (median per-symbol correlation
   0.744). Fixed by reporting the per-symbol correlation distribution as the primary comparison and
   excluding `|basis| > 20%` bars from pooled statistics, with the exclusion count disclosed rather
   than silently absorbed.

One additional near-miss, not a code bug: a naive read of Phase 4's extreme DSR sample skew/kurtosis
could have been reported as "the estimator is unreliable here" and left at that. Tracing it instead
(Phase 4's concentration diagnostic) found a real, structural, disclosable finding — a
median-liquidity-screen blind spot during a sudden perp-market liquidity collapse — rather than
either hand-waving the number away or, worse, silently patching the liquidity screen to make it go
away (which sec 12 explicitly forbids as post-hoc tuning).

Also disclosed, not a bug: `join(..., suffix="_basket")` in Phase 4/5/6's beta computation does
nothing unless the joined frames' column names already collide — three call sites initially relied on
it and would have silently computed beta against the wrong (unrenamed) column name, caught by running
the scripts (`ColumnNotFoundError`, not a silent wrong number) before any result was trusted. Fixed by
renaming the basket column explicitly before joining, in all three phase scripts.

## Bottom line

**Gate FA-1 fires — a genuine, statistically significant funding carry exists in this repo's own
data, and it is driven by funding, not basis drift, exactly as E1 claims.** But **Gates FA-2 and FA-3
do not fire**, so this is not a tradeable strategy by this notebook's own pre-declared bar, and the
holdout stays unspent. The verdict is more informative than a flat null: net Sharpe clears the
absolute 0.5 bar at every origin offset, and timing shows a large, economically sensible edge over
simply holding the trade always-on (12× the gross Sharpe) — but neither the bootstrap CI on net
return nor the paired CI on timing's own value-add can yet rule out zero, and the deflated-Sharpe
estimator (already flagged in this programme's own methodology notes as likely too harsh for a
same-strategy, near-identical-offset trial family) fails decisively. **Gate FA-4 fires cleanly: this
is a genuinely delta-neutral book, not a disguised long, confirmed two independent ways** (direct beta
measurement, and the perp-leg-only ablation's beta collapse to -1 when the hedge is removed). This is
the first gate in this programme's thirty-one-gate history where a real, structurally different
mechanism (a cash flow, not a forecast) shows up statistically significant before costs and survives
its own most direct validity check (neutrality) — it simply does not yet clear the higher bar of
being a demonstrably tradeable edge net of realistic costs and multiple-testing correction. The
Sharpe-1.2-to-2.4 development-window expectation this notebook's own pre-registration set going in
(sec 1) was not met; the honest net Sharpe is 0.577, inside the wider "plausibly under 1.0" band the
same section flagged as the realistic floor.

Real, structural reasons for caution about extrapolating even the point estimate: a $5,000,000/day
liquidity floor with no diversification requirement below the 10-position cap leaves the book
occasionally (5.4% of bars) concentrated in a single name, and the worst outcomes in this whole
backtest come from exactly that combination meeting a genuine perp-market liquidity collapse the
screen is too slow to catch. Reverse carry (negative funding) was never tested, by design (sec 3.5) —
the strategy as built is flat, not short, in negative-funding regimes, which caps the upside relative
to E1's own 6.45 figure (which does not appear to impose this constraint). Leverage, which is the
only way to make this size-competitive with E1's headline, carries a 1% expected shortfall on a
single 8h period equal to 42% (3×) to 70% (5×) of deployed capital on the unhedged leg alone.

Machinery: `src/data.py` (`download_and_unzip_klines` gained a `market` param, own commit/test),
`src/research/tmp/basis_lib18.py` (the library — carry, paired return, hysteresis, two-leg costs, book
construction, beta), `src/research/tmp/run_phase_{1,3,4,5,6}_18_*.py` (fetch driver, mechanism probe,
backtest, ablations, gated holdout runner), `tests/test_basis_lib18.py` (the six required tests plus
one cross-check), `scripts/fetch_basis_data.sh` / `run_backtest.sh` / `run_ablations.sh` (background-
safe runners with `scratch/018/status.json` heartbeats). `src/risk/` used only as machinery
(`ewma_vol` implicitly via the repo's existing conventions was not needed; `risk.model.fit_risk_model`
+ `risk.densities` for the sec 6.3 liquidation-tail fit, cited to notebook 006) and never to time
entries, exits, or position size (sec 6.4). `src/risk/` and `src/regime/` were imported only, never
modified. The 2025-07-01+ crypto holdout remains exactly as unspent as every notebook before this one
left it, now pre-fetched into a mechanically fenced-off directory for whichever future notebook next
has a fired dev gate.

## What to test next

- **A joint diversification floor**, e.g. a minimum-N-symbols requirement below which the book stands
  down entirely rather than concentrating, would directly address the concentration finding — but
  this is a new, pre-registerable design choice for a future notebook, not a retroactive edit to this
  one's frozen construction (sec 12).
- **A faster or dual-signal liquidity screen** (e.g. a same-day zero-volume veto layered on top of the
  30-day trailing median) would close the specific detection-lag gap this notebook found, at the cost
  of a second pre-registered parameter and a corresponding `n_trials` increment.
- **The DSR question this notebook's own numbers sharpen**: with FA-2 failing specifically on a
  DSR leg already flagged as likely-too-harsh for a near-identical-offset trial family, notebook 017's
  deferred estimator correction is now motivated by a live, borderline case rather than only a
  methodological concern — worth prioritizing.
- **A genuinely lower-turnover carry construction** (a slower EWMA, or a wider band re-derived from a
  longer target hold) might clear the bootstrap-CI leg the current design misses by a small margin —
  but any such change needs its own pre-registration and gate, not a retroactive tune of this one.
- **Fee-tier sensitivity as a standing exhibit**: this notebook's own cost-sensitivity table already
  shows the trade clears zero around 43-44bp and the current retail-tier round turn is 34bp — worth
  tracking against Binance's actual VIP tier schedule (Tier 3/4 sourcing caveats apply, per 009's own
  near-miss on this exact question) rather than re-deriving it per notebook.
