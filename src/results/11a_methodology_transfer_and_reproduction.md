# Notebook 11a — Methodology Transfer and Reproduction: Results Summary

**This notebook is descriptive and infrastructural only — no gate verdicts, no Sharpe-based
strategy conclusions (NEXT_PROMPT.md sec 1 rule 1, unchanged from the 10a/10b split).** Its
purpose is to absorb a second, independent codebase's spread-trading research
(`~/Documents/ultron/apps/trading-labs`) into this repo: port its evaluation machinery, build
its evaluation harness, reproduce its control book on our own data and statistics, settle the
data-quality question its own v3 correction opened, and — its actual deliverable — pre-register
11b/11c/11d's full gate table, DSR trial counts, and the sec 4.3 include/exclude decision
**before any of those three notebooks' backtests exist**. That pre-registration (Phase 6,
`phase_6_11a_results.json`) is committed as part of this notebook and may not be edited once
11b/11c/11d starts running.

## The half-life corroboration is real, and it closes their own top open item

The external programme's own v3 correction (2026-07-27) found a contract-substitution bug that
corrupted its entire 12.6-year spread series; on correction, three of six of its strategy
verdicts changed materially, and two of those changes traced back to a half-life measured on
the corrupted data (`brent_calendar`: 1.7–4.7 days corrupted vs. 28–73 days, mean 59.5,
corrected). Their v4's own highest-priority follow-up was to remeasure half-life on the
corrected series as a standalone check.

This repo already had that measurement — 10a Phase 2's AR(1)-in-differences half-life, built
independently from Databento `ohlcv`/`contracts`/`roll_calendar` via
`commod_lib8.build_continuous_series`, with no code shared between the two repos at any point.
Phase 0 re-derives it here and Phase 1 reproduces it a second way (`spread_lib11.rolling_stability`,
a from-spec reimplementation, calling the same `research_lib9.ols_ar1_diff` primitive on the
same roll-window-excluded series): every one of the five half-lives (`brent_calendar` 42.7d,
`brent_wti` 79.3d, `corn_wheat` 45.4d, `bean_corn` 118.2d, `kc_chicago_wheat` 113.4d) lands
inside their *corrected* ranges and nowhere near their *corrupted* ones. This is a genuine
cross-repo validation result, obtained without either programme ever reading the other's data
pipeline, and it substantially closes their v4's own open item.

## We cannot validate their reported control-book Sharpe

Phase 4 runs NEXT_PROMPT.md sec 4.1's pre-declared trading rule — entry/exit z-score
thresholds, per-spread ATR stops (`brent_calendar` 4.0×, `kc_chicago_wheat` 12.0×, 6.0× global
default elsewhere), fixed-fractional risk sizing, the vol/vol-regime suppression filters, the
cooldown/gated-reentry mechanism, and `brent_calendar`'s backwardation-only regime gate — on
the five live spreads (`brent_wti`, `brent_calendar`, `corn_wheat`, `bean_corn`,
`kc_chicago_wheat`), on our own dev-window data (2010-06-06 to 2024-12-31), under two cost
models: ours (`commod_lib8.round_turn_cost_per_contract`, materially more conservative per sec
3.1's own cost table) and a reimplementation of their stated $2/contract + 5bps + 2bps flat
cost.

| metric | theirs (tune, 2014–2023) | ours, our costs | ours, their costs |
|---|---:|---:|---:|
| fixed-notional return | +85.1% | −1.1% | −1.1% |
| equity-path return | +122.4% | −1.1% | −1.1% |
| Sharpe | 0.889 | −0.16 | −0.16 |
| max drawdown | −7.30% | −1.9% | −1.9% |
| n trades | 333 | 57 | 57 |

**This is a material divergence on every axis, and it is reported honestly rather than tuned
away** — exactly the outcome NEXT_PROMPT.md sec 3 Phase 4 explicitly flagged as "a live
possibility" given the data-corruption history. `phase_4_11a_results.json`'s reconciliation
record lists the candidate, non-exhaustive explanations without isolating one: a dev window
that is longer and differently dated than their 2014–2023 tune window; an independently-built
spread series; this notebook's own reimplementation-from-specification (not their code) of the
suppression filters and the gated-reentry mechanism, including an approximate ADF p-value
(`spread_lib11.approx_adf_pvalue`, a linear interpolation against three tabulated critical
values, not the exact Dickey-Fuller CDF); and `simulate_book`'s documented joint-sizing
simplification — five independently-sized per-spread books pooled by dollar P&L, not one
shared-equity risk engine enforcing the real joint `max_gross_exposure_pct`/
`daily_drawdown_limit_pct` caps.

The practical consequence: **every 11b comparison against this control book is internal**
(structured vs. unconditional for Gate TS, sign-flipped vs. unconditional for Gate BF,
screen-inclusive vs. screen-exclusive for Gate SCR, and so on) — never a validation of their
absolute reported numbers. The noise floor on our own book (95% CI half-width of a few
percentage points around a near-zero point estimate) states plainly what any of those internal
comparisons can resolve.

## The new screen is strict enough to reject their own flagship spread

Phase 3 rebuilds both screens. The **old screen** — ADF on the 30-day deviation from a 30-day
rolling mean — fails its own random-walk check completely: run on 20 synthetic pure random
walks, it flags all 20 as stationary (median t-stat deeply negative, matching the ≈1e-19-p-value
order of magnitude the external repo reported for the same construction). Detrending a random
walk against its own rolling mean manufactures a bounded, spuriously stationary residual almost
by construction — a screen built this way carries no information about genuine mean reversion.

The **new screen** — ADF-on-level AND variance ratio (q=5, q=20, one-sided z=1.645) AND
Hurst<0.5 AND half-life stability (full-sample half-life in a 3–60-day band AND ≥3 of 4
contiguous sub-periods also in band) — calibrates close to its nominal 5% false-positive rate
on 500 seeded random walks at both q=5 (5.8%) and q=20 (4.6%), so the variance-ratio component
itself is not miscalibrated. But applied to all 30 of our spreads it passes only 8, against the
old screen's 23. Strikingly, `brent_calendar` — one of the external repo's own five live
spreads, with a clean ADF rejection (t=−5.22) and the 42.7-day half-life corroborated above —
fails the new screen on its variance-ratio leg alone (VR(5) z=+1.71, positive rather than the
required <−1.645). This is a real finding, not a screen bug: `brent_calendar`'s daily changes
show short-horizon positive autocorrelation (momentum at the 5-day horizon) layered on top of
genuine long-horizon mean reversion measured by the AR(1)/half-life test — the two diagnostics
measure different things and can legitimately disagree. Whether this stricter screen earns its
place in 11b's trading universe, or is itself an example of sec 0.3's "any mechanism that
improves quality by deletion should be expected to fail," is Gate SCR's decision, not assumed
here.

## The trade-shape corroboration is the strongest evidence the mechanism, not just the number, is real

Phase 5 replicates their `pattern-summary.md` analysis on our own 57-trade Phase 4 book.
Despite the aggregate-P&L divergence above, the trade *shape* corroborates theirs closely:

| | ours (n=57) | theirs (n=9,545 pooled) |
|---|---:|---:|
| entry `\|z\|` discriminates winners/losers | no (2.13 vs. 2.23 median) | no (2.06 vs. 1.95 median) |
| stop-exit fraction, top half vs. worst half | 0% / 82% | 0% / 85% |
| loss:win `pnl_atr` asymmetry | 2.09× | ≈3.4× (698 stop trades at −6.94 ATR vs. 6,837 z-score trades at +2.05 ATR) |

An independently-built book, roughly 170× smaller, under a materially stricter cost model,
still reproduces the central pattern: entry extremity does not discriminate winners from
losers, and the catastrophic tail comes almost entirely from stop-exits while the bulk of
trades exit cleanly via z-score normalization. That the *shape* survives even though the
*magnitude* diverges is meaningful — it suggests the underlying mechanism (a small number of
adverse continuations doing most of the damage) is a genuine property of this trading rule
family, not an artifact of one repo's particular parameterization or data. This is the
empirical basis for 11c's entry-time loss classifier: worth building properly even though sec
6's own honest prior, informed by this exact pattern, is that entry-time features will not
predict the tail.

## What is pre-registered for 11b/11c/11d (Phase 6)

Ten gates (TS, TS-S, BF, BF-X, SCR, VA, RE for 11b; LC for 11c; MB, MB-E for 11d), transcribed
verbatim from NEXT_PROMPT.md sec 4.2/6/7, with their firing criteria fixed before any backtest.
DSR trial counts total **85** across all ten gates, transcribed verbatim from NEXT_PROMPT.md
sec 9 — Gate RE's count of 36 (a genuine 3×3×4 grid) is the largest single component and is not
to be reduced even if it proves unreachable; an unreachable grid is itself the finding. Gate
SCR's two competing universes (the 10a ADF-passing screen vs. the full eligible universe
including `kc_chicago_wheat`, `gc_cal_m2m3`, and `es_calendar`) are fixed here, with resolution
of the two named cross-repo conflicts (`gc_cal_m2m3` and `es_calendar`, not `kc_chicago_wheat`,
which is simply one of the five live spreads and not itself in conflict) left to Gate SCR's own
paired comparison, not assumed. COT positioning extremes are recorded as a hard data gap — this
repo's `data/market/cot/` holds only CL, not the corn/wheat/soybeans series their proposal
needs — and are out of scope for every 11b/c/d gate, not proxied.

The holdout (2025-01-01 to 2026-07-28) is untouched by this notebook and remains unspent, but
its independence for commodity-spread strategies specifically is disclosed as reduced: the
external programme's own held-out window (2024-01-01 to 2026-07-21) overlaps it, and their
held-out numbers were read during this notebook's design (NEXT_PROMPT.md sec 8). Every future
write-up touching the holdout for 11a/11b/11c must carry that disclosure; 11d's crypto momentum
work is unaffected, since the external crypto programme produced no held-out numbers.

## Also flagged, not resolved, here

`spread_lib11.carry_ratio`'s literal implementation of `c_t = -value_t / full_carry_t` — as
specified in NEXT_PROMPT.md sec 3 Phase 1 — evaluates to approximately −1 at the deep-contango
"full carry" boundary under this repo's leg1-front sign convention, the opposite of the "+1 at
the contango ceiling" description in the same spec. The function is implemented literally, and
the discrepancy is documented in its own docstring rather than silently corrected; it does not
affect anything computed in 11a and is left for 11b — the first notebook to actually consume
`carry_ratio` for Gate BF — to resolve against the external repo's own live output.

Machinery: `src/research/tmp/run_phase_{0..6}_11a_*.py` (Phase 0 reproduction check; Phase 1
ported primitives; Phase 2 evaluation harness demo; Phase 3 screen rebuild/calibration; Phase 4
control-book reproduction; Phase 5 trade-shape atlas; Phase 6 pre-registration),
`src/research/tmp/spread_lib11.py` (the new computation this notebook needed — z-score/ATR
primitives, fixed-fractional sizing, carry fair value/ratio, term-structure regime label,
variance ratio, Hurst exponent, rolling half-life/ADF/stability, the paired block bootstrap and
noise-floor construction, and the full single-position-per-spread backtest engine), with unit
tests in `tests/test_spread_lib11.py`.
