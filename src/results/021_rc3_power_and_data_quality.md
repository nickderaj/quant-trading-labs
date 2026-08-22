# Notebook 021 — Is RC-3 Blocked by Data Artifacts, or by Statistical Power?: Results Summary

## What

020 built a refined basis book (`A3`) that beats 018's own construction on every *absolute* measure —
net Sharpe 3.89 vs. 0.58, DSR 0.9999997, worst single-bar loss down from −5.7% to −0.38% — and clears
its own tradeability gate (RC-2) cleanly. But 020's RC-3, the *paired* comparison against 018's own
frozen reproduction, did not clear its 95% bootstrap CI: `[-2.20e-05, +5.62e-05]`, point estimate
favouring the refined book but the CI still straddling zero. 021 asks one narrow, pre-registered
question about that single gap: is the non-fire driven by data-quality artifacts in the baseline's own
return series, or is it a statistical power problem that no amount of exclusion can fix? One rule, one
answer, one of three pre-decided branches.

## Why

018's own detection signature for a frozen perp feed (`open == high == low == close AND volume == 0`)
was never applied to the *paired comparison* between 018's baseline and 020's refinement — only to
018's own descriptive statistics. If `A0` (the baseline) happened to be holding a symbol whose feed
froze during 020's dev window, its recorded "return" on those bars is a mark-to-market illusion, not
captured funding — and if enough of those bars sit in the diff series, they could be inflating the
baseline's own apparent performance just enough to keep the diff CI straddling zero. That is a
mechanically checkable, falsifiable hypothesis, cheap to test against data already on disk, and it has
a genuinely different fix than "the sample is too small" — which is why it gets pre-registered as its
own notebook rather than folded into 020 (whose own frozen numbers stand exactly as recorded) or left
as a shrug.

> **H-P**: RC-3's non-fire in 020 is driven materially by data-quality artifacts in the baseline's own
> return series — bars where `A0` holds a symbol whose perp feed is frozen, producing apparent gains
> that are mark-to-market illusions rather than captured funding. Removing those bars from the paired
> comparison only (never from either backtest) moves the diff CI toward, and possibly past, zero. If
> it does not, statistical power — not data quality — is the binding constraint.

## How

Pre-registered the exclusion rule, all four gates (PW-1..PW-4), the power-formula constants, and the
6-trial itemisation (`phase_0_21_preregistration.json`) before Phase 1 ran, then reproduced 020's own
RC-3 diff CI from the two stored returns parquets to 1e-12 as this notebook's tripwire. Scanned all 128
cached full-range perp files with 018's own frozen-feed signature — no halo, no minimum run length, no
widening, verified sufficient because it covers both of 018's own documented events (`ICPUSDT`,
`MATICUSDT`) as single contiguous runs with no free parameter needed. The detector script
(`run_phase_1_21_catalogue.py`) names neither symbol anywhere in its own code, checked by grep, not
just asserted. Rebuilt only `A0`'s and `A3`'s *weights* (no book rebuild — the diff series is reused
verbatim from 020's own stored returns parquets) to intersect the flagged catalogue with `A0`'s actual
holdings under the pre-registered `{T-1, T}` book-return alignment rule. Computed the diff CI with and
without the exclusion, the closed-form MDE and required sample size for both, and a 200-draw
same-size-random-exclusion placebo (seed 0, means only) to test whether the exclusion's effect is
specific to the flagged bars or just generic outlier-trimming. Total measured wall time for a clean,
from-scratch run: 50.7 seconds.

## Results

**PW-1 fires cleanly.** The mechanical detector, blind to both symbol names, independently rediscovers
018's own two documented events: `ICPUSDT` flagged inside 2022-06-20..2022-06-30 (31 bars in that
10-day window, part of the full 247-bar 2022-06-10..2022-08-31 run) and `MATICUSDT` flagged inside
2024-09-01..2024-09-30 (all 21 bars of its 2024-09-04..2024-09-11 run).

**PW-2 nominally fires.** Excluding the 22 book-return bars where `A0` held a frozen-feed symbol
(0.57% of the 3,840-bar series) moves the diff CI from `[-2.20e-05, +5.62e-05]` (point estimate
1.99e-05, p=0.249) to `[1.02e-05, 3.10e-05]` (point estimate 2.03e-05, p=0.000) — the lower bound
clears zero.

**But PW-4's placebo catches it, and PW-4 does not fire.** The exclusion easily clears its own 5% cap
(0.57% ≪ 5%), but the flagged-exclusion mean diff (2.032e-05) sits *just inside* the 95th percentile of
200 same-size random exclusions (2.087e-05) — meaning dropping *any* 22 bars from this series tends to
move the mean by roughly this much, not something specific to the flagged bars. Per the pre-registered
cap/placebo rule, PW-2's number is reported but declared non-load-bearing.

**PW-3 does not fire on the original series**, and this is the headline: the observed mean diff
(1.99e-05) is well below the minimum detectable effect at 80% power (5.59e-05) given the series' own
noise. Closed-form: detecting an effect this size at 80% power, 5% two-sided, would need ≈30,205 paired
8h bars — about **27.6 years** of paired history — against the **3.50 years** the paired series
actually has. (For reference, the excluded series alone — non-load-bearing per PW-4 — would need only
≈1.9 years, i.e. would already look adequately powered; that number is exactly why PW-4's placebo
control matters here rather than being decorative.)

**Branch (b): statistical power, not data quality, is the binding constraint.** No holdout access is
granted (021's own pre-registered policy grants none under any outcome).

## Gate verdicts — the full table

| gate | claim | fires? | number behind it |
|---|---|:---:|---|
| **PW-1** (detector sound) | mechanical catalogue covers both documented events, no symbol named in detector code | **YES** | ICPUSDT: 31 bars in 06-20..06-30; MATICUSDT: 21 bars in 09-01..09-30; grep for both names over `run_phase_1_21_catalogue.py` returns nothing |
| **PW-2** (data-quality-corrected RC-3) | excluded-bar diff CI excludes zero | **YES** (non-load-bearing, see PW-4) | 95% CI `[1.02e-05, 3.10e-05]` |
| **PW-3** (adequately powered) | observed mean diff ≥ MDE at 80% power | **No** | mean 1.99e-05 < MDE 5.59e-05 (without exclusion) |
| **PW-4** (surgical exclusion, not a reshape) | <5% excluded AND beats 95th-pct placebo | **No** | 0.57% excluded (cap OK) but mean 2.032e-05 < placebo p95 2.087e-05 (placebo fails) |
| **Holdout access** | granted under no outcome (021's own policy) | **Not granted** | pre-registered, unconditional |

---

## Phase 0 — Pre-registration + reproduction tripwire

`phase_0_21_preregistration.json`, committed before Phase 1 ran. Freezes hypothesis H-P, the
zero-free-parameter exclusion rule and its `{T-1, T}` book-return alignment (`A0`'s holdings only,
never the union with `A3`), the four gates PW-1..PW-4 with the pre-derived 5% PW-4 cap (018's own
single-symbol-bar fraction, 5.4%, fixed as the cap before this notebook's own count was known), the
power-formula constants (`z_0.975=1.959964`, `z_0.80=0.841621`, MDE multiplier 2.801585, 1095.75 bars/
year), the placebo spec (200 draws, seed 0, means only), and the 6-trial itemisation with an explicit
note that no DSR is computed anywhere in this notebook, so no deflation applies. `run_phase_0_21_
preregistration.py` then reproduced 020's own RC-3 diff CI from `scratch/020/cells/phase4/{A0,A3}-0.
returns.parquet` and asserted it matches `(-2.199972533099506e-05, 5.617934418888986e-05)` to 1e-12 —
it did, exactly, on n=3840 bars.

## Phase 1 — The mechanical catalogue

`run_phase_1_21_catalogue.py` scans all 128 files matching `src/research/cache/basis18/dev/*-perp-8h-
2021-07-01-2025-06-30.parquet` with the bare signature (no halo, no widening) in under two seconds and
flags **19,157 symbol-bars across 21 symbols**. As sec 1 anticipated, most of that is post-delisting
dead-feed tail (`SCUSDT` 3,326 bars, `FTTUSDT`/`RAYUSDT` ~2,875 bars each, `DGBUSDT` 1,364, `WAVESUSDT`
1,151, `OCEANUSDT` 1,109, and a dozen more single-contiguous-run tails running to 2025-06-30) — the
quantity that matters is book-return bars, computed in Phase 3. The catalogue is written as per-symbol
contiguous runs (start, end, count) — a compact, lossless disclosure of the full flagged list — to
`phase_1_21_catalogue.json` (committed) and the raw per-bar frame to a gitignored parquet for Phase 3's
reuse. The detector script itself names neither `ICPUSDT` nor `MATICUSDT` anywhere; the coverage check
against those two specific events runs in Phase 3, which is not subject to that grep.

## Phase 2 — `power_lib21.py` and its tests

Four small, independently testable pieces, 9 tests, no network: `flag_frozen_feed_bars` (the detector
itself, reused by both Phase 1 and Phase 3 so the two never drift), `excluded_book_bars` (the `{T-1,T}`
alignment rule, pinned by a hand-built 5-bar synthetic frame with symbols held/not-held at chosen bars
worked out by hand — the one place a silent off-by-one would produce a plausible-looking wrong answer),
and the three closed-form power one-liners (`bootstrap_se_from_ci`, `mde`, `n_required`), each checked
against a hand-derived normal-case value, plus `placebo_mean_diffs`, checked for shape and seed
reproducibility.

## Phase 3 — Diff CIs, MDE, placebo — the results themselves

`run_phase_3_21_results.py` refuses to start if `phase_1_21_catalogue.json` is missing (the one guard
kept from 020's runner architecture — pre-registration ordering discipline, not performance
machinery). Rebuilds `A0`'s weights (`n_min=1`, `bl20.THETA_IN`/`THETA_OUT` on the `hl=21` panel) and
`A3`'s weights (`n_min=bl20.N_MIN`, `THETA_IN_SLOW`/`THETA_OUT_SLOW` on the `hl=42` panel), copied
verbatim in shape from `run_phase_4_20_book.py:144-186` — weights only, no book rebuild. Intersects the
catalogue with `A0`'s non-zero-weight bars to get the exclusion set (22 book-return bars out of 3,840,
0.57%), computes both diff CIs and their bootstrap p-values, both MDE/`n_required`/`years_required`
pairs, and the 200-draw placebo. As a diagnostic only (never used for any exclusion), it also computes
how many of the same flagged bars `A3`'s own, more diversified holdings would have been exposed to: 18
of 22 — `A3`'s floor reduces but does not eliminate exposure to the same events, consistent with a
floor that stands down entirely below `N_MIN` symbols rather than filtering by feed quality directly.

## Bugs found

One near-miss, caught before it shipped: **a self-referential false positive in the sec 6 holdout-
literal grep.** The pre-registration's own `holdout_policy` note originally quoted the grep pattern
(`"basis18/holdout\|bybit20/holdout\|HOLDOUT_START"`) verbatim to document what Phase 4's check would
run — which meant that once `scripts/run_021.sh` existed, `grep -rn "basis18/holdout\|bybit20/holdout\
|HOLDOUT_START" src/research/tmp/*21*.py scripts/run_021.sh` would match the pre-registration file
itself and produce a false "holdout literal present" alarm on a file that has no holdout access logic
at all. Caught by running the grep as part of finishing Phase 3, before Phase 4 existed to trigger it.
Fixed by describing the check in `phase_0_21_preregistration.json` without spelling out the literal
substrings; re-verified the grep returns nothing once every 021 file existed.

No bug affected any reported number: the reproduction tripwire matched 020's stored RC-3 CI to 1e-12 on
the first run, and Phase 1's catalogue independently rediscovered both of 018's documented events
without any hand-tuning.

## Bottom line

**Data quality is not what is holding RC-3 back — statistical power is, and the notebook now has an
exact number for it.** The mechanical detector is sound (PW-1) and, taken at face value, excluding the
handful of bars it flags in `A0`'s own holdings does move the diff CI past zero (PW-2). But the
pre-registered placebo control — the single most informative thing this notebook adds over an ad-hoc
version of the same probe — shows that effect is not specific to those bars: removing *any* same-size
random subset of the 3,840-bar series produces a comparably large mean shift, because a diff series
this short and this noisy is sensitive to which handful of its own tail bars happen to be in or out.
PW-4 fails on the placebo leg alone (the 5% cap was never close to binding), so PW-2's apparent fire is
declared non-load-bearing per the pre-registered rule, and the notebook lands on branch (b): the
paired comparison would need roughly 27.6 years of history to detect an effect this size at 80% power,
against the 3.50 years actually available. That is a complete, quantitative, publishable answer — not
a failure, and not a licence to go looking for a second exclusion rule. 020's own frozen RC-3 verdict
stands exactly as recorded; no holdout access is granted under any outcome, per 021's own pre-
registered policy (never contingent on this notebook's own result, since a single corrected diagnostic
was never the same conjunction 020's own holdout gate required).

Machinery: `src/research/tmp/power_lib21.py` (imports `research`, `basis_lib18`, `basis_lib20`, edits
none of them — detector, alignment rule, closed-form power helpers, placebo; 9 tests),
`src/research/tmp/run_phase_{0,1,3}_21_*.py` (pre-registration+tripwire, mechanical catalogue,
results), `tests/test_power_lib21.py`, `scripts/run_021.sh` (five phases, sequential, foreground,
idempotent by output-file existence — no background runner, no heartbeat, no `status.json`, no
`xargs -P`; this notebook's entire measured compute budget is under one minute). `src/research.py`,
`basis_lib18.py`, `basis_lib20.py`, and every `*_17_*`/`*_18_*`/`*_19_*`/`*_20_*` file were imported
only, never edited. `A0`'s and `A3`'s frozen construction was never touched. `src/risk/` and
`src/regime/` were never touched. No bar was ever excluded from either book's own backtest, Sharpe,
DSR, or universe — only from the paired comparison series. Mechanism B was out of scope throughout.

## What to test next

- **A fresh, full-conjunction notebook was the pre-registered next step if PW-2 had fired cleanly
  (sec 6) — but it didn't, so that path is closed for now.** The data-quality route is exhausted for
  this exclusion rule; per this notebook's own scope discipline, no second or third exclusion rule
  should be hunted for. If a future notebook wants to revisit the data-quality angle, it needs a
  materially different, independently-motivated signal (not a variant of the frozen-feed flag), not a
  retry of this one with different parameters.
- **The 27.6-years-required number is the sharpest actionable fact this notebook produced.** RC-3 will
  not clear on Binance dev-window history alone, ever, without either (a) a genuinely lower-variance
  diff estimator (e.g. a paired/coupled bootstrap conditioning on the shared market-regime component
  both books are exposed to — 020's own "what to test next" flagged this same idea), or (b) the
  eventual 2025-07-01+ holdout window, once some future gate legitimately earns access to it under its
  *own* notebook's conjunction, not this one's.
- **020's two deferred items carry forward untouched** (on-chain/exchange-netflow data survey, and
  market-making/inventory-managed liquidity provision) — nothing in 021 bears on either; see
  `NEXT_PROMPT.md`.
- **020's `B2` Mechanism-B candidate (joint floor + slow-carry cross-venue construction, mirroring
  what worked for Mechanism A) was not reached** — 021 was scoped to Mechanism A's RC-3 gap only, per
  its own pre-registration (sec 2: "Touch Mechanism B — Does not"). Still open, deferred again.
