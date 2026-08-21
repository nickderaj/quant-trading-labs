# Notebook 020 — Refined Basis Construction + Cross-Venue Funding Dispersion: Results Summary

## What

This notebook tests two independent, pre-registered fixes to 018's funding basis trade — the closest
thing to real alpha this repo has found in thirty-one gates, but one that failed the tradeability bar
(FA-2) and the "does timing add value" bar (FA-3). **Mechanism A** refines 018's own single-venue
construction: a diversification floor (`N_MIN=3`, stand down entirely rather than concentrate below
it) and a slower, lower-turnover carry (half-life 42 vs. 21, thresholds re-derived from a 30-day
rather than 15-day target hold). **Mechanism B** is a structurally different trade: the *spread*
between Binance's and Bybit's funding rates on the same underlying — short the expensive-funding
venue's perp, long the cheap one's, no spot leg needed, cheaper (25bp round-turn vs. 34bp) because two
perp legs hedge each other instead of a spot+perp pair.

## Why

018's own concentration diagnostic found *why* its sample moments were so extreme (skew −11.5,
kurtosis 816.9, capping any dispersion-based DSR repair below 0.95 per 018's own addendum): the book
is equal-weight among however many symbols currently qualify, with a cap of 10 above but **no floor
below**. It holds a single symbol on 5.4% of bars, and the five worst single-bar net returns are
exactly those bars, each coinciding with a verified real perp-market liquidity collapse the 30-day
trailing-*median* liquidity screen is too slow to catch. 018 correctly refused to patch that after the
fact (its own sec 12 forbids exactly that). This notebook is the pre-registered place to test the fix,
paired with a second, independent mechanism (cross-venue dispersion) that a repo-wide Bybit fetcher now
makes cheap to test for the first time.

Two hypotheses, written down before any backtest ran:

> **H-A**: A diversification floor removes the 1- and 2-symbol bars that produce 018's extreme skew
> and kurtosis. Better moments raise the DSR mechanically, and fewer catastrophic single-bar losses
> tighten the bootstrap CI. Both of FA-2's failing legs are therefore addressable by construction, not
> by estimator repair.
>
> **H-B**: A funding *spread* between venues is a genuinely different, more arbitrage-like return
> driver than a funding *level*, with lower directional and basis exposure — and therefore possibly a
> better-behaved return distribution — at the cost of a shorter usable history and a smaller universe.

## How

Pre-registered every constant, gate, and the itemised 32-trial `n_trials` budget
(`phase_0_20_preregistration.json`) before a single byte of Bybit data was fetched. Probed Bybit live
first (interval=480 is silently rejected by its kline endpoint — worked around with interval=240 and
2→1 aggregation, disclosed below), then fetched 8h-equivalent perp klines and funding history for the
93-symbol Binance∩Bybit universe, single-threaded and rate-limited. Built `basis_lib20.py` (imports
`basis_lib18`, never edits it) with twelve required tests, proved it reduces exactly to 018 at
`n_min=1` — both synthetically and on the real panel — and reproduced 018's stored net Sharpe
(0.5766182328943011) to 1e-6 before proceeding. Probed both mechanisms without costs, **wrote down a
falsifiable DSR prediction before running the backtest** (019's device, reused), then ran 14 books and
13 ablations, scoring RC-1…RC-4, XD-1…XD-4, and FUND at `n_trials=32`. Invoked the holdout gate once,
per its own "demonstrate the fence" rule, regardless of outcome.

## Results

**Mechanism A: the construction refinement works exactly as hypothesized, but doesn't clear its own
paired-comparison bar.** The diversification floor alone (A1) collapses the single-symbol-bar
fraction from 018's 5.4% to exactly 0.0%, and improves gross skew/kurtosis from −11.65/849 to
+0.53/18.7; combining it with the slow carry (A3, the headline) improves them further (+0.34/16.1).
The pre-registered prediction was explicit and testable: at 018's own Sharpe (0.5766) with A3's
predicted moments, the counterfactual DSR was only 0.154 — better moments alone would not be enough.
A3's *actual* net Sharpe came in at 3.89, driven almost entirely by a 4x reduction in standard
deviation (018's gross mean return is essentially unchanged: 0.434 vs. 0.435) rather than a change in
mean — and that absolute improvement is what the prediction said would be needed. **RC-2 fires**
(DSR=0.9999997) and **RC-4 fires** (neutral) and **FUND-A fires**, but **RC-3 does not**: the *paired*
per-bar difference against 018's own frozen reproduction still has a 95% bootstrap CI that includes
zero, because the diff series inherits much of 018's own volatility bar-by-bar even though A3's own
absolute risk profile is far better. Per sec 9's strict conjunction (RC-2 AND RC-3), **the holdout
stays locked for Mechanism A.**

**Mechanism B: H-B is not supported.** The raw, undirected pooled funding-spread statistic (XD-1, a
fixed short-Binance/long-Bybit orientation across the whole panel) is significantly *negative*
(mean −2.21e-05/period, t=−6.73) — pooled, Bybit funding runs higher than Binance's, not the other way
round. The strategy's own direction-following return (following the sign of the spread symbol-by-
symbol, bar-by-bar) is separately significant and positive (mean 7.15e-05, t=10.8), so the mechanism
is real in that narrower sense — but the headline book (B0) is weak (net Sharpe 0.354, DSR 0.082) and,
decisively, **loses outright to `B_single`** (the plain 018-style single-venue trade restricted to the
same 93-symbol Bybit-intersected universe and window), which alone scores net Sharpe 0.854. XD-3
("the spread beats the level") therefore fails in the *wrong direction* — the level wins. XD-4
(neutrality) fires. No gate combination unlocks Mechanism B's holdout access.

Per sec 9 point 7, `run_phase_6_20_holdout.py` was invoked once anyway (neither mechanism qualified)
and refused correctly: exit code 1, no path into either holdout directory constructed, verified by
grep.

## Gate verdicts — the full table

| gate | claim | fires? | number behind it |
|---|---|:---:|---|
| **RC-1** (mechanism preserved) | Carry mechanism survives the construction change | **YES** | pooled mean gross paired return 1.13e-04/period, Newey-West \|t\|=11.71 (>3) |
| **RC-2** (tradeable) | Restates FA-2, all three legs | **YES** | net Sharpe 3.887–3.889 at every offset (clears >0.5) **and** 95% bootstrap CI [2.06e-05, 6.98e-05] excludes zero **and** DSR=0.9999997 (>0.95) |
| **RC-3** (refinement adds value) | Restates FA-3's shape, vs. 018's own reproduction | **No** | (refined − 018-reproduction) 95% CI is [-2.20e-05, +5.62e-05] — point estimate favours refined, CI includes zero |
| **RC-4** (genuinely neutral) | Not a disguised long | **YES** | \|beta\| to crypto basket 0.0022, to BTC 0.0020, both < 0.10, at every offset |
| **FUND-A** (fundable, Mechanism A) | Net Sharpe > 0.5 AND DSR > 0.95 | **YES** | Sharpe 3.89, DSR 0.9999997 |
| **XD-1** (mechanism, raw/undirected) | Pooled cross-venue spread > 0 net of basis drift | **No** | pooled mean −2.21e-05/period, Newey-West \|t\|=−6.73 — *significant but negative* |
| **XD-2** (tradeable) | Restates FA-2, all three legs, cross-venue book | **No** | net Sharpe 0.354 (fails >0.5) **and** 95% CI [-6.17e-06, +1.67e-05] includes zero **and** DSR=0.082 |
| **XD-3** (spread beats level) | Cross-venue book beats single-venue on B's own universe | **No** | (xvenue − single-venue) 95% CI is [-6.75e-05, +8.36e-06] — point estimate *favours the single-venue book* |
| **XD-4** (genuinely neutral) | Not a disguised directional bet | **YES** | \|beta\| to crypto basket 0.0007, to BTC 0.0009, both < 0.10 |
| **FUND-B** (fundable, Mechanism B) | Net Sharpe > 0.5 AND DSR > 0.95 | **No** | Sharpe 0.354 < 0.5 |
| **Holdout access, Mechanism A** | requires RC-2 AND RC-3 | **Not granted** | RC-3 did not fire |
| **Holdout access, Mechanism B** | requires XD-2 AND XD-3 | **Not granted** | neither fired |

---

## Phase 0 — Pre-registration

`phase_0_20_preregistration.json`, committed before Phase 2 ran, not edited since. Freezes every
constant inherited unchanged from 018, `N_MIN=3` (derived from 018's own concentration diagnostic —
the smallest integer eliminating both the 1- and 2-symbol tails), the slow-carry constants
(`SLOW_CARRY_HALF_LIFE=42`, `TARGET_HOLD_SLOW=90`, both an exact 2x scaling of 018's own ratio —
`THETA_IN_SLOW` equals 018's `THETA_OUT` to the last digit, an expected coincidence, not a bug), the
Mechanism B constants (`ROUND_TURN_BP_XV=25.0`, cheaper than 018's 34bp since two perp legs hedge each
other with no spot leg), the eight gates plus FUND, and the itemised 32-trial `n_trials` budget:
9 (Mechanism A Phase 4) + 7 (Mechanism A Phase 5) + 8 (Mechanism B Phase 4) + 6 (Mechanism B Phase 5) +
2 (holdout) = 32.

## Phase 1 — Data

**Mechanism A** needed no new data: `src/research/cache/basis18/dev/` (018's own 151MB cache) reused
unchanged through `basis_lib18.load_basis_panel`.

**Mechanism B** needed a new Bybit fetcher. Live probe first (sec 4.2): `interval=480` (8h) is
**silently rejected** by Bybit's kline endpoint — `retCode=0`, an empty `result.list`, no error —
worked around by fetching native `interval=240` (4h) bars and aggregating 2→1 into Binance-aligned 8h
buckets (open=first, high=max, low=min, close=last, volume/turnover=sum). Bybit's `turnover` field is
quote-denominated volume directly, used for its side of the liquidity screen — genuinely better than
Binance's `close*volume` approximation (018 had to approximate because `data.download_and_unzip_klines`
drops Binance's native `quote_volume`); **this asymmetry is disclosed, not "fixed" on either side.**
`api.bybit.com` answered every probe directly; no geo-block, no `api.bytick.com` fallback needed.

Universe: intersecting 018's 126 spot-usable symbols with Bybit's 725 USDT linear perps gave **93
symbols** (slightly above the sec 4.3 "roughly 60–90" expectation, recorded honestly rather than
trimmed). Funding-interval distribution among the intersection: 85 symbols at 480min (matching
Binance's native cadence), 8 at 240min — both divide the 480-minute Binance bucket evenly, so no
symbol needed exclusion on that basis (sec 4.4's non-divisor exclusion rule was never triggered here).
Bybit's published taker fee (0.055%, the pre-registered `BYBIT_TAKER_BP`) could not be independently
confirmed via the public market-data API (no endpoint exposes it) — disclosed as unconfirmed-by-probe,
not silently assumed-confirmed.

Fetched single-threaded, ≤5 req/s, 3 retries with exponential backoff — **all 93/93 symbols completed
in both dev and holdout windows with zero truncation and zero errors**, well inside the 90-minute time
box (finished in roughly 20 minutes). The funding sum-into-8h-bucket resampling (sec 4.4) and the
4h→8h kline aggregation are deliberately NOT done at fetch time — they live in `basis_lib20.py` as
testable, pinned functions (`test_bybit_funding_resample_to_8h`), so the fetcher's only job is a
faithful, resumable copy of Bybit's raw response.

## Phase 2 — Library (`basis_lib20.py`) and the reproduction tripwire

Imports `basis_lib18` and `research`, never edits either. `add_trade_features_v2` and
`build_book_weights_v2` parameterise 018's own functions by carry half-life and add the
diversification floor as a strict no-op at `n_min<=1` — proved by `test_n_min_1_reproduces_018_weights`
(synthetic) and, more importantly, by Phase 2b's own real-panel identity check (element-wise
identical, not just close). `xvenue_paired_log_return`, `xvenue_carry_estimate`,
`build_xvenue_book_weights`, and `apply_xvenue_costs` implement Mechanism B: the book's `weight`
column is **signed** (positive = short Binance/long Bybit), so a sign flip while held is charged
exactly like a close+reopen through the ordinary `|Δweight|` turnover accounting, with no special-cased
branch — pinned by `test_xvenue_sign_flip`. Twelve tests, all passing.

**The reproduction tripwire (sec 5.6): 018's own `build_book_weights` on the cached panel reproduced
net Sharpe 0.5766182328943011 to the full stored precision (abs diff 0.0).** Panel-build timing:
2812–3283 bars/sec, comfortably fast enough that the sec 3 rule-7 wall-time refusal never came close
to triggering (predicted remaining wall time for the entire Phase 4 grid: ~10 seconds).

## Phase 3 — Mechanism probes (no cost model, no Sharpe, no strategy verdict except RC-1/XD-1)

**Mechanism A**: A0 (018-baseline reproduction, hl=21, n_min=1) reproduces 018's 5.4% single-symbol
figure closely (5.44%, within 0.4pp) — a free tripwire that passed. A1 (floor only) drops that to
0.00% and improves gross skew/kurtosis from −11.65/849 to +0.53/18.7. A2 (slow carry only) improves
moments modestly (−10.66/529) — most of the improvement comes from the floor, not the slower carry.
A3 (both) is best: +0.34/16.1. **Gate RC-1 fires**: pooled mean gross paired return 1.13e-04/period,
Newey-West \|t\|=11.71.

**Mechanism B**: the raw undirected pooled statistic (XD-1) is significant but *negative*
(−2.21e-05/period, t=−6.73) — this is a genuine, disclosed finding, not a bug (confirmed by checking
the decomposition below). The funding-spread term dominates the price-divergence term both by
magnitude (2.20e-05 vs. 1.00e-07) and by significance (t=−15.53 vs. 0.05), exactly as 018's own single-
venue decomposition found for funding vs. basis-change — funding is the drift, price divergence is
noise around it, on both mechanisms. 551 bars were excluded from the decomposition's descriptive
statistics for `|xvenue basis| > 20%` (018's own sanity bound, reused verbatim) — two symbols
(WAVESUSDT, ICPUSDT), consistent with 018's own finding that a handful of feed-artifact bars can
dominate a pooled statistic without being real tradeable relationships; never excluded from the
backtest universe itself. Spread persistence: 32% of `|carry|>θ_in_xv` runs clear the 25-period
break-even (018's single-venue figure was 44% against a break-even of 34) — plausible, somewhat
weaker persistence, consistent with the headline result. The sign convention was asserted against a
real bar (largest \|spread\| in the panel) and matches the formula exactly.

## Phase 3b — The written-down prediction

Made before any Phase 4 cell ran (`scripts/run_020_books.sh` structurally refuses to start without
this file on disk). Using only Phase 3's no-cost numbers:

| | predicted skew | predicted kurtosis | Sharpe used | counterfactual DSR | clears 0.95? |
|---|---|---|---|---|---|
| A3 | +0.339 | 16.08 | 0.5766 (018's own) | **0.154** | No |
| B0 | +0.168 | 51.20 | 0.854 (`B_single`'s) | **0.346** | No |

**Prediction: RC-2's DSR leg would not fire unless A3's actual Sharpe substantially exceeded 018's;
XD-2's DSR leg would not fire unless B0's actual Sharpe substantially exceeded `B_single`'s.** Both
predictions were falsifiable and both turned out to matter directly: A3's actual Sharpe (3.89) did
substantially exceed 018's, and RC-2 fired; B0's actual Sharpe (0.354) did *not* exceed `B_single`'s
(0.854) — it came in lower — and XD-2 failed. **The prediction machinery worked exactly as intended
and its qualitative call was right in both directions.**

## Phase 4 — The books

14 book builds (17 itemised `n_trials` cells; origin offsets reuse one book build each, trimmed
post-hoc, per 018's own non-refit convention) ran in ~10 seconds total (predicted and actual, sec 3
rule 7's wall-time gate never came close to triggering).

| variant | net Sharpe | net MDD | ann. turnover | note |
|---|---:|---:|---:|---|
| A0 (018 repro) | 0.577 | −8.6% | 56.9/yr | matches 018 exactly |
| A1 (floor only) | 2.757 | — | — | |
| A2 (slow carry only) | 1.352 | — | — | |
| **A3 (headline)** | **3.887–3.889** | **−3.4%** | **44.2/yr** | offsets agree to 3 decimals |
| A_alwayson | −0.415 | — | — | timing still matters within the refined universe |
| B0 (headline) | 0.354 | −3.5% | 58.1/yr | below the 0.5 bar |
| B1 (slow) | 0.961 | −2.4% | 47.6/yr | notably stronger than the headline — not itself pre-registered as the headline, disclosed here, not substituted for it |
| `B_single` | 0.854 | −8.3% | 53.4/yr | beats B0 outright |
| B_alwayson | −1.869 | — | — | |

**Concentration diagnostic, A3 vs. 018**: median symbols held is unchanged (10, at the cap), but the
worst single-bar net loss drops from 018's −5.7% (ICPUSDT alone) to −0.38% (a 10-symbol bar including
ICPUSDT — the floor kept it diversified through exactly the kind of event that broke 018). The book
sits in cash 5.2% of bars (the stand-down cost of the floor, cheap here). **Concentration diagnostic,
B0**: median 5 symbols held (well below the 10 cap — this book rarely fills up), cash 19.1% of bars —
substantially more stand-down than Mechanism A, consistent with a thinner, more marginal opportunity
set.

Beta: A3 to basket 0.0022, to BTC 0.0020 (both offsets 0–3, all inside ±0.10). B0 to basket 0.0007, to
BTC 0.0009. Both mechanisms are genuinely delta-neutral.

## Phase 5 — Ablations (13, offset 0 only, 018's convention)

**Mechanism A (A3 base 3.887)**: `N_MIN∈{2,5}` barely moves it (3.884, 3.920) — the headline stays at
`N_MIN=3` regardless, per sec 8's own rule, this is reported as robustness, not as grounds to retune.
Excluding LUNA/FTT barely moves it either (3.985) — the floor already prevents dependence on any one
name, a direct, disclosed contrast with 018 (whose worst bars were driven by exactly these kinds of
single-symbol events). No-hysteresis costs about half the Sharpe (1.748). Cost sensitivity never
crosses the 0.5 bar within the tested 0–51bp range (still 0.812 at 51bp, 1.5x the real 34bp cost) — no
computable break-even inside the tested range; the true break-even is somewhere beyond 51bp, reported
as a bound, not extrapolated.

**Mechanism B (B0 base 0.354)**: no-hysteresis is deeply negative (−3.679, a much larger relative
collapse than Mechanism A's equivalent ablation — sign-flipping without a band is expensive).
Excluding B0's own top-2 contributing symbols (TRBUSDT, DOTUSDT) flips it negative (−0.278) — a real
concentration finding, this book leans on a small number of names more than its median-5-held
diagnostic alone suggests. The one-venue-leg-only neutrality control is negative and structurally
unlike B0's own return (−0.589 vs. +0.354) — evidence against a disguised single-leg directional bet.
Interpolated break-even: 24.95bp, essentially identical to the real 25bp cost — **this single number
explains the weak headline Sharpe directly**: B0 operates almost exactly at its own break-even.

## Phase 6 — Holdout: not spent

Neither mechanism's conjunction fired (Mechanism A: RC-2 fired, RC-3 did not; Mechanism B: neither
XD-2 nor XD-3 fired). Per sec 9 point 7, `run_phase_6_20_holdout.py` was invoked anyway, to
demonstrate the fence rather than merely assert it: it read `phase_4_20_results.json`'s
`holdout_access` block, found both mechanisms `False`, and exited 1 without constructing any path into
`basis18/holdout` or `bybit20/holdout`. Verified live:

```
$ grep -rn "basis18/holdout\|bybit20/holdout\|HOLDOUT_START" src/research/tmp/*20*.py
```

shows the two literal holdout directory strings appearing in exactly one file,
`run_phase_6_20_holdout.py` — nowhere else in the notebook's codebase.

## Bugs found

One real bug, caught by suspicious output before being trusted (this repo's own standing tripwire
discipline):

1. **A stray unused holdout-directory literal in `basis_lib20.py`** — while writing the library,
   `BYBIT_HOLDOUT_CACHE_DIR = "src/research/cache/bybit20/holdout"` was declared alongside
   `BYBIT_DEV_CACHE_DIR` for symmetry, but never referenced anywhere in the module. It didn't cause a
   wrong number (it was dead code), but it violated sec 9 point 6's mechanical-fencing invariant that
   `run_phase_6_20_holdout.py` is the *only* file permitted to name a holdout directory literal.
   Caught by running the sec 9 point 6 grep as part of finishing Phase 6, before it was ever exercised
   — removed.

One near-miss, not a bug: RC-3 failing to fire despite A3's dramatically higher absolute Sharpe (3.89
vs. 018's 0.577) looked, on first read, like a possible computation error in the paired bootstrap.
Investigated by comparing CI widths: A0's own CI width (≈8.3e-05) and the diff series' CI width
(≈7.8e-05) are close, consistent with the diff series inheriting most of A0's own volatility bar-by-
bar — a real statistical property of a paired comparison against a noisy baseline, not a bug in the
comparison itself.

## Bottom line

**Mechanism A confirms H-A at every level the notebook could test it, and is, on any absolute
reading, a dramatically better book than 018's** — RC-1, RC-2, RC-4, and FUND-A all fire, DSR clears
0.95 by six nines, and the worst single-bar loss drops by more than an order of magnitude (−5.7% to
−0.38%). But this notebook's own pre-registered bar for "the refinement adds value" is a stricter,
paired-comparison question — does the difference from 018's own book clear a bootstrap CI — and that
one does not fire. **The holdout stays exactly as unspent as 018 left it, and per sec 9's own rule,
that is the correct, honest outcome of a conjunction gate, not a partial credit situation.**

**Mechanism B does not support its own hypothesis.** The cross-venue funding-spread mechanism is real
in a narrow, direction-following sense (a significant, positive, sign-following return exists), but
the headline book is weak, sits almost exactly at its own break-even cost, and loses outright to the
much simpler single-venue trade restricted to the identical universe. This is a clean, informative
negative result, not a data-quality artifact — 93/93 symbols fetched cleanly, the reproduction
tripwire passed exactly, and every neutrality/robustness check behaves as expected.

**The central methodological result of this notebook is that the pre-registered DSR prediction worked
correctly, in both directions.** Phase 3b's counterfactual DSR computation, made before Phase 4 ran,
correctly anticipated that neither mechanism's headline book would clear the DSR bar on moments
improvement alone — and then correctly anticipated exactly what *would* need to happen for each (a
Sharpe well above the counterfactual baseline for A3; a Sharpe above `B_single`'s for B0) — and both of
those conditional predictions resolved correctly against Phase 4's actual numbers. This notebook
therefore closes cleanly on both mechanisms without needing the DSR estimator question at all: 018's
own hinge — "the DSR leg was never contingent on the estimator, it was contingent on the return
distribution" — is now validated twice, once by fixing the distribution and clearing DSR (Mechanism A)
and once by fixing the distribution, not clearing the higher paired bar anyway (Mechanism A's RC-3),
and once by a genuinely different mechanism simply not being strong enough regardless of estimator
questions (Mechanism B).

Machinery: `src/research/tmp/basis_lib20.py` (imports `basis_lib18`, never edits it — carry/floor/
xvenue primitives, twelve tests), `src/research/tmp/run_phase_{0,1,2,3,4,5,6}_20_*.py` (pre-
registration, Bybit fetch, panels+tripwire, mechanism probes, prediction, books, ablations, gated
holdout runner), `tests/test_basis_lib20.py` (twelve required tests), `scripts/fetch_bybit_data.sh` /
`run_020_books.sh` / `run_020_ablations.sh` (background-safe runners with `scratch/020/status.json`
heartbeats). `src/research.py`, `basis_lib18.py`, and every `*_17_*`/`*_18_*`/`*_19_*` file were
imported only, never edited. `src/risk/` and `src/regime/` were never touched. 018's own frozen book
never received holdout access from this notebook, and 018's numbers, verdicts, and text stand exactly
as recorded. The 2025-07-01+ crypto holdout remains exactly as unspent as 018 left it, now with a
second, independently-fenced Bybit cache alongside the Binance one for whichever future notebook next
has a fired dev gate.

## What to test next

- **Mechanism A's RC-3 gap is the sharpest, most directly actionable open question this notebook
  found.** A3 is unambiguously a better absolute book than 018's; the open question is purely
  statistical power on the paired comparison. A longer paired series (the holdout window, once *some*
  future gate legitimately earns it) or a variance-reduction technique on the diff series itself (e.g.
  a paired/coupled bootstrap that conditions on the shared market-regime component both books are
  exposed to) could sharpen this without touching the frozen construction.
- **B1 (the slow-carry-only cross-venue variant) scored net Sharpe 0.961, nearly 3x the pre-registered
  headline B0's 0.354**, and was not itself gated as a headline. That is exactly the kind of
  after-the-fact-visible number this repo's own discipline forbids adopting retroactively (sec 8: "the
  headline stays at the pre-registered value regardless of what the sensitivity shows") — but it is a
  legitimate, pre-registerable candidate headline for a future notebook that starts fresh with B1 (or
  a joint floor+slow-carry Mechanism B variant, mirroring what worked for Mechanism A) as its own
  pre-declared headline, not a retroactive substitution here.
- **A joint Mechanism-B floor+slow-carry construction**, mirroring exactly what worked for Mechanism A
  (H-A generalized), is a natural next pre-registered test — B0 alone already shows the slow-carry
  variant (B1) outperforming, and Mechanism A's own result shows floor+slow-carry compounding rather
  than just adding.
- **The B0-vs-B_single result deserves a mechanism-level explanation**, not just a score: why does
  restricting to the intersected universe make the *plain* single-venue trade (0.854) so much stronger
  than 018's full-universe number (0.577)? A possible survivorship/liquidity story (Bybit-listed names
  skew toward larger, more liquid symbols) is plausible but untested here — worth its own descriptive
  pass before building more Mechanism B machinery on top of it.
