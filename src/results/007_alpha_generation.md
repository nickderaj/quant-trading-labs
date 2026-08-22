# 007 — Is Transaction Cost the Thing Blocking Every Edge?

## The hypothesis

**Transaction cost, not signal absence, is what has blocked every alpha attempt in this programme
so far.**

Notebook 003 said it outright: the signal is real pre-cost, costs erase it. Notebook 006 said it
again in different words — its risk overlay's signal was genuinely better calibrated and *still*
lost to buy-and-hold on Sharpe once turnover was charged.

If that's true, the cheapest and most direct fix is obvious: trade the same profitable signal less
often. This notebook tests that directly, then asks whether a structurally different kind of signal
survives where price-based ones haven't.

Four things were pre-declared as the tests that would count:

1. **Turnover reduction** — can cutting how often a known gross-profitable signal trades produce a
   statistically distinguishable net edge?
2. **Risk gating** — does standing down during predicted high-tail-risk periods improve
   risk-adjusted return?
3. **Carry** — does the funding rate, a payment rather than a price prediction, survive costs?
4. **Tail-shape factors** — do the fitted tail parameters from notebooks 005–006 work as
   cross-sectional signals?

A fifth, spending the frozen holdout period, would only run if one of the four fired.

Universe: notebook 003's frozen 30-symbol panel, unchanged.

## First: does notebook 003 reproduce?

Before building on those numbers, the headline net and gross Sharpe (+0.42 / +1.32), the
origin-shift flip to −2.45, the realised annual fee drag (0.335–0.375%), the two negative
configurations at every offset, and the holdout result (−0.47 net / +0.74 gross) were all
re-derived from the committed outputs. **All reproduced to within 0.005.**

---

## Test 1 — Cut turnover on a signal known to be gross-profitable

The signal's own out-of-sample predictions were generated **once per origin offset**, with a fixed
seed, and never refitted per intervention.

That matters more than it sounds. Notebook 003's model was unseeded, and its own reproducibility
check found that re-running the identical code flipped the headline Sharpe from +0.42 to −1.22.
Refitting per turnover variant would have confounded "did the trading mechanics help" with "did
this particular refit get lucky". One frozen signal per offset; four position-construction variants
tested against it.

The three interventions, plus one combination declared in advance:

- **No-trade bands** — only move a position when the target differs from the current one by more
  than a threshold. Bands of 0.05, 0.10, 0.15, 0.20 were tested, with 0.0 verified to reproduce the
  unmodified weights exactly. That check is the correctness foundation every other number here
  depends on.
- **Weight quantisation** — round each weight to the nearest 0.05 of gross exposure.
- **Rebalance throttling** — recompute positions only every k-th bar and hold flat between, for
  k = 2, 3 and 6.
- **Combined** — the mid-points of all three (band 0.10, grid 0.05, k = 3), tested once, before any
  individual result was seen.

### Results

Baseline, no intervention:

| Origin offset | Net Sharpe | Turnover/year |
|---|---|---|
| 0 | −0.180 | 492.5 |
| 7 | +0.134 | 486.1 |
| 14 | −0.040 | 488.5 |
| 21 | −0.343 | 487.0 |

(This doesn't exactly match notebook 003's logged numbers, which is expected rather than a bug —
that model was unseeded, so a fresh seeded run of identical code is methodologically identical but
not bit-exact.)

| Variant | Turnover reduction | Net Sharpe range across offsets |
|---|---|---|
| No-trade band (best, 0.20) | 22% | −0.31 to +0.08 |
| Quantisation (grid 0.05) | ~0% | −0.29 to +0.26 |
| Throttle, k = 2 | 37% | −0.04 to +0.47 |
| Throttle, k = 3 | 51% | +0.49 to +0.89 |
| **Throttle, k = 6** | **71%** | **+0.27 to +1.05** |
| Combined | 55% | +0.37 to +0.90 |

**Rebalance throttling is the effective lever.** At k = 6 turnover falls 71% a year and net Sharpe
goes consistently positive at every offset (+1.011, +1.053, +0.272, +0.587).

The other two barely move. No-trade bands cut turnover only 10–23% across the whole band grid,
short of the 30% bar set for counting as a genuine intervention. Quantisation changes turnover by
essentially nothing, because the book's per-symbol weights — the top and bottom 6 of 30 symbols at
gross exposure 1.0 — are already coarser than a 0.05 grid.

**And it still fails.** No variant's bootstrap 95% confidence interval on excess return over the
basket excludes zero, at any of the four origin offsets, despite point-estimate net Sharpe reaching
above 1.0 in the best configuration.

This was written down in advance as the expected outcome: turnover would fall substantially, net
Sharpe would improve somewhat, and whether it cleared zero was genuinely unknown. It did not clear
zero.

The k = 6 throttle is carried forward into the next test as the best available baseline — not a
certified one, but the gating test needs *a* book to gate.

---

## Test 2 — Gate the signal on predicted tail risk

The first use of this programme's risk findings as an *alpha input* rather than a risk report. The
throttled book is gated on a causal 1% conditional Value-at-Risk path from GARCH-NIG, the
best-certified density from notebook 006 at this interval, fitted per symbol across all 30 symbols.

Two gating variants, both declared in advance:

- **Stand down** — zero the whole book when the cross-sectional median predicted tail loss exceeds
  its own trailing 250-bar median by a factor k (1.25, 1.5 or 2.0).
- **Per-symbol tilt** — shrink each symbol's size in proportion to its own predicted risk before
  dollar-neutralising.

Every comparison is against the **identical ungated** throttled book on the same signal, never
against the raw baseline — otherwise the previous test's improvement would be miscredited here.

| Offset | Ungated | Stand down k=1.25 | k=1.5 | k=2.0 | Per-symbol tilt |
|---|---|---|---|---|---|
| 0 | +1.011 | +0.919 (−0.092) | +0.928 (−0.083) | +1.007 (−0.004) | +0.924 (−0.087) |
| 7 | +1.053 | +0.829 (−0.224) | +0.904 (−0.149) | +1.070 (+0.017) | +1.006 (−0.047) |
| 14 | +0.272 | +0.020 (−0.252) | +0.003 (−0.269) | +0.256 (−0.016) | +0.326 (+0.054) |
| 21 | +0.587 | +0.678 (+0.091) | +0.639 (+0.052) | +0.583 (−0.004) | +0.589 (+0.002) |

**Gating fails, and mostly hurts.** No variant improves net Sharpe by the pre-declared 0.20 margin
at any offset. Twelve of the sixteen differences are actually negative, and the tighter stand-down
thresholds cost as much as 0.27 of Sharpe.

Drawdown doesn't improve consistently either, so this isn't even the "helps drawdown, hurts Sharpe"
result notebook 006's overlay found. **Standing down during high predicted tail risk costs more in
forgone signal than it saves in avoided drawdown.**

That is a genuinely informative negative result. It directly answers a question posed in advance:
whether standing down during high-tail-risk periods helps depends on whether those periods are
*also* high-expected-return periods. On this book, they at least partially are — which qualifies
the intuitive case for using the risk science as a timing overlay.

---

## Test 3 — Carry (the funding rate) as a primary signal

Structurally different from every signal tested in this programme so far: a payment, not a price
prediction. Tested as a transparent single-feature cross-sectional ranking rather than a fitted
model, deliberately — so a result can't be a fitting artefact and a null can't be blamed on an
under-trained network. Run at 4h, 12h and 1d, all four origin offsets, raw and z-scored.

**A sign correction, decided before any number was seen.** Perpetual funding pays shorts when the
rate is positive and longs when it is negative, so the carry-consistent ranking is the *negative*
of the funding rate — short the payers, long the receivers. A naive ranking on the raw rate would
long the payers, the opposite of the intended trade. This also matches notebook 003's screening
result, where the raw funding rate's correlation with forward return was negative (−0.0095 at 4h).
Both variants use the corrected sign.

| Interval | Signal | Best net Sharpe | Worst | Gross range | Turnover/year |
|---|---|---|---|---|---|
| 4h | Raw | −2.479 | −2.533 | −0.14 to −0.04 | ~965–971 |
| 4h | Z-scored | −3.396 | −3.629 | −0.33 to −0.11 | ~1,260 |
| 12h | Raw | −1.079 | −1.339 | +0.23 to +0.45 | ~674–681 |
| 12h | Z-scored | −2.188 | −2.336 | −0.37 to −0.25 | ~751 |
| 1d | Raw | −0.733 | −0.990 | −0.02 to +0.25 | ~373–377 |
| 1d | Z-scored | −1.636 | −1.809 | −0.73 to −0.56 | ~406–408 |

**All 24 configurations are net-Sharpe-negative.** Applying the throttling intervention from test 1
narrows the losses substantially (4h raw goes from −2.48 to −0.66 at offset 0) but flips the sign in
only one isolated cell out of 48. No confidence interval on excess return excludes zero anywhere.

Funding-rate data coverage came back clean — 30 of 30 symbols, 100% of panel rows — so this isn't
the coverage-limited caveat the test was built to watch for. Instead, a genuinely surprising
finding replaces it:

**At the same 12h interval, the carry book's realised turnover (~674–681/year) is about 40%
*higher* than the price-based signal's (~487–493/year), not lower** — despite funding being a
slow-moving payment. A rank-based top/bottom book still churns when funding rates cluster closely
together across symbols. "Low turnover by construction" turned out to be a claim about the
underlying payment, not about every way of ranking and trading it.

Some gross Sharpes are genuinely positive (up to +0.45). Costs, not signal absence, erase the edge
again.

---

## Test 4 — Tail shape as a cross-sectional factor

The slowest-moving signal available, built from a rolling GARCH fit with Hansen skew-t innovations
per symbol at 4h — the interval where that family most robustly beat GARCH-t in notebook 006. The
information coefficient is computed **first**, before any portfolio is built, by rule.

**Tail quality** (long symbols with symmetric innovations, short skewed ones): correlation not
significant (mean −0.0032, t = −1.43 across 8,766 periods). **Portfolio correctly skipped** —
backtesting it would only produce a spurious Sharpe on what the correlation test already says is
noise.

**Tail premium** (long thin-tailed symbols, short fat-tailed ones): correlation significant but
**negative** (mean −0.0089, t = −3.26) — the opposite sign from the hypothesis. Read literally,
thinner tails predict *lower* forward returns across the cross-section. Yet the top/bottom-quintile
portfolio, built exactly as pre-declared, came back **net-Sharpe-positive at all four offsets**
(+0.402, +0.463, +0.395, +0.348) — which would technically clear the numeric bar.

**Investigated rather than taken at face value.** A significant full-sample correlation and a
profitable extreme-quantile portfolio are not the same test: the correlation uses the whole ranked
cross-section every bar, while the portfolio only trades the extremes (6 of 30 symbols per leg).

Checking leg composition found that **one symbol, FTT, accounts for about 19% of the short leg's
weight** — a symbol with only 6–7 successful fits across its entire history, most of it during the
FTX collapse, whose unstable and thinly identified tail parameter kept landing it in the extreme
leg.

**Excluding that one symbol flips net Sharpe to clearly negative at every offset** (−0.338, −0.342,
−0.450, −0.505).

This is the same "spectacular on one symbol, does not generalise" pattern the programme keeps
finding — but caught here for the first time on a symbol that wasn't already known to be special,
by checking leg composition after the fact rather than assuming a passing number is a real finding.
This single-symbol exclusion check is now run automatically for every factor and offset.

**Direct risk ranking** (long low predicted expected shortfall, short high): correlation significant
(mean −0.0310, t = −9.17) but the portfolio is net-Sharpe-negative at every offset (−1.552 to
−1.678), with or without the problem symbol. A clean, unambiguous null.

---

## The holdout was not touched

The holdout period was gated on at least one of the four tests firing. All four are null, decided
programmatically by reading the four verdicts back out of the already-committed results rather than
re-deriving fresh numbers.

No data past the holdout start date was read anywhere in this notebook. It remains unspent.

---

## Bugs found

Two instances of the same mistake in this notebook's own driver code: a helper indexed into an
already-subsetted test panel using row positions that had been computed against the *full* panel.
An immediate out-of-bounds error, caught by running the script rather than by a test. Fixed by
passing the same full frame the splits were computed on to both call sites. A loud failure rather
than a silent wrong number, but the same class of "index computed against one frame, applied to
another" mistake worth watching for wherever a fold-then-subset pattern appears.

The more consequential catch wasn't a code bug at all: the tail-premium factor would have been
wrongly credited with a positive result if its raw numbers had been trusted without checking leg
composition.

---

## Bottom line

**The hypothesis does not survive its own most direct test.** Test 1 held a known gross-profitable
signal fixed and changed only its trading mechanics — the cheapest, most targeted possible test of
whether the cost problem is solvable at all. Turnover fell 71% a year, net Sharpe improved from
roughly flat-to-negative to consistently positive at every origin offset, and it still did not
clear the pre-declared bar. **Cost reduction is real and mechanically reliable; it was not, on its
own, enough.**

Four tests, four nulls, four different and informative reasons — not four repeats of one finding:

- **Turnover** — the underlying gross edge isn't large enough for even a 71% turnover cut to produce
  a statistically distinguishable net edge.
- **Risk gating** — standing down during predicted high-tail-risk periods mostly *hurt* net Sharpe.
  That's a direct answer to the open question notebook 006's overlay raised, not a repeat of it.
- **Carry** — its own turnover turned out higher, not lower, than the price-based signal it was
  meant to contrast with.
- **Tail factors** — the one that technically cleared the numeric bar was a single-symbol artefact,
  caught by checking leg composition rather than assumed away.

**Six notebooks now have found no tradeable edge in liquid crypto majors that survives its own
pre-declared robustness bar, with costs charged honestly and the bar declared before any number was
seen, every single time.**

This is evidence about the market — a claim about how hard cross-sectional and time-series edges are
to extract from liquid, well-arbitraged crypto majors net of realistic execution costs — not
evidence of a failed research programme. The risk findings from notebooks 004–006 remain exactly as
valid as they were: crypto's tails are fat, real and increasingly well characterised; thin-tailed
models understate 1% expected shortfall almost universally; violations cluster; and the newer
distribution families genuinely beat GARCH-t cross-sectionally at 4h and 12h. Risk modelling and
alpha generation are not the same test, and one failing to produce a tradeable strategy says
nothing against the other's separately certified findings.

## What to test next

- **A joint search over turnover and gating**, rather than the sequential structure used here.
  Gating was only ever applied to a single already-chosen throttle setting; a joint search might
  find an interaction this design couldn't see — though it would also multiply the configurations
  tried and correspondingly deflate any resulting Sharpe.
- **Why gating hurt specifically.** A direct check of whether the signal's gross returns are
  *higher* during high predicted-risk periods would confirm or rule out the risk-premium mechanism
  suggested above, rather than leaving it as an inference from the Sharpe differences.
- **A carry construction with genuinely lower turnover** — a slower rolling window on the funding
  z-score, or a no-trade band tuned to funding's own update frequency (roughly every 8 hours, not
  every bar). The finding that a naive rank-based book churns despite a slow underlying signal
  suggests the construction, not the payment, is the fixable part.
- **A tail-premium factor with an explicit short-history exclusion rule** rather than a post-hoc
  single-symbol check. The catch above would have been structural rather than incidental if symbols
  with too few successful fits were excluded before any number was seen.

*Notebook: `src/research/007_alpha_generation.ipynb`. Terminology — turnover budgeting, no-trade
bands, carry and basis trades, the tail-premium factor — is defined in `docs/`.*
