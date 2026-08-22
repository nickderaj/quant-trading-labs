# 021 — Is the Paired Comparison Blocked by Bad Data, or by Sample Size?

## The narrow question

Notebook 020 built a refined basis book that beats notebook 018's construction on **every absolute
measure** — net Sharpe 3.89 against 0.58, a deflation of 0.9999997, worst single-bar loss down from
−5.7% to −0.38% — and cleared its own tradeability bar cleanly.

But it failed the **paired** comparison against notebook 018's own frozen reproduction. The interval
on the per-bar difference was [−2.20e−05, +5.62e−05]: the point estimate favours the refined book, but
the interval still straddles zero.

This notebook asks one pre-registered question about that single gap.

**The hypothesis:** notebook 018 documented a detection signature for a frozen perpetual feed — open,
high, low and close all identical with zero volume — but only ever applied it to its own descriptive
statistics, never to the paired comparison. If the *baseline* book happened to hold a symbol whose
feed froze, its recorded "return" on those bars is a **mark-to-market illusion, not captured funding.**
Enough of those bars in the difference series could inflate the baseline's apparent performance just
enough to keep the interval straddling zero.

That is mechanically checkable, falsifiable, cheap to test against data already on disk, and has a
genuinely different fix from "the sample is too small".

Four checks were fixed in advance, along with the exclusion rule, the power-formula constants, and a
6-trial budget. **No holdout access is granted under any outcome** — that policy is unconditional and
pre-registered.

## The answer

**Statistical power, not data quality, is the binding constraint — and there is now an exact number
for it.**

| Check | Question | Result |
|---|---|:---:|
| **Is the detector sound?** | Does a mechanical scan rediscover the documented events, without naming them? | **Yes** |
| **Does exclusion move the interval past zero?** | | **Nominally yes** |
| **Is the comparison adequately powered?** | Is the observed effect above the minimum detectable one? | **No** |
| **Is the exclusion surgical rather than a reshape?** | Does it beat a same-size random-exclusion control? | **No — and this is decisive** |

## A tripwire first

Notebook 020's own paired interval was reproduced from its two stored return files and asserted to
match **to 1e−12**. It did, exactly, on all 3,840 bars.

## The mechanical catalogue

All 128 cached perpetual files were scanned with the bare signature — **no halo, no minimum run length,
no widening, zero free parameters.** That was verified sufficient because it covers both of notebook
018's documented events as single contiguous runs without needing any tuning.

**The detector script names neither problem symbol anywhere in its own code**, checked by search rather
than asserted.

It flags **19,157 symbol-bars across 21 symbols** in under two seconds. As anticipated, most of that is
post-delisting dead-feed tail — one symbol at 3,326 bars, two more at about 2,875 each, and a dozen
smaller tails running to the end of the window. **The quantity that actually matters is book-return
bars**, computed separately below.

And it independently rediscovers both documented events: one symbol flagged inside a 10-day window in
June 2022 (31 bars, part of a full 247-bar run), and the other flagged across all 21 bars of its run in
September 2024.

## The results themselves

Only the two books' **weights** were rebuilt — no book was re-run. The difference series is reused
verbatim from notebook 020's stored returns. The catalogue is then intersected with the *baseline's*
actual holdings under the pre-registered alignment rule.

**The exclusion set is 22 book-return bars out of 3,840 — 0.57% of the series.**

**Excluding them moves the interval from [−2.20e−05, +5.62e−05] (point estimate 1.99e−05, p = 0.249) to
[1.02e−05, 3.10e−05] (point estimate 2.03e−05, p = 0.000).** The lower bound clears zero.

**And the placebo control catches it.**

The exclusion easily clears its 5% size cap. But the flagged-exclusion mean of 2.032e−05 sits **just
inside** the 95th percentile of 200 same-size random exclusions, at 2.087e−05.

**Dropping *any* 22 bars from this series tends to move the mean by roughly this much.** The effect is
not specific to the flagged bars — it's generic outlier trimming. Per the pre-registered rule, the
apparent improvement is reported but declared **non-load-bearing.**

### The power calculation — the headline

On the original, unexcluded series, the observed mean difference (1.99e−05) is **well below the
minimum detectable effect at 80% power** (5.59e−05), given the series' own noise.

Detecting an effect this size at 80% power and 5% two-sided significance would require **roughly 30,205
paired 8-hour bars — about 27.6 years of paired history.**

**The paired series has 3.50 years.**

For reference, the *excluded* series alone would need only about 1.9 years, i.e. would already look
adequately powered. **That number is exactly why the placebo control matters here rather than being
decorative** — without it, the exclusion would have looked like a genuine fix.

### A diagnostic, used for nothing

Purely as an observation: of the same 22 flagged bars, the refined book's own more diversified
holdings would have been exposed to **18**. Its floor reduces but does not eliminate exposure to the
same events — consistent with a rule that stands down below a symbol count rather than filtering by
feed quality directly.

## Bugs found

**One near-miss, caught before it shipped: a self-referential false positive in the holdout-literal
check.**

The pre-registration document originally quoted the search pattern verbatim, to document what the
final check would run. Which meant that once the driver script existed, running that search across the
notebook's files would **match the pre-registration document itself** and raise a false alarm on a file
with no holdout logic at all.

Caught by running the check as part of finishing the analysis, before the phase that would trigger it
existed. Fixed by describing the check without spelling out the literal strings, and re-verified clean
once every file existed.

**No bug affected any reported number.** The reproduction tripwire matched to 1e−12 on the first run,
and the catalogue independently rediscovered both documented events without hand-tuning.

Total measured wall time for a clean from-scratch run: **50.7 seconds.**

## Bottom line

**Data quality is not what is holding the paired comparison back. Statistical power is.**

The detector is sound, and taken at face value, excluding the handful of bars it flags in the
baseline's holdings does move the interval past zero.

**But the pre-registered placebo control — the single most informative thing this notebook adds over an
ad-hoc version of the same probe — shows that effect is not specific to those bars.** Removing any
same-size random subset produces a comparably large shift, because a difference series this short and
this noisy is sensitive to which handful of its own tail bars happen to be in or out.

So the apparent improvement is non-load-bearing, and the answer is the other branch: **the paired
comparison would need roughly 27.6 years of history to detect an effect this size at 80% power,
against the 3.50 years actually available.**

That is a complete, quantitative, publishable answer — not a failure, and not a licence to go looking
for a second exclusion rule. Notebook 020's verdict stands exactly as recorded.

**No bar was ever excluded from either book's own backtest, Sharpe, deflation, or universe — only from
the paired comparison series.**

## What to test next

- **The data-quality route is exhausted for this rule.** A fresh full-conjunction notebook was the
  pre-registered next step had the exclusion held up cleanly. It didn't, so that path is closed. If a
  future notebook wants to revisit the data-quality angle, it needs a **materially different,
  independently motivated signal** — not a variant of this flag with different parameters.
- **The 27.6-years-required figure is the sharpest actionable fact here.** The paired comparison will
  never clear on this development window alone, without either (a) a genuinely lower-variance
  difference estimator — such as a coupled bootstrap conditioning on the shared market-regime component
  both books are exposed to, which notebook 020 flagged independently — or (b) the eventual holdout
  window, once some future test legitimately earns access under its *own* conjunction, not this one's.
- **The joint floor-plus-slow-carry cross-venue construction was not reached.** This notebook was
  scoped to the single paired-comparison gap. Still open, deferred again.
- **Two further deferred items carry forward untouched** — an on-chain and exchange-flow data survey,
  and market-making and inventory-managed liquidity provision. Nothing here bears on either.

*Notebook: `src/research/021_rc3_power_and_data_quality.ipynb`. Both books' frozen construction was
never touched, and no holdout access is granted under any outcome.*
