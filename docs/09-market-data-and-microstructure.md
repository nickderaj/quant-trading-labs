# 09 — Market data and microstructure

This file explains what an OHLCV bar actually is, and the crypto-specific facts that
shape why this research programme's data is shaped the way it is. Useful background for
[01](01-probability-and-distributions.md) onward, but can also be read first if you want
to understand the data before the maths.

---

### OHLCV bar

**In one sentence.** A summary of everything that happened to a price during one fixed
time window, compressed into five numbers: **O**pen (first trade price), **H**igh
(highest trade price), **L**ow (lowest trade price), **C**lose (last trade price), and
**V**olume (total amount traded) — the standard building block of nearly all financial
time-series data.

**The maths.** No formula; a data-aggregation convention. Given every individual trade
in a window $[t, t+\Delta)$: $O = $ price of the first trade, $C = $ price of the last,
$H = \max$(prices), $L = \min$(prices), $V = \sum$(trade sizes).

**Why it is here.** Every single model in this repo starts from OHLCV bars — the range
estimators (Parkinson, Garman-Klass, ...) exist specifically to use the *whole* bar
(H and L, not just C) rather than throwing that information away and using only
close-to-close returns.

**Worked example.** `build_asset_frame`'s `hl_log` column (`log(high/low)`) is the
foundational quantity every range estimator builds from — information genuinely present
in the raw bar data that a close-only model discards entirely.

**Pitfalls.** A bar's H and L only capture the *extremes actually reached*, not the full
path of the price within the bar — two very different intrabar paths (one that visited
its high early and drifted down, one that visited it late after drifting up) produce
identical OHLCV numbers, a fundamental information loss any range estimator inherits.

---

### Interval / timeframe

**In one sentence.** How much real time each bar covers — 1 hour, 4 hours, 12 hours, 1
day, in this repo's case — a choice that changes not just how many bars you have, but the
statistical character of the data itself (fatter tails at finer intervals, per this
programme's own repeated finding).

**The maths.** No formula; `INTERVAL_HOURS = {"1h": 1, "4h": 4, "12h": 12, "1d": 24}` in
this repo's own code is the literal mapping used throughout.

**Why it is here.** Every refit-cadence calculation, every `bars_per_day` conversion,
and the entire [aggregational Gaussianity](01-probability-and-distributions.md#aggregational-gaussianity)
finding depend on comparing behavior *across* intervals consistently — which is why
cadence is declared in calendar days and converted per-interval rather than fixed in bar
counts.

**Worked example.** The same calendar month is 720 bars at 1h but only 30 bars at 1d —
this repo's [refit cadence](02-estimation-and-fitting.md#refit-cadence) convention
exists specifically to keep comparisons fair across this range.

**Pitfalls.** A finding that holds at one interval doesn't automatically hold at
another — this programme repeatedly checks claims across all four intervals rather than
assuming a 1d result generalizes down to 1h or vice versa (e.g. fitted Student-t $\nu$
ranges from 1.98 at 1h to 2.88 at 1d — a materially different number depending on
interval).

---

### Log return and why logs

**In one sentence.** The standard way of measuring "how much did the price change,"
using the logarithm of the price ratio rather than the raw percentage change — chosen
because log returns compound additively over time and behave more symmetrically for
gains vs. losses.

**The maths.** $r_t = \log(P_t / P_{t-1})$, as opposed to the
[simple return](#simple-vs-log-return) $(P_t - P_{t-1})/P_{t-1}$. Log returns over
consecutive periods simply *add*: the two-period log return is exactly $r_{t} + r_{t-1}$,
whereas simple returns must be compounded multiplicatively.

**Why it is here.** Every `log_return` column throughout this repo (`build_asset_frame`)
uses this convention — it's the input to every GARCH fit, every distributional fit, and
every scoring rule in this whole research programme.

**Worked example.** A price that goes from 100 to 110 then back to 100 has a simple
return of $+10\%$ then $-9.09\%$ (not symmetric); the log returns are $+0.0953$ then
$-0.0953$ (exactly symmetric, and they sum to exactly 0, matching the fact that the price
ended up unchanged) — a cleaner algebraic property that makes log returns the natural
choice for compounding and for symmetric statistical modeling.

**Pitfalls.** Log returns are an *approximation* to simple returns that gets worse for
very large moves (a 100% simple gain is a $\log(2)\approx 0.693$ log return, not $1.0$) —
for the modest per-bar moves typical in this data this distinction rarely matters in
practice, but it's worth remembering when interpreting a fitted parameter's scale for an
unusually large move.

---

### Simple vs. log return

**In one sentence.** Two different (but for small moves, nearly identical) ways of
expressing "how much did the price change": simple return is the plain percentage
change; log return (above) is its logarithmic cousin, chosen throughout this repo for its
additive compounding property.

**The maths.** Simple: $R_t = (P_t - P_{t-1})/P_{t-1}$. Log: $r_t = \log(1+R_t) =
\log(P_t/P_{t-1})$. For small $R_t$ (a few percent or less), $r_t \approx R_t$ — they're
nearly interchangeable at typical per-bar magnitudes.

**Why it is here.** This repo consistently uses log returns for modeling (additive,
symmetric, well-behaved for statistical fitting) but reports some results in more
intuitive percentage terms in write-ups (e.g. "0.58% per hour" for BTC's 1h standard
deviation) — worth knowing which convention a given number uses when comparing across
this repo's write-ups.

**Worked example.** BTC's fitted 1h log-return standard deviation of $\approx 0.00584$
translates to a nearly identical $\approx 0.58\%$ simple-return standard deviation at
this small magnitude — the distinction genuinely doesn't matter much at 1h, but would
start to for a much larger, rarer move.

**Pitfalls.** Never average simple returns and log returns together, or convert between
them casually for a series of multi-period returns — only log returns sum correctly
across time; simple returns must be compounded (multiplied, not added).

---

### Perpetual futures

**In one sentence.** A crypto derivative contract that mimics holding the underlying
asset directly (no expiration date, unlike traditional futures) while trading
continuously, 24/7 — this repo's actual instrument for BTC/ETH/etc., not spot price.

**The maths.** No formula; a contract-design fact. A perpetual future's price is kept
close to the underlying spot price via a periodic
[funding rate](#funding-rate) payment between long and short holders, rather than by
convergence to a fixed expiration/settlement date the way a traditional future works.

**Why it is here.** This is the actual instrument every dataset in this repo is built
from — its "no overnight gap" property (below) directly shapes several distributional
findings (tiny gap-return std, Yang-Zhang's gap term being nearly irrelevant).

**Worked example.** Notebook 4's own gap-vs-intrabar decomposition (gap std 400-2,000x
smaller than intrabar std, at every interval) is a direct, measured consequence of
trading on a perpetual future that never actually closes for the day.

**Pitfalls.** Findings from this dataset (near-zero gap variance, continuous trading)
don't automatically transfer to markets with genuine trading-session structure (a
traditional stock exchange with a real overnight gap, for instance) — this is a
data-specific, not universal, fact worth keeping in mind before generalizing any finding
built on it.

---

### Funding rate

**In one sentence.** A periodic payment exchanged between long and short holders of a
perpetual future, designed to keep its price anchored close to the underlying spot
price — when the future trades above spot, longs pay shorts (and vice versa), creating an
economic incentive pulling the two back together.

**The maths.** No universal closed form; typically computed from the futures-spot price
premium/discount and charged at fixed intervals (e.g. every 8 hours), though the exact
formula is exchange-specific.

**Why it is here.** `funding_rate` is one of the 27 candidate features screened for
cross-sectional predictive power in notebook 3's Phase 4 — found to survive the IC
screen (weakly, negative sign) only at 4h, the weakest of the surviving signals reported
there.

**Worked example.** Notebook 3 found `funding_rate` (negative sign, mean IC $-0.0095$,
NW t-stat $-4.4$) — a statistically real but economically modest cross-sectional
predictor, surviving at only one of three intervals tested.

**Pitfalls.** A negative funding-rate coefficient's economic interpretation (elevated
funding tends to precede weaker forward returns, plausibly a crowded-long unwind
dynamic) is a real, but separate, question from whether it's *statistically* significant
— this repo reports both the sign and the statistic together rather than either alone.

---

### Taker vs. maker

**In one sentence.** Two roles in a trade: a "maker" places a resting order that sits on
the order book waiting to be matched; a "taker" places an order that immediately matches
against an existing resting order — makers add liquidity to the book, takers consume it.

**The maths.** No formula; an exchange-mechanics distinction, usually reflected in
different fee rates (makers typically pay lower fees, or are even paid a rebate, since
they're providing liquidity; takers typically pay higher fees for consuming it).

**Why it is here.** `taker_buy_volume` (part of `build_asset_frame`'s input columns) is
the basis for `taker_buy_ratio` below — a signal of whether recent trading pressure was
predominantly aggressive buying (taking from the ask side) or aggressive selling.

**Worked example.** A market order that immediately executes against the best available
resting price is a taker trade; a limit order placed away from the current price, sitting
until someone else trades into it, is a maker trade.

**Pitfalls.** Taker/maker volume splits say something about *aggression* (who initiated
the trade), not necessarily about the trade's ultimate directional information content —
a large taker-buy volume doesn't guarantee the price will continue up, which is
consistent with `taker_buy_ratio`'s own weak/uninformative showing in this repo's
distributional analysis (see below).

---

### Taker buy ratio

**In one sentence.** The fraction of a bar's total trading volume that came from
aggressive (taker) buy orders, rather than taker sell orders — a per-bar summary of
whether buying or selling pressure was more aggressive during that bar.

**The maths.** $\mathrm{taker\_buy\_ratio} = \frac{\text{taker buy volume}}{\text{total
volume}}$ — bounded in $(0,1)$ by construction, which is exactly why it's fit with a
[Beta distribution](01-probability-and-distributions.md#beta-distribution) rather than a
normal.

**Why it is here.** Notebook 4's Phase 1 fits Beta to this column directly, finding it
tightly concentrated near 0.5 (balanced buy/sell pressure) with concentration
*increasing* as the bar widens (more individual trades averaged in).

**Worked example.** Notebook 4's fitted $(a,b)$ growing from $(245.8, 247.7)$ at 4h to
$(869.3, 875.8)$ at 1d, always with $a \approx b$ — read as "buy/sell pressure is close
to balanced in aggregate, consistent with a two-sided perpetual futures market," not a
persistently directional signal.

**Pitfalls.** `distributions.py`'s Beta fitter requires every observation strictly
inside $(0,1)$ — an all-taker-sell or all-taker-buy bar (a genuine occurrence at finer
granularities) breaks a whole-sample fit; notebook 4's "fit failed (boundary obs)" cells
at 1h/4h document exactly this real data-boundary effect rather than a code bug.

---

### Order book

**In one sentence.** The live list of all resting (unmatched) buy and sell orders for an
asset at any given moment, organized by price — the underlying structure that trades
against, though this repo's data doesn't work with order-book snapshots directly, only
their aggregated OHLCV consequences.

**The maths.** No formula; a market-structure concept. The "best bid" is the highest
resting buy order's price; the "best ask" is the lowest resting sell order's price.

**Why it is here.** Mentioned here mainly to define
[bid-ask spread](#bid-ask-spread-and-bounce) below — this repo works entirely from
aggregated OHLCV bar data, never raw order-book snapshots, so this entry is background
context rather than a directly-used quantity anywhere in the codebase.

**Worked example.** A quoted price of "$50,000 bid / $50,001 ask" means the highest
resting buy order is at $50,000 and the lowest resting sell order is at $50,001 — a taker
buy order would immediately execute at $50,001 (crossing the spread), not at $50,000.

**Pitfalls.** Not applicable to this repo's own data pipeline directly, since only
aggregated bar-level data (not live order-book depth) is used anywhere in this research
programme.

---

### Bid-ask spread and bounce

**In one sentence.** The gap between the best available buy price and best available
sell price at any instant (the spread), and the resulting artifact where consecutive
trade prices appear to bounce back and forth between the bid and ask even when the
"true" underlying price hasn't moved at all (the bounce).

**The maths.** Spread $= P_{\mathrm{ask}} - P_{\mathrm{bid}}$. Bid-ask bounce: if
consecutive trades alternate between hitting the bid and hitting the ask, the observed
trade-price series shows artificial back-and-forth movement of roughly the spread's
size, unrelated to any genuine price change.

**Why it is here.** Cited in notebook 4's write-up as the likely microstructure
explanation for two separate, related findings: BTC's intrabar range running
systematically *below* the driftless-Brownian prediction (bid-ask bounce can suppress the
realized high-low spread relative to a pure random walk), and the rejected-geometric,
excess-short-run-reversal pattern in sign-run lengths (bounce mechanically creates extra
sign flips at the smallest scales).

**Worked example.** If the "true" price is $50,000.50 and trades alternate between
hitting a $50,000 bid and a $50,001 ask, the raw trade-price series shows apparent moves
of about $1 back and forth, purely from which side of the spread each trade happened to
hit — not genuine price discovery.

**Pitfalls.** Bid-ask bounce is a microstructure artifact most relevant at very fine
granularities (individual trades, or bars close to trade-level resolution) — its
influence is expected to matter less at coarser intervals (4h/12h/1d), consistent with
notebook 4's own finding that the normalized-range departure from the Brownian prediction
shrinks (from $-13.5\%$ at 1h to $-6.1\%$ at 1d) as bars widen.

---

### Intrabar vs. gap return

**In one sentence.** A bar's total return can be split into two pieces: the **gap**
return (the jump from the previous bar's close to this bar's own open) and the
**intrabar** return (the move from this bar's open to its own close) — a useful
decomposition for understanding *where*, within a bar, a price move actually happened.

**The maths.** Gap: $g_t = \log(O_t / C_{t-1})$. Intrabar: $i_t = \log(C_t/O_t)$. The two
sum exactly to the bar's total log return: $r_t = g_t + i_t$.

**Why it is here.** `build_asset_frame`'s `gap_return` and `intrabar_return` columns are
exactly this split, feeding both the Garman-Klass/Yang-Zhang range estimators (which use
the intrabar term specifically) and notebook 4's own direct gap-vs-intrabar comparison.

**Worked example.** Notebook 4 found gap std 400-2,000x smaller than intrabar std at
every interval — on this instrument, essentially all of the return happens *within* a
bar, none of it in the gap between bars, a direct, measured consequence of continuous
perpetual-futures trading (see [why crypto perps have no overnight gap](#why-crypto-perps-have-no-overnight-gap)).

**Pitfalls.** This decomposition's usefulness depends heavily on the instrument actually
having a meaningful gap in the first place — for a traditional market with real
overnight closure, gap return would carry genuine information (news arriving while
closed); for BTC perpetuals, the gap term is close to a formality, which is exactly why
Yang-Zhang's gap-variance addition over Rogers-Satchell buys little here.

---

### Why crypto perps have no overnight gap

**In one sentence.** Traditional markets close overnight and on weekends, so news can
accumulate while trading is halted, producing a genuine "gap" at the next open — crypto
perpetual futures trade continuously, 24/7, with no scheduled close, so there is
structurally almost nothing for a "gap" to capture.

**The maths.** No formula; a market-structure fact directly measured in this repo's own
data: gap standard deviation is 400-2,000x smaller than intrabar standard deviation at
every interval tested.

**Why it is here.** This is the specific, repo-grounded justification cited wherever a
range estimator's gap-handling behavior is discussed (Yang-Zhang's gap term, the
gap-vs-intrabar decomposition above) — a structural fact about the instrument, not an
assumption.

**Worked example.** A traditional equity that closes at 4pm and reopens at 9:30am the
next day has 17.5 hours of "closed" time during which real information can accumulate
and only be reflected at the next open, producing a genuine overnight gap; BTC perpetual
futures have essentially zero such closed time.

**Pitfalls.** This is a fact specific to *this instrument* (continuously-traded crypto
perpetuals) — it should not be assumed to generalize to every crypto product (a
traditionally-listed crypto ETF, for instance, would have real market-hours structure and
a genuine overnight gap) or to any non-crypto asset class.

---

### Trade count

**In one sentence.** The number of individual executed trades within a bar — a measure
of trading *activity*, distinct from volume (which measures the total size traded, not
the number of separate transactions).

**The maths.** A simple count, `count` in this repo's own OHLCV data — the number of
matched trades, whatever their individual size, within the bar's time window.

**Why it is here.** `dist_lib.activity_forecast`'s "activity-based" volatility rung uses
trade count and its dispersion as regression features, and notebook 4's Phase 1 fits
[Poisson](01-probability-and-distributions.md#poisson-distribution)/[negative binomial](01-probability-and-distributions.md#negative-binomial)
to this column directly.

**Worked example.** Notebook 4 found trade-count dispersion indices (Var/Mean) of
114,393 (1h) up to 891,044 (1d) — massively overdispersed relative to Poisson's
predicted value of 1, meaning trade activity arrives in far burstier clusters than a
simple, constant-rate arrival process would produce.

**Pitfalls.** `NEXT_RUN_PROMPT.md`'s own tripwire convention flags a dispersion index
*near* 1 as suspicious (an aggregation bug erasing genuine clustering), not the enormous
observed values — the extreme overdispersion here is the expected, correct result, not a
red flag.

---

### Frozen-price bar

**In one sentence.** A bar during which the price genuinely didn't move at all (open,
high, low, and close all identical, or realized variance of exactly zero) — a real,
recurring data phenomenon at fine granularities that breaks several statistical
calculations if not handled explicitly, because a handful of formulas in this whole
research programme are undefined at exactly zero variance.

**The maths.** No formula; the defining condition is $H=L=O=C$ for a bar (or,
equivalently, `bar_squared_return == 0` / `rv_target == 0`).

**Why it is here — the source of (at least) two separate, real bugs in this repo:**

1. **Notebook 3's `realized_vol_24 == 0` bug**: a frozen-price rolling window with zero
   variance broke every distribution's variance-based moments downstream — the original
   motivation for `distributions.py`'s entire "detect a degenerate window and return
   `None` rather than propagating NaN/inf" convention, applied consistently across every
   `_fit_*` function in the module.
2. **Notebook 4's QLIKE bug #4**: `qlike_mse`'s mask originally allowed `actual == 0`,
   but QLIKE's `log(ratio)` term is undefined exactly there — 13 frozen-price bars at 1h
   (`rv_target == 0`) poisoned the whole-series mean QLIKE to `+inf` for *every single
   rung* until the mask was tightened to `actual > 0` specifically for the QLIKE
   calculation (MSE, unaffected by this issue, kept the wider `actual >= 0` mask).

**Worked example.** `distributions.py`'s Beta fitter's own "fit failed (boundary obs)"
cells in notebook 4's Phase 1 table are a closely related manifestation: a bar with
taker-buy ratio exactly 0 or 1 (an extreme version of the same "the data hit a hard
boundary" phenomenon) breaks a whole-sample fit the same way a frozen-price bar breaks a
variance-based one.

**Pitfalls.** `NEXT_RUN_PROMPT.md`'s own framing treats this as a class of bug that is
"certain to recur" in any new code touching realized variance or ratio-type features —
worth checking for explicitly (a strict `> 0` mask where a formula requires strictly
positive variance/ratios) in every new model this repo adds, rather than assuming
existing guards automatically cover a new computation.

---

### Realized variance from sub-bars

**In one sentence.** Building a more accurate variance estimate for a coarse bar (say,
1 day) by summing the squared returns of many finer sub-bars (say, 24 hourly bars) inside
it, rather than relying on the coarse bar's own single close-to-close squared return —
much less noisy, since it's built from many observations instead of one.

**The maths.** For a coarse bar composed of $m$ finer sub-returns $r_1,\dots,r_m$:
$\mathrm{RV} = \sum_{i=1}^m r_i^2$ — see
[realized variance/volatility](04-volatility-models.md#realized-variance-volatility)
for the general definition; this entry is specifically about *how* this repo constructs
it from cached data.

**Why it is here.** `dist_lib.realized_variance_from_subbars` implements exactly this,
using cached 1h data to build the RV target for 4h/12h/1d bars — used throughout
`build_asset_frame` (`rv_target` coalesces the sub-bar RV where available, falling back
to the bar's own squared return otherwise).

**Worked example.** At 1h itself, no finer cached series exists in this repo, so
`rv_target` falls back to `bar_squared_return` — explicitly noted, both in code and in
every write-up that uses it, as a noisier proxy specifically at that one interval,
a real, acknowledged limitation rather than something papered over.

**Pitfalls.** This construction requires the finer sub-bar data to actually be reliably
cached and complete — a gap in the underlying 1h data (this repo's own write-up notes a
real, documented ~120-hour gap for SOL/XRP in Feb-Apr 2022, a genuine hole in Binance's
own archive) would silently propagate into a slightly less accurate coarser-bar RV target
for that period, worth remembering when interpreting results spanning that window.
