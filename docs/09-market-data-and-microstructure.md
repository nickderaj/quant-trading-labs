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

### Carry / basis trade

**In one sentence.** A trade whose expected profit comes from a structural payment
(here, [funding rate](#funding-rate)) rather than from predicting which way price will
move — go long the assets that pay you to hold them, short the assets that charge you.

**The maths.** No new formula beyond funding rate itself; the *ranking* signal is
$-\text{funding\_rate}$ (or its rolling z-score), not funding rate directly, because of
how the payment is directional: funding positive means longs pay shorts, so being SHORT
collects it and being LONG pays it — the sign a genuine carry ranking needs is the
opposite of funding rate's own raw sign.

**Why it is here.** Notebook 7's Phase C tests carry as a PRIMARY cross-sectional
signal for the first time in this research programme (notebook 3 only ever included
raw `funding_rate` as one of eight features inside a fitted model). The sign convention
above is a pre-declared correction to notebook 7's own runbook, decided from two
independent sources before any Phase C number was seen: the economic mechanism itself,
and notebook 3's own screening result that raw `funding_rate`'s IC against forward
return is *negative* (`src/results/003_cross_sectional_ic.md` Phase 4) — ranking directly
on funding_rate points the wrong way for a book that wants to profit from the payment.

**Worked example.** `src/results/007_alpha_generation.md`'s Phase C: ranking the frozen
30-symbol universe by $-\text{funding\_rate}$ (long the most negative-funding names,
short the most positive-funding ones) at 4h/12h/1d, all four origin offsets — full
funding coverage (30/30 symbols, 100% of panel rows), so this is not a
coverage-limited result the way `funding_rate`'s own weak notebook-3 IC survival was.

**Pitfalls.** The whole appeal of a carry trade is supposed to be low turnover (a
payment schedule changes slowly, unlike a price-momentum rank) — but a *rank-based*
carry book can still churn heavily if funding rates cluster closely together
cross-sectionally, so small changes flip which symbols sit in the top/bottom
`top_frac`. Notebook 7's own Phase C found exactly this: realized turnover on the
carry book (~970-1260/year at 4h) was *higher* than the price-based cfg2_12h signal
it was meant to contrast with, not lower — "structurally low-turnover" is a claim about
the underlying payment, not automatically true of every way you might rank and trade it.

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

---

### Futures roll, front month, and continuous contract

**In one sentence.** A futures contract expires on a fixed date, so a series that wants
to represent "the price of crude oil" for years at a time has to splice together a
sequence of individual contracts — the **front month** is whichever contract is
currently the nearest-to-expiry one still being actively traded, and **rolling** is the
act of switching from the expiring front month to the next one before delivery risk
arrives; a **continuous contract** is the resulting spliced series.

**The maths.** No single formula — it's a rule, applied daily. Notebook 8's rule: roll
$N$ calendar days before the contract's first-notice date (or last-trade date where
first-notice isn't populated), snapped backward off a weekend since no exchange trading
calendar is available. $N=5$ in production; sensitivity checked at $N \in \{3, 5, 10\}$.

**Why it is here.** Every tail statistic, every backtest, and the risk engine itself sit
on top of this series. Get the roll wrong and every number built on it is wrong in a way
that doesn't announce itself — notebook 8's own Phase 0 found this out directly: rolling
into a contract-month that was never actually traded (see
[roll_calendar vs. contracts.parquet](#seasonal-and-liquid-months-vs-nominally-listed-months)
below) silently deleted 60% of one product's front-month series before the bug was caught.

**Worked example.** CL's May-2020 contract (the one that famously settled at
$-\$37.63/\text{bbl}$ on 2020-04-20) had its roll date fall on 2020-04-17 under the
production $N=5$ rule — a real book following this rule was already three trading days
into the June contract by the time the crash happened, and never touched the negative
print at all. The row itself survives in the raw per-contract data (see
[hygiene filter](#negative-and-contaminated-settlement-rows) below); it just never
enters the *continuous* front-month series, because rolling early is what a real book
that respects physical delivery risk actually does.

**Pitfalls.** "Front month" is not "most-traded contract" — those disagree constantly.
On any given day the most heavily-traded platinum contract might be four months out
while the nominal front month trades a token handful of contracts; a naive
max-volume-per-date selection rule (rather than a fixed roll calendar) will happily
follow that liquidity around and produce a series with implausible month-to-month jumps.

---

### Back-adjustment vs. ratio-adjustment

**In one sentence.** Splicing contracts together (see [futures roll](#futures-roll-front-month-and-continuous-contract))
creates an artificial price jump at every roll date (the old and new contract rarely
trade at exactly the same price); **back-adjustment** removes the jump by adding a
constant offset to every price before the roll, while **ratio-adjustment** removes it by
multiplying by a constant ratio instead.

**The maths.** Back-adjustment: at each roll, $\text{offset} \mathrel{+}= P_{\text{new}} - P_{\text{old}}$,
then every pre-roll price has the *cumulative* offset subtracted. Ratio-adjustment: at
each roll, $\text{ratio} \mathrel{*}= P_{\text{old}} / P_{\text{new}}$, then every
pre-roll price is *multiplied* by the cumulative ratio. Both make the return **at** the
roll date well-defined and continuous; neither is the return an account actually earned
that day (see the third convention below).

**Why it is here.** Notebook 8 needed three separate return conventions and got the
choice between these two wrong on the first attempt. The *unadjusted* return — computed
within a single contract only, `null` exactly at every roll boundary — is the correct
series for backtests and cost accounting, since P&L and fees are charged on the contract
actually held. Back- and ratio-adjustment exist only to make a *continuous* series for
charting and for statistics (tail index, ACF, ...) that need an unbroken series.

**Worked example.** Over a 16-year, ~200-roll history, back-adjustment's additive offset
can grow large enough to push an old contract's *adjusted* price negative or through
zero — nonsensical for a commodity, and it happened in this notebook: several products'
back-adjusted 2010-era prices went negative purely from the accumulated splice offset,
producing single-day "returns" of 100%+ that were pure adjustment artifact, not a real
price move. Ratio-adjustment (multiplicative) cannot cross zero as long as the raw price
never does, and was the series notebook 8 actually used for every tail/ACF/density
statistic once this was caught.

**Pitfalls.** Never compute a log return across a roll boundary on **unadjusted**
prices — that silently invents a jump equal to the roll's own price gap, which can dwarf
a normal day's move and will single-handedly corrupt a kurtosis or Hill-tail-index
estimate. And never use **back-adjusted** returns for a multi-decade tail statistic
without first checking the adjusted price hasn't drifted through zero.

---

### Contango and backwardation

**In one sentence.** **Contango**: futures prices increase with maturity (a farther-out
contract costs more) — the "normal" state for a storable good under positive cost of
carry. **Backwardation**: futures prices decrease with maturity — the market is paying a
premium for immediate delivery, the classic signature of scarce current inventory.

**The maths.** This notebook's operational definition: the annualised F1→F2 roll slope,
$\text{slope} = \dfrac{\ln(F_2/F_1)}{\text{dte}_2 - \text{dte}_1} \times 365$.
$\text{slope} < 0$ (F2 cheaper than F1) is backwardation; $\text{slope} > 0$ is contango.

**Why it is here.** Inventory theory (Kaldor-Working; Deaton-Laroque) predicts that
backwardation — low inventory — should carry higher volatility and a fatter right tail
(a supply shock has less buffer to absorb it), the central conditional-tail question
notebook 8's Phase 4 exists to test. It is also the raw signal behind the oldest
commodity risk premium in the literature: go long backwardated markets, short contangoed
ones (Keynes' "normal backwardation"; Erb-Harvey 2006; Gorton-Rouwenhorst 2006).

**Worked example.** A CL curve with $F_1 = \$75.00$ (10 days to expiry) and
$F_2 = \$74.50$ (40 days to expiry) has
$\text{slope} = \ln(74.50/75.00) / 30 \times 365 \approx -0.081$, i.e. roughly 8.1%
annualised backwardation — the front month is pricier than the deferred month by an
amount that, extrapolated, corresponds to an 8%/year roll yield for a long-front
position.

**Pitfalls.** The sign convention is easy to flip. A carry *trading* signal wants to be
long backwardation, so the predictive score fed into a ranking function needs the
**negative** of the roll slope (very negative slope → very positive score → long leg) —
using the raw slope as the ranking score silently reverses the whole book's positions.

---

### Convenience yield

**In one sentence.** The non-monetary benefit of physically holding a commodity right
now rather than a promise to receive it later — insurance against a stockout, the
ability to keep a factory running — which is what allows backwardation to persist
without being an arbitrage.

**The maths.** From the cost-of-carry identity (below), convenience yield $y$ is
whatever makes it balance: $F = S \, e^{(r + u - y)T}$, so
$y = r + u - \dfrac{1}{T}\ln(F/S)$ where $S$ is spot, $r$ the risk-free rate, $u$
physical storage cost. $y$ is not directly observable — it is inferred residually from
the other four quantities, all of which *are* observable.

**Why it is here.** It is the theoretical reason backwardation and contango exist at all
rather than being pure arbitrage. This notebook does not attempt to estimate $y$
directly (that needs a storage-cost estimate this dataset does not provide); the F1/F2
roll slope is used as the *observable proxy* for the inventory state $y$ is meant to
summarize.

**Worked example.** If crude spot is $75, one-year storage costs 2%/year, the risk-free
rate is 5%/year, and the one-year future trades at $73, then
$y = 0.05 + 0.02 - \ln(73/75) \approx 0.097$, roughly 9.7%/year — inventory is scarce
enough that holders are effectively being paid a large convenience yield to keep barrels
on hand rather than sell the future.

**Pitfalls.** Convenience yield is a *residual*, not a free-standing measurement — any
error in the storage-cost or financing-rate assumption goes straight into the estimate.
It is also not constant: it rises sharply exactly when inventory is tight, which is
precisely the regime this notebook's Phase 4 conditions on.

---

### Cost of carry

**In one sentence.** The net cost of holding a physical position from today until a
future delivery date — financing plus storage minus convenience yield — which is what a
futures price has to compensate for relative to spot.

**The maths.** $F = S \, e^{(r + u - y)T}$ — the same identity convenience yield is
solved from, read the other direction: given spot, financing, storage, and convenience
yield, this pins down the no-arbitrage futures price.

**Why it is here.** It's the textbook explanation for why $F \neq S$ at all, and the
frame every other term on this page (contango, backwardation, convenience yield, the
roll slope itself) sits inside.

**Worked example.** Gold has near-zero convenience yield (it isn't consumed, and storage
is cheap and standardized) and near-zero physical storage cost relative to its price —
so gold's forward curve should track almost purely the financing rate $r$, and does:
gold trades in near-permanent contango, unlike crude or natural gas which flip sign
depending on inventory.

**Pitfalls.** Cost-of-carry is a no-arbitrage *bound*, not a prediction of what the
curve will do — it says what $F$ *must* be given the other four inputs, not which way
inventory will move next. Confusing the identity for a forecast is the same category of
mistake as confusing an accounting relationship for a causal one.

---

### The Samuelson effect

**In one sentence.** Volatility rises as a futures contract approaches expiry, because
there is less and less time left for supply and demand to arbitrage away fresh news
before delivery — a contract with one day left reacts to news the same size move it
always did, but that move now represents a much larger fraction of its remaining
"time to mean-revert."

**The maths.** No single closed form; operationalised here as realised volatility
bucketed by days-to-expiry (`dte_f1`), e.g. $[0,5), [5,10), \ldots, [120, \infty)$ days,
comparing mean realised vol across buckets. A genuine Samuelson effect shows monotonic
(or near-monotonic) rise in vol as the near bucket is approached.

**Why it is here.** It is a first-order reason a naive front-month series is
contaminated near every roll — if the roll rule waits too long, the series inherits a
burst of expiry-driven volatility that has nothing to do with the underlying commodity's
"true" risk that week, one more argument (alongside the delivery-risk argument) for
rolling before expiry rather than at it.

**Worked example.** Predicted strongest in seasonal/storage-driven products (natural
gas, grains) where expiry-week supply/demand information is most information-dense, and
weakest in metals where physical delivery logistics dominate less of the price
formation process.

**Pitfalls.** The effect is about time-to-expiry, not calendar time — a contract can
show elevated vol in its final week regardless of *which* calendar month that week falls
in, so it must not be confused with (or allowed to contaminate) a month-of-year
seasonality estimate computed from the same series.

---

### Basis and basis-momentum

**In one sentence.** **Basis** is the gap between a nearby and a deferred futures price
(or between spot and futures) — essentially the same object as the roll slope above,
viewed as a level rather than an annualised rate. **Basis-momentum** (Boons-Prado 2019)
is trend-following applied to that gap itself: has the basis been widening or narrowing
recently, independent of whether the *level* is currently in backwardation or contango.

**The maths.** Basis-momentum signal: the trailing return differential between the F1
and F2 legs, $\sum_t (r_1_t - r_2_t)$ over some lookback — a distinct object from the
plain roll-slope carry signal, and from price-level momentum on F1 alone.

**Why it is here.** It's a **distinct, more recent factor** from carry per se (sec 3.3
of this notebook's own pre-registration) — carry bets on the *level* of backwardation
persisting; basis-momentum bets on the *direction of change* in that level continuing,
which can point a different way than the level itself.

**Worked example.** A market that is contangoed but rapidly *moving toward*
backwardation (inventory draining) gives a basis-momentum signal in the opposite
direction to what a pure level-based carry signal would say — carry says "short" (still
contangoed), basis-momentum says "long" (moving the right way for a future backwardation
bet to pay off first).

**Pitfalls.** Declared in this notebook's pre-registration as in scope but not run in
this pass, given the volume of machinery Phase 5 already covers with carry and
time-series momentum — see the results MD for the explicit scope tradeoff, not a silent
omission.

---

### Hedging pressure and the COT report

**In one sentence.** Keynes' "normal backwardation" theory: commodity producers are
natural hedgers who want to sell their future output forward, so they must pay
speculators a risk premium to take the other side — **hedging pressure** measures how
lopsided that natural hedging demand is, typically via the CFTC's weekly
**Commitment of Traders (COT)** report's split of open interest between commercial
(hedger) and non-commercial (speculator) positions.

**The maths.** A common operationalisation: net non-commercial position as a fraction of
open interest, $\frac{\text{noncomm\_long} - \text{noncomm\_short}}{\text{open\_interest}}$,
or its rolling z-score — no single canonical formula, but this notebook's data only
supports one market's worth of it (see pitfalls).

**Why it is here.** It's a third, independent theoretical account of *why* commodity
risk premia might exist (alongside the storage/convenience-yield account behind
contango/backwardation), operationalisable here for exactly one product.

**Worked example.** CFTC code 067651 (light sweet crude, NYMEX) is the one COT series
with weekly history back to 2006 this notebook's data provides. It must be lagged by at
least a full week before touching any signal: the report is **as-of Tuesday, released
Friday 15:30 ET**, and is itself subject to later revision.

**Pitfalls.** COT coverage in this dataset is CL and ES only — nowhere near enough to
build a cross-sectional hedging-pressure factor across the 16-product panel. Declared
explicitly as a single-market, CL-only time-series test (or out of scope entirely), never
silently extrapolated into a panel-wide claim the data can't support.

---

### Crack and crush spreads

**In one sentence.** **Crack spreads** are refining margins — the price gap between
crude oil and its refined products (gasoline, heating oil) — and **crush spreads** are
the analogous processing margin for soybeans into soybean meal and oil. Both are
"physical process" spreads: the legs are linked by an actual industrial conversion, not
just by both being commodities.

**The maths.** The 3-2-1 crack spread: $3 \times \text{CL} - 2 \times \text{RB} - 1 \times \text{HO}$
(3 barrels of crude yield roughly 2 of gasoline and 1 of heating oil, in the ratio a
refinery actually processes them). Crush spread: a similar weighted combination of ZS
(soybeans) against ZM (meal) and ZL (oil), reflecting a crushing plant's actual output
ratios.

**Why it is here.** These spreads are structurally lower-volatility, lower-directional-
exposure trades than an outright position — the two/three legs share most of their
common price risk, leaving mostly the refining/crushing *margin* itself as the exposed
factor, which is exactly the kind of "cheap in vol terms" opportunity sec 3.3 flags as
the most plausible place a genuine edge could survive transaction costs.

**Worked example.** `spreads/crack_321.parquet` and `spreads/crush_soy.parquet` are
pre-built in this dataset, with `regime` and `roll_window_flag` columns marking exactly
the dates where a spread's *own* roll mechanics (each leg rolls on its own calendar)
contaminate the spread's measured return — not incidental data, the single most
important column for trading any of these spreads honestly.

**Pitfalls.** A spread backtest that looks profitable only *inside* `roll_window == True`
dates is not profitable — it is a roll-mechanics artifact, and this notebook's own
standard requires reporting every spread result both including and excluding those
dates specifically to catch this.

---

### Calendar spreads

**In one sentence.** The price gap between two delivery months of the *same*
underlying commodity (e.g. CL's December contract vs. its following June) — a pure play
on the shape of one product's own term structure, with none of the cross-commodity
processing-ratio complexity of a crack or crush spread.

**The maths.** $\text{spread} = F_{\text{near}} - F_{\text{far}}$ (or a ratio), tracked
as its own time series with its own roll calendar (the *spread's* roll, distinct from
either individual leg's roll) — this is exactly the same quantity as the roll slope
behind [contango and backwardation](#contango-and-backwardation), just expressed as a
raw price difference rather than an annualised rate.

**Why it is here.** It is the vehicle notebook 8's spread mean-reversion strategy (sec 4
Phase 5, strategy E) would trade — cointegrated legs, mostly-hedged directional
exposure, cheap in vol terms, one of the two places sec 3.3 flags as most plausible for
a surviving edge (alongside the crack/crush spreads above).

**Worked example.** Not every product has a full calendar-spread ladder in this
dataset — `cl_cal_m1m2.parquet` (the nearest calendar spread) does not exist for CL,
only `cl_cal_m2m3` and `wti_calendar`; checking each file's actual existence before
assuming symmetry across products is a real, documented gotcha in this dataset.

**Pitfalls.** Same roll-window discipline as crack/crush spreads applies, and a
stationarity/cointegration check on the two legs is a precondition for trading a
calendar spread as mean-reverting at all — an uncointegrated pair can drift arbitrarily
far apart with no reversion ever arriving.

---

### Cointegration and the Engle-Granger test

**In one sentence.** Two individually non-stationary price series (each wanders like a
random walk on its own) are **cointegrated** if a fixed linear combination of them is
stationary — i.e. they wander *together*, so the gap between them keeps returning to a
stable level even though neither leg does on its own. This is the actual statistical
license to trade a spread as mean-reverting; without it, "the spread looks like it
reverts" is an unfounded eyeball impression.

**The maths.** The **Engle-Granger two-step test**: (1) regress one leg on the other,
$\text{leg}_1 = c + h \cdot \text{leg}_2 + u_t$ ($h$ is the hedge ratio); (2) run the
[augmented Dickey-Fuller test](03-statistical-inference.md#stationarity-and-the-augmented-dickey-fuller-test)
on the residual $\hat u_t$. If $\hat u_t$ is stationary, the two legs are cointegrated and
$\hat u_t$ itself — a stationary series with a well-defined mean to revert to — is the
tradeable spread. In this repo's own pre-built spread data, step (1) is already done
upstream (`hedge_ratio` and the `value` column *are* $\hat u_t$), so
`spread_lib10.adf_test` run directly on `value` performs step (2) and completes the test.

**Why it is here.** It is the difference between "this spread's history happens to show
mean reversion" (a backward-looking, possibly spurious observation — two independent
random walks can drift together for a long stretch by pure chance) and "there is a
structural reason this gap cannot grow without bound" (the actual claim a mean-reversion
strategy needs to be true prospectively, not just historically).

**Worked example.** brent_wti clears the test comfortably (ADF t = −3.39 vs. the 5%
critical value of −2.86) — BZ and CL crude are genuinely linked (substitutable grades, one
global oil market) and their spread is a legitimate mean-reversion candidate. gold_silver
does not (t = −1.76) — gold and silver track a common macro factor loosely, but nothing
structurally pins their *ratio*, and notebook 10a excludes it from any regime-gated
backtest on exactly this basis.

**Pitfalls.** Cointegration is a **property of the pair as constructed**, not of either
leg alone — a hedge ratio estimated on one sample window and never re-checked can drift
out of the relationship that made the pair cointegrated in the first place. It is also not
transitive in the naive sense: A cointegrated with B and B cointegrated with C does not
guarantee A cointegrated with C. And a passing ADF test says the *level* reverts — it says
nothing about how fast (see half-life, next) or how large the round-trip transaction cost
is relative to the amplitude of the reversion, both separate questions a cointegration
test alone cannot answer.

---

### Ornstein-Uhlenbeck process and half-life of mean reversion

**In one sentence.** The continuous-time model behind "mean reversion with a speed": a
process that is constantly pulled back toward a long-run mean at a rate proportional to
how far it currently is from that mean, and **half-life** is the plain-English translation
of that pull-back rate into "how many days until half of today's deviation is gone."

**The maths.** SDE form: $dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t$, where $\theta>0$ is
the mean-reversion speed and $\mu$ the long-run mean. Its discrete-time analogue is
exactly the AR(1)-in-differences regression this repo already fits
(`research_lib9.ols_ar1_diff`): $\Delta v_t = \alpha + \beta v_{t-1} + \varepsilon_t$, with
$\beta$ playing $-\theta$'s role (a more negative $\beta$ = faster pull-back). Half-life
follows from solving $(1+\beta)^k = 0.5$:
$$\text{half-life} = \frac{-\ln 2}{\ln(1+\beta)}, \quad -1 < \beta < 0.$$

**Why it is here.** Half-life is the number that turns a statistically-significant
$\beta$ into a *tradeable* fact: it sets the natural holding period, and therefore roughly
how many round trips per year the strategy needs, and therefore how much of its edge a
per-trade cost model will eat. A significant but glacially slow-reverting spread
(platinum_palladium, half-life 552 days even before failing its own cointegration test) is
a very different proposition from one reverting in weeks (brent_wti under a both-legs-
agree backwardation regime, 9.5 days in notebook 10a's own Phase 3) — the same
significance test, wildly different practical implications.

**Worked example.** brent_wti's pooled (unconditional) half-life is ~79 days
($\beta \approx -0.0087$); notebook 8's own carry strategy on the *same underlying
commodities* held positions with a comparable multi-week horizon at 21-day rebalance —
half-lives in this range are long enough that per-trade transaction costs, not
signal-strength, become the deciding factor for whether a spread strategy nets positive.

**Pitfalls.** The half-life formula requires $-1 < \beta < 0$ — a $\beta \geq 0$ (no mean
reversion at all) or $\beta \leq -1$ (oscillatory/unstable) has no valid half-life and must
be reported as `None`, not coerced into a number. Half-life is also a *population*
property estimated with real sampling uncertainty from a finite series — a point estimate
of "77 days" from ~2,500 observations carries a wide implicit confidence interval that a
single reported number obscures if not paired with the underlying $t$-statistic.

---

### The commodity inverse-leverage effect

**In one sentence.** In equities, volatility tends to *rise* when price *falls* (the
classic "leverage effect" — a falling stock raises its own debt-to-equity ratio,
mechanically raising equity volatility). Commodities are hypothesised to show the
**opposite** sign: volatility rising when price *rises*, because a price spike signals
scarcity, and scarcity is itself a volatile state.

**The maths.** Estimated here as $\text{corr}(r_t, \sigma_{t+1})$ per product, with a
bootstrap CI. Negative correlation = equity-style leverage effect; positive = inverse
leverage.

**Why it is here.** It is one of the most-cited "commodities are structurally different
from equities" stylised facts (alongside the [contango/backwardation](#contango-and-backwardation)
skew-flip prediction) — and one this notebook can test directly, since GJR-GARCH's own
asymmetry parameter $\gamma$ (see [conditional EVT](07-extreme-value-theory.md#conditional-evt-mcneil-frey-two-stage))
implicitly assumes the *equity* sign unless refit; a wrong-signed $\gamma$ on commodity
data would be fitting the asymmetry term backwards.

**Worked example.** Notebook 8's own leverage-correlation table (Phase 1) found this
prediction only weakly and inconsistently supported across the 16-product panel — most
correlations sat close to zero with wide, mostly zero-including CIs, and where a
significant correlation *did* appear (palladium) it was **negative** (equity-sign), not
the predicted positive inverse-leverage sign. Reported as a genuine disagreement with
the consensus prior, not smoothed over.

**Pitfalls.** A single correlation estimate over a 14-year window can be dominated by a
handful of extreme episodes (2020 COVID crash, 2022 energy crisis) — before trusting a
per-product sign, check whether it is stable across sub-periods or an artifact of one or
two dominant events.

---

### Negative and contaminated settlement rows

**In one sentence.** Not every negative or near-zero price in a raw futures OHLCV feed
is a data error — some are genuine (WTI settled at $-\$37.63$ on 2020-04-20) — but many
are spread/differential settlement values mistakenly carrying an outright contract's
ticker, and the two cases must be told apart by evidence, not by a sign check.

**The maths.** Notebook 8's hygiene rule is two-tier, both relative to each date's
highest-volume contract (the *anchor*, almost always a genuine liquid outright):
(1) **contract-level** — if a contract_id deviates from the anchor by more than 30% on
over half its trading days (with at least 10 days of history to judge), every row of
that contract is junk; (2) **row-level** — for contracts that pass (1), a single day is
still flagged if it deviates from the anchor by >30% *and* its volume is below 50,000
contracts.

**Why it is here.** Volume alone cannot separate the real case from the fake one: CL's
genuine 2020-04-20 crash traded on 8.4% of that day's total CL volume; NG's mislabeled
spread-differential contract (`NG202507`, 2025-05-23) traded on a *comparable* 9.9% of
that day's NG volume. An absolute or relative volume cutoff flags both or neither. The
signal that actually separates them is **persistence**: `NG202507` prints a near-zero or
negative close on 97% of the ~575 days it appears — a differential series mislabeled as
an outright, not an outright having one bad day. CL's contract deviates like this on
0.6% of its ~343 days — one genuine event in an otherwise normally-trading contract.

**Worked example.** Pulled directly from the audit trail (`exact_statistics/raw`) for
one confirmed junk contract (GC201511): its `settlement_price` stat prints exactly 0.0,
while its `trading_session_low/high` and best-bid/best-offer stats print ~$1127-1128,
in line with real gold spot at the time — direct evidence the settlement feed for that
contract is broken, not that gold outright traded near zero.

**Pitfalls.** A rule based purely on `close <= 0` fails both ways: it wrongly keeps
plenty of near-zero *junk* prints that happen to be barely positive, and — worse — it
would wrongly discard CL's real April-2020 print if applied to the outright series
itself rather than to hygiene screening upstream of curve construction.

---

### Seasonal/liquid months vs. nominally listed months

**In one sentence.** A product's `contracts.parquet` listing (every ticker that was ever
issued) is not the same thing as which of those tickers were ever *genuinely traded* —
some products (platinum, palladium) list a contract for every calendar month but only
trade real size on a quarterly cycle, leaving the "in-between" months technically listed
but functionally dead.

**The maths.** A contract-month is treated as liquid if its lifetime total volume
clears a threshold (5,000 contracts in this notebook); months below that threshold are
excluded from the roll sequence entirely, on top of (not instead of) the
`roll_calendar`-vs-`contracts.parquet` membership check.

**Why it is here.** `roll_calendar.parquet` lists an entry for every calendar month for
every product — including seasonal/quarterly-cycle products like the grains, whose real
delivery months (corn: Mar/May/Jul/Sep/Dec) are a strict subset of what's listed.
Rolling into a month with zero real contracts, or one with a token handful of trades
spread over a few days, leaves the front-month series with a hole for that whole month.

**Worked example.** Before this filter, platinum's front-month series was null on 57% of
trading days. The pattern was stark once found: `PL`'s real active months (Jan/Apr/Jul/
Oct) traded total lifetime volume in the hundreds of thousands of contracts; the
"in-between" months (Feb/Mar/May/Jun/Aug/Nov/Dec) traded total lifetime volume in the
tens to low hundreds, over a handful of days each, before going silent for the rest of
that contract's life.

**Pitfalls.** This is a *coarse, contract-lifetime* liquidity filter, deliberately
distinct from a *daily* liquidity screen (minimum volume per day) — applying a daily
screen upstream of continuous-series construction was tried first and created the same
kind of hole for a different reason (a single quiet day on an otherwise-legitimate front
month), which is why the two screens are kept separate: one decides which contract-months
are eligible to be rolled into at all; the other is a downstream per-row filter for
signal/backtest use, never applied to the series-construction step itself.

---

### Structural cash-and-carry arbitrage

**In one sentence.** A delta-neutral trade that holds an asset long in one venue/form
and short in another (spot vs. futures, or spot vs. a perpetual future) in equal dollar
size, profiting from a structural payment or convergence rather than from ranking or
predicting direction — distinct from [carry as a cross-sectional ranking
signal](#carry-basis-trade), which bets on the *relative* size of a payment across many
assets while still carrying full directional exposure to whichever asset it's long or
short.

**The maths.** No new formula beyond [cost of carry](#cost-of-carry) and [funding
rate](#funding-rate): hold $+1$ unit spot and $-1$ unit future/perpetual (dollar-matched),
so the position's P&L from the *underlying's own price move* cancels to (approximately)
zero, leaving only the basis convergence (futures) or the accumulated funding payments
(perpetuals) as the source of return.

**Why it is here.** Notebook 9's external research review found unusually strong,
unusually consistent Tier 1 (peer-reviewed/regulatory) evidence for this category at
institutional scale — the Treasury cash-futures basis trade alone represented roughly
$4 trillion of hedge funds' gross Treasury exposure by late 2025 (Federal Reserve, Office
of Financial Research, Dallas Fed, CFTC sources, `src/results/009_external_research_review.md`)
— a real, large, structurally-motivated (not directional) return source this research
programme has never tested in any form. Notebook 7's own Gate CY tested funding rate only
as a *ranking* signal (§ above) and found the resulting book's own turnover exceeded the
signal it was meant to replace; the structural, delta-neutral version is a different
trade with a different (and untested) cost/turnover profile.

**Worked example.** The Treasury basis trade: short a Treasury futures contract, long a
repo-financed Treasury security deliverable into that future, at (near) zero net exposure
to the level of interest rates — the trade's real risk is not "will rates move" but "will
the futures-implied and cash-market prices actually converge before financing conditions
change," which is precisely why regulators (not just academics) track its scale as a
financial-stability question, not a return-prediction one.

**Pitfalls.** "Delta-neutral" only cancels *price* risk from the underlying, not every
risk in the trade — the Treasury version's real risk is repo-funding-rate and dealer
intermediation-capacity risk (Dallas Fed, `src/results/009_external_research_review.md`
Gate FA discussion), and a crypto spot-vs-perpetual version of the same idea carries
exchange-counterparty risk and the possibility that funding turns negative, reversing the
trade from collecting a payment to making one. A cash-and-carry trade being "structural"
does not mean it is risk-free — only that its risk is a different kind than the
directional bet a ranking-based carry signal still carries.

---

### Market making and inventory risk

**In one sentence.** Continuously quoting both a buy (bid) and sell (ask) price for an
asset, profiting from the spread between them when both sides get filled, while managing
the risk that one side fills much more than the other and leaves an unwanted directional
position (inventory) exposed to the next price move.

**The maths.** The Avellaneda-Stoikov (2008) formulation skews the market maker's quoted
mid-price away from the true mid by an amount proportional to current inventory $q$,
risk aversion $\gamma$, and remaining time-to-horizon: reservation price
$r = s - q\gamma\sigma^2(T-t)$, where $s$ is the observed mid-price and $\sigma^2$ the
asset's variance — a market maker long inventory ($q>0$) quotes *below* the true mid on
both sides, making the ask more attractive to sell into (reducing inventory) and the bid
less attractive to buy into (not adding to it).

**Why it is here.** Notebook 9's external research review lists this among the
structural/mechanical return sources this research programme has never tested at all
(`src/results/009_external_research_review.md`, Gate MM) — and, unusually for that
notebook's shortlist, explicitly could NOT be made testable with this repo's existing
data: every backtest in this programme runs on OHLCV bars, but a market maker's entire
risk (inventory versus the *actual order book*, fill probability at a given quote
distance from mid) cannot be reconstructed from bars at any frequency, no matter how
fine — it requires level-2 (full order-book depth) data this repo has never had.

**Worked example.** A market maker holding a large long BTC inventory after a run of buy
orders should NOT keep quoting symmetric bid/ask around the observed mid — by the formula
above, it should shift its whole quote ladder down, making its ask price closer to the
current mid (encouraging sells that reduce inventory) and its bid price further below the
mid (discouraging further buys) — actively steering its own inventory back toward zero
rather than passively accepting whatever flow arrives.

**Pitfalls.** The model's edge (the spread) is earned from *uninformed* flow (traders who
just want to transact now) but lost to *informed* flow (traders who know something the
quotes don't yet reflect) — a market maker who cannot tell the two apart, or who quotes
through a genuine regime shift without widening, can lose far more on the informed side of
its book than the spread ever earned on the uninformed side, which is why every real
implementation of this model layers volatility- and news-aware quote-widening on top of
the base inventory-skew formula, not covered by the formula alone.
