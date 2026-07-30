# 05 — Regime models

This file assumes [01](01-probability-and-distributions.md)-[04](04-volatility-models.md).
It covers models of a hidden, switching state behind the data — used in notebook 4's
Phase 4 to test whether crypto behaves differently in identifiable "regimes" (e.g.
calm vs. turbulent) and whether that distinction is predictive of anything.

---

### Latent/hidden state

**In one sentence.** A variable the model believes exists and matters, but that you never
directly observe — you only see its *effects* (like which distribution a return seems to
have come from), and have to infer the state itself from those effects.

**The maths.** A latent variable $S_t$ (the state at time $t$) takes one of a fixed set
of values ($1,\dots,k$), and the *observed* data $X_t$ is generated conditional on $S_t$
(its [emission distribution](#emission-distribution)) — but $S_t$ itself is never seen
directly, only inferred from $X_t$.

**Why it is here.** This is the foundational idea behind every model in this file — a
"regime" (calm vs. turbulent market) is exactly a latent state: no bar is ever labeled
"this is regime 2," it is inferred from the pattern of returns.

**Worked example.** BTC's returns are the only thing actually observed; "which volatility
regime is currently active" is a latent state that a
[Gaussian mixture model](#gaussian-mixture-model) or [hidden Markov model](#hidden-markov-model)
infers from the pattern of those returns, never observed directly.

**Pitfalls.** A latent state is a modeling *choice*, not a directly falsifiable fact
about the world in the same way an observed variable is — two different, equally
reasonable choices of how many states to posit ($k=2$ vs. $k=3$) can both fit the data
reasonably well; see [identifiability](02-estimation-and-fitting.md#identifiability) for
the related concern of whether a given state structure is even uniquely determined by
the data at all.

---

### Markov chain

**In one sentence.** A model of a sequence of states where the next state only depends on
the *current* state, not on the whole history of how you got there — a simple, useful
"memorylessness" assumption for how regimes switch over time.

**The maths.** $P(S_{t+1} \mid S_t, S_{t-1}, \dots, S_1) = P(S_{t+1} \mid S_t)$ — the
Markov property. Given the present state, the future is independent of the past.

**Why it is here.** This is the structural backbone of the
[hidden Markov model](#hidden-markov-model) used in notebook 4's Phase 4 — regime
transitions (calm $\to$ turbulent, turbulent $\to$ calm) are modeled as exactly this kind
of memoryless process.

**Worked example.** Under a Markov assumption, knowing "we've been in the turbulent state
for the last 10 bars" tells you nothing extra about whether the *next* bar stays
turbulent, beyond just knowing "we're currently turbulent" — a genuinely restrictive
assumption.

**Pitfalls.** Notebook 4 explicitly found this assumption doesn't hold cleanly: observed
regime durations reject a geometric distribution (the duration pattern a true Markov
chain implies — see [state persistence and expected duration](#state-persistence-and-expected-duration))
at every model and interval tested. This is reported as an honest, expected departure
from the model's own null assumption, not a modeling failure — the Markov chain is the
baseline being tested against, not the claim being made.

---

### Transition matrix

**In one sentence.** A table of probabilities describing exactly how likely the process
is to move from each state to each other state (including staying put) — the complete
description of a Markov chain's dynamics.

**The maths.** A $k \times k$ matrix $A$ where entry $A_{ij} = P(S_{t+1}=j \mid S_t=i)$ —
each row sums to 1 (from state $i$, the process must go *somewhere*, including possibly
staying at $i$).

**Why it is here.** `dist_lib.fit_hmm`'s `A` variable is exactly this matrix, estimated
via [Baum-Welch](#baum-welch); its diagonal entries ($A_{ii}$, probability of staying in
state $i$) directly determine [state persistence](#state-persistence-and-expected-duration).

**Worked example.** A transition matrix with a diagonal entry of 0.9 for the "calm"
state means: given you're calm this bar, there's a 90% chance you're still calm next
bar — a state that, on average, lasts about 10 bars before switching (see the geometric
expected-duration formula below).

**Pitfalls.** A fitted transition matrix's diagonal being close to 1 for every state can
look, superficially, like a strong finding of persistence — but per the
[Markov chain](#markov-chain) entry above, notebook 4 found that even a transition matrix
capturing *more* persistence than the naive threshold baseline still underestimates true
regime persistence (durations remain non-geometric, i.e. more persistent than even the
fitted Markov structure implies).

---

### State persistence and expected duration

**In one sentence.** How long a regime, once entered, tends to last before switching to
another — a direct, interpretable consequence of a Markov chain's transition
probabilities.

**The maths.** If state $i$ has self-transition probability $A_{ii} = p$, the number of
bars spent in state $i$ before switching follows a
[geometric distribution](01-probability-and-distributions.md#geometric-distribution) with
mean duration $\frac{1}{1-p}$.

**Why it is here.** Notebook 4's Phase 4 reports exactly this per model/interval —
HMM-Gaussian's mean state duration (e.g. 6.62 bars at 1h) compared against the naive
threshold baseline's (2.34 bars at 1h) — and tests whether the *actual* observed
durations match what a Markov chain's own geometric-duration implication would predict
([Markov chain](#markov-chain)'s stated caveat: they consistently don't, rejected by KS
at effectively every model/interval).

**Worked example.** HMM-Gaussian's 1.7-2.8x longer mean state duration than the naive
threshold baseline, at every interval, is read in notebook 4's write-up as "real
structure" — states that actually persist, rather than the threshold's near-random 2-3
bar flip-flopping around its own trailing median.

**Pitfalls.** A longer *mean* duration doesn't by itself mean the *distribution* of
durations is well-described by the model — notebook 4 explicitly separates these two
claims, reporting both the (favorable) mean-duration comparison and the (unfavorable,
consistently rejected) geometric-shape test, rather than only the flattering one.

---

### Gaussian mixture model

**In one sentence.** A model where the data is assumed to come from a blend of several
different normal distributions, each representing a distinct regime — but *without* any
notion of time or persistence: it says "how much of each regime is present overall," not
"when."

**The maths.** Exactly a [mixture distribution](01-probability-and-distributions.md#mixture-distribution)
where every component is normal:
$f(x) = \sum_{j=1}^k w_j \cdot \mathrm{Normal}(x \mid \mu_j, \sigma_j^2)$, fit via
[EM](02-estimation-and-fitting.md#em-algorithm).

**Why it is here.** `dist_lib.fit_gmm_em` is exactly this, used both as a static
Phase 1 descriptive fit (two components: low-vol/high-vol) and as the *emission*
distribution choice inside [hidden Markov models](#hidden-markov-model) in Phase 4.

**Worked example.** Notebook 4's Phase 1 GMM ($k=2$) at 1h found weight $\approx 0.79$ on
a low-variance component and $\approx 0.21$ on a high-variance component (roughly 10x
larger variance) — a static snapshot of "how much of the time is BTC calm vs. turbulent,"
with no statement about clustering in time (that's what the HMM adds).

**Pitfalls.** A GMM alone cannot distinguish "regimes alternate in bursts over time" from
"regimes are scattered randomly with no temporal pattern" — both would produce an
identical mixture fit, since a GMM has no time-ordering built in at all. This is exactly
why the [hidden Markov model](#hidden-markov-model) below exists: it adds a
[transition matrix](#transition-matrix) on top of essentially the same per-state emission
idea.

---

### Hidden Markov model

**In one sentence.** A Gaussian mixture model with time added: instead of just saying
"here's the overall blend of regimes," it says "here's how the process actually moves
between regimes over time" — combining a Markov chain (governing *when* states switch)
with per-state emission distributions (governing *what* the data looks like in each
state).

**The maths.** A latent state sequence $S_1, S_2, \dots$ following a
[Markov chain](#markov-chain) with [transition matrix](#transition-matrix) $A$, where
each observed $X_t$ is drawn from state $S_t$'s own
[emission distribution](#emission-distribution) — the combination lets the model both
describe "what each regime looks like" and "how regimes actually persist and switch over
time," unlike a plain [Gaussian mixture](#gaussian-mixture-model) which only does the
former.

**Why it is here.** `dist_lib.fit_hmm` is this model, fit via
[Baum-Welch](#baum-welch), with either Gaussian or Student-t emissions
(`emission="gaussian"` or `"t"`), and is the model showing the clearest improvement over
the naive threshold baseline in notebook 4's Phase 4 (longer, more realistic state
persistence).

**Worked example.** HMM-Gaussian at 1h found a mean state duration of 6.62 bars (vs. the
naive threshold's 2.34) and comparable-or-better volatility discrimination (Kruskal-Wallis
p effectively 0) — real, useful structure, though notebook 4's write-up is careful to
note no formal significance test was built comparing HMM against the threshold baseline
head-to-head, so it's reported as "suggestive," not a certified winner.

**Pitfalls.** Fitting more states ($k=3$ instead of $k=2$) will generally fit the
training data at least as well by construction (more flexibility) — the same
[overparameterization](02-estimation-and-fitting.md#overparameterization) caution that
applies to distribution fitting generally applies here too; a bigger $k$ isn't
automatically a better regime model.

---

### Baum-Welch

**In one sentence.** The specific EM-style algorithm used to fit a hidden Markov
model's parameters (transition probabilities and emission distributions) from data,
without ever observing the hidden state sequence directly.

**The maths.** Alternates a forward-backward pass (computing, for every $t$, the
probability of being in each state given the *entire* observed sequence — both past and
future) with an update step that re-estimates the transition matrix and emission
parameters using those probabilities as soft weights, exactly analogous to
[EM](02-estimation-and-fitting.md#em-algorithm)'s E-step/M-step structure.

**Why it is here.** `dist_lib.fit_hmm`'s core loop (`alpha`, `beta`, `gamma`, `xi_sum`
variables) is a from-scratch implementation of exactly this — forward pass (`alpha`),
backward pass (`beta`), combined into state probabilities (`gamma`) and
transition-probability estimates (`xi_sum`), iterated until the parameters stop changing
meaningfully.

**Worked example.** The backward pass (`beta`) in `fit_hmm` genuinely looks at *future*
observations relative to time $t$ — this is legitimate and necessary for *fitting* the
model's own parameters (the fit itself is done once, on an already-closed training
window, so no live forecasting causality is violated), but it is exactly why the
resulting smoothed state probabilities (`gamma`) must never be reused as a live,
tradeable state estimate — see
[forward filtering vs. smoothing](#forward-filtering-vs-smoothing-and-why-smoothing-is-lookahead)
below for why.

**Pitfalls.** Like EM generally, Baum-Welch converges to a local, not necessarily
global, maximum of the likelihood — the specific initialization scheme used
(`fit_hmm`'s evenly-spaced-quantile starting means) is a deliberate choice to reduce the
chance of a degenerate local optimum, not a guarantee against one.

---

### Forward filtering vs. smoothing (and why smoothing is lookahead)

**In one sentence.** Two different questions about a hidden state's probability at time
$t$: **filtering** asks "given everything up to and including $t$, what's the probability
of each state?" (causal, usable live); **smoothing** asks "given the *entire* dataset,
including everything after $t$, what's the probability?" (non-causal, only valid for
after-the-fact analysis or model-fitting).

**The maths.** Filtering: $P(S_t \mid X_1,\dots,X_t)$. Smoothing: $P(S_t \mid
X_1,\dots,X_T)$ for the full series length $T$, using data both before and after $t$.
Smoothing is always at least as accurate as filtering (it uses strictly more
information) — which is exactly the problem when the goal is a real-time, tradeable
estimate: that extra accuracy comes from information you would not actually have had at
the time.

**Why it is here.** This is a named, explicit guardrail in notebook 4's Phase 4:
"filtered, never smoothed" — `dist_lib.hmm_filter_step` implements only the one-step
forward recursion, applied to an *already-fit* (frozen, trained-on-past-data-only) model.
Baum-Welch's own smoothed `gamma` (computed during the fitting process, using the full
training window in both directions) is explicitly discarded after fitting and never used
as a live state estimate.

**Worked example.** Using smoothed state probabilities to trade would mean, e.g., "knowing"
at bar 100 that the regime is about to switch at bar 105, because the smoothing pass
already looked at bars 101-110 to compute bar 100's probability — a direct violation of
[causality](08-research-methodology.md#lookahead-bias-leakage), even though it happens
inside a superficially reasonable-looking statistical procedure (Baum-Welch) rather than
an obvious same-bar leak.

**Pitfalls.** The distinction is subtle exactly because both filtering and smoothing use
the *same* fitted model parameters — the leak isn't in *how the model was fit* (fitting
on a closed past window is fine), it's in *which state-probability calculation* gets used
live. This is precisely why `hmm_filter_step` is a clearly separate function from the
fitting routine, so the causal, live-usable computation can't be accidentally swapped for
the smoothed one.

---

### Viterbi

**In one sentence.** An algorithm for finding the single *most likely entire sequence* of
hidden states, given all the observed data — a different question from "what's the
probability of each state at each time," and, like smoothing, inherently non-causal if
used across the whole dataset.

**The maths.** Finds $\arg\max_{s_1,\dots,s_T} P(S_1=s_1,\dots,S_T=s_T \mid
X_1,\dots,X_T)$ via dynamic programming — efficiently searching over all possible state
sequences without literally enumerating every one.

**Why it is here.** Mentioned for completeness as a standard HMM tool a reader may
encounter elsewhere; **not used** in this repo's own Phase 4 work, which relies on
[forward filtering](#forward-filtering-vs-smoothing-and-why-smoothing-is-lookahead) (a
live, causal, per-bar probability) rather than a single best full-sequence reconstruction.

**Worked example.** Not applicable — not used in this codebase's actual driver scripts.

**Pitfalls.** Like smoothing, a Viterbi path computed over an entire historical dataset
uses future information to label past states — reasonable for retrospective
"characterize what regime the market was probably in on this date" analysis, but not a
live, causal signal; this repo avoids it for exactly that reason.

---

### Emission distribution

**In one sentence.** The distribution of the *observed* data, given which hidden state
is currently active — the "what does state $j$ look like" half of a regime model,
paired with the [transition matrix](#transition-matrix)'s "how do states switch" half.

**The maths.** $P(X_t \mid S_t = j) = f_j(X_t)$ — a separate distribution (with its own
parameters) for each state $j$.

**Why it is here.** `dist_lib.fit_hmm`'s `emission` argument (`"gaussian"` or `"t"`)
picks exactly this per-state distribution's family — a Gaussian emission means each
state's returns are modeled as normal (with that state's own mean/variance); a
Student-t emission (fixed $\nu$, estimated once and held constant across states, per the
docstring's own documented simplification) lets each state's returns be fat-tailed.

**Worked example.** A 2-state HMM's emission distributions might be
$\mathrm{Normal}(0, 0.001^2)$ for the "calm" state and $\mathrm{Normal}(0, 0.01^2)$ for
the "turbulent" state — the same return value (say, $-0.005$) is far more "ordinary"
under the turbulent state's emission than the calm one's, which is exactly the
information the filter uses to infer which state is more likely active.

**Pitfalls.** `fit_hmm`'s Student-t emission's $\nu$ is estimated once, globally, and held
fixed across states and across the EM iterations for the location/scale update (a
documented, explicit simplification, not a full t-MLE) — a genuine trade-off for
tractability, and worth remembering when comparing HMM-Gaussian and HMM-t results as not
perfectly symmetric in how thoroughly each was fit.

---

### Posterior state probability

**In one sentence.** The model's own best current guess, as a probability for each
possible state, of which hidden state is active — not a hard yes/no assignment, but a
full probability distribution over the possibilities (e.g. "73% calm, 27% turbulent"),
updated as new data arrives.

**The maths.** $P(S_t = j \mid \text{data available})$ for each state $j$, summing to 1
across all $j$ — a "posterior" in the sense that it's the state distribution *after*
(post-) folding in the observed data, as opposed to the model's prior (before-any-data)
assumption of each state's baseline probability.

**Why it is here.** `hmm_filter_step`'s return value is exactly this — a live,
[filtered](#forward-filtering-vs-smoothing-and-why-smoothing-is-lookahead) posterior over
states at each bar, computed causally from only past and current data.

**Worked example.** `gmm_posterior` computes the same idea for a static Gaussian mixture
(not time-indexed): given one observed return value, how much of the mixture's total
density at that point came from each component — read as "how likely is each regime,
given just this one observation."

**Pitfalls.** A posterior probability close to 50/50 between two states signals genuine
model uncertainty about which regime is active — this is meaningfully different from
(and less actionable than) a confident 95/5 posterior, even though both are valid
"filtered state estimates"; any downstream use of a posterior probability should account
for this rather than collapsing it to a hard state label without regard to confidence.
