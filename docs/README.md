# docs/ — a from-scratch glossary for this research programme

This folder explains, from first principles, every technical term used across this repo's
research programme (`src/results/*.md`, `src/research/*.ipynb`) and its production risk
engine (`src/risk/`). It assumes **high-school algebra and nothing else** — no prior
statistics, no prior probability, no prior finance. If you have never seen the word
"distribution" used in a mathematical sense, start at the top of
`01-probability-and-distributions.md` and read in order; by the end of
`07-extreme-value-theory.md` you should be able to follow the argument in
`src/results/005_tail_risk_evt.md` (why Student-t innovations beat normal ones on tail
calibration) without leaving this folder.

Files `01`–`07` are the statistical core, `08`–`09` cover methodology and market data, and
`10` is an operator document for the running risk engine rather than a teaching file.

## How to read this

- **Read in file order the first time.** Each numbered file only depends on files with a
  smaller number. `04-volatility-models.md` uses terms defined in `01` through `03`;
  it does not use anything from `05` onward.
- **Every term is defined before it is used.** If a word in an entry looks technical and
  isn't a link, that is a bug in this documentation — file it as such rather than
  guessing at the meaning.
- **`GLOSSARY.md` is the flat lookup**, not a teaching document: one line per term, in
  alphabetical order, linking to the full entry. Use it once you already know the term
  and just want to jump to it, or to check whether a word you've hit in a notebook is
  defined here at all.
- **Every entry cites this repo.** The "why it is here" section of each entry names the
  actual file and the actual number that made the concept matter to this research
  programme, not a textbook example. That is the whole reason this folder exists instead
  of a link to Wikipedia.

## Reading order (teaching path)

1. [01 — Probability and distributions](01-probability-and-distributions.md) — what a
   distribution *is*, and the specific shapes (normal, Student-t, generalized Pareto, ...)
   this programme fits to data.
2. [02 — Estimation and fitting](02-estimation-and-fitting.md) — how you go from raw data
   to a fitted distribution's parameters, and what can go wrong doing it.
3. [03 — Statistical inference](03-statistical-inference.md) — how you decide whether a
   fitted difference between two things is real or noise.
4. [04 — Volatility models](04-volatility-models.md) — models of how much a return series
   moves, not which direction (GARCH, HAR-RV, range estimators, ...).
5. [05 — Regime models](05-regime-models.md) — models of a hidden, switching state behind
   the data (Gaussian mixtures, hidden Markov models).
6. [06 — Scoring rules and calibration](06-scoring-rules-and-calibration.md) — how a
   forecast (of a whole distribution, not just a number) is graded, and what "well
   calibrated" means precisely.
7. [07 — Extreme value theory](07-extreme-value-theory.md) — the mathematics of tails
   specifically: what happens far from the middle of a distribution, and why that needs
   its own machinery instead of just using a fatter-tailed family for everything.
8. [08 — Research methodology](08-research-methodology.md) — the discipline (holdouts,
   lookahead bias, pre-registration, multiple testing, ...) that keeps a backtest from
   lying to you. Read this even if you skip the maths in 01-07 — it's the one file that
   explains *why* this whole programme is run the way it is.
9. [09 — Market data and microstructure](09-market-data-and-microstructure.md) — what an
   OHLCV bar actually is, and the crypto- and futures-specific facts (perpetual futures,
   funding rates, no overnight gap, frozen-price bars, roll adjustment, spreads and
   cash-and-carry trades) that shape every other file's "why it is here."
10. [10 — The risk engine](10-risk-engine.md) — an **operator document**, not a teaching
    file: what `src/risk/` computes, its data-cleaning contract, how a density family was
    chosen per product, how the calibration monitor works, and what the engine has and
    has not been validated for. Read `06` and `07` first for the statistics behind it.

[GLOSSARY.md](GLOSSARY.md) — every term above, one line each, alphabetical, linked.

## Notation conventions used throughout `docs/`

These symbols recur in nearly every entry, so they are defined once, here, rather than
re-defined in each file.

- $x, y, r$ — a single observed data value. In this programme $r$ is almost always a
  **log return** (see [log return](09-market-data-and-microstructure.md#log-return-and-why-logs)).
- $x_t, r_t$ — the subscript $t$ is a **time index**: bar number $t$ in a sequence, so
  $r_t$ means "the return observed in bar $t$" and $r_{t-1}$ means "the bar immediately
  before it." Time only ever runs forward in this documentation: anything computed "at
  time $t$" is only allowed to use $r_1, \dots, r_t$ (see
  [causality / lookahead bias](08-research-methodology.md#lookahead-bias-leakage)).
- $n$ — the number of observations in whatever sample is under discussion.
- $\theta$ (Greek letter "theta") — a generic stand-in for "some parameter of a model,"
  used when the entry is making a point that applies to *any* parameter, not a specific
  named one like $\omega$ or $\nu$.
- $\hat{\theta}$ ("theta hat") — an **estimate** of $\theta$ computed from data, as
  opposed to $\theta$ itself, which is the true (unknown, unobservable) value. See
  [parameter vs estimate vs estimator](02-estimation-and-fitting.md#parameter-vs-estimate-vs-estimator).
- $\mathbb{E}[\cdot]$ — "expectation": the long-run average value of whatever is inside
  the brackets, if you could repeat the random process that generated it infinitely many
  times. See [expectation](01-probability-and-distributions.md#expectation).
- $\mid$ (a vertical bar) — "conditional on" / "given." $P(A \mid B)$ reads "the
  probability of $A$, given that $B$ is true" — a different, generally smaller-scoped
  question than the plain probability of $A$. Used constantly from
  [conditional vs unconditional variance](04-volatility-models.md#conditional-vs-unconditional-variance)
  onward.
- $\sum$ (capital Greek sigma) — "sum": $\sum_{i=1}^n x_i$ means $x_1 + x_2 + \dots + x_n$.
- $\prod$ (capital Greek pi) — "product": $\prod_{i=1}^n x_i$ means $x_1 \times x_2 \times
  \dots \times x_n$. Note this is unrelated to the number $\pi \approx 3.14159$, which
  also appears in this documentation (e.g. the normal density) — same-looking letter,
  different meaning, distinguished by context.
- $\log(\cdot)$ — natural logarithm throughout (base $e$), never base 10, matching
  `numpy`'s `np.log` and every formula in this repo's code.
- $\hat{\ }$ over a Greek letter (e.g. $\hat{\alpha}$, $\hat{\nu}$) — always means
  "the fitted/estimated value of this parameter," same convention as $\hat{\theta}$
  above.

## A note on symbols vs. code

Wherever a formula uses a Greek letter that also appears as a variable name in this
repo's code, the entry says so explicitly: for
example $\omega$ in [GARCH(1,1)](04-volatility-models.md#garch11) is exactly the `omega`
argument in `dist_lib._garch_negloglik`. You should be able to hold these docs next to
the source and map every symbol to a variable one-to-one.
