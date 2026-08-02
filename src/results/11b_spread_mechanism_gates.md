# Notebook 11b — Spread Mechanism Gates: Results Summary

Seven pre-registered gates (`phase_6_11a_results.json`, committed before this notebook ran and
not edited since), all costed backtests on this repo's own spread series and its own, stricter
cost model (`commod_lib8.round_turn_cost_per_contract`). **All seven return a fired=False
verdict.** Every comparison here is internal — the pre-declared trading rule vs 10b's own
continuous benchmark, sign-flipped vs unconditional, screen-inclusive vs screen-exclusive,
vol-adaptive vs static stop, reentry-swept vs baseline — never a validation of the external
programme's absolute reported numbers, which 11a's Phase 4 reproduction already found diverge
materially on every axis. DSR trial counts total **65** across the seven gates, cross-checked
line-by-line against `phase_6_11a_results.json`'s pre-registered breakdown (Phase 7,
`phase_7_11b_results.json`): every count matches exactly, none shrunk.

## Gate TS: the discrete-trade structure does not explain Gate SP's weak Sharpe — it explains a worse one

The pre-declared trading rule (entry/exit z-thresholds, per-spread ATR stops, fixed-fractional
sizing, suppression filters, gated reentry), run on the five live spreads under this repo's
costs, produces a **negative** Sharpe at every origin offset (−0.165 / −0.185 / −0.180 / −0.209).
10b's own continuous Gate SP book, restricted to the *same five spreads* and the *same costs*,
runs **+0.552 to +0.555** at every offset — a large, one-sided gap. The paired block-bootstrap
95% CI on (structured − continuous) is **[−0.520, −0.124]** in fixed-notional return units,
excluding zero in the direction that says the discrete/stopped packaging is *worse*, not better.
**Gate TS does not fire, and it fails in the opposite direction sec 0.1 hypothesized**: this
repo's own continuous, always-on position — the one thing sec 0.1 called strictly inferior to
their mechanism — outperforms their own pre-declared mechanism by a wide, bootstrapped margin on
this repo's data and costs.

**Gate TS-S makes this sharper, not muddier.** Disabling only the stop (keeping every other
piece of the discrete-trade packaging) turns the structured book's Sharpe *positive* (+0.304,
n=107 trades, zero stop exits by construction) — better than the stopped variant, but *still*
significantly below the continuous benchmark (paired CI **[−0.468, −0.092]**, still excluding
zero in the same direction). Because Gate TS itself did not fire, Gate TS-S's own firing
condition (`Gate TS fires AND the stop-disabled variant fails`) cannot be met and it is recorded
as not fired by construction — but the diagnostic itself is informative: the stop is A drag here,
just not the WHOLE story. The remaining gap between the stop-disabled discrete book (+0.304) and
the continuous book (+0.552-0.555) is attributable to the entry/exit thresholds and
trade-boundary discipline themselves, not the stop. **This is the opposite of sec 0.1's own
worked hypothesis** ("their mechanisms are better than ours; our statistics are better than
theirs") — on this repo's own data and costs, the packaging is a net negative at every layer
tested, and the plainest-possible always-on position remains the best-performing book in this
notebook.

## Gate BF: reversing mild backwardation makes the book monotonically worse, not better

Tested on Gate SP's own 16-spread calendar universe (a declared, non-cherry-picked pooled list),
sign-flipping only the entries opened in the mild-backwardation bucket (`0 < carry_ratio < 0.5`
on this repo's own sign convention — see the resolved-sign note below) against three storage
constants, all reported:

| storage | Sharpe by offset (0/7/14/21) | n trades | vs unconditional (0.236/0.186/0.188/0.182) |
|---|---|---:|---|
| unconditional (no flip) | 0.236 / 0.186 / 0.188 / 0.182 | 436 | — |
| LOW | 0.200 / 0.139 / 0.142 / 0.132 | 378 | worse at every offset |
| **MID (headline)** | −0.146 / −0.208 / −0.206 / −0.216 | 378 | worse at every offset, sign-flipped |
| HIGH | −0.185 / −0.248 / −0.245 / −0.256 | 377 | worse still |

The degradation is monotone in the storage assumption (LOW → MID → HIGH), and even LOW never
exceeds the unconditional book at any offset. **Gate BF does not fire — flipping the signal in
this bucket hurts, and hurts more the more contango-cost is assumed.** The trade count also
drops ~13% (378 vs 436, storage-independent), failing sec 0.3's own anti-throughput-collapse
check on its own terms: flipping direction changes which trades get stopped out
(`adverse = (value − entry_value) × −direction` is direction-dependent), so a pure sign flip is
*not* throughput-neutral in this engine the way it is assumed to be — a mechanical finding worth
recording for any future proposal that reverses rather than filters a signal. Gate BF-X
(per-spread improvement, headline storage) finds only 4 of 16 spreads individually improved,
short of the ≥3-of-a-different-kind bar its own criterion implies once BF itself has already
failed; **BF-X does not fire.**

**Sign convention, resolved.** 11a's Phase 1 flagged that `spread_lib11.carry_ratio`'s literal
formula evaluates positive in backwardation and negative in contango on this repo's own
leg1-front `value` convention — the opposite of the external repo's own "+1 at the contango
ceiling" description. Verified again here on the full calendar universe: applying their bucket
boundaries requires negating our raw ratio first (`c_corrected = −carry_ratio`), i.e. their
"mild backwardation" (−0.5 < c_corrected < 0) is `0 < carry_ratio < 0.5` on our data. This
notebook uses the corrected mapping throughout; the result above is not an artifact of the sign
confusion.

## Gate SCR: the ADF screen cannot beat its own absence, and does not survive

Screen-inclusive (4 ADF-passing live spreads) and screen-exclusive (+`kc_chicago_wheat`,
`gc_cal_m2m3`, `es_calendar`, resolving both named cross-repo conflicts) books are, to three
decimal places, **the same book** at every offset (Sharpe −0.165 vs −0.164 at offset 0; paired
CI **[−0.0002, +0.00005]**, a needle's width around zero). This is because `kc_chicago_wheat`,
`gc_cal_m2m3` and `es_calendar` contribute almost no trades under the pre-declared risk rule (see
Gate 4.3 below) — the screen's presence or absence is close to observationally void for this
particular parameterization. **Gate SCR does not KEEP the screen** — sec 4.2's own bar ("a screen
that cannot beat its own absence does not survive") is failed on a technicality (the screen
barely matters either way, not that dropping it helps), which is itself worth stating plainly:
this notebook cannot currently distinguish whether the ADF screen is good, bad, or simply inert
under this specific trading rule and universe.

## Gate VA: the vol-adaptive stop moves in the right direction, but not far enough to clear the bar

Scaling `stop_atr_mult` by 0.75×–1.25× against a rolling realized-vol percentile improves both
Sharpe (−0.111 to −0.157 across offsets, vs control's −0.165 to −0.209) and max drawdown
(−1.63% vs control's −1.88%) at every single offset — directionally exactly what sec 2.1's
reopened, corrected-data finding anticipated. But the improved Sharpe never crosses zero, and the
paired CI **[−0.004, +0.013]** still straddles it. **Gate VA does not fire.** Unlike Gate BF, this
is a genuinely close, informative near-miss rather than a reversal: the mechanism helps on both
axes sec 5's three-way risk gate cares about, just not by enough, on this universe, to clear a
Sharpe-positive-every-offset bar that the underlying book was never going to clear anyway (recall
the control itself is Sharpe-negative at every offset here).

## Gate RE: the reentry-grid sweep — an unreachable DSR at its full, honest denominator

The full 3×3×4 = 36-trial grid's best non-baseline cell (`half_life_max=30, adf_pmax=0.20`)
improves on the baseline (45, 0.10) at every offset (−0.114 / −0.134 / −0.129 / −0.160 vs
baseline's −0.165 / −0.185 / −0.180 / −0.209) and its paired CI **[−0.000004, +0.0095]** sits a
hair's width from excluding zero on the positive side — but the best cell's own Sharpe is still
negative at every offset, so the "positive every offset" leg fails outright, and the resulting
DSR at n_trials=36 is **0.0138**, nowhere near the 0.95 bar. **Gate RE does not fire.** Per sec 9's
own binding instruction, this n_trials=36 is reported in full and not reduced because the grid
turned out unreachable — that unreachability, at its honest denominator, is itself the finding,
in the same tradition as Gate SP's own DSR=0.562 in 10b.

## Sec 4.3: `es_calendar` and `gc_cal_m2m3` are trade-starved under the pre-declared risk caps, not merely under-tested

Neither of the two named cross-repo-conflict spreads is meaningfully tradeable under this
notebook's pre-declared parameterization, and the reason is mechanical, not statistical:
`max_single_name_pct=12%` of the assumed `$1,000,000` `START_EQUITY` caps notional per contract
at `$120,000`, while `es_calendar`'s median per-contract notional is **$196,275** (within the cap
on only 0.2% of dev-window bars) and `gc_cal_m2m3`'s is **$147,870** (9.2% of bars) — `qty` floors
to zero almost everywhere regardless of `risk_pct`, which this notebook confirmed directly by
re-running both spreads at `risk_pct=0.15` (5× the default) with **zero change** in trade count
(`es_calendar`: 0 trades at both 0.03 and 0.15; `gc_cal_m2m3`: 5 trades at both). The
drawdown-matched `risk_pct` search sec 4.3 asked for is reported for completeness but is not a
meaningful edge-vs-edge comparison for either spread given this — the single-name cap, not
`risk_pct`, is the binding constraint, and this notebook does not relax it to work around a
problem that is itself part of the pre-declared trading rule. This also means neither of their
own two conflicting claims about `gc_cal_m2m3` (v4 §14.2's +0.56/+0.62 mean ATR vs their own
atlas's reported −0.04 over 224 pooled trades) can be adjudicated from this side — that
disagreement lives entirely inside data this repo does not have. What this notebook independently
establishes is a *different*, prior question: under the pre-declared position-sizing rule and a
plausible $1M capital assumption, both spreads are barely tradeable at all, which is itself a
reason their v4 §14.2 "standalone, edge-vs-leverage" framing may not transfer cleanly across
repos with different assumed capital bases.

## Sec 5: Gate VS's drawdown convention, reconciled

10b's Gate VS (vol-scaled carry) reported a max drawdown of −5.41 in continuously-compounded
log-return cumsum units — `exp(−5.41)−1 ≈ −99.55%` of peak once exponentiated, which this
notebook confirms is the correct reading of that published number (not a bug), while also
building the capital-bounded, fixed-notional alternative sec 5 asked for:
`equity_t = max(0, equity_{t-1} + simple_return_t × start_equity)`, with an absorbing floor (an
account that hits zero stays at zero, rather than reviving whenever the arithmetic cumsum later
recovers). **Sharpe and DSR are unchanged by this recomputation** (recomputed Sharpe 1.16478 vs
published 1.16466 — a ~1e-4 relative difference from non-bit-deterministic rolling/ranking
internals, not from this notebook's own methodology; both round to the same three significant
figures). The fixed-notional reading is materially different in kind, not just degree: the
account is wiped out at bar 49 of 4488 (2010-08-25) and never recovers under the absorbing floor,
giving max drawdown −100.0% outright, worse than the already-extreme log-cumsum reading. This is
not a contradiction — it is the concrete illustration of *why* the two conventions diverge this
much. A daily-rebalanced, always-reinvested portfolio's real-world equity path is genuinely
closer to the compounding reading (that IS how a live account marks itself to market daily,
which is why it survives past bar 49); the fixed-notional convention this notebook's own
episodic single-trade spread bookkeeping uses everywhere else produces a qualitatively different,
more pessimistic answer when applied instead to a strategy that re-levers every single day. Both
numbers are reported here as labelled hypothetical recomputations of an already-published 10b
result, neither replacing 10b's own number.

## The three-way risk gate, all seven 11b books, offset 0

| gate | Sharpe | max drawdown | return/drawdown | fires |
|---|---:|---:|---:|---|
| TS (structured, stopped) | −0.165 | −1.88% | −0.57 | no |
| TS-S (structured, stop-disabled) | +0.304 | −1.83% | +1.81 | no (TS didn't fire) |
| BF (mid storage, sign-flipped) | −0.146 | −5.80% | −0.44 | no |
| SCR (screen-inclusive) | −0.165 | −1.88% | −0.57 | no |
| VA (vol-adaptive stop) | −0.111 | −1.63% | −0.62 | no |
| RE (best reentry cell) | −0.114 | −1.57% | −0.61 | no |

No book here clears Sharpe > 0.5, so the institutionally-fundable flag is moot for all seven —
none is even in range of the tradeable-alpha gate, let alone the fundable one. Consistent with
sec 5's own worked example (conviction-sizing's return-CI-clearing-but-correctly-rejected result
in 10b): had any of these cleared a bootstrap CI on Sharpe alone without this table, that would
have been the wrong basis for a verdict. None did, so the three-way table here is confirmatory
rather than load-bearing — but is reported in full per sec 5's standing instruction regardless.

## What this notebook establishes, plainly

**Fifteen-plus gates across ten prior notebooks fired none; this notebook's seven fire none
either**, extending the run to twenty-two-plus. But sec 11's own stated stakes for this specific
notebook were higher than a routine extension: Gate TS was "the first proposition in this
programme with a specific, mechanically-identified reason to expect a positive result," and Gate
BF "the first with independent out-of-sample support from a separate codebase." Both came back
null against a properly paired control — and both came back null in the *informative* direction
sec 11 flagged as the live alternative: **the external book's advantage is universe selection and
parameterization choices fitted on their own data, not a portable mechanism**, which is exactly
what this repo's methodology is positioned to distinguish and theirs is not. The continuous,
un-stopped, un-structured position this whole programme has been implicitly comparing against
since Gate SP remains, on this repo's data and costs, the best-performing book in this notebook.

Machinery: `src/research/tmp/run_phase_{0..7}_11b_*.py` (Phase 0 Gates TS/TS-S; Phase 1 Gates
BF/BF-X; Phase 2 Gate SCR; Phase 3 Gate VA; Phase 4 Gate RE; Phase 5 sec 4.3 standalone
diagnostics; Phase 6 Gate VS drawdown reconciliation and the three-way summary; Phase 7 the final
gate table cross-checked against 11a's pre-registration), extending `spread_lib11.py`
(per-bar conditional sign-flip masks and vol-adaptive stop-multiplier scaling, both added to
`simulate_single_spread`/`simulate_book` this notebook, unit-tested in `tests/test_spread_lib11.py`).
The holdout (2025-01-01 to 2026-07-28) remains untouched and unspent; its reduced independence
for commodity-spread strategies specifically, disclosed in 11a, is unchanged by this notebook.
