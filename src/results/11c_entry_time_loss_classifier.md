# Notebook 11c — Entry-Time Loss Classifier: Results Summary

Gate LC (`phase_6_11a_results.json`'s pre-registration, committed before this notebook ran
and not edited since — cross-checked verbatim in Phase 2, not re-typed by hand):
**"a classifier trained walk-forward achieves out-of-sample AUC > 0.60 on the stop-exit
label AND a book that suppresses its top-decile predicted-loss entries beats the
unsuppressed book on the three-way risk gate."** DSR `n_trials=4` (four origin offsets, one
pre-declared feature set, one model class — none swept). **Gate LC does not fire.** The
AUC leg clears its bar at every offset by its literal, point-estimate text; the suppression
leg needs all four offsets to clear (this notebook's own reading of the gate's "every
offset" convention, applied consistently with the rest of the programme) and only 3 of 4
do — so the gate's own `AND` is not satisfied.

## The trade log

11a's own 57-trade control book (`run_phase_4_11a_control_book.py`) is not persisted to a
committed file — it is regenerated, not stored — so Phase 0 reruns the identical
pre-declared trading rule (same five live spreads, same dev window, same cost model, same
`TradingRuleParams`) to reproduce it exactly: **57 trades, 23 stop-exits, 34 zscore-exits**,
matching 11a Phase 4/Phase 5's own counts. Every feature attached to a trade — 15 in total
(entry |z|, corrected carry ratio at MID storage with real FRED `DFF` financing, realized-vol
percentile, ADF t-stat, rolling half-life, half-life sub-period stability fraction,
full-sample in-band flag, roll-window proximity, 60-day leg correlation, variance-ratio
z-stats at q=5 and q=20, Hurst exponent, spread-level percentile within its own trailing
range, and 5-/20-day pre-move in ATR units) — is computed from data strictly at or before
the trade's own entry bar, using a 252-day trailing window feeding the same primitives 11a
built (`rolling_adf_stat`, `rolling_half_life`, `rolling_stability`, `variance_ratio`,
`hurst_exponent`). 55 of 57 trades have every feature finite (the earliest 2 trades predate
enough history for some rolling statistics); those 2 are dropped from the classifier rather
than imputed, since imputation would quietly change what "known at entry" means.

## Gate LC, leg 1: the classifier clears its bar, but the bar alone should not be trusted

No walk-forward classification infrastructure existed in this repo — `research.py`'s own
`walk_forward_run`/`batch_train_reg` are regression-only (hardcoded `MSELoss`, built around a
continuous-return trading rule, no AUC anywhere). This notebook reuses `research.py`'s
`walk_forward_splits` unchanged for fold indices (an anchored/expanding window, exactly the
no-lookahead discipline `2_walk_forward_multi_asset.ipynb` already established) applied over
trades rather than daily bars — 55 trades over 14.5 years is this classifier's native
sampling unit — and adds one new, minimal primitive, `spread_lib11.roc_auc_score` (the
Mann-Whitney U / rank-sum identity; no `sklearn` in this repo's environment). The model class
is logistic regression (`torch.nn.Linear` + `BCEWithLogitsLoss`, strong L2 via
`weight_decay=1.0`, chosen because 15 features against ~30-trade train folds is a genuine
n≪p regime). Four origin offsets (0-3 trades, this notebook's own discretization of the
programme's day-offset convention, since the sampling unit here is trades, not bars) each
give an anchored walk-forward grid (train 30 trades, test 5, step 5):

| offset | n folds | n OOS trades | stitched OOS AUC |
|---|---:|---:|---:|
| 0 | 5 | 25 | **0.757** |
| 1 | 4 | 20 | **0.667** |
| 2 | 4 | 20 | **0.738** |
| 3 | 4 | 20 | **0.707** |

All four clear the pre-registered ">0.60" bar on its literal, point-estimate text — **the
AUC leg fires.** But per-fold AUCs are themselves exactly the small handful of values a
5-trade test fold can produce (1.0, 0.75, 0.5, 0.25, 0.0 — there is no room for anything
else with `n=5`), and inspecting the actual predicted probabilities behind a perfect
1.0-AUC fold shows values clustered in **0.41-0.56** — barely distinguishable from a coin
flip in absolute terms, correctly *ranked* only by tiny margins. Bootstrapping the
already-collected stitched out-of-sample predictions (resampling trades, not refitting) at
each offset gives a 95% CI that **includes 0.5 at every single offset** (offset 0: point
0.757, CI [0.487, 0.960]; offset 1: point 0.667, CI [0.389, 0.920]; offset 2: point 0.738,
CI [0.458, 0.956]; offset 3: point 0.707, CI [0.374, 0.944]). **The AUC leg fires by the
pre-registered criterion's literal text, and that verdict is reported honestly — but the
same scrutiny this programme applies to every other small-sample result says this point
estimate alone is not distinguishable from chance on 20-25 out-of-sample trades.** This is
disclosed as a fragility caveat on a leg that mechanically fires, not smoothed over.

## Gate LC, leg 2: the suppression book does not clear its bar at every offset

Per offset, the top decile of that offset's own out-of-sample predicted P(stop) is vetoed at
entry (2-3 trades per offset; trades that predate any out-of-sample prediction are left
untouched — suppressing them would require a prediction that does not exist without
lookahead) via a new, minimal `spread_lib11` addition: `veto_entry_mask`
(`simulate_single_spread`) / `veto_entry_masks` (`simulate_book`), a per-bar boolean checked
last, after every other suppression/regime/reentry condition, so it can only remove an entry
the pre-declared rule would otherwise take, never add one.

| offset | n vetoed | Sharpe (unsupp. → supp.) | max DD (unsupp. → supp.) | ret/DD (unsupp. → supp.) | n trades (unsupp. → supp.) | beats control on 3-way |
|---|---:|---|---|---|---|---|
| 0 | 3 | −0.1646 → −0.1635 | −1.877% → −1.877% | −0.580 → −0.576 | 57 → 57 | **yes** |
| 1 | 2 | −0.1646 → −0.1591 | −1.877% → −1.877% | −0.580 → −0.561 | 57 → 57 | **yes** |
| 2 | 2 | −0.1646 → −0.1939 | −1.877% → −1.877% | −0.580 → −0.680 | 57 → 56 | **no** |
| 3 | 2 | −0.1646 → −0.1639 | −1.877% → −1.878% | −0.580 → −0.578 | 57 → 57 | **yes** |

Three of four offsets show a marginal improvement (Sharpe moves a few thousandths, well
within any bootstrap noise floor this programme has measured elsewhere); offset 2 makes the
book *worse*. Because the rest of this programme's every-offset convention is the operative
standard everywhere else, and Gate LC's own text does not explicitly relax it, this notebook
requires all four offsets to clear before the suppression leg fires. **It does not — leg 2
fails, and per the gate's own `AND`, Gate LC does not fire**, independent of leg 1's already-
qualified pass.

One mechanical finding worth recording, in the same tradition as Gate BF's sec 0.3
surprise: at offset 2, vetoing 2 entries removed only 1 net trade from the final book (57 →
56), not 2 — suppressing a specific entry bar does not guarantee the underlying z-score
crossing never trades at all, only that *this* bar's attempt is skipped; if the crossing
persists past the vetoed bar, a later entry can still open. A "delete this entry" filter is
not perfectly throughput-neutral in this engine, echoing sec 0.3's own general point about
filters that improve quality by deletion.

## What this notebook establishes, plainly

This was v4 §14.3's own proposed idea, never built by the external programme, and NEXT_PROMPT.md's
own honest prior — informed by 11a Phase 5's trade-shape atlas, where entry extremity failed
to discriminate winners from losers and the catastrophic tail traced entirely to stop-exits
— was that entry-time features would not predict which trades stop out, because the atlas's
own evidence points to the *first post-entry response*, not anything knowable at entry, as
the true separator. **The result here is more nuanced than a clean null**: a point-estimate
AUC comfortably above the pre-registered bar at every offset, on a genuinely walk-forward,
no-lookahead classifier — but one that a direct bootstrap check shows is not reliably
distinguishable from chance on this sample size, and whose practical payoff (a suppression
book) fails to clear its own bar at 1 of 4 offsets regardless. Reported together, not
separately, this is a well-powered near-miss in the AUC leg and an outright miss in the
suppression leg, and Gate LC's overall verdict is a clean **does not fire** — consistent
with sec 6's own prior that the catastrophic tail is not predictable at entry-time, and
directly supporting the case for keeping the stop rather than trying to avoid the trades
that trigger it.

Machinery: `src/research/tmp/run_phase_{0,1,2}_11c_*.py` (Phase 0 trade-log reproduction and
entry-time feature engineering; Phase 1 the walk-forward logistic classifier and its AUC
robustness check; Phase 2 the suppression book, the three-way risk-gate comparison, and the
final verdict, cross-checked programmatically against `phase_6_11a_results.json`), extending
`spread_lib11.py` with `roc_auc_score` and the `veto_entry_mask`/`veto_entry_masks`
parameters on `simulate_single_spread`/`simulate_book`, unit-tested in
`tests/test_spread_lib11.py`. The holdout (2025-01-01 to 2026-07-28) remains untouched and
unspent; its reduced independence for commodity-spread strategies specifically, disclosed in
11a, is unchanged by this notebook.
