# Notebook 021 — Run Summary

Read this first. Full detail is in `src/results/021_rc3_power_and_data_quality.md`; this file is the
run log — what happened, what I assumed, what broke, what I'd do differently.

## What ran

All four phases ran to completion in one continuous session on branch `notebook-021-rc3-power`,
committed after each phase:

1. Step 0: verified 020's two frozen returns parquets and the 128-file perp cache were present, ran
   the four CI checks clean (763 tests), then branched.
2. Phase 0: `phase_0_21_preregistration.json` written and committed before any catalogue scan ran;
   `run_phase_0_21_preregistration.py` reproduced 020's own RC-3 diff CI from the two stored parquets
   to 1e-12 (exact match) as this notebook's reproduction tripwire.
3. Phase 1: `run_phase_1_21_catalogue.py` scanned all 128 cached perp files with 018's own
   frozen-feed signature (0.4s measured), flagging 19,157 symbol-bars across 21 symbols. Committed the
   catalogue (as compact per-symbol contiguous runs) before Phase 3 touched the diff series.
4. Phase 2: `power_lib21.py` (4 functions, 9 tests) built and tested — network-free, synthetic frames
   only.
5. Phase 3: `run_phase_3_21_results.py` rebuilt `A0`'s and `A3`'s weights (no book rebuild), computed
   the exclusion set, both diff CIs, MDE/`n_required`/`years_required` for both, and the 200-draw
   placebo.
6. Phase 4: `scripts/run_021.sh` (five-step sequential driver, idempotent by output-file existence)
   verified end to end — a clean, from-scratch run measured at 50.7 seconds and reproduced every
   output JSON byte-for-byte. Notebook built via `build_notebook21.py`, then ruff-formatted in place
   (020's own precedent for notebook code cells). Write-up, README row, this document.

## Gate results and why

| gate | fired? | why |
|---|:---:|---|
| PW-1 (detector sound) | **yes** | mechanically rediscovers both of 018's documented events, names neither symbol in its own code |
| PW-2 (data-quality-corrected RC-3) | **yes, nominally** | excluded-bar diff CI `[1.02e-05, 3.10e-05]` clears zero |
| PW-3 (adequately powered) | no | observed mean (1.99e-05) < MDE (5.59e-05) on the original series |
| PW-4 (exclusion surgical, not a reshape) | **no** | cap OK (0.57% ≪ 5%) but placebo fails: flagged mean (2.032e-05) < placebo p95 (2.087e-05) |

**Branch (b): statistical power, not data quality, is the binding constraint.** PW-2's nominal fire is
declared non-load-bearing because PW-4 fails on the placebo leg — the CI move from excluding those 22
bars is not distinguishable from the CI move you'd get excluding any random 22 bars out of 3,840. The
without-exclusion series would need ≈27.6 years of paired history to detect an effect this size at 80%
power; it has 3.50.

## Holdout: not spent

021's own pre-registered policy (sec 6) grants no holdout access under any outcome, including a PW-2
fire — a single corrected diagnostic on a frozen return series was never the RC-2-AND-RC-3 conjunction
020's own holdout gate required. No 021 file names a holdout directory or `HOLDOUT_START`; verified by
grep (see "everything that broke" below for a near-miss on this exact check).

## Every assumption I made, and why

1. **PW-3's gate verdict is scored on the without-exclusion (original RC-3) series, not the
   with-exclusion one** — NEXT_PROMPT.md's sec 3 table says "compute this both with and without the
   exclusion" but doesn't pin which one the gate itself fires on. Read PW-3 as answering "is the
   *original* RC-3 sample adequately powered for the effect it observed" (independent of whether the
   exclusion itself turns out to be legitimate under PW-4) — both numbers are reported in the JSON and
   the write-up regardless, so the choice doesn't hide anything, it only decides the boolean.
2. **PW-1's coverage check (does the catalogue actually flag `ICPUSDT`/`MATICUSDT` in their documented
   windows) lives in `run_phase_3_21_results.py`, not in the detector script** — the pre-registration
   explicitly permits this ("neither symbol named anywhere in the detector's own code"; the grep
   discipline only targets `run_phase_1_21_catalogue.py`), so the check needed a home that's allowed to
   name both symbols. Phase 3 was the natural place since it already computes every other gate.
3. **The flagged-bar catalogue is disclosed in the committed JSON as compact per-symbol contiguous
   runs (start, end, count), not as 19,157 individual per-bar entries** — mathematically lossless
   (8h-bar spacing makes runs and individual bars interconvertible) and dramatically smaller. The
   full per-bar frame is still regenerable deterministically from the same detector function in 0.4s,
   which Phase 3 does rather than depending on the gitignored parquet cache.
4. **Phase 3 recomputes the catalogue directly via `power_lib21.flag_frozen_feed_bars` rather than
   reading Phase 1's cached parquet** — the parquet is gitignored (matches `.gitignore`'s `*.parquet`
   rule) so it wouldn't exist on a fresh clone; recomputing costs 0.4s and guarantees Phase 1 and
   Phase 3 can never silently drift from each other.
5. **`A3`'s weights are rebuilt in Phase 3 even though they're not used in the exclusion set** (which
   is defined by `A0`'s holdings only, per pre-registration) — used instead for a diagnostic-only
   comparison (how many of the same flagged bars `A3`'s more diversified holdings would have been
   exposed to: 18 of 22, vs. `A0`'s 22) that the pre-registration's n_trials itemisation counts as one
   of the 6 trials. Clearly labeled non-load-bearing in the output JSON.

## Everything that broke, and what I found

1. **A self-referential false positive in the sec 6 holdout-literal grep, caught before it could ever
   fire for real.** The pre-registration's own `holdout_policy` note originally quoted the grep
   pattern verbatim (`"basis18/holdout\|bybit20/holdout\|HOLDOUT_START"`) to document what Phase 4's
   check would run. Once `scripts/run_021.sh` existed, running that exact grep over
   `src/research/tmp/*21*.py scripts/run_021.sh` matched the pre-registration file itself — a false
   "holdout literal present" alarm on a file with no holdout access logic at all. Caught while
   re-verifying the grep at the end of Phase 3, before Phase 4 (the driver script) existed to trigger
   it for real. Fixed by rephrasing the note to describe the check without spelling out the literal
   substrings; re-ran the grep after every subsequent file was added and it returns nothing.
2. **No other bug.** The reproduction tripwire matched 020's stored RC-3 CI to 1e-12 on the first run;
   Phase 1's catalogue independently rediscovered both of 018's documented events without any
   hand-tuning of the signature; a from-scratch rerun of the full driver reproduced every output byte
   -for-byte.

## What I'd do next

See the write-up's own "What to test next" for the full list. Top of it: the 27.6-years-required
number is the sharpest fact this notebook produced — RC-3 will not clear on Binance dev-window history
alone without either a genuinely lower-variance diff estimator or the eventual holdout window, earned
by some future notebook's own conjunction, not this one's. 020's two deferred items (on-chain data
survey, market-making) and its `B2` Mechanism-B candidate all carry forward untouched into the
rewritten `NEXT_PROMPT.md`.

## Checks

`uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy .`, `uv run pytest tests/ -q` all
green before branching and before every push. Final full-repo state: 349 source files clean under
ruff/mypy, tests passing including 9 new to this notebook (`tests/test_power_lib21.py`).
