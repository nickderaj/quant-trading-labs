# Notebook 018 — Run Summary (unattended overnight run)

Read this first. Full detail is in `src/results/018_funding_basis_trade.md`; this file is the
run log — what happened, what I assumed, what broke, what I'd do differently.

## What ran

All seven phases ran to completion in one continuous session on branch `notebook-018-funding-basis`,
committed and pushed after each phase:

1. `src/data.py`: added `market` param to `download_and_unzip_klines` (own commit, own test,
   `tests/test_data.py`, 4 tests).
2. Phase 0: `phase_0_18_preregistration.json` written and committed before Phase 1 finished.
3. Phase 1: fetch driver (`run_phase_1_18_fetch.py`) launched in the background; Phase 2 library
   (`basis_lib18.py`) built while it ran.
4. Phase 3: mechanism probe. Gate FA-1 fires. Continued to Phase 4 regardless (it would have either
   way, per instructions).
5. Phases 4 and 5: backtest and ablations, via background scripts, one JSON per runner.
6. Phase 6 gate check: FA-2 and FA-3 did **not** both fire, so Phase 6 was skipped — verified by
   invoking `run_phase_6_18_holdout.py` directly and confirming it refuses (exit 1) without reading
   the holdout directory.
7. Phase 7: notebook, results write-up, README row (this document).

## Gate results and why

| gate | fired? | why |
|---|:---:|---|
| FA-1 (mechanism) | **yes** | pooled gross paired return significantly positive (NW t=3.27) |
| FA-2 (tradeable) | no | Sharpe leg passes (+0.577 net at every offset); bootstrap-CI leg and DSR leg (0.186 vs. 0.95) both fail |
| FA-3 (timing adds value) | no | large point-estimate gap (timed +0.577 vs. always-on −0.415 net Sharpe) but the paired bootstrap CI on the difference includes zero |
| FA-4 (genuinely neutral) | **yes** | beta to crypto basket 0.0005, to BTC 0.0016 — confirmed independently by the perp-leg-only ablation (beta collapses to ≈−1 without the hedge) |
| FUND | no | Sharpe leg passes, DSR leg fails |

## Holdout: not spent

Rule: requires FA-2 **and** FA-3. Neither fired, so Phase 6 did not run. Verified mechanically, not
just by discipline: `run_phase_6_18_holdout.py` reads `phase_4_18_results.json`'s own
`holdout_access.access_granted` field and refuses before ever constructing a path into
`src/research/cache/basis18/holdout/`. That directory is populated (Phase 1 fetched it in the same
pass as dev, per the run instructions) but has never been read by any script that computes a return,
a Sharpe, or a gate verdict.

## Every assumption I made, and why

1. **`scripts/fetch_basis_data.sh` takes no `[dev|holdout]` argument**, unlike NEXT_PROMPT.md's own
   §9.2 skeleton. The overriding run instructions explicitly say to fetch the holdout window "in the
   same pass" as dev; I resolved the tension by having the one script/driver fetch both windows
   internally in a single invocation, rather than requiring two separate invocations. Only raw I/O
   (spot/perp/premium/funding fetch) happens for the holdout window in Phase 1 — no return, gate, or
   Sharpe is ever computed from it outside Phase 6.
2. **Phase 1 uses a Python `ThreadPoolExecutor` driver, not the literal bash+curl+xargs skeleton**
   NEXT_PROMPT.md sketches. The skeleton downloads raw zips with no unzip/parse step; my driver reuses
   `data.download_and_unzip_klines` (already tested, already caches to parquet) directly, which needed
   Python anyway. I judged this more robust (proper per-month caching, structured manifest, retry
   logic) than a two-pass bash-then-parse pipeline, at the cost of literally following the given
   skeleton. Same idempotency/resumability/heartbeat/404-as-data properties are preserved.
3. **premiumIndexKlines gets its own fetch function in `basis_lib18.py`**, not a `data.py` change,
   since its URL path (`.../premiumIndexKlines/...`) is a different endpoint family from
   `.../klines/...`, not just a different `market` — the one sanctioned `data.py` change was narrowly
   the `market` param.
4. **dollar volume is approximated as `close * volume`** for the liquidity screen, since
   `data.download_and_unzip_klines`'s existing `.select()` does not retain Binance's own
   `quote_volume` column, and extending that select was judged out of scope for the one sanctioned
   `data.py` change (the `market` param only). Disclosed in `basis_lib18.py`'s docstring, not silently
   assumed.
5. **"always-on" book membership is uncapped by `N_max`** (every currently-liquid symbol, not just the
   top 10 by carry), read literally from NEXT_PROMPT §5.5's "hold the paired position in every
   screened symbol at all times." This is an implementation reading of an already-frozen book
   definition, not a swept parameter — disclosed in code and in the write-up.
6. **Phase 5's ablations are evaluated at origin_offset=0 only** (not all four offsets), matching
   013's own convention for ablation-table exhibits and keeping the pre-declared `n_trials=18` count
   exact (the offsets themselves are vacuous for this fixed-parameter design anyway, confirmed in
   Phase 4).
7. **DSR's `n_trials=18`** uses the pre-registered baseline exactly; nothing run in this notebook fell
   outside the 18 declared configurations, so no upward revision was needed.
8. **Cost sensitivity's "breakeven round-turn cost"** is reported as a linear interpolation between
   the two bracketing tested grid points (34bp positive, 51bp negative) — labeled as interpolated, not
   as a fifth tested configuration (would have required incrementing `n_trials`).

## Everything that broke, and what I found

1. **A live transient DNS failure mid-Phase-1** (116/256 symbol-windows failed with
   `NameResolutionError`). Not a code bug — recovered completely on a resumed run via the existing
   per-symbol/series/month parquet cache, in under 10 minutes once network connectivity returned.
   Added retry-with-backoff to the fetch driver afterward so a similar blip doesn't need a manual
   resume next time.
2. **A units bug in the break-even-periods constant** (Phase 3): computed 1133 periods instead of the
   intended 34. Caught because the persistence check's clearing fraction was implausibly low (0.1%)
   against the sec 3.4 prior of "feasible but not by a wide margin." Fixed; re-ran; clearing fraction
   is now 44%, consistent with the prior.
3. **Two symbols' price-feed artifacts (`DGBUSDT`, `LUNAUSDT`) distorted two pooled Phase 3
   statistics** — a frozen/stale perp price for `DGBUSDT` (verified: volume=0 for several consecutive
   days) and `LUNAUSDT`'s real 2022 collapse (dividing by a near-zero spot price) both produce basis
   values in the hundreds-of-percent range. The pooled premium-index correlation came back −0.08 (a
   "materially disagree" verdict by the pre-declared bar) even though 124 of 126 symbols individually
   agree well. Fixed by reporting the per-symbol correlation distribution as the primary comparison
   and excluding `|basis| > 20%` bars from pooled statistics, with the exclusion count disclosed.
4. **The same class of artifact reappeared in the actual Phase 4 backtest**, not just the Phase 3
   diagnostic: the timed book's worst single-bar losses trace to `ICPUSDT` (2022-06-25) and
   `MATICUSDT` (Sept 2024, matching 013's own documented MATIC→POL rebrand feed gap) being the *only*
   symbol held that bar, during a verified real perp-market zero-volume stretch the 30-day trailing-
   median liquidity screen was too slow to catch. This is reported as a capacity/robustness finding
   (see the write-up's "Concentration finding"), **not fixed** by changing the frozen liquidity floor,
   lookback window, or hysteresis parameters — that would be exactly the after-the-fact tuning sec 12
   forbids.
5. **`join(..., suffix="_basket")` silently does nothing** when the joined frames' column names don't
   already collide — three beta-computation call sites (Phase 4, 5, 6) relied on it and would have
   computed beta against a wrongly-named column. Caught immediately by `ColumnNotFoundError` when the
   scripts were run (a loud failure, not a silent wrong number), before any result was trusted. Fixed
   by renaming the basket column explicitly before joining, in all three scripts.

All five items are also recorded in `src/results/018_funding_basis_trade.md`'s own "Bugs found"
section, in the house format.

6. **A git staging slip, not a code bug**: the "Phases 3-5" commit's `git add` list omitted
   `basis_lib18.py`, even though `book_metrics` and `build_book_weights`'s `theta_in`/`theta_out`
   params (both used by the Phase 4/5 runners that commit's message describes) had already been added
   to it. The working tree was correct throughout (ruff/mypy/pytest all ran against it and passed),
   but that one pushed commit, checked out in isolation, would have failed to run Phase 4/5 with an
   `AttributeError`. Caught before Phase 7's commit by `git status` showing an unexpected modified
   file; fixed by including it in the Phase 7 commit rather than rewriting history.

## What I'd do next

See the write-up's own "What to test next" section for the full list. Top of it: a diversification
floor below `N_max` (directly addresses the concentration finding), and prioritizing notebook 017's
deferred DSR-estimator correction now that a live, borderline case (FA-2 failing specifically and only
on a DSR leg already flagged as likely-too-harsh) exists to motivate it.

## Checks

`uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy .`, `uv run pytest tests/ -q` all
green after every phase, before every push. Final full-repo state: 236 source files clean under
ruff/mypy, 740 tests passing (11 of them new to this notebook: 4 in `tests/test_data.py`, 7 in
`tests/test_basis_lib18.py`).
