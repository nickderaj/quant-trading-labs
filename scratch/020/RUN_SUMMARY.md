# Notebook 020 — Run Summary (unattended overnight run)

Read this first. Full detail is in `src/results/020_basis_refinement_and_cross_venue.md`; this file
is the run log — what happened, what I assumed, what broke, what I'd do differently.

## What ran

All seven phases ran to completion in one continuous session on branch `notebook-020-basis-refinement`,
committed after each phase:

1. Step 0: committed and pushed 019's outstanding work on `notebook-018-funding-basis`
   (four CI checks green, one commit), then branched.
2. Phase 0: `phase_0_20_preregistration.json` written and committed before any data was fetched.
3. Phase 1a: live Bybit probe (`scratch/020/phase1a_probe.json`) — found `interval=480` silently
   rejected, `interval=240`+aggregation the fix; `api.bybit.com` reachable directly.
4. Phase 1b: Bybit fetch (`run_phase_1_20_fetch_bybit.py`), backgrounded via `nohup`; 93/93 symbols
   completed cleanly in ~20 minutes (well inside the 90-minute time box), zero truncation.
5. Phase 2: `basis_lib20.py` (12 tests) built and tested *while the fetch ran in the background* —
   Mechanism A needs no Bybit data, so this was free parallel work.
6. Phase 2b: panel build + reproduction tripwire (passed exactly: 0.5766182328943011, abs diff 0.0)
   run once on Mechanism A immediately; re-run after the Bybit fetch completed to rebuild the
   cross-venue panel with the full 93-symbol universe (the first pass only saw 3 symbols, a leftover
   smoke-test manifest — not a bug, just sequencing, described below).
7. Phase 3 + 3b: mechanism probes and the written-down DSR prediction, for both mechanisms.
8. Phase 4 + 5: 14 book builds + 13 ablations, via `scripts/run_020_books.sh` / `run_020_ablations.sh`.
   Both completed in seconds — see "wall-time note" below.
9. Phase 6: holdout gate invoked once, per sec 9 point 7, and refused correctly (exit 1).
10. Phase 7: notebook, write-up, README row, this document.

## Gate results and why

| gate | fired? | why |
|---|:---:|---|
| RC-1 (mechanism preserved) | **yes** | pooled gross paired return significantly positive (NW t=11.71) |
| RC-2 (tradeable) | **yes** | net Sharpe 3.887–3.889 at every offset; 95% CI excludes zero; DSR=0.9999997 |
| RC-3 (refinement adds value, paired vs. 018) | no | point estimate favours refined but the 95% CI on the diff includes zero |
| RC-4 (genuinely neutral) | **yes** | \|beta\| 0.0022 basket, 0.0020 BTC, all offsets |
| FUND-A | **yes** | Sharpe and DSR legs both pass |
| XD-1 (mechanism, raw/undirected) | no | pooled spread significant but *negative* (Bybit funds higher than Binance, pooled) |
| XD-2 (tradeable) | no | Sharpe 0.354 < 0.5; CI includes zero; DSR=0.082 |
| XD-3 (spread beats level) | no | point estimate *favours the single-venue comparator*, not cross-venue |
| XD-4 (genuinely neutral) | **yes** | \|beta\| 0.0007 basket, 0.0009 BTC |
| FUND-B | no | Sharpe leg fails |

## Holdout: not spent, either mechanism

Mechanism A needs RC-2 AND RC-3; RC-3 did not fire. Mechanism B needs XD-2 AND XD-3; neither fired.
Verified mechanically: `run_phase_6_20_holdout.py` reads `phase_4_20_results.json`'s own
`holdout_access` block and refuses (exit 1) before constructing any path into either
`basis18/holdout/` or `bybit20/holdout/`. Both directories are populated (018 fetched the first,
this notebook fetched the second in the same pass as dev) but neither has been read by any script
that computes a return, Sharpe, or gate verdict.

## Every assumption I made, and why

1. **Mechanism B's "book build" grid has 8 itemised cells (sec 7), not the wall-clock table's
   rough "14 book builds" total (sec 3) split evenly** — I read sec 7's itemisation (9 Mechanism A +
   8 Mechanism B = 17 cells) as the load-bearing, precise count (it's the one "used for every DSR
   computed anywhere in this notebook"), and the wall-clock table's "14" as a rough pre-run estimate.
   Implemented all 17 cells; disclosed the discrepancy in the write-up rather than silently picking one.
2. **Origin offsets are computed by re-running the (cheap, ~1s) book build per offset**, not by
   caching one build and trimming in-process — since `run_phase_4_20_book.py` runs one cell per
   process invocation (sec 8's own architecture) and panel loads are already parquet-cached, redoing
   the full pipeline per offset costs nothing measurable and keeps every cell independently
   resumable/parallelisable, matching the sec 3 idempotence rule more literally than sharing in-memory
   state across offsets would.
3. **The "equal-weight crypto basket" benchmark for both RC-4 and XD-4 is the same, single,
   repo-wide construct** (018's own full 126-symbol Binance perp basket), not a mechanism-specific
   basket restricted to each mechanism's own universe — read literally from sec 6's phrasing ("the
   equal-weight crypto basket" as one fixed thing, not "a" basket per mechanism).
4. **`B_single` (Mechanism B's single-venue comparator) is computed once, in Phase 3b, ahead of its
   official Phase 4 cell** — needed for the written-down prediction's B0 counterfactual DSR, which
   sec 8 requires *before* Phase 4 runs. The Phase 4 cell recomputes/stores it formally (cached
   parquet, so the actual computation only runs once); not double-counted in `n_trials`.
5. **The Bybit funding-interval normalisation and 4h→8h kline aggregation live in `basis_lib20.py`
   as testable functions, not baked into the Phase 1b fetcher** — the fetcher caches raw native-cadence
   data only. This keeps sec 4.4's "get this right or Mechanism B is nonsense" logic pinned by a unit
   test (`test_bybit_funding_resample_to_8h`) rather than buried in an un-tested fetch script.
6. **`B0`'s "one-venue-leg-only" neutrality control (Phase 5) uses the Binance leg's isolated price
   return** (dropping the Bybit leg and funding entirely, same position timing as B0) as the
   comparison series — read as the natural cross-venue analogue of 018's own perp-leg-only ablation,
   since Mechanism B has no spot leg to isolate against.
7. **Phase 5's cost-sensitivity ablations use three explicit round-turn levels each** (0/17/51bp for
   A3, 0/12.5/37.5bp for B0, bracketing each mechanism's real cost), and the reported break-even is a
   linear interpolation between the two points bracketing the Sharpe=0.5 crossing — labeled
   interpolated, not a fourth/fifth tested configuration, keeping `n_trials` exact.
8. **`n_trials=32` throughout**, exactly the pre-registered itemisation — nothing run fell outside the
   32 declared configurations, so no upward revision was needed.

## Everything that broke, and what I found

1. **The cross-venue panel was built once, prematurely, on a stale 3-symbol manifest.** I ran the
   Phase 1b fetcher's `--smoke` test (3 symbols) to validate the pipeline before launching the full
   93-symbol background fetch, which correctly wrote `scratch/020/phase1_manifest.json` with those 3
   symbols. I then ran Phase 2b's panel builder *before* the full fetch finished, and
   `load_xvenue_universe()` read that leftover smoke manifest (which the fetcher only overwrites once,
   at completion) — reporting "xvenue universe: 3/3 symbols ok" instead of the eventual 93. Not a
   code bug: the manifest was doing exactly what it was told. Caught immediately by the suspiciously
   low symbol count; fixed by re-running Phase 2b (and Phase 3/3b) after the full fetch completed, this
   time correctly picking up all 93 symbols. No incorrect number was ever carried into Phase 4/5.
2. **A stray unused holdout-directory literal in `basis_lib20.py`.** While writing the library,
   `BYBIT_HOLDOUT_CACHE_DIR = "src/research/cache/bybit20/holdout"` was declared alongside
   `BYBIT_DEV_CACHE_DIR` for interface symmetry, but never referenced anywhere in the module — dead
   code, no wrong number produced, but it violated sec 9 point 6's mechanical-fencing invariant (only
   `run_phase_6_20_holdout.py` may name a holdout directory literal). Caught by running the sec 9
   point 6 grep as part of finishing Phase 6, before Phase 6 was ever exercised for real; removed, and
   the grep re-run to confirm only one file remains.
3. **Wall-time note, not a bug**: the sec 3 rule-7 wall-time-prediction/refusal machinery
   (`scripts/run_020_books.sh`) is built and does run its prediction, but at 2800+ bars/sec on this
   repo's panel sizes, the entire 17-cell Phase 4 grid predicted (and took) about 10 seconds — nowhere
   near the 6-hour refusal threshold. The elaborate detached-background/heartbeat/xargs-parallel
   architecture sec 3 specifies for *expensive* Monte Carlo work (017/019's DSR confirmation grids)
   is present and correct here too, but this notebook's actual compute cost never came close to
   needing it — disclosed so a future reader isn't confused about why Phase 4/5 finished almost
   instantly.

## What I'd do next

See the write-up's own "What to test next" section for the full list. Top of it: Mechanism A's RC-3
gap (a purely statistical-power question on an already-strong absolute book) and B1's un-gated but
notably stronger score (0.961 vs. B0's 0.354) as a legitimate future pre-registered headline
candidate — not a retroactive substitution here.

## Checks

`uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy .`, `uv run pytest tests/ -q` all
green after every phase, before every push. Final full-repo state: 273 source files clean under
ruff/mypy, 763 tests passing (12 of them new to this notebook, `tests/test_basis_lib20.py`).
