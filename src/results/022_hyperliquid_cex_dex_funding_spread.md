# 022 — A CEX/DEX Funding Spread: Hyperliquid vs. Binance

## The idea

020 tested whether a funding *spread* between two exchanges is a better carry driver than a funding
*level* (018's own book), and found nothing: Binance vs. Bybit, two large, arbitraged CEXs, produced a
spread with the wrong sign and a book that sat at its own break-even cost.

A 2026-08-23 planning pass (`NEXT_PROMPT.md`) proposed testing the same idea on a genuinely different
pair: **Hyperliquid, an on-chain perpetual exchange, against Binance.** The structural story is that
on-chain flow is retail and directional while a large CEX's is institutional and two-sided, and that
institutions find a DEX hard to onboard to — so, unlike Binance-vs-Bybit, the gap should not be
competed away. A planning-time probe found a spread on BTC roughly four times the size of 020's, with
the right sign: **+9.41% annualised, t≈21.**

This notebook builds a fresh pre-registration and tests that, on real Hyperliquid data, for the first
time in this repo.

## The answer

**Every pre-registered gate fired. The book is real, cheap, neutral, and survives every ablation this
notebook threw at it — including one it should not have survived if the earlier headline number had
been trusted uncritically.**

| Gate | Question | Result |
|---|---|:---:|
| **HD-1** | Does the gross spread exist, pooled, before costs? | **Yes** (t=13.6) |
| **HD-3** | Is the window adequately powered? | **Yes** (Sharpe 4.49 vs. MDE 1.04) |
| **HD-2** | Is the headline book tradeable? (Sharpe, CI, DSR) | **Yes**, all three legs |
| **HD-4** | Is it genuinely market-neutral? | **Yes** (\|β\| < 0.003 to both benchmarks) |
| **HD-5** | Does it survive dropping its top-2 symbols? | **Yes** (Sharpe 5.90 → 3.77) |
| **HD-7** | Does it survive a 40bp long-tail slippage stress? | **Yes** (Sharpe 4.63) |
| **FUND-HL** | Institutionally fundable (Sharpe>0.5, DSR>0.95)? | **Yes** |

**Headline (`HL_ALWAYSON`, offset 0, 2023-07-01 → 2025-06-30, 47 symbols):** net Sharpe **5.90**,
deflated Sharpe probability **>0.99999** at 12 pre-registered trials, max drawdown **-0.47%**, 2-year
net compound return **16.6%**, turnover **15.7 round-turns/year**.

**Holdout access is unlocked** under the pre-registered conjunction (HD-2 ∧ HD-4 ∧ HD-5). No holdout
pass has been run yet — see "What to test next."

## A tripwire, and what it caught

The very first run of the headline book produced a number that should not have been trusted at face
value: net Sharpe 4.29, but **kurtosis (non-excess) 80.4 and skew 2.08** — a signature of a handful of
extreme bars, not a smooth carry return. Per this repo's standing rule ("an implausible number gets
investigated, not reported and not silently patched"), the worst and best bars were traced directly: a
cluster of ±3-8% single-bar moves, all inside **2025-01-06 to 2025-01-11**, all in `FTMUSDT`.

**Fantom's "Sonic" migration froze `FTMUSDT`'s Binance perpetual feed** — price flat, volume zero —
starting 2025-01-06, while Hyperliquid's own FTM market kept trading normally. The "spread" the book
was capturing on those bars was a real, moving Hyperliquid price against a stale, non-updating Binance
mark: a mark-to-market illusion, not a captured funding spread. This is exactly the frozen-feed
signature notebooks 018 and 021 already documented and built a detector for
(`power_lib21.flag_frozen_feed_bars`, open==high==low==close and volume==0 on the perp leg) — reused
here unmodified, import-only.

**Checking all 50 fetched symbols, not just FTM, found the problem was not isolated:**

| Symbol | Frozen fraction of dev-window bars | Cause |
|---|---:|---|
| `FTTUSDT` | **100%** | FTX token — Binance's futures market never had real activity in this window |
| `BLZUSDT` | 25.8% | unexplained, not investigated further |
| `FTMUSDT` | 23.9% | Fantom → Sonic migration |
| `MATICUSDT` | 0.96% | (kept — see below) |
| every other symbol | 0% | — |

A disclosed, mechanical rule — **exclude any symbol frozen on more than 5% of its own dev-window
bars** — was applied, decided *before* checking whether exclusion would help or hurt the headline. It
drops 3 symbols (`FTT`, `BLZ`, `FTM`) from the 50-symbol fetched universe, leaving 47. `MATICUSDT`
clears the bar easily (0.96%) and was kept.

**Why `HL_ALWAYSON` was exposed and 018/020's books were not, by construction, not by luck:**
018/020's books are threshold-gated — a symbol only enters when its carry crosses a hysteresis band.
A frozen, unchanging funding rate rarely crosses that threshold, so a dead symbol mostly just sits
out. `HL_ALWAYSON` has no such filter by design: every liquid symbol is held every bar. That is exactly
the trade-off flagged in `NEXT_PROMPT.md`'s own design section — "always-on is a live candidate for the
first time" — and this notebook found the corresponding cost: an always-on book is maximally exposed
to exactly this failure mode. That is a genuine, reportable property of the construction, not an
artefact of the screen.

**After the fix, the book got healthier, not weaker — the contaminated bars were adding noise, not
inflating the mean.**

| | Before (50 symbols, contaminated) | After (47 symbols, screened) |
|---|---:|---:|
| HD-1 pooled t-stat | 6.15 | **13.60** |
| Headline net Sharpe | 4.29 | **5.90** |
| Skew | 2.08 | **0.65** |
| Kurtosis (non-excess) | 80.4 | **6.69** |
| Max drawdown (net) | -1.11% | **-0.47%** |
| DSR | 0.999988 | **>0.999999** |

The contaminated run's raw output is kept on disk for audit
(`src/research/tmp/{phase_2,phase_4}_22_results_PRE_FROZEN_FIX.json`), not deleted.

## Reproducing the planning probe, and where its headline number actually came from

BTC and ETH's per-symbol numbers reproduce closely: **BTC +10.62% ann (t=8.23), ETH +7.81% ann
(t=6.30)**, against the planning probe's +9.41%/+8.01%. But the probe's other headline figure — **72%
of bars positive** — did *not* reproduce on the tradeable quantity. The actual paired return (funding
spread *plus* the realized price-convergence noise between the two venues) is positive on only
**56-57%** of BTC bars.

Investigating the gap found it exactly: **the planning probe's 72%-positive figure describes the raw
funding-RATE spread alone** (`hl_funding - binance_funding`), which *does* reproduce to three
significant figures — 72.0% positive, t=7.65, +9.38% annualised, matching the probe almost exactly.
The full tradeable position also carries the two venues' relative price movement each bar, which adds
noise around a still-solidly-positive mean, pulling the sign-hit-rate down without hurting the pooled
mean or Sharpe much. **The probe's number was accurate for what it measured; it measured the wrong
thing for judging tradeability.** This notebook's own gates were built on the full paired return
throughout, not the funding rate alone.

## The book grid

| Configuration | Net Sharpe | Note |
|---|---:|---|
| `HL_ALWAYSON` (headline) | **5.90** | no timing, every liquid symbol, every bar |
| `HL_TIMED_FAST` | 4.58 | bl20's 15-day-target-hold thetas |
| `HL_TIMED_SLOW` | 4.90 | 30-day target hold |

**HD-6, the falsification check this notebook pre-registered specifically to interrogate the
structural story, comes back clean: `HL_ALWAYSON` beats both timed variants outright.** Fast timing is
not needed to make the numbers work — if anything, timing *costs* Sharpe here, consistent with a
spread that is genuinely structural rather than something that needs to be traded around funding-sign
inversions the way 018's funding-level book did.

Offset robustness (018/020's own non-refit convention): Sharpe **5.897 / 5.898 / 5.899 / 5.901** across
origin offsets 0-3 — agrees to 3 decimals, as established practice predicts for a fixed-parameter
construction.

## Cost and ablation sensitivity

| Check | Net Sharpe | |
|---|---:|---|
| 0bp (no cost) | 7.45 | gross ceiling |
| 19bp (bare-fee, no slippage) | 6.19 | the planning survey's optimistic comparator |
| **23bp (headline, fully costed)** | **5.90** | — |
| 40bp (long-tail slippage stress) | 4.63 | **HD-7 fires** |
| excluding top-2 symbols (SOL, BTC) | 3.77 | **HD-5 fires** — CI still excludes zero |
| BTC only | 4.68 | |
| ETH only | 2.42 | |

Even at a 40bp stress cost — roughly 1.7x the pre-registered 23bp round-turn, standing in for
unmodelled long-tail slippage this notebook did not build a per-symbol depth model for — the book
clears Sharpe 4.6. Even with its two largest contributors removed, it clears Sharpe 3.8 with a CI that
still excludes zero. Neither BTC nor ETH alone is doing all the work: both sub-books are independently
profitable.

## Neutrality

|β| to the equal-weight crypto basket and to BTC are both **≈0.0023-0.0025** — an order of magnitude
inside the 0.10 bound, and cleaner than 018 or 020's own neutrality numbers. This book is not a
disguised directional bet by any reasonable margin.

## The JELLY force-settlement event

On 2025-03-26, Hyperliquid validators voted to force-settle the JELLY perpetual at an administratively
chosen price to protect the platform's own vault (TVL fell ~$540M → ~$150M). JELLY itself is not
Binance-listed and never enters this notebook's universe, so this is a platform-risk characterisation,
not a JELLY trade outcome: the headline book's realized net return through 2025-03-24..29 was **+0.044%
over 16 bars**, and its single worst bar in that window ranks only at the **11th percentile** of the
book's own full return distribution. Diversification across 47 symbols meant this particular event
barely registered — but the underlying risk (a venue that can administratively settle a market) is real
and carries no analogue in this book's cost model.

## Bugs found

**One, load-bearing, caught by the tripwire discipline rather than by luck:** the frozen-feed
contamination described above. Nothing else was found on inspection of the fetcher, the resampling, or
the join logic — though one genuine implementation bug was caught and fixed *before* any gate was
scored: the first draft of `basis_lib22.py` joined Hyperliquid's native **hourly** funding directly
against 8h candles on an exact-datetime match, which — since only a small, jittered fraction of hourly
timestamps coincide with an 8h boundary — left the panel at roughly 1% of its correct size (BTC: 12
rows instead of 2,191). The fix reuses `bl20.resample_funding_to_8h` (already built and tested for
Bybit's own finer-cadence funding in 020) to sum hourly payments into 8h buckets before joining — a
direct, unmodified reuse of existing, tested machinery, not new logic.

## What to test next

- **The pre-registered holdout is unlocked and has not been spent.** HD-2, HD-4, and HD-5 all fired
  cleanly under the conjunction this notebook committed to before Phase 4 ran. Unlike 018/020, the
  2025-07-01+ Hyperliquid holdout is not pre-cached anywhere in this repo — it is freely fetchable from
  the same public endpoint Phase 1b already used, so the fence protecting it is a matter of discipline,
  not of what is on disk. `run_phase_6_22_holdout.py` has not been written. Spending a one-shot resource
  on the strongest result this research programme has produced in 22 notebooks is exactly the kind of
  decision this repo's own conjunction-gate discipline exists to make mechanical rather than
  discretionary — but it is still irreversible, and is deliberately left to a separate, explicit step
  rather than run automatically in the same pass that unlocked it.
- **`BLZUSDT`'s 25.8% frozen fraction was not investigated further.** It was excluded by the same
  mechanical rule as `FTM`/`FTT`, which is sufficient for this notebook's own purposes, but *why* its
  Binance feed froze for a quarter of the window is an open, minor loose end.
- **The long-tail slippage stress (HD-7) is a flat 40bp proxy, not a modelled cost curve.** A per-symbol
  slippage model calibrated to each market's own book depth remains the highest-risk piece of
  engineering this candidate still lacks, exactly as flagged at planning time.
- **SOL and BTC together are roughly a third of the headline's total contribution** (HD-5's own finding).
  The book does not depend on them exclusively, but a future notebook extending this universe should
  watch whether that concentration grows.

## Inputs and reproducibility

`scripts/run_022.sh` runs every phase in order, idempotently. New data: `src/research/cache/hyperliquid22/dev/`
(funding + 8h candles for 50 mapped symbols, fetched from `api.hyperliquid.xyz/info`, hard-guarded
against reaching `research.HOLDOUT_START`). Reused unmodified: `src/research.py`, `basis_lib18.py`,
`basis_lib20.py`'s cross-venue machinery (`xvenue_paired_log_return`, `xvenue_carry_estimate`,
`build_xvenue_book_weights`), and `power_lib21.flag_frozen_feed_bars`. New library:
`basis_lib22.py`. 7 pinned, network-free tests in `tests/test_basis_lib22.py`.

*Notebook: `src/research/022_hyperliquid_cex_dex_funding_spread.ipynb`. Pre-registration:
`src/research/tmp/phase_0_22_preregistration.json`, committed before Phase 4 ran and not edited since —
the frozen-feed screen this document describes is disclosed as a Phase 2b addendum, not a retroactive
edit to the frozen pre-registration file.*
