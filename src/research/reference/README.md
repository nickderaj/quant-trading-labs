# Reference material

Source material for research notebooks, kept in-repo so a notebook's inputs are
reproducible without chasing external links that rot.

## `papers/` (local only, not committed)

Gabillon, J. (1991), *The Term Structures of Oil Futures Prices*, Oxford
Institute for Energy Studies, Working Paper WPM 17. ISBN 0 948061 59 6.
Available from <https://www.oxfordenergy.org/publications/the-term-structures-of-oil-futures-prices/>.

The PDF is **not** committed — it is all-rights-reserved OIES material and this
repo is public. `src/research/reference/papers/` is gitignored; put your own
copy there as
`papers/gabillon-1991-term-structures-of-oil-futures-prices.pdf` if you want the
notebook's citations to resolve to a local file.

The Gabillon paper is the two-factor (spot, long-term price) model of the crude
futures curve used by notebook 023. Its key results, by paper page:

- §4.3 p.20–27 — the model. Closed form at eq (26); the short-term-shock
  variant at eq (27)/(28).
- eq (23) p.25 — the implied term structure of futures price volatilities.
- §5.1 p.33 — estimation with a non-stochastic long-term price `L`.
- §5.2 p.35 — estimation with a daily, extrapolated `L`.
- Figures 14–15 p.40–41 — parameter stability and RMSE, stochastic vs
  non-stochastic `L`.

## `data/`

| File | Source |
|---|---|
| `DCOILWTICO.csv` | Crude Oil Prices: West Texas Intermediate (WTI) — Cushing, Oklahoma. U.S. EIA, retrieved from FRED (series `DCOILWTICO`). Daily, 1986-01-02 to 2026-09-15, USD/barrel, not seasonally adjusted. Blank values are non-trading days. |

`DCOILWTICO` is a physical spot assessment. It is **not** the state variable `S`
in the Gabillon model — that is the futures curve extrapolated back to zero
maturity (paper p.33). The two differ by a basis. This series is kept as an
independent cross-check on the fitted `S`, and for the 1986–2010 history the
per-contract futures panel does not cover.

Note: this file is committed despite the repo-wide `*.csv` ignore rule, because
it is a fixed reference input rather than regenerable cache.
