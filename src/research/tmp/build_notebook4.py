import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(md("""\
# Notebook 4 - Distributional Models for Volatility and Regime

Notebooks 1-3 all asked distributions (implicitly) to predict the **first moment** - the
next return - and all three found nothing. This notebook resets: single-asset,
deliberately basic, and asks the two questions those notebooks never asked:

1. Can distributional modelling forecast **volatility** better than the trivial
   baselines already in use? (Phase 3, a forecasting contest with proper scoring rules.)
2. Can distributional modelling identify persistent **regimes**, causally, that are
   informative about anything? (Phase 4.)

Both are judged as forecasting contests, not backtests - a backtest happens only in
Phase 5, only if something won its contest first.

Full narrative and numbers: `src/results/4_distributional_models.md`. This notebook
recomputes the lightweight parts live and reloads the heavier rolling-refit artifacts
(`phase1_results.json`, `phase3_results.json`, `phase4_results.json` in
`src/research/tmp/`) that a Raspberry Pi shouldn't recompute on every notebook run -
those were produced by `run_phase1.py`/`run_phase3.py`/`run_phase4.py` in the same
directory, which this notebook's cells mirror at smaller scale for a live demonstration.
"""))

cells.append(code("""\
import sys
from datetime import UTC, datetime

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")
import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats as st

import dist_lib as L
import distributions as dist
import research

research.set_seed(123)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

SYMBOL = "BTCUSDT"
INTERVALS = ["1h", "4h", "12h", "1d"]
"""))

cells.append(md("""\
## Phase 1 - Descriptive: what these series actually look like

Fit once on the full pre-holdout history (causal-to-date, not rolling - Phase 1
characterizes the data; Phase 3/4 are the rolling, out-of-sample forecasts). Full
numbers for all 4 intervals: `phase1_results.json` (produced by `run_phase1.py`); this
cell reproduces the fat-tails fit live for 1d as a demonstration and loads the rest.
"""))

cells.append(code("""\
df_1d = L.build_asset_frame(SYMBOL, "1d", end=research.HOLDOUT_START)
r = df_1d["log_return"].drop_nulls().to_numpy()

normal_p = L.fit_once(df_1d, "log_return", "normal")
t_p = L.fit_once(df_1d, "log_return", "t")
skewt_p = L.fit_once(df_1d, "log_return", "skewt")
print("normal (mu, sigma):", normal_p)
print("t (df, loc, scale):", t_p)
print("skew-t (a, b, loc, scale):", skewt_p)

mu, sigma = normal_p
z = (r - mu) / sigma
p_5sigma_normal = 2 * st.norm.sf(5)
n_5sigma_obs = int(np.sum(np.abs(z) >= 5))
print(f"observed frac |z|>=5sigma: {n_5sigma_obs / len(r):.6%}  "
      f"normal-implied: {p_5sigma_normal:.9%}  "
      f"ratio: {(n_5sigma_obs / len(r)) / p_5sigma_normal:.0f}x")
"""))

cells.append(code("""\
xs = np.linspace(r.min(), r.max(), 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(r, bins=80, density=True, alpha=0.5, label="observed")
axes[0].plot(xs, st.norm(*normal_p).pdf(xs), label="normal fit", lw=2)
axes[0].plot(xs, st.t(*t_p).pdf(xs), label="t fit", lw=2)
axes[0].set_yscale("log")
axes[0].legend()
axes[0].set_title("BTC 1d log returns: density (log scale)")

st.probplot(r, dist=st.t, sparams=t_p, plot=axes[1])
axes[1].set_title("QQ plot vs fitted Student-t")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
with open("tmp/phase1_results.json") as f:
    phase1 = json.load(f)

rows = []
for iv in INTERVALS:
    r_ = phase1["intervals"][iv]
    rows.append({
        "interval": iv, "n": r_["n_obs"],
        "t_df": r_["t_params"][0] if r_["t_params"] else None,
        "5sigma_ratio_vs_normal": r_.get("5sigma_ratio_obs_to_normal"),
        "count_dispersion_index": r_["count_dispersion_index"],
        "normalized_range_excess_pct": r_["normalized_range_excess_pct"],
        "run_length_mean": r_.get("run_length_mean"),
        "ks_geom_pvalue": r_.get("ks_geom", [None, None])[1],
    })
stylized_facts = pl.DataFrame(rows)
stylized_facts
"""))

cells.append(md("""\
Fitted t-df rises monotonically from 1h (1.98) to 1d (2.88) - aggregational
Gaussianity, but crypto starts and stays far from Gaussian. Count dispersion index is
in the hundreds of thousands at every interval (Poisson predicts 1) - trade arrivals are
massively overdispersed. Run lengths reject the geometric null at 3 of 4 intervals - a
distributional echo of notebook 3's short-horizon mean-reversion finding. Full table
with beta/gap/clustering numbers: `src/results/4_distributional_models.md`.
"""))

cells.append(md("""\
## Phase 2 - Machinery

`src/distributions.py` (families normal/t/skewt/poisson/nbinom/beta via `fit_rolling`;
scoring: `log_score`, `crps`, `pit_values`/`pit_ks_test`, `qlike`, `kupiec_test`,
`christoffersen_*`) was built and committed separately (commit 37fdc8b, 77/77 tests
passing) and is not modified here. `dist_lib.py` (this directory) is this notebook's own
supporting library: OHLCV feature engineering, the Phase 3 forecasting rungs, and the
from-scratch GARCH(1,1)/Gaussian-mixture/HMM fits (no `arch`/`hmmlearn`/`sklearn` in this
environment). One bug found and fixed while building it: `fit_once` looked up fitted
parameter columns by their bare name instead of `fit_rolling`'s actual
`f"{col}_{family}_{name}"` naming - see the results doc for detail.
"""))

cells.append(md("""\
## Phase 3 - Volatility forecasting contest

Full 7-rung ladder, BTC, all 4 intervals - every rung implemented and scored, no
exceptions. Refit cadence bounded by calendar time (weekly for cheap OLS rungs, monthly
on a 500-bar-capped window for the MLE rungs - GARCH, RV-distribution fits), so the
number of expensive fits is constant across intervals rather than scaling with bar
count. Produced by `run_phase3.py`; reloaded here (full recompute takes ~15 minutes on
this machine, most of it GARCH-t/skew-t MLE at 1h).

**Bugs found while building this** (full detail in the results doc): a lookahead leak
in the HAR-RV features (unshifted rolling means, so the "daily" component was
momentarily identical to the same-bar target at 1d), a QLIKE-undefined-at-zero bug from
BTC 1h's 13 frozen-price bars (the same *class* of bug as notebook 3's
`realized_vol_24==0`), and a Diebold-Mariano p-value that was actually a raw t-statistic
due to a return-value mislabeling in `research.newey_west_tstat`'s caller. All three are
fixed in the numbers below.
"""))

cells.append(code("""\
with open("tmp/phase3_results.json") as f:
    phase3 = json.load(f)

rows = []
for iv, r in phase3["intervals"].items():
    for rung, rep in r["ladder_reps"].items():
        s = r["scores"][rep]
        rows.append({"interval": iv, "rung": rung, "rep": rep, "qlike": s["qlike"], "mse": s["mse"]})
qlike_table = pl.DataFrame(rows).pivot("interval", index=["rung", "rep"], values="qlike")
qlike_table
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, iv in zip(axes, ["1h", "4h", "12h", "1d"]):
    r = phase3["intervals"][iv]
    rungs = list(r["ladder_reps"].keys())
    q = [r["scores"][r["ladder_reps"][rg]]["qlike"] for rg in rungs]
    ax.bar(rungs, q, color="#4C72B0")
    ax.set_title(f"BTC {iv}: QLIKE by rung")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
Lower is better. HAR-RV (rung2) has the lowest QLIKE at every interval, with the range
estimators (rung3) and GARCH-normal (rung5) close behind - but the all-pairs
Diebold-Mariano test (21 pairs per interval, not just adjacent-rung comparisons) shows
**no rung beats every other rung with significance at any interval** - see
`winner_verdict` below and the results doc for the full pairwise table.
"""))

cells.append(code("""\
for iv, r in phase3["intervals"].items():
    wv = r["winner_verdict"]
    print(f"{iv}: best-by-QLIKE={wv['best_by_qlike']} ({wv['best_rep']}), "
          f"beats every other rung significantly = {wv['beats_every_other_rung_significantly']}")
"""))

cells.append(md("""\
### Density scoring: where a real (narrower) result shows up

Point-forecast QLIKE found no ladder winner, but scoring GARCH-t under its **own**
fitted Student-t innovation distribution (rather than forcing every rung through a
normal density) shows a real calibration edge - best log score at 3 of 4 intervals, and
its 5% VaR exceedance rate is never rejected by the Kupiec coverage test.
"""))

cells.append(code("""\
rows = []
for iv, r in phase3["intervals"].items():
    d = r["density"]
    best_normal = min(
        (r["ladder_reps"][rg] for rg in r["ladder_reps"]),
        key=lambda rep: d[rep]["log_score"] if np.isfinite(d[rep]["log_score"]) else -np.inf,
        default=None,
    )
    row = {"interval": iv, "best_normal_density_rung": best_normal,
           "best_normal_log_score": d[best_normal]["log_score"] if best_normal else None}
    if "rung5_garch_t_own_dist" in d:
        row["garch_t_own_dist_log_score"] = d["rung5_garch_t_own_dist"]["log_score"]
        row["garch_t_kupiec_p"] = d["rung5_garch_t_own_dist"]["kupiec_p"]
    rows.append(row)
pl.DataFrame(rows)
"""))

cells.append(md("""\
### Frozen transfer check (ETH/SOL/DOGE/BNB/XRP, 1d only - scoped down for wall-clock)
"""))

cells.append(code("""\
with open("tmp/phase3_transfer_results.json") as f:
    phase3_transfer = json.load(f)

rows = []
for sym, r in phase3_transfer["symbols"].items():
    rows.append({
        "symbol": sym, "best_rung": r["best_rung"], "best_rep": r["best_rep"],
        "qlike": round(r["qlike_by_rung"][r["best_rung"]], 4),
        "beats_every_other_rung": r["beats_every_other_rung_significantly"],
    })
pl.DataFrame(rows)
"""))

cells.append(md("""\
HAR-RV is the best-by-QLIKE rung at 5 of 6 symbols, but whether it's a *significant*
all-beating winner flips symbol by symbol (yes on ETH/SOL, no on BTC/BNB/XRP/DOGE) -
per notebook 3's "stability outranks magnitude" standard, this is **not a stable
winner**.
"""))

cells.append(md("""\
## Phase 4 - Regime estimation

Threshold baseline, Gaussian mixture (K=2,3), HMM (Gaussian/Student-t emissions),
activity regime - all rolling-refit (monthly, 500-bar cap), filtered-only (never
smoothed), canonically ordered by ascending fitted variance (never full-sample fit).
Produced by `run_phase4.py`.

**Bug found**: the rolling-refit sufficiency check compared window length against
`min_train // 2`, but the window itself is capped at `max_train=500` bars - at 1h,
`min_train` (90 days x 24 bars/day = 2160) made `min_train // 2` (1080) unreachable by
a 500-bar-capped window, so every GMM/HMM refit at 1h silently no-opped for the entire
series. Fixed by flooring the check at `min(min_train, max_train) // 2`.
"""))

cells.append(code("""\
with open("tmp/phase4_results.json") as f:
    phase4 = json.load(f)

rows = []
for iv, r in phase4["intervals"].items():
    for model in ["baseline_threshold", "gmm_k2", "gmm_k3", "hmm_gaussian", "hmm_t", "activity_regime"]:
        v = r[model]
        gd = v["geometric_duration"]
        rows.append({
            "interval": iv, "model": model,
            "mean_duration": gd.get("mean_duration"),
            "geom_ks_p": gd.get("ks_pvalue"),
            "predicts_vol_p": v["predicts"]["vol_kruskal"].get("pvalue"),
            "predicts_direction_p": v["predicts"]["direction_anova"].get("pvalue"),
        })
regime_table = pl.DataFrame(rows)
regime_table.filter(pl.col("model").is_in(["baseline_threshold", "hmm_gaussian"]))
"""))

cells.append(code("""\
fig, ax = plt.subplots(figsize=(8, 4))
piv = regime_table.filter(pl.col("model").is_in(["baseline_threshold", "hmm_gaussian"]))
for model, color in [("baseline_threshold", "#999999"), ("hmm_gaussian", "#C44E52")]:
    sub = piv.filter(pl.col("model") == model).sort("interval")
    ax.plot(sub["interval"], sub["mean_duration"], marker="o", label=model, color=color)
ax.set_ylabel("mean state duration (bars)")
ax.set_title("BTC regime persistence: threshold baseline vs HMM-Gaussian")
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
**Every model at every interval predicts next-bar volatility with overwhelming
significance and shows no consistent direction-prediction effect** - regimes predict
risk, not return, exactly the clean expected finding NEW_PROMPT called likely. HMM-
Gaussian shows 1.7-2.8x longer mean state duration than the naive threshold at every
interval (more persistent, less flip-floppy regimes) and a comparable-or-better vol-
discrimination statistic at 3 of 4 intervals - real structure, but no formal
significance test was built to certify it as beating the baseline (unlike Phase 3's
DM-test apparatus), so it is not reported as a certified Phase 4 "winner."
"""))

cells.append(code("""\
with open("tmp/phase4_transfer_results.json") as f:
    phase4_transfer = json.load(f)

rows = []
for sym, r in phase4_transfer["symbols"].items():
    rows.append({
        "symbol": sym,
        "baseline_vol_p": r["baseline_threshold"]["predicts"]["vol_kruskal"].get("pvalue"),
        "hmm_vol_p": r["hmm_gaussian"]["predicts"]["vol_kruskal"].get("pvalue"),
        "baseline_dir_p": r["baseline_threshold"]["predicts"]["direction_anova"].get("pvalue"),
        "hmm_dir_p": r["hmm_gaussian"]["predicts"]["direction_anova"].get("pvalue"),
    })
pl.DataFrame(rows)
"""))

cells.append(md("""\
"Predicts vol overwhelmingly, direction inconsistently" replicates at all 5 transfer
symbols - the single most stable finding in this notebook.
"""))

cells.append(md("""\
## Phase 5 - Does any of it pay?

**Not run.** Pre-declared to run only if Phase 3 or Phase 4 produced a certified
winner. Neither did, held to a consistent standard: Phase 3's all-pairs DM test found
no rung that beats every other rung significantly at any BTC interval (and the ranking
that does exist isn't stable across symbols); Phase 4's HMM-Gaussian shows real,
replicated structure but was never put through a formal head-to-head significance test
against the baseline. Running Phase 5 on either "almost" result would mean backtesting
a forecast this notebook itself declined to certify - the "no tuning until the backtest
looks good" failure mode NEW_PROMPT explicitly warns against. Full reasoning (including
why the GARCH-t density-calibration result and the HMM persistence result each still
fall short of what Phase 5's own pre-declared gates require) is in the results doc.
"""))

cells.append(md("""\
## Bottom line

**Volatility**: no single rung wins the mandatory 7-rung ladder outright on BTC at any
interval, and the closest thing to a leader (HAR-RV) isn't a stable winner across the 5
transfer symbols either. HAR-RV, the range estimators, and GARCH-normal form a cluster
of similarly-good point forecasts, all beating EWMA and RV-distribution fits with
significance but not beating each other. A real, narrower win exists in density
calibration: GARCH-t's own Student-t innovation distribution gives better-calibrated
tail forecasts than any normal-density alternative (best log score at 3/4 intervals,
VaR coverage never rejected) - confirming the crypto-GARCH literature's standard finding
without rescuing the point-forecast contest.

**Regime**: distributional regime models find real, more persistent structure than a
naive threshold, and the same clean pattern replicates on BTC and all 5 transfer
symbols: **regimes predict volatility, not direction.** No regime model was formally
certified as beating the baseline, so Phase 4 also produced no certified winner.

**Phase 5 did not run** - a legitimate outcome per NEW_PROMPT's own framing. This
extends notebooks 1-3's "no validated tradeable edge" pattern while adding genuinely new
knowledge: crypto's tails, clustering, and regime structure are real, extreme, and now
measured with proper scoring rules, even though none of it clears this notebook's own
bar for calling something a winner.

Full numbers, all bugs found, and "what to test next": `src/results/4_distributional_models.md`.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
            "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.12.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("4_distributional_models.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"total cells: {len(cells)}")
