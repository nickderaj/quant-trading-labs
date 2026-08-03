import json


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(
    md("""\
# Notebook 5 - Tail Risk and Conditional Non-Normality

Notebook 4 ran a 7-rung volatility **point**-forecast contest (QLIKE) and found no
clear winner at any interval - HAR-RV, the range estimators, and GARCH-normal all sit in
one statistically indistinguishable cluster, at every interval, on BTC. That contest is
exhausted: conditional variance at these horizons has an R^2 of 0.004-0.19, and every
reasonable point estimator lands in the same place.

Notebook 4 did surface one real, narrower result along the way: GARCH-t's own
Student-t innovation distribution scored better than a normal density. That is a
**density/calibration** result, not a point-forecast one, and it is the reason this
notebook exists. The question changes from "who forecasts variance best" to:

> Given that the tails are the dominant feature of this data (fitted t-df of 2-3,
> 5-sigma moves thousands of times more frequent than normal implies), which model gives
> the best-calibrated conditional tail - and can that be certified with the same rigor
> the variance ladder was?

Full narrative and every number: `src/results/005_tail_risk_evt.md`. This notebook
recomputes the lightweight, cheap-to-demonstrate pieces live and reloads the heavier
rolling-refit artifacts (`phase1_tails_results.json`, `phase3_density_results.json`,
`phase4_coverage_results.json`, `phase5_transfer_results.json` in `src/research/tmp/`)
that a Raspberry Pi shouldn't recompute on every notebook run - those were produced by
`run_phase1_tails.py` / `run_phase3_density.py` / `run_phase4_coverage.py` /
`run_phase5_transfer.py` in the same directory. Terminology is defined from scratch,
grounded in this repo's own numbers, in `docs/` (start at `docs/README.md`).
""")
)

cells.append(
    code("""\
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "tmp")
import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats as st

import dist_lib as L
import dist_lib5 as L5
import distributions as dist
import research

research.set_seed(123)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

SYMBOL = "BTCUSDT"
INTERVALS = ["1h", "4h", "12h", "1d"]
""")
)

cells.append(
    md("""\
## Correction to notebook 4 (landed before any new model code, per this notebook's own
mandatory-first rule)

Two bugs in already-committed code were found and fixed first, because Phase 3's
density contest below is built directly on the same scoring machinery they touch.
""")
)

cells.append(
    code("""\
# The GARCH-t degrees-of-freedom lookahead: run_phase3.py scored GARCH-t's
# own-distribution log score using fits[-1]["params"][3] - the df from the FINAL
# training window - applied to score the entire evaluation period. Fixed with a
# causal, forward-filled path. Confirm directly (not just trust the fix ran) that the
# fitted nu path genuinely varies across refits, rather than being the single constant
# value the bug effectively assumed:
df_1h = L.build_asset_frame(SYMBOL, "1h", end=research.HOLDOUT_START)
ret_1h = df_1h["log_return"].fill_null(0.0).to_numpy()
_, fits_t_1h = L.rolling_garch_forecast(
    ret_1h, refit_every=30 * 24, min_train=90 * 24, innovation="t", max_train=500,
)
nu_path_1h = L.nu_path_from_fits(fits_t_1h, len(ret_1h))
print("n refits:", len(fits_t_1h))
print("nu at each refit:", [round(f["params"][3], 2) for f in fits_t_1h])
print("bug used ONLY the last value:", round(fits_t_1h[-1]["params"][3], 2), "for the WHOLE sample")
""")
)

cells.append(
    code("""\
with open("tmp/phase3_results.json") as f:
    phase3_old_new = json.load(f)  # notebook 4's own file, already amended in place

rows = []
for iv in INTERVALS:
    d = phase3_old_new["intervals"][iv]["density"]["rung5_garch_t_own_dist"]
    rows.append({"interval": iv, "log_score": d["log_score"], "kupiec_p": d["kupiec_p"]})
pl.DataFrame(rows)
""")
)

cells.append(
    md("""\
The corrected numbers (this table) replace notebook 4's originally-reported ones - see
`src/results/004_distributional_models.md`'s own "Correction" section for the old-vs-new
comparison. Both halves of the original claim moved, in different directions: the
log-score win **strengthened** to all 4 intervals (was 3 of 4); the VaR-coverage claim
**weakened** - Kupiec now rejects at 1h/4h (was never rejected anywhere).

The second correction, `distributions.crps_normal_closed_form`/`crps_t_closed_form`
(closed-form CRPS, replacing a numerical grid that was 13-79% wrong on a t(2.1)
forecast at the old default `n_points`), is used throughout Phase 3 below rather than
demonstrated separately - it's what makes the 28-pair, bootstrap-heavy density contest
computationally tractable on this hardware.
""")
)

cells.append(
    md("""\
## Phase 1 - Foundations: does the variance even exist?

Fit-once on the full pre-holdout history (descriptive - Phase 2 onward is rolling and
causal). The Hill estimator is independent of the scipy `t.fit` optimizer notebook 4
already caught pinning at a boundary on the near-degenerate gap-return series.
""")
)

cells.append(
    code("""\
with open("tmp/phase1_tails_results.json") as f:
    phase1 = json.load(f)

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, iv in zip(axes, INTERVALS):
    df_iv = L.build_asset_frame(SYMBOL, iv, end=research.HOLDOUT_START)
    ret_iv = df_iv["log_return"].fill_null(0.0).to_numpy()
    path = L5.hill_alpha_path(ret_iv, tail="upper", k_min=20, k_max=max(20, len(ret_iv) // 10))
    ax.plot(path["k"], path["alpha"], lw=1, color="#4C72B0")
    ax.axhline(2.0, color="crimson", ls="--", lw=1, label="alpha=2 (finite-variance boundary)")
    plateau = phase1["intervals"][iv]["hill"]["upper"]["plateau"]
    if plateau.get("found"):
        ax.axvspan(plateau["k_lo"], plateau["k_hi"], color="orange", alpha=0.15, label="plateau")
    ax.set_title(f"{iv}: Hill plot (upper tail)")
    ax.set_xlabel("k")
    ax.set_xscale("log")
axes[0].set_ylabel("alpha-hat")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
""")
)

cells.append(
    code("""\
rows = []
for iv in INTERVALS:
    for tail in ["upper", "lower"]:
        h = phase1["intervals"][iv]["hill"][tail]
        p = h["plateau"]
        rows.append({
            "interval": iv, "tail": tail, "plateau_found": p.get("found"),
            "k_lo": p.get("k_lo"), "k_hi": p.get("k_hi"),
            "alpha_point": h.get("alpha_point"),
            "ci_lo": h.get("ci_95", [None, None])[0], "ci_hi": h.get("ci_95", [None, None])[1],
        })
hill_table = pl.DataFrame(rows)
hill_table
""")
)

cells.append(
    md("""\
Every point estimate sits above 2, and every computed CI either sits fully above 2 (1h,
4h) or dips only barely below it. **Gate E does not fire at any interval** (0/4 meet
"alpha<=2 with CI excluding 2"). This sits in mild tension with notebook 4's own
Student-t MLE (df as low as 1.98 at 1h) - the non-parametric estimate is consistently a
bit further from the boundary. Variance most likely exists, though not by a wide margin.
""")
)

cells.append(
    code("""\
rows = []
for iv in INTERVALS:
    dv = phase1["intervals"][iv]["dm_validity"]
    rows.append({
        "interval": iv, "pair": " vs ".join(dv["pair"]),
        "normal_pvalue": dv["normal_pvalue"], "bootstrap_pvalue": dv["bootstrap_pvalue"],
        "materially_disagree": dv["materially_disagree"],
        "loss_diff_hill_alpha_upper": dv["loss_diff_hill_alpha_upper"],
    })
pl.DataFrame(rows)
""")
)

cells.append(
    md("""\
DM's normal-approximation and the block-bootstrap p-value materially disagree only at
12h - both still say "not significant" there, so no notebook 4 conclusion is
overturned - but the loss differential's own Hill-estimated tail index runs as low as
~1.2-1.9, low enough that Phase 3 below uses bootstrap p-values as primary throughout,
not the normal approximation alone.
""")
)

cells.append(
    code("""\
rows = []
for iv in INTERVALS:
    lr = phase1["intervals"][iv]["log_rv_vs_rv"]
    for fam in ["normal", "t", "skewt"]:
        rows.append({
            "interval": iv, "family": fam,
            "log_rv_ks_pvalue": lr["log_rv_ks_pvalue"][fam], "rv_ks_pvalue": lr["rv_ks_pvalue"][fam],
        })
pl.DataFrame(rows)
""")
)

cells.append(
    md("""\
Raw RV is rejected outright, every family, every interval (KS p effectively 0 in every
`rv_ks_pvalue` cell). log(RV) is dramatically better-calibrated at every interval except
1h - not rejected at any family at 4h/12h, and skew-t not rejected even at 1d. This
directly motivates the HAR-log-RV rung (`d2`) added to Phase 3 below.

**Gate E verdict: does not fire.**
""")
)

cells.append(
    md("""\
## Phase 2 - New models: GJR-GARCH (leverage) and conditional EVT

Live GJR fit on 1d, with the likelihood-ratio test on gamma=0 printed directly (GJR
nests plain GARCH(1,1) exactly at gamma=0, which is what makes this a genuine, direct
test rather than an eyeballed comparison).
""")
)

cells.append(
    code("""\
df_1d = L.build_asset_frame(SYMBOL, "1d", end=research.HOLDOUT_START)
ret_1d = df_1d["log_return"].fill_null(0.0).to_numpy()

# fit on the most recent 500-bar window, same convention as the rolling refit
window_1d = ret_1d[np.isfinite(ret_1d)][-500:]
gjr_fit = L5.fit_gjr11(window_1d, innovation="normal")
print("GJR-GARCH(1,1,1) fit (last 500 bars, 1d, normal innovation):")
print(f"  omega={gjr_fit['omega']:.3e}  alpha={gjr_fit['alpha']:.4f}  "
      f"gamma={gjr_fit['gamma']:.4f}  beta={gjr_fit['beta']:.4f}")
print(f"  LR test on gamma=0: stat={gjr_fit['lr_gamma0_stat']:.3f}  "
      f"p={gjr_fit['lr_gamma0_pvalue']:.4f}  "
      f"({'significant leverage' if gjr_fit['lr_gamma0_pvalue'] < 0.05 else 'no significant leverage'} on this window)")
""")
)

cells.append(
    code("""\
with open("tmp/phase3_density_results.json") as f:
    phase3_density = json.load(f)

rows = []
for iv in INTERVALS:
    g = phase3_density["intervals"][iv]["gjr_leverage"]
    rows.append({
        "interval": iv,
        "frac_refits_significant_gamma0_normal": g["frac_refits_significant_gamma0_normal"],
        "frac_refits_significant_gamma0_t": g["frac_refits_significant_gamma0_t"],
    })
pl.DataFrame(rows)
""")
)

cells.append(
    md("""\
Leverage is significant in a non-trivial minority of individual refits (17-43%,
strongest at 1h) - but does NOT survive as a net pooled improvement: all-pairs DM
(Phase 3 below) shows GARCH-t significantly beats GJR-t at 1h/4h/12h. A real,
occasionally-significant refit-level effect whose extra parameter's estimation noise
outweighs its benefit once rolled forward - overparameterization with real numbers
behind it.
""")
)

cells.append(
    code("""\
# Live GPD fit + mean-excess plot on 1d standardized residuals - the standard
# threshold-selection diagnostic, and it visually justifies the fixed 10% cutoff.
fc_1d, fits_normal_1d = L.rolling_garch_forecast(
    ret_1d, refit_every=30, min_train=90, innovation="normal", max_train=500,
)
last_fit = fits_normal_1d[-1]
start = max(0, last_fit["t"] - 500)
window = ret_1d[start:last_fit["t"]]
window = window[np.isfinite(window)]
sig2_path = L5._variance_path_for_fit(last_fit, window, model="garch")
z = window / np.sqrt(sig2_path)

gpd_fit = L5.fit_gpd_tail(z, tail_frac=0.10, tail="upper")
print("GPD upper-tail fit (last training window, 1d):", gpd_fit)

thresholds = np.linspace(np.quantile(np.abs(z), 0.75), np.quantile(np.abs(z), 0.98), 30)
mean_excess = [np.mean(z[z > u] - u) if (z > u).sum() > 5 else np.nan for u in thresholds]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thresholds, mean_excess, marker="o", ms=3, color="#4C72B0")
ax.axvline(gpd_fit["u"], color="crimson", ls="--", label=f"10% threshold (u={gpd_fit['u']:.2f})")
ax.set_xlabel("threshold u")
ax.set_ylabel("mean excess above u")
ax.set_title("Mean excess plot, BTC 1d standardized residuals (upper tail)")
ax.legend()
plt.tight_layout()
plt.show()
""")
)

cells.append(
    code("""\
rows = []
for iv in INTERVALS:
    for tail in ["upper", "lower"]:
        # summary already computed in the results write-up from a full rolling
        # refit; recomputed here compactly for the live table
        bpd = {"1h": 24, "4h": 6, "12h": 2, "1d": 1}[iv]
        df_iv = L.build_asset_frame(SYMBOL, iv, end=research.HOLDOUT_START)
        ret_iv = df_iv["log_return"].fill_null(0.0).to_numpy()
        fc_iv, fits_iv = L.rolling_garch_forecast(
            ret_iv, refit_every=30 * bpd, min_train=90 * bpd, innovation="normal", max_train=500,
        )
        _paths, gpd_fits = L5.rolling_gpd_paths(ret_iv, fits_iv, model="garch", max_train=500, tail_frac=0.10)
        xis = [f["xi"] for f in gpd_fits[tail]]
        rows.append({
            "interval": iv, "tail": tail, "n_refits": len(xis),
            "xi_median": float(np.median(xis)) if xis else None,
            "xi_min": float(np.min(xis)) if xis else None, "xi_max": float(np.max(xis)) if xis else None,
            "frac_xi_negative": float(np.mean([x < 0 for x in xis])) if xis else None,
        })
gpd_summary = pl.DataFrame(rows)
gpd_summary
""")
)

cells.append(
    md("""\
**Tripwire investigated, not ignored**: a substantial and growing (with interval width)
fraction of individual 500-bar-window refits show xi<0 (formally a bounded tail -
implausible for crypto). Each refit estimates xi from only ~50 exceedances - a
genuinely small sample. The *median* xi stays sensible and tracks aggregational
Gaussianity (thinner at coarser intervals); the scattered negative estimates are
consistent with small-sample noise around a true xi close to zero, not a real per-refit
finding. Cross-checked against Phase 1's Hill estimator (`1/alpha` approx 0.37-0.45 at
every interval, on RAW returns) - notably higher than the GPD's own median xi on
GARCH-*standardized residuals*, consistent with volatility clustering itself explaining
much of the raw, unconditional fat-tailedness (see the results write-up for the full
argument).
""")
)

cells.append(
    md("""\
## Phase 3 - The density contest

Log score is primary (QLIKE kept only as a secondary column). d8/d9 (GARCH-EVT,
GJR-EVT) are not entered here - continuously normalizing their semiparametric density
proved as fiddly as expected, and this notebook uses its own sanctioned fallback rather
than force a shaky density through. An 8-model, 28-pair-per-interval contest.
""")
)

cells.append(
    code("""\
rows = []
for iv in INTERVALS:
    for name, s in phase3_density["intervals"][iv]["scores"].items():
        rows.append({"interval": iv, "model": name, "log_score": s["log_score_mean"], "qlike": s["qlike_mean"]})
log_score_table = pl.DataFrame(rows).pivot("interval", index="model", values="log_score").sort("1h", descending=True)
log_score_table
""")
)

cells.append(
    code("""\
# All-pairs DM heatmap at 1d (bootstrap p-values), the third required visual.
iv = "1d"
pairs = phase3_density["intervals"][iv]["all_pairs_dm"]
models = sorted(phase3_density["intervals"][iv]["scores"].keys())
n = len(models)
heat = np.full((n, n), np.nan)
for i, a in enumerate(models):
    for j, b in enumerate(models):
        if i == j:
            continue
        key = f"{a}_vs_{b}" if f"{a}_vs_{b}" in pairs else f"{b}_vs_{a}"
        if key in pairs:
            heat[i, j] = pairs[key]["bootstrap_pvalue"]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(heat, cmap="RdYlGn_r", vmin=0, vmax=0.2)
ax.set_xticks(range(n))
ax.set_xticklabels(models, rotation=90, fontsize=8)
ax.set_yticks(range(n))
ax.set_yticklabels(models, fontsize=8)
ax.set_title(f"BTC {iv}: all-pairs DM bootstrap p-value (log score)\\ngreen=tied, red=significant")
plt.colorbar(im, ax=ax, label="bootstrap p-value")
plt.tight_layout()
plt.show()
""")
)

cells.append(
    code("""\
rows = []
for iv in INTERVALS:
    g = phase3_density["intervals"][iv]["gate_a_verdict"]
    rows.append({
        "interval": iv, "best_by_log_score": g["best_by_log_score"],
        "fires_bootstrap_bh": g["beats_every_other_significantly_bootstrap_bh"],
        "fires_normal_bh": g["beats_every_other_significantly_normal_bh"],
    })
pl.DataFrame(rows)
""")
)

cells.append(
    md("""\
**Gate A fires at 3 of 4 intervals** (1h/4h/12h - GARCH-t, both bootstrap- and
normal-approximation-adjusted verdicts agree). Only at 1d does nothing win
significantly (HAR-log-RV edges narrowly ahead, not significantly). This is the
cleanest result this research programme has produced: the identical variance
recursions, scored on log score instead of QLIKE, surface a real, ordered,
mostly-replicating ranking rather than the ties QLIKE found everywhere.
""")
)

cells.append(
    md("""\
## Phase 4 - The tail calibration battery

Full grid: Kupiec + Christoffersen independence + conditional coverage, 6 quantile
levels, 10 models (d8/d9's quantile/ES forecasts ARE well-defined from their GPD fits,
even though their density wasn't entered in Phase 3), 4 intervals - 1,440 tests total.
""")
)

cells.append(
    code("""\
with open("tmp/phase4_coverage_results.json") as f:
    phase4 = json.load(f)

rows = []
for iv in INTERVALS:
    gb = phase4["intervals"][iv]["gate_b_verdict"]
    rows.append({"interval": iv, "clears_all_36": [k for k, v in gb.items() if v]})
pl.DataFrame(rows)
""")
)

cells.append(
    code("""\
# PIT/tail QQ plot: GARCH-t (the Phase 3 winner) vs GARCH-normal (the normal-innovation
# baseline), at 1h - the single most persuasive visual for "does accounting for fat
# tails actually matter."
iv = "1h"
df_iv = L.build_asset_frame(SYMBOL, iv, end=research.HOLDOUT_START)
ret_iv = df_iv["log_return"].fill_null(0.0).to_numpy()
n_iv = len(ret_iv)
fc_normal, _ = L.rolling_garch_forecast(ret_iv, refit_every=30 * 24, min_train=90 * 24, innovation="normal", max_train=500)
fc_t, fits_t = L.rolling_garch_forecast(ret_iv, refit_every=30 * 24, min_train=90 * 24, innovation="t", max_train=500)
nu_path_iv = L.nu_path_from_fits(fits_t, n_iv)

mask_n = np.isfinite(ret_iv) & np.isfinite(fc_normal) & (fc_normal > 0)
pit_normal = st.norm.cdf(ret_iv[mask_n], loc=0, scale=np.sqrt(fc_normal[mask_n]))

mask_t = np.isfinite(ret_iv) & np.isfinite(fc_t) & (fc_t > 0) & np.isfinite(nu_path_iv) & (nu_path_iv > 2)
nu_m, v_m = nu_path_iv[mask_t], fc_t[mask_t]
c_m = np.sqrt(nu_m / (nu_m - 2))
pit_t = st.t.cdf(ret_iv[mask_t] / (np.sqrt(v_m) / c_m), df=nu_m)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, pit, title in zip(axes, [pit_normal, pit_t], ["GARCH-normal", "GARCH-t"]):
    st.probplot(pit, dist="uniform", plot=ax)
    ax.set_title(f"PIT QQ vs Uniform(0,1): {title}, BTC {iv}")
plt.tight_layout()
plt.show()
""")
)

cells.append(
    md("""\
GARCH-normal's PIT values deviate visibly from the Uniform(0,1) reference line in the
tails (the classic signature of a model that doesn't assign enough probability to
extreme moves); GARCH-t tracks the reference line much more closely - exactly what a
better-calibrated tail forecast should look like.
""")
)

cells.append(
    code("""\
rows = []
for iv in INTERVALS:
    for name, es in phase4["intervals"][iv]["es_backtest"].items():
        e = es["0.01"]
        rows.append({"interval": iv, "model": name, "Z": e["z"], "bootstrap_pvalue": e["bootstrap_pvalue"]})
es_table = pl.DataFrame(rows).pivot("interval", index="model", values="Z")
es_table
""")
)

cells.append(
    md("""\
Recall Z~=0 means well-calibrated ES; **Z>0 means realized 1%-tail losses are worse
than the model's own expected-shortfall prediction** (a genuine sign bug in the
runbook's own pseudocode was caught and fixed here - see
`docs/06-scoring-rules-and-calibration.md#acerbi-székely` and the results write-up).
Every non-fat-tailed model is significantly positive (p<0.05) at every single interval
with no exceptions - the cleanest single result in this notebook: models that ignore
fat tails concretely, measurably underestimate how bad the worst days get, every time
this was checked.
""")
)

cells.append(
    md("""\
## Phase 5 - Transfer / stability

ETH/SOL/DOGE/BNB/XRP, 1d only. A real scoping limit: BTC's actual Gate A win
(1h/4h/12h) is never transfer-tested here, since Phase 5 only covers 1d - the one
interval BTC itself found no significant density winner at.
""")
)

cells.append(
    code("""\
with open("tmp/phase5_transfer_results.json") as f:
    phase5 = json.load(f)

rows = []
for s, r in phase5["symbols"].items():
    clears = [k for k, v in r["gate_b_by_model"].items() if v]
    rows.append({
        "symbol": s, "best_by_log_score": r["best_by_log_score"], "gate_a_fires": r["gate_a_fires"],
        "gate_b_clears": ", ".join(clears) if clears else "none",
    })
pl.DataFrame(rows)
""")
)

cells.append(
    md("""\
**Gate A fires at 0 of 6 symbols at 1d** - a perfectly stable null. **Gate B clears on 4
of 5 transfer symbols** (not ETH), with an EVT model present in every clearing set
except XRP's. Read plainly: EVT-based tail calibration replicates as a genuinely good
idea across most of this symbol set, but which specific model clears (or whether any
does) is asset-specific, not a single portable number - notebook 4's own standard
(tail-shape findings replicate more readily than rankings) confirmed again here.
""")
)

cells.append(
    md("""\
## Phase 6 - Application (gated, does not run)

Pre-declared: an EVT-conditional risk-limit overlay on buy-and-hold BTC, judged against
buy-and-hold and a normal-GARCH-driven overlay on Sharpe, max drawdown, 1% exceedance
count, and turnover cost. **Gate D requires a Gate A or B winner AND Gate C stability at
the same interval.** Gate A fired at 1h/4h/12h (BTC only, never transfer-tested there)
and Gate B fired at 12h (BTC) and on 4/5 transfer symbols at 1d - but never jointly, at
the same interval, across all six symbols. **Gate D does not fire. Phase 6 does not
run** - written up rather than silently skipped, per this notebook's own convention.
""")
)

cells.append(
    md("""\
## Bugs found

1. **Acerbi-Szekely sign error in the runbook's own pseudocode** - both the additive
   constant's sign and the stated failure-mode direction were backwards. Caught by
   verifying numerically (a deliberately mis-specified model with a known-wrong
   volatility) rather than trusting the pseudocode or re-deriving on paper alone.
2. **GPD xi<0 tripwire, investigated rather than ignored** - a growing (12% to 78%
   across intervals) fraction of individual 500-bar-window refits produced an
   implausible bounded-tail estimate. Traced to small-sample noise (~50 exceedances per
   refit) around a sensible median, not a real per-refit finding, and cross-checked
   against Phase 1's independent Hill estimate.

Full detail, and the two mandatory notebook-4 corrections that landed before any of
this: `src/results/005_tail_risk_evt.md`.
""")
)

cells.append(
    md("""\
## Bottom line

The point-forecast question notebook 4 asked is exhausted; the density/tail question
this notebook asks is not. Reframing the identical variance models' scoring from QLIKE
to log score surfaces a real, replicating, statistically certified density winner
(GARCH-t) at 3 of 4 BTC intervals. The tail-calibration battery is stricter still, and
still finds something: GARCH-EVT clears every one of 36 coverage tests at 12h, and -
more practically than any single gate - every model that ignores fat tails
significantly underestimates how bad the worst 1% of days actually are, at every
interval, with no exceptions. GJR's leverage effect is real in a meaningful minority of
individual refits but does not survive as a net pooled improvement, and its prevalence
varies enormously by asset.

**Phase 6 did not run** - Gate D's dual requirement (a certified winner, plus
cross-sectional stability, at the same interval) was never jointly satisfied, a
legitimate outcome under this notebook's own pre-declared rule. This notebook adds
genuinely new, certified knowledge notebook 4 could not produce alone: crypto's
conditional tails are real, extreme, and non-normal in a way that concretely costs
risk-unaware models measurable accuracy on the outcomes that matter most - established
with the same pre-declared, multiple-testing-corrected rigor the variance ladder was
held to, even though no tradeable application clears the bar this notebook set for it.

Full numbers, every table, and "what to test next": `src/results/005_tail_risk_evt.md`.
""")
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("005_tail_risk_evt.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"total cells: {len(cells)}")
