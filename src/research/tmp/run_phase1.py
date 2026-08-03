"""Phase 1 driver: descriptive stylized-fact table for BTC across all 4
intervals. Fit-once (causal-to-date, pre-holdout sample), not rolling -
see dist_lib.fit_once. Dumps results to phase1_results.json.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")

import dist_lib as L
import numpy as np
from scipy import stats as st

import distributions as dist
import research

SYMBOL = "BTCUSDT"
INTERVALS = ["1h", "4h", "12h", "1d"]

out: dict = {"symbol": SYMBOL, "intervals": {}}

for interval in INTERVALS:
    t0 = time.time()
    df = L.build_asset_frame(SYMBOL, interval, end=research.HOLDOUT_START)
    r = df["log_return"].drop_nulls().to_numpy()
    n = len(r)
    res: dict = {"n_obs": n}

    # ---- fat tails: normal / t / skewt fits (causal-to-date, i.e. on the
    # whole pre-holdout sample = the last row of an expanding fit_rolling)
    normal_p = L.fit_once(df, "log_return", "normal")
    t_p = L.fit_once(df, "log_return", "t")
    skewt_p = L.fit_once(df, "log_return", "skewt")
    res["normal_params"] = normal_p
    res["t_params"] = t_p
    res["skewt_params"] = skewt_p

    if normal_p is not None:
        mu, sigma = normal_p
        z = (r - mu) / sigma
        p_5sigma_normal = 2 * st.norm.sf(5)
        n_5sigma_obs = int(np.sum(np.abs(z) >= 5))
        res["frac_5sigma_obs"] = n_5sigma_obs / n
        res["p_5sigma_normal_implied"] = p_5sigma_normal
        res["5sigma_ratio_obs_to_normal"] = (
            (n_5sigma_obs / n) / p_5sigma_normal if p_5sigma_normal > 0 else np.nan
        )
        # PIT/KS under each fitted family
        res["ks_normal"] = list(dist.pit_ks_test(st.norm(loc=mu, scale=sigma), r))
    if t_p is not None:
        df_t, loc_t, scale_t = t_p
        res["ks_t"] = list(dist.pit_ks_test(st.t(df=df_t, loc=loc_t, scale=scale_t), r))
    if skewt_p is not None:
        a, b, loc_s, scale_s = skewt_p
        res["ks_skewt"] = list(
            dist.pit_ks_test(st.jf_skew_t(a=a, b=b, loc=loc_s, scale=scale_s), r)
        )

    # Gaussian scale mixture (2-component, weights/vars): reuse fit_gmm_em
    gsm = L.fit_gmm_em(r, k=2, n_iter=80)
    res["gaussian_scale_mixture"] = gsm

    # ---- volatility clustering: waiting times between k-sigma moves,
    # exponential vs gamma fit
    clustering = {}
    for k in [2.0, 3.0]:
        wt = L.waiting_times_between_k_sigma(r, k)
        if len(wt) >= 20:
            loc_e, scale_e = st.expon.fit(wt, floc=0)
            shape_g, loc_g, scale_g = st.gamma.fit(wt, floc=0)
            ks_e = st.kstest(wt, "expon", args=(loc_e, scale_e))
            ks_g = st.kstest(wt, "gamma", args=(shape_g, loc_g, scale_g))
            clustering[f"k={k}"] = {
                "n_events": len(wt) + 1,
                "n_waits": len(wt),
                "expon_scale": scale_e,
                "gamma_shape": shape_g,
                "gamma_scale": scale_g,
                "ks_expon": [float(ks_e.statistic), float(ks_e.pvalue)],
                "ks_gamma": [float(ks_g.statistic), float(ks_g.pvalue)],
            }
    res["clustering"] = clustering

    # ---- overdispersed activity: count Poisson vs NB
    counts = df["count"].drop_nulls().to_numpy().astype(float)
    mean_c, var_c = counts.mean(), counts.var()
    pois_p = L.fit_once(df.with_columns(), "count", "poisson")
    nb_p = L.fit_once(df, "count", "nbinom")
    res["count_dispersion_index"] = float(var_c / mean_c)
    res["poisson_params"] = pois_p
    res["nbinom_params"] = nb_p
    if pois_p is not None:
        res["ks_poisson"] = list(dist.pit_ks_test(st.poisson(mu=pois_p[0]), counts))
    if nb_p is not None:
        res["ks_nbinom"] = list(
            dist.pit_ks_test(st.nbinom(n=nb_p[0], p=nb_p[1]), counts)
        )

    # ---- bounded observables: beta fits
    tbr = (
        df["taker_buy_ratio"]
        .drop_nulls()
        .filter(
            (df["taker_buy_ratio"].drop_nulls() > 0)
            & (df["taker_buy_ratio"].drop_nulls() < 1)
        )
        .to_numpy()
    )
    icp = (
        df["intrabar_close_pos"]
        .drop_nulls()
        .filter(
            (df["intrabar_close_pos"].drop_nulls() > 0)
            & (df["intrabar_close_pos"].drop_nulls() < 1)
        )
        .to_numpy()
    )
    beta_tbr = L.fit_once(df, "taker_buy_ratio", "beta")
    beta_icp = L.fit_once(df, "intrabar_close_pos", "beta")
    res["beta_taker_buy_ratio"] = beta_tbr
    res["beta_intrabar_close_pos"] = beta_icp
    if beta_tbr is not None:
        res["taker_buy_ratio_alpha_plus_beta"] = beta_tbr[0] + beta_tbr[1]
        res["taker_buy_ratio_mean"] = float(np.mean(tbr))
    if beta_icp is not None:
        res["intrabar_close_pos_alpha_plus_beta"] = beta_icp[0] + beta_icp[1]
        res["intrabar_close_pos_mean"] = float(np.mean(icp))

    # ---- intrabar range distributionally: normalized range vs Brownian prediction
    # normalized range = (high-low)/close-to-close sigma (full-sample sigma,
    # descriptive). Brownian E[range/sigma] = 2*sqrt(2/pi) for driftless BM
    # over the sampling interval (Parkinson's constant).
    sigma_full = float(np.std(r))
    hl = (
        df["hl_log"].drop_nulls().to_numpy()
    )  # this is ln(high/low), a proxy range in log space
    norm_range = hl / sigma_full if sigma_full > 0 else np.array([])
    brownian_expected = 2 * np.sqrt(2 / np.pi)
    res["normalized_range_mean"] = (
        float(np.mean(norm_range)) if len(norm_range) else np.nan
    )
    res["normalized_range_brownian_expected"] = float(brownian_expected)
    res["normalized_range_excess_pct"] = (
        float((np.mean(norm_range) / brownian_expected - 1) * 100)
        if len(norm_range)
        else np.nan
    )

    # ---- gap vs intrabar decomposition
    gap = df["gap_return"].drop_nulls().to_numpy()
    intrabar = df["intrabar_return"].drop_nulls().to_numpy()
    gap_t = L.fit_once(df, "gap_return", "t")
    intrabar_t = L.fit_once(df, "intrabar_return", "t")
    res["gap_return_std"] = float(np.std(gap))
    res["intrabar_return_std"] = float(np.std(intrabar))
    res["gap_t_params"] = gap_t
    res["intrabar_t_params"] = intrabar_t
    res["gap_t_df"] = gap_t[0] if gap_t else None
    res["intrabar_t_df"] = intrabar_t[0] if intrabar_t else None

    # ---- run lengths: geometric null
    runs = L.run_lengths(r)
    if len(runs) >= 20:
        p_geom = 1.0 / np.mean(runs)
        ks_geom = st.kstest(runs, "geom", args=(p_geom,))
        res["run_length_mean"] = float(np.mean(runs))
        res["run_length_geom_p"] = float(p_geom)
        res["ks_geom"] = [float(ks_geom.statistic), float(ks_geom.pvalue)]
        res["run_length_n"] = len(runs)

    res["elapsed_sec"] = time.time() - t0
    out["intervals"][interval] = res
    print(
        f"{interval}: n={n} elapsed={res['elapsed_sec']:.1f}s "
        f"t_df={t_p[0] if t_p else None} 5sigma_ratio={res.get('5sigma_ratio_obs_to_normal')}"
    )

with open("src/research/tmp/phase1_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("written phase1_results.json")
