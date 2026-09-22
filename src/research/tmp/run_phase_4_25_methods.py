"""Phase 4 -- methods: speed, convergence, variance reduction, and teaching comparisons.

This is a teaching notebook (bespoke commodity derivatives, notebook 025). Phase 2
already proved the pricers agree; Phase 4 is about HOW the methods behave: speed,
convergence, variance reduction, not whether they're correct.

Uses the reference scenario: F=70, K=68, T=1.0, sigma=0.30, r=0.03.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib25
import pricers_25_american as pam
import pricers_25_barrier as pba
import pricers_25_exotic as pex

TMP = Path(__file__).resolve().parent
SEED = 25

# Reference scenario
F = 70.0
K = 68.0
T = 1.0
sigma = 0.30
r = 0.03
df = np.exp(-r * T)


def main():
    results = {}

    # ======================================================================
    # 1. Wall-clock timing per method per instrument, at matched accuracy
    # ======================================================================
    print("1. Timing (Black76, binomial_lattice, MC)...", end=" ", flush=True)
    timing_data = []

    # Black-76 (closed form)
    t0 = time.perf_counter()
    price_b76 = lib25.black76(F, K, T, sigma, df, "put")
    t1 = time.perf_counter()
    timing_data.append(
        {"method": "black76", "seconds": t1 - t0, "price": float(price_b76)}
    )

    # American binomial at n_steps=500 (label as binomial_lattice)
    t0 = time.perf_counter()
    price_binom = pam.american_binomial(F, K, T, sigma, r, "put", n_steps=500)
    t1 = time.perf_counter()
    timing_data.append(
        {"method": "binomial_lattice", "seconds": t1 - t0, "price": float(price_binom)}
    )

    # MC via simulate_gbm at n_paths=100_000
    t0 = time.perf_counter()
    paths = lib25.simulate_gbm(F, sigma, T, n_steps=100, n_paths=100_000, seed=SEED)
    terminal = paths[:, -1]
    payoff = np.maximum(K - terminal, 0.0)  # Put payoff
    price_mc = np.mean(payoff * df)
    t1 = time.perf_counter()
    timing_data.append(
        {"method": "mc_100k", "seconds": t1 - t0, "price": float(price_mc)}
    )

    results["timing"] = timing_data
    print("done")

    # ======================================================================
    # 2. Convergence traces
    # ======================================================================
    print("2. Convergence (binomial, MC, PDE)...", end=" ", flush=True)

    # Binomial convergence: n_steps in [5,10,20,50,100,200,500,1000,2000,4000]
    n_steps_list = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
    binom_prices = []
    for n in n_steps_list:
        p = pam.american_binomial(F, K, T, sigma, r, "call", n_steps=n)
        binom_prices.append(float(p))

    # MC convergence: n_paths in [500, 2000, 10000, 50000, 200000]
    n_paths_list = [500, 2000, 10000, 50000, 200000]
    mc_prices = []
    mc_ses = []
    for n in n_paths_list:
        paths = lib25.simulate_gbm(
            F, sigma, T, n_steps=100, n_paths=n, seed=SEED, antithetic=False
        )
        terminal = paths[:, -1]
        payoff = np.maximum(terminal - K, 0.0)  # Call payoff
        discounted = payoff * df
        mc_prices.append(float(np.mean(discounted)))
        mc_ses.append(float(np.std(discounted) / np.sqrt(n)))

    # PDE convergence: down-and-out put (B=55.0)
    B = 55.0
    ref_analytic = pba.barrier_analytic(F, K, B, T, sigma, df, "put", "do")
    n_grid_list = [50, 100, 200, 400, 800]
    pde_prices = []
    for n in n_grid_list:
        p = pba.barrier_pde(F, K, B, T, sigma, r, "put", "do", n_s=n, n_t=n)
        pde_prices.append(float(p))

    convergence = {
        "binomial_convergence": {
            "n_steps": n_steps_list,
            "price": binom_prices,
        },
        "mc_convergence": {
            "n_paths": n_paths_list,
            "price": mc_prices,
            "se": mc_ses,
        },
        "pde_convergence": {
            "n_grid": n_grid_list,
            "price": pde_prices,
            "reference_analytic": float(ref_analytic),
        },
    }
    results["convergence"] = convergence
    print("done")

    # ======================================================================
    # 3. Variance-reduction efficiency at n_paths=50_000
    # ======================================================================
    print("3. Variance reduction...", end=" ", flush=True)
    n_paths_var = 50_000

    # (a) Plain MC (antithetic=False)
    paths_plain = lib25.simulate_gbm(
        F, sigma, T, n_steps=1, n_paths=n_paths_var, seed=SEED, antithetic=False
    )
    payoff_plain = np.maximum(paths_plain[:, -1] - K, 0.0) * df
    var_plain = np.var(payoff_plain)

    # (b) Antithetic
    paths_anti = lib25.simulate_gbm(
        F, sigma, T, n_steps=1, n_paths=n_paths_var, seed=SEED, antithetic=True
    )
    payoff_anti = np.maximum(paths_anti[:, -1] - K, 0.0) * df
    var_anti = np.var(payoff_anti)

    # (c) Control variate: using terminal price as control (known mean = F)
    payoff_cv = np.maximum(paths_plain[:, -1] - K, 0.0) * df
    control = paths_plain[:, -1]
    beta = np.cov(payoff_cv, control)[0, 1] / np.var(control)
    adjusted_cv = payoff_cv - beta * (control - F)
    var_cv = np.var(adjusted_cv)

    # (d) Sobol QMC with single step (terminal value only)
    z_sobol = lib25.sobol_normals(n_paths_var, n_steps=1, seed=SEED)
    terminal_sobol = F * np.exp(
        -0.5 * sigma**2 * T + sigma * np.sqrt(T) * z_sobol[:, 0]
    )
    payoff_sobol = np.maximum(terminal_sobol - K, 0.0) * df
    var_sobol = np.var(payoff_sobol)

    variance_reduction = {
        "method": ["plain", "antithetic", "control_variate", "sobol"],
        "variance": [
            float(var_plain),
            float(var_anti),
            float(var_cv),
            float(var_sobol),
        ],
        "variance_ratio_vs_plain": [
            1.0,
            float(var_plain / var_anti) if var_anti > 1e-12 else 1.0,
            float(var_plain / var_cv) if var_cv > 1e-12 else 1.0,
            float(var_plain / var_sobol) if var_sobol > 1e-12 else 1.0,
        ],
    }
    results["variance_reduction"] = variance_reduction
    print("done")

    # ======================================================================
    # 4. LSM basis-function sensitivity and duality gap
    # ======================================================================
    print("4. LSM sensitivity...", end=" ", flush=True)

    # Generate paths for LSM (50 time steps, n_paths=50_000)
    paths_lsm = lib25.simulate_gbm(
        F, sigma, T, n_steps=50, n_paths=50_000, seed=SEED, antithetic=False
    )

    # Basis sensitivity: degree in [1,2,3,4,5] at n_paths=50_000
    basis_degrees = [1, 2, 3, 4, 5]
    basis_prices = []
    basis_duals = []
    basis_gaps = []
    for degree in basis_degrees:
        result = pam.american_lsm(
            paths_lsm, K, "put", df, basis_degree=degree, seed=SEED
        )
        basis_prices.append(float(result["price"]))
        basis_duals.append(float(result["dual_upper_bound"]))
        basis_gaps.append(float(result["duality_gap"]))

    # Duality gap vs paths: degree=3, n_paths in [2000, 10000, 50000, 200000]
    n_paths_duality = [2000, 10000, 50000, 200000]
    duality_prices = []
    duality_duals = []
    duality_gaps = []
    for n in n_paths_duality:
        paths_d = lib25.simulate_gbm(
            F, sigma, T, n_steps=50, n_paths=n, seed=SEED, antithetic=False
        )
        result = pam.american_lsm(paths_d, K, "put", df, basis_degree=3, seed=SEED)
        duality_prices.append(float(result["price"]))
        duality_duals.append(float(result["dual_upper_bound"]))
        duality_gaps.append(float(result["duality_gap"]))

    lsm_sensitivity = {
        "basis_sensitivity": {
            "basis_degree": basis_degrees,
            "price": basis_prices,
            "dual_upper_bound": basis_duals,
            "duality_gap": basis_gaps,
        },
        "duality_gap_vs_paths": {
            "n_paths": n_paths_duality,
            "price": duality_prices,
            "dual_upper_bound": duality_duals,
            "duality_gap": duality_gaps,
        },
    }
    results["lsm_sensitivity"] = lsm_sensitivity
    print("done")

    # ======================================================================
    # 5. Markov functional (1-factor, P-measure)
    # ======================================================================
    print("5. Markov functional...", end=" ", flush=True)

    # Generate realized terminal marginals at T1=0.5 and T2=1.0
    T1, T2 = 0.5, 1.0
    n_samples = 20_000

    # T1 samples
    paths_t1 = lib25.simulate_gbm(
        F, sigma, T1, n_steps=1, n_paths=n_samples, seed=SEED, antithetic=False
    )
    samples_t1 = paths_t1[:, -1]

    # T2 samples
    paths_t2 = lib25.simulate_gbm(
        F, sigma, T2, n_steps=1, n_paths=n_samples, seed=SEED, antithetic=False
    )
    samples_t2 = paths_t2[:, -1]

    # Calibrate Markov functional
    marginals = {T1: samples_t1, T2: samples_t2}
    driver_grid = np.linspace(-3, 3, 50)
    mf = pex.markov_functional_1f(marginals, driver_grid)

    # Verify reprice: empirical z-quantiles for T1
    sorted_t1 = np.sort(samples_t1)
    empirical_quantiles_t1 = (np.arange(1, n_samples + 1) - 0.5) / n_samples
    z_quantiles_t1 = sstats.norm.ppf(empirical_quantiles_t1)
    reprice_t1 = mf(T1, z_quantiles_t1)
    reprice_error_t1 = float(np.max(np.abs(reprice_t1 - sorted_t1)))

    # Verify reprice: empirical z-quantiles for T2
    sorted_t2 = np.sort(samples_t2)
    empirical_quantiles_t2 = (np.arange(1, n_samples + 1) - 0.5) / n_samples
    z_quantiles_t2 = sstats.norm.ppf(empirical_quantiles_t2)
    reprice_t2 = mf(T2, z_quantiles_t2)
    reprice_error_t2 = float(np.max(np.abs(reprice_t2 - sorted_t2)))

    # Teaching comparison: price an Asian-style average payoff via MF and plain GBM
    n_teacher = 20_000
    z_teacher = np.random.default_rng(SEED).standard_normal(n_teacher)

    # MF: one-factor assumption (same z drives both T1 and T2)
    mf_t1_vals = mf(T1, z_teacher)
    mf_t2_vals = mf(T2, z_teacher)
    mf_avg = 0.5 * (mf_t1_vals + mf_t2_vals)
    mf_payoff = np.maximum(mf_avg - K, 0.0) * df
    mf_price = float(np.mean(mf_payoff))

    # GBM: two-step paths to approximate correlation
    paths_gbm_2step = lib25.simulate_gbm(
        F, sigma, T2, n_steps=2, n_paths=n_teacher, seed=SEED, antithetic=False
    )
    # T1 approximation: step 1 (at T1 * n_steps / T2 ≈ step 1)
    gbm_t1_vals = paths_gbm_2step[:, 1]
    # T2: final step
    gbm_t2_vals = paths_gbm_2step[:, -1]
    gbm_avg = 0.5 * (gbm_t1_vals + gbm_t2_vals)
    gbm_payoff = np.maximum(gbm_avg - K, 0.0) * df
    gbm_price = float(np.mean(gbm_payoff))

    markov_functional = {
        "reprice_max_error_T1": reprice_error_t1,
        "reprice_max_error_T2": reprice_error_t2,
        "mf_price_estimate": mf_price,
        "gbm_price_estimate": gbm_price,
        "note": "gap between MF and GBM prices shows joint-dynamics error from one-factor assumption; MF reprices its calibration marginals exactly (errors should be tiny)",
    }
    results["markov_functional"] = markov_functional
    print("done")

    # ======================================================================
    # 6. Historical bootstrap pricer
    # ======================================================================
    print("6. Bootstrap pricer...", end=" ", flush=True)
    try:
        # Load bootstrap paths for CL (crude oil) with T=1.0
        paths_bootstrap = lib25.bootstrap_paths(
            "CL", T=1.0, n_paths=20_000, block_days=63, seed=SEED
        )
        F0_bootstrap = paths_bootstrap[:, 0].mean()
        K_bootstrap = F0_bootstrap  # ATM at bootstrap's own starting level

        # Price via bootstrap
        terminal_bootstrap = paths_bootstrap[:, -1]
        payoff_bootstrap = np.maximum(terminal_bootstrap - K_bootstrap, 0.0)
        df_bootstrap = np.exp(-r * T)
        bootstrap_price = np.mean(payoff_bootstrap * df_bootstrap)

        # Price via Black76 at the same F0 and K
        inputs = lib25.inputs_as_of("CL", pd.Timestamp("2026-01-02"))
        sigma_cl = inputs["sigma"](T)
        if hasattr(sigma_cl, "__len__"):
            sigma_cl = sigma_cl[0]
        parametric_price = lib25.black76(
            F0_bootstrap, K_bootstrap, T, float(sigma_cl), df_bootstrap, "call"
        )

        bootstrap_pricer = {
            "bootstrap_price": float(bootstrap_price),
            "parametric_black76_price": float(parametric_price),
            "F0": float(F0_bootstrap),
            "K": float(K_bootstrap),
            "sigma_used": float(sigma_cl),
            "note": "gap between bootstrap and parametric prices shows where the lognormal assumption is doing the work",
        }
    except (ValueError, KeyError, FileNotFoundError) as e:
        # Fallback: if bootstrap fails, note the error
        bootstrap_pricer = {
            "bootstrap_price": float("nan"),
            "parametric_black76_price": float("nan"),
            "F0": float("nan"),
            "K": float("nan"),
            "sigma_used": float("nan"),
            "note": f"bootstrap pricer failed: {e!s}; this is expected if CL market data is unavailable",
        }

    results["bootstrap_pricer"] = bootstrap_pricer
    print("done")

    # ======================================================================
    # Write output
    # ======================================================================
    output_file = TMP / "phase_4_25_methods.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nOutput written to {output_file}")
    print(
        f"JSON size: {output_file.stat().st_size} bytes, top-level keys: {list(results.keys())}"
    )


if __name__ == "__main__":
    main()
