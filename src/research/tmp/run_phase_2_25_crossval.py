"""Phase 2 -- the cross-validation matrix (the referee) for notebook 025.

With no market option prices to check against, the only available proof that
a pricer is right is that independent methods agree, and that each reduces to
a known closed form in its degenerate limit. This script is the ONE pass/fail
phase in the notebook; nothing downstream may start until it is green.

Written and run by the main session (not delegated) -- Stage 3 of
NEXT_PROMPT.md's build plan.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib25
import pricers_25_american as pam
import pricers_25_asian as pas
import pricers_25_barrier as pba
import pricers_25_exotic as pex

TMP = Path(__file__).resolve().parent
N = sstats.norm.cdf
SEED = 25

OPTS: tuple[lib25.OptType, lib25.OptType] = ("call", "put")
BARRIER_KINDS: tuple[
    tuple[lib25.BarrierKind, str],
    tuple[lib25.BarrierKind, str],
    tuple[lib25.BarrierKind, str],
    tuple[lib25.BarrierKind, str],
] = (("do", "down"), ("di", "down"), ("uo", "up"), ("ui", "up"))


# --------------------------------------------------------------------------- #
# European lattice methods (built here, not delegated -- pricers_25_american
# only exposes AMERICAN binomial/PDE, which include early exercise and are
# therefore not exact European cross-checks). Both are unconstrained versions
# of the same numerics: remove the early-exercise projection.
# --------------------------------------------------------------------------- #


def euro_binomial(F, K, T, sigma, r, opt, n_steps) -> float:
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (1.0 - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(n_steps + 1)
    F_final = F * u ** (n_steps - j) * d**j
    payoff = (
        np.maximum(F_final - K, 0.0) if opt == "call" else np.maximum(K - F_final, 0.0)
    )
    for step in range(n_steps, 0, -1):
        payoff = disc * (p * payoff[:-1] + (1 - p) * payoff[1:])
    return float(payoff[0])


def euro_pde_cn(F, K, T, sigma, r, opt, n_s, n_t) -> float:
    """Crank-Nicolson on x=ln(F): dV/dtau = 0.5*sigma^2*Vxx - 0.5*sigma^2*Vx - r*V,
    tau=T-t running forward from the terminal payoff. Dirichlet boundaries at
    the domain edges use the known deep-ITM/OTM asymptotics."""
    from scipy.linalg import solve_banded

    x0 = np.log(F)
    width = 6 * sigma * np.sqrt(T)
    xs = np.linspace(x0 - width, x0 + width, n_s + 1)
    dx = xs[1] - xs[0]
    dtau = T / n_t
    Fgrid = np.exp(xs)
    V = np.maximum(Fgrid - K, 0.0) if opt == "call" else np.maximum(K - Fgrid, 0.0)

    alpha = 0.5 * sigma**2 / dx**2
    beta = 0.25 * sigma**2 / dx
    A = alpha - beta  # coeff of V_{i+1}
    Bc = -2 * alpha - r  # coeff of V_i
    C = alpha + beta  # coeff of V_{i-1}

    n_int = n_s - 1
    ab = np.zeros((3, n_int))
    ab[0, 1:] = -0.5 * dtau * A
    ab[1, :] = 1 - 0.5 * dtau * Bc
    ab[2, :-1] = -0.5 * dtau * C

    tau = 0.0
    for _ in range(n_t):
        tau_new = tau + dtau
        rhs = (
            0.5 * dtau * C * V[:-2]
            + (1 + 0.5 * dtau * Bc) * V[1:-1]
            + 0.5 * dtau * A * V[2:]
        )
        if opt == "call":
            V0_new, Vn_new = 0.0, Fgrid[-1] - K * np.exp(-r * tau_new)
        else:
            V0_new, Vn_new = K * np.exp(-r * tau_new) - Fgrid[0], 0.0
        rhs[0] += 0.5 * dtau * C * (V0_new - V[0])
        rhs[-1] += 0.5 * dtau * A * (Vn_new - V[-1])
        V_new = np.empty(n_s + 1)
        V_new[0], V_new[-1] = V0_new, Vn_new
        V_new[1:-1] = solve_banded((1, 1), ab, rhs)
        V = V_new
        tau = tau_new
    return float(np.interp(x0, xs, V))


# --------------------------------------------------------------------------- #
# Cell helpers
# --------------------------------------------------------------------------- #


def cell(name, methods, tolerance_desc, passed, detail):
    return {
        "name": name,
        "methods": methods,
        "tolerance": tolerance_desc,
        "passed": bool(passed),
        "detail": detail,
    }


def main() -> None:
    results = []

    # Reference scenario, fixed for the whole matrix.
    F, K, T, sigma, r = 70.0, 68.0, 1.0, 0.30, 0.03
    df = float(np.exp(-r * T))
    n_paths_mc = 300_000
    n_steps_path = 100

    # --- 1. European call/put: black76 vs binomial vs PDE vs MC ---
    for opt in OPTS:
        closed = lib25.black76(F, K, T, sigma, df, opt)
        tree = euro_binomial(F, K, T, sigma, r, opt, n_steps=2000)
        pde = euro_pde_cn(F, K, T, sigma, r, opt, n_s=400, n_t=400)
        paths = lib25.simulate_gbm(F, sigma, T, n_steps_path, n_paths_mc, seed=SEED)
        term = paths[:, -1]
        payoff = (
            np.maximum(term - K, 0.0) if opt == "call" else np.maximum(K - term, 0.0)
        )
        pv = payoff * df
        mc_price, mc_se = float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv)))
        tree_ok = abs(tree - closed) / closed < 0.001
        pde_ok = abs(pde - closed) / closed < 0.001
        mc_ok = abs(mc_price - closed) < 3 * mc_se
        results.append(
            cell(
                f"european_{opt}",
                ["black76", "binomial_n2000", "pde_cn", "mc"],
                "mc within 3 SE; tree and PDE within 0.1% of closed form",
                tree_ok and pde_ok and mc_ok,
                {
                    "closed": closed,
                    "tree": tree,
                    "pde": pde,
                    "mc": mc_price,
                    "mc_se": mc_se,
                    "tree_ok": tree_ok,
                    "pde_ok": pde_ok,
                    "mc_ok": mc_ok,
                },
            )
        )

    # --- 2. European, negative-capable: bachelier vs MC arithmetic ---
    F_neg, K_neg, sigma_n = 5.0, 3.0, 8.0
    for opt in OPTS:
        closed = lib25.bachelier(F_neg, K_neg, T, sigma_n, df, opt)
        paths = lib25.simulate_bachelier(
            F_neg, sigma_n, T, n_steps_path, n_paths_mc, seed=SEED
        )
        term = paths[:, -1]
        payoff = (
            np.maximum(term - K_neg, 0.0)
            if opt == "call"
            else np.maximum(K_neg - term, 0.0)
        )
        pv = payoff * df
        mc_price, mc_se = float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv)))
        ok = abs(mc_price - closed) < 3 * mc_se
        results.append(
            cell(
                f"european_negative_capable_{opt}",
                ["bachelier", "mc_arithmetic"],
                "3 standard errors",
                ok,
                {"closed": closed, "mc": mc_price, "mc_se": mc_se},
            )
        )

    # --- 3. Displaced diffusion limits ---
    black = lib25.black76(F, K, T, sigma, df, "call")
    disp_small = lib25.displaced_black(F, K, T, sigma, df, "call", shift=1e-6)
    ok_small = abs(disp_small - black) / black < 0.001
    # The shift->large limit needs the SHIFTED process's proportional vol
    # (sigma) to be small relative to 1/sqrt(T) for the lognormal-vs-Gaussian
    # gap to actually vanish -- at sigma=0.30 the relative vol over T=1yr is
    # not small, so the two models retain an irreducible ~0.4% gap regardless
    # of shift size. Use a small-sigma scenario to isolate the true limit.
    sigma_small, shift_large = 0.02, 100_000.0
    disp_large = lib25.displaced_black(
        F, K, T, sigma_small, df, "call", shift=shift_large
    )
    sigma_n_equiv = sigma_small * (F + shift_large)
    normal_equiv = lib25.bachelier(F, K, T, sigma_n_equiv, df, "call")
    ok_large = abs(disp_large - normal_equiv) / normal_equiv < 0.001
    results.append(
        cell(
            "displaced_diffusion_limits",
            [
                "displaced_black->black76(shift->0)",
                "displaced_black->bachelier(shift->large)",
            ],
            "0.1%",
            ok_small and ok_large,
            {
                "black76": black,
                "displaced_shift_small": disp_small,
                "displaced_shift_large": disp_large,
                "bachelier_equiv": normal_equiv,
                "ok_small": ok_small,
                "ok_large": ok_large,
            },
        )
    )

    # --- 4/5. Barriers: analytic vs PDE vs MC(+BGK), and in-out parity ---
    B_down, B_up = 55.0, 90.0
    dt_step = T / n_steps_path
    barrier_detail = {}
    barrier_ok_all = True
    parity_ok_all = True
    for kind, direction in BARRIER_KINDS:
        B = B_down if direction == "down" else B_up
        analytic = pba.barrier_analytic(F, K, B, T, sigma, df, "call", kind)
        # n=1500: the knock-in kinds are small numbers obtained by subtracting
        # two close vanilla/knock-out values, which amplifies the PDE's
        # absolute truncation error into a much larger relative error -- needs
        # finer resolution than the knock-out kinds alone would.
        pde_val = pba.barrier_pde(
            F, K, B, T, sigma, r, "call", kind, n_s=1500, n_t=1500
        )
        pde_ok = abs(pde_val - analytic) / max(analytic, 1e-8) < 0.005
        # BGK: discrete monitoring catches fewer breaches than continuous
        # monitoring, so a raw discretely-monitored MC price at B differs
        # systematically from the continuous analytic price at B. The
        # correction is applied to the CONTINUOUS formula (move its barrier
        # further from spot) so it matches the raw discrete MC, not the
        # other way around.
        B_shifted = pba.barrier_continuity_shift(B, sigma, dt_step, kind)
        analytic_bgk = pba.barrier_analytic(F, K, B_shifted, T, sigma, df, "call", kind)
        paths = lib25.simulate_gbm(F, sigma, T, n_steps_path, n_paths_mc, seed=SEED)
        mc_out = pba.barrier_mc(paths, K, B, kind, "call", df)
        mc_ok = abs(mc_out["price"] - analytic_bgk) < 3 * mc_out["se"]
        barrier_ok_all = barrier_ok_all and pde_ok and mc_ok
        barrier_detail[kind] = {
            "analytic": analytic,
            "pde": pde_val,
            "pde_ok": pde_ok,
            "mc_discrete_raw": mc_out["price"],
            "mc_se": mc_out["se"],
            "analytic_bgk_shifted": analytic_bgk,
            "mc_ok": mc_ok,
        }
    results.append(
        cell(
            "barrier_all_four_types",
            [
                "reiner_rubinstein_analytic",
                "pde_barrier_on_grid_node",
                "mc_bgk_continuity",
            ],
            "0.5% analytic vs PDE; MC within 3 SE",
            barrier_ok_all,
            barrier_detail,
        )
    )

    parity_detail = {}
    for down_up, B in (("down", B_down), ("up", B_up)):
        kind_ko: lib25.BarrierKind = "do" if down_up == "down" else "uo"
        kind_ki: lib25.BarrierKind = "di" if down_up == "down" else "ui"
        ko = pba.barrier_analytic(F, K, B, T, sigma, df, "call", kind_ko)
        ki = pba.barrier_analytic(F, K, B, T, sigma, df, "call", kind_ki)
        vanilla = lib25.black76(F, K, T, sigma, df, "call")
        diff = abs((ko + ki) - vanilla)
        ok = diff < 1e-10
        parity_ok_all = parity_ok_all and ok
        parity_detail[down_up] = {
            "ko": ko,
            "ki": ki,
            "vanilla": vanilla,
            "diff": diff,
            "ok": ok,
        }
    results.append(
        cell(
            "barrier_in_out_parity",
            ["knock_in + knock_out == vanilla"],
            "1e-10",
            parity_ok_all,
            parity_detail,
        )
    )

    # --- 6. Geometric Asian: closed form vs MC ---
    avg_start, n_fix = 0.75, 21
    t_fix = avg_start + np.arange(n_fix) * (T - avg_start) / (n_fix - 1)
    idx_fix = np.clip(np.round(t_fix / T * n_steps_path).astype(int), 0, n_steps_path)
    for opt in OPTS:
        closed = pas.asian_geometric(F, K, T, sigma, df, opt, avg_start, n_fix)
        paths = lib25.simulate_gbm(F, sigma, T, n_steps_path, n_paths_mc, seed=SEED)
        geo_avg = np.exp(np.log(paths[:, idx_fix]).mean(axis=1))
        payoff = (
            np.maximum(geo_avg - K, 0.0)
            if opt == "call"
            else np.maximum(K - geo_avg, 0.0)
        )
        pv = payoff * df
        mc_price, mc_se = float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv)))
        ok = abs(mc_price - closed) < 3 * mc_se
        results.append(
            cell(
                f"geometric_asian_{opt}",
                ["closed_form", "mc"],
                "3 SE",
                ok,
                {"closed": closed, "mc": mc_price, "mc_se": mc_se},
            )
        )

    # --- 7. Arithmetic Asian: Turnbull-Wakeman vs MC (geometric control variate) ---
    for opt in OPTS:
        tw = pas.asian_turnbull_wakeman(F, K, T, sigma, df, opt, avg_start, n_fix)
        paths = lib25.simulate_gbm(F, sigma, T, n_steps_path, n_paths_mc, seed=SEED)
        mc_out = pas.asian_mc(paths, K, opt, df, idx_fix, control_variate=True)
        ok = abs(mc_out["price"] - tw) / tw < 0.01
        results.append(
            cell(
                f"arithmetic_asian_{opt}",
                ["turnbull_wakeman", "mc_geometric_control_variate"],
                "1%",
                ok,
                {"tw": tw, "mc": mc_out["price"], "mc_se": mc_out["se"]},
            )
        )

    # --- 8. Asian ordering ---
    arith = pas.asian_turnbull_wakeman(F, K, T, sigma, df, "call", avg_start, n_fix)
    geo = pas.asian_geometric(F, K, T, sigma, df, "call", avg_start, n_fix)
    euro = lib25.black76(F, K, T, sigma, df, "call")
    ok_ordering = (arith >= geo) and (arith <= euro) and (geo <= euro)
    results.append(
        cell(
            "asian_ordering",
            ["arithmetic >= geometric", "asian <= european same strike"],
            "strict",
            ok_ordering,
            {"arithmetic": arith, "geometric": geo, "european": euro},
        )
    )

    # --- 9. American: binomial vs PDE vs LSM (with dual upper bound) ---
    opt = "put"
    binom = pam.american_binomial(F, K, T, sigma, r, opt, n_steps=1000)
    pde_amer = pam.american_pde(F, K, T, sigma, r, opt, n_s=300, n_t=300)
    paths = lib25.simulate_gbm(F, sigma, T, n_steps_path, n_paths_mc, seed=SEED)
    lsm = pam.american_lsm(paths, K, opt, df, basis_degree=3, seed=SEED)
    binom_se_proxy = (
        0.01 * binom
    )  # binomial has no natural SE; use a small tolerance band
    lower = binom - 3 * max(binom_se_proxy, 1e-6)
    upper = lsm["dual_upper_bound"]
    ok_lsm_band = (lsm["price"] >= lower - 1e-6) and (lsm["price"] <= upper + 1e-6)
    ok_pde = abs(pde_amer - binom) / binom < 0.02
    results.append(
        cell(
            "american",
            ["binomial", "pde", "longstaff_schwartz_with_dual_upper_bound"],
            "LSM inside [binomial - 3 SE, dual upper bound]",
            ok_lsm_band and ok_pde,
            {
                "binomial": binom,
                "pde": pde_amer,
                "lsm": lsm["price"],
                "dual_upper_bound": lsm["dual_upper_bound"],
                "duality_gap": lsm["duality_gap"],
                "ok_lsm_band": ok_lsm_band,
                "ok_pde": ok_pde,
            },
        )
    )

    euro_put = lib25.black76(F, K, T, sigma, df, "put")
    intrinsic = max(K - F, 0.0)
    ok_amer_ordering = (binom >= euro_put - 1e-8) and (euro_put >= intrinsic - 1e-8)
    results.append(
        cell(
            "american_ordering",
            ["american >= european >= intrinsic"],
            "strict",
            ok_amer_ordering,
            {"american": binom, "european": euro_put, "intrinsic": intrinsic},
        )
    )

    # --- 10. Spread option: Margrabe (K=0) vs Kirk vs bivariate MC ---
    F1, F2, s1, s2, rho_spread = 70.0, 30.0, 0.30, 0.35, 0.5
    marg = pex.margrabe(F1, F2, T, s1, s2, rho_spread, df)
    paths2 = lib25.simulate_correlated(
        [F1, F2], [s1, s2], rho_spread, T, n_steps_path, n_paths_mc, seed=SEED
    )
    mc_spread0 = pex.spread_mc(paths2[0], paths2[1], 0.0, "call", df)
    ok_marg = abs(marg - mc_spread0["price"]) < 3 * mc_spread0["se"]
    # Kirk near-zero-spread/high-correlation corner: deviation is expected, recorded not failed.
    F1c, F2c, rho_c, Kc = 30.0, 29.5, 0.95, 0.5
    kirk_val = pex.kirk(F1c, F2c, Kc, T, s1, s2, rho_c, df, "call")
    pathsc = lib25.simulate_correlated(
        [F1c, F2c], [s1, s2], rho_c, T, n_steps_path, n_paths_mc, seed=SEED
    )
    mc_spreadc = pex.spread_mc(pathsc[0], pathsc[1], Kc, "call", df)
    kirk_deviation_pct = abs(kirk_val - mc_spreadc["price"]) / max(
        mc_spreadc["price"], 1e-8
    )
    results.append(
        cell(
            "spread_option",
            ["margrabe_at_k0", "kirk", "bivariate_mc"],
            "Margrabe vs MC 3 SE; Kirk allowed to deviate near zero spread/high correlation -- recorded, not a failure",
            ok_marg,
            {
                "margrabe_k0": marg,
                "mc_k0": mc_spread0["price"],
                "mc_k0_se": mc_spread0["se"],
                "kirk_corner_value": kirk_val,
                "mc_corner_value": mc_spreadc["price"],
                "kirk_corner_deviation_pct": kirk_deviation_pct,
            },
        )
    )

    # --- 11. Quanto: analytic drift adjustment vs MC correlated FX ---
    sigma_fx, rho_fx = 0.12, 0.4
    F0_fx = 1.10
    quanto_analytic = pex.quanto_black76(F, K, T, sigma, sigma_fx, rho_fx, df, "call")
    pathsq = lib25.simulate_correlated(
        [F, F0_fx], [sigma, sigma_fx], rho_fx, T, n_steps_path, n_paths_mc, seed=SEED
    )
    F_quanto_terminal = pathsq[0][:, -1] * np.exp(-rho_fx * sigma * sigma_fx * T)
    payoff_q = np.maximum(F_quanto_terminal - K, 0.0) * df
    mc_q_price, mc_q_se = (
        float(payoff_q.mean()),
        float(payoff_q.std(ddof=1) / np.sqrt(len(payoff_q))),
    )
    ok_quanto = abs(quanto_analytic - mc_q_price) < 3 * mc_q_se
    results.append(
        cell(
            "quanto",
            ["analytic_drift_adjustment", "mc_correlated_fx"],
            "3 SE",
            ok_quanto,
            {"analytic": quanto_analytic, "mc": mc_q_price, "mc_se": mc_q_se},
        )
    )

    # --- 12. Put-call parity, every European pricer ---
    parity_pricers = {}
    c = lib25.black76(F, K, T, sigma, df, "call")
    p = lib25.black76(F, K, T, sigma, df, "put")
    diff = abs((c - p) - df * (F - K))
    parity_pricers["black76"] = {"diff": diff, "ok": diff < 1e-8}

    c = lib25.bachelier(F_neg, K_neg, T, sigma_n, df, "call")
    p = lib25.bachelier(F_neg, K_neg, T, sigma_n, df, "put")
    diff = abs((c - p) - df * (F_neg - K_neg))
    parity_pricers["bachelier"] = {"diff": diff, "ok": diff < 1e-8}

    shift = 20.0
    c = lib25.displaced_black(F, K, T, sigma, df, "call", shift)
    p = lib25.displaced_black(F, K, T, sigma, df, "put", shift)
    diff = abs((c - p) - df * (F - K))
    parity_pricers["displaced_black"] = {"diff": diff, "ok": diff < 1e-8}

    ok_parity_all = all(v["ok"] for v in parity_pricers.values())
    results.append(
        cell(
            "put_call_parity",
            ["black76", "bachelier", "displaced_black"],
            "1e-8",
            ok_parity_all,
            parity_pricers,
        )
    )

    # ----------------------------------------------------------------- #
    # Teaching-chart auxiliary data (small grids, for Part C figures)
    # ----------------------------------------------------------------- #
    charts: dict = {}

    # Kirk's error surface over (spread/strike ratio proxy, correlation)
    kirk_surface = []
    for rho_g in (0.0, 0.3, 0.6, 0.8, 0.95):
        for spread_pct in (0.02, 0.05, 0.10, 0.20, 0.40):
            F1g, F2g = 50.0, 48.0
            Kg = spread_pct * F1g
            kirk_g = pex.kirk(F1g, F2g, Kg, T, s1, s2, rho_g, df, "call")
            pathsg = lib25.simulate_correlated(
                [F1g, F2g], [s1, s2], rho_g, T, 60, 60_000, seed=SEED
            )
            mc_g = pex.spread_mc(pathsg[0], pathsg[1], Kg, "call", df)
            err_pct = float((kirk_g - mc_g["price"]) / max(mc_g["price"], 1e-8))
            kirk_surface.append(
                {
                    "rho": rho_g,
                    "spread_pct": spread_pct,
                    "kirk": kirk_g,
                    "mc": mc_g["price"],
                    "err_pct": err_pct,
                }
            )
    charts["kirk_error_surface"] = kirk_surface

    # Turnbull-Wakeman error surface over (vol, maturity)
    tw_surface = []
    for sigma_g in (0.15, 0.25, 0.35, 0.50):
        for T_g in (0.25, 0.5, 1.0, 2.0):
            n_fix_g = 12
            avg_start_g = max(T_g - 0.1, 0.0)
            tw_g = pas.asian_turnbull_wakeman(
                F, K, T_g, sigma_g, df, "call", avg_start_g, n_fix_g
            )
            t_fix_g = avg_start_g + np.arange(n_fix_g) * (T_g - avg_start_g) / (
                n_fix_g - 1
            )
            n_steps_g = 60
            idx_g = np.clip(
                np.round(t_fix_g / T_g * n_steps_g).astype(int), 0, n_steps_g
            )
            paths_g = lib25.simulate_gbm(F, sigma_g, T_g, n_steps_g, 60_000, seed=SEED)
            mc_g = pas.asian_mc(paths_g, K, "call", df, idx_g, control_variate=True)
            err_pct = float((tw_g - mc_g["price"]) / max(mc_g["price"], 1e-8))
            tw_surface.append(
                {
                    "sigma": sigma_g,
                    "T": T_g,
                    "tw": tw_g,
                    "mc": mc_g["price"],
                    "err_pct": err_pct,
                }
            )
    charts["turnbull_wakeman_error_surface"] = tw_surface

    # Binomial convergence and oscillation vs n
    binom_conv = []
    for n_g in (5, 10, 20, 50, 100, 200, 500, 1000, 2000):
        price_g = euro_binomial(F, K, T, sigma, r, "call", n_g)
        binom_conv.append({"n_steps": n_g, "price": price_g, "closed_form": black})
    charts["binomial_convergence"] = binom_conv

    # MC error vs paths, plain / antithetic / control variate (geometric) / Sobol
    mc_error = []
    for n_p in (1_000, 5_000, 20_000, 100_000):
        paths_plain = lib25.simulate_gbm(
            F, sigma, T, n_steps_path, n_p, seed=SEED, antithetic=False
        )
        payoff_plain = np.maximum(paths_plain[:, -1] - K, 0.0) * df
        se_plain = float(payoff_plain.std(ddof=1) / np.sqrt(n_p))

        paths_anti = lib25.simulate_gbm(
            F, sigma, T, n_steps_path, n_p, seed=SEED, antithetic=True
        )
        payoff_anti = np.maximum(paths_anti[:, -1] - K, 0.0) * df
        se_anti = float(payoff_anti.std(ddof=1) / np.sqrt(n_p))

        # The control variate must be measured on a payoff it does NOT
        # reproduce exactly. Passing a single averaging index (the terminal
        # point) makes the geometric and arithmetic averages the same number,
        # so beta = Cov/Var = 1, the adjusted payoff collapses to its own mean,
        # and the reported "standard error" is floating-point noise that
        # scales like 1/n instead of 1/sqrt(n) -- an impossible convergence
        # rate that made the chart claim a ~1000x variance reduction.
        # Average over the whole path, which is the payoff the geometric
        # control variate is actually a control for.
        avg_idx_cv = np.arange(1, paths_plain.shape[1])
        cv_out = pas.asian_mc(
            paths_plain, K, "call", df, avg_idx_cv, control_variate=True
        )
        plain_asian = pas.asian_mc(
            paths_plain, K, "call", df, avg_idx_cv, control_variate=False
        )
        se_cv = cv_out["se"]
        se_plain_asian = plain_asian["se"]

        z = lib25.sobol_normals(n_p, n_steps_path, seed=SEED)
        dt = T / n_steps_path
        increments = (-0.5 * sigma**2 * dt) + sigma * np.sqrt(dt) * z
        sobol_paths_terminal = F * np.exp(np.cumsum(increments, axis=1)[:, -1])
        payoff_sobol = np.maximum(sobol_paths_terminal - K, 0.0) * df
        se_sobol = float(payoff_sobol.std(ddof=1) / np.sqrt(n_p))

        mc_error.append(
            {
                "n_paths": n_p,
                "se_plain": se_plain,
                "se_antithetic": se_anti,
                "se_control_variate": se_cv,
                # The control variate prices an ARITHMETIC ASIAN, not the
                # European the other three legs price, so its SE is only
                # meaningful against the plain Asian on the same payoff.
                "se_plain_asian": se_plain_asian,
                "cv_variance_reduction_x": float((se_plain_asian / se_cv) ** 2)
                if se_cv > 0
                else float("nan"),
                "se_sobol": se_sobol,
            }
        )
    charts["mc_error_vs_paths"] = mc_error

    # Discrete-vs-continuous barrier gap vs monitoring frequency
    gap_vs_freq = []
    for n_monitor in (12, 26, 52, 126, 252):
        idx_monitor = np.linspace(0, n_steps_path, n_monitor, dtype=int)
        paths_b = lib25.simulate_gbm(F, sigma, T, n_steps_path, n_paths_mc, seed=SEED)
        sub_paths = paths_b[:, idx_monitor]
        mc_raw = pba.barrier_mc(sub_paths, K, B_down, "do", "call", df)
        dt_g = T / n_monitor
        B_shift_g = pba.barrier_continuity_shift(B_down, sigma, dt_g, "do")
        analytic_do = pba.barrier_analytic(F, K, B_down, T, sigma, df, "call", "do")
        analytic_do_shifted = pba.barrier_analytic(
            F, K, B_shift_g, T, sigma, df, "call", "do"
        )
        gap_vs_freq.append(
            {
                "n_monitor": n_monitor,
                "raw_gap": float(mc_raw["price"] - analytic_do),
                "bgk_corrected_gap": float(mc_raw["price"] - analytic_do_shifted),
            }
        )
    charts["barrier_discrete_vs_continuous_gap"] = gap_vs_freq

    # ----------------------------------------------------------------- #
    report = {
        "seed": SEED,
        "n_paths_mc": n_paths_mc,
        "n_steps_path": n_steps_path,
        "scenario": {"F": F, "K": K, "T": T, "sigma": sigma, "r": r, "df": df},
        "matrix": results,
        "all_passed": all(c["passed"] for c in results),
        "charts": charts,
    }

    out = TMP / "phase_2_25_crossval.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out}")

    for c in results:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{status}  {c['name']}")
    print(f"\nALL PASSED: {report['all_passed']}")

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
