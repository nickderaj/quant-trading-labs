"""Barrier option pricing for bespoke commodity derivatives (notebook 025).

Implements closed-form (Reiner-Rubinstein), Monte Carlo, and finite-difference
pricing for four barrier types: down-and-out, down-and-in, up-and-out, up-and-in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import linalg as sp_linalg
from scipy import stats as sstats

_TMP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TMP_DIR))

import lib25

OptType = Literal["call", "put"]
BarrierKind = Literal["do", "di", "uo", "ui"]


def barrier_analytic(
    F: float,
    K: float,
    B: float,
    T: float,
    sigma: float,
    df: float,
    opt: OptType,
    kind: BarrierKind,
) -> float:
    """Reiner-Rubinstein closed-form barrier option price.

    Args:
        F: Futures price
        K: Strike price
        B: Barrier level
        T: Time to expiry (years)
        sigma: Volatility (annualized)
        df: Discount factor
        opt: "call" or "put"
        kind: "do" (down-out), "di" (down-in), "uo" (up-out), "ui" (up-in)

    Returns:
        Option price (knock-out or knock-in depending on kind).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Check if barrier already breached
    breached = (F <= B and kind in ("do", "di")) or (F >= B and kind in ("uo", "ui"))

    if breached:
        # Barrier already hit: knock-out is worthless, knock-in equals vanilla
        vanilla = lib25.black76(F, K, T, sigma, df, opt)
        if kind in ("do", "uo"):
            return 0.0
        else:  # knock-in
            return vanilla

    # Compute knock-in price, then derive knock-out if needed
    if kind in ("do", "uo"):
        # For knock-out, compute knock-in and subtract from vanilla
        knock_in_kind: Literal["di", "ui"] = "di" if kind == "do" else "ui"
        knock_in_price = _barrier_knock_in(F, K, B, T, sigma, df, opt, knock_in_kind)
        vanilla = lib25.black76(F, K, T, sigma, df, opt)
        return vanilla - knock_in_price
    else:
        # For knock-in, compute directly
        knock_in_kind_2: Literal["di", "ui"] = "di" if kind == "di" else "ui"
        return _barrier_knock_in(F, K, B, T, sigma, df, opt, knock_in_kind_2)


def _barrier_knock_in(
    F: float,
    K: float,
    B: float,
    T: float,
    sigma: float,
    df: float,
    opt: OptType,
    kind: Literal["di", "ui"],
) -> float:
    """Compute knock-in price using Reiner-Rubinstein formula.

    Args:
        kind: "di" (down-in) or "ui" (up-in)
    """
    sT = sigma * np.sqrt(T)
    mu = -0.5

    # Define the static functions A, B, C, D from Reiner-Rubinstein
    N = sstats.norm.cdf

    x1 = np.log(F / K) / sT + (1 + mu) * sT
    x2 = np.log(F / B) / sT + (1 + mu) * sT
    y1 = np.log(B**2 / (F * K)) / sT + (1 + mu) * sT
    y2 = np.log(B / F) / sT + (1 + mu) * sT

    def A(phi):
        return phi * F * df * N(phi * x1) - phi * K * df * N(phi * x1 - phi * sT)

    def Bterm(phi):
        return phi * F * df * N(phi * x2) - phi * K * df * N(phi * x2 - phi * sT)

    def C(phi, eta):
        H = B
        return phi * F * df * (H / F) ** (2 * (mu + 1)) * N(eta * y1) - phi * K * df * (
            H / F
        ) ** (2 * mu) * N(eta * y1 - eta * sT)

    def D(phi, eta):
        H = B
        return phi * F * df * (H / F) ** (2 * (mu + 1)) * N(eta * y2) - phi * K * df * (
            H / F
        ) ** (2 * mu) * N(eta * y2 - eta * sT)

    # phi = 1 for call, -1 for put
    # eta = 1 for down barriers, -1 for up barriers
    phi_val = 1 if opt == "call" else -1
    eta_val = 1 if kind == "di" else -1

    if opt == "call":
        if kind == "di":
            knock_in = (
                C(phi_val, eta_val)
                if K >= B
                else (A(phi_val) - Bterm(phi_val) + D(phi_val, eta_val))
            )
        else:  # "ui"
            knock_in = (
                A(phi_val)
                if K > B
                else (Bterm(phi_val) - C(phi_val, eta_val) + D(phi_val, eta_val))
            )
    else:  # put
        if kind == "di":
            knock_in = (
                (Bterm(phi_val) - C(phi_val, eta_val) + D(phi_val, eta_val))
                if K > B
                else A(phi_val)
            )
        else:  # "ui"
            knock_in = A(phi_val) if K > B else C(phi_val, eta_val)

    return float(knock_in)


def barrier_continuity_shift(
    B: float,
    sigma: float,
    dt: float,
    kind: BarrierKind,
) -> float:
    """Broadie-Glasserman-Kou discretization correction.

    For discrete monitoring, adjust the barrier level to account for the drift
    between monitoring points in continuous monitoring.

    Args:
        B: Barrier level
        sigma: Volatility
        dt: Time step (years)
        kind: "do", "di", "uo", "ui"

    Returns:
        Adjusted barrier level.
    """
    correction = 0.5826 * sigma * np.sqrt(dt)
    if kind in ("do", "di"):  # Down barriers
        return B * np.exp(-correction)
    else:  # Up barriers
        return B * np.exp(correction)


def barrier_mc(
    paths: np.ndarray,
    K: float,
    B: float,
    kind: BarrierKind,
    opt: OptType,
    df: float,
    monitor: str = "daily",
) -> dict:
    """Monte Carlo barrier option pricing.

    Args:
        paths: Shape (n_paths, n_steps+1), paths[i, 0] = F0
        K: Strike price
        B: Barrier level
        kind: "do", "di", "uo", "ui"
        opt: "call" or "put"
        df: Discount factor (or callable)
        monitor: "daily" (check every column)

    Returns:
        {"price": mean, "se": std_err, "n": n_paths}
    """
    n_paths = paths.shape[0]

    # Check barrier hits for each path
    if kind in ("do", "di"):
        hits = np.any(paths <= B, axis=1)
    else:  # "uo", "ui"
        hits = np.any(paths >= B, axis=1)

    # Determine which paths are alive for payout
    if kind in ("do", "uo"):  # Knock-out: payoff if NOT hit
        alive = ~hits
    else:  # Knock-in: payoff if hit
        alive = hits

    # Compute payoff at terminal time
    terminal = paths[:, -1]
    if opt == "call":
        payoff = np.maximum(terminal - K, 0.0)
    else:  # put
        payoff = np.maximum(K - terminal, 0.0)

    # Apply barrier condition
    payoff = payoff * alive

    # Discount (assume scalar df for now)
    pv = payoff * df
    price = np.mean(pv)
    se = np.std(pv) / np.sqrt(n_paths)

    return {"price": float(price), "se": float(se), "n": n_paths}


def barrier_pde(
    F: float,
    K: float,
    B: float,
    T: float,
    sigma: float,
    r: float,
    opt: OptType,
    kind: BarrierKind,
    n_s: int,
    n_t: int,
) -> float:
    """Implicit (backward-Euler) finite-difference solver for a barrier
    option, on x=ln(S), backward in tau=T-t: dV/dtau = 0.5*sigma^2*Vxx -
    0.5*sigma^2*Vx - r*V. The grid is built so the barrier sits EXACTLY on a
    node (never just the nearest one) -- placing a barrier off-node biases
    a finite-difference barrier price by an amount that scales with the grid
    spacing and looks like a convergence problem when it is really a
    discretisation-placement bug. Domain edges use Dirichlet boundaries at
    the known deep-ITM/OTM asymptotics.

    Args:
        F: Initial futures price
        K: Strike
        B: Barrier level
        T: Time to expiry
        sigma: Volatility
        r: Risk-free rate
        opt: "call" or "put"
        kind: "do", "di", "uo", "ui"
        n_s: Number of space grid points (approximate; the grid is snapped so
            the barrier lands on a node, so the actual count may differ
            slightly)
        n_t: Number of time steps

    Returns:
        Barrier option price (knock-out solved directly; knock-in derived via
        vanilla - knock-out, consistent with `barrier_analytic`).
    """
    x_F = np.log(F)
    x_B = np.log(B)
    half_width = 6 * sigma * np.sqrt(T)
    raw_dx = 2 * half_width / max(n_s - 1, 1)

    n_to_barrier = max(round(abs(x_F - x_B) / raw_dx), 1)
    dx = abs(x_F - x_B) / n_to_barrier

    k_min = int(np.floor((x_F - half_width - x_B) / dx))
    k_max = int(np.ceil((x_F + half_width - x_B) / dx))
    ks = np.arange(k_min, k_max + 1)
    x = x_B + ks * dx
    n_s_actual = len(x)
    barrier_idx = int(np.argmin(np.abs(ks)))
    x_F_idx = int(np.argmin(np.abs(x - x_F)))

    S = np.exp(x)
    V = np.maximum(S - K, 0.0) if opt == "call" else np.maximum(K - S, 0.0)

    dt = T / n_t
    nu = 0.5 * sigma**2
    alpha = nu * dt / (dx**2)
    beta = nu * dt / (2 * dx)
    # (I - dt*L) V_new = V_old, with L V_i = alpha*(V_{i+1}-2V_i+V_{i-1})
    #                                        - beta*(V_{i+1}-V_{i-1}) - r*V_i
    a = -(alpha + beta)  # coefficient of V[i-1]
    b = 1.0 + r * dt + 2 * alpha  # coefficient of V[i]
    c = -(alpha - beta)  # coefficient of V[i+1]

    # Always solve the KNOCK-OUT PDE (barrier enforced); knock-in kinds
    # derive from parity below. 'di'->'do', 'ui'->'uo' -- the barrier
    # direction ('down' vs 'up') is the same for a kind and its in/out pair.
    ko_kind = "do" if kind in ("do", "di") else "uo"

    tau = 0.0
    for _step in range(n_t):
        tau += dt
        rhs = V.copy()

        ab = np.zeros((3, n_s_actual))
        ab[1, :] = b
        ab[0, 1:] = c
        ab[2, :-1] = a

        if opt == "call":
            rhs[0] = 0.0
            rhs[-1] = S[-1] - K * np.exp(-r * tau)
        else:
            rhs[0] = K * np.exp(-r * tau) - S[0]
            rhs[-1] = 0.0
        ab[1, 0] = 1.0
        ab[0, 1] = 0.0
        ab[1, -1] = 1.0
        ab[2, -2] = 0.0

        ab[1, barrier_idx] = 1.0
        if barrier_idx > 0:
            ab[2, barrier_idx - 1] = 0.0
        if barrier_idx < n_s_actual - 1:
            ab[0, barrier_idx + 1] = 0.0
        rhs[barrier_idx] = 0.0

        V = sp_linalg.solve_banded((1, 1), ab, rhs)

        if ko_kind == "do":
            V[: barrier_idx + 1] = 0.0
        else:
            V[barrier_idx:] = 0.0

    price = V[x_F_idx]

    if kind in ("di", "ui"):
        vanilla = lib25.black76(F, K, T, sigma, np.exp(-r * T), opt)
        price = vanilla - price

    return float(price)


def ko_collar(
    F: float,
    K_put: float,
    K_call: float,
    B: float,
    T: float,
    sigma: float,
    df: float,
) -> dict:
    """Down-and-out put + short vanilla call collar.

    Args:
        F: Futures price
        K_put: Put strike
        K_call: Call strike
        B: Down-and-out barrier
        T: Time to expiry
        sigma: Volatility
        df: Discount factor

    Returns:
        {"premium": net_premium, "legs": [leg1, leg2, ...]}
    """
    do_put = barrier_analytic(F, K_put, B, T, sigma, df, "put", "do")
    call = lib25.black76(F, K_call, T, sigma, df, "call")

    premium = do_put - call

    return {
        "premium": float(premium),
        "legs": [
            {
                "instrument": "down-and-out put",
                "strike": K_put,
                "barrier": B,
                "price": do_put,
            },
            {"instrument": "short vanilla call", "strike": K_call, "price": -call},
        ],
    }
