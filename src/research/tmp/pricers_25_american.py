"""American option pricing for bespoke commodity derivatives.

Three methods: binomial tree (Cox-Ross-Rubinstein for futures), finite-difference
PDE solver (Crank-Nicolson), and Longstaff-Schwartz Monte Carlo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np

_TMP_DIR = Path(__file__).resolve().parent
if str(_TMP_DIR) not in sys.path:
    sys.path.insert(0, str(_TMP_DIR))


OptType = Literal["call", "put"]


def american_binomial(
    F: float, K: float, T: float, sigma: float, r: float, opt: OptType, n_steps: int
) -> float:
    """Cox-Ross-Rubinstein binomial tree for an American futures option.

    The underlying is a futures price, so under the risk-neutral/futures measure
    the futures price itself is driftless (no drift term in the up-move probability).
    At each node, American value = max(intrinsic, discounted continuation value).

    Args:
        F: futures price
        K: strike
        T: time to expiry (years)
        sigma: volatility
        r: continuously-compounded annual rate
        opt: "call" or "put"
        n_steps: number of binomial steps

    Returns:
        American option price
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    dt = T / n_steps
    df_step = np.exp(-r * dt)  # discount factor per step

    # Handle zero or near-zero volatility
    if sigma <= 0:
        intrinsic = max(F - K, 0.0) if opt == "call" else max(K - F, 0.0)
        return df_step**n_steps * intrinsic

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (1.0 - d) / (u - d)  # risk-neutral probability (driftless)

    # Initialize value at expiry nodes (level n_steps)
    # Node j at level i corresponds to price F * u^j * d^(i-j)
    values = np.zeros(n_steps + 1)
    for j in range(n_steps + 1):
        f_node = F * (u**j) * (d ** (n_steps - j))
        if opt == "call":
            values[j] = max(f_node - K, 0.0)
        else:  # put
            values[j] = max(K - f_node, 0.0)

    # Work backward through the tree
    # At each time step, values[j] represents the option value at node j (j up-moves)
    # To get to node j at time i from time i+1:
    # - Node j can come from node j+1 at time i+1 via a DOWN move (prob 1-p)
    # - Node j can come from node j at time i+1 via an UP move (prob p)
    # Wait, let me reconsider: at time i+1, we have i+2 nodes (j=0..i+1)
    # At time i, node j (j up-moves) has price F*u^j*d^(i-j)
    # At time i+1, node j has price F*u^j*d^(i+1-j), and node j+1 has price F*u^(j+1)*d^(i-j)
    # Going from time i to i+1:
    # - From node j at time i, UP move goes to node j+1 at time i+1 (price gets multiplied by u)
    # - From node j at time i, DOWN move goes to node j at time i+1 (price gets multiplied by d)
    # So at time i, node j gets value from:
    # - Coming from node j at time i+1 (which was reached by UP from node j-1? No...)
    # Actually, the recombination happens both ways:
    # - Node j at time i+1 can be reached from node j at time i (via DOWN) or node j-1 at time i (via UP)
    # But we only have one value per node.
    # Let me re-think: we're at time i, node j. At time i+1:
    # - If we go UP, we go to node j+1
    # - If we go DOWN, we go to node j
    # So: V(i,j) = df * [p * V(i+1, j+1) + (1-p) * V(i+1, j)]
    for i in range(n_steps - 1, -1, -1):
        # Create a new array for this time step (to avoid overwriting values used in RHS)
        values_new = np.zeros(i + 1)
        for j in range(i + 1):
            # Continuation value: discounted average of up and down nodes
            # p is probability of UP move (j -> j+1), (1-p) is prob of DOWN move (j -> j)
            cont_value = df_step * (p * values[j + 1] + (1.0 - p) * values[j])
            # Current node price
            f_node = F * (u**j) * (d ** (i - j))
            if opt == "call":
                intrinsic = max(f_node - K, 0.0)
            else:  # put
                intrinsic = max(K - f_node, 0.0)
            # American early-exercise decision
            values_new[j] = max(intrinsic, cont_value)

        # Replace values array for next iteration
        values = values_new

    return float(values[0])


def american_pde(
    F: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    opt: OptType,
    n_s: int,
    n_t: int,
) -> float:
    """Finite-difference solver (implicit Euler) for an American futures option.

    Solves the log-transformed PDE on a uniform grid:
        dV/dt + 0.5*sigma^2*d2V/dx^2 - 0.5*sigma^2*dV/dx - r*V = 0
    where x = ln(F). The underlying is a futures price (zero drift).

    At each time step, applies the American early-exercise constraint:
    V = max(V, intrinsic) node-wise.

    Args:
        F: futures price
        K: strike
        T: time to expiry (years)
        sigma: volatility
        r: continuously-compounded annual rate
        opt: "call" or "put"
        n_s: number of space steps
        n_t: number of time steps

    Returns:
        American option price (at x = ln(F))
    """
    if n_s <= 0 or n_t <= 0:
        raise ValueError("n_s and n_t must be > 0")

    from scipy.linalg import solve_banded

    x_center = np.log(F)
    x_range = 6.0 * sigma * np.sqrt(max(T, 0.001))
    x_min = x_center - x_range
    x_max = x_center + x_range
    dx = (x_max - x_min) / n_s
    dt = T / n_t

    x = x_min + np.arange(n_s + 1) * dx
    F_vals = np.exp(x)
    if opt == "call":
        v = np.maximum(F_vals - K, 0.0).astype(float)
    else:  # put
        v = np.maximum(K - F_vals, 0.0).astype(float)

    sigma2 = sigma * sigma
    alpha = 0.5 * sigma2 * dt / (dx * dx)
    beta = 0.5 * sigma2 * dt / (2.0 * dx)
    # (I - dt*L) v_new = v_old, L v_i = alpha_notdt*(v_{i+1}-2v_i+v_{i-1})
    #                                    - beta_notdt*(v_{i+1}-v_{i-1}) - r*v_i
    lo = -(alpha + beta)  # coefficient of v[i-1]
    di = 1.0 + r * dt + 2.0 * alpha  # coefficient of v[i]
    up = -(alpha - beta)  # coefficient of v[i+1]

    n_pts = n_s + 1
    tau = 0.0
    for _step in range(n_t):
        tau += dt
        rhs = v.copy()

        ab = np.zeros((3, n_pts))
        ab[1, :] = di
        ab[0, 1:] = up
        ab[2, :-1] = lo

        # Dirichlet boundaries at the known deep-ITM/OTM asymptotics for the
        # EUROPEAN value there (the American constraint below still applies
        # at every interior node, so early exercise is still captured).
        if opt == "call":
            rhs[0] = 0.0
            rhs[-1] = F_vals[-1] - K * np.exp(-r * tau)
        else:
            rhs[0] = K * np.exp(-r * tau) - F_vals[0]
            rhs[-1] = 0.0
        ab[1, 0] = 1.0
        ab[0, 1] = 0.0
        ab[1, -1] = 1.0
        ab[2, -2] = 0.0

        v_new = solve_banded((1, 1), ab, rhs)

        intrinsic = (
            np.maximum(F_vals - K, 0.0)
            if opt == "call"
            else np.maximum(K - F_vals, 0.0)
        )
        v = np.maximum(v_new, intrinsic)

    return float(np.interp(x_center, x, v))


def american_lsm(
    paths: np.ndarray,
    K: float,
    opt: OptType,
    df: float,
    basis_degree: int = 3,
    seed: int = 25,
) -> dict:
    """Longstaff-Schwartz Monte Carlo for an American option (Bermudan approximation).

    `paths` has shape (n_paths, n_steps+1), where column 0 is the current price
    (all paths start at the same level), and each subsequent column is an
    exercise opportunity.

    Per-step discount factor is dstep = df ** (1/n_steps), which assumes a flat
    term structure implied by the overall df to the final time step.

    Algorithm:
    1. Initialize cashflow to intrinsic at expiry.
    2. Walk backward from column n_steps-1 to column 1.
    3. At each step, among in-the-money paths, regress discounted future cashflow
       on a polynomial basis in the price.
    4. Exercise where intrinsic > regression prediction.
    5. Discount cashflow by dstep after each step (all paths, exercised or not).
    6. Return price, dual upper bound (perfect-foresight), duality gap, and
       exercise boundary.

    Args:
        paths: shape (n_paths, n_steps+1) of simulated futures prices
        K: strike
        opt: "call" or "put"
        df: discount factor to the final time step
        basis_degree: polynomial degree for regression
        seed: random seed for regression

    Returns:
        Dict with keys:
        - 'price': LSM option price (low-biased)
        - 'dual_upper_bound': crude perfect-foresight upper bound
        - 'duality_gap': dual_upper_bound - price
        - 'exercise_boundary': array of (smallest ITM put / largest ITM call) prices
                                where exercise occurred, by time column (NaN where no exercise)
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1

    if n_steps <= 0:
        raise ValueError("paths must have at least 2 columns (initial + 1 step)")

    # Per-step discount factor
    dstep = df ** (1.0 / n_steps)

    # Initialize cashflow at expiry
    f_expiry = paths[:, -1]
    if opt == "call":
        cashflow = np.maximum(f_expiry - K, 0.0).astype(float)
    else:  # put
        cashflow = np.maximum(K - f_expiry, 0.0).astype(float)

    # Track exercise boundary: smallest (put) or largest (call) ITM price where exercised
    exercise_boundary = np.full(n_steps_plus_1, np.nan)

    # Walk backward from step n_steps-1 to step 1 (do not exercise at step 0 = now)
    for t in range(n_steps - 1, 0, -1):
        # Intrinsic value at this step
        f_t = paths[:, t]
        if opt == "call":
            intrinsic = np.maximum(f_t - K, 0.0)
        else:  # put
            intrinsic = np.maximum(K - f_t, 0.0)

        # Identify in-the-money paths
        itm = intrinsic > 0

        # Regression on ITM paths
        if np.any(itm):
            X = f_t[itm]
            # Build polynomial basis: [1, X, X^2, ..., X^degree]
            poly_features = np.column_stack([X**k for k in range(basis_degree + 1)])
            Y = cashflow[itm]  # Already discounted to time t+1; use as-is

            # Fit polynomial regression
            try:
                # Use least squares to avoid explicit inversion
                coeffs, _, _, _ = np.linalg.lstsq(poly_features, Y, rcond=None)
                # Regression prediction for all ITM paths
                continuation = np.polyval(coeffs[::-1], X)
            except np.linalg.LinAlgError:
                # Degenerate case: assume zero continuation value
                continuation = np.zeros_like(X)

            # Early-exercise decision: exercise where intrinsic > continuation
            exercise = itm.copy()
            exercise[itm] = intrinsic[itm] > continuation

            # Record exercise boundary: smallest (put) or largest (call) price where exercised
            if np.any(exercise):
                if opt == "call":
                    exercise_boundary[t] = np.max(f_t[exercise])
                else:  # put
                    exercise_boundary[t] = np.min(f_t[exercise])

            # Update cashflow: exercise where early exercise is optimal
            cashflow[exercise] = intrinsic[exercise]

        # Discount cashflow by one step (all paths, before next iteration)
        cashflow *= dstep

    # Option value at time 0
    price = float(np.mean(cashflow))

    # Dual upper bound: crude perfect-foresight approximation
    # Compute the discounted intrinsic at each time for each path, take the max per path,
    # then average. This is the value if we could exercise at the optimal time with perfect
    # hindsight (but using a simple discount schedule: dstep^t for t steps).
    perfect_foresight_values = np.zeros(n_paths)
    for p in range(n_paths):
        max_payoff = 0.0
        for t in range(1, n_steps + 1):
            f_t = paths[p, t]
            if opt == "call":
                intrinsic_t = max(f_t - K, 0.0)
            else:  # put
                intrinsic_t = max(K - f_t, 0.0)
            # Discount back to time 0: dstep^t
            discount = dstep**t
            payoff_t = discount * intrinsic_t
            max_payoff = max(max_payoff, payoff_t)
        perfect_foresight_values[p] = max_payoff

    dual_upper_bound = max(price, float(np.mean(perfect_foresight_values)))
    duality_gap = dual_upper_bound - price

    return {
        "price": price,
        "dual_upper_bound": dual_upper_bound,
        "duality_gap": duality_gap,
        "exercise_boundary": exercise_boundary,
    }
