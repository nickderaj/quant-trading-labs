"""Exotic commodity derivative pricers for teaching notebook 025.

Includes: Margrabe (exchange), Kirk (spread), Monte Carlo pricers, FX-adjusted
options (quanto, composite), variance-swap strikes, Markov-functional models,
and autocallable structures.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy import stats as sstats

# Setup path: lib25.py is in the same directory
_TMP_DIR = Path(__file__).resolve().parent
if str(_TMP_DIR) not in sys.path:
    sys.path.insert(0, str(_TMP_DIR))

import lib25


def margrabe(
    F1: float, F2: float, T: float, s1: float, s2: float, rho: float, df: float
) -> float:
    """Margrabe (1978) exchange option: payoff max(F1-F2, 0) at T.

    Uses the closed-form formula specialised to two driftless futures/forwards.
    Reduces to Black-76 with a spread volatility.

    Args:
        F1: Forward/futures price of asset 1 (numeraire in payoff).
        F2: Forward/futures price of asset 2 (denominator).
        T: Time to maturity (years).
        s1: Volatility of asset 1.
        s2: Volatility of asset 2.
        rho: Correlation between asset 1 and asset 2.
        df: Discount factor.

    Returns:
        Price of the exchange option.
    """
    if T <= 0:
        return df * max(F1 - F2, 0.0)

    # Spread volatility: sqrt(s1^2 + s2^2 - 2*rho*s1*s2)
    sigma_spread = np.sqrt(s1**2 + s2**2 - 2 * rho * s1 * s2)

    # Use Black-76 on F1 vs F2 with spread volatility
    return lib25.black76(F1, F2, T, sigma_spread, df, "call")


def kirk(
    F1: float,
    F2: float,
    K: float,
    T: float,
    s1: float,
    s2: float,
    rho: float,
    df: float,
    opt: lib25.OptType,
) -> float:
    """Kirk (1995) approximation for a spread option with nonzero strike.

    Payoff: max(F1-F2-K, 0) (call) or max(K-(F1-F2), 0) (put).

    Args:
        F1: Forward/futures price of asset 1.
        F2: Forward/futures price of asset 2.
        K: Strike on the spread (not on individual assets).
        T: Time to maturity (years).
        s1: Volatility of asset 1.
        s2: Volatility of asset 2.
        rho: Correlation between asset 1 and asset 2.
        df: Discount factor.
        opt: "call" or "put".

    Returns:
        Price of the spread option.
    """
    if T <= 0:
        intrinsic = max(F1 - F2 - K, 0.0) if opt == "call" else max(K - (F1 - F2), 0.0)
        return df * intrinsic

    F2_adj = F2 + K
    if F2_adj <= 0:
        # Fall back to intrinsic: avoid log(non-positive)
        intrinsic = max(F1 - F2 - K, 0.0) if opt == "call" else max(K - (F1 - F2), 0.0)
        return df * intrinsic

    # Kirk's approximation: adjusted spread volatility
    sigma_kirk = np.sqrt(
        s1**2 + (s2 * F2 / F2_adj) ** 2 - 2 * rho * s1 * s2 * (F2 / F2_adj)
    )

    # Price as a call on F1 vs F2_adj strike
    call_price = lib25.black76(F1, F2_adj, T, sigma_kirk, df, "call")

    if opt == "call":
        return call_price
    else:
        # Put-call parity: put = call - df * (F1 - F2_adj)
        return call_price - df * (F1 - F2_adj)


def spread_mc(
    paths_1: np.ndarray,
    paths_2: np.ndarray,
    K: float,
    opt: lib25.OptType,
    df: float,
) -> dict:
    """Bivariate Monte Carlo pricer for a spread option.

    Payoff: max(spread - K, 0) (call) or max(K - spread, 0) (put).

    Args:
        paths_1: (n_paths, n_steps+1) price paths for asset 1.
        paths_2: (n_paths, n_steps+1) price paths for asset 2.
        K: Strike on the spread.
        opt: "call" or "put".
        df: Discount factor.

    Returns:
        Dict with 'price' (mean), 'se' (standard error), 'n' (n_paths).
    """
    paths_1 = np.asarray(paths_1)
    paths_2 = np.asarray(paths_2)
    n_paths = paths_1.shape[0]

    # Terminal values
    S1_T = paths_1[:, -1]
    S2_T = paths_2[:, -1]
    spread = S1_T - S2_T

    if opt == "call":
        payoffs = np.maximum(spread - K, 0.0)
    else:
        payoffs = np.maximum(K - spread, 0.0)

    # Discount
    discounted = payoffs * df

    mean_price = np.mean(discounted)
    std_price = np.std(discounted, ddof=1)
    se = std_price / np.sqrt(n_paths)

    return {"price": float(mean_price), "se": float(se), "n": n_paths}


def quanto_black76(
    F: float,
    K: float,
    T: float,
    sigma: float,
    sigma_fx: float,
    rho: float,
    df_foreign: float,
    opt: lib25.OptType,
) -> float:
    """Quanto (fixed-FX) adjustment on a Black-76 option.

    The commodity is priced in its natural (foreign) currency but settled at a
    FIXED exchange rate. This shifts the effective drift by -rho*sigma*sigma_fx*T
    under the domestic-currency-hedged measure.

    Args:
        F: Forward price in foreign currency.
        K: Strike in foreign currency.
        T: Time to maturity (years).
        sigma: Volatility of the commodity.
        sigma_fx: Volatility of the FX rate.
        rho: Correlation between commodity and FX.
        df_foreign: Discount factor (foreign currency).
        opt: "call" or "put".

    Returns:
        Price of the quanto option (in domestic currency units).
    """
    # Adjust forward: F_quanto = F * exp(-rho * sigma * sigma_fx * T)
    F_quanto = F * np.exp(-rho * sigma * sigma_fx * T)

    # Price using adjusted forward
    return lib25.black76(F_quanto, K, T, sigma, df_foreign, opt)


def composite_black76(
    F: float,
    K_foreign: float,
    T: float,
    sigma: float,
    sigma_fx: float,
    rho: float,
    df_foreign: float,
    opt: lib25.OptType,
) -> float:
    """Composite (floating-FX) option with foreign-currency strike.

    The strike is fixed in foreign currency and payoff converts through
    the floating FX rate. The effective volatility combines commodity and FX vols.

    Args:
        F: Forward price in foreign currency.
        K_foreign: Strike in foreign currency.
        T: Time to maturity (years).
        sigma: Volatility of the commodity.
        sigma_fx: Volatility of the FX rate.
        rho: Correlation between commodity and FX.
        df_foreign: Discount factor (foreign currency).
        opt: "call" or "put".

    Returns:
        Price of the composite option (in domestic currency units).
    """
    # Combined volatility
    sigma_composite = np.sqrt(sigma**2 + sigma_fx**2 + 2 * rho * sigma * sigma_fx)

    # Price with combined volatility
    return lib25.black76(F, K_foreign, T, sigma_composite, df_foreign, opt)


def varswap_strike(
    sigma_fn: Callable[[float], float],
    T: float,
    n_strikes: int | None = None,
) -> dict:
    """Fair variance-swap strike as a FORECAST (not a replicated strike).

    Approximates the integrated variance by evaluating sigma_fn on a grid and
    averaging sigma(t)^2. This is a teaching implementation (no option strip
    exists in this repo to replicate with).

    Args:
        sigma_fn: Callable vol term structure, sigma_fn(t) -> volatility.
        T: Maturity (years).
        n_strikes: Number of grid points (default 50).

    Returns:
        Dict with 'strike_vol' (sqrt of mean variance), 'strike_variance',
        'T' (maturity), 'n_strikes' (number of points used).
    """
    if n_strikes is None:
        n_strikes = 50

    # Grid of time points in (0, T]
    times = np.linspace(1e-6, T, n_strikes)

    # Evaluate volatility at each point
    sigmas = np.asarray([sigma_fn(t) for t in times])

    # Mean of sigma^2 (Riemann sum approximation to integrated variance)
    mean_variance = np.mean(sigmas**2)

    # Strike vol
    strike_vol = np.sqrt(mean_variance)

    return {
        "strike_vol": float(strike_vol),
        "strike_variance": float(mean_variance),
        "T": float(T),
        "n_strikes": n_strikes,
    }


def markov_functional_1f(
    marginals: dict[float, np.ndarray],
    driver_grid: np.ndarray,
) -> Callable:
    """One-factor Markov-functional model calibrated to realized terminal marginals.

    Builds monotone quantile mappings from standard-normal driver to empirical
    distributions for each calibrated maturity. This is a P-measure object
    (not risk-neutral), a teaching implementation that only reprices its
    calibration marginals exactly.

    Args:
        marginals: Dict {maturity_T: np.ndarray_of_samples}.
        driver_grid: Suggested grid of Z values (for API completeness; can be ignored).

    Returns:
        Callable driver_to_value(T, z) that maps (T, z) -> F(T, z).
        Raises ValueError if T is not a calibrated maturity.
    """
    driver_grid = np.asarray(driver_grid)

    # Build per-maturity interpolators
    interpolators = {}
    for T, samples in marginals.items():
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        # Empirical quantiles
        empirical_quantiles = (np.arange(1, n + 1) - 0.5) / n
        z_of_quantile = sstats.norm.ppf(empirical_quantiles)
        # Store for later interpolation
        interpolators[T] = (z_of_quantile, sorted_samples)

    def driver_to_value(T: float, z: float | np.ndarray) -> np.ndarray:
        """Evaluate the mapping at maturity T and driver value(s) z."""
        if T not in interpolators:
            raise ValueError(
                f"Maturity {T} not in calibration set: {list(interpolators.keys())}"
            )

        z_of_quantile, sorted_samples = interpolators[T]
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        # Interpolate (both arrays are sorted/monotone by construction)
        return np.interp(z_arr, z_of_quantile, sorted_samples)

    return driver_to_value


def autocallable(
    paths: np.ndarray,
    coupon: float,
    autocall_level: float,
    barrier: float,
    obs_idx: np.ndarray,
    df: float,
) -> dict:
    """Autocallable reverse convertible structure.

    Initial level normalized to 1.0 (normalizes internally if needed).
    On each observation date, if the level >= autocall_level and not yet
    autocalled, redeem early at 1 + coupon*(fraction of dates observed).
    At final date, if never autocalled: payoff = level if level < barrier,
    else 1 + coupon (principal protected above barrier).

    Args:
        paths: (n_paths, n_steps+1) price paths. If paths[:,0] != 1, normalizes.
        coupon: Annual coupon rate.
        autocall_level: Level at which early redemption is triggered.
        barrier: Level below which principal is not protected at maturity.
        obs_idx: Sorted array of observation date column indices.
        df: Discount factor for the full maturity.

    Returns:
        Dict with 'price', 'se', 'n', 'autocall_frequency_by_obs',
        'never_autocalled_frequency'.
    """
    paths = np.asarray(paths)
    obs_idx = np.asarray(obs_idx, dtype=int)
    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    # Normalize paths to start at 1.0
    initial_levels = paths[:, 0:1]
    paths_norm = paths / initial_levels

    # Track which paths have autocalled and at which observation
    autocalled = np.zeros(n_paths, dtype=bool)
    redemption_step = np.full(n_paths, n_steps, dtype=int)  # Default: maturity

    # Autocall on observation dates (in order)
    autocall_freq = np.zeros(len(obs_idx))
    for i, obs_step in enumerate(obs_idx[:-1]):  # All but the final observation
        # Check which paths are at autocall level and haven't autocalled yet
        to_call = (~autocalled) & (paths_norm[:, obs_step] >= autocall_level)
        autocalled[to_call] = True
        redemption_step[to_call] = obs_step
        autocall_freq[i] = np.sum(to_call) / n_paths

    # Handle the final observation for paths that never autocalled
    final_obs_step = obs_idx[-1]
    final_levels = paths_norm[:, final_obs_step]

    # Paths that autocalled early
    early_payoffs = np.zeros(n_paths)
    early_payoffs[autocalled] = 1.0 + coupon * (
        (np.searchsorted(obs_idx, redemption_step[autocalled]) + 1) / len(obs_idx)
    )

    # Paths that reach final observation without autocalling
    at_maturity = ~autocalled
    final_payoffs = np.where(
        final_levels[at_maturity] >= barrier,
        1.0 + coupon,  # Principal protected
        final_levels[at_maturity],  # Principal loss
    )

    payoffs = np.zeros(n_paths)
    payoffs[autocalled] = early_payoffs[autocalled]
    payoffs[at_maturity] = final_payoffs

    # Discount: use redemption_step / n_steps as fraction of time
    discount_factors = df ** (redemption_step / n_steps)
    discounted = payoffs * discount_factors

    mean_price = np.mean(discounted)
    std_price = np.std(discounted, ddof=1)
    se = std_price / np.sqrt(n_paths)

    never_autocalled = np.sum(~autocalled) / n_paths

    return {
        "price": float(mean_price),
        "se": float(se),
        "n": n_paths,
        "autocall_frequency_by_obs": autocall_freq,
        "never_autocalled_frequency": float(never_autocalled),
    }
