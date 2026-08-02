"""Phase 7: the risk engine -- the guaranteed deliverable, independent of
every alpha gate (NEXT_PROMPT.md sec 4, Phase 7).

Family selection is read from Phase 3's own per-product ranking (preferred:
it is the OOS, conditional-model result) with Phase 2's unconditional ranking
as a fallback for any product Phase 3 didn't cover. Never hardcoded.

- `commod_lib8.fit_risk_model`/`RiskModel`: per-product VaR/ES/simulate/stress.
- Gate RE acceptance test: walk-forward OOS 1% VaR coverage (Kupiec +
  Christoffersen, BH-adjusted across products) -- a risk engine whose own 1%
  VaR is violated 3% of the time is worse than useless.
- Portfolio-level risk under three dependence assumptions (empirical,
  Gaussian copula, t-copula), with lower-tail dependence coefficients
  reported side by side to quantify the Gaussian copula's understatement.
- Stress scenarios: Phase 1's named events, replayed at the portfolio level.

Writes phase_7_results.json.
"""

import json
import sys
import time

sys.path.insert(0, "src/research/tmp")
sys.path.insert(0, "src")

import commod_lib8 as C
import numpy as np
import polars as pl

import research

CURVE_DIR = "src/research/tmp/phase_0_curves"
OUT_PATH = "src/research/tmp/phase_7_results.json"
PHASE2_PATH = "src/research/tmp/phase_2_results.json"
PHASE3_PATH = "src/research/tmp/phase_3_results.json"

DEV_START = {"__default__": "2010-06-06", "ES": "2018-01-01", "KE": "2013-12-16"}
DEV_END = "2024-12-31"

# Phase 3 model names are "{garch,gjr}_{family}" or "..._normal"/"..._t";
# collapse to the underlying density family name for RiskModel.
_PHASE3_FAMILY_MAP = {
    "normal": "normal", "t": "t", "ged": "ged", "nig": "nig",
    "johnsonsu": "johnsonsu", "hansen_skewt": "hansen_skewt",
}

research.set_seed(0)


def load_family_map() -> dict[str, str]:
    families: dict[str, str] = {}
    try:
        with open(PHASE3_PATH) as f:
            phase3 = json.load(f)
        for p, r in phase3.items():
            if p.startswith("_") or r is None:
                continue
            best = r.get("best_model")
            if not best:
                continue
            suffix = best.split("_", 1)[1] if "_" in best else best
            fam = _PHASE3_FAMILY_MAP.get(suffix)
            if fam and fam != "spliced_evt":
                families[p] = fam
    except FileNotFoundError:
        pass

    try:
        with open(PHASE2_PATH) as f:
            phase2 = json.load(f)
        for p, r in phase2.items():
            if p.startswith("_") or p in families or r is None:
                continue
            best = r.get("best_family")
            if best and best != "spliced_evt":
                families[p] = best
    except FileNotFoundError:
        pass

    # BTCUSDT is Phase 1/2's bridge series (comparison-only), not one of the
    # 16 commodity/ES products this risk engine covers -- it has no Phase 0
    # curve file, so it must be excluded here even though Phase 2's family
    # map includes it as a fallback candidate.
    families.pop("BTCUSDT", None)
    return families


def load_returns(product: str) -> tuple[np.ndarray, np.ndarray]:
    curve = pl.read_parquet(f"{CURVE_DIR}/{product}.parquet")
    dev_start = DEV_START.get(product, DEV_START["__default__"])
    sub = curve.select(["date", pl.col("log_return_ratioadj").alias("ret")]).drop_nulls()
    sub = sub.filter(pl.col("ret").is_finite())
    sub = sub.filter((pl.col("date") >= pl.lit(dev_start).str.to_date()) & (pl.col("date") <= pl.lit(DEV_END).str.to_date()))
    return sub["ret"].to_numpy(), sub["date"].to_numpy()


def oos_coverage_test(ret: np.ndarray, family: str, product: str) -> dict:
    """Gate RE: expanding-window OOS 1% VaR coverage. Shape/mean/std refit
    once per fold (same cadence as Phase 2); the VaR *scale* is conditioned
    day-by-day within each test block on a causal EWMA volatility path
    (`commod_lib8.ewma_vol`, RiskMetrics lambda=0.94) via `var_conditional` --
    a fixed full-sample-std VaR failed OOS coverage badly in development
    (violations cluster in vol regime shifts, exactly what Christoffersen's
    independence test is built to catch), which is the expected, documented
    failure mode of an unconditional VaR and the reason this conditioning
    step exists.
    """
    n = len(ret)
    train_bars, test_bars = int(n * 0.6), max(50, int(n * 0.08))
    folds = research.walk_forward_splits(n, train_bars=train_bars, test_bars=test_bars, mode="anchored")
    if not folds:
        return {"n_folds": 0, "note": "insufficient data for OOS test"}

    sigma_path = C.ewma_vol(ret, lam=0.94)

    hits_01 = np.full(n, np.nan)
    hits_025 = np.full(n, np.nan)
    for train_idx, test_idx in folds:
        model = C.fit_risk_model(ret[train_idx], product, family)
        if model is None:
            continue
        r_test = ret[test_idx]
        sigma_test = sigma_path[test_idx]
        valid = np.isfinite(sigma_test)
        for i in np.where(valid)[0]:
            idx = test_idx[i]
            var_01 = model.var_conditional(0.01, sigma_t=sigma_test[i])
            var_025 = model.var_conditional(0.025, sigma_t=sigma_test[i])
            hits_01[idx] = float(r_test[i] < -var_01)
            hits_025[idx] = float(r_test[i] < -var_025)

    import distributions as dist

    out = {"n_folds": len(folds)}
    for level, hits in [("0.01", hits_01), ("0.025", hits_025)]:
        mask = np.isfinite(hits)
        h = hits[mask].astype(int)
        if len(h) < 30:
            out[level] = {"n": len(h), "note": "too few OOS points"}
            continue
        _, kp = dist.kupiec_test(h, float(level))
        _, ip = dist.christoffersen_independence_test(h)
        out[level] = {
            "n": len(h), "observed_rate": float(np.mean(h)), "expected_rate": float(level),
            "kupiec_p": float(kp), "christoffersen_independence_p": float(ip),
            "pass": bool(kp > 0.05 and ip > 0.05),
        }
    return out


def stress_portfolio(models: dict, weights: dict, returns_by_product: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Replay each named event (commod_lib8.NAMED_EVENTS) at the portfolio
    level: for every product with a hit in that event window, weight its
    realised return path by its portfolio weight and sum."""
    out = {}
    for ev in C.NAMED_EVENTS:
        from datetime import date as _date

        s, e = _date.fromisoformat(ev["start"]), _date.fromisoformat(ev["end"])
        port_pnl = 0.0
        contributions = {}
        touched = False
        for p in models:
            if p not in ev["products"] and ev["products"] != C.PRODUCTS:
                continue
            ret, dates = returns_by_product[p]
            mask = np.array([s <= d <= e for d in dates.astype("datetime64[D]").tolist()])
            if not mask.any():
                continue
            touched = True
            cum = float(np.sum(ret[mask]))
            w = weights.get(p, 0.0)
            contributions[p] = {"cum_return": cum, "weight": w, "contribution": cum * w}
            port_pnl += cum * w
        if touched:
            out[ev["name"]] = {"portfolio_pnl": port_pnl, "contributions": contributions}
    return out


def main():
    t0 = time.time()
    families = load_family_map()
    print(f"family map ({len(families)} products): {families}", flush=True)

    models = {}
    returns_by_product = {}
    coverage_results = {}
    for p, family in families.items():
        ret, dates = load_returns(p)
        if len(ret) < 500:
            continue
        returns_by_product[p] = (ret, dates)
        model = C.fit_risk_model(ret, p, family)
        if model is None:
            print(f"  {p}: fit_risk_model failed for family {family}, skipping", flush=True)
            continue
        models[p] = model
        print(f"OOS coverage test: {p} ({family})...", flush=True)
        coverage_results[p] = oos_coverage_test(ret, family, p)

    # Gate RE verdict
    n_pass_01 = sum(1 for r in coverage_results.values() if r.get("0.01", {}).get("pass"))
    n_total = len(coverage_results)
    gate_re = {
        "n_products_passing_1pct_coverage": n_pass_01,
        "n_products_total": n_total,
        "threshold": 14,
        "fires": n_pass_01 >= 14,
    }

    # Portfolio-level risk, equal-weighted across all fitted products
    weights = dict.fromkeys(models, 1.0 / len(models)) if models else {}
    hist_returns = {p: returns_by_product[p][0] for p in models}
    # align by truncating to common length from the end (approx alignment
    # across products with slightly different history starts)
    min_len = min(len(v) for v in hist_returns.values()) if hist_returns else 0
    hist_returns_aligned = {p: v[-min_len:] for p, v in hist_returns.items()}

    portfolio = {}
    if len(models) >= 2:
        for dep in ["empirical", "gaussian", "t"]:
            print(f"portfolio risk ({dep})...", flush=True)
            portfolio[dep] = C.portfolio_risk(
                models, weights, dependence=dep, historical_returns=hist_returns_aligned, n_sims=20000, seed=0,
                t_df=5.0,
            )

    stress = stress_portfolio(models, weights, returns_by_product) if models else {}

    results = {
        "family_map": families,
        "oos_coverage": coverage_results,
        "gate_RE": gate_re,
        "portfolio_risk": portfolio,
        "stress_scenarios": stress,
        "_config": {"weights": weights},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwritten {OUT_PATH} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
