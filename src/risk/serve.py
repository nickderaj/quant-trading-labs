"""The dashboard's data producer (NEXT_PROMPT.md sec 7.2): one function,
`build_snapshot`, producing the single JSON document the dashboard reads.
No server, no API framework, no database.

Reads from `risk.ingest`'s output (`src/risk/data/*.parquet` by default --
run `risk.ingest.refresh()` first, or pass `data_dir=` explicitly); this
module never reaches into `research/tmp/`.

NEXT_PROMPT.md sec 7.4: this module is explicitly permitted to display
current dates, including dates inside and after the spent futures holdout
window (2025-01-01 onward) -- it fits no model on that data, chooses no
threshold from it, and makes no gate decision. It only *evaluates* the
already-frozen (development-window-fit) models forward, which is what 008
Phase 8 already did once and recorded. What is forbidden, and what this
module does not do, is feed any displayed number back into a fitting,
selection, or threshold decision.
"""

from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from risk import ingest
from risk.calibration import CalibrationMonitor
from risk.families import load_family_map
from risk.model import RiskModel, ewma_vol, fit_risk_model
from risk.portfolio import portfolio_risk

__all__ = ["build_snapshot", "render_dashboard", "write_snapshot"]

LEVELS = (0.01, 0.025)
HORIZONS = (1, 5, 10)
RECENT_SERIES_DAYS = 250
PRODUCTS = [
    "CL", "BZ", "NG", "HO", "RB", "GC", "SI", "PL", "PA",
    "ZC", "ZW", "KE", "ZS", "ZL", "ZM", "ES",
]  # fmt: skip

# Mirrors commod_lib8.NAMED_EVENTS (NEXT_PROMPT.md sec 1 Phase 1) -- risk/
# does not import from research/tmp/ (sec 12), so this reference-data table
# (not logic) is duplicated here, not imported.
NAMED_EVENTS: list[dict[str, Any]] = [
    {
        "name": "2011 Libya supply shock",
        "start": "2011-02-15",
        "end": "2011-04-30",
        "products": ["CL", "BZ"],
    },
    {
        "name": "2014-15 OPEC collapse",
        "start": "2014-11-01",
        "end": "2015-03-01",
        "products": ["CL", "BZ", "RB", "HO"],
    },
    {
        "name": "2020-04-20 negative WTI",
        "start": "2020-04-15",
        "end": "2020-04-25",
        "products": ["CL"],
    },
    {
        "name": "2021-02 Uri freeze",
        "start": "2021-02-10",
        "end": "2021-02-20",
        "products": ["NG", "CL"],
    },
    {
        "name": "2022-02 Ukraine invasion",
        "start": "2022-02-20",
        "end": "2022-04-01",
        "products": ["NG", "BZ", "CL", "ZW", "ZC"],
    },
    {
        "name": "2022 nickel-style squeeze era (energy crunch)",
        "start": "2022-08-01",
        "end": "2022-09-15",
        "products": ["NG"],
    },
    {
        "name": "2023-24 normalisation",
        "start": "2023-01-01",
        "end": "2024-06-30",
        "products": PRODUCTS,
    },
]


def _load_product_curve(data_dir: Path, product: str) -> pl.DataFrame | None:
    path = data_dir / f"{product}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _product_snapshot(
    product: str,
    family: str,
    curve: pl.DataFrame,
    monitor: CalibrationMonitor,
) -> tuple[dict[str, Any], RiskModel | None, np.ndarray]:
    ret = curve["log_return"].to_numpy()
    dates = curve["date"].to_numpy()
    finite = np.isfinite(ret)
    ret_clean = ret[finite]
    dates_clean = dates[finite]

    model = fit_risk_model(ret_clean, product, family)
    if model is None:
        return (
            {
                "product": product,
                "family": family,
                "status": "fit_failed",
                "last_observation": str(dates_clean[-1]) if len(dates_clean) else None,
            },
            None,
            ret_clean,
        )

    sigma_path = ewma_vol(ret_clean)
    last_finite_sigma = sigma_path[np.isfinite(sigma_path)]
    sigma_t = float(last_finite_sigma[-1]) if len(last_finite_sigma) else model.std

    var_es: dict[str, dict[str, float]] = {}
    for level in LEVELS:
        var_es[str(level)] = {
            f"var_h{h}": model.var_conditional(level, sigma_t=sigma_t, horizon=h)
            for h in HORIZONS
        } | {
            f"es_h{h}": model.es_conditional(level, sigma_t=sigma_t, horizon=h)
            for h in HORIZONS
        }

    # rolling calibration status from the model's own conditional VaR path
    # over its full fitted history (sigma_t is causal throughout).
    # compute_acerbi=False: the Acerbi-Szekely bootstrap (n_boot simulated
    # paths, each root-finding a standardized family's numerical ppf over
    # the full history) costs seconds per product and is exactly what
    # run_risk_04_monitor.py already validated offline (gate MB) -- the
    # live snapshot needs to regenerate in around a second, not minutes, so
    # it reports coverage/clustering status only; a full battery re-run
    # (including shape) is the offline job's job, not this one's.
    calib = monitor.evaluate(
        product, model, ret_clean, sigma_path, levels=LEVELS, compute_acerbi=False
    )

    trailing_violations = {}
    for level in LEVELS:
        lr = calib.levels.get(level)
        if lr is None:
            continue
        trailing_violations[str(level)] = {
            "n": lr.n,
            "observed_rate": lr.observed_rate,
            "expected_rate": lr.expected_rate,
            "max_cluster_length": lr.max_cluster_length,
        }

    n_recent = min(RECENT_SERIES_DAYS, len(ret_clean))
    recent_returns = ret_clean[-n_recent:]
    recent_dates = dates_clean[-n_recent:]
    recent_sigma = sigma_path[-n_recent:]
    var_band_01 = np.array(
        [
            -model.var_conditional(0.01, sigma_t=s) if np.isfinite(s) else np.nan
            for s in recent_sigma
        ]
    )

    snapshot = {
        "product": product,
        "family": family,
        "status": "ok",
        "fit_window": {
            "start": str(dates_clean[0]) if len(dates_clean) else None,
            "end": str(dates_clean[-1]) if len(dates_clean) else None,
            "n_obs": len(ret_clean),
        },
        "last_observation": str(dates_clean[-1]) if len(dates_clean) else None,
        "sigma_t": sigma_t,
        "var_es": var_es,
        "trailing_violations": trailing_violations,
        "monitor": {"status": calib.status, "failure_mode": calib.failure_mode},
        "recent_series": {
            "dates": [str(d) for d in recent_dates],
            "returns": [float(r) for r in recent_returns],
            "var_band_01": [float(v) if np.isfinite(v) else None for v in var_band_01],
        },
    }
    return snapshot, model, ret_clean


def _stress_scenarios(
    models: dict[str, RiskModel],
    weights: dict[str, float],
    returns_by_product: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ev in NAMED_EVENTS:
        s = _date.fromisoformat(str(ev["start"]))
        e = _date.fromisoformat(str(ev["end"]))
        port_pnl = 0.0
        contributions: dict[str, Any] = {}
        touched = False
        for p in models:
            if p not in ev["products"] and ev["products"] != PRODUCTS:
                continue
            if p not in returns_by_product:
                continue
            ret, dates = returns_by_product[p]
            mask = np.array(
                [s <= d <= e for d in dates.astype("datetime64[D]").tolist()]
            )
            if not mask.any():
                continue
            touched = True
            cum = float(np.sum(ret[mask]))
            w = weights.get(p, 0.0)
            contributions[p] = {"cum_return": cum, "weight": w, "contribution": cum * w}
            port_pnl += cum * w
        if touched:
            out[str(ev["name"])] = {
                "portfolio_pnl": port_pnl,
                "contributions": contributions,
            }
    return out


def build_snapshot(
    as_of: str | None = None, data_dir: Path | None = None
) -> dict[str, Any]:
    """Build the single JSON document the dashboard reads. `as_of` is a
    display label only (sec 7.4 -- no fitting/selection/threshold decision
    reads it); `data_dir` defaults to `risk.ingest.OUT_DIR`
    (`src/risk/data/`), assumed already populated by `risk.ingest.refresh()`.
    """
    data_dir = data_dir or ingest.OUT_DIR
    family_map = load_family_map("v1")
    monitor = CalibrationMonitor()

    products_out: dict[str, Any] = {}
    models: dict[str, RiskModel] = {}
    returns_by_product: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for product in sorted(family_map.products.keys()):
        curve = _load_product_curve(data_dir, product)
        if curve is None:
            products_out[product] = {
                "product": product,
                "status": "no_data",
                "family": family_map.family_for(product),
            }
            continue
        family = family_map.family_for(product)
        snap, model, ret_clean = _product_snapshot(product, family, curve, monitor)
        products_out[product] = snap
        if model is not None:
            models[product] = model
            dates_clean = curve["date"].to_numpy()[
                np.isfinite(curve["log_return"].to_numpy())
            ]
            returns_by_product[product] = (ret_clean, dates_clean)

    book: dict[str, Any] = {"n_products": len(models)}
    if len(models) >= 2:
        weights = dict.fromkeys(models, 1.0 / len(models))
        min_len = min(len(v[0]) for v in returns_by_product.values())
        hist_aligned = {p: returns_by_product[p][0][-min_len:] for p in models}
        portfolio = {}
        for dep in ["empirical", "gaussian", "t"]:
            portfolio[dep] = portfolio_risk(
                models,
                weights,
                dependence=dep,
                historical_returns=hist_aligned,
                n_sims=20000,
                seed=0,
                t_df=5.0,
            )
        book["weights"] = weights
        book["portfolio_risk"] = portfolio
        book["stress_scenarios"] = _stress_scenarios(
            models, weights, returns_by_product
        )

    return {
        "as_of": as_of,
        "family_map_version": family_map.version,
        "validated_envelope": {
            "products": sorted(family_map.products.keys()),
            "claim": (
                "Conditional VaR/ES coverage validated on these 16 daily "
                "commodity/equity-index futures only (008 Phase 7/8). Not "
                "validated on any other product, asset class, or frequency."
            ),
        },
        "products": products_out,
        "book": book,
    }


def write_snapshot(
    out_path: Path, as_of: str | None = None, data_dir: Path | None = None
) -> dict[str, Any]:
    """Convenience wrapper: build the snapshot and write it as JSON."""
    snapshot = build_snapshot(as_of=as_of, data_dir=data_dir)
    out_path.write_text(json.dumps(snapshot, indent=2, default=str))
    return snapshot


_DASHBOARD_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "dashboard" / "template.html"
)
_SNAPSHOT_PLACEHOLDER = "__RISK_SNAPSHOT_JSON__"


def render_dashboard(
    snapshot: dict[str, Any] | None = None,
    out_path: Path | None = None,
    as_of: str | None = None,
    data_dir: Path | None = None,
    template_path: Path | None = None,
) -> str:
    """Generate the self-contained dashboard HTML: `template.html`
    (`src/risk/dashboard/template.html` -- static HTML+CSS+vanilla JS, no
    build step) with the snapshot JSON inlined at the `__RISK_SNAPSHOT_JSON__`
    placeholder, numbers formatted once here in Python (NEXT_PROMPT.md sec
    7.3), not in JS. Inlined rather than `fetch()`-ed beside the file so the
    page opens correctly from `file://` (fetch of a local file is blocked by
    most browsers' CORS policy for `file://` origins).
    """
    if snapshot is None:
        snapshot = build_snapshot(as_of=as_of, data_dir=data_dir)
    template_path = template_path or _DASHBOARD_TEMPLATE_PATH
    template = template_path.read_text()
    if _SNAPSHOT_PLACEHOLDER not in template:
        raise ValueError(
            f"dashboard template is missing the {_SNAPSHOT_PLACEHOLDER!r} placeholder"
        )
    payload = json.dumps(snapshot, default=str)
    # </script> can never legally appear inside a JSON string value, but
    # guard the injection point anyway against a payload containing that
    # literal substring (e.g. a stress-scenario name), which would
    # otherwise prematurely close the inlining <script> tag.
    payload = payload.replace("</", "<\\/")
    html = template.replace(_SNAPSHOT_PLACEHOLDER, payload)
    if out_path is not None:
        out_path.write_text(html)
    return html
