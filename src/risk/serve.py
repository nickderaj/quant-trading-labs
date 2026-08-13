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
from risk.calibration import CalibrationMonitor, MonitorStateStore, apply_persistence
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

# Crypto panel (research/tmp/build_family_map_crypto_v1.py,
# research/tmp/ingest_crypto_risk_data.py): the frozen 6-symbol panel from
# notebooks 004-006's crypto research, run through this same risk-engine
# pipeline for monitoring purposes. Deliberately kept out of `PRODUCTS`,
# `family_map_v1`, `products`/`book` above -- see `_build_crypto_products`'s
# docstring for why this is not the validated envelope.
CRYPTO_FAMILY_MAP_VERSION = "crypto_v1"
CRYPTO_ENVELOPE_CLAIM = (
    "These 6 crypto perpetuals are shown for research/monitoring purposes "
    "only, run through the same VaR/ES and calibration-monitor machinery as "
    "the validated futures engine above. This is explicitly NOT the "
    "validated envelope: each symbol's density family is an OOS log-score "
    "pick (the same contest Phase 3 ran for the futures), but no "
    "walk-forward VaR-coverage gate has ever been run for this panel at "
    "daily frequency -- notebooks 004/005 only cleared that battery for BTC "
    "at 12h, and for SOL/DOGE/BNB (weakly XRP) at 1d; never for BTC or ETH "
    "at 1d. Treat the calibration status below as a live diagnostic, not a "
    "certified result (see family_map_crypto_v1.json's own validation_note)."
)

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
) -> tuple[dict[str, Any], RiskModel | None, np.ndarray, np.ndarray]:
    """Fits the model and computes everything that does *not* depend on
    other products (VaR/ES, recent series). Calibration status is
    deliberately not computed here: it must be BH-corrected across all 16
    products at once (`CalibrationMonitor.evaluate_batch`) and persistence-
    gated across runs (`calibration.apply_persistence`), both of which need
    every product's inputs together -- see `build_snapshot`, which fills in
    `monitor`/`trailing_violations` after this returns."""
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
            np.array([]),
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
        "recent_series": {
            "dates": [str(d) for d in recent_dates],
            "returns": [float(r) for r in recent_returns],
            "var_band_01": [float(v) if np.isfinite(v) else None for v in var_band_01],
        },
    }
    return snapshot, model, ret_clean, sigma_path


def _attach_monitor_status(
    products_out: dict[str, Any],
    product_inputs: dict[str, tuple[RiskModel, np.ndarray, np.ndarray]],
    monitor: CalibrationMonitor,
    data_dir: Path,
) -> None:
    """BH-corrects calibration status across every fitted product in this
    run (`evaluate_batch`), then persistence-gates it against the streak
    counts carried over from the previous run at `data_dir /
    "_monitor_state.json"` (`apply_persistence`) -- both pre-registered in
    `risk_engine_preregistration.json`'s `calibration_monitor` block.
    compute_acerbi=False for the same reason as before: the Acerbi-Szekely
    bootstrap costs seconds per product and is the offline job's job (gate
    MB), not this one's, which needs to regenerate in about a second.
    Mutates `products_out` in place."""
    if not product_inputs:
        return
    raw_statuses = monitor.evaluate_batch(
        product_inputs, levels=LEVELS, compute_acerbi=False
    )
    state_store = MonitorStateStore(data_dir / "_monitor_state.json")
    k = int(monitor.thresholds["persistence_rule"]["k_consecutive_breaching_windows"])
    final_statuses, new_state = apply_persistence(raw_statuses, state_store.load(), k)
    state_store.save(new_state)

    for product, status in final_statuses.items():
        trailing_violations = {}
        for level, lr in status.levels.items():
            trailing_violations[str(level)] = {
                "n": lr.n,
                "observed_rate": lr.observed_rate,
                "expected_rate": lr.expected_rate,
                "max_cluster_length": lr.max_cluster_length,
            }
        products_out[product]["trailing_violations"] = trailing_violations
        products_out[product]["monitor"] = {
            "status": status.status,
            "failure_mode": status.failure_mode,
        }


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


def _build_crypto_products(
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The crypto counterpart of `build_snapshot`'s main futures loop: same
    per-symbol fit/VaR-ES/monitor pipeline, but against
    `family_map_crypto_v1` (an OOS-log-score pick, not a gate-validated
    family map) and a separate `data_dir`
    (`research/tmp/ingest_crypto_risk_data.py`'s output, not
    `risk.ingest.refresh()`'s). Returns `(products, envelope)`, both kept
    entirely out of the main `products`/`book`/`validated_envelope` so that
    the validated envelope's own invariant (`products.keys() ==
    validated_envelope.products`) is never diluted by an unvalidated panel.
    If `family_map_crypto_v1.json` isn't present, returns empty results
    rather than failing the whole snapshot -- this section is additive.
    """
    try:
        family_map = load_family_map(CRYPTO_FAMILY_MAP_VERSION)
    except (FileNotFoundError, ValueError):
        return {}, {"products": [], "claim": ""}

    monitor = CalibrationMonitor()
    products_out: dict[str, Any] = {}
    product_inputs: dict[str, tuple[RiskModel, np.ndarray, np.ndarray]] = {}

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
        snap, model, ret_clean, sigma_path = _product_snapshot(product, family, curve)
        products_out[product] = snap
        if model is not None:
            product_inputs[product] = (model, ret_clean, sigma_path)

    _attach_monitor_status(products_out, product_inputs, monitor, data_dir)

    envelope = {
        "products": sorted(family_map.products.keys()),
        "claim": CRYPTO_ENVELOPE_CLAIM,
    }
    return products_out, envelope


def build_snapshot(
    as_of: str | None = None,
    data_dir: Path | None = None,
    crypto_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the single JSON document the dashboard reads. `as_of` is a
    display label only (sec 7.4 -- no fitting/selection/threshold decision
    reads it); `data_dir` defaults to `risk.ingest.OUT_DIR`
    (`src/risk/data/`), assumed already populated by `risk.ingest.refresh()`.
    `crypto_data_dir` defaults to `data_dir / "crypto"` (so passing a fresh
    `data_dir` in a test isolates the crypto panel too, rather than reading
    real production crypto data by accident).
    """
    data_dir = data_dir or ingest.OUT_DIR
    crypto_data_dir = crypto_data_dir or (data_dir / "crypto")
    family_map = load_family_map("v1")
    monitor = CalibrationMonitor()

    products_out: dict[str, Any] = {}
    models: dict[str, RiskModel] = {}
    returns_by_product: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    product_inputs: dict[str, tuple[RiskModel, np.ndarray, np.ndarray]] = {}

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
        snap, model, ret_clean, sigma_path = _product_snapshot(product, family, curve)
        products_out[product] = snap
        if model is not None:
            models[product] = model
            dates_clean = curve["date"].to_numpy()[
                np.isfinite(curve["log_return"].to_numpy())
            ]
            returns_by_product[product] = (ret_clean, dates_clean)
            product_inputs[product] = (model, ret_clean, sigma_path)

    _attach_monitor_status(products_out, product_inputs, monitor, data_dir)

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

    crypto_products_out, crypto_envelope = _build_crypto_products(crypto_data_dir)

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
        "crypto_envelope": crypto_envelope,
        "crypto_products": crypto_products_out,
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
