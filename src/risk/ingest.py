"""The refresh pipeline (NEXT_PROMPT.md sec 7.1): turns the one-off
`run_phase_0_hygiene.py` build into a supported, repeatable operation.

Reads the same databento parquet root Phase 0 used
(`src/research/data/market/databento/{ohlcv,contracts.parquet,
roll_calendar.parquet}`), runs `hygiene.build_risk_inputs` per product, and
writes versioned outputs to `src/risk/data/` -- a durable location, not
`research/tmp/`. No new vendor integration and no network calls: if the
databento cache has not been updated, `refresh` reports the stale
`last_observation` date rather than fetching anything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

from risk import hygiene
from risk.families import load_family_map

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = _REPO_ROOT / "src" / "research" / "data" / "market" / "databento"
OUT_DIR = Path(__file__).resolve().parent / "data"

__all__ = ["IngestReport", "ProductIngestResult", "refresh"]


@dataclass(frozen=True)
class ProductIngestResult:
    product: str
    status: str  # "written" | "unchanged" | "rejected"
    rows: int
    last_observation: str | None
    rows_after_hygiene_filter: int | None
    rows_after_liquidity_screen: int | None
    content_hash: str | None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class IngestReport:
    as_of: str
    products: dict[str, ProductIngestResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "products": {p: asdict(r) for p, r in self.products.items()},
        }


def _content_hash(df: pl.DataFrame) -> str:
    """A stable hash of `df`'s content (not file bytes/timestamps), used to
    decide whether a re-run actually changed anything -- the idempotency
    check NEXT_PROMPT.md sec 7.1/gate IR needs, without depending on parquet
    writers embedding a deterministic timestamp."""
    h = hashlib.sha256()
    for col in sorted(df.columns):
        h.update(col.encode())
        series = df[col]
        h.update(
            series.cast(pl.Utf8, strict=False).fill_null("").hash().to_numpy().tobytes()
        )
    return h.hexdigest()


def refresh(
    products: list[str] | None = None,
    as_of: str | None = None,
    data_root: Path | None = None,
    out_dir: Path | None = None,
) -> IngestReport:
    """Re-read the databento parquet root, re-run `build_risk_inputs` per
    product, and write versioned outputs to `out_dir` (default
    `src/risk/data/`).

    Idempotent: if a product's freshly-built frame content-hashes identical
    to what is already on disk, the file is left untouched (`status ==
    "unchanged"`) rather than rewritten -- true byte-identity by
    construction, not by hoping two writes serialize the same way.

    Fail loud, never silently partial: a product whose contract check fails
    (`assert_risk_inputs` raises) is *absent* from the written outputs, with
    the reason recorded (`status == "rejected"`), never present with a
    quietly shortened series.
    """
    data_root = data_root or DATA_ROOT
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if products is None:
        products = sorted(load_family_map("v1").products.keys())

    ohlcv = pl.read_parquet(str(data_root / "ohlcv"))
    contracts = pl.read_parquet(str(data_root / "contracts.parquet"))
    roll_calendar = pl.read_parquet(str(data_root / "roll_calendar.parquet"))

    # Computed once for the whole multi-product frame and reused for every
    # product's reporting counts below, rather than recomputed per product
    # -- flag_contaminated_rows/liquidity_screen operate on the full frame
    # regardless of which product is being reported on.
    clean_all = hygiene.apply_hygiene_filter(ohlcv)
    screened_all = hygiene.liquidity_screen(clean_all)

    results: dict[str, ProductIngestResult] = {}
    for product in products:
        try:
            rows_after_hygiene = clean_all.filter(pl.col("product") == product).height
            rows_after_liquidity = screened_all.filter(
                pl.col("product") == product
            ).height

            curve = hygiene.build_risk_inputs(ohlcv, contracts, roll_calendar, product)
            hygiene.assert_risk_inputs(curve)
        except hygiene.RiskInputError as exc:
            results[product] = ProductIngestResult(
                product=product,
                status="rejected",
                rows=0,
                last_observation=None,
                rows_after_hygiene_filter=None,
                rows_after_liquidity_screen=None,
                content_hash=None,
                rejection_reason=str(exc),
            )
            continue

        new_hash = _content_hash(curve)
        out_path = out_dir / f"{product}.parquet"
        hash_path = out_dir / f"{product}.hash"
        prev_hash = hash_path.read_text().strip() if hash_path.exists() else None

        last_obs = curve["date"].max()
        last_obs_str = str(last_obs) if last_obs is not None else None

        if prev_hash == new_hash and out_path.exists():
            status = "unchanged"
        else:
            curve.write_parquet(out_path)
            hash_path.write_text(new_hash)
            status = "written"

        results[product] = ProductIngestResult(
            product=product,
            status=status,
            rows=curve.height,
            last_observation=last_obs_str,
            rows_after_hygiene_filter=rows_after_hygiene,
            rows_after_liquidity_screen=rows_after_liquidity,
            content_hash=new_hash,
        )

    report = IngestReport(as_of=as_of or "unspecified", products=results)
    report_path = out_dir / "_ingest_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return report
