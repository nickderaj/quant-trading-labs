"""The family map: a versioned, provenance-carrying artifact (NEXT_PROMPT.md
sec 5), not a bare dict. `load_family_map` is the only way `fit_risk_model`'s
family argument should be sourced in production; a caller override is
supported but must be explicit and is logged.

**Validated envelope.** `v1` covers exactly the 16 products in
`configs/family_map_v1.json` -- the same 16 commodity/ES products 008 Phase 7
certified. A product not in the map is not covered by any validated result
and must go through `fit_new_product()` (NEXT_PROMPT.md sec 5.3) rather than
silently defaulting to `ged`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

__all__ = [
    "FamilyMap",
    "UnseenProductError",
    "config_hash",
    "fit_new_product",
    "load_family_map",
]

# NEXT_PROMPT.md sec 5.3: production refit cadence, decided here and not left
# to a future caller. Phase 7 used 5 walk-forward folds over ~1,805
# observations/product (development window); the minimum history required
# for a refit mirrors fit_risk_model's own >=100-observation guard, several
# times over so a refit is not fit to a bare-minimum window.
REFIT_INTERVAL_TRADING_DAYS = 63  # ~1 calendar quarter
MIN_HISTORY_FOR_REFIT = 500


class UnseenProductError(ValueError):
    """Raised when a product outside the validated family-map envelope is
    requested without going through `fit_new_product()`."""


def config_hash(payload: dict[str, Any]) -> str:
    """sha256 over the sorted-key, compact-JSON encoding of `payload` --
    the same construction as `regime.config.RegimeConfig.config_hash`
    (014 established config-hash equality as this repo's port-fidelity
    primitive; the risk engine uses the same one, NEXT_PROMPT.md sec 5.1)."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class FamilyMap:
    version: str
    created: str
    source_notebook: str
    policy: str
    config_hash_value: str
    products: dict[str, dict[str, Any]]

    def family_for(self, product: str) -> str:
        if product not in self.products:
            raise UnseenProductError(
                f"{product!r} is not in family map {self.version!r}'s validated "
                f"envelope ({sorted(self.products)}). Use fit_new_product() to "
                "extend the map, or refuse -- never default to a family."
            )
        return str(self.products[product]["family"])

    def __contains__(self, product: str) -> bool:
        return product in self.products


@lru_cache(maxsize=8)
def load_family_map(version: str = "v1") -> FamilyMap:
    """Load `configs/family_map_v{version}.json` as a `FamilyMap`. Cached --
    the map is a frozen artifact for the lifetime of the process; a new
    version requires a new file, not an in-place edit."""
    resource = files("risk").joinpath("configs", f"family_map_{version}.json")
    with resource.open("r") as f:
        payload = json.load(f)

    stored_hash = payload.get("config_hash")
    verifiable = {k: v for k, v in payload.items() if k != "config_hash"}
    computed_hash = config_hash(verifiable)
    if stored_hash != computed_hash:
        raise ValueError(
            f"family_map_{version}.json config_hash mismatch: stored="
            f"{stored_hash!r} computed={computed_hash!r} -- the file was "
            "edited without regenerating its hash"
        )

    return FamilyMap(
        version=payload["version"],
        created=payload["created"],
        source_notebook=payload["source_notebook"],
        policy=payload.get("policy", "P1"),
        config_hash_value=stored_hash,
        products=payload["products"],
    )


def fit_new_product(
    product: str, returns: Any, ranking_fn: Any = None
) -> dict[str, Any]:
    """Extend the family map to a product not in `v1`'s validated envelope.

    NEXT_PROMPT.md sec 5.3: a product not in the map must not silently
    default to `ged`. This is the explicit, refuse-or-extend path: run the
    same Phase 3 OOS-log-score ranking procedure the map itself was built
    from (`ranking_fn`, supplied by the caller -- this module does not
    reimplement Phase 3's ranking battery, only requires that a new
    product's map entry come from running it, not from guessing) and append
    to a *new* map version. Unimplemented here deliberately: doing this for
    real is a modelling exercise (sec 2 ground rule 3: no new modelling
    inside this project), not a plumbing one -- this function documents and
    enforces the contract, and raises until a ranking procedure is wired in.
    """
    raise NotImplementedError(
        f"fit_new_product({product!r}) has no wired ranking procedure. "
        "Per NEXT_PROMPT.md sec 5.3, an unseen product must not silently "
        "default to a family -- run 008 Phase 3's OOS-log-score ranking on "
        "the new product's own history and append a new family_map version "
        "explicitly, as a separate, reviewed change."
    )
