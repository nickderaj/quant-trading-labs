"""Registry of notebook-6's Phase 3 innovation-density families.

Each module (ged.py, nig.py, johnsonsu.py, hansen_skewt.py) exposes the fixed
interface declared in NEXT_RUN_PROMPT.md's Phase 3 section:

    NAME: str
    N_SHAPE: int
    def fit(z: np.ndarray) -> tuple[float, ...] | None
    def logpdf(z: np.ndarray, shape: tuple[float, ...]) -> np.ndarray
    def ppf(q: float | np.ndarray, shape: tuple[float, ...]) -> np.ndarray
    def es(q: float, shape: tuple[float, ...]) -> float

All are standardized to unit variance (mean 0, var 1) before use, so they can
be composed with a rolling GARCH/GJR variance forecast exactly the way
dist_lib's own Student-t scaling (`sqrt(nu/(nu-2))`) is - see
`_garch_negloglik` for the discipline being matched.

Each was implemented independently (one subagent per family, per
NEXT_RUN_PROMPT.md section 1's fan-out instruction) and is imported here into
one registry dict so dist_lib6.py / the Phase 3 driver can iterate over the
family set without hardcoding four import statements at every call site.
"""

from . import ged, hansen_skewt, johnsonsu, nig

REGISTRY = {
    ged.NAME: ged,
    nig.NAME: nig,
    johnsonsu.NAME: johnsonsu,
    hansen_skewt.NAME: hansen_skewt,
}
