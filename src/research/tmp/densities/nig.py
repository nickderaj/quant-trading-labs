"""Shim: promoted to `src/risk/densities/nig.py` (NEXT_PROMPT.md sec 3.4) as
durable, tested, production code. Re-exported here, unchanged, because
`tests/test_dist_lib6_nig.py` imports this file directly by path
(`sys.path.insert(0, ".../densities"); import nig`), bypassing the
`densities` package. See `docs/10-risk-engine.md`.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from risk.densities.nig import N_SHAPE, NAME, es, fit, logpdf, ppf

__all__ = ["NAME", "N_SHAPE", "es", "fit", "logpdf", "ppf"]
