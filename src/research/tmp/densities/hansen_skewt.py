"""Shim: promoted to `src/risk/densities/hansen_skewt.py` (NEXT_PROMPT.md sec
3.4) as durable, tested, production code. Re-exported here, unchanged,
because `tests/test_dist_lib6_hansen_skewt.py` imports this file directly by
path (`sys.path.insert(0, ".../densities"); import hansen_skewt`), bypassing
the `densities` package. See `docs/10-risk-engine.md`.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from risk.densities.hansen_skewt import N_SHAPE, NAME, es, fit, logpdf, ppf

__all__ = ["NAME", "N_SHAPE", "es", "fit", "logpdf", "ppf"]
