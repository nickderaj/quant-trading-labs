"""Shim: promoted to `src/risk/densities/` (NEXT_PROMPT.md sec 3.4) as
durable, tested, production code. Re-exported here, unchanged, because nine
scratch scripts still import from this path (`run_phase_2_density_selection.py`,
`run_phase3_zoo.py`, `run_phase6_application.py`, `run_phase_b_risk_gated.py`,
`run_phase_d_tail_factor.py`, `run_phase_3_conditional_battery.py`,
`build_notebook8.py`, `dist_lib6.py`, and `commod_lib8.py` via `risk.model`).
See `docs/10-risk-engine.md`.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from risk.densities import REGISTRY, ged, hansen_skewt, johnsonsu, nig

__all__ = ["REGISTRY", "ged", "hansen_skewt", "johnsonsu", "nig"]
