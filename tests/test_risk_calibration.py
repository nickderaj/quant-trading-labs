"""Unit tests for `src/risk/calibration.py`: `kupiec_by_state` (split from
`tests/test_commod_lib8.py`, NEXT_PROMPT.md sec 3.6) and `CalibrationMonitor`
(NEXT_PROMPT.md sec 6.3-6.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from risk.calibration import kupiec_by_state

SEED = 0


class TestKupiecByState:
    def test_flags_a_state_with_bad_coverage(self):
        rng = np.random.default_rng(SEED)
        n = 2000
        # state A: correctly calibrated 1% hit rate; state B: hits at 5% (miscalibrated)
        states = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
        hits = np.concatenate([rng.random(n // 2) < 0.01, rng.random(n // 2) < 0.05])
        result = kupiec_by_state(hits, states, expected_rate=0.01)
        assert result["A"]["kupiec_p"] > 0.01
        assert result["B"]["kupiec_p"] < 0.01
