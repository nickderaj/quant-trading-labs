"""NEXT_PROMPT.md sec 12: "The import shim creates a cycle" is the risks
section's second-listed failure mode. `commod_lib8.py` importing from
`risk/` (and a notebook importing both) is fine; `risk/` importing anything
from `commod_lib8.py` is not -- the dependency must stay strictly
one-directional. Verified two ways: `risk` imports cleanly in a fresh
subprocess with no notebook-scratch path on `sys.path`, and importing `risk`
never pulls `commod_lib8`/`dist_lib5`/the `densities` shim into `sys.modules`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_risk_imports_cleanly_without_research_tmp_on_sys_path():
    # A clean interpreter with ONLY src/ on sys.path (no src/research/tmp/):
    # if risk/ secretly depended on commod_lib8.py or dist_lib5.py, this
    # would fail with ModuleNotFoundError.
    result = subprocess.run(
        [sys.executable, "-c", "import risk, risk.hygiene, risk.calibration"],
        cwd=str(_ROOT),
        env={"PYTHONPATH": str(_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_importing_risk_does_not_pull_in_research_tmp_scratch():
    probe = (
        "import sys; import risk, risk.hygiene, risk.calibration; "
        "leaked = [m for m in sys.modules if m in "
        "('commod_lib8', 'dist_lib5', 'densities')]; "
        "print(leaked); sys.exit(1 if leaked else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_ROOT),
        env={"PYTHONPATH": str(_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"risk/ pulled in scratch modules: {result.stdout}"
