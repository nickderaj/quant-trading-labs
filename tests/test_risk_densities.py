"""Tests for the promoted `src/risk/densities/` registry (NEXT_PROMPT.md sec
3.4): the registry is complete, every family round-trips `ppf(cdf(x)) ~= x`,
and every `es(q)` is more negative than its own `ppf(q)`.

The four family modules already have dedicated test files
(`tests/test_dist_lib6_{ged,nig,johnsonsu,hansen_skewt}.py`), which keep
passing unchanged through the `src/research/tmp/densities/` shim -- this file
only covers the registry-level contract, not per-family numerics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from risk.densities import REGISTRY
from risk.model import numerical_pit

SEED = 0
EXPECTED_FAMILIES = {"ged", "nig", "johnsonsu", "hansen_skewt"}


class TestRegistry:
    def test_registry_is_complete(self):
        assert set(REGISTRY.keys()) == EXPECTED_FAMILIES

    def test_every_family_exposes_the_fixed_interface(self):
        for name, mod in REGISTRY.items():
            assert mod.NAME == name
            assert isinstance(mod.N_SHAPE, int) and mod.N_SHAPE > 0
            assert callable(mod.fit)
            assert callable(mod.logpdf)
            assert callable(mod.ppf)
            assert callable(mod.es)


class TestPpfCdfRoundTrip:
    def _default_shape(self, name: str, mod) -> tuple:
        # A representative, well-conditioned shape per family (not fit from
        # data -- this test is about the ppf/cdf plumbing, not calibration).
        defaults = {
            "ged": (1.5,),
            "nig": (1.2, 0.0),
            "johnsonsu": (0.0, 1.5),
            "hansen_skewt": (8.0, 0.0),
        }
        return defaults[name]

    def test_ppf_cdf_round_trip(self):
        u = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        for name, mod in REGISTRY.items():
            shape = self._default_shape(name, mod)
            z = np.asarray(mod.ppf(u, shape), dtype=float)
            back = numerical_pit(
                lambda zz, mod=mod, shape=shape: mod.logpdf(zz, shape), z
            )
            np.testing.assert_allclose(
                back, u, atol=0.02, err_msg=f"{name}: ppf/cdf round-trip failed"
            )

    def test_es_more_negative_than_ppf_in_the_lower_tail(self):
        for name, mod in REGISTRY.items():
            shape = self._default_shape(name, mod)
            for q in (0.01, 0.025, 0.05):
                es_q = float(mod.es(q, shape))
                ppf_q = float(mod.ppf(q, shape))
                assert es_q <= ppf_q, (
                    f"{name}: es({q})={es_q} is not more negative than ppf({q})={ppf_q}"
                )
