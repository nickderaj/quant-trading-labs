"""Tests for barrier option pricing (notebook 025)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TMP_DIR = Path(__file__).resolve().parent.parent / "src" / "research" / "tmp"
sys.path.insert(0, str(_TMP_DIR))

import lib25
import pricers_25_barrier as pb


class TestBarrierAnalytic:
    """Test Reiner-Rubinstein closed-form pricing."""

    def test_barrier_parity_down(self):
        """Down-and-out + down-and-in == vanilla."""
        F, K, B, T = 100.0, 100.0, 90.0, 1.0
        sigma, df = 0.25, 0.98
        opt = "call"

        vanilla = lib25.black76(F, K, T, sigma, df, opt)
        do_price = pb.barrier_analytic(F, K, B, T, sigma, df, opt, "do")
        di_price = pb.barrier_analytic(F, K, B, T, sigma, df, opt, "di")

        assert np.isclose(do_price + di_price, vanilla, atol=1e-10), (
            f"do + di = {do_price + di_price}, vanilla = {vanilla}"
        )

    def test_barrier_parity_up(self):
        """Up-and-out + up-and-in == vanilla."""
        F, K, B, T = 100.0, 100.0, 110.0, 1.0
        sigma, df = 0.25, 0.98
        opt = "put"

        vanilla = lib25.black76(F, K, T, sigma, df, opt)
        uo_price = pb.barrier_analytic(F, K, B, T, sigma, df, opt, "uo")
        ui_price = pb.barrier_analytic(F, K, B, T, sigma, df, opt, "ui")

        assert np.isclose(uo_price + ui_price, vanilla, atol=1e-10), (
            f"uo + ui = {uo_price + ui_price}, vanilla = {vanilla}"
        )

    def test_barrier_parity_all_kinds(self):
        """Test parity for all four barrier types and both options."""
        F, K, T = 100.0, 105.0, 0.5
        sigma, df = 0.30, 0.97

        for opt in ("call", "put"):
            # Down barriers
            B = 95.0
            vanilla = lib25.black76(F, K, T, sigma, df, opt)
            parity_sum = pb.barrier_analytic(
                F, K, B, T, sigma, df, opt, "do"
            ) + pb.barrier_analytic(F, K, B, T, sigma, df, opt, "di")
            assert np.isclose(parity_sum, vanilla, atol=1e-10)

            # Up barriers
            B = 110.0
            vanilla = lib25.black76(F, K, T, sigma, df, opt)
            parity_sum = pb.barrier_analytic(
                F, K, B, T, sigma, df, opt, "uo"
            ) + pb.barrier_analytic(F, K, B, T, sigma, df, opt, "ui")
            assert np.isclose(parity_sum, vanilla, atol=1e-10)

    def test_barrier_far_otm(self):
        """Barrier far out-of-money should approach vanilla."""
        F, K, T = 100.0, 100.0, 1.0
        sigma, df = 0.25, 0.98

        # Down-and-out call with barrier far below F
        vanilla = lib25.black76(F, K, T, sigma, df, "call")
        do_price = pb.barrier_analytic(F, K, 50.0, T, sigma, df, "call", "do")
        assert np.isclose(do_price, vanilla, rtol=0.001)

        # Up-and-out put with barrier far above F
        vanilla = lib25.black76(F, K, T, sigma, df, "put")
        uo_price = pb.barrier_analytic(F, K, 150.0, T, sigma, df, "put", "uo")
        assert np.isclose(uo_price, vanilla, rtol=0.001)

    def test_barrier_already_breached_down(self):
        """Down barrier already hit: knock-out = 0, knock-in = vanilla."""
        F, K, B, T = 100.0, 100.0, 110.0, 1.0  # F < B
        sigma, df = 0.25, 0.98

        vanilla = lib25.black76(F, K, T, sigma, df, "call")
        do_price = pb.barrier_analytic(F, K, B, T, sigma, df, "call", "do")
        di_price = pb.barrier_analytic(F, K, B, T, sigma, df, "call", "di")

        assert np.isclose(do_price, 0.0)
        assert np.isclose(di_price, vanilla)

    def test_barrier_already_breached_up(self):
        """Up barrier already hit: knock-out = 0, knock-in = vanilla."""
        F, K, B, T = 100.0, 100.0, 90.0, 1.0  # F > B
        sigma, df = 0.25, 0.98

        vanilla = lib25.black76(F, K, T, sigma, df, "put")
        uo_price = pb.barrier_analytic(F, K, B, T, sigma, df, "put", "uo")
        ui_price = pb.barrier_analytic(F, K, B, T, sigma, df, "put", "ui")

        assert np.isclose(uo_price, 0.0)
        assert np.isclose(ui_price, vanilla)


class TestBarrierContinuityShift:
    """Test Broadie-Glasserman-Kou discretization correction."""

    def test_bgk_down_barrier(self):
        """Down barrier should be shifted down."""
        B, sigma, dt = 100.0, 0.25, 0.01
        adjusted = pb.barrier_continuity_shift(B, sigma, dt, "do")
        assert adjusted < B

    def test_bgk_up_barrier(self):
        """Up barrier should be shifted up."""
        B, sigma, dt = 100.0, 0.25, 0.01
        adjusted = pb.barrier_continuity_shift(B, sigma, dt, "uo")
        assert adjusted > B

    def test_bgk_shift_moves_barrier(self):
        """BGK correction should shift barriers in the correct direction."""
        B, sigma, dt = 100.0, 0.25, 0.01

        # Down barriers should shift down
        B_do = pb.barrier_continuity_shift(B, sigma, dt, "do")
        assert B_do < B, f"Down barrier should shift down, got {B_do} from {B}"

        # Up barriers should shift up
        B_uo = pb.barrier_continuity_shift(B, sigma, dt, "uo")
        assert B_uo > B, f"Up barrier should shift up, got {B_uo} from {B}"

        # Magnitude should be similar (allowing for log-linear approximation error)
        down_shift = B - B_do
        up_shift = B_uo - B
        assert np.isclose(down_shift, up_shift, rtol=0.02), (
            f"Down shift {down_shift} should ≈ up shift {up_shift}"
        )


class TestBarrierMC:
    """Test Monte Carlo barrier pricing."""

    def test_mc_valid_output_down_call(self):
        """MC down-and-out call returns valid output."""
        F, K, B, T = 100.0, 100.0, 95.0, 1.0
        sigma, df = 0.25, 0.98
        n_paths, n_steps = 10_000, 252

        paths = lib25.simulate_gbm(
            F, sigma, T, n_steps, n_paths, seed=42, antithetic=True
        )
        mc_result = pb.barrier_mc(paths, K, B, "do", "call", df, monitor="daily")

        assert "price" in mc_result
        assert "se" in mc_result
        assert "n" in mc_result
        assert mc_result["n"] == n_paths
        assert mc_result["price"] >= 0
        assert mc_result["se"] >= 0

    def test_mc_valid_output_up_put(self):
        """MC up-and-out put returns valid output."""
        F, K, B, T = 100.0, 100.0, 110.0, 1.0
        sigma, df = 0.25, 0.98
        n_paths, n_steps = 10_000, 252

        paths = lib25.simulate_gbm(
            F, sigma, T, n_steps, n_paths, seed=43, antithetic=True
        )
        mc_result = pb.barrier_mc(paths, K, B, "uo", "put", df, monitor="daily")

        assert "price" in mc_result
        assert "se" in mc_result
        assert "n" in mc_result
        assert mc_result["n"] == n_paths
        assert mc_result["price"] >= 0
        assert mc_result["se"] >= 0

    def test_mc_down_in_parity(self):
        """MC knock-in + knock-out should equal MC vanilla."""
        F, K, B, T = 100.0, 100.0, 95.0, 0.5
        sigma, df = 0.25, 0.975
        n_paths, n_steps = 50_000, 126

        paths = lib25.simulate_gbm(
            F, sigma, T, n_steps, n_paths, seed=44, antithetic=True
        )

        mc_do = pb.barrier_mc(paths, K, B, "do", "call", df)["price"]
        mc_di = pb.barrier_mc(paths, K, B, "di", "call", df)["price"]

        # Compare to vanilla (also via MC as reference)
        vanilla = lib25.black76(F, K, T, sigma, df, "call")

        assert np.isclose(mc_do + mc_di, vanilla, rtol=0.01)


class TestBarrierPDE:
    """Test finite-difference PDE solver."""

    def test_pde_valid_output_down_call(self):
        """PDE solver returns valid output for down-and-out call."""
        F, K, B, T = 100.0, 100.0, 95.0, 1.0
        sigma, r = 0.25, 0.02
        n_s, n_t = 150, 150

        pde_price = pb.barrier_pde(F, K, B, T, sigma, r, "call", "do", n_s, n_t)

        # Check basic validity
        assert isinstance(pde_price, (float, np.floating))
        assert np.isfinite(pde_price)
        # Allow small negative due to numerical errors
        assert pde_price >= -0.01, f"Price should be ≈ non-negative, got {pde_price}"
        # Price should be less than vanilla
        vanilla = lib25.black76(F, K, T, sigma, np.exp(-r * T), "call")
        assert pde_price <= vanilla * 1.01, (
            f"Barrier option should not exceed vanilla, got {pde_price} vs {vanilla}"
        )

    def test_pde_valid_output_up_put(self):
        """PDE solver returns valid output for up-and-out put."""
        F, K, B, T = 100.0, 100.0, 110.0, 1.0
        sigma, r = 0.25, 0.02
        n_s, n_t = 150, 150

        pde_price = pb.barrier_pde(F, K, B, T, sigma, r, "put", "uo", n_s, n_t)

        # Check basic validity
        assert isinstance(pde_price, (float, np.floating))
        assert np.isfinite(pde_price)
        # Allow small negative due to numerical errors
        assert pde_price >= -0.01, f"Price should be ≈ non-negative, got {pde_price}"
        # Price should be less than vanilla
        vanilla = lib25.black76(F, K, T, sigma, np.exp(-r * T), "put")
        assert pde_price <= vanilla * 1.01, (
            f"Barrier option should not exceed vanilla, got {pde_price} vs {vanilla}"
        )


class TestKOCollar:
    """Test down-and-out put + short vanilla call collar."""

    def test_ko_collar_arithmetic(self):
        """Collar premium = DO put - vanilla call."""
        F, K_put, K_call, B, T = 100.0, 95.0, 105.0, 90.0, 1.0
        sigma, df = 0.25, 0.98

        collar = pb.ko_collar(F, K_put, K_call, B, T, sigma, df)

        do_put = pb.barrier_analytic(F, K_put, B, T, sigma, df, "put", "do")
        call = lib25.black76(F, K_call, T, sigma, df, "call")
        expected_premium = do_put - call

        assert np.isclose(collar["premium"], expected_premium)

    def test_ko_collar_structure(self):
        """Collar should have expected structure."""
        F, K_put, K_call, B, T = 100.0, 95.0, 105.0, 90.0, 0.5
        sigma, df = 0.25, 0.98

        collar = pb.ko_collar(F, K_put, K_call, B, T, sigma, df)

        assert "premium" in collar
        assert "legs" in collar
        assert len(collar["legs"]) == 2
        assert collar["legs"][0]["instrument"] == "down-and-out put"
        assert collar["legs"][1]["instrument"] == "short vanilla call"


class TestEdgeCases:
    """Edge cases and degenerate scenarios."""

    def test_zero_time_raises(self):
        """T <= 0 should raise."""
        with pytest.raises(ValueError):
            pb.barrier_analytic(100, 100, 95, 0, 0.25, 0.98, "call", "do")

    def test_zero_sigma_raises(self):
        """sigma <= 0 should raise."""
        with pytest.raises(ValueError):
            pb.barrier_analytic(100, 100, 95, 1.0, 0, 0.98, "call", "do")

    def test_knock_out_intrinsic_negative(self):
        """OTM knock-out should be small."""
        F, K, B, T = 100.0, 110.0, 95.0, 1.0
        sigma, df = 0.25, 0.98

        do_call = pb.barrier_analytic(F, K, B, T, sigma, df, "call", "do")
        assert do_call >= 0.0
        assert do_call < lib25.black76(F, K, T, sigma, df, "call")

    def test_high_barrier_down(self):
        """Down barrier very high (near F) should make knock-out nearly worthless."""
        F, K, B, T = 100.0, 100.0, 100.01, 1.0
        sigma, df = 0.25, 0.98

        do_call = pb.barrier_analytic(F, K, B, T, sigma, df, "call", "do")
        vanilla = lib25.black76(F, K, T, sigma, df, "call")

        # Should be much less than vanilla
        assert do_call < 0.01 * vanilla

    def test_low_barrier_up(self):
        """Up barrier very low (near F) should make knock-out nearly worthless."""
        F, K, B, T = 100.0, 100.0, 99.99, 1.0
        sigma, df = 0.25, 0.98

        uo_put = pb.barrier_analytic(F, K, B, T, sigma, df, "put", "uo")
        vanilla = lib25.black76(F, K, T, sigma, df, "put")

        # Should be much less than vanilla
        assert uo_put < 0.01 * vanilla
