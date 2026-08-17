"""Notebook 017 sec 7.3: the seven load-bearing tests for dsr_lib17.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "research" / "tmp")
)

import dsr_lib17 as L

import research


def test_v0_is_bit_for_bit_unchanged():
    """Test 1: the V0 path reproduces the two constants this repo has
    already published (sec 8's backward-compatibility contract as an
    executable assertion)."""
    annualized_rate = float(np.sqrt(252))
    out_ac = L.dsr_variant(
        0.9042246451305482 / annualized_rate, n_trials=4, n_obs=4507, variant="v0"
    )
    assert out_ac["probability"] == research.deflated_sharpe_prob(
        0.9042246451305482 / annualized_rate, n_trials=4, n_obs=4507
    )
    assert abs(out_ac["probability"] - 0.997) < 0.002
    assert out_ac["probability"] == 0.9971831492023995

    out_18 = L.dsr_variant(
        0.01743215308672331,
        n_trials=18,
        n_obs=3837,
        skew=-11.516325584172863,
        kurtosis=816.8538707698766,
        variant="v0",
    )
    assert out_18["probability"] == research.deflated_sharpe_prob(
        0.01743215308672331,
        n_trials=18,
        n_obs=3837,
        skew=-11.516325584172863,
        kurtosis=816.8538707698766,
    )
    assert out_18["probability"] == 0.18590973717553716


def test_v1_reduces_to_v0_when_dispersion_equals_se():
    """Test 2: construct trial Sharpes whose sample std equals se_hat(SR);
    V1 and V0 must agree to floating-point. Pins that the only thing
    changed is the dispersion input."""
    sharpe, n_trials, n_obs = 0.06, 8, 500
    se = float(L._sr_se(sharpe, n_obs, 0.0, 3.0))

    # 8 values with sample std (ddof=1) exactly equal to se, mean = sharpe:
    # +-se*sqrt(n/(n-1))/2 alternating around the mean reproduces a known std.
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(n_trials)
    raw -= raw.mean()
    raw *= se / raw.std(ddof=1)
    trial_sharpes = (sharpe + raw).tolist()
    assert abs(float(np.std(trial_sharpes, ddof=1)) - se) < 1e-12

    v0 = L.dsr_variant(sharpe, n_trials, n_obs, variant="v0")
    v1 = L.dsr_variant(
        sharpe, n_trials, n_obs, variant="v1", trial_sharpes=trial_sharpes
    )
    assert abs(v0["probability"] - v1["probability"]) < 1e-10
    assert abs(v0["dispersion_used"] - v1["dispersion_used"]) < 1e-10


def test_identical_trials_impose_no_penalty():
    """Test 3: N identical trial Sharpes => dispersion 0 => SR*=0 =>
    DSR = PSR(0). The whole thesis, pinned."""
    sharpe, n_trials, n_obs = 0.05, 12, 1000
    trial_sharpes = [sharpe] * n_trials

    out = L.dsr_variant(
        sharpe, n_trials, n_obs, variant="v1", trial_sharpes=trial_sharpes
    )
    assert (
        out["dispersion_used"] < 1e-9
    )  # float noise on identical inputs, not exactly 0
    assert out["sr_star"] < 1e-9
    assert abs(out["probability"] - L.psr_upper_bound(sharpe, n_obs)) < 1e-9


def test_dispersed_trials_penalise_more_than_v0():
    """Test 4: a family with cross-sectional std well above se_hat yields a
    LOWER DSR than V0. Pins sec 1.3: not a one-way ratchet upward."""
    sharpe, n_trials, n_obs = 0.05, 8, 1000
    se = float(L._sr_se(sharpe, n_obs, 0.0, 3.0))

    rng = np.random.default_rng(1)
    raw = rng.standard_normal(n_trials)
    raw -= raw.mean()
    wide_std = 10 * se
    raw *= wide_std / raw.std(ddof=1)
    trial_sharpes = (sharpe + raw).tolist()
    assert float(np.std(trial_sharpes, ddof=1)) > se

    v0 = L.dsr_variant(sharpe, n_trials, n_obs, variant="v0")
    v1 = L.dsr_variant(
        sharpe, n_trials, n_obs, variant="v1", trial_sharpes=trial_sharpes
    )
    assert v1["probability"] < v0["probability"]


def test_family_mismatch_is_flagged_not_silent():
    """Test 5: n_trials=18 with 12 trial Sharpes uses 18 in the bracket
    (never len(trial_sharpes)) and sets family_mismatch=True (sec 2.2
    failure mode 1)."""
    rng = np.random.default_rng(2)
    trial_sharpes = rng.normal(0.01, 0.02, size=12).tolist()

    out = L.dsr_variant(
        0.05, n_trials=18, n_obs=1000, variant="v1", trial_sharpes=trial_sharpes
    )
    assert out["family_mismatch"] is True
    assert out["n_trials_used_in_bracket"] == 18
    assert out["n_trial_sharpes_provided"] == 12

    match = L.dsr_variant(
        0.05, n_trials=12, n_obs=1000, variant="v1", trial_sharpes=trial_sharpes
    )
    assert match["family_mismatch"] is False


def test_mc_cell_is_seed_and_resume_reproducible():
    """Test 6: same cell key => identical output. Resumability (sec 9.1) is
    per-cell, not per-replication: a cell that was interrupted and re-run
    from scratch must give bit-identical results to one that ran straight
    through, which holds iff the seed is a deterministic function of the
    cell's own key rather than a running counter."""
    key = (8, 500, 0.5, L.GAUSSIAN_MOMENTS, 300, 0.0)
    seed_a = L.seed_for_cell(key)
    seed_b = L.seed_for_cell(key)
    assert seed_a == seed_b

    n_trials, n_obs, rho, moments, n_reps, true_sharpe = key
    out_a = L.mc_cell(n_trials, n_obs, rho, moments, n_reps, seed_a, true_sharpe)
    out_b = L.mc_cell(n_trials, n_obs, rho, moments, n_reps, seed_b, true_sharpe)
    assert out_a["rate"] == out_b["rate"]
    assert out_a["achieved_moments"] == out_b["achieved_moments"]

    other_key = (8, 500, 0.9, L.GAUSSIAN_MOMENTS, 300, 0.0)
    assert L.seed_for_cell(other_key) != seed_a


def test_psr_upper_bound_dominates_every_variant():
    """Test 7: over a randomized sweep of inputs, psr_upper_bound is >=
    every variant's output. Phase 5's "cannot flip" shortcut (sec 5.4 item
    2) rests entirely on this."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        sharpe = float(rng.uniform(-0.1, 0.15))
        n_trials = int(rng.integers(2, 100))
        n_obs = int(rng.integers(50, 4000))
        skew = float(rng.uniform(-3, 1))
        kurtosis = float(rng.uniform(3, 20))

        upper = L.psr_upper_bound(sharpe, n_obs, skew, kurtosis)

        trial_sharpes = (sharpe + rng.normal(0, 0.05, size=n_trials)).tolist()
        for variant, kwargs in [
            ("v0", {}),
            ("v1", {"trial_sharpes": trial_sharpes}),
            ("v2", {"mean_pairwise_corr": float(rng.uniform(0.0, 0.99))}),
            ("v1b", {"trial_sharpes": trial_sharpes, "shrinkage_c": 0.25}),
            ("v1b", {"trial_sharpes": trial_sharpes, "shrinkage_c": 0.5}),
        ]:
            out = L.dsr_variant(
                sharpe, n_trials, n_obs, skew, kurtosis, variant=variant, **kwargs
            )
            assert out["probability"] <= upper + 1e-9, (variant, kwargs, out, upper)
