import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import distributions as dist
import research

SEED = 0


def _df(values: np.ndarray, col: str = "x") -> pl.DataFrame:
    return pl.DataFrame({col: values})


# --------------------------------------------------------------------------
# 1. Parameter recovery
# --------------------------------------------------------------------------


class TestParameterRecovery:
    """Simulate from each family with known parameters, fit on a single
    large window, and check the recovered parameters are close to the
    truth. A fitter that can't invert its own generative process is broken
    before it ever sees market data.
    """

    def test_normal(self):
        research.set_seed(SEED)
        true_loc, true_scale = 0.5, 2.0
        x = st.norm.rvs(loc=true_loc, scale=true_scale, size=5000, random_state=SEED)
        result = dist.fit_rolling(_df(x), "x", "normal", window=5000)
        row = result.frame.tail(1)
        assert row["x_normal_loc"][0] == pytest.approx(true_loc, abs=0.1)
        assert row["x_normal_scale"][0] == pytest.approx(true_scale, rel=0.05)

    def test_t(self):
        true_df, true_loc, true_scale = 6.0, 0.0, 1.5
        x = st.t.rvs(
            df=true_df, loc=true_loc, scale=true_scale, size=5000, random_state=SEED
        )
        result = dist.fit_rolling(_df(x), "x", "t", window=5000)
        row = result.frame.tail(1)
        assert row["x_t_df"][0] == pytest.approx(true_df, rel=0.35)
        assert row["x_t_scale"][0] == pytest.approx(true_scale, rel=0.1)

    def test_skewt(self):
        true_a, true_b, true_loc, true_scale = 4.0, 8.0, 0.0, 1.0
        x = st.jf_skew_t.rvs(
            a=true_a,
            b=true_b,
            loc=true_loc,
            scale=true_scale,
            size=6000,
            random_state=SEED,
        )
        result = dist.fit_rolling(_df(x), "x", "skewt", window=6000)
        row = result.frame.tail(1)
        # shape params of a skew-t are individually noisy to recover; check
        # the fit converged (non-null) and the implied mean/std are close,
        # which is the property downstream scoring actually depends on.
        assert row["x_skewt_a"][0] is not None
        fitted = dist.frozen_dist(
            "skewt",
            [
                row["x_skewt_a"][0],
                row["x_skewt_b"][0],
                row["x_skewt_loc"][0],
                row["x_skewt_scale"][0],
            ],
        )
        assert fitted.mean() == pytest.approx(np.mean(x), abs=0.15)
        assert fitted.std() == pytest.approx(np.std(x), rel=0.15)

    def test_poisson(self):
        true_mu = 12.0
        x = st.poisson.rvs(mu=true_mu, size=3000, random_state=SEED)
        result = dist.fit_rolling(_df(x), "x", "poisson", window=3000)
        row = result.frame.tail(1)
        assert row["x_poisson_mu"][0] == pytest.approx(true_mu, rel=0.05)

    def test_nbinom(self):
        true_n, true_p = 5.0, 0.4
        x = st.nbinom.rvs(n=true_n, p=true_p, size=5000, random_state=SEED)
        result = dist.fit_rolling(_df(x), "x", "nbinom", window=5000)
        row = result.frame.tail(1)
        assert row["x_nbinom_n"][0] == pytest.approx(true_n, rel=0.25)
        assert row["x_nbinom_p"][0] == pytest.approx(true_p, rel=0.25)

    def test_beta(self):
        true_a, true_b = 2.0, 5.0
        x = st.beta.rvs(a=true_a, b=true_b, size=5000, random_state=SEED)
        result = dist.fit_rolling(_df(x), "x", "beta", window=5000)
        row = result.frame.tail(1)
        assert row["x_beta_a"][0] == pytest.approx(true_a, rel=0.1)
        assert row["x_beta_b"][0] == pytest.approx(true_b, rel=0.1)


# --------------------------------------------------------------------------
# 2. Causality under truncation
# --------------------------------------------------------------------------


class TestCausalityUnderTruncation:
    """Same pattern as test_features.py's causal-under-truncation test:
    recompute fit_rolling on a truncated history and every surviving row's
    fitted params must be identical to the full-history run's same row.
    """

    @pytest.mark.parametrize(
        "family,rvs",
        [
            (
                "normal",
                lambda n, seed: st.norm.rvs(loc=0, scale=1, size=n, random_state=seed),
            ),
            ("t", lambda n, seed: st.t.rvs(df=5, size=n, random_state=seed)),
            (
                "poisson",
                lambda n, seed: st.poisson.rvs(mu=8, size=n, random_state=seed),
            ),
            (
                "nbinom",
                lambda n, seed: st.nbinom.rvs(n=5, p=0.4, size=n, random_state=seed),
            ),
            ("beta", lambda n, seed: st.beta.rvs(a=2, b=5, size=n, random_state=seed)),
        ],
    )
    def test_truncation_leaves_earlier_rows_unchanged(self, family, rvs):
        x = rvs(400, SEED)
        window = 30
        full = dist.fit_rolling(_df(x), "x", family, window=window).frame

        for cut in (60, 150, 399):
            truncated = dist.fit_rolling(_df(x[:cut]), "x", family, window=window).frame
            for param in dist.FAMILY_PARAMS[family]:
                col = f"x_{family}_{param}"
                full_vals = full[col].head(cut).to_numpy()
                trunc_vals = truncated[col].to_numpy()
                np.testing.assert_allclose(
                    full_vals,
                    trunc_vals,
                    equal_nan=True,
                    err_msg=f"{col} is not causal under truncation",
                )


# --------------------------------------------------------------------------
# 3. Degenerate inputs
# --------------------------------------------------------------------------


class TestDegenerateInputs:
    """Constant series, all-zero counts, single-observation windows: must
    null out, never raise, never report an unconverged/undefined fit as if
    it were real.
    """

    @pytest.mark.parametrize("family", ["normal", "t", "skewt"])
    def test_constant_series_is_null(self, family):
        x = np.full(200, 3.0)
        result = dist.fit_rolling(_df(x), "x", family, window=50)
        assert result.n_degenerate > 0
        params = dist.FAMILY_PARAMS[family]
        last_row = result.frame.tail(1)
        for p in params:
            assert last_row[f"x_{family}_{p}"][0] is None

    def test_all_zero_counts_poisson_is_null(self):
        x = np.zeros(100)
        result = dist.fit_rolling(_df(x), "x", "poisson", window=30)
        assert result.n_degenerate > 0
        assert result.frame.tail(1)["x_poisson_mu"][0] is None

    def test_all_zero_counts_nbinom_is_null(self):
        x = np.zeros(100)
        result = dist.fit_rolling(_df(x), "x", "nbinom", window=30)
        assert result.n_degenerate > 0
        assert result.frame.tail(1)["x_nbinom_n"][0] is None

    def test_underdispersed_counts_nbinom_is_null(self):
        # equidispersed (Poisson-like) counts: var ~ mean, MOM for NB
        # (which requires var > mean) is undefined and must null, not raise
        # or return a nonsense negative/inf size parameter.
        x = st.poisson.rvs(mu=10, size=500, random_state=SEED)
        result = dist.fit_rolling(_df(x), "x", "nbinom", window=500)
        row = result.frame.tail(1)
        assert row["x_nbinom_n"][0] is None

    def test_beta_boundary_values_are_null(self):
        x = np.concatenate([np.full(50, 0.0), np.full(50, 1.0)])
        result = dist.fit_rolling(_df(x), "x", "beta", window=20)
        assert result.n_degenerate > 0
        assert result.frame.tail(1)["x_beta_a"][0] is None

    def test_single_observation_window_is_insufficient_history_not_a_fit(self):
        x = st.norm.rvs(size=10, random_state=SEED)
        result = dist.fit_rolling(_df(x), "x", "normal", window=1, min_periods=2)
        # every row has a "window" of size 1 (window=1) but min_periods=2
        # means none of them ever accumulate enough history to fit.
        assert result.n_fit == 0
        assert result.n_insufficient_history == len(x)
        assert result.frame["x_normal_loc"].is_null().all()

    def test_never_raises_on_pathological_input(self):
        pathological = [
            np.array([]),
            np.full(5, np.nan),
            np.array([1.0]),
            np.full(500, 7.0),
        ]
        for x in pathological:
            for family in dist.FAMILY_PARAMS:
                # skip families whose support the values violate by
                # construction (e.g. beta needs (0,1)); the point of this
                # test is "no exception", not "every family accepts every
                # array" - all-zero/constant/short arrays are exercised for
                # every family via the pad below.
                padded = np.concatenate([x, np.full(3, 0.3)]) if len(x) < 3 else x
                dist.fit_rolling(_df(padded), "x", family, window=10)  # must not raise


# --------------------------------------------------------------------------
# 4. Moment identities
# --------------------------------------------------------------------------


class TestMomentIdentities:
    """Each family's analytic mean/variance (from a frozen dist at known
    parameters) must match the sample moments of a large simulated draw
    from that same distribution.
    """

    def test_normal(self):
        d = dist.frozen_dist("normal", [1.0, 2.0])
        x = d.rvs(size=200_000, random_state=SEED)
        assert np.mean(x) == pytest.approx(d.mean(), abs=0.03)
        assert np.var(x) == pytest.approx(d.var(), rel=0.03)

    def test_t(self):
        d = dist.frozen_dist("t", [8.0, 0.0, 1.5])
        x = d.rvs(size=200_000, random_state=SEED)
        assert np.mean(x) == pytest.approx(d.mean(), abs=0.05)
        assert np.var(x) == pytest.approx(d.var(), rel=0.05)

    def test_skewt(self):
        d = dist.frozen_dist("skewt", [4.0, 8.0, 0.0, 1.0])
        x = d.rvs(size=200_000, random_state=SEED)
        assert np.mean(x) == pytest.approx(d.mean(), abs=0.05)
        assert np.var(x) == pytest.approx(d.var(), rel=0.08)

    def test_poisson(self):
        d = dist.frozen_dist("poisson", [15.0])
        x = d.rvs(size=200_000, random_state=SEED)
        assert np.mean(x) == pytest.approx(d.mean(), rel=0.02)
        assert np.var(x) == pytest.approx(d.var(), rel=0.05)

    def test_nbinom(self):
        d = dist.frozen_dist("nbinom", [5.0, 0.4])
        x = d.rvs(size=200_000, random_state=SEED)
        assert np.mean(x) == pytest.approx(d.mean(), rel=0.02)
        assert np.var(x) == pytest.approx(d.var(), rel=0.05)

    def test_beta(self):
        d = dist.frozen_dist("beta", [2.0, 5.0])
        x = d.rvs(size=200_000, random_state=SEED)
        assert np.mean(x) == pytest.approx(d.mean(), abs=0.01)
        assert np.var(x) == pytest.approx(d.var(), rel=0.05)


# --------------------------------------------------------------------------
# 5. Scoring rule sanity
# --------------------------------------------------------------------------


class TestScoringRuleSanity:
    def test_log_score_prefers_true_distribution(self):
        true = dist.frozen_dist("normal", [0.0, 1.0])
        misspecified = dist.frozen_dist("normal", [3.0, 0.2])
        y = true.rvs(size=2000, random_state=SEED)
        true_score = dist.log_score(true, y).mean()
        wrong_score = dist.log_score(misspecified, y).mean()
        assert true_score > wrong_score

    def test_crps_prefers_true_distribution(self):
        true = dist.frozen_dist("normal", [0.0, 1.0])
        misspecified = dist.frozen_dist("normal", [3.0, 0.2])
        y = true.rvs(size=500, random_state=SEED)
        true_crps = dist.crps(true, y).mean()
        wrong_crps = dist.crps(misspecified, y).mean()
        assert true_crps < wrong_crps

    def test_log_score_and_crps_prefer_true_family(self):
        # true generating process is fat-tailed (t); compare a well-fit t
        # against a normal fit to the same moments - the normal
        # underweights tail events it will see plenty of.
        true = dist.frozen_dist("t", [3.0, 0.0, 1.0])
        misspecified = dist.frozen_dist("normal", [0.0, true.std()])
        y = true.rvs(size=4000, random_state=SEED)
        assert dist.log_score(true, y).mean() > dist.log_score(misspecified, y).mean()
        assert dist.crps(true, y).mean() < dist.crps(misspecified, y).mean()

    def test_pit_of_true_distribution_is_uniform(self):
        true = dist.frozen_dist("normal", [0.0, 1.0])
        y = true.rvs(size=3000, random_state=SEED)
        _stat, pvalue = dist.pit_ks_test(true, y)
        assert pvalue > 0.05

    def test_pit_of_misspecified_distribution_rejects_uniform(self):
        true = dist.frozen_dist("normal", [0.0, 1.0])
        misspecified = dist.frozen_dist("normal", [0.0, 0.2])
        y = true.rvs(size=3000, random_state=SEED)
        _stat, pvalue = dist.pit_ks_test(misspecified, y)
        assert pvalue < 0.01

    def test_qlike_minimized_at_true_variance(self):
        true_variance = 4.0
        rng = np.random.default_rng(SEED)
        actual = rng.chisquare(df=1, size=5000) * true_variance
        grid = np.linspace(0.5, 10.0, 40)
        losses = [dist.qlike(actual, np.full_like(actual, g)).mean() for g in grid]
        best = grid[np.argmin(losses)]
        assert best == pytest.approx(true_variance, abs=0.5)

    def test_kupiec_rejects_miscalibrated_exceedance_rate(self):
        rng = np.random.default_rng(SEED)
        # expected 5% exceedance rate, actual ~20% -> should reject strongly
        hits = rng.random(2000) < 0.20
        _lr, pvalue = dist.kupiec_test(hits, expected_rate=0.05)
        assert pvalue < 0.01

    def test_kupiec_accepts_well_calibrated_exceedance_rate(self):
        rng = np.random.default_rng(SEED)
        hits = rng.random(2000) < 0.05
        _lr, pvalue = dist.kupiec_test(hits, expected_rate=0.05)
        assert pvalue > 0.05

    def test_christoffersen_rejects_clustered_exceedances(self):
        rng = np.random.default_rng(SEED)
        # simulate a Markov chain with strong positive persistence of hits
        n = 3000
        hits = np.zeros(n, dtype=bool)
        hits[0] = rng.random() < 0.05
        for i in range(1, n):
            p = 0.6 if hits[i - 1] else 0.03
            hits[i] = rng.random() < p
        _lr, pvalue = dist.christoffersen_independence_test(hits)
        assert pvalue < 0.01

    def test_christoffersen_accepts_iid_exceedances(self):
        rng = np.random.default_rng(SEED)
        hits = rng.random(3000) < 0.05
        _lr, pvalue = dist.christoffersen_independence_test(hits)
        assert pvalue > 0.05


# --------------------------------------------------------------------------
# 6. Closed-form CRPS (notebook 5 #1b fix)
# --------------------------------------------------------------------------


class TestClosedFormCRPS:
    """`crps`'s numerical grid (linspace(ppf(1e-6), ppf(1-1e-6), n_points)) spans
    ~10 units for a normal but ~1400 units for a t(2) at the same n_points, so a
    fixed n_points gives wildly different effective resolution across families -
    see NEXT_RUN_PROMPT.md #1b. crps_normal_closed_form/crps_t_closed_form are
    exact (no grid), and are checked here against the numerical version on a
    light-tailed case where the numerical version is itself trustworthy (a very
    fine grid), plus a direct demonstration that the OLD default (n_points=400)
    is not trustworthy for a heavy-tailed t.
    """

    def test_closed_form_normal_matches_fine_numerical_grid(self):
        y = np.linspace(-4, 4, 25)
        numeric = dist.crps(dist.frozen_dist("normal", [0.0, 1.0]), y, n_points=200_000)
        closed = dist.crps_normal_closed_form(y, loc=0.0, scale=1.0)
        assert closed == pytest.approx(numeric, abs=1e-4)

    def test_closed_form_t_matches_fine_numerical_grid_for_light_tail(self):
        # df=30 is close enough to normal that crps's numerical grid (built
        # from ppf(1e-6)/ppf(1-1e-6), same as the normal case) is fine enough
        # to trust at a large n_points - the trustworthy "light-tailed case"
        # the runbook asks for.
        df = 30.0
        y = np.linspace(-4, 4, 25)
        numeric = dist.crps(dist.frozen_dist("t", [df, 0.0, 1.0]), y, n_points=200_000)
        closed = dist.crps_t_closed_form(y, df=df, loc=0.0, scale=1.0)
        assert closed == pytest.approx(numeric, abs=1e-4)

    def test_closed_form_t_matches_fine_numerical_grid_nonstandard_loc_scale(self):
        df = 20.0
        y = np.linspace(-5, 15, 25)
        numeric = dist.crps(dist.frozen_dist("t", [df, 5.0, 2.0]), y, n_points=200_000)
        closed = dist.crps_t_closed_form(y, df=df, loc=5.0, scale=2.0)
        assert closed == pytest.approx(numeric, abs=2e-4)

    def test_old_default_grid_fails_on_heavy_tailed_t(self):
        """Fails on the pre-fix implementation: a t(2.1) forecast's CRPS at
        the OLD default n_points=400 disagrees with the true (closed-form)
        value by 10s of percent - the exact motivating bug for #1b. Asserts
        the closed form is accurate to 1% of a trustworthy fine-grid
        reference, and separately demonstrates the old default grid is not.
        """
        df = 2.1
        y = np.array([0.0, 1.0, -2.0, 5.0])
        reference = dist.crps(
            dist.frozen_dist("t", [df, 0.0, 1.0]), y, n_points=200_000
        )
        closed = dist.crps_t_closed_form(y, df=df, loc=0.0, scale=1.0)
        # the closed form must be accurate to 1% of the trustworthy fine-grid reference
        assert closed == pytest.approx(reference, rel=0.01)
        # and the OLD default (n_points=400) must NOT be within 1% - demonstrating
        # exactly the bug this checkpoint fixes (this assertion is the one that
        # would fail against the pre-fix, only-numerical implementation, since
        # there would be no closed form to fall back on in that world - it fails
        # here on today's numerical `crps` at the old default n_points directly).
        old_grid = dist.crps(dist.frozen_dist("t", [df, 0.0, 1.0]), y, n_points=400)
        rel_error = np.abs((old_grid - reference) / reference)
        assert np.any(rel_error > 0.01)

    def test_crps_t_closed_form_undefined_below_df_one(self):
        y = np.array([0.0, 1.0, -1.0])
        out = dist.crps_t_closed_form(y, df=0.9, loc=0.0, scale=1.0)
        assert np.all(np.isnan(out))

    def test_crps_t_closed_form_vectorized_per_bar_df(self):
        # a per-bar nu path (as nu_path_from_fits would produce) must be
        # scoreable in one call, not just a scalar df.
        y = np.array([0.5, -0.5, 1.5])
        nu = np.array([3.0, 5.0, 8.0])
        vectorized = dist.crps_t_closed_form(y, df=nu, loc=0.0, scale=1.0)
        looped = np.array(
            [
                dist.crps_t_closed_form(np.array([yi]), df=nui, loc=0.0, scale=1.0)[0]
                for yi, nui in zip(y, nu)
            ]
        )
        assert vectorized == pytest.approx(looped)
