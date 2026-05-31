"""Tests for the Tier II model-based source inversion."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.lagrangian.inversion import (
    GaussianPosterior,
    LognormalPosterior,
    linear_gaussian_inversion,
    lognormal_inversion,
    matern32_covariance,
    observation_covariance,
)


# ── Matérn-3/2 prior ─────────────────────────────────────────────────────────


def test_matern32_is_spd_with_unit_diagonal_variance():
    coords = np.linspace(0.0, 1000.0, 12)
    b = np.asarray(matern32_covariance(coords, variance=4.0, length_scale=200.0))
    # Symmetric.
    np.testing.assert_allclose(b, b.T, atol=1e-12)
    # Diagonal equals the marginal variance (r = 0 → k = σ²).
    np.testing.assert_allclose(np.diag(b), 4.0, atol=1e-10)
    # Positive-definite.
    assert np.all(np.linalg.eigvalsh(b) > 0.0)


def test_matern32_decays_with_distance():
    coords = np.linspace(0.0, 1000.0, 6)
    b = np.asarray(matern32_covariance(coords, variance=1.0, length_scale=150.0))
    # Correlation to cell 0 is monotone non-increasing with separation.
    row = b[0]
    assert np.all(np.diff(row) <= 1e-12)


def test_matern32_longer_length_scale_is_more_correlated():
    coords = np.linspace(0.0, 1000.0, 6)
    short = np.asarray(matern32_covariance(coords, variance=1.0, length_scale=100.0))
    long = np.asarray(matern32_covariance(coords, variance=1.0, length_scale=400.0))
    # At a fixed separation the longer length-scale keeps more correlation.
    assert long[0, -1] > short[0, -1]


def test_matern32_validates_positive_hyperparameters():
    coords = np.linspace(0.0, 1.0, 4)
    with pytest.raises(ValueError, match="variance"):
        matern32_covariance(coords, variance=0.0, length_scale=1.0)
    with pytest.raises(ValueError, match="length_scale"):
        matern32_covariance(coords, variance=1.0, length_scale=0.0)


# ── Observation covariance ──────────────────────────────────────────────────


def test_observation_covariance_adds_representation_error():
    retr = np.array([1.0, 2.0, 3.0])
    r = np.asarray(observation_covariance(retr, representation_variance=0.5))
    np.testing.assert_allclose(r, [1.5, 2.5, 3.5])


def test_observation_covariance_defaults_to_retrieval_only():
    retr = np.array([1.0, 2.0])
    np.testing.assert_allclose(np.asarray(observation_covariance(retr)), retr)


# ── Linear Gaussian inversion ───────────────────────────────────────────────


def _toy_problem(n_grid=8, n_obs=12, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.linspace(0.0, 1000.0, n_grid)
    f = rng.uniform(0.0, 1.0, size=(n_obs, n_grid))
    q_true = rng.uniform(0.5, 2.0, size=n_grid)
    return coords, f, q_true


def test_linear_inversion_recovers_truth_low_noise():
    # Twin experiment: synthesise obs from a known source, invert, recover it.
    coords, f, q_true = _toy_problem()
    n_obs, n_grid = f.shape
    y = f @ q_true  # noiseless observations
    b = matern32_covariance(coords, variance=1.0, length_scale=300.0)
    r = observation_covariance(np.full(n_obs, 1e-6))
    post = linear_gaussian_inversion(
        f, y, prior_mean=np.full(n_grid, 1.0), prior_cov=b, obs_variance=r
    )
    assert isinstance(post, GaussianPosterior)
    # Tight likelihood → posterior mean tracks the truth closely.
    np.testing.assert_allclose(np.asarray(post.mean), q_true, rtol=5e-2, atol=5e-2)


def test_linear_inversion_returns_to_prior_when_obs_uninformative():
    coords, f, _ = _toy_problem()
    n_obs, n_grid = f.shape
    qa = np.full(n_grid, 1.3)
    y = f @ qa  # innovation is exactly zero
    b = matern32_covariance(coords, variance=1.0, length_scale=300.0)
    r = observation_covariance(np.full(n_obs, 1e3))  # very loose
    post = linear_gaussian_inversion(f, y, prior_mean=qa, prior_cov=b, obs_variance=r)
    # Zero innovation → posterior mean stays at the prior mean.
    np.testing.assert_allclose(np.asarray(post.mean), qa, atol=1e-6)


def test_linear_inversion_reduces_variance():
    coords, f, q_true = _toy_problem()
    n_obs, n_grid = f.shape
    y = f @ q_true
    b = np.asarray(matern32_covariance(coords, variance=1.0, length_scale=300.0))
    r = observation_covariance(np.full(n_obs, 1e-2))
    post = linear_gaussian_inversion(
        f, y, prior_mean=np.full(n_grid, 1.0), prior_cov=b, obs_variance=r
    )
    # Observing data cannot increase the posterior variance over the prior.
    assert np.all(np.asarray(post.std) <= np.sqrt(np.diag(b)) + 1e-8)
    # Posterior covariance stays symmetric PSD.
    p = np.asarray(post.covariance)
    np.testing.assert_allclose(p, p.T, atol=1e-9)
    assert np.min(np.linalg.eigvalsh(p)) > -1e-8


def test_linear_inversion_shape_validation():
    coords, f, q_true = _toy_problem()
    n_obs, n_grid = f.shape
    b = matern32_covariance(coords, variance=1.0, length_scale=300.0)
    r = observation_covariance(np.full(n_obs, 1.0))
    with pytest.raises(ValueError, match="observation"):
        linear_gaussian_inversion(
            f, np.zeros(n_obs + 1), prior_mean=q_true, prior_cov=b, obs_variance=r
        )
    with pytest.raises(ValueError, match="prior_mean"):
        linear_gaussian_inversion(
            f,
            f @ q_true,
            prior_mean=np.zeros(n_grid + 1),
            prior_cov=b,
            obs_variance=r,
        )


# ── Lognormal (sign-constrained) inversion ──────────────────────────────────


def test_lognormal_inversion_is_nonnegative_and_recovers_truth():
    # The linear-in-log estimator is valid for *moderate* enhancements over the
    # prior, so the twin truth is a small (≤ ~15 %) perturbation of q_a.
    coords, f, _ = _toy_problem()
    n_obs, n_grid = f.shape
    rng = np.random.default_rng(11)
    qa = np.full(n_grid, 1.0)
    q_true = qa * np.exp(rng.uniform(-0.15, 0.15, size=n_grid))
    y = f @ q_true
    b_log = matern32_covariance(coords, variance=0.25, length_scale=300.0)
    r = observation_covariance(np.full(n_obs, 1e-6))
    post = lognormal_inversion(f, y, prior_mean=qa, prior_log_cov=b_log, obs_variance=r)
    assert isinstance(post, LognormalPosterior)
    # Non-negativity is guaranteed by q = q_a · exp(δ).
    assert np.all(np.asarray(post.mean) >= 0.0)
    # Linearised about the prior, so it tracks a moderate-enhancement truth.
    np.testing.assert_allclose(np.asarray(post.mean), q_true, rtol=0.1, atol=0.05)


def test_lognormal_inversion_strictly_positive_under_large_negative_increment():
    # Even when the data push δ strongly negative, q = q_a · exp(δ) stays > 0
    # (the property a Gaussian inversion on q would violate by going negative).
    coords, f, _ = _toy_problem()
    n_obs, n_grid = f.shape
    qa = np.full(n_grid, 1.0)
    # Observations consistent with a near-zero source → large negative δ.
    y = f @ (qa * 0.05)
    b_log = matern32_covariance(coords, variance=1.0, length_scale=300.0)
    r = observation_covariance(np.full(n_obs, 1e-6))
    post = lognormal_inversion(f, y, prior_mean=qa, prior_log_cov=b_log, obs_variance=r)
    assert np.all(np.asarray(post.mean) > 0.0)


def test_lognormal_inversion_rejects_nonpositive_prior_mean():
    coords, f, _ = _toy_problem()
    n_obs, n_grid = f.shape
    b_log = matern32_covariance(coords, variance=0.25, length_scale=300.0)
    r = observation_covariance(np.full(n_obs, 1.0))
    qa = np.full(n_grid, 1.0)
    qa[0] = 0.0
    with pytest.raises(ValueError, match="q_a"):
        lognormal_inversion(
            f, np.zeros(n_obs), prior_mean=qa, prior_log_cov=b_log, obs_variance=r
        )


def test_lognormal_zero_innovation_returns_prior_mean():
    coords, f, _ = _toy_problem()
    n_obs, n_grid = f.shape
    qa = np.full(n_grid, 1.2)
    y = f @ qa  # innovation zero → δ* = 0 → q* = q_a
    b_log = matern32_covariance(coords, variance=0.25, length_scale=300.0)
    r = observation_covariance(np.full(n_obs, 1.0))
    post = lognormal_inversion(f, y, prior_mean=qa, prior_log_cov=b_log, obs_variance=r)
    np.testing.assert_allclose(np.asarray(post.log_increment), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(post.mean), qa, rtol=1e-6)


def test_inversion_with_real_footprint_matrix():
    # End-to-end: build F from actual backward footprints (one row per
    # receptor), synthesise observations from a known surface source, and
    # recover it. Exercises the full footprint → inversion handoff.
    from plumax.lagrangian.footprint import compute_footprint
    from plumax.lagrangian.particles import wind_from_speed_direction
    from plumax.lagrangian.turbulence import HomogeneousTurbulence

    turb = HomogeneousTurbulence.isotropic(sigma=0.6, tau=30.0)
    wind = wind_from_speed_direction(4.0, 270.0)  # from west → flows east
    domain_x = (-100.0, 500.0, 12)
    domain_y = (-200.0, 200.0, 8)
    receptors = [(300.0, 0.0, 20.0), (350.0, 50.0, 20.0), (250.0, -50.0, 20.0)]

    rows, xc, yc = [], None, None
    for r in receptors:
        fp, xc, yc = compute_footprint(
            r,
            turb,
            domain_x=domain_x,
            domain_y=domain_y,
            wind=wind,
            n_particles=2000,
            t_back=150.0,
            dt=1.0,
            seed=0,
        )
        rows.append(np.asarray(fp).reshape(-1))
    f = np.stack(rows, axis=0)  # (n_obs, n_grid)
    n_obs, n_grid = f.shape

    # Coordinates for the Matérn prior over the flattened (x, y) surface grid.
    gx, gy = np.meshgrid(xc, yc, indexing="ij")
    coords = np.column_stack([gx.reshape(-1), gy.reshape(-1)])

    q_true = np.full(n_grid, 0.1)
    q_true[n_grid // 2] = 1.0  # a localised hot-spot
    y = f @ q_true

    b = matern32_covariance(coords, variance=1.0, length_scale=150.0)
    r = observation_covariance(np.full(n_obs, 1e-8))
    post = linear_gaussian_inversion(
        f, y, prior_mean=np.full(n_grid, 0.1), prior_cov=b, obs_variance=r
    )
    # The inversion reproduces the observations it was given (data consistency).
    np.testing.assert_allclose(np.asarray(f @ post.mean), y, rtol=1e-3, atol=1e-3)


def test_inversion_is_jittable():
    import jax

    coords, f, q_true = _toy_problem()
    n_obs, n_grid = f.shape
    b = matern32_covariance(coords, variance=1.0, length_scale=300.0)
    r = observation_covariance(np.full(n_obs, 1e-3))
    fj, yj, qaj = jnp.asarray(f), jnp.asarray(f @ q_true), jnp.ones(n_grid)

    @jax.jit
    def mean_of(y):
        return linear_gaussian_inversion(
            fj, y, prior_mean=qaj, prior_cov=b, obs_variance=r
        ).mean

    out = mean_of(yj)
    assert out.shape == (n_grid,)
