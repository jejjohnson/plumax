"""The met-ensemble runner and the inversion that marginalises over it (#89)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.lagrangian import (
    GriddedWind,
    HomogeneousTurbulence,
    ensemble_footprints,
    ensemble_inversion,
    linear_gaussian_inversion,
    matern32_covariance,
    transport_covariance,
    uniform_wind,
)
from plumax.les_fvm import make_grid


pytest.importorskip("vardax")

GRID = make_grid(
    (-400.0, 400.0, 8), (-200.0, 200.0, 4), (0.0, 200.0, 4), dtype=jnp.float64
)
TURBULENCE = HomogeneousTurbulence(0.5, 0.5, 0.3, 20.0, 20.0, 10.0)
RECEPTORS = jnp.array([[150.0, 0.0, 10.0], [250.0, 50.0, 10.0]])
FOOTPRINT_KWARGS = {
    "n_particles": 200,
    "t_back": 60.0,
    "dt": 5.0,
    "pbl_height": 150.0,
}


def _gridded(u: float, v: float) -> GriddedWind:
    nz, ny, nx = GRID.interior_shape
    shape = (2, nz, ny, nx)
    return GriddedWind.from_grid(
        GRID,
        jnp.array([-3600.0, 0.0]),
        jnp.full(shape, u),
        jnp.full(shape, v),
        jnp.zeros(shape),
    )


def _toy_problem():
    footprint = jnp.array(
        [[1.0, 0.5, 0.1], [0.2, 1.0, 0.4], [0.1, 0.3, 1.0], [0.6, 0.2, 0.2]]
    )
    coordinates = jnp.array([0.0, 1.0, 2.0])
    prior_mean = jnp.array([1.0, 2.0, 1.5])
    prior_cov = matern32_covariance(coordinates, variance=0.5, length_scale=1.0)
    obs_variance = jnp.full(4, 0.05)
    return footprint, prior_mean, prior_cov, obs_variance


def test_ensemble_footprints_shape_and_common_draws():
    winds = [_gridded(3.0, 0.0), _gridded(3.0, 0.0), uniform_wind(3.0, 0.0)]
    footprints = ensemble_footprints(
        winds,
        RECEPTORS,
        TURBULENCE,
        (-400.0, 400.0, 8),
        (-200.0, 200.0, 4),
        **FOOTPRINT_KWARGS,
    )
    assert footprints.shape == (3, 2, 32)
    # Same wind, same particle draws: the members agree exactly
    np.testing.assert_array_equal(footprints[0], footprints[1])
    np.testing.assert_array_equal(footprints[0], footprints[2])
    assert float(footprints.sum()) > 0.0


def test_ensemble_footprints_differ_with_the_wind():
    winds = [_gridded(3.0, 0.0), _gridded(3.0, 1.5)]
    footprints = ensemble_footprints(
        winds,
        RECEPTORS,
        TURBULENCE,
        (-400.0, 400.0, 8),
        (-200.0, 200.0, 4),
        **FOOTPRINT_KWARGS,
    )
    assert not np.array_equal(footprints[0], footprints[1])
    with pytest.raises(ValueError, match="at least one"):
        ensemble_footprints([], RECEPTORS, TURBULENCE, (0.0, 1.0, 1), (0.0, 1.0, 1))


def test_identical_members_reduce_to_single():
    footprint, prior_mean, prior_cov, obs_variance = _toy_problem()
    observation = footprint @ jnp.array([1.4, 1.8, 1.0])
    single = linear_gaussian_inversion(
        footprint,
        observation,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        obs_variance=obs_variance,
    )
    for n_members in (1, 4):
        posterior = ensemble_inversion(
            jnp.broadcast_to(footprint, (n_members, *footprint.shape)),
            observation,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            obs_variance=obs_variance,
        )
        # F̄ = F_0 and R_transport = 0 exactly, so the result is bit-identical
        np.testing.assert_array_equal(posterior.mean, single.mean)
        np.testing.assert_array_equal(posterior.covariance, single.covariance)


def test_transport_covariance_is_the_sample_covariance():
    footprint, prior_mean, _, _ = _toy_problem()
    perturbation = jax.random.normal(jax.random.PRNGKey(0), (5, *footprint.shape))
    footprints = footprint + 0.1 * perturbation
    mean_footprint, covariance = transport_covariance(footprints, prior_mean)
    np.testing.assert_allclose(mean_footprint, footprints.mean(axis=0), rtol=1e-12)
    responses = np.asarray(footprints @ prior_mean)
    np.testing.assert_allclose(
        covariance, np.cov(responses, rowvar=False), rtol=1e-10, atol=1e-14
    )


def test_cv_grows_with_spread():
    # y = F̄ q_a: the innovation is zero, the posterior mean stays at q_a and
    # CV(q) = std / q_a tracks the posterior spread alone.
    footprint, prior_mean, prior_cov, obs_variance = _toy_problem()
    perturbation = jax.random.normal(jax.random.PRNGKey(1), (6, *footprint.shape))
    # zero ensemble mean, so F̄ = F at every spread
    perturbation = perturbation - perturbation.mean(axis=0)
    observation = footprint @ prior_mean
    cvs = []
    for spread in (0.0, 0.05, 0.1, 0.2, 0.4):
        posterior = ensemble_inversion(
            footprint + spread * perturbation,
            observation,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            obs_variance=obs_variance,
        )
        np.testing.assert_allclose(posterior.mean, prior_mean, rtol=1e-10)
        cvs.append(np.asarray(posterior.std / posterior.mean))
    cvs = np.stack(cvs)
    assert np.all(np.diff(cvs, axis=0) > 0.0)


def test_rejects_a_two_dimensional_footprint():
    footprint, prior_mean, prior_cov, obs_variance = _toy_problem()
    # ValueError at run time; the jaxtyping hook in the tests catches it first
    with pytest.raises((ValueError, TypeError)):
        ensemble_inversion(
            footprint,
            jnp.zeros(4),
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            obs_variance=obs_variance,
        )
