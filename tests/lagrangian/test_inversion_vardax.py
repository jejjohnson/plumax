"""Tier II inversions as vardax analyses (#163).

``linear_gaussian_inversion`` and ``lognormal_inversion`` now run
``vardax.OptimalInterpolation`` over :func:`footprint_operator`. These tests
pin them to the closed form they replaced: the gaussx Kalman gain with the
Joseph covariance update, kept here as the reference.
"""

from __future__ import annotations

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from plumax.lagrangian.inversion import (
    footprint_operator,
    linear_gaussian_inversion,
    lognormal_inversion,
    matern32_covariance,
    observation_covariance,
)


pytest.importorskip("vardax")


def _reference_blue(jacobian, innovation, prior_cov, obs_var):
    """The pre-vardax update: K = B Jᵀ (J B Jᵀ + R)⁻¹, Joseph-form P*."""
    psd = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    gain = gx.kalman_gain(
        lx.MatrixLinearOperator(prior_cov, tags=psd),
        lx.MatrixLinearOperator(jacobian),
        lx.DiagonalLinearOperator(obs_var),
    )
    covariance = gx.joseph_update(prior_cov, gain, jacobian, jnp.diag(obs_var))
    return gain @ innovation, covariance


def _scene(n_grid, n_obs, seed, variance, length_scale, noise):
    rng = np.random.default_rng(seed)
    coords = np.linspace(0.0, 1000.0, n_grid)
    f = jnp.asarray(rng.uniform(0.0, 1.0, size=(n_obs, n_grid)))
    q_true = rng.uniform(0.5, 2.0, size=n_grid)
    b = matern32_covariance(coords, variance=variance, length_scale=length_scale)
    r = observation_covariance(np.full(n_obs, noise))
    y = f @ q_true + np.sqrt(noise) * rng.standard_normal(n_obs)
    return f, jnp.asarray(y), jnp.full(n_grid, 1.0), b, r


# the Tier II test scenes: overdetermined, underdetermined, low noise
SCENES = {
    "overdetermined": (8, 12, 0, 1.0, 300.0, 1e-3),
    "underdetermined": (10, 4, 1, 0.5, 200.0, 1e-2),
    "low_noise": (8, 12, 2, 1.0, 300.0, 1e-6),
}


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def test_source_receptor_shapes():
    f, *_ = _scene(*SCENES["overdetermined"])
    q = jnp.linspace(0.5, 1.5, f.shape[1])
    predicted = footprint_operator(f)(q[None])
    assert predicted.shape == (1, f.shape[0])
    np.testing.assert_allclose(predicted[0], f @ q, rtol=1e-14)


@pytest.mark.parametrize("scene", SCENES)
def test_oi_matches_blue(scene):
    f, y, qa, b, r = _scene(*SCENES[scene])
    post = linear_gaussian_inversion(f, y, prior_mean=qa, prior_cov=b, obs_variance=r)
    increment, covariance = _reference_blue(f, y - f @ qa, b, r)
    assert _rel(post.mean - qa, increment) < 1e-10
    assert _rel(post.covariance, covariance) < 1e-10


@pytest.mark.parametrize("scene", SCENES)
def test_lognormal_matches_linearised_blue(scene):
    f, y, qa, b, r = _scene(*SCENES[scene])
    b_log = 0.25 * b
    post = lognormal_inversion(f, y, prior_mean=qa, prior_log_cov=b_log, obs_variance=r)
    increment, covariance = _reference_blue(f * qa[None, :], y - f @ qa, b_log, r)
    assert _rel(post.log_increment, increment) < 1e-10
    assert _rel(post.log_covariance, covariance) < 1e-10
    np.testing.assert_allclose(post.mean, qa * jnp.exp(increment), rtol=1e-10)
