"""Tests for the Tier III Eulerian 4D-Var inversion.

These use a deliberately small grid / short window so the differentiable FV
solve stays fast; the point is to exercise the cost, the exact-adjoint gradient,
the whitening transform, and end-to-end twin recovery — not large-scale physics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.lagrangian.inversion import matern32_covariance
from plumax.les_fvm.fourdvar import (
    ColumnObservationOperator,
    EulerianForward4DVar,
    FourDVarProblem,
    FourDVarResult,
    build_forward,
    build_problem,
    solve_4dvar,
)


def _forward(n_t: int = 5):
    save_times = jnp.linspace(0.0, 40.0, n_t)
    return build_forward(
        domain_x=(0.0, 400.0, 20),
        domain_y=(-100.0, 100.0, 10),
        domain_z=(0.0, 80.0, 4),
        save_times=save_times,
        source_location=(50.0, 0.0, 20.0),
        uniform_wind=(4.0, 0.0, 0.0),
        eddy_diffusivity=2.0,
        advection_scheme="upwind1",
    )


# ── observation operator ─────────────────────────────────────────────────────


def test_column_operator_full_grid():
    # Column integral = sum over z * dz, flattened over (ny, nx).
    op = ColumnObservationOperator(dz=2.0)
    field = jnp.ones((4, 3, 5))  # (nz, ny, nx)
    obs = op(field)
    assert obs.shape == (15,)
    np.testing.assert_allclose(np.asarray(obs), 4 * 2.0)  # nz * dz


def test_column_operator_receptor_subset():
    op = ColumnObservationOperator(dz=1.0, receptor_index=jnp.array([[0, 0], [2, 4]]))
    field = jnp.arange(4 * 3 * 5, dtype=float).reshape(4, 3, 5)
    obs = op(field)
    assert obs.shape == (2,)
    col = np.asarray(field).sum(axis=0)
    np.testing.assert_allclose(np.asarray(obs), [col[0, 0], col[2, 4]])


# ── forward ──────────────────────────────────────────────────────────────────


def test_forward_predict_shape_and_positivity():
    fwd = _forward()
    assert isinstance(fwd, EulerianForward4DVar)
    q = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    y = fwd.predict(q)
    assert y.shape == (5, 200)  # (n_t, ny*nx)
    # No emission before t=0 release builds up → first frame is ~zero.
    np.testing.assert_allclose(np.asarray(y[0]), 0.0, atol=1e-12)
    # Concentration is non-negative and grows once emission starts.
    assert np.all(np.asarray(y) >= -1e-12)
    assert float(y.max()) > 0.0


def test_forward_scales_with_emission():
    fwd = _forward()
    q = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    y1 = fwd.predict(q)
    y2 = fwd.predict(2.0 * q)
    # Transport + column integral are linear in the emission rate.
    np.testing.assert_allclose(np.asarray(y2), 2.0 * np.asarray(y1), rtol=1e-5)


# ── cost + adjoint gradient ──────────────────────────────────────────────────


def _problem(fwd, q_true, *, length_scale=20.0, obs_floor=1e-12):
    y = fwd.predict(q_true)
    n_t = fwd.save_times.shape[0]
    b = matern32_covariance(fwd.save_times, variance=1.0, length_scale=length_scale)
    var = (float(y.max()) * 1e-3) ** 2 + obs_floor
    return build_problem(
        forward=fwd,
        observations=y,
        prior_mean=jnp.full(n_t, 0.5),
        prior_covariance=b,
        obs_variance=var,
    )


def test_cost_and_grad_are_finite():
    fwd = _forward()
    prob = _problem(fwd, jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    j, g = prob.value_and_grad(jnp.zeros(5))
    assert np.isfinite(float(j))
    assert g.shape == (5,)
    assert np.all(np.isfinite(np.asarray(g)))


def test_cost_minimised_at_truth_in_whitened_space():
    fwd = _forward()
    q_true = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    prob = _problem(fwd, q_true)
    # χ that maps exactly to q_true: χ* = L⁻¹ (q_true − S_b).
    chi_true = jax.scipy.linalg.solve_triangular(
        prob.prior_chol, q_true - prob.prior_mean, lower=True
    )
    # The observation term vanishes at the truth, so J(χ_true) ≤ J(0).
    assert float(prob.cost(chi_true)) < float(prob.cost(jnp.zeros(5)))


def test_gradient_matches_finite_difference():
    fwd = _forward(n_t=4)
    prob = _problem(fwd, jnp.array([0.0, 1.0, 0.5, 0.5]), obs_floor=1e-6)
    chi = jnp.array([0.1, -0.2, 0.05, 0.0])
    _, g = prob.value_and_grad(chi)
    # Central finite-difference check of the adjoint gradient.
    eps = 1e-3
    g_fd = np.zeros(4)
    for i in range(4):
        e = jnp.zeros(4).at[i].set(eps)
        g_fd[i] = (float(prob.cost(chi + e)) - float(prob.cost(chi - e))) / (2 * eps)
    np.testing.assert_allclose(np.asarray(g), g_fd, rtol=1e-2, atol=1e-2)


def test_whitening_roundtrip():
    fwd = _forward()
    prob = _problem(fwd, jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    s = jnp.array([0.3, 1.2, 0.8, 0.4, 0.6])
    chi = jax.scipy.linalg.solve_triangular(
        prob.prior_chol, s - prob.prior_mean, lower=True
    )
    np.testing.assert_allclose(
        np.asarray(prob.source_from_whitened(chi)), np.asarray(s), rtol=1e-5
    )


# ── end-to-end twin experiment ───────────────────────────────────────────────


def test_twin_experiment_recovers_emission_time_series():
    fwd = _forward()
    q_true = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    prob = _problem(fwd, q_true)
    res = solve_4dvar(prob, max_steps=60)
    assert isinstance(res, FourDVarResult)
    # The MAP emission signal recovers the truth that generated the obs.
    np.testing.assert_allclose(np.asarray(res.source), np.asarray(q_true), atol=0.05)
    # The optimiser made progress from the prior-mean start.
    assert res.cost < float(prob.cost(jnp.zeros(5)))


def test_solve_from_explicit_initial_source():
    fwd = _forward()
    q_true = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    prob = _problem(fwd, q_true)
    res = solve_4dvar(prob, initial_source=jnp.full(5, 0.5), max_steps=40)
    assert res.source.shape == (5,)
    assert np.all(np.isfinite(np.asarray(res.source)))


# ── validation ───────────────────────────────────────────────────────────────


def test_build_problem_shape_validation():
    fwd = _forward()
    y = fwd.predict(jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    b = matern32_covariance(fwd.save_times, variance=1.0, length_scale=20.0)
    with pytest.raises(ValueError, match="prior_covariance"):
        build_problem(
            forward=fwd,
            observations=y,
            prior_mean=jnp.zeros(5),
            prior_covariance=jnp.eye(4),  # wrong size
            obs_variance=1.0,
        )
    with pytest.raises(ValueError, match="observations"):
        build_problem(
            forward=fwd,
            observations=y[:3],  # n_t mismatch
            prior_mean=jnp.zeros(5),
            prior_covariance=b,
            obs_variance=1.0,
        )


def test_problem_is_a_frozen_dataclass():
    fwd = _forward()
    prob = _problem(fwd, jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    assert isinstance(prob, FourDVarProblem)
    with pytest.raises((AttributeError, TypeError)):
        prob.prior_mean = jnp.zeros(5)  # frozen
