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
    PosteriorCovariance,
    build_forward,
    build_problem,
    laplace_sample,
    posterior_covariance,
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


def test_build_forward_rejects_bad_save_times():
    common = dict(
        domain_x=(0.0, 400.0, 20),
        domain_y=(-100.0, 100.0, 10),
        domain_z=(0.0, 80.0, 4),
        source_location=(50.0, 0.0, 20.0),
        uniform_wind=(4.0, 0.0, 0.0),
        eddy_diffusivity=2.0,
    )
    with pytest.raises(ValueError, match="save_times"):
        build_forward(save_times=jnp.array([5.0]), **common)  # singleton
    with pytest.raises(ValueError, match="strictly increasing"):
        build_forward(save_times=jnp.array([0.0, 10.0, 5.0]), **common)  # non-monotone


def test_build_problem_rejects_nonpositive_obs_variance():
    fwd = _forward()
    y = fwd.predict(jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    b = matern32_covariance(fwd.save_times, variance=1.0, length_scale=20.0)
    with pytest.raises(ValueError, match="obs_variance"):
        build_problem(
            forward=fwd,
            observations=y,
            prior_mean=jnp.zeros(5),
            prior_covariance=b,
            obs_variance=0.0,
        )
    with pytest.raises(ValueError, match="obs_variance"):
        build_problem(
            forward=fwd,
            observations=y,
            prior_mean=jnp.zeros(5),
            prior_covariance=b,
            obs_variance=-1.0,
        )


def test_build_problem_per_overpass_variance_vector():
    # A length-n_t obs_variance is one variance per overpass (R_t): it must
    # broadcast across receptors (the (n_t, 1) interpretation), not align with
    # the trailing receptor axis.
    fwd = _forward()
    y = fwd.predict(jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    n_t, n_obs = y.shape
    assert n_t != n_obs  # the case where the two interpretations differ
    b = matern32_covariance(fwd.save_times, variance=1.0, length_scale=20.0)
    per_overpass = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prob = build_problem(
        forward=fwd,
        observations=y,
        prior_mean=jnp.zeros(n_t),
        prior_covariance=b,
        obs_variance=per_overpass,
    )
    assert prob.obs_variance.shape == (n_t, n_obs)
    # Row t holds R_t across all receptors.
    for t in range(n_t):
        np.testing.assert_allclose(
            np.asarray(prob.obs_variance[t]), float(per_overpass[t])
        )


def test_build_problem_rejects_ambiguous_square_variance_vector():
    # When n_t == n_obs a bare 1-D obs_variance is ambiguous (per-overpass vs
    # per-receptor); build_problem must reject it and demand an explicit shape.
    fwd = _forward()
    # Receptor subset sized to n_t makes the observation grid square (n_t, n_t).
    n_t = fwd.save_times.shape[0]
    receptors = jnp.array([[0, i] for i in range(n_t)])
    fwd_sq = build_forward(
        domain_x=(0.0, 400.0, 20),
        domain_y=(-100.0, 100.0, 10),
        domain_z=(0.0, 80.0, 4),
        save_times=fwd.save_times,
        source_location=(50.0, 0.0, 20.0),
        uniform_wind=(4.0, 0.0, 0.0),
        eddy_diffusivity=2.0,
        receptor_index=receptors,
    )
    y = fwd_sq.predict(jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    assert y.shape == (n_t, n_t)
    b = matern32_covariance(fwd_sq.save_times, variance=1.0, length_scale=20.0)
    with pytest.raises(ValueError, match="ambiguous"):
        build_problem(
            forward=fwd_sq,
            observations=y,
            prior_mean=jnp.zeros(n_t),
            prior_covariance=b,
            obs_variance=jnp.arange(1.0, n_t + 1.0),
        )
    # An explicit (n_t, 1) column is accepted as per-overpass R_t.
    col = jnp.arange(1.0, n_t + 1.0)[:, None]
    prob = build_problem(
        forward=fwd_sq,
        observations=y,
        prior_mean=jnp.zeros(n_t),
        prior_covariance=b,
        obs_variance=col,
    )
    for t in range(n_t):
        np.testing.assert_allclose(np.asarray(prob.obs_variance[t]), float(t + 1))


def test_build_problem_per_receptor_variance_vector():
    # A length-n_obs obs_variance stays per-receptor (broadcast across time).
    fwd = _forward()
    y = fwd.predict(jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    n_t, n_obs = y.shape
    b = matern32_covariance(fwd.save_times, variance=1.0, length_scale=20.0)
    per_receptor = jnp.full(n_obs, 2.5)
    prob = build_problem(
        forward=fwd,
        observations=y,
        prior_mean=jnp.zeros(n_t),
        prior_covariance=b,
        obs_variance=per_receptor,
    )
    assert prob.obs_variance.shape == (n_t, n_obs)
    np.testing.assert_allclose(np.asarray(prob.obs_variance), 2.5)


# ── posterior covariance (Gauss-Newton Laplace) ──────────────────────────────


def test_posterior_covariance_symmetric_psd_shape():
    fwd = _forward()
    q_true = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    prob = _problem(fwd, q_true)
    res = solve_4dvar(prob, max_steps=40)
    post = posterior_covariance(prob, res.whitened)
    assert isinstance(post, PosteriorCovariance)
    n_t = fwd.save_times.shape[0]
    assert post.whitened_covariance.shape == (n_t, n_t)
    assert post.source_covariance.shape == (n_t, n_t)
    for cov in (post.whitened_covariance, post.source_covariance):
        c = np.asarray(cov)
        # Symmetric and positive semidefinite (all eigenvalues ≥ 0).
        np.testing.assert_allclose(c, c.T, atol=1e-10)
        eig = np.linalg.eigvalsh(0.5 * (c + c.T))
        assert eig.min() > -1e-8


def test_posterior_source_std_positive_finite():
    fwd = _forward()
    q_true = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    prob = _problem(fwd, q_true)
    res = solve_4dvar(prob, max_steps=40)
    post = posterior_covariance(prob, res.whitened)
    std = np.asarray(post.source_std)
    assert std.shape == (fwd.save_times.shape[0],)
    assert np.all(np.isfinite(std))
    assert np.all(std > 0.0)


def test_observing_data_does_not_increase_whitened_variance():
    # The whitened prior is the identity, so the GN-Hessian update can only
    # shrink (never inflate) the marginal whitened posterior variances.
    fwd = _forward()
    q_true = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    prob = _problem(fwd, q_true)
    res = solve_4dvar(prob, max_steps=40)
    post = posterior_covariance(prob, res.whitened)
    whitened_var = np.diagonal(np.asarray(post.whitened_covariance))
    assert np.all(whitened_var <= 1.0 + 1e-6)


def test_laplace_sample_shape_and_mean():
    fwd = _forward()
    q_true = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    prob = _problem(fwd, q_true)
    res = solve_4dvar(prob, max_steps=40)
    n_t = fwd.save_times.shape[0]
    samples = laplace_sample(prob, res.whitened, jax.random.PRNGKey(0), n_samples=4000)
    assert samples.shape == (4000, n_t)
    assert np.all(np.isfinite(np.asarray(samples)))
    # The sample mean of N(S*, P_S) draws concentrates on the MAP source.
    sample_mean = np.asarray(samples).mean(axis=0)
    np.testing.assert_allclose(sample_mean, np.asarray(res.source), atol=0.05)


def test_laplace_sample_rejects_nonpositive_n_samples():
    fwd = _forward()
    prob = _problem(fwd, jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    res = solve_4dvar(prob, max_steps=20)
    with pytest.raises(ValueError, match="n_samples"):
        laplace_sample(prob, res.whitened, jax.random.PRNGKey(0), n_samples=0)


def test_posterior_covariance_matches_dense_inverse():
    # gaussx is on the path: its inverse of the GN-Hessian must agree with a
    # dense jnp.linalg.inv reference (P_χ = H_GN⁻¹, P_S = L P_χ Lᵀ).
    from plumax.les_fvm.fourdvar import _gauss_newton_hessian

    fwd = _forward(n_t=4)
    q_true = jnp.array([0.0, 1.0, 0.5, 0.5])
    prob = _problem(fwd, q_true, obs_floor=1e-6)
    res = solve_4dvar(prob, max_steps=30)
    post = posterior_covariance(prob, res.whitened)
    h_gn, _ = _gauss_newton_hessian(prob, res.whitened)
    p_chi_ref = np.linalg.inv(np.asarray(h_gn))
    np.testing.assert_allclose(
        np.asarray(post.whitened_covariance), p_chi_ref, rtol=1e-4, atol=1e-6
    )
    chol = np.asarray(prob.prior_chol)
    p_source_ref = chol @ p_chi_ref @ chol.T
    np.testing.assert_allclose(
        np.asarray(post.source_covariance), p_source_ref, rtol=1e-4, atol=1e-6
    )


def test_solve_4dvar_attaches_posterior():
    fwd = _forward()
    q_true = jnp.array([0.0, 1.0, 1.0, 0.5, 0.5])
    prob = _problem(fwd, q_true)
    res = solve_4dvar(prob, max_steps=40, compute_posterior=True)
    assert isinstance(res.posterior, PosteriorCovariance)
    n_t = fwd.save_times.shape[0]
    assert res.posterior.source_covariance.shape == (n_t, n_t)
    # Default path leaves it unset.
    res_none = solve_4dvar(prob, max_steps=5)
    assert res_none.posterior is None
