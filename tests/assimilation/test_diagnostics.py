"""Posterior-diagnostics tests.

Light-touch checks: each diagnostic should produce finite, sane values on a
trivial setup. The textbook identities (χ²_red ≈ 1 at the truth, DFS ≤ dim(y))
are easy to break with a sign flip or transposed index, so we test those even
on toy inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.assimilation.background import build_diagonal_background
from plumax.assimilation.control import WhiteningTransform
from plumax.assimilation.cost import build_cost_x, build_cost_xi
from plumax.assimilation.diagnostics import (
    degrees_of_freedom_for_signal,
    posterior_covariance_proxy,
    reduced_chi_squared,
)


jax.config.update("jax_enable_x64", True)


def test_chi2_at_truth_is_zero(obs_model_no_optics):
    model = obs_model_no_optics
    truth = jnp.full((3, 3), 1e-7)
    y = model.forward(truth, linear=False)
    chi2 = reduced_chi_squared(
        forward_fn=model.make_forward(linear=False),
        estimated_state=truth,
        observation=y,
        obs_inv_variance=1e8,
    )
    assert chi2 < 1e-10


def test_dfs_zero_in_no_information_limit(obs_model_no_optics):
    """When the prior dominates (obs infinitely noisy), Hess → B⁻¹ and DFS → 0.

    This is the key regression test: the old implementation computed
    ``trace(B · Hess)`` which returned ``state_size`` instead of ``0`` in
    this limit, so the sanity check was systematically biased high.
    """
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1.0, n_pixels=ny * nx)  # well-scaled prior
    W = WhiteningTransform.from_background(B)
    y = model.forward(jnp.zeros((ny, nx)), linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1e-12,  # effectively no obs info
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    dfs = degrees_of_freedom_for_signal(
        hessian_vector_product=lambda v: cost.hvp(jnp.zeros(ny * nx), v),
        state_size=ny * nx,
        whitened=True,  # cost is build_cost_xi → ξ-space Hessian, prior = I
        n_probes=8,
    )
    assert np.isfinite(dfs)
    # No observations → posterior == prior → DFS ≈ 0. Hutchinson variance
    # plus CG tolerance give ~5% of state_size as a comfortable bound.
    assert abs(dfs) < 0.5


def test_dfs_approaches_state_size_in_high_information_limit(obs_model_no_optics):
    """When obs dominate (prior very loose), Hess → H'ᵀR⁻¹H' and DFS → state_size."""
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1.0, n_pixels=ny * nx)  # well-scaled prior
    W = WhiteningTransform.from_background(B)
    truth = jnp.full((ny, nx), 1e-7)
    y = model.forward(truth, linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1e12,  # tight observations
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    dfs = degrees_of_freedom_for_signal(
        hessian_vector_product=lambda v: cost.hvp(jnp.zeros(ny * nx), v),
        state_size=ny * nx,
        whitened=True,  # cost is build_cost_xi → ξ-space Hessian, prior = I
        n_probes=16,
    )
    # Should be close to state_size = 9; allow Hutchinson + CG slack.
    assert 7.0 < dfs < ny * nx + 0.5


def test_dfs_nonidentity_background(obs_model_no_optics):
    """DFS agrees between the model-space and whitened paths for B ≠ I.

    The old code combined a ξ-space Hessian with a model-space B⁻¹ solve, which
    is only correct when B = I. Here B is strongly heteroscedastic, so the two
    paths must be computed in their own spaces (``whitened`` flag) to agree —
    and both must match an exact dense reference.
    """
    model = obs_model_no_optics
    ny, nx = 3, 3
    n = ny * nx
    # Wide, non-uniform prior variances → B ≠ I in a way that the space-mixing
    # bug cannot hide behind.
    variances = np.array([0.05, 0.1, 0.3, 0.7, 1.0, 1.5, 3.0, 6.0, 12.0])
    B = build_diagonal_background(variances)
    W = WhiteningTransform.from_background(B)
    fwd = model.make_forward(linear=False)
    y = model.forward(jnp.full((ny, nx), 1e-7), linear=False)
    obs_inv = 1e-11  # partial-information regime (chosen so 0 < DFS < n)

    kw = dict(
        forward_fn=fwd,
        obs_inv_variance=obs_inv,
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    cost_x = build_cost_x(background_op=B, **kw)
    cost_xi = build_cost_xi(whitening=W, **kw)

    dfs_x = degrees_of_freedom_for_signal(
        hessian_vector_product=lambda v: cost_x.hvp(jnp.zeros(n), v),
        state_size=n,
        background_op=B,
        whitened=False,
        n_probes=64,
        seed=0,
    )
    dfs_xi = degrees_of_freedom_for_signal(
        hessian_vector_product=lambda v: cost_xi.hvp(jnp.zeros(n), v),
        state_size=n,
        whitened=True,
        n_probes=64,
        seed=0,
    )

    # Exact reference: dense ξ-space Hessian, DFS = n − trace(Hess_ξ⁻¹).
    basis = np.eye(n)
    hess_xi = np.stack(
        [np.asarray(cost_xi.hvp(jnp.zeros(n), jnp.asarray(col))) for col in basis],
        axis=1,
    )
    hess_xi_inv = np.linalg.inv(hess_xi)
    exact = n - np.trace(hess_xi_inv)

    # The setup must be genuinely partial for the test to exercise the bug.
    assert 0.5 < exact < n - 0.5
    np.testing.assert_allclose(dfs_x, exact, atol=0.6)
    np.testing.assert_allclose(dfs_xi, exact, atol=0.6)
    np.testing.assert_allclose(dfs_x, dfs_xi, atol=0.8)

    # Non-vacuous: the old ξ-hvp + B⁻¹ mix gives a materially different number.
    buggy = n - np.trace(hess_xi_inv @ np.diag(1.0 / variances))
    assert abs(exact - buggy) > 0.5


def test_dfs_rejects_inconsistent_background_args(obs_model_no_optics):
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1.0, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    y = model.forward(jnp.zeros((ny, nx)), linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1.0,
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    hvp = lambda v: cost.hvp(jnp.zeros(ny * nx), v)
    with pytest.raises(ValueError, match=r"whitened=True.*do not pass `background_op`"):
        degrees_of_freedom_for_signal(
            hessian_vector_product=hvp,
            state_size=ny * nx,
            background_op=B,
            whitened=True,
            n_probes=2,
        )
    with pytest.raises(ValueError, match=r"model-space mode.*requires"):
        degrees_of_freedom_for_signal(
            hessian_vector_product=hvp,
            state_size=ny * nx,
            whitened=False,
            n_probes=2,
        )


def test_posterior_covariance_proxy_whitened_matches_model_space(obs_model_no_optics):
    """Bₐ = U Hess_ξ⁻¹ Uᵀ: the whitened proxy reproduces the model-space one."""
    model = obs_model_no_optics
    ny, nx = 3, 3
    n = ny * nx
    variances = np.array([0.05, 0.2, 0.5, 0.8, 1.0, 1.3, 2.0, 4.0, 9.0])
    B = build_diagonal_background(variances)
    W = WhiteningTransform.from_background(B)
    fwd = model.make_forward(linear=False)
    y = model.forward(jnp.full((ny, nx), 1e-7), linear=False)
    kw = dict(
        forward_fn=fwd,
        obs_inv_variance=1e-11,
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    cost_x = build_cost_x(background_op=B, **kw)
    cost_xi = build_cost_xi(whitening=W, **kw)

    proxy_x = posterior_covariance_proxy(
        hessian_vector_product=lambda v: cost_x.hvp(jnp.zeros(n), v),
        state_size=n,
        cg_max_steps=500,
    )
    proxy_xi = posterior_covariance_proxy(
        hessian_vector_product=lambda v: cost_xi.hvp(jnp.zeros(n), v),
        state_size=n,
        whitening=W,
        cg_max_steps=500,
    )
    v = jnp.asarray(np.linspace(-1.0, 1.0, n))
    np.testing.assert_allclose(
        np.asarray(proxy_xi(v)), np.asarray(proxy_x(v)), rtol=1e-3, atol=1e-8
    )


def test_posterior_covariance_proxy_runs(obs_model_no_optics):
    model = obs_model_no_optics
    ny, nx = 3, 3
    B = build_diagonal_background(1e-12, n_pixels=ny * nx)
    W = WhiteningTransform.from_background(B)
    y = model.forward(jnp.zeros((ny, nx)), linear=False)
    cost = build_cost_xi(
        forward_fn=model.make_forward(linear=False),
        whitening=W,
        obs_inv_variance=1.0,
        background_state=jnp.zeros((ny, nx)),
        observation=y,
        state_shape=(ny, nx),
    )
    # The HVP is whitened (build_cost_xi), so pass `whitening` to get the
    # model-space covariance Bₐ = U Hess_ξ⁻¹ Uᵀ rather than the raw ξ-space one.
    matvec = posterior_covariance_proxy(
        hessian_vector_product=lambda v: cost.hvp(jnp.zeros(ny * nx), v),
        state_size=ny * nx,
        whitening=W,
        cg_max_steps=50,
    )
    Bv = matvec(jnp.ones(ny * nx))
    assert Bv.shape == (ny * nx,)
    assert np.all(np.isfinite(np.asarray(Bv)))
