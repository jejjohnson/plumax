"""Tier III 4D-Var as a vardax analysis (#163).

``solve_4dvar`` / ``posterior_covariance`` now run ``vardax.ThreeDVar`` over
the whitened source–receptor map. The reference below is the algorithm they
replaced: L-BFGS on the whitened cost through ``optx.minimise`` and the dense
Gauss–Newton Hessian ``I + J̃ᵀ R⁻¹ J̃`` by ``jax.jacrev``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

from plumax.lagrangian.inversion import matern32_covariance
from plumax.les_fvm.fourdvar import build_forward, build_problem, solve_4dvar


pytest.importorskip("vardax")


@pytest.fixture(scope="module")
def problem():
    save_times = jnp.linspace(0.0, 40.0, 5)
    forward = build_forward(
        domain_x=(0.0, 400.0, 20),
        domain_y=(-100.0, 100.0, 10),
        domain_z=(0.0, 80.0, 4),
        save_times=save_times,
        source_location=(50.0, 0.0, 20.0),
        uniform_wind=(4.0, 0.0, 0.0),
        eddy_diffusivity=2.0,
        advection_scheme="upwind1",
    )
    y = forward.predict(jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    return build_problem(
        forward=forward,
        observations=y,
        prior_mean=jnp.full(5, 0.5),
        prior_covariance=matern32_covariance(
            save_times, variance=1.0, length_scale=20.0
        ),
        obs_variance=(float(y.max()) * 1e-3) ** 2 + 1e-12,
    )


def _reference(problem, max_steps):
    """The pre-vardax solve and Laplace standard deviations."""
    solution = optx.minimise(
        lambda chi, _: problem.cost(chi),
        optx.LBFGS(rtol=1e-6, atol=1e-8),
        jnp.zeros(5),
        max_steps=max_steps,
        throw=False,
    )
    chi = solution.value

    def predict_whitened(c):
        return problem.forward.predict(problem.source_from_whitened(c)).reshape(-1)

    jac = jax.jacrev(predict_whitened)(chi)
    r_inv = (1.0 / problem.obs_variance).reshape(-1)
    p_chi = jnp.linalg.inv(jnp.eye(5) + jac.T @ (r_inv[:, None] * jac))
    L = problem.prior_chol
    std = jnp.sqrt(jnp.diag(L @ p_chi @ L.T))
    return problem.source_from_whitened(chi), std, int(solution.stats["num_steps"])


@pytest.mark.slow
def test_map_and_laplace_match(problem):
    source_ref, std_ref, steps_ref = _reference(problem, max_steps=500)
    result = solve_4dvar(problem, max_steps=500, compute_posterior=True)
    np.testing.assert_allclose(result.source, source_ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(result.posterior.source_std, std_ref, rtol=1e-3)
    # same optimiser on the same whitened cost: same iteration count
    assert result.n_steps == steps_ref
    assert result.converged
