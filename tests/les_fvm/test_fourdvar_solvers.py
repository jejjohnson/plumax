"""Gauss–Newton 4D-Var on vardax's IncrementalFourDVar (#93)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from plumax.lagrangian.inversion import matern32_covariance
from plumax.les_fvm import (
    BackgroundControl,
    SpatialControl,
    build_forward,
    build_problem,
    solve_4dvar,
)


pytest.importorskip("vardax")


def _relative(estimate: jax.Array, reference: jax.Array) -> float:
    return float(jnp.linalg.norm(estimate - reference) / jnp.linalg.norm(reference))


def _time_series_problem(n_t: int = 3):
    save_times = jnp.linspace(0.0, 30.0, n_t)
    forward = build_forward(
        domain_x=(0.0, 120.0, 6),
        domain_y=(0.0, 80.0, 4),
        domain_z=(0.0, 80.0, 4),
        save_times=save_times,
        source_location=(40.0, 40.0, 20.0),
        uniform_wind=(4.0, 0.0, 0.0),
        eddy_diffusivity=1.0,
        advection_scheme="upwind1",
    )
    observations = forward.predict(jnp.linspace(0.3, 0.9, n_t))
    return build_problem(
        forward=forward,
        observations=observations,
        prior_mean=jnp.full(n_t, 0.5),
        prior_covariance=matern32_covariance(
            save_times, variance=0.25, length_scale=20.0
        ),
        obs_variance=(1e-2 * float(observations.max())) ** 2,
    )


def _dense_solution(problem) -> tuple[jax.Array, jax.Array]:
    """MAP and covariance of the flat control, from the dense Jacobian."""
    predict = problem.forward.predict
    mean = problem.prior_mean
    covariance = problem.prior_chol @ problem.prior_chol.T
    obs_variance = problem.obs_variance.ravel()

    def flat(control: jax.Array) -> jax.Array:
        return predict(control).ravel()

    jacobian = jax.jacrev(flat)(mean)
    # μ + B Jᵀ (J B Jᵀ + R)⁻¹ (y − 𝒢(μ))
    obs_covariance = jacobian @ covariance @ jacobian.T + jnp.diag(obs_variance)
    gain = jnp.linalg.solve(obs_covariance, jacobian @ covariance).T
    dense_map = mean + gain @ (problem.observations.ravel() - flat(mean))
    return dense_map, covariance - gain @ jacobian @ covariance


@pytest.mark.slow
def test_methods_agree_linear():
    # the forward is affine, so the linearised cost is the cost itself: one outer
    # iteration of Gauss–Newton must reach the dense linear Gaussian MAP
    problem = _time_series_problem()
    dense_map, dense_covariance = _dense_solution(problem)
    lbfgs = solve_4dvar(problem, rtol=1e-10, atol=1e-12, max_steps=500)
    newton = solve_4dvar(
        problem,
        method="gauss_newton",
        rtol=1e-10,
        atol=1e-12,
        outer_iters=1,
        compute_posterior=True,
    )
    assert _relative(newton.source, lbfgs.source) < 1e-6
    assert _relative(newton.source, dense_map) < 1e-6
    assert newton.n_steps == 1
    assert newton.converged
    # the result carries the same whitened control and cost as the direct formulas
    np.testing.assert_allclose(
        problem.source_from_whitened(newton.whitened), newton.source, rtol=1e-12
    )
    np.testing.assert_allclose(
        newton.cost, float(problem.cost(newton.whitened)), rtol=1e-12
    )
    assert newton.posterior is not None
    np.testing.assert_allclose(
        newton.posterior.source_covariance, dense_covariance, rtol=1e-6, atol=1e-12
    )


@pytest.mark.slow
def test_gauss_newton_outer_iters():
    # the twin problem of tests/les_fvm: same MAP as L-BFGS within 5 outer steps
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
    observations = forward.predict(jnp.array([0.0, 1.0, 1.0, 0.5, 0.5]))
    problem = build_problem(
        forward=forward,
        observations=observations,
        prior_mean=jnp.full(5, 0.5),
        prior_covariance=matern32_covariance(
            save_times, variance=1.0, length_scale=20.0
        ),
        obs_variance=(float(observations.max()) * 1e-3) ** 2 + 1e-12,
    )
    lbfgs = solve_4dvar(problem, rtol=1e-10, atol=1e-12, max_steps=500)
    newton = solve_4dvar(
        problem, method="gauss_newton", rtol=1e-10, atol=1e-12, outer_iters=5
    )
    assert newton.n_steps <= 5
    assert _relative(newton.source, lbfgs.source) < 1e-6


@pytest.mark.slow
def test_gauss_newton_on_a_kronecker_prior_with_a_block_diagonal_background():
    # gaussx.Kronecker inside gaussx.BlockDiag as the vardax square root
    ny, nx, nz = 4, 5, 4
    control = SpatialControl(
        mask=jnp.ones((ny, nx), dtype=bool),
        prior_mean=1e-3 * jnp.ones((ny, nx)),
        length_scale_m=30.0,
        prior_std=2e-3 * jnp.ones((ny, nx)),
    )
    forward = build_forward(
        domain_x=(0.0, 100.0, nx),
        domain_y=(0.0, 80.0, ny),
        domain_z=(0.0, 40.0, nz),
        save_times=jnp.array([0.0, 20.0, 40.0]),
        uniform_wind=(2.0, 0.0, 0.0),
        control=control,
        eddy_diffusivity=1.0,
        bc_x=(("dirichlet", 1e-3), "outflow"),
    )
    n_cells = nz * ny * nx
    background = BackgroundControl(
        jnp.zeros((nz, ny, nx)),
        lx.DiagonalLinearOperator(jnp.full(n_cells, (1e-4) ** 2)),
    )
    truth = jnp.concatenate(
        [
            control.flat_prior_mean() * 1.5,
            jnp.zeros(n_cells),
            jnp.array([1.1, 1.0, 1.0, 1.0]),
        ]
    )
    observations = forward.with_background(background).predict(truth)
    problem = build_problem(
        forward=forward,
        observations=observations,
        obs_variance=(1e-2 * float(observations.max())) ** 2,
        control=control,
        background=background,
    )
    dense_map, _ = _dense_solution(problem)
    result = solve_4dvar(
        problem,
        method="gauss_newton",
        rtol=1e-10,
        atol=1e-12,
        outer_iters=2,
        inner_iters=200,
    )
    assert result.background is not None
    assert _relative(result.source, dense_map[: problem.n_source]) < 1e-6
    assert _relative(result.background.bc_scale, dense_map[-4:]) < 1e-6
    assert result.source_field.shape == (1, ny, nx)


def _unsolved_problem():
    save_times = jnp.linspace(0.0, 20.0, 2)
    forward = build_forward(
        domain_x=(0.0, 120.0, 6),
        domain_y=(0.0, 80.0, 4),
        domain_z=(0.0, 80.0, 4),
        save_times=save_times,
        source_location=(40.0, 40.0, 20.0),
        uniform_wind=(1.0, 0.0, 0.0),
    )
    return build_problem(
        forward=forward,
        observations=jnp.zeros((2, 24)),
        prior_mean=jnp.zeros(2),
        prior_covariance=jnp.eye(2),
        obs_variance=1.0,
    )


def test_psas_names_the_vardax_issue_it_waits_on():
    with pytest.raises(ValueError, match="vardax/issues/72"):
        solve_4dvar(_unsolved_problem(), method="psas")


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        solve_4dvar(_unsolved_problem(), method="newton")


@pytest.mark.parametrize(
    "options", [{"initial_source": jnp.zeros(2)}, {"solver": object()}]
)
def test_gauss_newton_rejects_options_it_cannot_honour(options):
    with pytest.raises(ValueError, match="L-BFGS only"):
        solve_4dvar(_unsolved_problem(), method="gauss_newton", **options)
