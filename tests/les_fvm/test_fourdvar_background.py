"""Initial concentration and lateral boundary scaling in the 4D-Var control (#92)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from finitevolx import Dirichlet1D, Outflow1D

from plumax._src.errors import PayloadError
from plumax.lagrangian.inversion import matern32_covariance
from plumax.les_fvm import (
    BackgroundControl,
    TemporalControl,
    build_default_concentration_bc,
    build_forward,
    build_problem,
    laplace_sample,
    solve_4dvar,
)
from plumax.les_fvm.boundary import scale_dirichlet_faces


pytest.importorskip("vardax")

INFLOW = 1e-3
NZ, NY, NX = 4, 4, 6
SAVE_TIMES = jnp.array([0.0, 15.0, 30.0])


def _forward(
    wind=(4.0, 0.0, 0.0), diffusivity=1.0, inflow=INFLOW, save_times=SAVE_TIMES
):
    return build_forward(
        domain_x=(0.0, 120.0, NX),
        domain_y=(0.0, 80.0, NY),
        domain_z=(0.0, 80.0, NZ),
        save_times=save_times,
        source_location=(40.0, 40.0, 20.0),
        uniform_wind=wind,
        eddy_diffusivity=diffusivity,
        bc_x=(("dirichlet", inflow), "outflow"),
        advection_scheme="upwind1",
    )


def _background(std: float = 1e-5, bc_prior=(1.0, 0.1)) -> BackgroundControl:
    n_cells = NZ * NY * NX
    return BackgroundControl(
        jnp.zeros((NZ, NY, NX)),
        lx.DiagonalLinearOperator(jnp.full(n_cells, std**2)),
        bc_prior,
    )


def _prior(n_t: int = 3) -> TemporalControl:
    times = jnp.linspace(0.0, 30.0, n_t)
    return TemporalControl(
        jnp.full(n_t, 0.5), matern32_covariance(times, variance=0.04, length_scale=20.0)
    )


def _flat(
    source: jax.Array,
    increment: jax.Array | None = None,
    bc_scale: jax.Array | None = None,
) -> jax.Array:
    if increment is None:
        increment = jnp.zeros(NZ * NY * NX)
    if bc_scale is None:
        bc_scale = jnp.ones(4)
    return jnp.concatenate([source, increment, bc_scale])


def test_scale_acts_on_dirichlet_faces_only():
    horizontal, _ = build_default_concentration_bc(
        bc_x=(("dirichlet", 2.0), "outflow"), bc_y=(("dirichlet", 3.0), "outflow")
    )
    # (west, east, south, north)
    scaled = scale_dirichlet_faces(horizontal, jnp.array([1.5, 7.0, 0.5, 7.0]))
    bc_set = scaled.bc_set
    assert isinstance(bc_set.west, Dirichlet1D)
    assert float(bc_set.west.value) == 3.0
    assert float(bc_set.south.value) == 1.5
    assert isinstance(bc_set.east, Outflow1D)
    assert isinstance(bc_set.north, Outflow1D)


def test_none_is_identity():
    forward = _forward()
    observations = jnp.zeros((3, NY * NX))
    prior = _prior()
    plain = build_problem(
        forward=forward, observations=observations, obs_variance=1.0, control=prior
    )
    explicit = build_problem(
        forward=forward,
        observations=observations,
        obs_variance=1.0,
        control=prior,
        background=None,
    )
    assert explicit.forward is forward
    np.testing.assert_array_equal(explicit.prior_mean, plain.prior_mean)
    np.testing.assert_array_equal(explicit.prior_chol, plain.prior_chol)
    assert plain.n_source == 3


@pytest.mark.slow
def test_background_at_its_prior_mean_reproduces_the_pinned_forward():
    # c_b = 0, δc₀ = 0 and α = 1 are exactly the forward without a background
    forward = _forward()
    source = jnp.array([0.5, 0.7, 0.4])
    with_background = forward.with_background(_background())
    np.testing.assert_array_equal(
        with_background.predict(_flat(source)), forward.predict(source)
    )


@pytest.mark.slow
def test_column_keeps_the_initial_concentration_in_still_air():
    # no horizontal transport: c_col(s, t) = Σ_z (c_b + δc₀) Δz + q(s, t), source off
    forward = _forward(wind=(0.0, 0.0, 0.0), diffusivity=(0.0, 1.0), inflow=0.0)
    key_b, key_d = jax.random.split(jax.random.PRNGKey(0))
    c_b = 1e-3 * jnp.abs(jax.random.normal(key_b, (NZ, NY, NX)))
    increment = 1e-4 * jax.random.normal(key_d, (NZ * NY * NX,))
    background = BackgroundControl(
        c_b, lx.DiagonalLinearOperator(jnp.ones(NZ * NY * NX))
    )
    columns = forward.with_background(background).predict(
        _flat(jnp.zeros(3), increment)
    )
    initial = c_b + increment.reshape(NZ, NY, NX)
    expected = jnp.sum(initial, axis=0).ravel() * 20.0
    np.testing.assert_allclose(columns, jnp.tile(expected, (3, 1)), rtol=1e-10)


@pytest.mark.slow
def test_inflow_scale_sets_the_flushed_concentration():
    # pure advection, no source: after the flush every column is α c_in Σ_z Δz
    times = jnp.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    forward = _forward(diffusivity=0.0, save_times=times)
    alpha = 1.3
    columns = forward.with_background(_background()).predict(
        _flat(jnp.zeros(7), bc_scale=jnp.array([alpha, 1.0, 1.0, 1.0]))
    )
    np.testing.assert_allclose(columns[-1], alpha * INFLOW * 80.0, rtol=1e-3)


@pytest.mark.slow
def test_adjoint_identity_over_the_whole_control():
    # the forward is affine in (S, δc₀, α), so F u = 𝒢(u) − 𝒢(0)
    forward = _forward().with_background(_background())
    size = 3 + NZ * NY * NX + 4
    u = jax.random.normal(jax.random.PRNGKey(1), (size,))
    baseline, adjoint = jax.vjp(forward.predict, jnp.zeros(size))
    v = jax.random.normal(jax.random.PRNGKey(2), baseline.shape)
    (adjoint_v,) = adjoint(v)
    forward_pairing = jnp.vdot(forward.predict(u) - baseline, v)
    np.testing.assert_allclose(forward_pairing, jnp.vdot(u, adjoint_v), rtol=1e-11)


def test_prior_is_block_diagonal_with_the_means_stacked():
    forward = _forward()
    prior = _prior()
    background = _background(std=1e-5, bc_prior=(1.1, 0.2))
    problem = build_problem(
        forward=forward,
        observations=jnp.zeros((3, NY * NX)),
        obs_variance=1.0,
        control=prior,
        background=background,
    )
    covariance = problem.prior_chol @ problem.prior_chol.T
    n_cells = NZ * NY * NX
    expected = jnp.zeros((3 + n_cells + 4, 3 + n_cells + 4))
    expected = expected.at[:3, :3].set(prior.prior_covariance)
    expected = expected.at[3:-4, 3:-4].set(1e-10 * jnp.eye(n_cells))
    expected = expected.at[-4:, -4:].set(0.04 * jnp.eye(4))
    np.testing.assert_allclose(covariance, expected, atol=1e-8, rtol=1e-8)
    np.testing.assert_array_equal(np.triu(np.asarray(problem.prior_chol), 1), 0.0)
    mean = jnp.concatenate([prior.prior_mean, jnp.zeros(n_cells), jnp.full(4, 1.1)])
    np.testing.assert_array_equal(problem.prior_mean, mean)
    assert problem.n_source == 3


@pytest.mark.parametrize(
    ("c_b", "covariance", "bc_prior"),
    [
        (jnp.zeros((2, 2)), jnp.ones(4), (1.0, 0.1)),
        (jnp.full((2, 2, 2), jnp.nan), jnp.ones(8), (1.0, 0.1)),
        (jnp.zeros((2, 2, 2)), jnp.ones(3), (1.0, 0.1)),
        (jnp.zeros((2, 2, 2)), jnp.ones(8), (1.0, 0.0)),
    ],
)
def test_invalid_background_is_rejected(c_b, covariance, bc_prior):
    with pytest.raises(ValueError):
        BackgroundControl(c_b, lx.DiagonalLinearOperator(covariance), bc_prior)


def test_background_on_another_grid_is_rejected():
    background = BackgroundControl(
        jnp.zeros((2, 2, 2)), lx.DiagonalLinearOperator(jnp.ones(8))
    )
    with pytest.raises(ValueError, match=r"background\.c_b"):
        build_problem(
            forward=_forward(),
            observations=jnp.zeros((3, NY * NX)),
            obs_variance=1.0,
            control=_prior(),
            background=background,
        )


@pytest.mark.slow
def test_map_and_posterior_match_the_dense_linear_gaussian_solution():
    forward = _forward()
    background = _background(std=1e-4)
    flat_forward = forward.with_background(background)
    truth = _flat(jnp.array([0.4, 0.7, 0.5]), bc_scale=jnp.array([1.1, 1.0, 1.0, 1.0]))
    clean = flat_forward.predict(truth)
    noise_std = 0.01 * float(clean.max())
    noise = noise_std * jax.random.normal(jax.random.PRNGKey(3), clean.shape)
    problem = build_problem(
        forward=forward,
        observations=clean + noise,
        obs_variance=noise_std**2,
        control=_prior(),
        background=background,
    )
    result = solve_4dvar(
        problem, max_steps=500, rtol=1e-8, atol=1e-10, compute_posterior=True
    )

    # dense reference: μ + B Jᵀ (J B Jᵀ + R)⁻¹ (y − 𝒢(μ))
    mean = problem.prior_mean
    jacobian = jax.jacrev(lambda q: flat_forward.predict(q).ravel())(mean)
    covariance = problem.prior_chol @ problem.prior_chol.T
    obs_covariance = jacobian @ covariance @ jacobian.T + noise_std**2 * jnp.eye(
        jacobian.shape[0]
    )
    gain = jnp.linalg.solve(obs_covariance, jacobian @ covariance).T
    innovation = (clean + noise - flat_forward.predict(mean)).ravel()
    dense_map = mean + gain @ innovation
    dense_posterior = covariance - gain @ jacobian @ covariance

    assert result.background is not None
    assert result.posterior is not None
    np.testing.assert_allclose(result.source, dense_map[:3], rtol=1e-3)
    np.testing.assert_allclose(result.background.bc_scale, dense_map[-4:], rtol=1e-3)
    posterior = result.posterior
    np.testing.assert_allclose(
        posterior.source_covariance, dense_posterior[:3, :3], rtol=1e-6, atol=1e-10
    )
    np.testing.assert_allclose(
        posterior.bc_scale_std, jnp.sqrt(jnp.diag(dense_posterior)[-4:]), rtol=1e-6
    )
    # the payload stays about the source block
    assert result.prior_mean.shape == (3,)
    assert result.prior_covariance.shape == (3, 3)
    assert posterior.source_std.shape == (3,)
    draws = laplace_sample(problem, result.whitened, jax.random.PRNGKey(4), 5)
    assert draws.shape == (5, 3)


@pytest.mark.slow
def test_bc_scale_std_needs_a_background():
    forward = _forward()
    problem = build_problem(
        forward=forward,
        observations=forward.predict(jnp.full(3, 0.5)),
        obs_variance=1.0,
        control=_prior(),
    )
    result = solve_4dvar(problem, max_steps=1, compute_posterior=True)
    assert result.background is None
    assert result.posterior is not None
    with pytest.raises(PayloadError):
        _ = result.posterior.bc_scale_std


@pytest.mark.slow
def test_inflow_bias_absorbed_by_bc_scale():
    # noise-free observations, so the outcome does not depend on a noise draw;
    # the error variance is 1 % of the plume, not of the inflow-dominated column,
    # so the source is well observed and the prior's pull does not hide the bias
    n_x, n_y = 12, 6
    times = jnp.array([0.0, 20.0, 40.0, 60.0, 80.0])
    forward = build_forward(
        domain_x=(0.0, 240.0, n_x),
        domain_y=(0.0, 120.0, n_y),
        domain_z=(0.0, 80.0, 4),
        save_times=times,
        source_location=(60.0, 60.0, 20.0),
        uniform_wind=(4.0, 0.0, 0.0),
        eddy_diffusivity=2.0,
        bc_x=(("dirichlet", INFLOW), "outflow"),
        advection_scheme="upwind1",
    )
    n_cells = 4 * n_y * n_x
    background = BackgroundControl(
        jnp.zeros((4, n_y, n_x)),
        lx.DiagonalLinearOperator(jnp.full(n_cells, (1e-5) ** 2)),
    )
    truth = jnp.array([0.4, 0.6, 0.7, 0.5, 0.4])
    biased = jnp.concatenate(
        [truth, jnp.zeros(n_cells), jnp.array([1.2, 1.0, 1.0, 1.0])]
    )
    flat_forward = forward.with_background(background)
    observations = flat_forward.predict(biased)
    plume = flat_forward.predict(biased.at[-4:].set(0.0))
    obs_std = 0.01 * float(plume.max())
    prior = TemporalControl(
        jnp.full(5, 0.5), matern32_covariance(times, variance=0.04, length_scale=40.0)
    )
    unscaled = solve_4dvar(
        build_problem(
            forward=forward,
            observations=observations,
            obs_variance=obs_std**2,
            control=prior,
        ),
        max_steps=300,
        compute_posterior=True,
    )
    scaled = solve_4dvar(
        build_problem(
            forward=forward,
            observations=observations,
            obs_variance=obs_std**2,
            control=prior,
            background=background,
        ),
        max_steps=1000,
        compute_posterior=True,
    )
    assert unscaled.posterior is not None
    assert scaled.posterior is not None
    assert scaled.background is not None
    # without the scaling the source absorbs the inflow: every sample is far off
    unscaled_error = jnp.abs(unscaled.source - truth) / unscaled.posterior.source_std
    assert float(unscaled_error.min()) > 3.0
    # with it the MAP is within σ, and the inflow factor is found
    scaled_error = jnp.abs(scaled.source - truth) / scaled.posterior.source_std
    assert float(scaled_error.max()) < 1.0
    bc_scale = scaled.background.bc_scale
    assert abs(float(bc_scale[0]) - 1.2) < float(scaled.posterior.bc_scale_std[0])
    assert abs(float(bc_scale[0]) - 1.2) < 0.01
    # the outflow and periodic faces see no inflow, so they keep their prior
    np.testing.assert_allclose(bc_scale[1:], 1.0, atol=1e-6)
