"""The per-cell surface flux control of the Tier III 4D-Var inversion (#91)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax._src.errors import PayloadError
from plumax.lagrangian.inversion import matern32_covariance
from plumax.les_fvm import (
    SpatialControl,
    TemporalControl,
    build_forward,
    build_problem,
    laplace_sample,
    solve_4dvar,
)


pytest.importorskip("vardax")

NY, NX = 4, 5
SPACING = 20.0
Y = (jnp.arange(NY) + 0.5) * SPACING
X = (jnp.arange(NX) + 0.5) * SPACING
SAVE_TIMES = jnp.array([0.0, 20.0, 40.0])
STD = 2e-3 * (1.0 + 0.1 * jnp.arange(NY * NX).reshape(NY, NX))


def _matern(distance: jax.Array, length_scale: float) -> jax.Array:
    scaled = jnp.sqrt(3.0) * distance / length_scale
    return (1.0 + scaled) * jnp.exp(-scaled)


def _control(**overrides) -> SpatialControl:
    options = {
        "mask": jnp.ones((NY, NX), dtype=bool),
        "prior_mean": 1e-3 * jnp.ones((NY, NX)),
        "length_scale_m": 30.0,
        "prior_std": STD,
    }
    return SpatialControl(**{**options, **overrides})


def _forward(control: SpatialControl, wind=(2.0, 0.0, 0.0), diffusivity=1.0):
    return build_forward(
        domain_x=(0.0, NX * SPACING, NX),
        domain_y=(0.0, NY * SPACING, NY),
        domain_z=(0.0, 40.0, 4),
        save_times=SAVE_TIMES,
        uniform_wind=wind,
        control=control,
        eddy_diffusivity=diffusivity,
    )


@functools.cache
def _toy() -> tuple[SpatialControl, object]:
    control = _control()
    return control, _forward(control)


def test_kronecker_matches_dense():
    # the Kronecker build is the product kernel, evaluated here cell by cell
    control = _control()
    covariance = control.prior_covariance(Y, X).as_matrix()
    grid_y, grid_x = jnp.meshgrid(Y, X, indexing="ij")
    flat_y, flat_x = grid_y.ravel(), grid_x.ravel()
    product = _matern(jnp.abs(flat_y[:, None] - flat_y[None, :]), 30.0) * _matern(
        jnp.abs(flat_x[:, None] - flat_x[None, :]), 30.0
    )
    std = STD.ravel()
    dense = std[:, None] * product * std[None, :]
    assert np.max(np.abs(np.asarray(covariance - dense))) < 1e-8 * float(std.max()) ** 2


def test_isotropic_control_is_the_dense_matern_not_the_product_kernel():
    separable = _control(prior_std=jnp.ones((NY, NX))).prior_covariance(Y, X)
    isotropic = _control(prior_std=jnp.ones((NY, NX)), separable=False)
    grid_y, grid_x = jnp.meshgrid(Y, X, indexing="ij")
    coordinates = jnp.stack([grid_y.ravel(), grid_x.ravel()], axis=-1)
    dense = matern32_covariance(coordinates, variance=1.0, length_scale=30.0)
    np.testing.assert_allclose(
        isotropic.prior_covariance(Y, X).as_matrix(), dense, atol=1e-12
    )
    # r = √(d_y² + d_x²) is no sum of per-axis terms, so the kernels differ
    gap = jnp.max(jnp.abs(separable.as_matrix() - dense))
    assert float(gap) > 0.01


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"separable": False},
        {
            "mask": jnp.ones((NY, NX), dtype=bool)
            .at[0, 1]
            .set(False)
            .at[2, 3]
            .set(False)
        },
        {"time_basis": jnp.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])},
    ],
    ids=["kronecker", "isotropic", "masked", "time-basis"],
)
def test_prior_sqrt_factorises_the_covariance(overrides):
    control = _control(**overrides)
    root = control.prior_sqrt(Y, X).as_matrix()
    covariance = control.prior_covariance(Y, X).as_matrix()
    assert root.shape == (control.size, control.size)
    scale = float(STD.max()) ** 2
    np.testing.assert_allclose(root @ root.T / scale, covariance / scale, atol=1e-8)
    # lower triangular, so the whitening solves of the posterior hold
    np.testing.assert_array_equal(np.triu(np.asarray(root), 1), 0.0)


def test_mask_restricts_the_covariance_to_the_active_cells():
    mask = jnp.ones((NY, NX), dtype=bool).at[1, 2].set(False)
    masked = _control(mask=mask).prior_covariance(Y, X).as_matrix()
    full = _control().prior_covariance(Y, X).as_matrix()
    active = np.flatnonzero(np.asarray(mask))
    np.testing.assert_allclose(masked, full[np.ix_(active, active)], atol=1e-14)


def test_time_basis_gives_the_kronecker_flux_covariance_and_the_mean():
    # Φ = L_t makes the flux covariance B_t ⊗ B_s at the save times
    temporal = matern32_covariance(SAVE_TIMES, variance=1.0, length_scale=30.0)
    basis = jnp.linalg.cholesky(temporal)
    control = _control(time_basis=basis)
    covariance = control.prior_covariance(Y, X).as_matrix()
    spatial = _control().prior_covariance(Y, X).as_matrix()
    # the flat flux at the save times is (Φ ⊗ I) q
    to_flux = jnp.kron(basis, jnp.eye(NY * NX))
    flux_covariance = to_flux @ covariance @ to_flux.T
    scale = float(STD.max()) ** 2
    np.testing.assert_allclose(
        flux_covariance / scale, jnp.kron(temporal, spatial) / scale, atol=1e-10
    )
    # and the prior mean of the flux is m(s) at every save time
    mean_flux = to_flux @ control.flat_prior_mean()
    np.testing.assert_allclose(mean_flux, 1e-3, rtol=1e-10)


def test_basis_that_misses_the_constant_is_rejected():
    with pytest.raises(ValueError, match="span the constant"):
        _control(time_basis=jnp.array([[0.0], [1.0], [2.0]]))


@pytest.mark.parametrize(
    "overrides",
    [
        {"mask": jnp.zeros((NY, NX), dtype=bool)},
        {"prior_mean": jnp.ones((2, 2))},
        {"prior_std": jnp.zeros((NY, NX))},
        {"prior_std": jnp.full((NY, NX), jnp.nan)},
        {"length_scale_m": 0.0},
        {"time_basis": jnp.ones(3)},
    ],
)
def test_invalid_control_is_rejected(overrides):
    with pytest.raises(ValueError):
        _control(**overrides)


def test_forward_needs_exactly_one_kind_of_source():
    with pytest.raises(ValueError, match="exactly one"):
        build_forward(
            domain_x=(0.0, 100.0, 5),
            domain_y=(0.0, 80.0, 4),
            domain_z=(0.0, 40.0, 4),
            save_times=SAVE_TIMES,
            uniform_wind=(1.0, 0.0, 0.0),
        )


def test_forward_rejects_a_control_on_another_grid():
    with pytest.raises(ValueError, match=r"\(ny, nx\)"):
        build_forward(
            domain_x=(0.0, 120.0, 6),
            domain_y=(0.0, 80.0, 4),
            domain_z=(0.0, 40.0, 4),
            save_times=SAVE_TIMES,
            uniform_wind=(1.0, 0.0, 0.0),
            control=_control(),
        )


def test_to_field_fills_inactive_cells_with_zero_and_keeps_batches():
    mask = jnp.array([[True, False], [False, True]])
    control = SpatialControl(
        mask=mask,
        prior_mean=jnp.zeros((2, 2)),
        length_scale_m=10.0,
        prior_std=jnp.ones((2, 2)),
        time_basis=jnp.ones((3, 2)),
    )
    flat = jnp.arange(1.0, 9.0).reshape(2, 4)
    fields = control.to_field(flat)
    assert fields.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(fields[1, 1], jnp.array([[7.0, 0.0], [0.0, 8.0]]))


@pytest.mark.slow
def test_adjoint_identity():
    # the forward is linear in the flux, so F u = 𝒢(u) − 𝒢(0)
    control, forward = _toy()
    u = jax.random.normal(jax.random.PRNGKey(0), (control.size,))
    zero = jnp.zeros(control.size)
    baseline, adjoint = jax.vjp(forward.predict, zero)
    v = jax.random.normal(jax.random.PRNGKey(1), baseline.shape)
    (adjoint_v,) = adjoint(v)
    forward_pairing = jnp.vdot(forward.predict(u) - baseline, v)
    adjoint_pairing = jnp.vdot(u, adjoint_v)
    np.testing.assert_allclose(forward_pairing, adjoint_pairing, rtol=1e-12)


def _still_air_forward(control: SpatialControl):
    # no horizontal transport, closed top and bottom: every column keeps its flux
    return _forward(control, wind=(0.0, 0.0, 0.0), diffusivity=(0.0, 1.0))


@pytest.mark.slow
def test_column_mass_integrates_the_flux_in_still_air():
    # c_col(s, t) = q(s) t, in kg m⁻², whatever the vertical mixing
    control = _control()
    flux = 1e-3 * (1.0 + jnp.arange(NY * NX, dtype=float))
    columns = _still_air_forward(control).predict(flux)
    expected = jnp.outer(SAVE_TIMES, flux)
    np.testing.assert_allclose(columns, expected, rtol=1e-10, atol=1e-13)


@pytest.mark.slow
def test_column_mass_integrates_a_time_varying_flux():
    # q(s, t) = q₀(s) + q₁(s) t / 40, so c_col(s, t) = q₀ t + q₁ t² / 80
    basis = jnp.stack([jnp.ones(3), SAVE_TIMES / 40.0], axis=-1)
    control = _control(time_basis=basis)
    first = 1e-3 * jnp.ones(NY * NX)
    second = 2e-3 * jnp.arange(NY * NX, dtype=float)
    columns = _still_air_forward(control).predict(jnp.concatenate([first, second]))
    expected = jnp.outer(SAVE_TIMES, first) + jnp.outer(SAVE_TIMES**2 / 80.0, second)
    np.testing.assert_allclose(columns, expected, rtol=1e-10, atol=1e-13)


@pytest.mark.slow
def test_map_and_posterior_match_the_dense_linear_gaussian_solution():
    control, forward = _toy()
    truth = control.flat_prior_mean() + 1e-3 * jnp.arange(control.size) / control.size
    clean = forward.predict(truth)
    noise_std = 0.02 * float(clean.max())
    noise = noise_std * jax.random.normal(jax.random.PRNGKey(2), clean.shape)
    problem = build_problem(
        forward=forward,
        observations=clean + noise,
        obs_variance=noise_std**2,
        control=control,
    )
    result = solve_4dvar(problem, max_steps=500, compute_posterior=True)

    # dense reference: μ + B Fᵀ (F B Fᵀ + R)⁻¹ (y − F μ)
    flat_predict = lambda q: forward.predict(q).ravel()
    jacobian = jax.jacrev(flat_predict)(jnp.zeros(control.size))
    covariance = control.prior_covariance(Y, X).as_matrix()
    mean = control.flat_prior_mean()
    innovation = (clean + noise).ravel() - jacobian @ mean
    obs_covariance = jacobian @ covariance @ jacobian.T + noise_std**2 * jnp.eye(
        jacobian.shape[0]
    )
    gain = jnp.linalg.solve(obs_covariance, jacobian @ covariance).T
    dense_map = mean + gain @ innovation
    dense_posterior = covariance - gain @ jacobian @ covariance

    assert result.posterior is not None
    np.testing.assert_allclose(result.source, dense_map, rtol=1e-3, atol=1e-5)
    scale = float(jnp.max(jnp.abs(dense_posterior)))
    np.testing.assert_allclose(
        result.posterior.source_covariance / scale, dense_posterior / scale, atol=1e-8
    )
    # the fields are the flat vectors on the grid
    np.testing.assert_array_equal(result.source_field[0].ravel(), result.source)
    np.testing.assert_array_equal(
        result.posterior.std_field[0].ravel(), result.posterior.source_std
    )
    draws = laplace_sample(problem, result.whitened, jax.random.PRNGKey(3), 4)
    assert draws.shape == (4, 1, NY, NX)


@pytest.mark.slow
def test_two_source_recovery():
    ny, nx, spacing = 20, 20, 20.0
    y = (jnp.arange(ny) + 0.5) * spacing
    x = (jnp.arange(nx) + 0.5) * spacing
    grid_y, grid_x = jnp.meshgrid(y, x, indexing="ij")
    centres = ((120.0, 100.0), (280.0, 180.0))

    def blob(centre: tuple[float, float], amplitude: float) -> jax.Array:
        distance_sq = (grid_y - centre[0]) ** 2 + (grid_x - centre[1]) ** 2
        return amplitude * jnp.exp(-distance_sq / (2.0 * 30.0**2))

    truth = blob(centres[0], 3e-3) + blob(centres[1], 2e-3)
    control = SpatialControl(
        mask=jnp.ones((ny, nx), dtype=bool),
        prior_mean=jnp.zeros((ny, nx)),
        length_scale_m=60.0,
        prior_std=2e-3 * jnp.ones((ny, nx)),
    )
    rows, columns = jnp.meshgrid(
        jnp.arange(0, ny, 2), jnp.arange(0, nx, 2), indexing="ij"
    )
    forward = build_forward(
        domain_x=(0.0, nx * spacing, nx),
        domain_y=(0.0, ny * spacing, ny),
        domain_z=(0.0, 80.0, 4),
        save_times=jnp.array([0.0, 30.0, 60.0, 90.0]),
        uniform_wind=(2.0, 0.5, 0.0),
        control=control,
        eddy_diffusivity=2.0,
        receptor_index=jnp.stack([rows.ravel(), columns.ravel()], axis=-1),
    )
    clean = forward.predict(truth.ravel())
    noise_std = 0.02 * float(clean.max())
    noise = noise_std * jax.random.normal(jax.random.PRNGKey(0), clean.shape)
    problem = build_problem(
        forward=forward,
        observations=clean + noise,
        obs_variance=noise_std**2,
        control=control,
    )
    result = solve_4dvar(problem, max_steps=300, compute_posterior=True)
    assert result.posterior is not None
    field = result.source_field[0]
    covariance = result.posterior.source_covariance
    # each source's flux over a 60 m disc lies within its own Laplace σ
    for centre in centres:
        distance_sq = (grid_y - centre[0]) ** 2 + (grid_x - centre[1]) ** 2
        disc = jnp.where(distance_sq < 60.0**2, 1.0, 0.0).ravel()
        recovered = jnp.vdot(disc, field.ravel())
        expected = jnp.vdot(disc, truth.ravel())
        sigma = jnp.sqrt(disc @ covariance @ disc)
        assert abs(float(recovered - expected)) < float(sigma)
    # the larger source is the field's peak, on the true cell
    peak = jnp.unravel_index(jnp.argmax(field), field.shape)
    assert (int(peak[0]), int(peak[1])) == (6, 5)


def test_temporal_control_is_the_prior_arguments():
    times = jnp.linspace(0.0, 40.0, 5)
    forward = build_forward(
        domain_x=(0.0, 400.0, 20),
        domain_y=(-100.0, 100.0, 10),
        domain_z=(0.0, 80.0, 4),
        save_times=times,
        source_location=(50.0, 0.0, 20.0),
        uniform_wind=(4.0, 0.0, 0.0),
        eddy_diffusivity=2.0,
    )
    mean = jnp.ones(5)
    covariance = matern32_covariance(times, variance=0.25, length_scale=20.0)
    observations = jnp.zeros((5, 3))
    by_arguments = build_problem(
        forward=forward,
        observations=observations,
        prior_mean=mean,
        prior_covariance=covariance,
        obs_variance=1.0,
    )
    by_control = build_problem(
        forward=forward,
        observations=observations,
        obs_variance=1.0,
        control=TemporalControl(mean, covariance),
    )
    np.testing.assert_array_equal(by_control.prior_chol, by_arguments.prior_chol)
    np.testing.assert_array_equal(by_control.prior_mean, by_arguments.prior_mean)
    assert by_control.control is None


def test_build_problem_rejects_mixed_or_mismatched_controls():
    control, forward = _toy()
    observations = jnp.zeros((3, NY * NX))
    with pytest.raises(ValueError, match="same SpatialControl"):
        build_problem(
            forward=forward,
            observations=observations,
            obs_variance=1.0,
            control=_control(),
        )
    with pytest.raises(ValueError, match="not both"):
        build_problem(
            forward=forward,
            observations=observations,
            obs_variance=1.0,
            control=control,
            prior_mean=jnp.zeros(control.size),
        )
    with pytest.raises(ValueError, match="spatial control"):
        build_problem(
            forward=forward,
            observations=observations,
            obs_variance=1.0,
            prior_mean=jnp.zeros(control.size),
            prior_covariance=jnp.eye(control.size),
        )


@pytest.mark.slow
def test_per_cell_result_refuses_the_time_mean_payload():
    control, forward = _toy()
    problem = build_problem(
        forward=forward,
        observations=forward.predict(control.flat_prior_mean()),
        obs_variance=1.0,
        control=control,
    )
    result = solve_4dvar(problem, max_steps=1)
    assert result.source_field.shape == (1, NY, NX)
    with pytest.raises(PayloadError):
        _ = result.emission_rate
