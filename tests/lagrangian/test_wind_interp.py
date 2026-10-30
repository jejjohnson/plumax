"""The C-grid-aware gridded wind for particles (#88)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.lagrangian import (
    GriddedWind,
    HomogeneousTurbulence,
    ParticleState,
    compute_footprint,
    gridded_wind_from_met,
    gridded_wind_from_prescribed,
    integrate_particles,
    simulate_lagrangian,
    uniform_wind,
)
from plumax.les_fvm import make_grid, wind_field_from_callable
from plumax.met import MetField, to_prescribed_wind


GRID = make_grid(
    (-500.0, 500.0, 20), (-400.0, 400.0, 16), (0.0, 200.0, 8), dtype=jnp.float64
)
STILL = HomogeneousTurbulence(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)


def _constant(value: float, n_time: int = 2) -> jax.Array:
    nz, ny, nx = GRID.interior_shape
    return jnp.full((n_time, nz, ny, nx), value)


def _positions(n: int = 64, seed: int = 0) -> jax.Array:
    """Random points inside every component's hull of sample points."""
    low = jnp.array([GRID.x[0] + GRID.dx, GRID.y[0] + GRID.dy, GRID.z[0]])
    high = jnp.array([GRID.x[-1] - GRID.dx, GRID.y[-1] - GRID.dy, GRID.z[-1]])
    unit = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3))
    return low + unit * (high - low)


def _linear_wind(t, x, y, z):
    del t
    return 0.01 * x - 0.02 * y + 0.03 * z, -0.04 * x + 0.05 * z, 0.001 * y


def test_uniform_field_no_stagger_offset():
    times = jnp.array([0.0, 3600.0])
    wind = GriddedWind.from_grid(
        GRID, times, _constant(3.0), _constant(-1.5), _constant(0.25)
    )
    # Inside, outside the grid and outside the time range alike
    positions = jnp.concatenate(
        [_positions(), jnp.array([[-900.0, 900.0, -5.0], [900.0, -900.0, 900.0]])]
    )
    reference = uniform_wind(3.0, -1.5, 0.25)
    for t in (-10.0, 0.0, 1234.5, 3600.0, 9999.0):
        sampled = np.asarray(wind(positions, t))
        expected = np.broadcast_to(np.asarray(reference(t)), sampled.shape)
        np.testing.assert_array_equal(sampled, expected)


def test_uniform_gridded_wind_integrates_like_uniform_wind():
    times = jnp.array([0.0, 3600.0])
    gridded = GriddedWind.from_grid(
        GRID, times, _constant(3.0), _constant(-1.5), _constant(0.0)
    )
    turbulence = HomogeneousTurbulence(0.5, 0.5, 0.2, 20.0, 20.0, 10.0)
    state = ParticleState(
        position=jnp.broadcast_to(jnp.array([0.0, 0.0, 20.0]), (32, 3)),
        velocity=jnp.zeros((32, 3)),
    )
    key = jax.random.PRNGKey(3)
    kwargs = {"t0": 0.0, "t1": 60.0, "dt": 2.0, "key": key, "pbl_height": 150.0}
    final_gridded, _ = integrate_particles(state, gridded, turbulence, **kwargs)
    final_uniform, _ = integrate_particles(
        state, uniform_wind(3.0, -1.5), turbulence, **kwargs
    )
    np.testing.assert_array_equal(
        np.asarray(final_gridded.position), np.asarray(final_uniform.position)
    )


def test_linear_field_is_interpolated_exactly():
    wind_field = wind_field_from_callable(
        GRID, lambda t, x, y, z: _linear_wind(t, x, y, z)
    )
    wind = gridded_wind_from_prescribed(wind_field, jnp.array([0.0]))
    positions = _positions()
    expected = np.stack(
        [np.asarray(c) for c in _linear_wind(0.0, *positions.T)], axis=-1
    )
    np.testing.assert_allclose(
        np.asarray(wind(positions, 0.0)), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("component", [0, 1])
def test_staggering_catches_a_half_cell_shift(component):
    # u = x on U-points and v = y on V-points. Read at a T-point they give the
    # T-point coordinate; read as if they sat on T-points they would be off
    # by half a cell, Δx/2 = 25 m and Δy/2 = 25 m here.
    nz, ny, nx = GRID.interior_shape
    x_u = GRID.x + 0.5 * GRID.dx
    y_v = GRID.y + 0.5 * GRID.dy
    u = jnp.broadcast_to(x_u, (1, nz, ny, nx))
    v = jnp.broadcast_to(y_v[:, None], (1, nz, ny, nx))
    zeros = jnp.zeros((1, nz, ny, nx))
    wind = GriddedWind.from_grid(GRID, jnp.array([0.0]), u, v, zeros)
    t_points = jnp.stack([GRID.x[2:6], GRID.y[3:7], jnp.full(4, GRID.z[1])], axis=-1)
    sampled = np.asarray(wind(t_points, 0.0))[:, component]
    expected = np.asarray(t_points[:, component])
    np.testing.assert_allclose(sampled, expected, rtol=1e-12)
    # The unstaggered reading is half a cell off
    unstaggered = expected + 0.5 * (GRID.dx if component == 0 else GRID.dy)
    assert np.all(np.abs(sampled - unstaggered) >= 0.49 * GRID.dx)


def test_piecewise_linear_in_time_and_held_outside():
    times = jnp.array([0.0, 100.0, 300.0])
    u = jnp.stack([_constant(1.0, 1)[0], _constant(3.0, 1)[0], _constant(-1.0, 1)[0]])
    wind = GriddedWind.from_grid(GRID, times, u, u, u)
    point = jnp.array([[0.0, 0.0, 50.0]])
    for t, expected in ((-50.0, 1.0), (50.0, 2.0), (200.0, 1.0), (500.0, -1.0)):
        np.testing.assert_allclose(np.asarray(wind(point, t))[0, 0], expected)


def test_edge_value_is_held_beyond_the_walls():
    times = jnp.array([0.0])
    nz, ny, nx = GRID.interior_shape
    u = jnp.broadcast_to(jnp.arange(nx, dtype=float), (1, nz, ny, nx))
    wind = GriddedWind.from_grid(GRID, times, u, u, u)
    east = jnp.array([[2000.0, 0.0, 50.0]])
    west = jnp.array([[-2000.0, 0.0, 50.0]])
    assert float(wind(east, 0.0)[0, 0]) == nx - 1
    assert float(wind(west, 0.0)[0, 0]) == 0.0


def test_streamlines_solid_body_rotation():
    # u = −ω y, v = ω x is linear, so the gridded wind is the analytic wind and
    # the zero-turbulence trajectory is the integrator's own: forward Euler,
    # whose one-step error against the circle is O(Δt²).
    omega = 2.0 * jnp.pi / 3600.0
    wind_field = wind_field_from_callable(
        GRID, lambda t, x, y, z: (-omega * y, omega * x, 0.0 * z)
    )
    gridded = gridded_wind_from_prescribed(wind_field, jnp.array([0.0]))
    start = jnp.array([[150.0, 0.0, 50.0], [0.0, -100.0, 50.0]])
    state = ParticleState(position=start, velocity=jnp.zeros_like(start))

    def step_error(dt: float) -> float:
        final, _ = integrate_particles(
            state, gridded, STILL, t0=0.0, t1=dt, dt=dt, key=jax.random.PRNGKey(0)
        )
        angle = omega * dt
        rotation = jnp.array(
            [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
        )
        exact = start[:, :2] @ rotation.T
        return float(jnp.max(jnp.abs(final.position[:, :2] - exact)))

    errors = [step_error(dt) for dt in (40.0, 20.0, 10.0)]
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    np.testing.assert_allclose(ratios, 4.0, rtol=0.05)

    # Over a quarter turn the particles stay on their circle to Euler's
    # amplification (1 + (ωΔt)²)^(N/2) and match the analytic wind's run.
    dt, n_steps = 10.0, 90
    final, _ = integrate_particles(
        state,
        gridded,
        STILL,
        t0=0.0,
        t1=dt * n_steps,
        dt=dt,
        key=jax.random.PRNGKey(0),
    )
    radius = jnp.linalg.norm(final.position[:, :2], axis=-1)
    growth = (1.0 + (omega * dt) ** 2) ** (n_steps / 2)
    np.testing.assert_allclose(
        np.asarray(radius), np.asarray(jnp.array([150.0, 100.0]) * growth), rtol=1e-5
    )
    angle = jnp.arctan2(final.position[:, 1], final.position[:, 0])
    np.testing.assert_allclose(np.asarray(angle), [jnp.pi / 2, 0.0], atol=0.02)


def test_from_met_and_from_prescribed_agree():
    times = jnp.array([0.0, 3600.0])
    nz, ny, nx = GRID.interior_shape
    key_u, key_v, key_w = jax.random.split(jax.random.PRNGKey(5), 3)
    shape = (2, nz, ny, nx)
    u = jax.random.normal(key_u, shape)
    v = jax.random.normal(key_v, shape)
    w = 0.1 * jax.random.normal(key_w, shape)
    field = MetField(
        plume_grid=GRID,
        times=times,
        u=u,
        v=v,
        w=w,
        temperature=jnp.full(shape, 290.0),
        pressure=jnp.full(shape, 1.0e5),
        pbl_height=jnp.full((2, ny, nx), 800.0),
    )
    from_met = gridded_wind_from_met(field)
    from_prescribed = gridded_wind_from_prescribed(to_prescribed_wind(field), times)
    positions = _positions(seed=6)
    for t in (0.0, 900.0, 3600.0):
        np.testing.assert_allclose(
            np.asarray(from_met(positions, t)),
            np.asarray(from_prescribed(positions, t)),
            rtol=1e-12,
        )
    # At a T-point and a met time, w is the stored value itself
    np.testing.assert_allclose(
        float(from_met(jnp.array([[GRID.x[3], GRID.y[2], GRID.z[1]]]), 3600.0)[0, 2]),
        float(w[1, 1, 2, 3]),
        rtol=1e-12,
    )


def test_jit_and_vmap_agree_with_a_loop():
    wind_field = wind_field_from_callable(GRID, _linear_wind)
    wind = gridded_wind_from_prescribed(wind_field, jnp.array([0.0, 60.0]))
    batches = jnp.stack([_positions(n=8, seed=s) for s in range(3)])
    looped = np.stack([np.asarray(wind(p, 10.0)) for p in batches])
    sampled = jax.jit(jax.vmap(lambda p: wind(p, 10.0)))(batches)
    np.testing.assert_allclose(np.asarray(sampled), looped, rtol=1e-12)


def test_position_gradient_matches_central_differences():
    wind_field = wind_field_from_callable(GRID, _linear_wind)
    wind = gridded_wind_from_prescribed(wind_field, jnp.array([0.0]))
    point = jnp.array([12.3, -45.6, 78.9])

    def sample(p):
        return wind(p[None], 0.0)[0]

    jacobian = np.asarray(jax.jacrev(sample)(point))
    step = 1e-3
    finite = np.stack(
        [
            (
                np.asarray(sample(point + step * e))
                - np.asarray(sample(point - step * e))
            )
            / (2 * step)
            for e in jnp.eye(3)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(jacobian, finite, rtol=1e-6, atol=1e-9)
    # The linear field's own gradient
    np.testing.assert_allclose(
        jacobian,
        [[0.01, -0.02, 0.03], [-0.04, 0.0, 0.05], [0.0, 0.001, 0.0]],
        atol=1e-12,
    )


def test_footprint_and_concentration_accept_a_gridded_wind():
    times = jnp.array([0.0, 3600.0])
    gridded = GriddedWind.from_grid(
        GRID, times, _constant(3.0), _constant(-1.0), _constant(0.0)
    )
    turbulence = HomogeneousTurbulence(0.5, 0.5, 0.3, 20.0, 20.0, 10.0)
    common = {
        "n_particles": 200,
        "t_back": 60.0,
        "dt": 5.0,
        "pbl_height": 150.0,
    }
    footprint_gridded, _, _ = compute_footprint(
        (100.0, 0.0, 10.0),
        turbulence,
        (-400.0, 400.0, 8),
        (-200.0, 200.0, 4),
        wind=gridded,
        **common,
    )
    footprint_uniform, _, _ = compute_footprint(
        (100.0, 0.0, 10.0),
        turbulence,
        (-400.0, 400.0, 8),
        (-200.0, 200.0, 4),
        wind=uniform_wind(3.0, -1.0),
        **common,
    )
    np.testing.assert_array_equal(footprint_gridded, footprint_uniform)

    domain = ((-400.0, 400.0, 8), (-200.0, 200.0, 4), (0.0, 100.0, 4))
    kwargs = {"n_particles": 200, "t_end": 60.0, "dt": 5.0}
    gridded_run = simulate_lagrangian(
        1.0, (0.0, 0.0, 10.0), turbulence, *domain, wind=gridded, **kwargs
    )
    uniform_run = simulate_lagrangian(
        1.0,
        (0.0, 0.0, 10.0),
        turbulence,
        *domain,
        wind=uniform_wind(3.0, -1.0),
        **kwargs,
    )
    np.testing.assert_array_equal(
        gridded_run["concentration"].values, uniform_run["concentration"].values
    )


def test_rejects_a_single_point_axis():
    one = jnp.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="at least 2"):
        GriddedWind(
            jnp.arange(2.0), jnp.arange(2.0), jnp.zeros(1), jnp.zeros(1), one, one, one
        )
