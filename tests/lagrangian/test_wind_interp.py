"""plumax's winds as velocities for particles: the C-grid stagger, and the builders."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tracerax.velocity import uniform

from plumax.lagrangian import (
    HomogeneousTurbulence,
    compute_footprint,
    gridded_wind_from_met,
    gridded_wind_from_prescribed,
    simulate_lagrangian,
    wind_from_speed_direction,
)
from plumax.les_fvm import make_grid, wind_field_from_callable
from plumax.met import MetField, to_prescribed_wind


GRID = make_grid(
    (-500.0, 500.0, 20), (-400.0, 400.0, 16), (0.0, 200.0, 8), dtype=jnp.float64
)


def _constant(value: float, n_time: int = 2) -> jax.Array:
    nz, ny, nx = GRID.interior_shape
    return jnp.full((n_time, nz, ny, nx), value)


def _met(times: jax.Array, u: jax.Array, v: jax.Array, w: jax.Array) -> MetField:
    """The meteorology of a wind on the analysis grid."""
    _, ny, nx = GRID.interior_shape
    return MetField(
        plume_grid=GRID,
        times=times,
        u=u,
        v=v,
        w=w,
        temperature=jnp.full(u.shape, 290.0),
        pressure=jnp.full(u.shape, 1.0e5),
        pbl_height=jnp.full((u.shape[0], ny, nx), 800.0),
    )


def _positions(n: int = 64, seed: int = 0) -> jax.Array:
    """Random points inside every component's hull of sample points."""
    low = jnp.array([GRID.x[0] + GRID.dx, GRID.y[0] + GRID.dy, GRID.z[0]])
    high = jnp.array([GRID.x[-1] - GRID.dx, GRID.y[-1] - GRID.dy, GRID.z[-1]])
    unit = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3))
    return low + unit * (high - low)


def _linear_wind(t, x, y, z):
    del t
    return 0.01 * x - 0.02 * y + 0.03 * z, -0.04 * x + 0.05 * z, 0.001 * y


@pytest.mark.slow
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
        np.asarray(wind(0.0, positions)), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.slow
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
    wind = gridded_wind_from_met(_met(jnp.array([0.0]), u, v, zeros))
    t_points = jnp.stack([GRID.x[2:6], GRID.y[3:7], jnp.full(4, GRID.z[1])], axis=-1)
    sampled = np.asarray(wind(0.0, t_points))[:, component]
    expected = np.asarray(t_points[:, component])
    np.testing.assert_allclose(sampled, expected, rtol=1e-12)
    # The unstaggered reading is half a cell off
    unstaggered = expected + 0.5 * (GRID.dx if component == 0 else GRID.dy)
    assert np.all(np.abs(sampled - unstaggered) >= 0.49 * GRID.dx)


@pytest.mark.slow
def test_from_met_and_from_prescribed_agree():
    times = jnp.array([0.0, 3600.0])
    nz, ny, nx = GRID.interior_shape
    key_u, key_v, key_w = jax.random.split(jax.random.PRNGKey(5), 3)
    shape = (2, nz, ny, nx)
    u = jax.random.normal(key_u, shape)
    v = jax.random.normal(key_v, shape)
    w = 0.1 * jax.random.normal(key_w, shape)
    field = _met(times, u, v, w)
    from_met = gridded_wind_from_met(field)
    from_prescribed = gridded_wind_from_prescribed(to_prescribed_wind(field), times)
    positions = _positions(seed=6)
    for t in (0.0, 900.0, 3600.0):
        np.testing.assert_allclose(
            np.asarray(from_met(t, positions)),
            np.asarray(from_prescribed(t, positions)),
            rtol=1e-12,
        )
    # At a T-point and a met time, w is the stored value itself
    np.testing.assert_allclose(
        float(from_met(3600.0, jnp.array([[GRID.x[3], GRID.y[2], GRID.z[1]]]))[0, 2]),
        float(w[1, 1, 2, 3]),
        rtol=1e-12,
    )


@pytest.mark.slow
def test_footprint_and_concentration_accept_a_gridded_wind():
    times = jnp.array([0.0, 3600.0])
    gridded = gridded_wind_from_met(
        _met(times, _constant(3.0), _constant(-1.0), _constant(0.0))
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
        wind=uniform(jnp.array([3.0, -1.0, 0.0])),
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
        wind=uniform(jnp.array([3.0, -1.0, 0.0])),
        **kwargs,
    )
    np.testing.assert_array_equal(
        gridded_run["concentration"].values, uniform_run["concentration"].values
    )


def test_wind_from_speed_direction_west_wind_flows_east():
    # 270° = wind FROM the west, i.e. blowing toward +x.
    wind = wind_from_speed_direction(5.0, 270.0)
    velocity = np.asarray(wind(0.0, jnp.zeros((2, 3))))
    np.testing.assert_allclose(velocity, [[5.0, 0.0, 0.0]] * 2, atol=1e-6)
