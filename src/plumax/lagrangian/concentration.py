"""Forward Lagrangian concentration field via residence-time binning.

For a continuous point source of strength ``Q`` [kg/s] in a statistically
steady flow, the steady concentration equals the time integral of the field
produced by an instantaneous release. We exploit that equivalence: release
``N`` particles at the source, integrate them, and accumulate the time each
particle spends in every grid cell. The steady concentration is then

    C(cell) = Q · (Σ_particles residence_time_in_cell) / (N · V_cell)     [kg/m³].

Releasing all particles at ``t = 0`` (each carrying the full source) keeps the
integration a single ``lax.scan`` with no inactive-particle masking; the ``1/N``
factor performs the ensemble average.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from plumax.lagrangian.particles import (
    ParticleState,
    uniform_wind,
    wind_from_speed_direction,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr

    from plumax.lagrangian.particles import TurbulenceModel


def _cell_indices(coord: jax.Array, edges: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return ``(index, in_range)`` for ``coord`` against bin ``edges``."""
    n = edges.shape[0] - 1
    idx = jnp.searchsorted(edges, coord, side="right") - 1
    in_range = (idx >= 0) & (idx < n)
    return jnp.clip(idx, 0, n - 1), in_range


def bin_positions(
    positions: jax.Array,
    x_edges: jax.Array,
    y_edges: jax.Array,
    z_edges: jax.Array,
) -> jax.Array:
    """Histogram particle ``positions`` ``(n, 3)`` onto a 3-D grid (counts)."""
    nx = x_edges.shape[0] - 1
    ny = y_edges.shape[0] - 1
    nz = z_edges.shape[0] - 1
    ix, gx = _cell_indices(positions[:, 0], x_edges)
    iy, gy = _cell_indices(positions[:, 1], y_edges)
    iz, gz = _cell_indices(positions[:, 2], z_edges)
    inside = gx & gy & gz
    flat = (ix * ny + iy) * nz + iz
    counts = jnp.zeros(nx * ny * nz).at[flat].add(inside.astype(float))
    return counts.reshape(nx, ny, nz)


def simulate_lagrangian(
    emission_rate: float,
    source_location: tuple[float, float, float],
    turbulence: TurbulenceModel,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    *,
    n_particles: int = 5000,
    t_end: float = 600.0,
    dt: float = 1.0,
    wind: Callable[[jax.Array], jax.Array] | None = None,
    wind_speed: float | None = None,
    wind_direction: float | None = None,
    pbl_height: float | None = None,
    background_conc: float = 0.0,
    seed: int = 0,
) -> xr.Dataset:
    """Simulate a steady Lagrangian concentration field on a 3-D grid.

    Args:
        emission_rate: Continuous source strength ``Q`` [kg/s], ``> 0``.
        source_location: ``(x, y, z)`` source coordinates [m], ``z ≥ 0``.
        turbulence: Turbulence model (e.g.
            :class:`~plumax.lagrangian.turbulence.HomogeneousTurbulence`).
        domain_x / domain_y / domain_z: ``(start, stop, n_cells)`` per axis.
        n_particles: Number of particles in the ensemble.
        t_end: Integration horizon [s].
        dt: Time step [s].
        wind: Mean-wind field ``t -> (u, v, w)``. Mutually exclusive with the
            ``wind_speed`` / ``wind_direction`` shortcut.
        wind_speed: Wind speed [m/s] (with ``wind_direction``) if ``wind`` is
            not given.
        wind_direction: Meteorological direction [deg from North].
        pbl_height: Optional reflecting PBL lid [m].
        background_conc: Additive background concentration [kg/m³].
        seed: PRNG seed.

    Returns:
        An :class:`xarray.Dataset` with ``concentration`` (x, y, z) [kg/m³] and
        ``column_concentration`` (x, y) [kg/m²].

    Raises:
        ValueError: on a non-positive emission rate, a negative source height,
            or an ambiguous / missing wind specification.
    """
    import xarray as xr

    if not (emission_rate > 0.0):
        raise ValueError("simulate_lagrangian: `emission_rate` must be > 0")
    if source_location[2] < 0.0:
        raise ValueError("simulate_lagrangian: source height must be ≥ 0")

    if wind is None:
        if wind_speed is None or wind_direction is None:
            raise ValueError(
                "simulate_lagrangian: provide either `wind` or both "
                "`wind_speed` and `wind_direction`."
            )
        wind = wind_from_speed_direction(wind_speed, wind_direction)
    elif wind_speed is not None or wind_direction is not None:
        raise ValueError(
            "simulate_lagrangian: pass `wind` or the speed/direction pair, not both."
        )

    x_edges = np.linspace(domain_x[0], domain_x[1], int(domain_x[2]) + 1)
    y_edges = np.linspace(domain_y[0], domain_y[1], int(domain_y[2]) + 1)
    z_edges = np.linspace(domain_z[0], domain_z[1], int(domain_z[2]) + 1)
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_c = 0.5 * (z_edges[:-1] + z_edges[1:])
    cell_volume = float(
        (x_edges[1] - x_edges[0])
        * (y_edges[1] - y_edges[0])
        * (z_edges[1] - z_edges[0])
    )

    key = jax.random.PRNGKey(seed)
    key, vkey = jax.random.split(key)

    src = jnp.asarray(source_location, dtype=float)
    sigma0, _ = turbulence.at(src[2])
    vel0 = jax.random.normal(vkey, (n_particles, 3)) * sigma0
    pos0 = jnp.broadcast_to(src, (n_particles, 3))
    state = ParticleState(position=pos0, velocity=vel0)

    xe = jnp.asarray(x_edges)
    ye = jnp.asarray(y_edges)
    ze = jnp.asarray(z_edges)
    residence = _accumulate_residence(
        state, wind, turbulence, t_end, dt, key, pbl_height, xe, ye, ze
    )

    conc = emission_rate * np.asarray(residence) / (n_particles * cell_volume)
    conc = conc + background_conc
    dz = float(z_edges[1] - z_edges[0])
    column = conc.sum(axis=2) * dz

    ds = xr.Dataset(
        data_vars={
            "concentration": (["x", "y", "z"], conc),
            "column_concentration": (["x", "y"], column),
        },
        coords={"x": x_c, "y": y_c, "z": z_c},
        attrs={
            "title": "Lagrangian particle dispersion (steady field, JAX)",
            "model": "markov-1-langevin",
            "emission_rate": emission_rate,
            "emission_rate_units": "kg/s",
            "n_particles": n_particles,
            "t_end": t_end,
            "dt": dt,
            "background_concentration": background_conc,
        },
    )
    ds["concentration"].attrs = {"long_name": "Mass concentration", "units": "kg/m^3"}
    ds["column_concentration"].attrs = {
        "long_name": "Column-integrated concentration",
        "units": "kg/m^2",
    }
    return ds


def _accumulate_residence(
    state: ParticleState,
    wind: Callable[[jax.Array], jax.Array],
    turbulence: TurbulenceModel,
    t_end: float,
    dt: float,
    key: jax.Array,
    pbl_height: float | None,
    x_edges: jax.Array,
    y_edges: jax.Array,
    z_edges: jax.Array,
) -> jax.Array:
    """Integrate the ensemble and accumulate per-cell residence time [s]."""
    from plumax.lagrangian.particles import langevin_step

    n_steps = max(int(np.ceil(t_end / dt)), 0)
    keys = jax.random.split(key, n_steps)
    # Step start times and per-step durations. When ``t_end`` is not an exact
    # multiple of ``dt`` the final step is shortened to the remainder so the
    # ensemble is advanced — and residence is accrued — over exactly ``t_end``,
    # not ``n_steps * dt`` (which would bias the field high).
    times = dt * jnp.arange(n_steps)
    dts = _step_durations(t_end, dt, n_steps)
    nx = x_edges.shape[0] - 1
    ny = y_edges.shape[0] - 1
    nz = z_edges.shape[0] - 1

    def body(carry, inputs):
        st, hist = carry
        t, k, dt_i = inputs
        st = langevin_step(
            st, jnp.asarray(wind(t)), turbulence, dt_i, k, pbl_height=pbl_height
        )
        hist = hist + bin_positions(st.position, x_edges, y_edges, z_edges) * dt_i
        return (st, hist), None

    hist0 = jnp.zeros((nx, ny, nz))
    (_, residence), _ = jax.lax.scan(body, (state, hist0), (times, keys, dts))
    return residence


def _step_durations(horizon: float, dt: float, n_steps: int) -> jax.Array:
    """Per-step durations summing to ``horizon``; the final step takes the rest.

    All steps are ``dt`` except the last, which is the remainder
    ``horizon - (n_steps - 1) * dt`` (equal to ``dt`` when ``horizon`` is an
    exact multiple). Returns an empty array when ``n_steps == 0``.
    """
    if n_steps == 0:
        return jnp.zeros((0,))
    dts = np.full(n_steps, dt, dtype=float)
    dts[-1] = horizon - dt * (n_steps - 1)
    return jnp.asarray(dts)


__all__ = [
    "bin_positions",
    "simulate_lagrangian",
    "uniform_wind",
    "wind_from_speed_direction",
]
