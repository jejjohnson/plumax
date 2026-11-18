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

import einx
import jax
import jax.numpy as jnp
import numpy as np

from plumax._src.tracing import require_positive
from plumax.lagrangian._src.wind_interp import mean_wind
from plumax.lagrangian.particles import (
    ParticleState,
    n_steps_for_horizon,
    step_durations,
    uniform_wind,
    wind_from_speed_direction,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr

    from plumax.lagrangian._src.wind_interp import GriddedWind
    from plumax.lagrangian.particles import TurbulenceModel


def _cell_indices(coord: jax.Array, edges: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return ``(index, in_range)`` for ``coord`` against bin ``edges``."""
    num_cells = edges.shape[0] - 1
    # i = max{ j : e_j ≤ c }
    cell_index = jnp.searchsorted(edges, coord, side="right") - 1
    in_range = (cell_index >= 0) & (cell_index < num_cells)
    return jnp.clip(cell_index, 0, num_cells - 1), in_range


def bin_positions(
    positions: jax.Array,
    x_edges: jax.Array,
    y_edges: jax.Array,
    z_edges: jax.Array,
) -> jax.Array:
    """Histogram particle ``positions`` ``(n, 3)`` onto a 3-D grid (counts)."""
    num_cols = x_edges.shape[0] - 1
    num_rows = y_edges.shape[0] - 1
    num_levels = z_edges.shape[0] - 1
    col_index, col_inside = _cell_indices(positions[:, 0], x_edges)
    row_index, row_inside = _cell_indices(positions[:, 1], y_edges)
    level_index, level_inside = _cell_indices(positions[:, 2], z_edges)
    inside = col_inside & row_inside & level_inside
    # s = (i_x n_y + i_y) n_z + i_z
    flat = (col_index * num_rows + row_index) * num_levels + level_index
    counts = jnp.zeros(num_cols * num_rows * num_levels)
    counts = counts.at[flat].add(inside.astype(float))
    return counts.reshape(num_cols, num_rows, num_levels)


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
    wind: Callable[[jax.Array], jax.Array] | GriddedWind | None = None,
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
        domain_x: ``(start, stop, n_cells)`` along x.
        domain_y: ``(start, stop, n_cells)`` along y.
        domain_z: ``(start, stop, n_cells)`` along z.
        n_particles: Number of particles in the ensemble.
        t_end: Integration horizon [s].
        dt: Time step [s].
        wind: Mean-wind field ``t -> (u, v, w)``, or a
            :class:`~plumax.lagrangian.GriddedWind` sampled at each particle.
            Mutually exclusive with the ``wind_speed`` / ``wind_direction``
            shortcut.
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

    require_positive("emission_rate", emission_rate, caller="simulate_lagrangian")
    require_positive(
        "source_location[2]",
        source_location[2],
        caller="simulate_lagrangian",
        allow_zero=True,
    )

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
    # c_i = (e_i + e_{i+1}) / 2
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    # V_cell = Δx Δy Δz
    cell_volume = float(
        (x_edges[1] - x_edges[0])
        * (y_edges[1] - y_edges[0])
        * (z_edges[1] - z_edges[0])
    )

    key = jax.random.PRNGKey(seed)
    key, vkey = jax.random.split(key)

    src = jnp.asarray(source_location, dtype=float)
    initial_std, _ = turbulence.at(src[2])
    vel0 = jax.random.normal(vkey, (n_particles, 3)) * initial_std
    pos0 = jnp.broadcast_to(src, (n_particles, 3))
    state = ParticleState(position=pos0, velocity=vel0)

    x_edges_jax = jnp.asarray(x_edges)
    y_edges_jax = jnp.asarray(y_edges)
    z_edges_jax = jnp.asarray(z_edges)
    residence = _accumulate_residence(
        state,
        wind,
        turbulence,
        t_end,
        dt,
        key,
        pbl_height,
        x_edges_jax,
        y_edges_jax,
        z_edges_jax,
    )

    # C = Q τ / (N V_cell)
    conc = emission_rate * np.asarray(residence) / (n_particles * cell_volume)
    # C ← C + C_bg
    conc = conc + background_conc
    spacing_z = float(z_edges[1] - z_edges[0])
    # C_col(x, y) = Δz Σ_z C(x, y, z)
    column = einx.sum("x y [z]", conc) * spacing_z

    dataset = xr.Dataset(
        data_vars={
            "concentration": (["x", "y", "z"], conc),
            "column_concentration": (["x", "y"], column),
        },
        coords={"x": x_centers, "y": y_centers, "z": z_centers},
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
    dataset["concentration"].attrs = {
        "long_name": "Mass concentration",
        "units": "kg/m^3",
    }
    dataset["column_concentration"].attrs = {
        "long_name": "Column-integrated concentration",
        "units": "kg/m^2",
    }
    return dataset


def _accumulate_residence(
    state: ParticleState,
    wind: Callable[[jax.Array], jax.Array] | GriddedWind,
    turbulence: TurbulenceModel,
    end_time: float,
    time_step: float,
    key: jax.Array,
    pbl_height: float | None,
    x_edges: jax.Array,
    y_edges: jax.Array,
    z_edges: jax.Array,
) -> jax.Array:
    """Integrate the ensemble and accumulate per-cell residence time [s]."""
    from plumax.lagrangian.particles import langevin_step

    num_steps = n_steps_for_horizon(end_time, time_step)
    keys = jax.random.split(key, num_steps)
    # Step start times and per-step durations. When ``end_time`` is not an exact
    # multiple of ``time_step`` the final step is shortened to the remainder so
    # the ensemble is advanced — and residence is accrued — over exactly
    # ``end_time``, not ``num_steps * time_step`` (which would bias the field
    # high).
    # t_k = k Δt
    times = time_step * jnp.arange(num_steps)
    durations = step_durations(end_time, time_step, num_steps)
    num_cols = x_edges.shape[0] - 1
    num_rows = y_edges.shape[0] - 1
    num_levels = z_edges.shape[0] - 1

    def body(carry, inputs):
        particles, histogram = carry
        time, step_key, duration = inputs
        particles = langevin_step(
            particles,
            mean_wind(wind, particles.position, time),
            turbulence,
            duration,
            step_key,
            pbl_height=pbl_height,
        )
        counts = bin_positions(particles.position, x_edges, y_edges, z_edges)
        # H ← H + counts · Δt_k
        histogram = histogram + counts * duration
        return (particles, histogram), None

    initial_histogram = jnp.zeros((num_cols, num_rows, num_levels))
    (_, residence), _ = jax.lax.scan(
        body, (state, initial_histogram), (times, keys, durations)
    )
    return residence


__all__ = [
    "bin_positions",
    "simulate_lagrangian",
    "uniform_wind",
    "wind_from_speed_direction",
]
