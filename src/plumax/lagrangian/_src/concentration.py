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
from tracerax.advect import Langevin, Walls, integrate
from tracerax.bin import Cells, Residence, column, steady_concentration
from tracerax.particles import at_point, with_turbulent_velocity

from plumax._src.dataset import label_concentrations
from plumax._src.domain import cell_centres, edge_axis
from plumax._src.tracing import require_positive
from plumax.lagrangian._src.wind_interp import wind_from_speed_direction


if TYPE_CHECKING:
    import xarray as xr
    from tracerax.advect import TurbulenceModel
    from tracerax.velocity import Velocity


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
    wind: Velocity | None = None,
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
        wind: Mean wind, a tracerax velocity ``wind(t, position)``: uniform, or
            gridded and sampled at each particle. Mutually exclusive with the
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
        ValueError: on a non-positive emission rate, a negative source height or
            horizon, or an ambiguous / missing wind specification.
    """
    import xarray as xr

    require_positive("emission_rate", emission_rate, caller="simulate_lagrangian")
    require_positive(
        "source_location[2]",
        source_location[2],
        caller="simulate_lagrangian",
        allow_zero=True,
    )
    require_positive("t_end", t_end, caller="simulate_lagrangian", allow_zero=True)

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

    x_edges = edge_axis(domain_x)
    y_edges = edge_axis(domain_y)
    z_edges = edge_axis(domain_z)
    x_centers = cell_centres(x_edges)
    y_centers = cell_centres(y_edges)
    z_centers = cell_centres(z_edges)
    cells = Cells((x_edges, y_edges, z_edges))

    # no lid is None, and so is an infinite one
    top = None if pbl_height is None or np.isinf(pbl_height) else pbl_height
    run_key, velocity_key = jax.random.split(jax.random.PRNGKey(seed))
    source = jnp.asarray(source_location, dtype=float)
    state = at_point(source, n_particles)
    state = with_turbulent_velocity(state, turbulence, key=velocity_key)
    # R_j = Σ_particles Σ_steps Δt 1[in cell j]
    result = integrate(
        state,
        Langevin(wind, turbulence, Walls(bottom=0.0, top=top)),
        t0=0.0,
        duration=float(t_end),
        step_size=float(dt),
        key=run_key,
        accumulate=Residence(cells),
    )
    # C = Q R / (N V_cell) + C_bg
    steady = steady_concentration(result.accumulated, emission_rate, n_particles, cells)
    steady = steady + background_conc
    # copies, because a view of a JAX array is read-only
    conc = np.array(steady)
    # C_col(x, y) = Σ_z C(x, y, z) Δz
    column_conc = np.array(column(steady, cells))

    dataset = xr.Dataset(
        data_vars={
            "concentration": (["x", "y", "z"], conc),
            "column_concentration": (["x", "y"], column_conc),
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
    label_concentrations(dataset)
    return dataset


__all__ = ["simulate_lagrangian"]
