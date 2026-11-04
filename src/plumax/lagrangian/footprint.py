"""Backward Lagrangian footprint (source–receptor sensitivity).

Releasing particles from a receptor and integrating *backward* in time yields
the source–receptor sensitivity — the workhorse of regional inversions
(FLEXPART / STILT). For a receptor ``r`` and surface source cell ``s`` the
footprint is the particle residence time in the well-mixed surface layer of
``s``, normalised by the layer volume and air density (after the design doc,
[stohl2005flexpart]):

    F(r, s) = (1/N) Σ_particles Σ_steps  1[in column s, z < f_pbl·h] · Δt
              / (ρ_air · f_pbl·h)                                   [s·m²·kg⁻¹].

This is the **flux-sensitivity** convention: there is no per-cell-area division,
so ``F`` is the sensitivity of the receptor value to a surface **flux**
``q`` in kg·m⁻²·s⁻¹ (``y = Σ_s F(r,s)·q(s)``), matching [stohl2005flexpart]. A
per-cell *emission-rate* sensitivity (kg/s) would divide by ``A_cell`` as well
and carry units s·kg⁻¹ — a different convention; the consumer
:mod:`plumax.lagrangian.inversion` assumes the flux one used here.

Backward integration reuses the forward integrator with the mean wind reversed;
for stationary turbulence the OU velocity process is statistically
time-reversible, so the same turbulence model applies. For a time-varying mean
wind the backward trajectory undoes the forward intervals in reverse order (the
partial final interval first) and samples ``−wind`` at each interval's physical
start time on the *receptor* clock, so it is the exact discrete reverse of a
forward run — which ``receptor_time`` anchors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from plumax.lagrangian._src.wind_interp import mean_wind
from plumax.lagrangian.concentration import _cell_indices
from plumax.lagrangian.particles import (
    ParticleState,
    langevin_step,
    n_steps_for_horizon,
    step_durations,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from plumax.lagrangian._src.wind_interp import GriddedWind
    from plumax.lagrangian.particles import TurbulenceModel


def _bin_surface(
    positions: jax.Array,
    x_edges: jax.Array,
    y_edges: jax.Array,
    below: jax.Array,
) -> jax.Array:
    """Histogram in-layer particles onto the 2-D surface grid (weighted counts)."""
    nx = x_edges.shape[0] - 1
    ny = y_edges.shape[0] - 1
    ix, gx = _cell_indices(positions[:, 0], x_edges)
    iy, gy = _cell_indices(positions[:, 1], y_edges)
    weight = (gx & gy & below).astype(float)
    flat = ix * ny + iy
    return jnp.zeros(nx * ny).at[flat].add(weight).reshape(nx, ny)


def _surface_edges(
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Cell edges along x and y of the surface grid."""
    x_edges = np.linspace(domain_x[0], domain_x[1], int(domain_x[2]) + 1)
    y_edges = np.linspace(domain_y[0], domain_y[1], int(domain_y[2]) + 1)
    return x_edges, y_edges


def _backward_steps(
    t_back: float, dt: float, receptor_time: float
) -> tuple[Float[Array, " time"], Float[Array, " time"]]:
    """Per-step durations of a backward run and the wind time of each step.

    Per-step durations sum to exactly ``t_back``: the final forward step is
    shortened to the remainder when ``t_back`` is not a multiple of ``dt``, so
    the footprint integrates over the requested horizon rather than
    ``n_steps * dt`` (which would over-count surface residence).

    Backward integration undoes the forward intervals in reverse order, so the
    first backward step consumes the (possibly partial) *final* forward
    interval, hence the reversal. ``wind_starts[j]`` is the physical start
    time of the interval step ``j`` undoes; sampling ``−wind(wind_starts[j])``
    (step-start, matching ``integrate_particles``) makes the backward run the
    exact discrete reverse of the forward one for any horizon.
    """
    n_steps = n_steps_for_horizon(t_back, dt)
    durations = step_durations(t_back, dt, n_steps)[::-1]
    wind_starts = receptor_time - jnp.cumsum(durations)
    return durations, wind_starts


def _surface_residence(
    wind: Callable[[jax.Array], jax.Array] | GriddedWind,
    receptor: Float[Array, " 3"],
    key: PRNGKeyArray,
    turbulence: TurbulenceModel,
    x_edges: Float[Array, " nx_edge"],
    y_edges: Float[Array, " ny_edge"],
    durations: Float[Array, " time"],
    wind_starts: Float[Array, " time"],
    *,
    n_particles: int,
    pbl_height: float,
    mix_height: float,
) -> Float[Array, "nx ny"]:
    """Summed residence time in the surface layer, a pure function of arrays.

    H(s) = Σ_particles Σ_steps 1[in column s, z < f_pbl·h] · Δt.

    It can be traced in ``wind``, ``receptor``, ``key`` and the steps, so a
    caller can compile it once and ``vmap`` it over receptors, which
    :class:`~plumax.lagrangian.FootprintSimulator` does.

    Args:
        wind: Forward mean wind, reversed here.
        receptor: Receptor position (x, y, z) [m].
        key: PRNG key of this receptor's particles.
        turbulence: Turbulence model.
        x_edges: Cell edges of the surface grid along x [m].
        y_edges: Cell edges of the surface grid along y [m].
        durations: Backward step durations Δt [s].
        wind_starts: Time [s] at which each backward step samples the wind.
        n_particles: Ensemble size.
        pbl_height: Boundary-layer height h [m], the reflecting lid.
        mix_height: Surface-layer depth f_pbl·h [m].

    Returns:
        The residence time per surface cell [s], summed over the particles.
    """
    key, velocity_key = jax.random.split(key)
    sigma0, _ = turbulence.at(receptor[2])
    vel0 = jax.random.normal(velocity_key, (n_particles, 3)) * sigma0
    pos0 = jnp.broadcast_to(receptor, (n_particles, 3))
    state = ParticleState(position=pos0, velocity=vel0)
    step_keys = jax.random.split(key, durations.shape[0])
    nx, ny = x_edges.shape[0] - 1, y_edges.shape[0] - 1

    def body(carry, inputs):
        st, hist = carry
        t_start, k, dt_i = inputs
        st = langevin_step(
            st,
            -mean_wind(wind, st.position, t_start),
            turbulence,
            dt_i,
            k,
            pbl_height=pbl_height,
        )
        below = st.position[:, 2] < mix_height
        hist = hist + _bin_surface(st.position, x_edges, y_edges, below) * dt_i
        return (st, hist), None

    (_, residence), _ = jax.lax.scan(
        body, (state, jnp.zeros((nx, ny))), (wind_starts, step_keys, durations)
    )
    return residence


def _flux_sensitivity(
    residence: Float[Array, "... nx ny"],
    n_particles: int,
    air_density: float,
    mix_height: float,
) -> np.ndarray:
    """Normalise residence time into the footprint [s m² kg⁻¹].

    F = H / (N ρ_air f_pbl·h): residence time per particle divided by the
    surface-layer air mass column density (no per-cell-area division), so ``F``
    pairs with a surface flux in kg m⁻² s⁻¹. It runs on the host so a compiled
    caller divides exactly as :func:`compute_footprint` does.
    """
    return np.asarray(residence) / (n_particles * air_density * mix_height)


def compute_footprint(
    receptor_location: tuple[float, float, float],
    turbulence: TurbulenceModel,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    *,
    wind: Callable[[jax.Array], jax.Array] | GriddedWind,
    n_particles: int = 5000,
    t_back: float = 600.0,
    dt: float = 1.0,
    pbl_height: float = 1000.0,
    pbl_fraction: float = 0.5,
    air_density: float = 1.2,
    receptor_time: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a single receptor's surface footprint by backward integration.

    Args:
        receptor_location: ``(x, y, z)`` receptor coordinates [m].
        turbulence: Turbulence model.
        domain_x: ``(start, stop, n_cells)`` along x for the surface grid.
        domain_y: ``(start, stop, n_cells)`` along y for the surface grid.
        wind: Forward mean-wind field ``t -> (u, v, w)`` or a
            :class:`~plumax.lagrangian.GriddedWind` sampled at each particle;
            integration reverses it internally and samples it on the receptor
            clock (see ``receptor_time``).
        n_particles: Ensemble size.
        t_back: Backward integration time [s].
        dt: Time step [s].
        pbl_height: Boundary-layer height ``h`` [m] (reflecting lid).
        pbl_fraction: Fraction ``f_pbl`` of ``h`` defining the surface layer in
            which surface flux is "seen".
        air_density: Air density ``ρ_air`` [kg/m³].
        receptor_time: Physical time ``T`` of the receptor observation [s].
            Backward steps undo the forward intervals in reverse order (the
            partial final interval first) and sample ``−wind`` at each interval's
            physical start time, so a time-varying ``wind`` is evaluated at the
            correct time and the run is the exact discrete reverse of a forward
            one. The default ``0.0`` is exact for a stationary ``wind`` (its
            argument is ignored) and simply reverses it.
        seed: PRNG seed.

    Returns:
        ``(footprint, x_centers, y_centers)`` where ``footprint`` has shape
        ``(nx, ny)`` and units ``s·m²·kg⁻¹`` (flux sensitivity; see the module
        docstring).
    """
    x_edges, y_edges = _surface_edges(domain_x, domain_y)
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    mix_height = pbl_fraction * pbl_height
    durations, wind_starts = _backward_steps(t_back, dt, receptor_time)
    residence = _surface_residence(
        wind,
        jnp.asarray(receptor_location, dtype=float),
        jax.random.PRNGKey(seed),
        turbulence,
        jnp.asarray(x_edges),
        jnp.asarray(y_edges),
        durations,
        wind_starts,
        n_particles=n_particles,
        pbl_height=pbl_height,
        mix_height=mix_height,
    )
    footprint = _flux_sensitivity(residence, n_particles, air_density, mix_height)
    return footprint, x_c, y_c


__all__ = ["compute_footprint"]
