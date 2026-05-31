"""Backward Lagrangian footprint (source–receptor sensitivity).

Releasing particles from a receptor and integrating *backward* in time yields
the source–receptor sensitivity — the workhorse of regional inversions
(FLEXPART / STILT). For a receptor ``r`` and surface source cell ``s`` the
footprint is the particle residence time in the well-mixed surface layer of
``s``, normalised by the layer volume and air density (after the design doc,
[stohl2005flexpart]):

    F(r, s) = (1/N) Σ_particles Σ_steps  1[in column s, z < f_pbl·h] · Δt
              / (ρ_air · A_cell · f_pbl·h)                          [s·m²·kg⁻¹].

Backward integration reuses the forward integrator with the mean wind reversed;
for stationary turbulence the OU velocity process is statistically
time-reversible, so the same turbulence model applies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from plumax.lagrangian.concentration import _cell_indices, _step_durations
from plumax.lagrangian.particles import ParticleState, langevin_step


if TYPE_CHECKING:
    from collections.abc import Callable

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


def compute_footprint(
    receptor_location: tuple[float, float, float],
    turbulence: TurbulenceModel,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    *,
    wind: Callable[[jax.Array], jax.Array],
    n_particles: int = 5000,
    t_back: float = 600.0,
    dt: float = 1.0,
    pbl_height: float = 1000.0,
    pbl_fraction: float = 0.5,
    air_density: float = 1.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a single receptor's surface footprint by backward integration.

    Args:
        receptor_location: ``(x, y, z)`` receptor coordinates [m].
        turbulence: Turbulence model.
        domain_x / domain_y: ``(start, stop, n_cells)`` for the surface grid.
        wind: Forward mean-wind field ``t -> (u, v, w)``; integration reverses
            it internally.
        n_particles: Ensemble size.
        t_back: Backward integration time [s].
        dt: Time step [s].
        pbl_height: Boundary-layer height ``h`` [m] (reflecting lid).
        pbl_fraction: Fraction ``f_pbl`` of ``h`` defining the surface layer in
            which surface flux is "seen".
        air_density: Air density ``ρ_air`` [kg/m³].
        seed: PRNG seed.

    Returns:
        ``(footprint, x_centers, y_centers)`` where ``footprint`` has shape
        ``(nx, ny)`` and units ``s·m²·kg⁻¹``.
    """
    x_edges = np.linspace(domain_x[0], domain_x[1], int(domain_x[2]) + 1)
    y_edges = np.linspace(domain_y[0], domain_y[1], int(domain_y[2]) + 1)
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    cell_area = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))
    mix_height = pbl_fraction * pbl_height

    def back_wind(t: jax.Array) -> jax.Array:
        return -jnp.asarray(wind(t))

    key = jax.random.PRNGKey(seed)
    key, vkey = jax.random.split(key)
    rec = jnp.asarray(receptor_location, dtype=float)
    sigma0, _ = turbulence.at(rec[2])
    vel0 = jax.random.normal(vkey, (n_particles, 3)) * sigma0
    pos0 = jnp.broadcast_to(rec, (n_particles, 3))
    state = ParticleState(position=pos0, velocity=vel0)

    xe, ye = jnp.asarray(x_edges), jnp.asarray(y_edges)
    n_steps = max(int(np.ceil(t_back / dt)), 0)
    keys = jax.random.split(key, n_steps)
    # Per-step durations summing to exactly ``t_back``: the final step is
    # shortened to the remainder when ``t_back`` is not a multiple of ``dt``, so
    # the footprint integrates over the requested horizon rather than
    # ``n_steps * dt`` (which would over-count surface residence).
    times = dt * jnp.arange(n_steps)
    dts = _step_durations(t_back, dt, n_steps)
    nx, ny = len(x_c), len(y_c)

    def body(carry, inputs):
        st, hist = carry
        t, k, dt_i = inputs
        st = langevin_step(st, back_wind(t), turbulence, dt_i, k, pbl_height=pbl_height)
        below = st.position[:, 2] < mix_height
        hist = hist + _bin_surface(st.position, xe, ye, below) * dt_i
        return (st, hist), None

    (_, residence), _ = jax.lax.scan(
        body, (state, jnp.zeros((nx, ny))), (times, keys, dts)
    )

    footprint = np.asarray(residence) / (
        n_particles * air_density * cell_area * mix_height
    )
    return footprint, x_c, y_c


__all__ = ["compute_footprint"]
