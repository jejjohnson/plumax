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
wind the backward trajectory must sample it on the *receptor* clock — at the
start of each forward interval it undoes, ``receptor_time − τ − Δt`` for
backward pseudo-time ``τ`` and step ``Δt`` — which ``receptor_time`` supplies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from plumax.lagrangian.concentration import _cell_indices
from plumax.lagrangian.particles import (
    ParticleState,
    langevin_step,
    n_steps_for_horizon,
    step_durations,
)


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
    receptor_time: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a single receptor's surface footprint by backward integration.

    Args:
        receptor_location: ``(x, y, z)`` receptor coordinates [m].
        turbulence: Turbulence model.
        domain_x / domain_y: ``(start, stop, n_cells)`` for the surface grid.
        wind: Forward mean-wind field ``t -> (u, v, w)``; integration reverses
            it internally and samples it on the receptor clock (see
            ``receptor_time``).
        n_particles: Ensemble size.
        t_back: Backward integration time [s].
        dt: Time step [s].
        pbl_height: Boundary-layer height ``h`` [m] (reflecting lid).
        pbl_fraction: Fraction ``f_pbl`` of ``h`` defining the surface layer in
            which surface flux is "seen".
        air_density: Air density ``ρ_air`` [kg/m³].
        receptor_time: Physical time ``T`` of the receptor observation [s]. The
            backward step at pseudo-time ``τ`` (duration ``Δt``) samples
            ``−wind(T − τ − Δt)`` — the start of the forward interval it undoes —
            so a time-varying ``wind`` is evaluated at the correct physical time
            and the run is the exact discrete reverse of a forward one. The
            default ``0.0`` is exact for a stationary ``wind`` (its argument is
            ignored) and simply reverses it.
        seed: PRNG seed.

    Returns:
        ``(footprint, x_centers, y_centers)`` where ``footprint`` has shape
        ``(nx, ny)`` and units ``s·m²·kg⁻¹`` (flux sensitivity; see the module
        docstring).
    """
    x_edges = np.linspace(domain_x[0], domain_x[1], int(domain_x[2]) + 1)
    y_edges = np.linspace(domain_y[0], domain_y[1], int(domain_y[2]) + 1)
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    mix_height = pbl_fraction * pbl_height

    def back_wind(t: jax.Array, dt_i: jax.Array) -> jax.Array:
        # Sample the mean wind on the receptor clock, at the START of the forward
        # interval this backward step undoes: the step at pseudo-time ``t`` with
        # duration ``dt_i`` reverses the forward interval ``[T − t − dt_i,
        # T − t]``, whose step-start value is ``wind(T − t − dt_i)`` — matching
        # the step-start sampling of ``integrate_particles``, so the backward run
        # is the exact discrete reverse of the forward one. Negated for the
        # reversed trajectory.
        return -jnp.asarray(wind(receptor_time - t - dt_i))

    key = jax.random.PRNGKey(seed)
    key, vkey = jax.random.split(key)
    rec = jnp.asarray(receptor_location, dtype=float)
    sigma0, _ = turbulence.at(rec[2])
    vel0 = jax.random.normal(vkey, (n_particles, 3)) * sigma0
    pos0 = jnp.broadcast_to(rec, (n_particles, 3))
    state = ParticleState(position=pos0, velocity=vel0)

    xe, ye = jnp.asarray(x_edges), jnp.asarray(y_edges)
    n_steps = n_steps_for_horizon(t_back, dt)
    keys = jax.random.split(key, n_steps)
    # Per-step durations summing to exactly ``t_back``: the final step is
    # shortened to the remainder when ``t_back`` is not a multiple of ``dt``, so
    # the footprint integrates over the requested horizon rather than
    # ``n_steps * dt`` (which would over-count surface residence).
    times = dt * jnp.arange(n_steps)
    dts = step_durations(t_back, dt, n_steps)
    nx, ny = len(x_c), len(y_c)

    def body(carry, inputs):
        st, hist = carry
        t, k, dt_i = inputs
        st = langevin_step(
            st, back_wind(t, dt_i), turbulence, dt_i, k, pbl_height=pbl_height
        )
        below = st.position[:, 2] < mix_height
        hist = hist + _bin_surface(st.position, xe, ye, below) * dt_i
        return (st, hist), None

    (_, residence), _ = jax.lax.scan(
        body, (state, jnp.zeros((nx, ny))), (times, keys, dts)
    )

    # Flux-sensitivity normalisation: residence-time per particle divided by the
    # surface-layer air mass column density ρ·h (no per-cell-area division), so
    # ``F`` has units s·m²·kg⁻¹ and pairs with a surface flux in kg·m⁻²·s⁻¹.
    footprint = np.asarray(residence) / (n_particles * air_density * mix_height)
    return footprint, x_c, y_c


__all__ = ["compute_footprint"]
