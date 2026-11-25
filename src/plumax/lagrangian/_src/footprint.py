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
from jaxtyping import Array, Float, PRNGKeyArray, UInt32
from tracerax.advect import Walls
from tracerax.bin import Cells
from tracerax.footprint import surface

from plumax._src.domain import cell_centres, edge_axis


if TYPE_CHECKING:
    from tracerax.advect import TurbulenceModel
    from tracerax.velocity import Velocity


def _surface_sensitivity(
    receptor: Float[Array, " 3"],
    wind: Velocity,
    turbulence: TurbulenceModel,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    *,
    n_particles: int,
    t_back: float,
    dt: float,
    pbl_height: float,
    pbl_fraction: float,
    air_density: float,
    receptor_time: float,
    key: PRNGKeyArray | UInt32[Array, " key"],
) -> Float[Array, "nx ny"]:
    """The footprint of one receptor [s m² kg⁻¹], a pure function of arrays.

    It can be traced in ``wind``, ``receptor`` and ``key``, so a caller can compile
    it once and ``vmap`` it over receptors, which
    :class:`~plumax.lagrangian.FootprintSimulator` does. The arguments are those of
    :func:`compute_footprint`, with the key of the receptor's particles.
    """
    cells = Cells((edge_axis(domain_x), edge_axis(domain_y)))
    # F = H / (N ρ_air f_pbl·h), H(s) = Σ_particles Σ_steps 1[in s, z < f_pbl·h] Δt
    footprint = surface(
        receptor,
        wind,
        turbulence,
        cells,
        n_particles=n_particles,
        receptor_time=receptor_time,
        duration=float(t_back),
        step_size=float(dt),
        surface_depth=pbl_fraction * pbl_height,
        air_density=air_density,
        key=key,
        walls=Walls(bottom=0.0, top=pbl_height),
    )
    return footprint.sensitivity


def compute_footprint(
    receptor_location: tuple[float, float, float],
    turbulence: TurbulenceModel,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    *,
    wind: Velocity,
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
        wind: Forward mean wind, a tracerax velocity ``wind(t, position)``:
            uniform, or gridded and sampled at each particle. Integration reverses
            it internally and samples it on the receptor clock (see
            ``receptor_time``).
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
    sensitivity = _surface_sensitivity(
        jnp.asarray(receptor_location, dtype=float),
        wind,
        turbulence,
        domain_x,
        domain_y,
        n_particles=n_particles,
        t_back=t_back,
        dt=dt,
        pbl_height=pbl_height,
        pbl_fraction=pbl_fraction,
        air_density=air_density,
        receptor_time=receptor_time,
        key=jax.random.PRNGKey(seed),
    )
    x_centers = cell_centres(edge_axis(domain_x))
    y_centers = cell_centres(edge_axis(domain_y))
    return np.asarray(sensitivity), x_centers, y_centers


__all__ = ["compute_footprint"]
