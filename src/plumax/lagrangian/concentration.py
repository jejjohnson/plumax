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

from plumax.lagrangian._src.concentration import (
    bin_positions,
    simulate_lagrangian,
    uniform_wind,
    wind_from_speed_direction,
)


__all__ = [
    "bin_positions",
    "simulate_lagrangian",
    "uniform_wind",
    "wind_from_speed_direction",
]
