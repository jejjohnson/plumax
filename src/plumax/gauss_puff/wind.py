"""Time-varying wind schedules with diffrax-based cumulative integrals.

A Gaussian puff released at time ``t_r`` under a time-varying wind field
``V(t) = (u(t), v(t))`` satisfies the trivial ODE ``dX/dt = V(t)``, whose
integral is

    X(t) − X(t_r) = ∫_{t_r}^{t} V(τ) dτ .

Because every puff sees the same wind, the per-puff offsets are differences
of a single *cumulative* integral evaluated at puff release times and the
current simulation time:

    I_u(t) = ∫_{t_0}^{t} u(τ) dτ                          [m]
    I_v(t) = ∫_{t_0}^{t} v(τ) dτ                          [m]
    S(t)   = ∫_{t_0}^{t} √(u(τ)² + v(τ)²) dτ              [m]   (travel distance)

We compute these with :func:`diffrax.diffeqsolve` driven by a piecewise
-linear wind interpolation, so the result is exact for piecewise-linear
winds up to the solver tolerance, and handles arbitrary continuous winds
with adaptive stepping.

Public surface
--------------
- ``WindSchedule``              : equinox.Module wrapping (t, u, v) arrays
- ``cumulative_wind_integrals`` : diffrax solve returning ``(I_u, I_v, S)``
"""

from __future__ import annotations

from plumax.gauss_puff._src.wind import (
    WindSchedule,
    cumulative_wind_integrals,
)


__all__ = [
    "WindSchedule",
    "cumulative_wind_integrals",
]
