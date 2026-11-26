"""Time-varying wind schedules and their cumulative integrals.

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

The wind is linear between its knots, so I_u and I_v are exact running
trapezoid sums on the knots and the save times, and S is integrated per
interval by adaptive Gauss–Kronrod (both from integrix). Both are
differentiable in reverse mode.

Public surface
--------------
- ``WindSchedule``              : equinox.Module wrapping (t, u, v) arrays
- ``cumulative_wind_integrals`` : running integrals ``(I_u, I_v, S)``
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
