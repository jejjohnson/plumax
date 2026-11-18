"""diffrax-compatible RHS for the Eulerian dispersion ODE system.

The full tendency ``∂_t C = RHS(t, C)`` is assembled by composing the
advection, diffusion, source, and boundary-application modules from this
sub-package.  Wrapping the RHS in an :class:`equinox.Module` keeps it a
single pytree leaf that diffrax can JIT as ``term = ODETerm(rhs)``.
"""

from __future__ import annotations

from plumax.les_fvm._src.dynamics import (
    EulerianDispersionRHS,
)


__all__ = [
    "EulerianDispersionRHS",
]
