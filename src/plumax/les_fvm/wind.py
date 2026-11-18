"""Prescribed 3-D wind fields for the Eulerian dispersion solver.

The L2 model does **not** resolve the flow — wind is an external input.
A :class:`PrescribedWindField` is a callable pytree that maps a query
time ``t`` to the three velocity components ``(u, v, w)`` on the grid,
with ``u`` at U-points, ``v`` at V-points, and ``w`` at T-points
(the C-grid convention adopted by ``les_fvm``; see ``grid.py``).

Three constructors are provided:

- :func:`uniform_wind_field`        — constant in space *and* time.
- :func:`wind_field_from_schedule`  — spatially uniform, time-varying via
  a ``gauss_puff.wind.WindSchedule``.
- :func:`wind_field_from_callable`  — general-purpose escape hatch for
  user-supplied ``(t, X, Y, Z) -> (u, v, w)`` closures.

All fields return arrays with the same ``[Nz, Ny, Nx]`` layout as the
tracer concentration, so they can be consumed directly by the advection
operators.
"""

from __future__ import annotations

from plumax.les_fvm._src.wind import (
    PrescribedWindField,
    uniform_wind_field,
    wind_field_from_callable,
    wind_field_from_schedule,
)


__all__ = [
    "PrescribedWindField",
    "uniform_wind_field",
    "wind_field_from_callable",
    "wind_field_from_schedule",
]
