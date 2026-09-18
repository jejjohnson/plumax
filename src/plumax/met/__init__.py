"""Meteorological forcing for the transport tiers (Tier 0 prerequisites).

The transport models are driven by an external meteorology; this package
owns the container those models consume and the loaders that populate it
from numerical-weather-prediction output.

- :class:`MetField` — gridded wind, temperature, pressure and PBL height on
  a ``les_fvm`` analysis grid, on the ``les_fvm`` C-grid stagger, with a
  time axis.
- :func:`load_wrf` — read a ``wrfout`` file onto an analysis grid.
- :func:`to_prescribed_wind` — expose a :class:`MetField` as the
  :class:`~plumax.les_fvm.wind.PrescribedWindField` the Eulerian solver
  takes, with piecewise-linear interpolation between met times.
"""

from __future__ import annotations

from plumax.met.field import MetField, to_prescribed_wind
from plumax.met.wrf import load_wrf


__all__ = ["MetField", "load_wrf", "to_prescribed_wind"]
