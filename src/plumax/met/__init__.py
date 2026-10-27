"""Meteorological forcing for the transport tiers (Tier 0 prerequisites).

The transport models are driven by an external meteorology; this package
owns the container those models consume and the loaders that populate it
from numerical-weather-prediction output.

- :class:`MetField` — gridded wind, temperature, pressure and PBL height on
  a ``les_fvm`` analysis grid, on the ``les_fvm`` C-grid stagger, with a
  time axis.
- :func:`load_wrf` — read a ``wrfout`` file onto an analysis grid.
- :func:`load_era5` — read ERA5 pressure- and single-level files onto an
  analysis grid placed by a :class:`LocalFrame`.
- :class:`LocalFrame` — the analysis grid's metric frame on lat/lon (a
  UTM zone shifted to an origin); built and applied outside ``jit``.
- :class:`TimeAxis` — seconds since ``t0`` ↔ UTC timestamps.
- :func:`surface_layer_from_met` — Monin–Obukhov ``u*``, ``1/L``, ``w*``
  (pure JAX) with the stability functions in :mod:`plumax.met.surface_layer`.
- :func:`pbl_height_bulk_richardson` / :func:`hanna_turbulence_from` —
  PBL height from the analysis profiles, and Tier II's Hanna turbulence
  built from the met.
- :func:`to_prescribed_wind` — expose a :class:`MetField` as the
  :class:`~plumax.les_fvm.wind.PrescribedWindField` the Eulerian solver
  takes, with piecewise-linear interpolation between met times.
"""

from __future__ import annotations

from plumax.met.era5 import load_era5
from plumax.met.field import MetField, to_prescribed_wind
from plumax.met.frames import LocalFrame
from plumax.met.pbl import hanna_turbulence_from, pbl_height_bulk_richardson
from plumax.met.surface_layer import SurfaceLayer, surface_layer_from_met
from plumax.met.time import TimeAxis
from plumax.met.wrf import load_wrf


__all__ = [
    "LocalFrame",
    "MetField",
    "SurfaceLayer",
    "TimeAxis",
    "hanna_turbulence_from",
    "load_era5",
    "load_wrf",
    "pbl_height_bulk_richardson",
    "surface_layer_from_met",
    "to_prescribed_wind",
]
