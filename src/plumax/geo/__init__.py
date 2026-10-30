"""Plume operators on GeoTensors, behind the ``[geo]`` extra.

- :mod:`plumax.geo.plume`: the ten plume operators ported from geotoolz's
  ``plume`` module (#134): masks, contours, footprints, shape filters,
  column statistics and QND features, wind cones, the integrated mass
  enhancement, the cross-sectional flux and column-unit conversion.
- :mod:`plumax.geo.plume_array`: the array functions under them
  (`plume_mask`, `wind_advection_cone`, `convert_column_units`, ...).
- :mod:`plumax.geo.vetting`: `PlumeVetting`, :mod:`plumax.vetting`'s
  candidate vetting and scene null on GeoTensors, as one Dataset (#157).

Importing this subpackage does not need the extra; the operators load on
first use, and a missing extra raises an ``ImportError`` naming it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any


_LAZY = {
    "ColumnToMass": "plume",
    "CrossSectionalFlux": "plume",
    "IMEEstimate": "plume",
    "PlumeColumnStats": "plume",
    "PlumeContours": "plume",
    "PlumeFootprint": "plume",
    "PlumeMask": "plume",
    "PlumeQNDFeatures": "plume",
    "PlumeShapeFilter": "plume",
    "PlumeVetting": "vetting",
    "WindAdvectionCone": "plume",
    "convert_column_units": "plume_array",
    "plume_mask": "plume_array",
    "wind_advection_cone": "plume_array",
}


def __getattr__(name: str) -> Any:  # PEP 562: the operators need [geo]
    module = _LAZY.get(name)
    if module is not None:
        value = getattr(importlib.import_module(f"plumax.geo.{module}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'plumax.geo' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    from plumax.geo.plume import (
        ColumnToMass,
        CrossSectionalFlux,
        IMEEstimate,
        PlumeColumnStats,
        PlumeContours,
        PlumeFootprint,
        PlumeMask,
        PlumeQNDFeatures,
        PlumeShapeFilter,
        WindAdvectionCone,
    )
    from plumax.geo.plume_array import (
        convert_column_units,
        plume_mask,
        wind_advection_cone,
    )
    from plumax.geo.vetting import PlumeVetting


__all__ = [
    "ColumnToMass",
    "CrossSectionalFlux",
    "IMEEstimate",
    "PlumeColumnStats",
    "PlumeContours",
    "PlumeFootprint",
    "PlumeMask",
    "PlumeQNDFeatures",
    "PlumeShapeFilter",
    "PlumeVetting",
    "WindAdvectionCone",
    "convert_column_units",
    "plume_mask",
    "wind_advection_cone",
]
