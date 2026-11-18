"""Plume operators on GeoArrays, behind the ``[geo]`` extra.

- :mod:`plumax.geo.plume`: the ten plume operators ported from geotoolz's
  ``plume`` module (#134): masks, contours, footprints, shape filters,
  column statistics and QND features, wind cones, the integrated mass
  enhancement, the cross-sectional flux and column-unit conversion.
- :mod:`plumax.geo.plume_array`: the array functions under them
  (`plume_mask`, `wind_advection_cone`, `convert_column_units`, ...).
- :mod:`plumax.geo.starts`: the starting values of an image fit, the
  direction from the source to the mask's centroid (`plume_direction`) and a
  grid of source positions (`source_offset_grid`) for a multistart solve.
- :mod:`plumax.geo.transects`: cross-plume transects of a column image
  (`cross_plume_transects`, `CrossPlumeProfile`), each one's line density,
  centre and width, the smoothed centreline and the width law.
- :mod:`plumax.geo.vetting`: `PlumeVetting`, :mod:`plumax.vetting`'s
  candidate vetting and scene null on GeoArrays, as one Dataset (#157).

Importing this subpackage does not need the extra; the operators load on
first use, and a missing extra raises an ``ImportError`` naming it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any


_LAZY = {
    "ColumnToMass": "plume",
    "CrossPlumeProfile": "transects",
    "CrossSectionalFlux": "plume",
    "IMEEstimate": "plume",
    "PlumeColumnStats": "plume",
    "PlumeContours": "plume",
    "PlumeFootprint": "plume",
    "PlumeMask": "plume",
    "PlumeQNDFeatures": "plume",
    "PlumeShapeFilter": "plume",
    "PlumeVetting": "vetting",
    "TransectProfiles": "transects",
    "WindAdvectionCone": "plume",
    "convert_column_units": "plume_array",
    "cross_plume_transects": "transects",
    "fit_transect_gaussians": "transects",
    "fit_width_law": "transects",
    "plume_direction": "starts",
    "plume_mask": "plume_array",
    "smooth_centreline": "transects",
    "source_offset_grid": "starts",
    "transect_moments": "transects",
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
    from plumax.geo.starts import plume_direction, source_offset_grid
    from plumax.geo.transects import (
        CrossPlumeProfile,
        TransectProfiles,
        cross_plume_transects,
        fit_transect_gaussians,
        fit_width_law,
        smooth_centreline,
        transect_moments,
    )
    from plumax.geo.vetting import PlumeVetting


__all__ = [
    "ColumnToMass",
    "CrossPlumeProfile",
    "CrossSectionalFlux",
    "IMEEstimate",
    "PlumeColumnStats",
    "PlumeContours",
    "PlumeFootprint",
    "PlumeMask",
    "PlumeQNDFeatures",
    "PlumeShapeFilter",
    "PlumeVetting",
    "TransectProfiles",
    "WindAdvectionCone",
    "convert_column_units",
    "cross_plume_transects",
    "fit_transect_gaussians",
    "fit_width_law",
    "plume_direction",
    "plume_mask",
    "smooth_centreline",
    "source_offset_grid",
    "transect_moments",
    "wind_advection_cone",
]
