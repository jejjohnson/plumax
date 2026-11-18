"""Plume masks, geometry and column units on NumPy arrays (behind ``[geo]``).

The array functions under the operators in :mod:`plumax.geo.plume`, ported
from geotoolz's ``plume`` module (#134) with their behaviour unchanged.
They take plain arrays and an affine transform, no carrier, but build on
geoops's public mask, measure and threshold helpers and on shapely, so they
need the ``[geo]`` extra too. Algorithms and unit conventions follow the
trace-gas plume literature cited per function:

- Varon et al. (2018), AMT: the integrated mass enhancement (IME) method.
- Frankenberg et al. (2016), PNAS: airborne CH4 plume detection.
- Thompson et al. (2015), AMT, and Foote et al. (2020), TGRS: the
  matched-filter CH4 retrievals whose columns these functions convert.
"""

from __future__ import annotations

from plumax.geo._src.plume_array import (
    MOLAR_MASS_KG_PER_MOL,
    STANDARD_MOLAR_VOLUME_M3_PER_MOL,
    ColumnUnit,
    Connectivity,
    ThresholdMode,
    convert_column_units,
    label_components as label_components,
    pixel_area,
    pixel_centers,
    plume_length,
    plume_mask,
    squeeze_single_band,
    wind_advection_cone,
)


__all__ = [
    "MOLAR_MASS_KG_PER_MOL",
    "STANDARD_MOLAR_VOLUME_M3_PER_MOL",
    "ColumnUnit",
    "Connectivity",
    "ThresholdMode",
    "convert_column_units",
    "pixel_area",
    "pixel_centers",
    "plume_length",
    "plume_mask",
    "squeeze_single_band",
    "wind_advection_cone",
]
