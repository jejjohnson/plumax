"""The plume operators on GeoArrays (behind the ``[geo]`` extra).

Ten operators ported from geotoolz's ``plume`` module (#134) with their
behaviour unchanged; its eleventh, ``SBMP``, moved to radtrax as
``radtrax.geo.MBMP``. They wrap the array functions in
:mod:`plumax.geo.plume_array` and are pipekit Operators built with geoops's
public carrier helpers, as radtrax's ``[geo]`` operators are.

Per-pixel operators accept a ``geoarray.GeoArray`` or a plain
``np.ndarray`` and return the same carrier kind; geo-dependent operators
(advection geometry, pixel-area maths, footprints) require a georeferenced
``GeoArray`` and raise ``TypeError`` otherwise. The area and distance
operators (``PlumeFootprint``, ``WindAdvectionCone``, ``IMEEstimate``,
``CrossSectionalFlux``) work in metres and raise ``ValueError`` on a
geographic CRS (see :func:`geoops.carrier.require_projected_crs`). Vector
outputs are ``geopandas.GeoDataFrame``. Algorithms follow the literature
cited in :mod:`plumax.geo.plume_array`:

- ``PlumeMask`` / ``PlumeContours`` / ``PlumeFootprint``: Frankenberg
  et al. (2016) detection and segmentation conventions.
- ``IMEEstimate``: Varon et al. (2018) integrated mass enhancement.
- ``CrossSectionalFlux``: Krings et al. (2011) / Varon (2018) appendix.
- ``PlumeShapeFilter`` / ``PlumeColumnStats`` / ``PlumeQNDFeatures``:
  Pérez Carrasco et al. (2026), the false-plume classifier's features.

The IME and cross-section operators expect the enhancement carrier to be
in kg/m^2. Use :class:`ColumnToMass` to convert from ppm m or mol/m^2.

Nodata: pixels that are non-finite or equal the carrier's
``fill_value`` (see :func:`geoops.carrier.valid_pixels`) never
enter a threshold, statistic or sum, and are never part of a plume mask or
contour. Output fill values follow the output's meaning, not the
input's: boolean masks declare ``fill_value=False``, float
products (scores, converted columns) hold ``NaN`` at nodata pixels and
declare ``fill_value=NaN``, and label images use ``0`` (see
:class:`PlumeContours` for its one documented exception).
"""

from __future__ import annotations

from plumax.geo._src.plume import (
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
    "WindAdvectionCone",
]
