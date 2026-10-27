"""A local metric frame on lat/lon: the analysis grid's map projection.

Every forward model works in metres (``PlumeGrid3D``), while every real
input (satellite pixels, catalog rows, reanalysis grids, inventories) is
lat/lon. A :class:`LocalFrame` is the one map between the two: the UTM
zone that contains a chosen origin, shifted so the origin sits at
``(x, y) = (0, 0)``.

``pyproj`` is not traceable by JAX, so frames are built and applied
outside ``jit``, on NumPy arrays, and their results enter the model as
static data. The canonical pattern::

    frame = LocalFrame.at(lat=31.95, lon=-102.15)    # Permian Basin site
    x, y = frame.to_xy(pixel_lat, pixel_lon)         # NumPy, outside jit
    receptors = jnp.asarray(np.stack([x, y], -1))    # into the model

``pyproj`` comes with the ``io`` extra (``pip install 'plumax[io]'``) and
is imported only when a frame transforms coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from plumax.les_fvm.grid import PlumeGrid3D


def utm_epsg(lat: float, lon: float) -> int:
    """EPSG code of the WGS 84 UTM zone containing ``(lat, lon)``.

    Zones are 6° wide from 180° W, ``zone = ⌊(lon + 180) / 6⌋ + 1``; the
    code is ``32600 + zone`` north of the equator and ``32700 + zone``
    south of it. The Norway and Svalbard exceptions are not applied.

    Args:
        lat: Latitude [degrees north], in [-80, 84].
        lon: Longitude [degrees east].

    Returns:
        The EPSG code.

    Raises:
        ValueError: If ``lat`` is outside UTM's [-80, 84] range.
    """
    if not -80.0 <= lat <= 84.0:
        raise ValueError(f"latitude {lat} is outside UTM's [-80, 84] range")
    zone = int(np.floor(((lon + 180.0) % 360.0) / 6.0)) + 1
    return (32600 if lat >= 0.0 else 32700) + min(zone, 60)


@dataclass(frozen=True)
class LocalFrame:
    """UTM metres relative to an origin, as the analysis grid's frame.

    ``x`` is easting and ``y`` northing in the UTM zone ``epsg``, minus the
    origin's easting and northing, so ``to_xy(origin_lat, origin_lon)`` is
    ``(0, 0)``. The axes are UTM grid axes, which differ from true east and
    north by the meridian convergence (under a degree near the zone's
    central meridian, up to about 3° at its edges at mid-latitudes).

    Attributes:
        origin_lat: Latitude of ``(x, y) = (0, 0)`` [degrees north].
        origin_lon: Longitude of ``(x, y) = (0, 0)`` [degrees east].
        epsg: EPSG code of the projected CRS; :meth:`at` picks the UTM
            zone containing the origin.
    """

    origin_lat: float
    origin_lon: float
    epsg: int

    @classmethod
    def at(cls, lat: float, lon: float) -> LocalFrame:
        """The frame in the UTM zone containing ``(lat, lon)``, origin there.

        Args:
            lat: Origin latitude [degrees north].
            lon: Origin longitude [degrees east].

        Returns:
            The frame.
        """
        return cls(
            origin_lat=float(lat), origin_lon=float(lon), epsg=utm_epsg(lat, lon)
        )

    @cached_property
    def _transformers(self) -> tuple[Any, Any, float, float]:
        try:
            from pyproj import Transformer
        except ImportError as err:
            raise ImportError(
                "LocalFrame transforms through pyproj; install it with "
                "`pip install 'plumax[io]'`"
            ) from err
        forward = Transformer.from_crs(4326, self.epsg, always_xy=True)
        inverse = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        e0, n0 = forward.transform(self.origin_lon, self.origin_lat)
        return forward, inverse, float(e0), float(n0)

    def to_xy(
        self, lat: np.ndarray | float, lon: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Frame metres of points given in lat/lon.

        Args:
            lat: Latitudes [degrees north], any shape.
            lon: Longitudes [degrees east], broadcastable with ``lat``.

        Returns:
            ``(x, y)`` [m], each of the broadcast shape.
        """
        forward, _, e0, n0 = self._transformers
        lat_b, lon_b = np.broadcast_arrays(
            np.asarray(lat, float), np.asarray(lon, float)
        )
        e, n = forward.transform(lon_b, lat_b)
        return np.asarray(e) - e0, np.asarray(n) - n0

    def to_latlon(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lat/lon of points given in frame metres.

        Args:
            x: Frame x [m], any shape.
            y: Frame y [m], broadcastable with ``x``.

        Returns:
            ``(lat, lon)`` [degrees], each of the broadcast shape.
        """
        _, inverse, e0, n0 = self._transformers
        x_b, y_b = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        lon, lat = inverse.transform(x_b + e0, y_b + n0)
        return np.asarray(lat), np.asarray(lon)

    def east_north_axes(
        self, lat: np.ndarray | float, lon: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors of true east and true north, in frame components.

        A wind ``(u_e, v_n)`` (eastward, northward) has frame components
        ``u_e · ê + v_n · n̂``. The vectors come from the frame's own
        projection by finite differences of :meth:`to_xy` over 1e-5°
        (about 1 m), so they carry the meridian convergence and nothing
        is assumed about the projection.

        Args:
            lat: Latitudes [degrees north], any shape.
            lon: Longitudes [degrees east], broadcastable with ``lat``.

        Returns:
            ``(east, north)``, each of shape ``(*shape, 2)`` holding the
            ``(x, y)`` components of a unit vector.
        """
        step = 1.0e-5
        lat_b, lon_b = np.broadcast_arrays(
            np.asarray(lat, float), np.asarray(lon, float)
        )
        x0, y0 = self.to_xy(lat_b, lon_b)
        xe, ye = self.to_xy(lat_b, lon_b + step)
        xn, yn = self.to_xy(lat_b + step, lon_b)
        east = np.stack([xe - x0, ye - y0], axis=-1)
        north = np.stack([xn - x0, yn - y0], axis=-1)
        east /= np.linalg.norm(east, axis=-1, keepdims=True)
        north /= np.linalg.norm(north, axis=-1, keepdims=True)
        return east, north

    def grid_indices(
        self, lat: np.ndarray | float, lon: np.ndarray | float, grid: PlumeGrid3D
    ) -> tuple[np.ndarray, np.ndarray]:
        """Indices of the analysis T-cells containing lat/lon points.

        A T-cell ``i`` spans ``[x_i − dx/2, x_i + dx/2)``, so
        ``i = ⌊(x − x₀ + dx/2) / dx⌋`` with ``x₀ = grid.x[0]``, and
        likewise for ``j`` along ``y``. Points outside the grid get ``-1``.

        Args:
            lat: Latitudes [degrees north], any shape.
            lon: Longitudes [degrees east], broadcastable with ``lat``.
            grid: Analysis grid, in this frame's metres.

        Returns:
            ``(i, j)`` integer arrays (x index, y index) of the broadcast
            shape, ``-1`` outside.
        """
        x, y = self.to_xy(lat, lon)
        gx = np.asarray(grid.x, float)
        gy = np.asarray(grid.y, float)
        i = np.floor((x - gx[0] + 0.5 * grid.dx) / grid.dx).astype(int)
        j = np.floor((y - gy[0] + 0.5 * grid.dy) / grid.dy).astype(int)
        outside = (i < 0) | (i >= gx.size) | (j < 0) | (j >= gy.size)
        return np.where(outside, -1, i), np.where(outside, -1, j)
