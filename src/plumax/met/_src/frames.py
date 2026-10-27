"""LocalFrame: the analysis grid's metric frame on lat/lon (a shifted UTM zone).

Forward models work in metres; real inputs are lat/lon. A frame is the one map
between them. ``pyproj`` is not traceable by JAX, so frames are built and
applied outside ``jit`` on NumPy arrays, and their results enter the model as
data.
"""

from __future__ import annotations

import dataclasses
from functools import cached_property
from typing import Any

import einx
import numpy as np
from jaxtyping import Float, Int

from plumax._src.errors import DomainError, MissingExtraError
from plumax.les_fvm.grid import PlumeGrid3D


#: Latitude/longitude step [degrees] of the finite differences in
#: :meth:`LocalFrame.east_north_axes` (about 1 m).
AXIS_STEP_DEGREES = 1.0e-5


def utm_epsg(lat: float, lon: float) -> int:
    """EPSG code of the WGS 84 UTM zone containing ``(lat, lon)``.

    zone = ⌊(λ + 180) / 6⌋ + 1,  EPSG = 32600 + zone (north) or 32700 + zone (south).

    The Norway and Svalbard exceptions are not applied.

    Args:
        lat: Latitude φ [degrees north], in [-80, 84].
        lon: Longitude λ [degrees east].

    Returns:
        The EPSG code.

    Raises:
        DomainError: If ``lat`` is outside UTM's [-80, 84] range.

    Examples:
        A Permian Basin site lies in zone 13 north:

        >>> utm_epsg(31.95, -102.15)
        32613

        Southern hemisphere, and the antimeridian wrapping to zone 1:

        >>> utm_epsg(-23.5, 133.9), utm_epsg(10.0, 180.0)
        (32753, 32601)

        UTM stops at 84° N:

        >>> utm_epsg(85.0, 0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        DomainError: latitude 85.0 is outside UTM's [-80, 84] range
    """
    if not -80.0 <= lat <= 84.0:
        raise DomainError(f"latitude {lat} is outside UTM's [-80, 84] range")
    # λ wrapped to [0, 360) from 180° W
    wrapped_lon = (lon + 180.0) % 360.0
    # zone = ⌊λ / 6⌋ + 1
    zone = int(np.floor(wrapped_lon / 6.0)) + 1
    hemisphere_base = 32600 if lat >= 0.0 else 32700
    return hemisphere_base + zone


@dataclasses.dataclass(frozen=True)
class LocalFrame:
    """UTM metres relative to an origin, as the analysis grid's frame.

    (x, y) = (E(φ, λ) − E₀, N(φ, λ) − N₀),

    with ``E``, ``N`` the easting and northing in the UTM zone ``epsg`` and
    ``(E₀, N₀)`` the origin's. The axes are UTM grid axes, which differ from
    true east and north by the meridian convergence (under a degree near the
    zone's central meridian, about 3° at its edges at mid-latitudes).

    Args:
        origin_lat: Latitude φ₀ of ``(x, y) = (0, 0)`` [degrees north].
        origin_lon: Longitude λ₀ of ``(x, y) = (0, 0)`` [degrees east].
        epsg: EPSG code of the projected CRS; :meth:`at` picks the UTM zone
            containing the origin.

    Raises:
        MissingExtraError: When a transform runs without ``pyproj``
            (``pip install 'plumax[io]'``).

    Examples:
        The origin maps to zero, and a round trip is exact to well under 1 mm:

        >>> import numpy as np
        >>> frame = LocalFrame.at(31.95, -102.15)
        >>> x, y = frame.to_xy(31.95, -102.15)
        >>> bool(np.allclose([x, y], 0.0, atol=1e-6))
        True
        >>> lat, lon = frame.to_latlon(250.0, -400.0)
        >>> bool(np.allclose(frame.to_xy(lat, lon), [250.0, -400.0], atol=1e-6))
        True

        Arrays keep their shape:

        >>> x, y = frame.to_xy(np.full((2, 3), 31.95), np.full((2, 3), -102.15))
        >>> x.shape
        (2, 3)

        The configuration round-trips:

        >>> LocalFrame.from_config(frame.get_config()) == frame
        True
    """

    origin_lat: float
    origin_lon: float
    epsg: int

    @classmethod
    def at(cls, lat: float, lon: float) -> LocalFrame:
        """The frame in the UTM zone containing ``(lat, lon)``, origin there."""
        epsg = utm_epsg(lat, lon)
        return cls(origin_lat=float(lat), origin_lon=float(lon), epsg=epsg)

    @cached_property
    def _transformers(self) -> tuple[Any, Any, float, float]:
        try:
            import pyproj
        except ImportError as err:
            raise MissingExtraError(
                "LocalFrame transforms through pyproj; install it with "
                "`pip install 'plumax[io]'`"
            ) from err
        forward = pyproj.Transformer.from_crs(4326, self.epsg, always_xy=True)
        inverse = pyproj.Transformer.from_crs(self.epsg, 4326, always_xy=True)
        # (E₀, N₀) = UTM(φ₀, λ₀)
        origin_easting, origin_northing = forward.transform(
            self.origin_lon, self.origin_lat
        )
        return forward, inverse, float(origin_easting), float(origin_northing)

    def to_xy(
        self,
        lat: Float[np.ndarray, "*#points"] | float,
        lon: Float[np.ndarray, "*#points"] | float,
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Frame metres ``(x, y)`` of points given by latitude φ and longitude λ."""
        forward, _, origin_easting, origin_northing = self._transformers
        lat_array, lon_array = np.broadcast_arrays(
            np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        )
        # (E, N) = UTM(φ, λ)
        easting, northing = forward.transform(lon_array, lat_array)
        # x = E − E₀
        x = np.asarray(np.asarray(easting) - origin_easting)
        # y = N − N₀
        y = np.asarray(np.asarray(northing) - origin_northing)
        return x, y

    def to_latlon(
        self,
        x: Float[np.ndarray, "*#points"] | float,
        y: Float[np.ndarray, "*#points"] | float,
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Latitude φ and longitude λ of points given in frame metres."""
        _, inverse, origin_easting, origin_northing = self._transformers
        x_array, y_array = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
        # E = x + E₀
        easting = x_array + origin_easting
        # N = y + N₀
        northing = y_array + origin_northing
        # (φ, λ) = UTM⁻¹(E, N)
        lon, lat = inverse.transform(easting, northing)
        return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)

    def east_north_axes(
        self,
        lat: Float[np.ndarray, "*#points"] | float,
        lon: Float[np.ndarray, "*#points"] | float,
    ) -> tuple[Float[np.ndarray, "*shape 2"], Float[np.ndarray, "*shape 2"]]:
        """Unit vectors of true east ê and true north n̂, in frame components.

        ê = ∂(x, y)/∂λ / ‖∂(x, y)/∂λ‖,  n̂ = ∂(x, y)/∂φ / ‖∂(x, y)/∂φ‖.

        A wind ``(u_e, v_n)`` has frame components ``u_e ê + v_n n̂``. The
        derivatives are forward differences of :meth:`to_xy` over
        :data:`AXIS_STEP_DEGREES`, so the vectors carry the meridian convergence
        and nothing is assumed about the projection.

        Args:
            lat: Latitudes φ [degrees north].
            lon: Longitudes λ [degrees east], broadcastable with ``lat``.

        Returns:
            ``(east, north)``, each holding the ``(x, y)`` components of a unit
            vector in its last axis.

        Examples:
            Near a zone's central meridian the frame barely rotates; 2.85° east
            of it, at 32° N, grid north leans by about sin φ · Δλ ≈ 1.5°:

            >>> import numpy as np
            >>> frame = LocalFrame.at(31.95, -102.15)
            >>> east, north = frame.east_north_axes(31.95, -102.15)
            >>> east.shape
            (2,)
            >>> lean = np.degrees(np.arctan2(north[0], north[1]))
            >>> bool(1.0 < abs(lean) < 2.0)
            True
        """
        lat_array, lon_array = np.broadcast_arrays(
            np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        )
        x0, y0 = self.to_xy(lat_array, lon_array)
        x_east, y_east = self.to_xy(lat_array, lon_array + AXIS_STEP_DEGREES)
        x_north, y_north = self.to_xy(lat_array + AXIS_STEP_DEGREES, lon_array)
        # ∂(x, y)/∂λ ≈ ((x, y)(φ, λ + δ) − (x, y)(φ, λ)) / δ
        east = einx.id("..., ... -> ... (1 + 1)", x_east - x0, y_east - y0)
        # ∂(x, y)/∂φ ≈ ((x, y)(φ + δ, λ) − (x, y)(φ, λ)) / δ
        north = einx.id("..., ... -> ... (1 + 1)", x_north - x0, y_north - y0)
        # ‖e‖ = √(Σ_c e_c²)
        east_norm = np.sqrt(einx.sum("... [c]", east**2))
        north_norm = np.sqrt(einx.sum("... [c]", north**2))
        # ê = e / ‖e‖
        east_unit = einx.divide("... c, ... -> ... c", east, east_norm)
        # n̂ = n / ‖n‖
        north_unit = einx.divide("... c, ... -> ... c", north, north_norm)
        return east_unit, north_unit

    def grid_indices(
        self,
        lat: Float[np.ndarray, "*#points"] | float,
        lon: Float[np.ndarray, "*#points"] | float,
        grid: PlumeGrid3D,
    ) -> tuple[Int[np.ndarray, "*shape"], Int[np.ndarray, "*shape"]]:
        """Indices ``(i, j)`` of the analysis T-cells containing lat/lon points.

        i = ⌊(x − x₀ + Δx/2) / Δx⌋,  j = ⌊(y − y₀ + Δy/2) / Δy⌋,

        with ``x₀ = grid.x[0]``; points outside the grid get ``-1``.

        Args:
            lat: Latitudes φ [degrees north].
            lon: Longitudes λ [degrees east], broadcastable with ``lat``.
            grid: Analysis grid, in this frame's metres.

        Returns:
            ``(i, j)``: x and y cell indices, ``-1`` outside.

        Examples:
            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> from plumax.les_fvm.grid import make_grid
            >>> frame = LocalFrame.at(31.95, -102.15)
            >>> grid = make_grid(
            ...     (0.0, 400.0, 4), (0.0, 300.0, 4), (0.0, 50.0, 4), dtype=jnp.float64
            ... )
            >>> x, y = np.array([60.0, -10.0]), np.array([140.0, 50.0])
            >>> lat, lon = frame.to_latlon(x, y)
            >>> i, j = frame.grid_indices(lat, lon, grid)
            >>> i.tolist(), j.tolist()
            ([0, -1], [1, -1])
        """
        x, y = self.to_xy(lat, lon)
        grid_x = np.asarray(grid.x, dtype=float)
        grid_y = np.asarray(grid.y, dtype=float)
        # i = ⌊(x − x₀ + Δx/2) / Δx⌋
        i = np.floor((x - grid_x[0] + 0.5 * grid.dx) / grid.dx).astype(int)
        # j = ⌊(y − y₀ + Δy/2) / Δy⌋
        j = np.floor((y - grid_y[0] + 0.5 * grid.dy) / grid.dy).astype(int)
        outside_x = (i < 0) | (i >= grid_x.size)
        outside_y = (j < 0) | (j >= grid_y.size)
        outside = outside_x | outside_y
        return np.where(outside, -1, i), np.where(outside, -1, j)

    def get_config(self) -> dict[str, Any]:
        """The frame's fields, for ``from_config``."""
        return dataclasses.asdict(self)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LocalFrame:
        """The frame described by ``get_config``'s output."""
        return cls(**config)
