"""Horizontal regrid onto analysis points placed by a ``LocalFrame``.

Each analysis T-, U- or V-point (frame metres) is mapped to lat/lon, then to a
fractional index ``(i, j)`` of the source grid, and sampled bilinearly there;
earth-relative winds are projected onto the frame's axes at the same points.
NumPy only: this runs at load time, outside ``jit``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import einx
import numpy as np
from jaxtyping import Float

from plumax._src.errors import DomainError
from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met._src.frames import LocalFrame


#: Frame metres ``(x, y)`` → source fractional indices ``(i, j)``, NaN outside.
IndexMap = Callable[
    [Float[np.ndarray, "y x"], Float[np.ndarray, "y x"]],
    tuple[Float[np.ndarray, "y x"], Float[np.ndarray, "y x"]],
]


@dataclasses.dataclass(frozen=True)
class StaggerPoints:
    """Source indices and true east/north axes at one stagger's analysis points.

    Args:
        i: Fractional index along the source's x (longitude) axis.
        j: Fractional index along the source's y (latitude) axis.
        east: Unit true-east vector ê in frame components.
        north: Unit true-north vector n̂ in frame components.
    """

    i: Float[np.ndarray, "y x"]
    j: Float[np.ndarray, "y x"]
    east: Float[np.ndarray, "y x 2"]
    north: Float[np.ndarray, "y x 2"]


def stagger_points(
    grid: PlumeGrid3D, frame: LocalFrame, to_index: IndexMap, *, loader: str
) -> dict[str, StaggerPoints]:
    """Source indices and axes of the analysis T-, U- and V-points.

    U-points sit at ``x + Δx/2`` and V-points at ``y + Δy/2`` of the T-points
    (the ``les_fvm`` stagger).

    Args:
        grid: Analysis grid, in ``frame`` metres.
        frame: The analysis grid's frame.
        to_index: Frame metres → source fractional indices; NaN outside the
            source domain.
        loader: Loader name for the error message.

    Returns:
        ``{"t": ..., "u": ..., "v": ...}``.

    Raises:
        DomainError: If any point falls outside the source domain.

    Examples:
        An identity map on a 4 × 4 grid of 100 m cells puts the U-points half a
        cell east of the T-points:

        >>> import jax.numpy as jnp
        >>> from plumax.les_fvm.grid import make_grid
        >>> frame = LocalFrame.at(31.95, -102.15)
        >>> grid = make_grid(
        ...     (0.0, 400.0, 4), (0.0, 400.0, 4), (0.0, 40.0, 4), dtype=jnp.float64
        ... )
        >>> points = stagger_points(
        ...     grid, frame, lambda x, y: (x / 100.0, y / 100.0), loader="demo"
        ... )
        >>> points["u"].i[0].tolist()
        [1.0, 2.0, 3.0, 4.0]
        >>> points["t"].east.shape
        (4, 4, 2)
    """
    x = np.asarray(grid.x, dtype=float)
    y = np.asarray(grid.y, dtype=float)
    offsets = {"t": (0.0, 0.0), "u": (0.5 * grid.dx, 0.0), "v": (0.0, 0.5 * grid.dy)}
    stagger: dict[str, StaggerPoints] = {}
    for name, (offset_x, offset_y) in offsets.items():
        point_x, point_y = np.meshgrid(x + offset_x, y + offset_y)
        i, j = to_index(point_x, point_y)
        if not (np.all(np.isfinite(i)) and np.all(np.isfinite(j))):
            raise DomainError(
                f"{loader}: some analysis {name.upper()}-points fall outside the "
                "source domain; shrink the grid or move the frame origin (the "
                "loader does not extrapolate horizontally)"
            )
        lat, lon = frame.to_latlon(point_x, point_y)
        east, north = frame.east_north_axes(lat, lon)
        stagger[name] = StaggerPoints(i=i, j=j, east=east, north=north)
    return stagger


def bilinear_at(
    field: Float[np.ndarray, "*lead src_y src_x"],
    i: Float[np.ndarray, "y x"],
    j: Float[np.ndarray, "y x"],
) -> Float[np.ndarray, "*lead y x"]:
    """Bilinear sample of ``field`` at fractional indices ``(j, i)``.

    f(i, j) = (1−a)(1−b) f₀₀ + a(1−b) f₀₁ + (1−a) b f₁₀ + a b f₁₁,

    with ``a = i − ⌊i⌋``, ``b = j − ⌊j⌋`` and ``f_{rc}`` the four corners. The
    lower corner is clipped into the grid, so ``i = n_x − 1`` uses the last cell.

    Args:
        field: Source values, last two axes (y, x).
        i: Fractional index along the source's x axis.
        j: Fractional index along the source's y axis.

    Returns:
        ``field`` at the points, leading axes kept.

    Examples:
        Exact for a field linear in both indices, with leading axes kept:

        >>> import numpy as np
        >>> rows, cols = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
        >>> field = np.stack([2.0 * cols + rows, cols])
        >>> sample = bilinear_at(field, np.array([[0.5, 3.0]]), np.array([[1.25, 2.0]]))
        >>> sample.shape
        (2, 1, 2)
        >>> sample[0].tolist()
        [[2.25, 8.0]]
    """
    n_rows, n_cols = field.shape[-2:]
    i_low = np.clip(np.floor(i).astype(int), 0, n_cols - 2)
    j_low = np.clip(np.floor(j).astype(int), 0, n_rows - 2)
    i_high = i_low + 1
    j_high = j_low + 1
    # a = i − ⌊i⌋
    frac_x = i - i_low
    # b = j − ⌊j⌋
    frac_y = j - j_low
    gather = "... [src_y src_x], y x, y x -> ... y x"
    corner_00 = einx.get_at(gather, field, j_low, i_low)
    corner_01 = einx.get_at(gather, field, j_low, i_high)
    corner_10 = einx.get_at(gather, field, j_high, i_low)
    corner_11 = einx.get_at(gather, field, j_high, i_high)
    # (1−a)(1−b) f₀₀
    term_00 = (1.0 - frac_x) * (1.0 - frac_y) * corner_00
    # a (1−b) f₀₁
    term_01 = frac_x * (1.0 - frac_y) * corner_01
    # (1−a) b f₁₀
    term_10 = (1.0 - frac_x) * frac_y * corner_10
    # a b f₁₁
    term_11 = frac_x * frac_y * corner_11
    return term_00 + term_01 + term_10 + term_11


def frame_wind(
    u_east: Float[np.ndarray, "time level src_y src_x"],
    v_north: Float[np.ndarray, "time level src_y src_x"],
    points: dict[str, StaggerPoints],
) -> tuple[Float[np.ndarray, "time level y x"], Float[np.ndarray, "time level y x"]]:
    """Frame-axis wind at U- and V-points from earth-relative source columns.

    u = u_e ê_x + v_n n̂_x  (at U-points),  v = u_e ê_y + v_n n̂_y  (at V-points).

    Args:
        u_east: Eastward wind u_e on the analysis levels, on source columns.
        v_north: Northward wind v_n, same layout.
        points: Output of :func:`stagger_points`.

    Returns:
        ``(u, v)``: frame-x wind at U-points and frame-y wind at V-points.

    Examples:
        With unrotated axes the wind passes through unchanged:

        >>> import numpy as np
        >>> axes = StaggerPoints(
        ...     i=np.zeros((1, 1)), j=np.zeros((1, 1)),
        ...     east=np.array([[[1.0, 0.0]]]), north=np.array([[[0.0, 1.0]]]),
        ... )
        >>> u, v = frame_wind(
        ...     np.full((1, 1, 2, 2), 3.0), np.full((1, 1, 2, 2), -1.0),
        ...     {"u": axes, "v": axes},
        ... )
        >>> float(u.squeeze()), float(v.squeeze())
        (3.0, -1.0)
    """
    at_u, at_v = points["u"], points["v"]
    u_east_at_u = bilinear_at(u_east, at_u.i, at_u.j)
    v_north_at_u = bilinear_at(v_north, at_u.i, at_u.j)
    u_east_at_v = bilinear_at(u_east, at_v.i, at_v.j)
    v_north_at_v = bilinear_at(v_north, at_v.i, at_v.j)
    east_x_at_u, _ = einx.id("y x c -> c y x", at_u.east)
    north_x_at_u, _ = einx.id("y x c -> c y x", at_u.north)
    _, east_y_at_v = einx.id("y x c -> c y x", at_v.east)
    _, north_y_at_v = einx.id("y x c -> c y x", at_v.north)
    # u = u_e ê_x + v_n n̂_x
    u = u_east_at_u * east_x_at_u + v_north_at_u * north_x_at_u
    # v = u_e ê_y + v_n n̂_y
    v = u_east_at_v * east_y_at_v + v_north_at_v * north_y_at_v
    return u, v
