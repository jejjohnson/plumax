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
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from plumax._src.errors import DomainError
from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met._src.frames import LocalFrame
from plumax.met._src.time import TimeAxis
from plumax.met.field import MetField


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
    grid: PlumeGrid3D, frame: LocalFrame | None, to_index: IndexMap, *, loader: str
) -> dict[str, StaggerPoints]:
    """Source indices and axes of the analysis T-, U- and V-points.

    U-points sit at ``x + Δx/2`` and V-points at ``y + Δy/2`` of the T-points
    (the ``les_fvm`` stagger).

    Args:
        grid: Analysis grid, in ``frame`` metres.
        frame: The analysis grid's frame, or ``None`` for a grid laid out on
            the source's own axes: east and north are then ``(1, 0)`` and
            ``(0, 1)``, so :func:`frame_wind` passes the wind through.
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
    grid_x = np.asarray(grid.x, dtype=float)
    grid_y = np.asarray(grid.y, dtype=float)
    offsets = {"t": (0.0, 0.0), "u": (0.5 * grid.dx, 0.0), "v": (0.0, 0.5 * grid.dy)}
    stagger: dict[str, StaggerPoints] = {}
    for name, (offset_x, offset_y) in offsets.items():
        point_x, point_y = np.meshgrid(grid_x + offset_x, grid_y + offset_y)
        index_x, index_y = to_index(point_x, point_y)
        if not (np.all(np.isfinite(index_x)) and np.all(np.isfinite(index_y))):
            raise DomainError(
                f"{loader}: some analysis {name.upper()}-points fall outside the "
                "source domain; shrink the grid or move the frame origin (the "
                "loader does not extrapolate horizontally)"
            )
        if frame is None:
            # ê = (1, 0), n̂ = (0, 1) at every point
            num_y, num_x = point_x.shape
            east = einx.id("c -> y x c", np.array([1.0, 0.0]), y=num_y, x=num_x)
            north = einx.id("c -> y x c", np.array([0.0, 1.0]), y=num_y, x=num_x)
        else:
            lat, lon = frame.to_latlon(point_x, point_y)
            east, north = frame.east_north_axes(lat, lon)
        stagger[name] = StaggerPoints(i=index_x, j=index_y, east=east, north=north)
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
    num_rows, num_cols = field.shape[-2:]
    col_low = np.clip(np.floor(i).astype(int), 0, num_cols - 2)
    row_low = np.clip(np.floor(j).astype(int), 0, num_rows - 2)
    col_high = col_low + 1
    row_high = row_low + 1
    # a = i − ⌊i⌋
    frac_x = i - col_low
    # b = j − ⌊j⌋
    frac_y = j - row_low
    gather = "... [src_y src_x], y x, y x -> ... y x"
    corner_00 = einx.get_at(gather, field, row_low, col_low)
    corner_01 = einx.get_at(gather, field, row_low, col_high)
    corner_10 = einx.get_at(gather, field, row_high, col_low)
    corner_11 = einx.get_at(gather, field, row_high, col_high)
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
    wind_x = u_east_at_u * east_x_at_u + v_north_at_u * north_x_at_u
    # v = u_e ê_y + v_n n̂_y
    wind_y = u_east_at_v * east_y_at_v + v_north_at_v * north_y_at_v
    return wind_x, wind_y


def analysis_heights(plume_grid: PlumeGrid3D) -> Float[np.ndarray, " level"]:
    """Heights above ground of the analysis grid's levels.

    The analysis grid's lower boundary is its surface (``make_grid``
    semantics), a half cell below the first level.

    Returns:
        h_k = z_k − (z₀ − Δz/2) [m], one per level.
    """
    grid_z = np.asarray(plume_grid.z, dtype=np.float64)
    # z_surface = z₀ − Δz/2
    grid_surface = grid_z[0] - 0.5 * plume_grid.dz
    # h_k = z_k − z_surface
    return grid_z - grid_surface


def interpolate_vertical(
    z_src: np.ndarray, field: np.ndarray, z_target: np.ndarray
) -> np.ndarray:
    """Per-column linear interpolation from levels ``z_src`` to ``z_target``.

    ``z_src`` and ``field`` are ``(n_time, n_lev, ny, nx)`` with ``z_src``
    increasing along the level axis; returns ``(n_time, nz, ny, nx)``.
    Targets outside a column's range take that column's end value.
    """
    num_levels = z_src.shape[1]
    out = np.empty((field.shape[0], z_target.shape[0], *field.shape[2:]), field.dtype)
    for target, target_height in enumerate(z_target):
        upper = np.clip((z_src <= target_height).sum(axis=1), 1, num_levels - 1)[
            :, None
        ]
        lower = upper - 1
        z_lower = np.take_along_axis(z_src, lower, axis=1)[:, 0]
        z_upper = np.take_along_axis(z_src, upper, axis=1)[:, 0]
        field_lower = np.take_along_axis(field, lower, axis=1)[:, 0]
        field_upper = np.take_along_axis(field, upper, axis=1)[:, 0]
        # θ = clip((z_k − z_lo) / (z_hi − z_lo), 0, 1)
        frac = np.clip((target_height - z_lower) / (z_upper - z_lower), 0.0, 1.0)
        # (1 − θ) f_lo + θ f_hi
        out[:, target] = (1.0 - frac) * field_lower + frac * field_upper
    return out


def onto_analysis_grid(
    plume_grid: PlumeGrid3D,
    points: dict[str, StaggerPoints],
    height_agl: Float[np.ndarray, "time source_level row col"],
    clock: TimeAxis,
    *,
    u_east: Float[np.ndarray, "time source_level row col"],
    v_north: Float[np.ndarray, "time source_level row col"],
    wind_up: Float[np.ndarray, "time source_level row col"],
    temperature: Float[np.ndarray, "time source_level row col"],
    pressure: Float[np.ndarray, "time source_level row col"],
    pbl_height: Float[np.ndarray, "time row col"],
) -> MetField:
    """A met source's fields on the analysis grid, as a :class:`MetField`.

    Each field is interpolated in height to the analysis levels, then sampled
    bilinearly at the analysis points; the earth-relative wind is projected
    onto the frame's axes at the U- and V-points (:func:`frame_wind`).

    Args:
        plume_grid: The analysis grid.
        points: The source indices of the T-, U- and V-points
            (:func:`stagger_points`).
        height_agl: Height above ground of each source level [m].
        clock: The source's time axis.
        u_east: Eastward wind on the source's mass points [m s⁻¹].
        v_north: Northward wind on the source's mass points [m s⁻¹].
        wind_up: Vertical wind [m s⁻¹].
        temperature: Air temperature [K].
        pressure: Air pressure [Pa].
        pbl_height: Boundary-layer height [m].

    Returns:
        The met field, in the analysis grid's dtype.
    """
    target_heights = analysis_heights(plume_grid)
    at_t = points["t"]

    def on_levels(field: np.ndarray) -> np.ndarray:
        return interpolate_vertical(height_agl, field, target_heights)

    u_frame, v_frame = frame_wind(on_levels(u_east), on_levels(v_north), points)
    dtype = plume_grid.x.dtype

    def as_jax(array: np.ndarray):
        return jnp.asarray(array, dtype=dtype)

    return MetField(
        plume_grid=plume_grid,
        times=as_jax(clock.seconds),
        u=as_jax(u_frame),
        v=as_jax(v_frame),
        w=as_jax(bilinear_at(on_levels(wind_up), at_t.i, at_t.j)),
        temperature=as_jax(bilinear_at(on_levels(temperature), at_t.i, at_t.j)),
        pressure=as_jax(bilinear_at(on_levels(pressure), at_t.i, at_t.j)),
        pbl_height=as_jax(bilinear_at(pbl_height, at_t.i, at_t.j)),
        t0=clock.t0_iso,
    )
