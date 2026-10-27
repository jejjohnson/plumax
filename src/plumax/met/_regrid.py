"""Shared horizontal regrid onto analysis points placed by a ``LocalFrame``.

Loaders bring a source grid's columns onto the analysis grid's T-, U- and
V-points. With a :class:`~plumax.met.frames.LocalFrame` the analysis
points are frame metres, so each point is mapped to lat/lon, then to a
fractional index ``(i, j)`` of the source grid (``i`` along x / longitude,
``j`` along y / latitude), and sampled bilinearly there. Earth-relative
winds are projected onto the frame's axes at the same points. NumPy only:
this runs at load time, outside ``jit``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met.frames import LocalFrame


#: Maps frame metres ``(x, y)`` to source fractional indices ``(i, j)``.
IndexMap = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class StaggerPoints:
    """Fractional source indices and true-north axes of one stagger.

    Attributes:
        i: Fractional index along the source's x axis, shape ``(ny, nx)``.
        j: Fractional index along the source's y axis, shape ``(ny, nx)``.
        east: Unit true-east vector in frame components, ``(ny, nx, 2)``.
        north: Unit true-north vector in frame components, ``(ny, nx, 2)``.
    """

    i: np.ndarray
    j: np.ndarray
    east: np.ndarray
    north: np.ndarray


def stagger_points(
    grid: PlumeGrid3D, frame: LocalFrame, to_index: IndexMap, *, loader: str
) -> dict[str, StaggerPoints]:
    """Source indices of the analysis T-, U- and V-points.

    U-points sit at ``x + dx/2`` and V-points at ``y + dy/2`` of the
    T-points (the ``les_fvm`` stagger).

    Args:
        grid: Analysis grid, in ``frame`` metres.
        frame: The analysis grid's frame.
        to_index: Frame metres → source fractional indices; NaN outside
            the source domain.
        loader: Loader name for the error message.

    Returns:
        ``{"t": ..., "u": ..., "v": ...}``.

    Raises:
        ValueError: If any point falls outside the source domain.
    """
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    offsets = {"t": (0.0, 0.0), "u": (0.5 * grid.dx, 0.0), "v": (0.0, 0.5 * grid.dy)}
    out: dict[str, StaggerPoints] = {}
    for name, (ox, oy) in offsets.items():
        xx, yy = np.meshgrid(x + ox, y + oy)  # (ny, nx)
        i, j = to_index(xx, yy)
        if not (np.all(np.isfinite(i)) and np.all(np.isfinite(j))):
            raise ValueError(
                f"{loader}: some analysis {name.upper()}-points fall outside the "
                "source domain; shrink the grid or move the frame origin (the "
                "loader does not extrapolate horizontally)"
            )
        lat, lon = frame.to_latlon(xx, yy)
        east, north = frame.east_north_axes(lat, lon)
        out[name] = StaggerPoints(i=i, j=j, east=east, north=north)
    return out


def bilinear_at(field: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``field[..., j, i]`` at 2-D fractional indices.

    Args:
        field: Source values ``(..., n_j, n_i)``.
        i: Fractional index along the last axis, shape ``(ny, nx)``.
        j: Fractional index along the second-to-last axis, ``(ny, nx)``.

    Returns:
        ``field`` sampled at ``(j, i)``, shape ``(..., ny, nx)``.
    """
    n_j, n_i = field.shape[-2:]
    i0 = np.clip(np.floor(i).astype(int), 0, n_i - 2)
    j0 = np.clip(np.floor(j).astype(int), 0, n_j - 2)
    wi, wj = i - i0, j - j0
    return (
        (1.0 - wj) * (1.0 - wi) * field[..., j0, i0]
        + (1.0 - wj) * wi * field[..., j0, i0 + 1]
        + wj * (1.0 - wi) * field[..., j0 + 1, i0]
        + wj * wi * field[..., j0 + 1, i0 + 1]
    )


def frame_wind(
    u_east: np.ndarray, v_north: np.ndarray, points: dict[str, StaggerPoints]
) -> tuple[np.ndarray, np.ndarray]:
    """Frame-axis wind at U- and V-points from earth-relative columns.

    With ``ê`` and ``n̂`` the unit true-east and true-north vectors in frame
    components, the wind ``u_e ê + v_n n̂`` has x component
    ``u_e ê_x + v_n n̂_x`` (taken at U-points) and y component
    ``u_e ê_y + v_n n̂_y`` (taken at V-points).

    Args:
        u_east: Eastward wind already on the analysis levels,
            ``(T, nz, n_j, n_i)`` on source columns.
        v_north: Northward wind, same shape.
        points: Output of :func:`stagger_points`.

    Returns:
        ``(u, v)``: frame-x wind at U-points and frame-y wind at V-points,
        each ``(T, nz, ny, nx)``.
    """
    p_u, p_v = points["u"], points["v"]
    u = (
        bilinear_at(u_east, p_u.i, p_u.j) * p_u.east[..., 0]
        + bilinear_at(v_north, p_u.i, p_u.j) * p_u.north[..., 0]
    )
    v = (
        bilinear_at(u_east, p_v.i, p_v.j) * p_v.east[..., 1]
        + bilinear_at(v_north, p_v.i, p_v.j) * p_v.north[..., 1]
    )
    return u, v
