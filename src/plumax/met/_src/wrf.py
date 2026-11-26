"""``wrfout`` reader onto a ``les_fvm`` analysis grid.

Reads the WRF variables the transport tiers need, destaggers them to mass
points, converts WRF's perturbation fields to physical units, and
interpolates — vertically from WRF's terrain-following mass levels to the
analysis grid's height-above-ground levels, then horizontally (bilinear)
onto the analysis T-, U- and V-points — so the result is already on the
``les_fvm`` stagger.

Horizontal frame (v1)
---------------------
WRF mass points form a regular ``DX`` × ``DY`` grid; this loader treats
that grid as a local Cartesian frame whose origin is the south-west mass
point.  The analysis grid is placed in that frame by ``origin``: an
analysis-grid coordinate ``(x, y)`` sits at WRF metres ``(x + origin[0],
y + origin[1])``, and the wind is grid-relative.

Lat/lon frame
-------------
Passing a :class:`~plumax.met.LocalFrame` as ``origin`` places the
analysis grid by lat/lon instead, so its axes are the frame's (UTM) axes
rather than WRF's. Each analysis point is mapped to WRF fractional
indices through the file's ``XLAT`` / ``XLONG``: the mass points are
projected into the frame and the bilinear map of each cell of them is
inverted, so WRF's own projection is never rebuilt and any
``MAP_PROJ`` works. The earth-relative wind (which needs ``COSALPHA`` /
``SINALPHA``) is projected onto the frame's axes at each point.

Vertical
--------
Analysis-grid ``z`` is height above ground (``make_grid`` treats ``z_min``
as the surface).  WRF heights above ground come from the geopotential
``(PH + PHB) / g`` on the staggered levels, averaged to mass levels, minus
terrain ``HGT`` — so terrain-following levels are handled per column.
Below the lowest mass level and above the highest the nearest level is
held (constant extrapolation).

Reading
-------
The file is read through :func:`xrops.atm.open_wrfout`, which destaggers
to mass points, rebuilds height above ground from the geopotential and the
terrain, converts WRF's perturbation fields and decodes the time axis. It
needs the ``io`` extra (``pip install 'plumax[io]'``). This module owns
only what is specific to the analysis grid: the vertical and horizontal
interpolation onto it, and a grid-relative wind.

All of this is NumPy: it is IO-side, not on any gradient path, and WRF
files are large.  Only the final fields become JAX arrays.
"""

from __future__ import annotations

from pathlib import Path

import einx
import jax.numpy as jnp
import numpy as np
import xarray as xr
from functax.geometry import CurvilinearGrid
from functax.interp import locate_curvilinear
from jaxtyping import Float

from plumax._src.constants import (
    KAPPA as KAPPA,
    P_REFERENCE as P_REFERENCE,
    WRF_GRAVITY,
)
from plumax._src.errors import (
    MissingVariableError,
    TimeAxisError,
)
from plumax._src.extras import require_extra
from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met._src.frames import LocalFrame
from plumax.met._src.regrid import (
    IndexMap,
    analysis_heights,
    interpolate_vertical,
    onto_analysis_grid,
    stagger_points,
)
from plumax.met._src.time import TimeAxis
from plumax.met.field import MetField


#: Gravitational acceleration used by WRF for the geopotential [m s⁻²].
GRAVITY = WRF_GRAVITY
#: WRF base-state potential temperature added to the perturbation ``T`` [K].
THETA_BASE = 300.0


def load_wrf(
    path: str | Path,
    *,
    plume_grid: PlumeGrid3D,
    origin: tuple[float, float] | LocalFrame = (0.0, 0.0),
    times: slice | None = None,
) -> MetField:
    """Load a ``wrfout`` file onto an analysis grid.

    Args:
        path: NetCDF ``wrfout`` file.
        plume_grid: Analysis grid (from :func:`plumax.les_fvm.make_grid`).
        origin: WRF-frame metres of the analysis grid's ``(x=0, y=0)``,
            or a :class:`~plumax.met.LocalFrame` the grid is laid
            out in; see the module docstring.
        times: Optional ``slice`` over the file's ``Time`` axis.

    Returns:
        A :class:`MetField` on ``plume_grid``.

    Raises:
        MissingExtraError: If the ``io`` extra (xrops) is not installed.
        TimeAxisError: If ``times`` selects nothing.
        MissingVariableError: With a ``LocalFrame``, if the file lacks
            ``XLAT`` / ``XLONG`` or the ``COSALPHA`` / ``SINALPHA`` its
            earth-relative wind needs.
        DomainError: With a ``LocalFrame``, if an analysis point falls outside
            the WRF domain.
        ValueError: With metre offsets, if an analysis point falls outside
            the WRF mass-point domain (the loader never extrapolates
            horizontally); or if the file has no usable time axis (``Times``,
            or ``XTIME`` with the ``SIMULATION_START_DATE`` attribute).
    """
    in_frame = isinstance(origin, LocalFrame)
    open_wrfout = _require_open_wrfout()

    # --- read ---
    rotation = ["COSALPHA", "SINALPHA"]
    with xr.open_dataset(path, decode_times=False) as raw:
        has_rotation = all(name in raw for name in rotation)
        selected_steps = range(raw.sizes["Time"])[times or slice(None)]
    if len(selected_steps) == 0:
        raise TimeAxisError("load_wrf: `times` selects no time steps")
    # xrops destaggers to mass points, rebuilds height above ground and
    # temperature, and decodes the time axis.
    with open_wrfout(
        path, times=times, variables=rotation if has_rotation else None
    ) as dataset:
        spacing_x = float(dataset.attrs["DX"])
        spacing_y = float(dataset.attrs["DY"])
        if in_frame:
            u_mass, v_mass = _earth_relative_wind(dataset)
            lat_mass, lon_mass = _latlon(dataset)
        else:
            u_mass, v_mass = _grid_relative_wind(dataset)
        w_mass = dataset["w"].values
        temperature = dataset["temperature"].values
        pressure = dataset["pressure"].values
        height_agl = np.broadcast_to(dataset["z_agl"].values, u_mass.shape)
        pbl_height = dataset["pbl_height"].values
        clock = TimeAxis.from_datetimes(dataset["time"].values)

    # --- vertical targets ---
    # The analysis grid's lower boundary is its surface (``make_grid``
    # semantics), and WRF heights are above ground.
    target_heights = analysis_heights(plume_grid)
    dtype = plume_grid.x.dtype
    as_jax = lambda array: jnp.asarray(array, dtype=dtype)

    if in_frame:
        # --- horizontal, by lat/lon ---
        to_index = _wrf_index_map(lat_mass, lon_mass, origin)
        points = stagger_points(plume_grid, origin, to_index, loader="load_wrf")
        return onto_analysis_grid(
            plume_grid,
            points,
            height_agl,
            clock,
            u_east=u_mass,
            v_north=v_mass,
            wind_up=w_mass,
            temperature=temperature,
            pressure=pressure,
            pbl_height=pbl_height,
        )

    # --- horizontal, by WRF metres ---
    num_y_wrf, num_x_wrf = u_mass.shape[-2:]
    x_t = np.asarray(plume_grid.x, dtype=np.float64) + origin[0]
    y_t = np.asarray(plume_grid.y, dtype=np.float64) + origin[1]
    # x_U = x_T + Δx/2
    x_u = x_t + 0.5 * plume_grid.dx
    # y_V = y_T + Δy/2
    y_v = y_t + 0.5 * plume_grid.dy
    _check_inside(
        x_u,
        y_v,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        num_x=num_x_wrf,
        num_y=num_y_wrf,
    )
    _check_inside(
        x_t,
        y_t,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        num_x=num_x_wrf,
        num_y=num_y_wrf,
    )

    def onto(field: np.ndarray, easting: np.ndarray, northing: np.ndarray):
        column = interpolate_vertical(height_agl, field, target_heights)
        # (fx, fy) = (x / DX, y / DY)
        return _bilinear(column, easting / spacing_x, northing / spacing_y)

    return MetField(
        plume_grid=plume_grid,
        times=as_jax(clock.seconds),
        u=as_jax(onto(u_mass, x_u, y_t)),
        v=as_jax(onto(v_mass, x_t, y_v)),
        w=as_jax(onto(w_mass, x_t, y_t)),
        temperature=as_jax(onto(temperature, x_t, y_t)),
        pressure=as_jax(onto(pressure, x_t, y_t)),
        pbl_height=as_jax(_bilinear(pbl_height, x_t / spacing_x, y_t / spacing_y)),
        t0=clock.t0_iso,
    )


def _require_open_wrfout():
    """``xrops.atm.open_wrfout``, or an error naming the ``io`` extra."""
    xrops_atm = require_extra(
        "xrops.atm", extra="io", caller="load_wrf", purpose="reads through xrops"
    )
    return xrops_atm.open_wrfout


def _earth_relative_wind(
    dataset: xr.Dataset,
) -> tuple[Float[np.ndarray, "time level y x"], Float[np.ndarray, "time level y x"]]:
    """Mass-point eastward and northward wind ``(u_e, v_n)``.

    Raises:
        MissingVariableError: If the file has no ``COSALPHA`` / ``SINALPHA``, so
            only WRF's grid-relative wind is known.
    """
    wind_east, wind_north = dataset["u"], dataset["v"]
    if wind_east.attrs.get("grid_relative") == "true":
        raise MissingVariableError(
            "load_wrf: placing the grid by a LocalFrame needs the earth-relative "
            "wind, and this file has no COSALPHA / SINALPHA to rotate WRF's "
            "grid-relative U / V; pass `origin=(x, y)` in WRF metres instead"
        )
    return wind_east.values, wind_north.values


def _latlon(
    dataset: xr.Dataset,
) -> tuple[Float[np.ndarray, "y x"], Float[np.ndarray, "y x"]]:
    """Static mass-point latitude and longitude (first step on a moving nest).

    Raises:
        MissingVariableError: If the file has no ``XLAT`` / ``XLONG``.
    """
    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        raise MissingVariableError(
            "load_wrf: placing the grid by a LocalFrame needs XLAT / XLONG in the file"
        )
    lat, lon = dataset["lat"], dataset["lon"]
    # a moving nest: v1 places the grid where the nest starts
    if "time" in lat.dims:
        lat, lon = lat.isel(time=0), lon.isel(time=0)
    return lat.values, lon.values


#: Index slack [cells] within which a target counts as on the edge of the WRF
#: mass points: the inverse of an edge cell's bilinear map returns about −1e-14
#: on its first row or column in float64, and −5e-6 in float32.
_EDGE_SLACK: float = 1e-4


def _wrf_index_map(
    lat: Float[np.ndarray, "src_y src_x"],
    lon: Float[np.ndarray, "src_y src_x"],
    frame: LocalFrame,
) -> IndexMap:
    """Frame metres → WRF fractional mass-point indices ``(i, j)``.

    (i, j) = 𝓘(x, y), the inverse of the bilinear map of each cell of mass
    points in frame coordinates (`functax.interp.locate_curvilinear`).

    WRF's own projection is never rebuilt, so any ``MAP_PROJ`` works. The
    index is exact where the WRF-to-frame map is affine and second-order
    accurate in the grid spacing elsewhere. A point outside the quadrilateral
    domain of the mass points comes back NaN; a triangulation of the points
    would also have accepted the sliver of their convex hull beyond an edge.
    """
    # (x, y) of every mass point in the frame
    mass_x, mass_y = frame.to_xy(lat, lon)
    grid = CurvilinearGrid(x=jnp.asarray(mass_x), y=jnp.asarray(mass_y))
    # the largest row and column index of a mass point
    last_index = np.array(lat.shape, dtype=float) - 1.0

    def to_index(
        frame_x: Float[np.ndarray, "y x"], frame_y: Float[np.ndarray, "y x"]
    ) -> tuple[Float[np.ndarray, "y x"], Float[np.ndarray, "y x"]]:
        targets = einx.id("y x, y x -> (y x) (1 + 1)", frame_x, frame_y)
        # u = (r + s, c + t), with x(s, t) = x_* in cell (r, c)
        cell, offset, _ = locate_curvilinear(grid, jnp.asarray(targets))
        position = np.asarray(cell + offset)
        # inside the mass points, with a slack of _EDGE_SLACK cells: a target on
        # the first row or column comes back at about −1e-14 (−5e-6 in float32)
        inside = einx.all(
            "t c -> t",
            (position >= -_EDGE_SLACK) & (position <= last_index + _EDGE_SLACK),
        )
        # NaN where the target is outside, and the edge targets onto the edge
        position = einx.where(
            "t, t c, -> t c", inside, np.clip(position, 0.0, last_index), np.nan
        )
        num_y, num_x = frame_x.shape
        # j along the rows of mass points, i along the columns
        index_y, index_x = einx.id("(y x) c -> c y x", position, y=num_y, x=num_x)
        return index_x, index_y

    return to_index


def _grid_relative_wind(
    dataset: xr.Dataset,
) -> tuple[Float[np.ndarray, "time level y x"], Float[np.ndarray, "time level y x"]]:
    """Mass-point ``(u, v)`` along the WRF grid axes.

    u = u_e cos α + v_n sin α,  v = v_n cos α − u_e sin α.

    The analysis grid sits in WRF's grid frame (see the module docstring), so
    the wind must be grid-relative. ``open_wrfout`` rotates to earth-relative
    (``u_e = u cos α − v sin α``, ``v_n = v cos α + u sin α``) when the file
    carries ``COSALPHA`` / ``SINALPHA``; the inverse rotation above undoes it.
    """
    wind_east, wind_north = dataset["u"], dataset["v"]
    if wind_east.attrs.get("grid_relative") == "true":
        return wind_east.values, wind_north.values
    cos_alpha, sin_alpha = dataset["COSALPHA"], dataset["SINALPHA"]
    # u = u_e cos α + v_n sin α
    u_grid = wind_east * cos_alpha + wind_north * sin_alpha
    # v = v_n cos α − u_e sin α
    v_grid = wind_north * cos_alpha - wind_east * sin_alpha
    return (
        u_grid.transpose(*wind_east.dims).values,
        v_grid.transpose(*wind_north.dims).values,
    )


def _check_inside(
    x: np.ndarray,
    y: np.ndarray,
    *,
    spacing_x: float,
    spacing_y: float,
    num_x: int,
    num_y: int,
) -> None:
    # x_max = (n_x − 1) DX
    x_max = (num_x - 1) * spacing_x
    # y_max = (n_y − 1) DY
    y_max = (num_y - 1) * spacing_y
    if x.min() < 0.0 or x.max() > x_max or y.min() < 0.0 or y.max() > y_max:
        raise ValueError(
            "load_wrf: the analysis grid (including its U-/V-point half-cell "
            f"offsets) spans x ∈ [{x.min():g}, {x.max():g}] m, y ∈ [{y.min():g}, "
            f"{y.max():g}] m in the WRF frame, but the WRF mass points cover "
            f"x ∈ [0, {x_max:g}] m, y ∈ [0, {y_max:g}] m. Shrink the grid or "
            "move `origin`; the loader does not extrapolate horizontally."
        )


def _bilinear(field: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``field[..., j, i]`` at fractional indices.

    ``fx`` (``(nx,)``) and ``fy`` (``(ny,)``) are index-space coordinates
    (WRF metres divided by ``DX`` / ``DY``); returns ``field`` resampled to
    ``(..., ny, nx)``.
    """
    num_y, num_x = field.shape[-2:]
    col_low = np.clip(np.floor(fx).astype(int), 0, num_x - 2)
    row_low = np.clip(np.floor(fy).astype(int), 0, num_y - 2)
    # a = fx − ⌊fx⌋
    weight_x = (fx - col_low)[None, :]
    # b = fy − ⌊fy⌋
    weight_y = (fy - row_low)[:, None]
    rows, cols = row_low[:, None], col_low[None, :]
    # (1−b)(1−a) f₀₀ + (1−b) a f₀₁ + b (1−a) f₁₀ + b a f₁₁
    return (
        (1.0 - weight_y) * (1.0 - weight_x) * field[..., rows, cols]
        + (1.0 - weight_y) * weight_x * field[..., rows, cols + 1]
        + weight_y * (1.0 - weight_x) * field[..., rows + 1, cols]
        + weight_y * weight_x * field[..., rows + 1, cols + 1]
    )
