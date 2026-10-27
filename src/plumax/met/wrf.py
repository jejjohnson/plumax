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
projected into the frame and a piecewise-linear interpolant (Delaunay)
inverts them, so WRF's own projection is never rebuilt and any
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
The file is read through :func:`xrtoolz.atm.open_wrfout`, which destaggers
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
from jaxtyping import Float

from plumax._src.errors import (
    MissingExtraError,
    MissingVariableError,
    TimeAxisError,
)
from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met._src.frames import LocalFrame
from plumax.met._src.regrid import bilinear_at, frame_wind, stagger_points
from plumax.met._src.time import TimeAxis
from plumax.met.field import MetField


#: Gravitational acceleration used by WRF for the geopotential [m s⁻²].
GRAVITY = 9.81
#: WRF reference pressure for potential temperature [Pa].
P_REFERENCE = 1.0e5
#: WRF base-state potential temperature added to the perturbation ``T`` [K].
THETA_BASE = 300.0
#: Dry-air gas constant over specific heat at constant pressure.
KAPPA = 287.04 / 1004.5


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
        MissingExtraError: If the ``io`` extra (xrtoolz) is not installed.
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
    # xrtoolz destaggers to mass points, rebuilds height above ground and
    # temperature, and decodes the time axis.
    with open_wrfout(
        path, times=times, variables=rotation if has_rotation else None
    ) as ds:
        dx_wrf = float(ds.attrs["DX"])
        dy_wrf = float(ds.attrs["DY"])
        if in_frame:
            u_mass, v_mass = _earth_relative_wind(ds)
            lat_mass, lon_mass = _latlon(ds)
        else:
            u_mass, v_mass = _grid_relative_wind(ds)
        w_mass = ds["w"].values
        temperature = ds["temperature"].values
        pressure = ds["pressure"].values
        height_agl = np.broadcast_to(ds["z_agl"].values, u_mass.shape)
        pbl_height = ds["pbl_height"].values
        clock = TimeAxis.from_datetimes(ds["time"].values)

    # --- vertical targets ---
    # The analysis grid's lower boundary is its surface (``make_grid``
    # semantics), and WRF heights are above ground.
    grid_z = np.asarray(plume_grid.z, dtype=np.float64)
    # z_surface = z₀ − Δz/2
    grid_surface = grid_z[0] - 0.5 * plume_grid.dz
    # h_k = z_k − z_surface
    target_heights = grid_z - grid_surface
    dtype = plume_grid.x.dtype
    as_jax = lambda a: jnp.asarray(a, dtype=dtype)

    if in_frame:
        # --- horizontal, by lat/lon ---
        to_index = _wrf_index_map(lat_mass, lon_mass, origin)
        points = stagger_points(plume_grid, origin, to_index, loader="load_wrf")
        at_t = points["t"]
        on_levels = lambda f: _interpolate_vertical(height_agl, f, target_heights)
        u_frame, v_frame = frame_wind(on_levels(u_mass), on_levels(v_mass), points)
        w_t = bilinear_at(on_levels(w_mass), at_t.i, at_t.j)
        temperature_t = bilinear_at(on_levels(temperature), at_t.i, at_t.j)
        pressure_t = bilinear_at(on_levels(pressure), at_t.i, at_t.j)
        pbl_height_t = bilinear_at(pbl_height, at_t.i, at_t.j)
        return MetField(
            plume_grid=plume_grid,
            times=as_jax(clock.seconds),
            u=as_jax(u_frame),
            v=as_jax(v_frame),
            w=as_jax(w_t),
            temperature=as_jax(temperature_t),
            pressure=as_jax(pressure_t),
            pbl_height=as_jax(pbl_height_t),
            t0=clock.t0_iso,
        )

    # --- horizontal, by WRF metres ---
    n_y_w, n_x_w = u_mass.shape[-2:]
    x_t = np.asarray(plume_grid.x, dtype=np.float64) + origin[0]
    y_t = np.asarray(plume_grid.y, dtype=np.float64) + origin[1]
    x_u = x_t + 0.5 * plume_grid.dx
    y_v = y_t + 0.5 * plume_grid.dy
    _check_inside(x_u, y_v, dx_w=dx_wrf, dy_w=dy_wrf, n_x_w=n_x_w, n_y_w=n_y_w)
    _check_inside(x_t, y_t, dx_w=dx_wrf, dy_w=dy_wrf, n_x_w=n_x_w, n_y_w=n_y_w)

    def onto(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        column = _interpolate_vertical(height_agl, field, target_heights)
        return _bilinear(column, x / dx_wrf, y / dy_wrf)

    return MetField(
        plume_grid=plume_grid,
        times=as_jax(clock.seconds),
        u=as_jax(onto(u_mass, x_u, y_t)),
        v=as_jax(onto(v_mass, x_t, y_v)),
        w=as_jax(onto(w_mass, x_t, y_t)),
        temperature=as_jax(onto(temperature, x_t, y_t)),
        pressure=as_jax(onto(pressure, x_t, y_t)),
        pbl_height=as_jax(_bilinear(pbl_height, x_t / dx_wrf, y_t / dy_wrf)),
        t0=clock.t0_iso,
    )


def _require_open_wrfout():
    """``xrtoolz.atm.open_wrfout``, or an error naming the ``io`` extra."""
    try:
        import xrtoolz.atm
    except ImportError as err:
        raise MissingExtraError(
            "load_wrf reads through xrtoolz; install it with `pip install 'plumax[io]'`"
        ) from err
    return xrtoolz.atm.open_wrfout


def _earth_relative_wind(
    ds: xr.Dataset,
) -> tuple[Float[np.ndarray, "time level y x"], Float[np.ndarray, "time level y x"]]:
    """Mass-point eastward and northward wind ``(u_e, v_n)``.

    Raises:
        MissingVariableError: If the file has no ``COSALPHA`` / ``SINALPHA``, so
            only WRF's grid-relative wind is known.
    """
    u, v = ds["u"], ds["v"]
    if u.attrs.get("grid_relative") == "true":
        raise MissingVariableError(
            "load_wrf: placing the grid by a LocalFrame needs the earth-relative "
            "wind, and this file has no COSALPHA / SINALPHA to rotate WRF's "
            "grid-relative U / V; pass `origin=(x, y)` in WRF metres instead"
        )
    return u.values, v.values


def _latlon(
    ds: xr.Dataset,
) -> tuple[Float[np.ndarray, "y x"], Float[np.ndarray, "y x"]]:
    """Static mass-point latitude and longitude (first step on a moving nest).

    Raises:
        MissingVariableError: If the file has no ``XLAT`` / ``XLONG``.
    """
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise MissingVariableError(
            "load_wrf: placing the grid by a LocalFrame needs XLAT / XLONG in the file"
        )
    lat, lon = ds["lat"], ds["lon"]
    # a moving nest: v1 places the grid where the nest starts
    if "time" in lat.dims:
        lat, lon = lat.isel(time=0), lon.isel(time=0)
    return lat.values, lon.values


def _wrf_index_map(
    lat: Float[np.ndarray, "src_y src_x"],
    lon: Float[np.ndarray, "src_y src_x"],
    frame: LocalFrame,
):
    """Frame metres → WRF fractional mass-point indices ``(i, j)``.

    (i, j) = 𝓘(x, y),  𝓘 piecewise linear on the Delaunay triangulation of the
    mass points' frame coordinates.

    WRF's own projection is never rebuilt, so any ``MAP_PROJ`` works. 𝓘 is
    exact where the WRF-to-frame map is affine and second-order accurate in
    the grid spacing elsewhere; points outside the WRF domain come back NaN.
    """
    import scipy.interpolate

    # (x, y) of every mass point in the frame
    mass_x, mass_y = frame.to_xy(lat, lon)
    n_rows, n_cols = lat.shape
    rows, cols = np.meshgrid(
        np.arange(n_rows, dtype=float), np.arange(n_cols, dtype=float), indexing="ij"
    )
    nodes = einx.id("y x, y x -> (y x) (1 + 1)", mass_x, mass_y)
    node_indices = einx.id("y x, y x -> (y x) (1 + 1)", cols, rows)
    interpolant = scipy.interpolate.LinearNDInterpolator(nodes, node_indices)

    def to_index(
        x: Float[np.ndarray, "y x"], y: Float[np.ndarray, "y x"]
    ) -> tuple[Float[np.ndarray, "y x"], Float[np.ndarray, "y x"]]:
        queries = einx.id("y x, y x -> (y x) (1 + 1)", x, y)
        # (i, j) = 𝓘(x, y)
        indices = interpolant(queries)
        n_y, n_x = x.shape
        i, j = einx.id("(y x) c -> c y x", indices, y=n_y, x=n_x)
        return i, j

    return to_index


def _grid_relative_wind(
    ds: xr.Dataset,
) -> tuple[Float[np.ndarray, "time level y x"], Float[np.ndarray, "time level y x"]]:
    """Mass-point ``(u, v)`` along the WRF grid axes.

    u = u_e cos α + v_n sin α,  v = v_n cos α − u_e sin α.

    The analysis grid sits in WRF's grid frame (see the module docstring), so
    the wind must be grid-relative. ``open_wrfout`` rotates to earth-relative
    (``u_e = u cos α − v sin α``, ``v_n = v cos α + u sin α``) when the file
    carries ``COSALPHA`` / ``SINALPHA``; the inverse rotation above undoes it.
    """
    u, v = ds["u"], ds["v"]
    if u.attrs.get("grid_relative") == "true":
        return u.values, v.values
    cos_alpha, sin_alpha = ds["COSALPHA"], ds["SINALPHA"]
    # u = u_e cos α + v_n sin α
    u_grid = u * cos_alpha + v * sin_alpha
    # v = v_n cos α − u_e sin α
    v_grid = v * cos_alpha - u * sin_alpha
    return u_grid.transpose(*u.dims).values, v_grid.transpose(*v.dims).values


def _check_inside(
    x: np.ndarray, y: np.ndarray, *, dx_w: float, dy_w: float, n_x_w: int, n_y_w: int
) -> None:
    x_max = (n_x_w - 1) * dx_w
    y_max = (n_y_w - 1) * dy_w
    if x.min() < 0.0 or x.max() > x_max or y.min() < 0.0 or y.max() > y_max:
        raise ValueError(
            "load_wrf: the analysis grid (including its U-/V-point half-cell "
            f"offsets) spans x ∈ [{x.min():g}, {x.max():g}] m, y ∈ [{y.min():g}, "
            f"{y.max():g}] m in the WRF frame, but the WRF mass points cover "
            f"x ∈ [0, {x_max:g}] m, y ∈ [0, {y_max:g}] m. Shrink the grid or "
            "move `origin`; the loader does not extrapolate horizontally."
        )


def _interpolate_vertical(
    z_src: np.ndarray, field: np.ndarray, z_target: np.ndarray
) -> np.ndarray:
    """Per-column linear interpolation from levels ``z_src`` to ``z_target``.

    ``z_src`` and ``field`` are ``(n_time, n_lev, ny, nx)`` with ``z_src``
    increasing along the level axis; returns ``(n_time, nz, ny, nx)``.
    Targets outside a column's range take that column's end value.
    """
    n_lev = z_src.shape[1]
    out = np.empty((field.shape[0], z_target.shape[0], *field.shape[2:]), field.dtype)
    for k, zk in enumerate(z_target):
        hi = np.clip((z_src <= zk).sum(axis=1), 1, n_lev - 1)[:, None]
        lo = hi - 1
        z_lo = np.take_along_axis(z_src, lo, axis=1)[:, 0]
        z_hi = np.take_along_axis(z_src, hi, axis=1)[:, 0]
        f_lo = np.take_along_axis(field, lo, axis=1)[:, 0]
        f_hi = np.take_along_axis(field, hi, axis=1)[:, 0]
        frac = np.clip((zk - z_lo) / (z_hi - z_lo), 0.0, 1.0)
        out[:, k] = (1.0 - frac) * f_lo + frac * f_hi
    return out


def _bilinear(field: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``field[..., j, i]`` at fractional indices.

    ``fx`` (``(nx,)``) and ``fy`` (``(ny,)``) are index-space coordinates
    (WRF metres divided by ``DX`` / ``DY``); returns ``field`` resampled to
    ``(..., ny, nx)``.
    """
    n_y, n_x = field.shape[-2:]
    i0 = np.clip(np.floor(fx).astype(int), 0, n_x - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, n_y - 2)
    wx = (fx - i0)[None, :]
    wy = (fy - j0)[:, None]
    jj, ii = j0[:, None], i0[None, :]
    return (
        (1.0 - wy) * (1.0 - wx) * field[..., jj, ii]
        + (1.0 - wy) * wx * field[..., jj, ii + 1]
        + wy * (1.0 - wx) * field[..., jj + 1, ii]
        + wy * wx * field[..., jj + 1, ii + 1]
    )
