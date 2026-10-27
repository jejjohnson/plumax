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
Passing a :class:`~plumax.met.frames.LocalFrame` as ``origin`` places the
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

import jax.numpy as jnp
import numpy as np
import xarray as xr

from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met._regrid import bilinear_at, frame_wind, stagger_points
from plumax.met.field import MetField
from plumax.met.frames import LocalFrame
from plumax.met.time import TimeAxis


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
            or a :class:`~plumax.met.frames.LocalFrame` the grid is laid
            out in; see the module docstring.
        times: Optional ``slice`` over the file's ``Time`` axis.

    Returns:
        A :class:`MetField` on ``plume_grid``.

    Raises:
        ImportError: If the ``io`` extra (xrtoolz) is not installed.
        ValueError: If any analysis T-, U- or V-point falls outside the
            WRF mass-point domain (the loader never extrapolates
            horizontally), if ``times`` selects nothing, or if the file
            has no usable time axis (``Times``, or ``XTIME`` with the
            ``SIMULATION_START_DATE`` attribute). With a ``LocalFrame``,
            also if the file lacks ``XLAT`` / ``XLONG`` or the
            ``COSALPHA`` / ``SINALPHA`` its earth-relative wind needs.
    """
    in_frame = isinstance(origin, LocalFrame)
    open_wrfout = _require_open_wrfout()
    rotation = ["COSALPHA", "SINALPHA"]
    with xr.open_dataset(path, decode_times=False) as raw:
        has_rotation = all(name in raw for name in rotation)
        n_selected = len(range(raw.sizes["Time"])[times or slice(None)])
    if n_selected == 0:
        raise ValueError("load_wrf: `times` selects no time steps")
    # xrtoolz destaggers to mass points, rebuilds height above ground and
    # temperature, and decodes the time axis.
    with open_wrfout(
        path, times=times, variables=rotation if has_rotation else None
    ) as ds:
        dx_w = float(ds.attrs["DX"])
        dy_w = float(ds.attrs["DY"])
        if in_frame:
            u_m, v_m = _earth_relative_wind(ds)
            lat_w, lon_w = _latlon(ds)
        else:
            u_m, v_m = _grid_relative_wind(ds)
        w_m = ds["w"].values
        temperature = ds["temperature"].values
        pressure = ds["pressure"].values
        z_agl = np.broadcast_to(ds["z_agl"].values, u_m.shape)
        pblh = ds["pbl_height"].values
        clock = TimeAxis.from_datetimes(ds["time"].values)

    # WRF heights are above ground; the analysis grid's lower boundary is its
    # surface (``make_grid`` semantics), so interpolate at heights above
    # ``z_min = z[0] - dz/2`` rather than at the raw ``z`` coordinates.
    z_abs = np.asarray(plume_grid.z, dtype=np.float64)
    z_t = z_abs - (z_abs[0] - 0.5 * plume_grid.dz)
    dtype = plume_grid.x.dtype
    as_jax = lambda a: jnp.asarray(a, dtype=dtype)

    if in_frame:
        points = stagger_points(
            plume_grid, origin, _wrf_index_map(lat_w, lon_w, origin), loader="load_wrf"
        )
        t = points["t"]
        columns = lambda f: _interpolate_vertical(z_agl, f, z_t)
        u_f, v_f = frame_wind(columns(u_m), columns(v_m), points)
        return MetField(
            plume_grid=plume_grid,
            times=as_jax(clock.seconds),
            u=as_jax(u_f),
            v=as_jax(v_f),
            w=as_jax(bilinear_at(columns(w_m), t.i, t.j)),
            temperature=as_jax(bilinear_at(columns(temperature), t.i, t.j)),
            pressure=as_jax(bilinear_at(columns(pressure), t.i, t.j)),
            pbl_height=as_jax(bilinear_at(pblh, t.i, t.j)),
            t0=clock.t0_iso,
        )

    n_y_w, n_x_w = u_m.shape[-2:]
    x_t = np.asarray(plume_grid.x, dtype=np.float64) + origin[0]
    y_t = np.asarray(plume_grid.y, dtype=np.float64) + origin[1]
    x_u = x_t + 0.5 * plume_grid.dx
    y_v = y_t + 0.5 * plume_grid.dy
    _check_inside(x_u, y_v, dx_w=dx_w, dy_w=dy_w, n_x_w=n_x_w, n_y_w=n_y_w)
    _check_inside(x_t, y_t, dx_w=dx_w, dy_w=dy_w, n_x_w=n_x_w, n_y_w=n_y_w)

    def onto(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        column = _interpolate_vertical(z_agl, field, z_t)
        return _bilinear(column, x / dx_w, y / dy_w)

    return MetField(
        plume_grid=plume_grid,
        times=as_jax(clock.seconds),
        u=as_jax(onto(u_m, x_u, y_t)),
        v=as_jax(onto(v_m, x_t, y_v)),
        w=as_jax(onto(w_m, x_t, y_t)),
        temperature=as_jax(onto(temperature, x_t, y_t)),
        pressure=as_jax(onto(pressure, x_t, y_t)),
        pbl_height=as_jax(_bilinear(pblh, x_t / dx_w, y_t / dy_w)),
        t0=clock.t0_iso,
    )


def _require_open_wrfout():
    """``xrtoolz.atm.open_wrfout``, or an error naming the ``io`` extra."""
    try:
        from xrtoolz.atm import open_wrfout
    except ImportError as err:
        raise ImportError(
            "load_wrf reads through xrtoolz; install it with `pip install 'plumax[io]'`"
        ) from err
    return open_wrfout


def _earth_relative_wind(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Mass-point eastward / northward wind, or an error if only grid-relative."""
    u, v = ds["u"], ds["v"]
    if u.attrs.get("grid_relative") == "true":
        raise ValueError(
            "load_wrf: placing the grid by a LocalFrame needs the earth-relative "
            "wind, and this file has no COSALPHA / SINALPHA to rotate WRF's "
            "grid-relative U / V; pass `origin=(x, y)` in WRF metres instead"
        )
    return u.transpose(*u.dims).values, v.transpose(*v.dims).values


def _latlon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Static 2-D mass-point latitude and longitude."""
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError(
            "load_wrf: placing the grid by a LocalFrame needs XLAT / XLONG in the file"
        )
    lat, lon = ds["lat"], ds["lon"]
    if "time" in lat.dims:  # a moving nest: v1 places the grid at the first step
        lat, lon = lat.isel(time=0), lon.isel(time=0)
    return lat.values, lon.values


def _wrf_index_map(lat: np.ndarray, lon: np.ndarray, frame: LocalFrame):
    """Frame metres → WRF fractional mass-point indices ``(i, j)``.

    The mass points' lat/lon are projected into ``frame`` and a
    piecewise-linear interpolant over that scattered set (Delaunay, via
    :class:`scipy.interpolate.LinearNDInterpolator`) returns fractional
    ``(i, j)``; points outside the WRF domain come back NaN. The
    interpolant is exact where the WRF-to-frame map is affine and
    second-order accurate in the grid spacing elsewhere.
    """
    from scipy.interpolate import LinearNDInterpolator

    x_w, y_w = frame.to_xy(lat, lon)
    jj, ii = np.meshgrid(
        np.arange(lat.shape[0], dtype=float),
        np.arange(lat.shape[1], dtype=float),
        indexing="ij",
    )
    interpolant = LinearNDInterpolator(
        np.column_stack([x_w.ravel(), y_w.ravel()]),
        np.column_stack([ii.ravel(), jj.ravel()]),
    )

    def to_index(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = interpolant(np.column_stack([x.ravel(), y.ravel()]))
        return out[:, 0].reshape(x.shape), out[:, 1].reshape(x.shape)

    return to_index


def _grid_relative_wind(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Mass-point ``(u, v)`` along the WRF grid axes.

    The analysis grid sits in WRF's grid frame (see the module docstring),
    so the wind must be grid-relative. ``open_wrfout`` rotates to
    earth-relative when the file carries ``COSALPHA`` / ``SINALPHA``
    (``u_e = u cos α − v sin α``, ``v_e = v cos α + u sin α``); the inverse
    rotation ``u = u_e cos α + v_e sin α``, ``v = v_e cos α − u_e sin α``
    undoes it.
    """
    u, v = ds["u"], ds["v"]
    if u.attrs.get("grid_relative") == "true":
        return u.values, v.values
    cos_a, sin_a = ds["COSALPHA"], ds["SINALPHA"]
    u_g = u * cos_a + v * sin_a
    v_g = v * cos_a - u * sin_a
    return u_g.transpose(*u.dims).values, v_g.transpose(*v.dims).values


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
