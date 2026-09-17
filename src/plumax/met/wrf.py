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
y + origin[1])``.  Projection-aware placement through a lat/lon frame is
the coordinate-frames issue (plumax#79); nothing here changes when it
lands except how ``origin`` is derived.

Vertical
--------
Analysis-grid ``z`` is height above ground (``make_grid`` treats ``z_min``
as the surface).  WRF heights above ground come from the geopotential
``(PH + PHB) / g`` on the staggered levels, averaged to mass levels, minus
terrain ``HGT`` — so terrain-following levels are handled per column.
Below the lowest mass level and above the highest the nearest level is
held (constant extrapolation).

All of this is NumPy: it is IO-side, not on any gradient path, and WRF
files are large.  Only the final fields become JAX arrays.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import xarray as xr

from plumax.les_fvm.grid import PlumeGrid3D
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
    origin: tuple[float, float] = (0.0, 0.0),
    times: slice | None = None,
) -> MetField:
    """Load a ``wrfout`` file onto an analysis grid.

    Args:
        path: NetCDF ``wrfout`` file.
        plume_grid: Analysis grid (from :func:`plumax.les_fvm.make_grid`).
        origin: WRF-frame metres of the analysis grid's ``(x=0, y=0)``;
            see the module docstring.
        times: Optional ``slice`` over the file's ``Time`` axis.

    Returns:
        A :class:`MetField` on ``plume_grid``.

    Raises:
        ValueError: If any analysis T-, U- or V-point falls outside the
            WRF mass-point domain (the loader never extrapolates
            horizontally), or if ``times`` selects nothing.
    """
    with xr.open_dataset(path) as ds:
        if times is not None:
            ds = ds.isel(Time=times)
        if ds.sizes["Time"] == 0:
            raise ValueError("load_wrf: `times` selects no time steps")
        dx_w = float(ds.attrs["DX"])
        dy_w = float(ds.attrs["DY"])
        u_stag = ds["U"].values
        v_stag = ds["V"].values
        w_stag = ds["W"].values
        theta_pert = ds["T"].values
        pressure = ds["P"].values + ds["PB"].values
        geopotential = ds["PH"].values + ds["PHB"].values
        terrain = ds["HGT"].values
        pblh = ds["PBLH"].values
        seconds, t0 = _time_axis(ds)

    # Destagger to mass points (WRF stores U/V/W on the faces).
    u_m = 0.5 * (u_stag[..., :-1] + u_stag[..., 1:])
    v_m = 0.5 * (v_stag[..., :-1, :] + v_stag[..., 1:, :])
    w_m = 0.5 * (w_stag[:, :-1] + w_stag[:, 1:])
    z_stag = geopotential / GRAVITY
    z_mass = 0.5 * (z_stag[:, :-1] + z_stag[:, 1:])
    if terrain.ndim == 3:  # WRF writes HGT with a Time axis
        terrain = terrain[:, None]
    z_agl = z_mass - terrain
    temperature = (theta_pert + THETA_BASE) * (pressure / P_REFERENCE) ** KAPPA

    n_y_w, n_x_w = u_m.shape[-2:]
    x_t = np.asarray(plume_grid.x, dtype=np.float64) + origin[0]
    y_t = np.asarray(plume_grid.y, dtype=np.float64) + origin[1]
    x_u = x_t + 0.5 * plume_grid.dx
    y_v = y_t + 0.5 * plume_grid.dy
    z_t = np.asarray(plume_grid.z, dtype=np.float64)
    _check_inside(x_u, y_v, dx_w=dx_w, dy_w=dy_w, n_x_w=n_x_w, n_y_w=n_y_w)
    _check_inside(x_t, y_t, dx_w=dx_w, dy_w=dy_w, n_x_w=n_x_w, n_y_w=n_y_w)

    def onto(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        column = _interpolate_vertical(z_agl, field, z_t)
        return _bilinear(column, x / dx_w, y / dy_w)

    dtype = plume_grid.x.dtype
    as_jax = lambda a: jnp.asarray(a, dtype=dtype)
    return MetField(
        plume_grid=plume_grid,
        times=as_jax(seconds),
        u=as_jax(onto(u_m, x_u, y_t)),
        v=as_jax(onto(v_m, x_t, y_v)),
        w=as_jax(onto(w_m, x_t, y_t)),
        temperature=as_jax(onto(temperature, x_t, y_t)),
        pressure=as_jax(onto(pressure, x_t, y_t)),
        pbl_height=as_jax(_bilinear(pblh, x_t / dx_w, y_t / dy_w)),
        t0=t0,
    )


def _time_axis(ds: xr.Dataset) -> tuple[np.ndarray, str]:
    """Seconds since the first selected time, plus that time's stamp."""
    if "XTIME" in ds:
        minutes = np.asarray(ds["XTIME"].values, dtype=np.float64)
        seconds = (minutes - minutes[0]) * 60.0
    else:
        seconds = np.arange(ds.sizes["Time"], dtype=np.float64)
    t0 = ""
    if "Times" in ds:
        first = ds["Times"].values[0]
        raw = first.tobytes() if isinstance(first, np.ndarray) else first
        t0 = raw.decode() if isinstance(raw, bytes) else str(raw)
    return seconds, t0.strip("\x00 ")


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
