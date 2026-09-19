"""``wrfout`` reader onto a ``les_fvm`` analysis grid.

The WRF-specific work — decoding ``Times`` / ``XTIME``, destaggering,
turning the perturbation fields into temperature and pressure, and
reconstructing height above ground per column — is `xrtoolz`_'s
(:func:`xrtoolz.atm.open_wrfout`), as is the per-column vertical remap
(:func:`xrtoolz.transforms.remap_axis` with ``source_coords="z_agl"``).
This module keeps only what is plumax-specific: placing the analysis grid
in the WRF frame, the ``les_fvm`` C-grid stagger, and the
:class:`~plumax.met.field.MetField` assembly. See the *xrtoolz boundary*
design page (``docs/design/00a_xrtoolz_boundary.md``).

.. _xrtoolz: https://github.com/jejjohnson/xrtoolz

Horizontal frame (v1)
---------------------
WRF mass points form a regular ``DX`` × ``DY`` grid; this loader treats
that grid as a local Cartesian frame whose origin is the south-west mass
point.  The analysis grid is placed in that frame by ``origin``: an
analysis-grid coordinate ``(x, y)`` sits at WRF metres ``(x + origin[0],
y + origin[1])``.  Because the analysis axes *are* the WRF grid axes, the
wind components are kept **grid-relative** (WRF's raw ``U`` / ``V``
destaggered to mass points) rather than rotated to earth-relative east /
north — on a rotated map projection the two differ, and it is the
grid-relative pair that drives a solver aligned with this frame.
Projection-aware placement through a lat/lon frame is the
coordinate-frames issue (plumax#79); nothing here changes when it lands
except how ``origin`` is derived.

Vertical
--------
Analysis-grid ``z`` is height above ground (``make_grid`` treats ``z_min``
as the surface).  WRF heights above ground come from the geopotential
``(PH + PHB) / g`` on the staggered levels, averaged to mass levels, minus
terrain ``HGT`` (the opener's ``z_agl`` coordinate), so terrain-following
levels are handled per column.  Below the lowest mass level and above the
highest the nearest level is held (``extrapolate="nearest"``).

All of this is xarray / NumPy: it is IO-side, not on any gradient path, and
WRF files are large.  Only the final fields become JAX arrays.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met.field import MetField


if TYPE_CHECKING:
    import xarray as xr


#: Gravitational acceleration used by WRF for the geopotential [m s⁻²].
GRAVITY = 9.81
#: WRF reference pressure for potential temperature [Pa].
P_REFERENCE = 1.0e5
#: WRF base-state potential temperature added to the perturbation ``T`` [K].
THETA_BASE = 300.0
#: Dry-air gas constant over specific heat at constant pressure.
KAPPA = 287.04 / 1004.5


def _require_xrtoolz() -> tuple[ModuleType, ModuleType]:
    """Import ``xrtoolz.atm`` / ``xrtoolz.transforms`` or explain the extra."""
    try:
        import xrtoolz.atm as atm
        import xrtoolz.transforms as transforms
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "plumax.met.load_wrf reads wrfout files through xrtoolz, which is "
            "an optional dependency: install it with `pip install 'plumax[data]'` "
            "(or `uv sync --extra data`)."
        ) from exc
    return atm, transforms


def load_wrf(
    path: str | Path,
    *,
    plume_grid: PlumeGrid3D,
    origin: tuple[float, float] = (0.0, 0.0),
    times: slice | None = None,
) -> MetField:
    """Load a ``wrfout`` file onto an analysis grid.

    Requires the ``data`` extra (``xrtoolz``).

    Args:
        path: NetCDF ``wrfout`` file.
        plume_grid: Analysis grid (from :func:`plumax.les_fvm.make_grid`).
        origin: WRF-frame metres of the analysis grid's ``(x=0, y=0)``;
            see the module docstring.
        times: Optional ``slice`` over the file's ``Time`` axis.

    Returns:
        A :class:`MetField` on ``plume_grid``.

    Raises:
        ImportError: If ``xrtoolz`` is not installed.
        ValueError: If any analysis T-, U- or V-point falls outside the
            WRF mass-point domain (the loader never extrapolates
            horizontally), or if ``times`` selects nothing.
    """
    atm, transforms = _require_xrtoolz()

    # Analysis-grid sample points in the WRF frame. The U- and V-points sit
    # half a cell east / north of the T-points (les_fvm C-grid stagger).
    x_t = np.asarray(plume_grid.x, dtype=np.float64) + origin[0]
    y_t = np.asarray(plume_grid.y, dtype=np.float64) + origin[1]
    x_u = x_t + 0.5 * plume_grid.dx
    y_v = y_t + 0.5 * plume_grid.dy
    # WRF heights are above ground; the analysis grid's lower boundary is its
    # surface (``make_grid`` semantics), so interpolate at heights above
    # ``z_min = z[0] - dz/2`` rather than at the raw ``z`` coordinates.
    z_abs = np.asarray(plume_grid.z, dtype=np.float64)
    z_t = z_abs - (z_abs[0] - 0.5 * plume_grid.dz)

    # The raw staggered ``U`` / ``V`` are passed through for the
    # grid-relative wind (see the module docstring); the opener's own ``u`` /
    # ``v`` are earth-relative wherever the file carries COSALPHA / SINALPHA.
    with atm.open_wrfout(path, variables=["U", "V"]) as ds:
        if times is not None:
            ds = ds.isel(time=times)
        if ds.sizes["time"] == 0:
            raise ValueError("load_wrf: `times` selects no time steps")
        seconds, t0 = _time_axis(ds["time"].values)
        _check_inside(x_u, y_v, ds)
        _check_inside(x_t, y_t, ds)

        u_m = atm.destagger(ds["U"], "x_stag").assign_coords(x=ds["x"])
        v_m = atm.destagger(ds["V"], "y_stag").assign_coords(y=ds["y"])

        def onto(field: xr.DataArray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            column = transforms.remap_axis(
                field.assign_coords(z_agl=ds["z_agl"]),
                source_dim="level",
                target_coords=z_t,
                target_name="height",
                source_coords="z_agl",
                extrapolate="nearest",
            )
            return _sample(column, x, y)

        dtype = plume_grid.x.dtype
        as_jax = lambda a: jnp.asarray(a, dtype=dtype)
        return MetField(
            plume_grid=plume_grid,
            times=as_jax(seconds),
            u=as_jax(onto(u_m, x_u, y_t)),
            v=as_jax(onto(v_m, x_t, y_v)),
            w=as_jax(onto(ds["w"], x_t, y_t)),
            temperature=as_jax(onto(ds["temperature"], x_t, y_t)),
            pressure=as_jax(onto(ds["pressure"], x_t, y_t)),
            pbl_height=as_jax(_sample(ds["pbl_height"], x_t, y_t)),
            t0=t0,
        )


def _time_axis(stamps: np.ndarray) -> tuple[np.ndarray, str]:
    """Seconds since the first selected time, plus that time's stamp.

    ``stamps`` is the opener's ``datetime64`` time axis (``Times`` rows, or
    ``XTIME`` plus the simulation start when the file has no ``Times``).
    """
    stamps = np.asarray(stamps, dtype="datetime64[s]")
    seconds = (stamps - stamps[0]) / np.timedelta64(1, "s")
    t0 = str(stamps[0]).replace("T", "_")
    return np.asarray(seconds, dtype=np.float64), t0


def _check_inside(x: np.ndarray, y: np.ndarray, ds: xr.Dataset) -> None:
    x_max = float(ds["x"].max())
    y_max = float(ds["y"].max())
    if x.min() < 0.0 or x.max() > x_max or y.min() < 0.0 or y.max() > y_max:
        raise ValueError(
            "load_wrf: the analysis grid (including its U-/V-point half-cell "
            f"offsets) spans x ∈ [{x.min():g}, {x.max():g}] m, y ∈ [{y.min():g}, "
            f"{y.max():g}] m in the WRF frame, but the WRF mass points cover "
            f"x ∈ [0, {x_max:g}] m, y ∈ [0, {y_max:g}] m. Shrink the grid or "
            "move `origin`; the loader does not extrapolate horizontally."
        )


def _sample(field: xr.DataArray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``field(..., y, x)`` at the analysis points.

    ``xarray.interp`` on the two horizontal coordinates is a bilinear
    interpolation on WRF's regular mass-point grid; the result is returned
    with ``(..., ny, nx)`` trailing axes as ``MetField`` expects.
    """
    out = field.interp(x=x, y=y, method="linear", assume_sorted=True)
    return np.asarray(out.transpose(..., "y", "x").values, dtype=np.float64)
