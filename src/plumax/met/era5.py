"""ERA5 reanalysis onto a ``les_fvm`` analysis grid placed by a ``LocalFrame``.

WRF is regional; where no WRF run exists, ERA5 is the met. This loader
produces the same :class:`~plumax.met.field.MetField` as
:func:`~plumax.met.wrf.load_wrf`, so downstream code never branches on the
source.

Inputs
------
Two NetCDF files as the Climate Data Store returns them (download them with
xrreader's ``reanalysis-era5-pressure-levels`` and
``reanalysis-era5-single-levels`` presets):

- pressure levels: ``u``, ``v`` [m s⁻¹], ``w`` (ω) [Pa s⁻¹], ``t`` [K] and,
  optionally, ``q`` [kg kg⁻¹] on ``(time, level, lat, lon)``;
- single levels: ``sp`` (surface pressure) [Pa] and ``blh`` (boundary-layer
  height) [m] on ``(time, lat, lon)``.

``valid_time`` / ``pressure_level`` / ``latitude`` / ``longitude`` are
accepted as the CDS spells them.

Vertical
--------
Height above ground of each pressure level comes from the hypsometric
equation integrated up from the surface pressure
(:func:`xrtoolz.atm.hypsometric_height`, with ``q`` when present). ERA5
extrapolates fields below ground, so levels with ``p > p_s`` get negative
heights; they stay in the column and only the analysis heights (all above
ground) are sampled. Vertical velocity is converted from ω with the
hydrostatic relation ``w = −ω / (ρ g)``, ``ρ = p / (R_d T)``.

Horizontal
----------
ERA5 is on a regular lat/lon grid, so each analysis point's fractional
index is ``(lon − lon₀) / Δλ`` and ``(lat − lat₀) / Δφ`` (after sorting
latitude ascending and putting the point's longitude in the file's
convention); fields are sampled bilinearly and the earth-relative wind is
projected onto the frame's axes (see :mod:`plumax.met._regrid`).

Needs the ``io`` extra (``pip install 'plumax[io]'``). NumPy only; the
final fields become JAX arrays.
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
from plumax.met.wrf import _interpolate_vertical


#: Standard gravity [m s⁻²], as xrtoolz's hypsometric integration uses.
GRAVITY = 9.80665
#: Dry-air gas constant [J kg⁻¹ K⁻¹].
R_DRY = 287.04

#: CDS spellings → the names this loader uses.
_RENAMES = {
    "valid_time": "time",
    "pressure_level": "level",
    "latitude": "lat",
    "longitude": "lon",
}


def load_era5(
    pressure_levels: str | Path,
    single_levels: str | Path,
    *,
    plume_grid: PlumeGrid3D,
    frame: LocalFrame,
    times: slice | None = None,
) -> MetField:
    """Load ERA5 pressure- and single-level files onto an analysis grid.

    Args:
        pressure_levels: NetCDF with ``u``, ``v``, ``w``, ``t`` (and
            optionally ``q``) on pressure levels in hPa.
        single_levels: NetCDF with ``sp`` and ``blh`` at the same times.
        plume_grid: Analysis grid, in ``frame`` metres.
        frame: Frame the analysis grid is laid out in.
        times: Optional ``slice`` over the shared time axis.

    Returns:
        A :class:`MetField` on ``plume_grid``.

    Raises:
        ImportError: If the ``io`` extra (xrtoolz) is not installed.
        KeyError: If a required variable is missing.
        ValueError: If the two files' times differ, ``times`` selects
            nothing, or an analysis point falls outside the ERA5 domain.
    """
    hypsometric_height = _require_hypsometric_height()
    with (
        xr.open_dataset(pressure_levels) as pl_raw,
        xr.open_dataset(single_levels) as sl_raw,
    ):
        pl, sl = _normalise(pl_raw), _normalise(sl_raw)
        _require(pl, ("u", "v", "w", "t"), pressure_levels)
        _require(sl, ("sp", "blh"), single_levels)
        if not np.array_equal(pl["time"].values, sl["time"].values):
            raise ValueError("load_era5: the two files cover different times")
        if not (
            np.array_equal(pl["lat"].values, sl["lat"].values)
            and np.array_equal(pl["lon"].values, sl["lon"].values)
        ):
            raise ValueError(
                "load_era5: the two files are on different lat/lon grids; request "
                "both with the same `area` and `grid` so `sp` and `blh` line up "
                "with the pressure levels"
            )
        if times is not None:
            pl, sl = pl.isel(time=times), sl.isel(time=times)
        if pl.sizes["time"] == 0:
            raise ValueError("load_era5: `times` selects no time steps")
        # Surface first, so heights increase along `level`.
        pl = pl.sortby("level", ascending=False)

        pressure = (100.0 * pl["level"]).broadcast_like(pl["t"])
        columns = pl.assign(pressure=pressure, sp=sl["sp"])
        z_agl = hypsometric_height(
            columns,
            pressure="pressure",
            temperature="t",
            specific_humidity="q" if "q" in pl else None,
            surface_pressure="sp",
            level="level",
            name="z_agl",
        )["z_agl"]
        dims = ("time", "level", "lat", "lon")
        u_e = pl["u"].transpose(*dims).values
        v_n = pl["v"].transpose(*dims).values
        temperature = pl["t"].transpose(*dims).values
        p = pressure.transpose(*dims).values
        rho = p / (R_DRY * temperature)
        w = -pl["w"].transpose(*dims).values / (rho * GRAVITY)
        z = z_agl.transpose(*dims).values
        pblh = sl["blh"].transpose("time", "lat", "lon").values
        lat = pl["lat"].values
        lon = pl["lon"].values
        clock = TimeAxis.from_datetimes(pl["time"].values)

    if _is_periodic(lon):
        # Close the seam: append the first longitude column after the last,
        # so points between the last column and 360° interpolate across it.
        seam = lambda f: np.concatenate([f, f[..., :1]], axis=-1)
        u_e, v_n, temperature, p, w, z, pblh = map(
            seam, (u_e, v_n, temperature, p, w, z, pblh)
        )

    points = stagger_points(
        plume_grid, frame, _regular_index_map(lat, lon, frame), loader="load_era5"
    )
    z_abs = np.asarray(plume_grid.z, dtype=np.float64)
    z_t = z_abs - (z_abs[0] - 0.5 * plume_grid.dz)
    on_levels = lambda f: _interpolate_vertical(z, f, z_t)
    u_f, v_f = frame_wind(on_levels(u_e), on_levels(v_n), points)
    t = points["t"]
    dtype = plume_grid.x.dtype
    as_jax = lambda a: jnp.asarray(a, dtype=dtype)
    return MetField(
        plume_grid=plume_grid,
        times=as_jax(clock.seconds),
        u=as_jax(u_f),
        v=as_jax(v_f),
        w=as_jax(bilinear_at(on_levels(w), t.i, t.j)),
        temperature=as_jax(bilinear_at(on_levels(temperature), t.i, t.j)),
        pressure=as_jax(bilinear_at(on_levels(p), t.i, t.j)),
        pbl_height=as_jax(bilinear_at(pblh, t.i, t.j)),
        t0=clock.t0_iso,
    )


def _require_hypsometric_height():
    try:
        from xrtoolz.atm import hypsometric_height
    except ImportError as err:
        raise ImportError(
            "load_era5 integrates heights through xrtoolz; install it with "
            "`pip install 'plumax[io]'`"
        ) from err
    return hypsometric_height


def _normalise(ds: xr.Dataset) -> xr.Dataset:
    """CDS names → ``time`` / ``level`` / ``lat`` / ``lon``, both axes ascending."""
    ds = ds.rename({k: v for k, v in _RENAMES.items() if k in ds.variables})
    drop = [c for c in ("number", "expver") if c in ds.coords and ds[c].ndim == 0]
    return ds.drop_vars(drop).sortby("lat").sortby("lon")


def _is_periodic(lon: np.ndarray) -> bool:
    """Whether a regular longitude axis wraps the globe (n · Δλ = 360°)."""
    if lon.size < 2:
        return False
    step = float(lon[1] - lon[0])
    return abs(lon.size * step - 360.0) < 1e-6 * 360.0


def _require(ds: xr.Dataset, names: tuple[str, ...], path: str | Path) -> None:
    missing = [n for n in names if n not in ds]
    if missing:
        raise KeyError(f"load_era5: {path!s} lacks {missing}")


def _regular_index_map(lat: np.ndarray, lon: np.ndarray, frame: LocalFrame):
    """Frame metres → fractional ``(i, j)`` on a regular ascending lat/lon grid.

    Longitudes are put in the file's convention ([0, 360) when its
    longitudes exceed 180) before indexing. On a global axis the index is
    taken modulo 360° and may reach ``n`` (the seam column the loader
    appends); elsewhere points outside the grid come back NaN.
    """
    d_lon = float(lon[1] - lon[0])
    d_lat = float(lat[1] - lat[0])
    wraps = bool(lon.max() > 180.0)
    periodic = _is_periodic(lon)
    i_max = lon.size if periodic else lon.size - 1

    def to_index(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p_lat, p_lon = frame.to_latlon(x, y)
        if wraps:
            p_lon = np.mod(p_lon, 360.0)
        offset = p_lon - lon[0]
        if periodic:
            offset = np.mod(offset, 360.0)
        i = offset / d_lon
        j = (p_lat - lat[0]) / d_lat
        inside = (i >= 0) & (i <= i_max) & (j >= 0) & (j <= lat.size - 1)
        return np.where(inside, i, np.nan), np.where(inside, j, np.nan)

    return to_index
