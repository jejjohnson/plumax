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

import einx
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Float

from plumax._src.errors import (
    GridMismatchError,
    MissingExtraError,
    MissingVariableError,
    TimeAxisError,
)
from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met._src.frames import LocalFrame
from plumax.met._src.regrid import IndexMap, bilinear_at, frame_wind, stagger_points
from plumax.met._src.time import TimeAxis
from plumax.met.field import MetField
from plumax.met.wrf import _interpolate_vertical


#: Standard gravity g [m s⁻²], as xrtoolz's hypsometric integration uses.
GRAVITY = 9.80665
#: Dry-air gas constant R_d [J kg⁻¹ K⁻¹].
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

    w = −ω / (ρ g),  ρ = p / (R_d T),  z_k = hypsometric height of level k.

    Algorithm:
        1. Read both files, check they share times and a lat/lon grid, and
           order the levels surface first.
        2. Integrate each level's height above ground from ``sp``.
        3. Convert ω to w; close the longitude seam on a global grid.
        4. Interpolate every column to the analysis heights, then sample the
           analysis T-, U- and V-points bilinearly and project the wind onto
           the frame's axes.

    Args:
        pressure_levels: NetCDF with ``u``, ``v``, ``w`` (ω), ``t`` and
            optionally ``q`` on pressure levels in hPa.
        single_levels: NetCDF with ``sp`` and ``blh`` at the same times and on
            the same grid.
        plume_grid: Analysis grid, in ``frame`` metres.
        frame: Frame the analysis grid is laid out in.
        times: Optional ``slice`` over the shared time axis.

    Returns:
        A :class:`MetField` on ``plume_grid``.

    Raises:
        MissingExtraError: If the ``io`` extra (xrtoolz) is not installed.
        MissingVariableError: If a required variable is missing.
        GridMismatchError: If the two files' times or lat/lon grids differ.
        TimeAxisError: If ``times`` selects nothing.
        DomainError: If an analysis point falls outside the ERA5 domain.
    """
    hypsometric_height = _require_hypsometric_height()

    # --- read and check ---
    with (
        xr.open_dataset(pressure_levels) as pressure_raw,
        xr.open_dataset(single_levels) as single_raw,
    ):
        levels = _normalise(pressure_raw)
        surface = _normalise(single_raw)
        _require(levels, ("u", "v", "w", "t"), pressure_levels)
        _require(surface, ("sp", "blh"), single_levels)
        if not np.array_equal(levels["time"].values, surface["time"].values):
            raise GridMismatchError("load_era5: the two files cover different times")
        same_lat = np.array_equal(levels["lat"].values, surface["lat"].values)
        same_lon = np.array_equal(levels["lon"].values, surface["lon"].values)
        if not (same_lat and same_lon):
            raise GridMismatchError(
                "load_era5: the two files are on different lat/lon grids; request "
                "both with the same `area` and `grid` so `sp` and `blh` line up "
                "with the pressure levels"
            )
        if times is not None:
            levels = levels.isel(time=times)
            surface = surface.isel(time=times)
        if levels.sizes["time"] == 0:
            raise TimeAxisError("load_era5: `times` selects no time steps")
        # surface first, so heights increase along `level`
        levels = levels.sortby("level", ascending=False)

        # --- heights ---
        # p_k = 100 · level_k  (hPa → Pa)
        pressure = (100.0 * levels["level"]).broadcast_like(levels["t"])
        columns = levels.assign(pressure=pressure, sp=surface["sp"])
        # z_k: hypsometric integration up from p_s
        height = hypsometric_height(
            columns,
            pressure="pressure",
            temperature="t",
            specific_humidity="q" if "q" in levels else None,
            surface_pressure="sp",
            level="level",
            name="z_agl",
        )["z_agl"]

        dims = ("time", "level", "lat", "lon")
        u_east = levels["u"].transpose(*dims).values
        v_north = levels["v"].transpose(*dims).values
        omega = levels["w"].transpose(*dims).values
        temperature = levels["t"].transpose(*dims).values
        pressure_pa = pressure.transpose(*dims).values
        height_agl = height.transpose(*dims).values
        pbl_height = surface["blh"].transpose("time", "lat", "lon").values
        lat = levels["lat"].values
        lon = levels["lon"].values
        clock = TimeAxis.from_datetimes(levels["time"].values)

    # --- vertical velocity ---
    # ρ = p / (R_d T)
    density = pressure_pa / (R_DRY * temperature)
    # w = −ω / (ρ g)
    w = -omega / (density * GRAVITY)

    # --- longitude seam ---
    if _is_periodic(lon):
        # append the first column after the last, so points between it and 360°
        # interpolate across the seam
        fields = (u_east, v_north, temperature, pressure_pa, w, height_agl)
        u_east, v_north, temperature, pressure_pa, w, height_agl = map(
            _close_seam, fields
        )
        pbl_height = _close_seam(pbl_height)

    # --- onto the analysis grid ---
    grid_z = np.asarray(plume_grid.z, dtype=np.float64)
    # z_surface = z₀ − Δz/2
    grid_surface = grid_z[0] - 0.5 * plume_grid.dz
    # h_k = z_k − z_surface
    target_heights = grid_z - grid_surface
    to_index = _regular_index_map(lat, lon, frame)
    points = stagger_points(plume_grid, frame, to_index, loader="load_era5")
    at_t = points["t"]
    on_levels = lambda f: _interpolate_vertical(height_agl, f, target_heights)
    u_frame, v_frame = frame_wind(on_levels(u_east), on_levels(v_north), points)
    w_t = bilinear_at(on_levels(w), at_t.i, at_t.j)
    temperature_t = bilinear_at(on_levels(temperature), at_t.i, at_t.j)
    pressure_t = bilinear_at(on_levels(pressure_pa), at_t.i, at_t.j)
    pbl_height_t = bilinear_at(pbl_height, at_t.i, at_t.j)
    dtype = plume_grid.x.dtype
    as_jax = lambda a: jnp.asarray(a, dtype=dtype)
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


def _require_hypsometric_height():
    try:
        import xrtoolz.atm
    except ImportError as err:
        raise MissingExtraError(
            "load_era5 integrates heights through xrtoolz; install it with "
            "`pip install 'plumax[io]'`"
        ) from err
    return xrtoolz.atm.hypsometric_height


def _normalise(ds: xr.Dataset) -> xr.Dataset:
    """CDS names → ``time`` / ``level`` / ``lat`` / ``lon``, both axes ascending."""
    renames = {old: new for old, new in _RENAMES.items() if old in ds.variables}
    ds = ds.rename(renames)
    scalar_coords = [
        name
        for name in ("number", "expver")
        if name in ds.coords and ds[name].ndim == 0
    ]
    return ds.drop_vars(scalar_coords).sortby("lat").sortby("lon")


def _require(ds: xr.Dataset, names: tuple[str, ...], path: str | Path) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise MissingVariableError(f"load_era5: {path!s} lacks {missing}")


def _is_periodic(lon: Float[np.ndarray, " lon"]) -> bool:
    """Whether a regular longitude axis wraps the globe: n · Δλ = 360°."""
    if lon.size < 2:
        return False
    # Δλ
    step = float(lon[1] - lon[0])
    # |n Δλ − 360| < ε
    return abs(lon.size * step - 360.0) < 1e-6 * 360.0


def _close_seam(
    field: Float[np.ndarray, "*lead lon"],
) -> Float[np.ndarray, "*lead lon_closed"]:
    """Append the first longitude column after the last."""
    first_column = field[..., :1]
    return einx.id("... lon, ... first -> ... (lon + first)", field, first_column)


def _regular_index_map(
    lat: Float[np.ndarray, " lat"], lon: Float[np.ndarray, " lon"], frame: LocalFrame
) -> IndexMap:
    """Frame metres → fractional ``(i, j)`` on a regular ascending lat/lon grid.

    i = (λ − λ₀) / Δλ,  j = (φ − φ₀) / Δφ.

    Longitudes are put in the file's convention ([0, 360) when its longitudes
    exceed 180) before indexing. On a global axis ``λ − λ₀`` is taken modulo
    360° and ``i`` may reach ``n`` (the seam column the loader appends);
    elsewhere points outside the grid come back NaN.
    """
    # Δλ, Δφ
    lon_step = float(lon[1] - lon[0])
    lat_step = float(lat[1] - lat[0])
    wraps = bool(lon.max() > 180.0)
    periodic = _is_periodic(lon)
    last_i = lon.size if periodic else lon.size - 1

    def to_index(
        x: Float[np.ndarray, "y x"], y: Float[np.ndarray, "y x"]
    ) -> tuple[Float[np.ndarray, "y x"], Float[np.ndarray, "y x"]]:
        point_lat, point_lon = frame.to_latlon(x, y)
        if wraps:
            point_lon = np.mod(point_lon, 360.0)
        # λ − λ₀
        lon_offset = point_lon - lon[0]
        if periodic:
            lon_offset = np.mod(lon_offset, 360.0)
        # i = (λ − λ₀) / Δλ
        i = lon_offset / lon_step
        # j = (φ − φ₀) / Δφ
        j = (point_lat - lat[0]) / lat_step
        inside_i = (i >= 0) & (i <= last_i)
        inside_j = (j >= 0) & (j <= lat.size - 1)
        inside = inside_i & inside_j
        return np.where(inside, i, np.nan), np.where(inside, j, np.nan)

    return to_index
