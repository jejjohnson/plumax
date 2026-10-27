"""Tests for the ERA5 loader on synthetic CDS-shaped files.

The synthetic atmosphere is isothermal and hydrostatic, so every field the
loader reconstructs has a closed form: with ``H = R_d T / g`` the pressure
at height ``z`` above ground is ``p(z) = p_s exp(−z / H)``, the vertical
velocity from a constant ω is ``w = −ω R_d T / (p g)``, and a uniform
eastward wind keeps its speed with a direction set by the frame's meridian
convergence. The expected direction comes from ``pyproj.Geod`` azimuths of
the frame's y axis, independent of the loader's own frame axes.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from plumax.les_fvm.grid import make_grid
from plumax.met import LocalFrame, load_era5
from plumax.met.era5 import GRAVITY, R_DRY


pytest.importorskip("xrtoolz")
Geod = pytest.importorskip("pyproj").Geod

SITE = (31.95, -102.15)  # Permian Basin
T_ISO = 290.0  # K
P_SURFACE = 99_500.0  # Pa: 1000 hPa sits just below ground
OMEGA = -0.5  # Pa s⁻¹ (rising air)
U_EAST = 5.0  # m s⁻¹
LEVELS = np.arange(900.0, 1001.0, 5.0)  # hPa, top-down as CDS writes them
LAT = np.array([32.5, 32.25, 32.0, 31.75, 31.5])  # descending, as ERA5
LON = np.arange(-102.75, -101.49, 0.25)
TIMES = np.array(
    ["2024-06-01T00:00", "2024-06-01T01:00", "2024-06-01T02:00"], dtype="datetime64[ns]"
)
DOMAIN = ((-300.0, 300.0, 6), (-200.0, 200.0, 4), (0.0, 80.0, 4))


def blh(lon: np.ndarray) -> np.ndarray:
    return 600.0 + 100.0 * (lon + 102.0)


def write_era5(
    tmp: Path,
    *,
    lon: np.ndarray = LON,
    times: np.ndarray = TIMES,
    lat: np.ndarray = LAT,
    sl_lon: np.ndarray | None = None,
) -> tuple[Path, Path]:
    sl_lon = lon if sl_lon is None else sl_lon
    shape = (times.size, LEVELS.size, lat.size, lon.size)
    coords = {
        "valid_time": times,
        "pressure_level": LEVELS,
        "latitude": lat,
        "longitude": lon,
    }
    dims = tuple(coords)
    pl = xr.Dataset(
        {
            "u": (dims, np.full(shape, U_EAST)),
            "v": (dims, np.zeros(shape)),
            "w": (dims, np.full(shape, OMEGA)),
            "t": (dims, np.full(shape, T_ISO)),
        },
        coords=coords,
    )
    sl_dims = ("valid_time", "latitude", "longitude")
    lon_wrapped = ((sl_lon + 180.0) % 360.0) - 180.0
    sl = xr.Dataset(
        {
            "sp": (sl_dims, np.full((times.size, lat.size, sl_lon.size), P_SURFACE)),
            "blh": (
                sl_dims,
                np.broadcast_to(blh(lon_wrapped), (times.size, lat.size, sl_lon.size)),
            ),
        },
        coords={**{k: coords[k] for k in sl_dims}, "longitude": sl_lon},
    )
    pl_path, sl_path = tmp / "era5_pl.nc", tmp / "era5_sl.nc"
    pl.to_netcdf(pl_path)
    sl.to_netcdf(sl_path)
    return pl_path, sl_path


@pytest.fixture(scope="module")
def frame() -> LocalFrame:
    return LocalFrame.at(*SITE)


@pytest.fixture(scope="module")
def grid():
    return make_grid(*DOMAIN, dtype=jnp.float64)


@pytest.fixture(scope="module")
def field(tmp_path_factory, frame, grid):
    pl, sl = write_era5(tmp_path_factory.mktemp("era5"))
    return load_era5(pl, sl, plume_grid=grid, frame=frame)


def _heights(grid) -> np.ndarray:
    z = np.asarray(grid.z, dtype=np.float64)
    return z - (z[0] - 0.5 * grid.dz)


def test_shapes_and_clock(field, grid):
    nz, ny, nx = grid.z.size, grid.y.size, grid.x.size
    assert field.u.shape == (3, nz, ny, nx)
    assert field.pbl_height.shape == (3, ny, nx)
    np.testing.assert_allclose(np.asarray(field.times), [0.0, 3600.0, 7200.0])
    assert field.t0 == "2024-06-01_00:00:00"


def test_hydrostatic_pressure_and_omega_conversion(field, grid):
    scale_height = R_DRY * T_ISO / GRAVITY
    p_expected = P_SURFACE * np.exp(-_heights(grid) / scale_height)
    p = np.asarray(field.pressure)
    np.testing.assert_allclose(
        p, np.broadcast_to(p_expected[None, :, None, None], p.shape), atol=1.0
    )
    w_expected = -OMEGA * R_DRY * T_ISO / (p_expected * GRAVITY)
    w = np.asarray(field.w)
    np.testing.assert_allclose(
        w, np.broadcast_to(w_expected[None, :, None, None], w.shape), rtol=1e-4
    )
    np.testing.assert_allclose(np.asarray(field.temperature), T_ISO, rtol=1e-12)


def test_eastward_wind_rotates_by_meridian_convergence(field, frame, grid):
    """A true-east wind of speed U has grid components (U cos a, U sin a).

    ``a`` is the azimuth of the frame's +y axis (grid north) clockwise from
    true north, measured on the ellipsoid with ``pyproj.Geod``.
    """
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    geod = Geod(ellps="WGS84")

    def azimuth_of_grid_north(xx, yy):
        lat0, lon0 = frame.to_latlon(xx, yy)
        lat1, lon1 = frame.to_latlon(xx, yy + 1.0)
        az, _, _ = geod.inv(lon0, lat0, lon1, lat1)
        return np.deg2rad(az)

    xu, yu = np.meshgrid(x + 0.5 * grid.dx, y)
    xv, yv = np.meshgrid(x, y + 0.5 * grid.dy)
    u_expected = U_EAST * np.cos(azimuth_of_grid_north(xu, yu))
    v_expected = U_EAST * np.sin(azimuth_of_grid_north(xv, yv))
    np.testing.assert_allclose(np.asarray(field.u)[0, 0], u_expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(field.v)[0, 0], v_expected, atol=1e-6)
    # The convergence is not negligible here: about 1.5°, ~0.13 m/s of v.
    assert np.abs(v_expected).min() > 0.1


def test_pbl_height_is_bilinear_exact_for_linear_fields(field, frame, grid):
    xx, yy = np.meshgrid(np.asarray(grid.x), np.asarray(grid.y))
    _, lon = frame.to_latlon(xx, yy)
    np.testing.assert_allclose(np.asarray(field.pbl_height)[1], blh(lon), rtol=1e-9)


def test_longitudes_in_0_360_are_handled(tmp_path, frame, grid, field):
    pl, sl = write_era5(tmp_path, lon=LON + 360.0)
    wrapped = load_era5(pl, sl, plume_grid=grid, frame=frame)
    np.testing.assert_allclose(np.asarray(wrapped.u), np.asarray(field.u), atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(wrapped.pbl_height), np.asarray(field.pbl_height), rtol=1e-9
    )


def test_time_slice_and_errors(tmp_path, frame, grid):
    pl, sl = write_era5(tmp_path)
    sliced = load_era5(pl, sl, plume_grid=grid, frame=frame, times=slice(1, 3))
    np.testing.assert_allclose(np.asarray(sliced.times), [0.0, 3600.0])
    assert sliced.t0 == "2024-06-01_01:00:00"
    with pytest.raises(ValueError, match="selects no time steps"):
        load_era5(pl, sl, plume_grid=grid, frame=frame, times=slice(5, 6))

    other = tmp_path / "other"
    other.mkdir()
    _, sl_shifted = write_era5(other, times=TIMES + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="different times"):
        load_era5(pl, sl_shifted, plume_grid=grid, frame=frame)

    with pytest.raises(KeyError, match="blh"):
        load_era5(pl, pl, plume_grid=grid, frame=frame)

    far = LocalFrame.at(SITE[0] + 2.0, SITE[1])
    with pytest.raises(ValueError, match="outside the source domain"):
        load_era5(pl, sl, plume_grid=grid, frame=far)


def test_global_grid_interpolates_across_the_seam(tmp_path):
    """A 0–359° global axis must sample across 359°/0° like a -180–179° one."""
    site = (5.0, -0.3)  # 0.3° west of Greenwich: between the 359° and 0° columns
    seam_frame = LocalFrame.at(*site)
    grid = make_grid(*DOMAIN, dtype=jnp.float64)
    lat = np.array([6.0, 5.0, 4.0])
    east = np.arange(0.0, 360.0, 1.0)  # global, periodic
    centred = ((east + 180.0) % 360.0) - 180.0
    order = np.argsort(centred)
    a, b = tmp_path / "global", tmp_path / "centred"
    a.mkdir()
    b.mkdir()
    across = load_era5(
        *write_era5(a, lon=east, lat=lat), plume_grid=grid, frame=seam_frame
    )
    inside = load_era5(
        *write_era5(b, lon=centred[order], lat=lat), plume_grid=grid, frame=seam_frame
    )
    for name in ("u", "v", "pressure", "pbl_height"):
        np.testing.assert_allclose(
            np.asarray(getattr(across, name)),
            np.asarray(getattr(inside, name)),
            rtol=1e-12,
        )


def test_single_levels_on_another_grid_are_rejected(tmp_path, frame, grid):
    pl, sl = write_era5(tmp_path, sl_lon=LON + 0.125)
    with pytest.raises(ValueError, match="different lat/lon grids"):
        load_era5(pl, sl, plume_grid=grid, frame=frame)
