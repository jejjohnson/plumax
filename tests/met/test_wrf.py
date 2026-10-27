"""Tests for the ``wrfout`` loader and the ``MetField`` wind adapter."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.errors import MissingVariableError
from plumax.les_fvm import simulate_eulerian_dispersion, wind_field_from_callable
from plumax.les_fvm.grid import make_grid
from plumax.met import MetField, load_wrf, to_prescribed_wind
from plumax.met.field import interpolate_in_time


DOMAIN_X = (0.0, 300.0, 6)
DOMAIN_Y = (0.0, 200.0, 4)
DOMAIN_Z = (0.0, 80.0, 4)
ORIGIN = (100.0, 100.0)


@pytest.fixture(scope="module")
def loaded(synthetic_wrf):
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z, dtype=jnp.float64)
    return grid, load_wrf(synthetic_wrf.path, plume_grid=grid, origin=ORIGIN)


@pytest.fixture(scope="module")
def loaded_f32(synthetic_wrf):
    """Same field on the float32 grid ``simulate_eulerian_dispersion`` builds."""
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z)
    return grid, load_wrf(synthetic_wrf.path, plume_grid=grid, origin=ORIGIN)


def test_shapes_times_and_stamp(loaded, synthetic_wrf):
    grid, field = loaded
    assert isinstance(field, MetField)
    nz, ny, nx = grid.interior_shape
    assert field.u.shape == (3, nz, ny, nx)
    assert field.pbl_height.shape == (3, ny, nx)
    np.testing.assert_allclose(np.asarray(field.times), synthetic_wrf.seconds)
    assert field.t0 == "2024-06-01_00:00:00"


def test_wind_lands_on_les_fvm_stagger_points(loaded, synthetic_wrf):
    """Loaded wind equals the analytic wind sampled the way les_fvm samples it.

    ``wind_field_from_callable`` evaluates ``u`` at U-points and ``v`` at
    V-points; the loader must put the destaggered, regridded WRF wind on the
    same points. The analytic wind is linear, so agreement is to round-off.
    """
    grid, field = loaded
    spec = synthetic_wrf

    def analytic(t, X, Y, Z):
        return (
            spec.u(X + ORIGIN[0], Y + ORIGIN[1], Z, 0),
            spec.v(X + ORIGIN[0], Y + ORIGIN[1], Z, 0),
            spec.w(X + ORIGIN[0], Y + ORIGIN[1], Z, 0),
        )

    expected = wind_field_from_callable(grid, analytic)(0.0)
    got = to_prescribed_wind(field)(0.0)
    for name, e, g in zip("uvw", expected, got, strict=True):
        np.testing.assert_allclose(
            np.asarray(g), np.asarray(e), rtol=0, atol=1e-9, err_msg=name
        )


def test_time_interpolation_is_piecewise_linear_and_clamped(loaded):
    _grid, field = loaded
    u0, u1 = field.u[0], field.u[1]
    mid = interpolate_in_time(field.times, field.u, 1800.0)
    np.testing.assert_allclose(np.asarray(mid), np.asarray(0.5 * (u0 + u1)), atol=1e-12)
    before = interpolate_in_time(field.times, field.u, -500.0)
    after = interpolate_in_time(field.times, field.u, 1.0e6)
    np.testing.assert_array_equal(np.asarray(before), np.asarray(u0))
    np.testing.assert_array_equal(np.asarray(after), np.asarray(field.u[-1]))


def test_pressure_column_integral_matches_analytic(loaded, synthetic_wrf):
    """Column integral of the loaded pressure equals the exact analytic integral.

    Pressure is linear in height in the fixture, so the vertical regrid is
    exact and the discrete column sum over the analysis levels equals the
    analytic ∫ p dz over [0, z_top] to round-off.
    """
    grid, field = loaded
    z_top = DOMAIN_Z[1]
    got = (field.pressure[0] * grid.dz).sum(axis=0)  # (ny, nx)
    p = synthetic_wrf.pressure
    exact = 0.5 * (p(0.0) + p(z_top)) * z_top  # trapezoid is exact for linear p
    np.testing.assert_allclose(np.asarray(got), exact, rtol=1e-12)


def test_temperature_and_pbl_are_physical(loaded, synthetic_wrf):
    grid, field = loaded
    temp = np.asarray(field.temperature)
    assert np.all(np.isfinite(temp))
    assert temp.min() > 250.0 and temp.max() < 310.0
    x_t = np.asarray(grid.x) + ORIGIN[0]
    expected = np.broadcast_to(
        synthetic_wrf.pbl_height(x_t, 0.0), field.pbl_height[0].shape
    )
    np.testing.assert_allclose(np.asarray(field.pbl_height[0]), expected, atol=1e-9)


def test_time_slice(synthetic_wrf):
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z, dtype=jnp.float64)
    field = load_wrf(
        synthetic_wrf.path, plume_grid=grid, origin=ORIGIN, times=slice(1, 3)
    )
    assert field.n_time == 2
    np.testing.assert_allclose(np.asarray(field.times), [0.0, 3600.0])
    with pytest.raises(ValueError, match="selects no time steps"):
        load_wrf(synthetic_wrf.path, plume_grid=grid, origin=ORIGIN, times=slice(5, 6))


def test_times_only_file_keeps_hourly_cadence(tmp_path):
    from tests.met.conftest import write_synthetic_wrfout

    spec = write_synthetic_wrfout(tmp_path / "wrfout_no_xtime.nc", with_xtime=False)
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z, dtype=jnp.float64)
    field = load_wrf(spec.path, plume_grid=grid, origin=ORIGIN)
    np.testing.assert_allclose(np.asarray(field.times), spec.seconds)
    assert field.t0 == "2024-06-01_00:00:00"


def test_vertical_targets_are_relative_to_grid_surface(loaded, synthetic_wrf):
    """A grid whose surface sits at z_min != 0 sees the same heights above ground."""
    _grid_ref, field_ref = loaded
    z_min, z_max, nz = DOMAIN_Z
    lifted = make_grid(
        DOMAIN_X, DOMAIN_Y, (z_min + 50.0, z_max + 50.0, nz), dtype=jnp.float64
    )
    field = load_wrf(synthetic_wrf.path, plume_grid=lifted, origin=ORIGIN)
    np.testing.assert_allclose(np.asarray(field.u), np.asarray(field_ref.u), atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(field.pressure), np.asarray(field_ref.pressure), rtol=1e-12
    )


def test_grid_outside_wrf_domain_raises(synthetic_wrf):
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z, dtype=jnp.float64)
    too_far = (synthetic_wrf.x_max - 100.0, 100.0)
    with pytest.raises(ValueError, match="does not extrapolate horizontally"):
        load_wrf(synthetic_wrf.path, plume_grid=grid, origin=too_far)


def test_simulate_eulerian_dispersion_runs_on_loaded_met(loaded_f32):
    _grid, field = loaded_f32
    ds = simulate_eulerian_dispersion(
        domain_x=DOMAIN_X,
        domain_y=DOMAIN_Y,
        domain_z=DOMAIN_Z,
        t_start=0.0,
        t_end=10.0,
        save_interval=5.0,
        emission_rate=0.1,
        source_location=(50.0, 100.0, 20.0),
        wind_field=to_prescribed_wind(field),
        # "pg" needs a scalar wind speed the runner cannot derive from a
        # prescribed field; use explicit (K_h, K_z) instead.
        eddy_diffusivity=(2.0, 0.5),
        solver="tsit5",
        dt0=0.5,
    )
    c = ds["concentration"].values
    assert np.all(np.isfinite(c))
    assert c[-1].max() > 0.0


def test_rotated_file_keeps_grid_relative_wind(tmp_path, loaded):
    """COSALPHA / SINALPHA in the file must not rotate the grid-frame wind."""
    from tests.met.conftest import write_synthetic_wrfout

    _grid_ref, field_ref = loaded
    spec = write_synthetic_wrfout(tmp_path / "wrfout_rotated.nc", alpha_deg=12.0)
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z, dtype=jnp.float64)
    field = load_wrf(spec.path, plume_grid=grid, origin=ORIGIN)
    np.testing.assert_allclose(np.asarray(field.u), np.asarray(field_ref.u), atol=1e-9)
    np.testing.assert_allclose(np.asarray(field.v), np.asarray(field_ref.v), atol=1e-9)


def test_missing_io_extra_names_it(synthetic_wrf, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "xrtoolz.atm", None)
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z, dtype=jnp.float64)
    with pytest.raises(ImportError, match=r"plumax\[io\]"):
        load_wrf(synthetic_wrf.path, plume_grid=grid, origin=ORIGIN)


# A Permian Basin site; the synthetic WRF grid is laid out in its UTM zone.
_SITE = (31.95, -102.15)
_SW = (-2000.0, -1500.0)  # frame metres of the WRF south-west mass point


@pytest.fixture(scope="module")
def latlon_wrf(tmp_path_factory):
    from plumax.met import LocalFrame
    from tests.met.conftest import write_synthetic_wrfout

    pytest.importorskip("pyproj")
    frame = LocalFrame.at(*_SITE)
    path = tmp_path_factory.mktemp("met") / "wrfout_latlon.nc"
    return frame, write_synthetic_wrfout(path, latlon=(frame, *_SW))


def test_frame_placement_matches_wrf_metres(latlon_wrf):
    """A LocalFrame on the WRF grid's own axes reproduces placement by metres.

    The analysis frame shares the WRF grid's UTM zone, so it differs from the
    WRF grid only by a shift: frame-axis winds equal WRF's grid-relative ones,
    and every field must match the metre-offset load. The wind path goes
    through COSALPHA (earth-relative) and back through the frame's east/north
    axes, so this checks both rotations.
    """
    from plumax.met import LocalFrame

    frame, spec = latlon_wrf
    lat0, lon0 = frame.to_latlon(_SW[0] + ORIGIN[0], _SW[1] + ORIGIN[1])
    analysis = LocalFrame(float(lat0), float(lon0), frame.epsg)
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z, dtype=jnp.float64)

    by_metres = load_wrf(spec.path, plume_grid=grid, origin=ORIGIN)
    by_frame = load_wrf(spec.path, plume_grid=grid, origin=analysis)
    for name in ("u", "v", "w", "temperature", "pressure", "pbl_height"):
        np.testing.assert_allclose(
            np.asarray(getattr(by_frame, name)),
            np.asarray(getattr(by_metres, name)),
            atol=1e-5,
            rtol=1e-9,
            err_msg=name,
        )
    assert by_frame.t0 == by_metres.t0


def test_frame_placement_needs_latlon_and_rotation(synthetic_wrf, tmp_path):
    from plumax.met import LocalFrame
    from tests.met.conftest import write_synthetic_wrfout

    pytest.importorskip("pyproj")
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z, dtype=jnp.float64)
    frame = LocalFrame.at(*_SITE)
    rotated_only = write_synthetic_wrfout(tmp_path / "rot.nc", alpha_deg=0.0)
    with pytest.raises(MissingVariableError, match="XLAT / XLONG"):
        load_wrf(rotated_only.path, plume_grid=grid, origin=frame)
    with pytest.raises(MissingVariableError, match="COSALPHA / SINALPHA"):
        load_wrf(synthetic_wrf.path, plume_grid=grid, origin=frame)


def test_frame_outside_wrf_domain_raises(latlon_wrf):
    from plumax.met import LocalFrame

    _frame, spec = latlon_wrf
    far = LocalFrame.at(_SITE[0] + 1.0, _SITE[1])  # ~110 km north
    grid = make_grid(DOMAIN_X, DOMAIN_Y, DOMAIN_Z, dtype=jnp.float64)
    with pytest.raises(ValueError, match="outside the source domain"):
        load_wrf(spec.path, plume_grid=grid, origin=far)
