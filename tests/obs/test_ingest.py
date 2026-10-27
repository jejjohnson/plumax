"""Tests for the L2 loaders: unit conversions, screening, and a fused twin.

The twin test writes a TROPOMI granule and an EMIT scene whose values are a
known plume's column enhancement converted back to each product's units,
ingests both, fuses them and recovers the emission rate. That exercises
pixel placement through the frame, both unit conversions, the TROPOMI
surface-layer kernel and the screening together.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from hypothesis import given, strategies as st

from plumax.coupled import (
    Instrument,
    PlumeSource,
    build_coupled_forward,
    default_prior,
    fuse_observations,
)
from plumax.errors import MissingVariableError
from plumax.met import LocalFrame, TimeAxis
from plumax.obs import (
    load_emit_l2b,
    load_ghgsat_l2,
    load_tropomi_l2,
    ppb_to_column_mass,
    ppm_m_to_column_mass,
)


pytest.importorskip("xrreader")
pytest.importorskip("pyproj")

SITE = (31.95, -102.15)  # Permian Basin
BACKGROUND = 1880.0  # ppb
SP = 90_000.0  # Pa, a high-plains surface
AK_SURFACE = 0.8  # TROPOMI kernel in the lowest (last) layer
N_LAYER = 12
Q_TRUE = 0.05  # kg s⁻¹
REF_TIME = "2024-06-01T18:00:00"


def dry_air(sp: float) -> float:
    # N_dry = p_s N_A / (g M_dry), molecules m⁻²
    return sp * 6.02214076e23 / (9.80665 * 0.0289644)


def test_unit_conversions_by_hand():
    # 1 ppb over a 1013.25 hPa dry column ≈ 5.72e-6 kg m⁻²
    assert ppb_to_column_mass(1.0, dry_air(101_325.0)) == pytest.approx(
        5.72e-6, rel=1e-3
    )
    # 1000 ppm·m at 1013.25 hPa, 288.15 K: 42.3 mol m⁻³ · 1e-3 m · M ≈ 6.79e-4 kg m⁻²
    got = ppm_m_to_column_mass(1000.0, pressure_pa=101_325.0, temperature_k=288.15)
    assert got == pytest.approx(42.29 * 1e-3 * 0.0160425, rel=1e-3)


# ---- synthetic products ---------------------------------------------------


def _source() -> PlumeSource:
    return PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=4.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=jnp.linspace(0.0, 1500.0, 61),
    )


def _pixels(frame, xs, ys):
    xx, yy = np.meshgrid(xs, ys)
    lat, lon = frame.to_latlon(xx, yy)
    return lat, lon, np.column_stack([xx.ravel(), yy.ravel()])


def _true_enhancement(receptors, ak):
    instrument = Instrument(
        "probe", jnp.asarray(receptors), averaging_kernel=jnp.asarray(ak)
    )
    forward = build_coupled_forward(_source(), [instrument])
    return Q_TRUE * np.asarray(forward.responses[0])


def write_tropomi(path: Path, frame: LocalFrame, *, low_qa=((0, 0),)) -> Path:
    n_scan, n_pix = 4, 5
    lat, lon, receptors = _pixels(
        frame, np.linspace(-500.0, 3500.0, n_pix), np.linspace(-1500.0, 1500.0, n_scan)
    )
    ak_flat = np.full(lat.size, AK_SURFACE)
    delta = _true_enhancement(receptors, ak_flat)
    xch4 = BACKGROUND + delta / ppb_to_column_mass(1.0, dry_air(SP))
    swath = ("time", "scanline", "ground_pixel")
    shape = (1, n_scan, n_pix)
    qa = np.full(shape, 0.9)
    for s, p in low_qa:
        qa[0, s, p] = 0.2
    ak = np.ones((*shape, N_LAYER))
    ak[..., -1] = AK_SURFACE  # layers run top of atmosphere first
    epoch = np.datetime64("2010-01-01T00:00:00")
    t_ref = int((np.datetime64(REF_TIME) - epoch) / np.timedelta64(1, "s"))
    product = xr.Dataset(
        {
            "latitude": (swath, lat.reshape(shape)),
            "longitude": (swath, lon.reshape(shape)),
            "delta_time": (
                ("time", "scanline"),
                (np.arange(n_scan) * 1000).reshape(1, n_scan).astype("int32"),
                {"units": f"milliseconds since {REF_TIME.replace('T', ' ')}"},
            ),
            "qa_value": (swath, qa),
            "methane_mixing_ratio": (
                swath,
                xch4.reshape(shape) + 10.0,
                {"units": "1e-9"},
            ),
            "methane_mixing_ratio_bias_corrected": (swath, xch4.reshape(shape)),
            "methane_mixing_ratio_precision": (swath, np.full(shape, 0.5)),
        },
        coords={
            "time": (
                "time",
                np.array([t_ref]),
                {"units": "seconds since 2010-01-01 00:00:00"},
            ),
            "scanline": np.arange(n_scan),
            "ground_pixel": np.arange(n_pix),
            "layer": np.arange(N_LAYER),
        },
    )
    layered = (*swath, "layer")
    detailed = xr.Dataset({"column_averaging_kernel": (layered, ak)})
    inputs = xr.Dataset(
        {
            "surface_pressure": (swath, np.full(shape, SP)),
            "methane_profile_apriori": (layered, np.full((*shape, N_LAYER), 0.01)),
            "dry_air_subcolumns": (layered, np.full((*shape, N_LAYER), 300.0)),
        }
    )
    xr.DataTree.from_dict(
        {
            "PRODUCT": product,
            "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS": detailed,
            "PRODUCT/SUPPORT_DATA/INPUT_DATA": inputs,
        }
    ).to_netcdf(path)
    return path


EMIT_P, EMIT_T = 90_000.0, 300.0


def write_emit(path: Path, frame: LocalFrame) -> Path:
    ny, nx = 6, 6
    lat, lon, receptors = _pixels(
        frame, np.linspace(100.0, 700.0, nx), np.linspace(-300.0, 300.0, ny)
    )
    delta = _true_enhancement(receptors, np.ones(lat.size))
    ppm_m = delta / ppm_m_to_column_mass(1.0, pressure_pa=EMIT_P, temperature_k=EMIT_T)
    xr.Dataset(
        {
            "ch4_enhancement": (("y", "x"), ppm_m.reshape(ny, nx), {"units": "ppm m"}),
            "ch4_uncertainty": (("y", "x"), np.full((ny, nx), 20.0)),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        attrs={"time_coverage_start": "2024-06-01T18:30:00Z"},
    ).to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def frame() -> LocalFrame:
    return LocalFrame.at(*SITE)


@pytest.fixture(scope="module")
def files(tmp_path_factory, frame):
    tmp = tmp_path_factory.mktemp("obs")
    return write_tropomi(tmp / "S5P_L2__CH4.nc", frame), write_emit(
        tmp / "emit.nc", frame
    )


# ---- tests ----------------------------------------------------------------


def test_tropomi_screening_kernel_and_variance(files, frame):
    tropomi, _ = files
    clock = TimeAxis.from_datetimes(["2024-06-01T17:00:00"])
    obs = load_tropomi_l2(tropomi, frame=frame, background_ppb=BACKGROUND, clock=clock)
    assert obs.instrument.n_obs == 4 * 5 - 1  # one low-QA pixel dropped
    np.testing.assert_allclose(obs.instrument.ak, AK_SURFACE)
    sigma = ppb_to_column_mass(0.5, dry_air(SP))
    np.testing.assert_allclose(obs.instrument.retrieval_variance, sigma**2, rtol=1e-6)
    # the middle scanline is ~1.5 s after the reference time, an hour after t0
    assert 3600.0 <= obs.instrument.observation_time <= 3603.0


def test_bbox_crop(files, frame):
    tropomi, _ = files
    lat0, lon0 = SITE
    box = (lon0 - 0.001, lat0 - 0.02, lon0 + 0.05, lat0 + 0.02)
    obs = load_tropomi_l2(tropomi, frame=frame, background_ppb=BACKGROUND, bbox=box)
    receptors = np.asarray(obs.instrument.receptors)
    assert 0 < obs.instrument.n_obs < 19 and receptors[:, 0].min() > -150.0
    with pytest.raises(ValueError, match="no pixel survives"):
        load_tropomi_l2(tropomi, frame=frame, bbox=(0.0, 0.0, 1.0, 1.0))


def test_fused_twin_recovers_the_emission_rate(files, frame):
    tropomi, emit = files
    trop = load_tropomi_l2(tropomi, frame=frame, background_ppb=BACKGROUND)
    em = load_emit_l2b(emit, frame=frame, pressure_pa=EMIT_P, temperature_k=EMIT_T)
    forward = build_coupled_forward(_source(), [trop.instrument, em.instrument])
    mean, cov = default_prior(
        n_instruments=2,
        emission_prior_mean=0.0,
        emission_prior_std=1.0,
        bias_prior_std=1e-9,
    )
    post = fuse_observations(
        forward,
        [trop.enhancement, em.enhancement],
        prior_mean=mean,
        prior_covariance=cov,
    )
    assert post.emission_rate == pytest.approx(Q_TRUE, rel=1e-4)


def test_emit_needs_an_uncertainty(tmp_path, frame):
    lat, lon, _ = _pixels(frame, np.array([0.0, 60.0]), np.array([0.0, 60.0]))
    path = tmp_path / "emit_no_unc.nc"
    xr.Dataset(
        {
            "ch4_enhancement": (("y", "x"), np.ones((2, 2))),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        }
    ).to_netcdf(path)
    with pytest.raises(MissingVariableError, match="retrieval_sigma_ppm_m"):
        load_emit_l2b(path, frame=frame)
    obs = load_emit_l2b(path, frame=frame, retrieval_sigma_ppm_m=50.0)
    assert obs.instrument.n_obs == 4


def test_ghgsat_background_and_conversion(tmp_path, frame):
    lat, lon, _ = _pixels(frame, np.linspace(0.0, 200.0, 3), np.linspace(0.0, 200.0, 3))
    xch4 = np.full((3, 3), 1900.0)
    xch4[1, 1] = 1950.0
    path = tmp_path / "ghgsat.nc"
    xr.Dataset(
        {
            "xch4": (("y", "x"), xch4, {"units": "ppb"}),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        attrs={"time_coverage_start": "2024-06-01T19:00:00"},
    ).to_netcdf(path)
    obs = load_ghgsat_l2(path, frame=frame, retrieval_sigma_ppb=10.0)
    enhancement = np.asarray(obs.enhancement)
    expected_peak = ppb_to_column_mass(50.0, dry_air(101_325.0))  # median is 1900
    assert enhancement.max() == pytest.approx(expected_peak, rel=1e-6)
    assert np.count_nonzero(enhancement) == 1
    assert obs.time == np.datetime64("2024-06-01T19:00:00")


def test_ghgsat_background_comes_from_the_kept_pixels(tmp_path, frame):
    """A crop with its own background must not inherit the whole scene's."""
    xs = np.linspace(0.0, 5000.0, 6)
    lat, lon, _ = _pixels(frame, xs, np.array([0.0, 100.0]))
    xch4 = np.where(xs[None, :] < 2500.0, 1880.0, 1920.0) * np.ones((2, 1))
    path = tmp_path / "ghgsat_two_backgrounds.nc"
    xr.Dataset(
        {
            "xch4": (("y", "x"), xch4, {"units": "ppb"}),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        }
    ).to_netcdf(path)
    lat_e, lon_e = frame.to_latlon(
        np.array([2600.0, 5600.0]), np.array([-100.0, 200.0])
    )
    east = (float(lon_e[0]), float(lat_e[0]), float(lon_e[1]), float(lat_e[1]))
    obs = load_ghgsat_l2(path, frame=frame, retrieval_sigma_ppb=10.0, bbox=east)
    assert obs.instrument.n_obs == 6  # the 1920 ppb half only
    np.testing.assert_allclose(obs.enhancement, 0.0, atol=1e-15)


@given(
    a=st.floats(-500.0, 500.0),
    b=st.floats(-500.0, 500.0),
    pressure=st.floats(50_000.0, 105_000.0),
)
def test_conversions_are_linear_and_scale_with_the_column(a, b, pressure):
    column = dry_air(pressure)
    total = ppb_to_column_mass(a + b, column)
    parts = ppb_to_column_mass(a, column) + ppb_to_column_mass(b, column)
    np.testing.assert_allclose(total, parts, rtol=1e-9, atol=1e-18)
    # ΔΩ ∝ p_s through N_dry, and ∝ p / T for ppm·m
    np.testing.assert_allclose(
        ppb_to_column_mass(a, column),
        ppb_to_column_mass(a, dry_air(101_325.0)) * pressure / 101_325.0,
        rtol=1e-9,
        atol=1e-18,
    )
    one = ppm_m_to_column_mass(a, pressure_pa=pressure, temperature_k=290.0)
    two = ppm_m_to_column_mass(a, pressure_pa=2.0 * pressure, temperature_k=290.0)
    np.testing.assert_allclose(two, 2.0 * one, rtol=1e-12, atol=1e-18)
