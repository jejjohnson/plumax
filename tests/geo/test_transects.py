"""Cross-plume transects: profiles, moments, Gaussian fits, smoothing (#269).

Synthetic ridges on 60 m grids. Needs the ``[geo]`` extra (in the dev group).
The functions are NumPy and SciPy code outside any traced path, so there are
no gradient, ``jit`` or ``vmap`` tests.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import rasterio
from geoarray import GeoArray
from hypothesis import given, settings, strategies as st
from jaxtyping import TypeCheckError
from scipy import stats

from plumax import geo
from plumax._src.errors import ParameterError
from plumax.errors import DomainError
from plumax.geo import transects
from plumax.geo.transects import (
    CrossPlumeProfile,
    TransectProfiles,
    cross_plume_transects,
    fit_transect_gaussians,
    fit_width_law,
    smooth_centreline,
    transect_moments,
)


PIXEL = 60.0
LINE_DENSITY = 0.1


def _grid(n_row, n_col, *, west=None, north=None):
    """North-up 60 m transform; centred on the origin unless placed."""
    west = -0.5 * n_col * PIXEL if west is None else west
    north = 0.5 * n_row * PIXEL if north is None else north
    return rasterio.Affine(PIXEL, 0.0, west, 0.0, -PIXEL, north)


def _centres(shape, transform):
    rows, cols = np.indices(shape, dtype=float)
    xs = transform.c + transform.a * (cols + 0.5)
    ys = transform.f + transform.e * (rows + 0.5)
    return xs, ys


def _ridge(shape, transform, *, direction, width, centreline=None, cut=False):
    """q / (√(2π) w) exp(−(c − m(d))² / 2w²) about the origin along φ."""
    xs, ys = _centres(shape, transform)
    # d = sᵀ e_∥,  c = sᵀ e_⊥
    downwind = xs * math.cos(direction) + ys * math.sin(direction)
    crosswind = -xs * math.sin(direction) + ys * math.cos(direction)
    offset = 0.0 if centreline is None else centreline(downwind)
    spread = width(downwind) if callable(width) else width
    profile = (
        LINE_DENSITY
        / (math.sqrt(2.0 * math.pi) * spread)
        * np.exp(-0.5 * ((crosswind - offset) / spread) ** 2)
    )
    return profile * (downwind > 0.0) if cut else profile


# ── oracles ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("degrees", [0.0, 30.0, 45.0])
@pytest.mark.parametrize("width_pixels", [2.0, 5.0, 10.0])
def test_gaussian_ridge_is_recovered(degrees, width_pixels):
    width = width_pixels * PIXEL
    shape = (140, 140)
    transform = _grid(*shape)
    image = _ridge(shape, transform, direction=math.radians(degrees), width=width)
    profiles = cross_plume_transects(
        image,
        transform,
        source=(0.0, 0.0),
        direction=math.radians(degrees),
        n_transects=10,
        spacing=120.0,
        half_length=6.0 * width,
    )
    assert profiles.values.shape[0] == 10
    assert np.isfinite(profiles.values).all()
    moments = transect_moments(profiles)
    fits = fit_transect_gaussians(profiles, min_width=PIXEL)
    assert fits["converged"].all()
    for line_density in (moments["line_density"], fits["line_density"]):
        np.testing.assert_allclose(line_density, LINE_DENSITY, rtol=0.01)
    np.testing.assert_allclose(fits["centre"], 0.0, atol=0.1 * PIXEL)
    np.testing.assert_allclose(fits["width"], width, rtol=0.02)


def test_width_law_is_recovered():
    distance = np.linspace(120.0, 6000.0, 50)
    initial_width = 17.0
    # ŵ² = w₀² + w_ref² (d / d_ref)^{2b}
    width = np.sqrt(initial_width**2 + (150.0 * (distance / 1000.0) ** 0.85) ** 2)
    reference_width, exponent = fit_width_law(
        distance, width, reference_distance=1000.0, initial_width=initial_width
    )
    np.testing.assert_allclose(reference_width, 150.0, rtol=1e-6)
    np.testing.assert_allclose(exponent, 0.85, rtol=1e-6)


@pytest.mark.slow
def test_line_density_times_wind_is_the_cross_sectional_flux():
    # an axis-aligned ridge three pixels wide, the source on a pixel centre
    shape = (41, 30)
    transform = _grid(*shape, west=-0.5 * PIXEL, north=20.5 * PIXEL)
    image = _ridge(shape, transform, direction=0.0, width=3.0 * PIXEL)
    carrier = GeoArray(
        image, transform=transform, crs="EPSG:32629", fill_value=-9999, dims=("y", "x")
    )
    mask = GeoArray(
        np.ones(shape, dtype=bool),
        transform=transform,
        crs="EPSG:32629",
        fill_value=None,
        dims=("y", "x"),
    )
    wind = 3.5
    flux = geo.CrossSectionalFlux(
        source=(0.0, 0.0),
        wind_u=wind,
        wind_v=0.0,
        n_transects=10,
        transect_spacing_m=2 * PIXEL,
    )(carrier, mask)
    table = CrossPlumeProfile(
        source=(0.0, 0.0),
        direction=0.0,
        n_transects=10,
        transect_spacing_m=2 * PIXEL,
        half_length_m=20 * PIXEL,
    )(carrier)
    np.testing.assert_allclose(table["distance_m"], flux["distance_m"])
    np.testing.assert_allclose(
        wind * table["line_density_kg_m"], flux["flux_kg_s"], rtol=0.02
    )


# ── properties ──────────────────────────────────────────────────────────────


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    a=st.floats(-3.0, 3.0),
    b=st.floats(-3.0, 3.0),
)
def test_line_density_is_linear_in_the_image(seed, a, b):
    rng = np.random.default_rng(seed)
    shape = (16, 16)
    transform = _grid(*shape)
    first, second = rng.normal(size=(2, *shape))
    options = {
        "source": (-300.0, 0.0),
        "direction": 0.4,
        "n_transects": 5,
        "spacing": PIXEL,
        "half_length": 240.0,
    }

    def line_density(image):
        profiles = cross_plume_transects(image, transform, **options)
        return transect_moments(profiles)["line_density"]

    combined = line_density(a * first + b * second)
    expected = a * line_density(first) + b * line_density(second)
    np.testing.assert_allclose(combined, expected, rtol=1e-9, atol=1e-9)


@settings(max_examples=25, deadline=None)
@given(shift_row=st.integers(-3, 3), shift_col=st.integers(-3, 3))
def test_shifting_the_image_by_whole_pixels_shifts_the_centres(shift_row, shift_col):
    # φ = 0 and the source on a pixel centre, so every sample is on one
    shape = (60, 40)
    transform = _grid(*shape, west=-10.5 * PIXEL, north=30.5 * PIXEL)
    image = _ridge(shape, transform, direction=0.0, width=2.0 * PIXEL)
    options = {
        "source": (0.0, 0.0),
        "direction": 0.0,
        "n_transects": 15,
        "spacing": PIXEL,
        "half_length": 20 * PIXEL,
    }
    before = transect_moments(cross_plume_transects(image, transform, **options))
    # the same pixels, moved shift_col east and shift_row south
    moved = transform @ rasterio.Affine.translation(shift_col, shift_row)
    after = transect_moments(cross_plume_transects(image, moved, **options))
    # a row south is a pixel lower: c decreases by g
    np.testing.assert_allclose(
        after["centre"], before["centre"] - shift_row * PIXEL, atol=1e-9
    )
    np.testing.assert_allclose(after["width"], before["width"], rtol=1e-9)
    np.testing.assert_allclose(after["line_density"], before["line_density"], rtol=1e-9)


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_moments_ignore_nan_samples(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(4, 9))
    missing = rng.random(size=values.shape) < 0.3
    offset = PIXEL * np.arange(-4.0, 5.0)
    distance = PIXEL * np.arange(1.0, 5.0)
    with_nan = TransectProfiles(distance, offset, np.where(missing, np.nan, values))
    with_zero = TransectProfiles(distance, offset, np.where(missing, 0.0, values))
    for key, value in transect_moments(with_nan).items():
        np.testing.assert_array_equal(value, transect_moments(with_zero)[key])


# ── statistics ──────────────────────────────────────────────────────────────


@pytest.mark.validation
def test_signed_sum_is_unbiased_and_the_positive_part_is_not():
    # φ = 0, the source on a pixel centre: the samples are the pixels, so the
    # sample noise is the pixel noise
    shape = (41, 12)
    transform = _grid(*shape, west=-0.5 * PIXEL, north=20.5 * PIXEL)
    clean = _ridge(shape, transform, direction=0.0, width=3.0 * PIXEL)
    sigma = 0.1 * float(np.max(clean))
    options = {
        "source": (0.0, 0.0),
        "direction": 0.0,
        "n_transects": 1,
        "spacing": 5 * PIXEL,
        "half_length": 20 * PIXEL,
    }
    signal = cross_plume_transects(clean, transform, **options).values[0]
    n_sample = signal.shape[0]
    rng = np.random.default_rng(0)
    signed, positive = [], []
    for _ in range(500):
        noisy = clean + sigma * rng.normal(size=shape)
        values = cross_plume_transects(noisy, transform, **options).values[0]
        signed.append(PIXEL * np.sum(values))
        positive.append(PIXEL * np.sum(np.maximum(values, 0.0)))
    signed, positive = np.asarray(signed), np.asarray(positive)
    clean_sum = PIXEL * np.sum(signal)

    # E[q̂] = q
    standard_error = signed.std(ddof=1) / math.sqrt(signed.size)
    assert abs(signed.mean() - clean_sum) < 3.0 * standard_error

    # E[max(v + ε, 0)] = v Φ(v/σ) + σ φ(v/σ): σ / √(2π) where v = 0
    z = signal / sigma
    expected = PIXEL * np.sum(signal * stats.norm.cdf(z) + sigma * stats.norm.pdf(z))
    standard_error = positive.std(ddof=1) / math.sqrt(positive.size)
    assert abs(positive.mean() - expected) < 3.0 * standard_error
    # the bias, at most S δ σ / √(2π), is many standard errors
    bias = positive.mean() - clean_sum
    assert bias > 10.0 * standard_error
    assert bias < n_sample * PIXEL * sigma / math.sqrt(2.0 * math.pi)


@pytest.mark.validation
def test_positive_part_bias_on_pure_noise_is_s_delta_sigma_over_root_two_pi():
    shape = (41, 12)
    transform = _grid(*shape, west=-0.5 * PIXEL, north=20.5 * PIXEL)
    sigma = 1e-5
    options = {
        "source": (0.0, 0.0),
        "direction": 0.0,
        "n_transects": 1,
        "spacing": 5 * PIXEL,
        "half_length": 20 * PIXEL,
    }
    rng = np.random.default_rng(1)
    positive = []
    for _ in range(500):
        values = cross_plume_transects(
            sigma * rng.normal(size=shape), transform, **options
        ).values[0]
        positive.append(PIXEL * np.sum(np.maximum(values, 0.0)))
    positive = np.asarray(positive)
    # S δ σ / √(2π)
    predicted = 41 * PIXEL * sigma / math.sqrt(2.0 * math.pi)
    standard_error = positive.std(ddof=1) / math.sqrt(positive.size)
    assert abs(positive.mean() - predicted) < 3.0 * standard_error


# ── smoothing ───────────────────────────────────────────────────────────────


def _bent_scene(noise_fraction, seed=0):
    """A ridge with a 200 m, 3 km sine centreline along 30°, widening downwind."""
    direction = math.radians(30.0)
    shape = (130, 150)
    transform = _grid(*shape, west=-1800.0, north=5400.0)

    def centreline(distance):
        # m(d) = A sin(2π d / λ) for d > 0
        return np.where(
            distance > 0.0, 200.0 * np.sin(2.0 * np.pi * distance / 3000.0), 0.0
        )

    def width(distance):
        # w(d) = 90 + 0.03 d
        return 90.0 + 0.03 * np.maximum(distance, 0.0)

    clean = _ridge(
        shape,
        transform,
        direction=direction,
        width=width,
        centreline=centreline,
        cut=True,
    )
    rng = np.random.default_rng(seed)
    image = clean + noise_fraction * float(np.max(clean)) * rng.normal(size=shape)
    profiles = cross_plume_transects(
        image,
        transform,
        source=(0.0, 0.0),
        direction=direction,
        n_transects=100,
        spacing=PIXEL,
        half_length=1200.0,
    )
    return profiles, centreline(profiles.distance)


@pytest.mark.parametrize("noise_fraction", [0.0, 0.1])
def test_smoothed_sinusoidal_centreline_is_within_half_a_pixel(noise_fraction):
    profiles, truth = _bent_scene(noise_fraction)
    assert profiles.values.shape[0] == 100
    fits = fit_transect_gaussians(profiles, min_width=PIXEL)
    smoothed = smooth_centreline(fits["centre"], fits["width"], fits["converged"])
    rmse = float(np.sqrt(np.mean((smoothed - truth) ** 2)))
    assert rmse < 0.5 * PIXEL


def test_rejected_centres_take_the_smoothed_line_not_zero():
    profiles, truth = _bent_scene(0.0)
    fits = fit_transect_gaussians(profiles, min_width=PIXEL)
    centre = fits["centre"].copy()
    # five fits gone wrong, a kilometre off
    wrong = np.array([12, 31, 47, 60, 85])
    centre[wrong] += 1000.0
    smoothed = smooth_centreline(centre, fits["width"], fits["converged"])
    spiire = smooth_centreline(
        centre, fits["width"], fits["converged"], rejected="zero"
    )
    error = float(np.sqrt(np.mean((smoothed - truth) ** 2)))
    spiire_error = float(np.sqrt(np.mean((spiire - truth) ** 2)))
    assert error < 0.5 * PIXEL
    assert spiire_error > error


def test_transects_without_signal_are_failed_fits():
    offset = PIXEL * np.arange(-10.0, 11.0)
    signal = np.exp(-0.5 * (offset / 120.0) ** 2)
    values = np.stack([signal, np.zeros_like(offset), -signal])
    profiles = TransectProfiles(PIXEL * np.arange(1.0, 4.0), offset, values)
    fits = fit_transect_gaussians(profiles, min_width=PIXEL)
    assert fits["converged"].tolist() == [True, False, False]
    for key in ("line_density", "centre", "width"):
        assert np.isfinite(fits[key][0])
        assert np.isnan(fits[key][1:]).all()


def test_failed_fits_are_interpolated():
    centre = np.linspace(0.0, 300.0, 40)
    converged = np.ones(40, dtype=bool)
    converged[[5, 6, 20]] = False
    broken = centre.copy()
    broken[[5, 6, 20]] = np.nan
    smoothed = smooth_centreline(broken, np.full(40, 100.0), converged, n_iter=0)
    reference = smooth_centreline(centre, np.full(40, 100.0), converged, n_iter=0)
    np.testing.assert_allclose(smoothed, reference, rtol=1e-12)


# ── the carrier operator ────────────────────────────────────────────────────


def _carrier(image, transform, crs="EPSG:32629"):
    return GeoArray(
        image, transform=transform, crs=crs, fill_value=-9999, dims=("y", "x")
    )


def test_operator_returns_the_transect_table():
    shape = (140, 140)
    transform = _grid(*shape)
    direction = math.radians(30.0)
    image = _ridge(shape, transform, direction=direction, width=4 * PIXEL, cut=True)
    table = CrossPlumeProfile(
        source=(0.0, 0.0),
        direction=direction,
        n_transects=20,
        transect_spacing_m=120.0,
        half_length_m=1500.0,
    )(_carrier(image, transform))
    assert list(table.columns) == [
        "geometry",
        "distance_m",
        "line_density_kg_m",
        "centre_m",
        "width_m",
        "centre_smooth_m",
        "converged",
    ]
    assert table.crs == "EPSG:32629"
    assert len(table) == 20
    assert table["converged"].all()
    # each LineString spans the transect, through s₀ + d_k e_∥
    line = table.geometry.iloc[0]
    np.testing.assert_allclose(line.length, 3000.0)
    np.testing.assert_allclose(
        np.asarray(line.centroid.coords[0]),
        120.0 * np.array([math.cos(direction), math.sin(direction)]),
        atol=1e-9,
    )
    # past the cut at the source, the line density is flat
    np.testing.assert_allclose(
        table["line_density_kg_m"].iloc[2:], LINE_DENSITY, rtol=0.01
    )


def test_operator_nodata_becomes_nan_samples():
    shape = (60, 60)
    transform = _grid(*shape)
    image = _ridge(shape, transform, direction=0.0, width=2 * PIXEL)
    image[30, 40] = -9999.0
    profiles = cross_plume_transects(
        np.where(image == -9999.0, np.nan, image),
        transform,
        source=(0.0, 0.0),
        direction=0.0,
        n_transects=20,
        spacing=PIXEL,
        half_length=600.0,
    )
    missing = ~np.isfinite(profiles.values)
    # within two pixels of the nodata pixel, in both directions
    assert missing.any()
    assert np.isfinite(profiles.values[:7]).all()
    table = CrossPlumeProfile(
        source=(0.0, 0.0),
        direction=0.0,
        n_transects=20,
        transect_spacing_m=PIXEL,
        half_length_m=600.0,
    )(_carrier(image, transform))
    expected = transect_moments(profiles)["line_density"]
    np.testing.assert_allclose(table["line_density_kg_m"], expected, rtol=1e-12)


def test_operator_config_round_trips_through_json():
    op = CrossPlumeProfile(
        source=(1.0, 2.0),
        direction=0.3,
        n_transects=7,
        transect_spacing_m=90.0,
        half_length_m=900.0,
        order=1,
    )
    config = json.loads(json.dumps(op.get_config()))
    rebuilt = CrossPlumeProfile(**config)
    assert rebuilt.get_config() == config
    assert rebuilt.source == (1.0, 2.0)


def test_exported_lazily_from_geo():
    assert geo.CrossPlumeProfile is CrossPlumeProfile
    assert geo.cross_plume_transects is transects.cross_plume_transects
    assert geo.fit_width_law is fit_width_law


# ── errors and dtypes ───────────────────────────────────────────────────────


def test_parameter_errors():
    shape = (20, 20)
    transform = _grid(*shape)
    image = np.zeros(shape)
    options = {
        "source": (0.0, 0.0),
        "direction": 0.0,
        "n_transects": 3,
        "spacing": PIXEL,
        "half_length": 300.0,
    }
    with pytest.raises(ParameterError, match="below one pixel"):
        cross_plume_transects(image, transform, **{**options, "spacing": 30.0})
    with pytest.raises(ParameterError, match="> 0"):
        cross_plume_transects(image, transform, **{**options, "half_length": 0.0})
    with pytest.raises(ParameterError, match="below one sample"):
        cross_plume_transects(image, transform, **{**options, "half_length": 20.0})
    with pytest.raises(ParameterError, match="n_transects"):
        cross_plume_transects(image, transform, **{**options, "n_transects": 0})
    with pytest.raises(ParameterError, match="order"):
        cross_plume_transects(image, transform, **options, order=7)
    with pytest.raises(ParameterError, match="min_width"):
        fit_transect_gaussians(
            cross_plume_transects(image, transform, **options), min_width=0.0
        )
    with pytest.raises(ParameterError, match="window"):
        smooth_centreline(np.zeros(5), np.ones(5), np.ones(5, dtype=bool), window=4)
    # the type-check hook of the test suite rejects the Literal first
    with pytest.raises((ParameterError, TypeCheckError), match="rejected"):
        smooth_centreline(
            np.zeros(5), np.ones(5), np.ones(5, dtype=bool), rejected="median"
        )
    with pytest.raises(ParameterError, match="reference_distance"):
        fit_width_law(np.ones(3), np.ones(3), reference_distance=0.0)


def test_domain_errors():
    shape = (20, 20)
    transform = _grid(*shape)
    with pytest.raises(DomainError, match="no transect"):
        cross_plume_transects(
            np.zeros(shape),
            transform,
            source=(1e5, 1e5),
            direction=0.0,
            n_transects=3,
            spacing=PIXEL,
            half_length=300.0,
        )
    with pytest.raises(DomainError, match="no fit converged"):
        smooth_centreline(np.zeros(5), np.ones(5), np.zeros(5, dtype=bool))
    with pytest.raises(DomainError, match="fewer than two"):
        fit_width_law(
            np.array([100.0, 200.0]),
            np.array([5.0, 50.0]),
            reference_distance=1000.0,
            initial_width=10.0,
        )


def test_operator_rejects_what_the_cross_sectional_flux_rejects():
    op = CrossPlumeProfile(
        source=(0.0, 0.0),
        direction=0.0,
        n_transects=2,
        transect_spacing_m=1e-4,
        half_length_m=1e-3,
    )
    geographic = GeoArray(
        np.zeros((10, 10)),
        transform=rasterio.Affine(1e-4, 0.0, 0.0, 0.0, -1e-4, 0.001),
        crs="EPSG:4326",
        dims=("y", "x"),
    )
    with pytest.raises(ValueError, match=r"CrossPlumeProfile.*geographic.*Reproject"):
        op(geographic)
    with pytest.raises(TypeError):
        op(np.zeros((10, 10)))


def test_float32_in_gives_float32_out():
    shape = (60, 60)
    transform = _grid(*shape)
    image = _ridge(shape, transform, direction=0.2, width=2 * PIXEL).astype(np.float32)
    profiles = cross_plume_transects(
        image,
        transform,
        source=(0.0, 0.0),
        direction=0.2,
        n_transects=10,
        spacing=PIXEL,
        half_length=600.0,
    )
    assert profiles.values.dtype == np.float32
    assert profiles.offset.dtype == profiles.distance.dtype == np.float32
    assert all(
        value.dtype == np.float32 for value in transect_moments(profiles).values()
    )
    fits = fit_transect_gaussians(profiles, min_width=PIXEL)
    assert fits["centre"].dtype == fits["width"].dtype == np.float32
    smoothed = smooth_centreline(fits["centre"], fits["width"], fits["converged"])
    assert smoothed.dtype == np.float32
