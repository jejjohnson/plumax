"""Tests for ``plumax.met.frames``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.les_fvm.grid import make_grid
from plumax.met import LocalFrame
from plumax.met.frames import utm_epsg


pytest.importorskip("pyproj")

# A Permian Basin site, the running example across the Tier 0 loaders.
LAT0, LON0 = 31.95, -102.15


@pytest.mark.parametrize(
    ("lat", "lon", "epsg"),
    [
        (LAT0, LON0, 32613),  # 102.15° W → zone 13 north
        (-23.5, 133.9, 32753),  # central Australia → zone 53 south
        (0.0, 179.9, 32660),  # last zone
        (10.0, -180.0, 32601),  # first zone
    ],
)
def test_utm_zone(lat, lon, epsg):
    assert utm_epsg(lat, lon) == epsg


def test_utm_rejects_polar_latitudes():
    with pytest.raises(ValueError, match="UTM"):
        utm_epsg(85.0, 0.0)


def test_roundtrip_mm_precision():
    frame = LocalFrame.at(LAT0, LON0)
    lat = LAT0 + np.linspace(-0.5, 0.5, 11)[:, None]
    lon = LON0 + np.linspace(-0.5, 0.5, 13)[None, :]
    x, y = frame.to_xy(lat, lon)
    lat2, lon2 = frame.to_latlon(x, y)
    x2, y2 = frame.to_xy(lat2, lon2)
    assert x.shape == (11, 13)
    assert np.max(np.hypot(x2 - x, y2 - y)) < 1e-3  # < 1 mm


def test_origin_is_zero_and_scale_is_metric():
    frame = LocalFrame.at(LAT0, LON0)
    x0, y0 = frame.to_xy(LAT0, LON0)
    assert abs(float(x0)) < 1e-6 and abs(float(y0)) < 1e-6
    # 0.01° of latitude is ~1.109 km at 32° N; UTM scale is within 0.1 % here.
    _, y = frame.to_xy(LAT0 + 0.01, LON0)
    assert float(y) == pytest.approx(1109.0, rel=2e-3)


def test_east_north_axes_are_unit_and_carry_convergence():
    frame = LocalFrame.at(LAT0, LON0)
    east, north = frame.east_north_axes(LAT0, LON0)
    np.testing.assert_allclose(np.linalg.norm(east), 1.0, rtol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(north), 1.0, rtol=1e-9)
    assert abs(float(east @ north)) < 1e-6  # conformal: still orthogonal
    # 102.15° W is 2.85° east of zone 13's central meridian (105° W): grid
    # north leans from true north by about sin(φ)·Δλ ≈ 1.5°, not zero.
    angle = np.degrees(np.arctan2(north[0], north[1]))
    assert 1.0 < abs(angle) < 2.0


def test_grid_indices_inside_and_outside():
    frame = LocalFrame.at(LAT0, LON0)
    grid = make_grid(
        (0.0, 400.0, 4), (0.0, 300.0, 4), (0.0, 50.0, 4), dtype=jnp.float64
    )
    # T-points at x = 50, 150, 250, 350 (dx 100); y = 37.5 … 262.5 (dy 75)
    lat, lon = frame.to_latlon(
        np.array([60.0, 399.0, -10.0]), np.array([140.0, 10.0, 50.0])
    )
    i, j = frame.grid_indices(lat, lon, grid)
    np.testing.assert_array_equal(i, [0, 3, -1])
    np.testing.assert_array_equal(j, [1, 0, -1])
