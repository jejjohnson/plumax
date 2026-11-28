"""Tests for ``plumax.met.frames``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from plumax.errors import DomainError
from plumax.les_fvm.grid import make_grid
from plumax.met import LocalFrame


pytest.importorskip("pyproj")

# A Permian Basin site, the running example across the Tier 0 loaders.
LAT0, LON0 = 31.95, -102.15


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


def test_grid_indices_on_the_grid_edges():
    # the frame origin maps to exactly (0, 0): the lower edge of a grid that
    # starts there (cell 0), and the upper edge of one that ends there (-1)
    frame = LocalFrame.at(LAT0, LON0)
    above = make_grid((0.0, 400.0, 4), (0.0, 300.0, 4), (0.0, 50.0, 4))
    below = make_grid((-400.0, 0.0, 4), (-300.0, 0.0, 4), (0.0, 50.0, 4))
    i, j = frame.grid_indices(LAT0, LON0, above)
    assert (int(i), int(j)) == (0, 0)
    i, j = frame.grid_indices(LAT0, LON0, below)
    assert (int(i), int(j)) == (-1, -1)


def test_a_crs_not_in_metres_is_refused_at_construction():
    # EPSG:4326 is geographic: its offsets were degrees labelled as metres
    with pytest.raises(DomainError, match="not a projected CRS in metres"):
        LocalFrame(LAT0, LON0, 4326)


# ---- properties -------------------------------------------------------------


offsets_m = st.floats(min_value=-50_000.0, max_value=50_000.0)


@settings(max_examples=50, deadline=None)
@given(x=offsets_m, y=offsets_m)
def test_roundtrip_property(x, y):
    frame = LocalFrame.at(LAT0, LON0)
    lat, lon = frame.to_latlon(x, y)
    x2, y2 = frame.to_xy(lat, lon)
    assert np.hypot(float(x2) - x, float(y2) - y) < 1e-3


@settings(max_examples=50, deadline=None)
@given(lat=st.floats(-60.0, 70.0), lon=st.floats(-179.0, 179.0))
def test_east_north_axes_are_orthonormal_everywhere(lat, lon):
    frame = LocalFrame.at(lat, lon)
    east, north = frame.east_north_axes(lat, lon)
    np.testing.assert_allclose(
        [np.linalg.norm(east), np.linalg.norm(north)], 1.0, rtol=1e-9
    )
    # UTM is conformal: the axes stay perpendicular
    assert abs(float(east @ north)) < 1e-5


def test_config_roundtrip():
    frame = LocalFrame.at(LAT0, LON0)
    assert LocalFrame.from_config(frame.get_config()) == frame
