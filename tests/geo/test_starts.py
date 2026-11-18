"""Starting values of an image fit, from a mask and a source (#271).

Needs the ``[geo]`` extra (in the dev group).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio

from plumax import geo
from plumax._src.errors import ParameterError
from plumax.errors import DomainError
from plumax.geo.starts import plume_direction, source_offset_grid


PIXEL = 60.0
# a north-up 60 m grid of 81 × 81 pixels centred on the origin
TRANSFORM = rasterio.Affine(PIXEL, 0.0, -40.5 * PIXEL, 0.0, -PIXEL, 40.5 * PIXEL)
SHAPE = (81, 81)


def _straight_plume(direction, length=1800.0, half_width=90.0):
    """Mask and enhancement of a straight plume from the origin."""
    rows, cols = np.indices(SHAPE, dtype=float)
    xs = TRANSFORM.c + PIXEL * (cols + 0.5)
    ys = TRANSFORM.f - PIXEL * (rows + 0.5)
    # d = sᵀ e_∥,  c = sᵀ e_⊥
    downwind = xs * math.cos(direction) + ys * math.sin(direction)
    crosswind = -xs * math.sin(direction) + ys * math.cos(direction)
    mask = (downwind > 0.0) & (downwind < length) & (np.abs(crosswind) < half_width)
    enhancement = np.exp(-0.5 * (crosswind / 60.0) ** 2) * (downwind > 0.0)
    return mask, enhancement


def test_exported_lazily_from_geo():
    assert geo.plume_direction is plume_direction
    assert geo.source_offset_grid is source_offset_grid


@pytest.mark.parametrize("degrees", [20.0, 115.0, -160.0, -70.0, 90.0, 180.0])
def test_direction_of_a_straight_plume_in_every_quadrant(degrees):
    direction = math.radians(degrees)
    mask, enhancement = _straight_plume(direction)
    for weights in (None, enhancement):
        angle = plume_direction(mask, TRANSFORM, (0.0, 0.0), weights=weights)
        # wrap the difference into (−π, π]
        error = math.atan2(math.sin(angle - direction), math.cos(angle - direction))
        assert abs(math.degrees(error)) < 1.0


def test_direction_errors():
    empty = np.zeros(SHAPE, dtype=bool)
    with pytest.raises(DomainError, match="empty"):
        plume_direction(empty, TRANSFORM, (0.0, 0.0))
    # a disc around the source: its centroid is the source
    rows, cols = np.indices(SHAPE, dtype=float)
    xs = TRANSFORM.c + PIXEL * (cols + 0.5)
    ys = TRANSFORM.f - PIXEL * (rows + 0.5)
    disc = np.hypot(xs, ys) < 600.0
    with pytest.raises(DomainError, match="within one pixel"):
        plume_direction(disc, TRANSFORM, (0.0, 0.0))
    mask, _ = _straight_plume(0.3)
    with pytest.raises(DomainError, match="≥ 0"):
        plume_direction(mask, TRANSFORM, (0.0, 0.0), weights=-np.ones(SHAPE))
    with pytest.raises(DomainError, match="sum to zero"):
        plume_direction(mask, TRANSFORM, (0.0, 0.0), weights=np.zeros(SHAPE))


def test_offset_grid_is_centred_on_the_source():
    source = (500.0, -250.0)
    grid = source_offset_grid(source, pixel_size=PIXEL, n=5, step=2)
    assert grid.shape == (25, 2)
    np.testing.assert_array_equal(grid[12], source)
    # offsets of step · g · {−2, …, 2} on each axis
    offsets = 2 * PIXEL * np.arange(-2.0, 3.0)
    np.testing.assert_array_equal(np.unique(grid[:, 0]), source[0] + offsets)
    np.testing.assert_array_equal(np.unique(grid[:, 1]), source[1] + offsets)
    # x runs first
    np.testing.assert_array_equal(grid[:5, 0], source[0] + offsets[0])


def test_offset_grid_errors():
    for n in (0, 2, -3):
        with pytest.raises(ParameterError, match="odd"):
            source_offset_grid((0.0, 0.0), pixel_size=PIXEL, n=n)
    with pytest.raises(ParameterError, match="step"):
        source_offset_grid((0.0, 0.0), pixel_size=PIXEL, step=0)
    with pytest.raises(ParameterError, match="pixel_size"):
        source_offset_grid((0.0, 0.0), pixel_size=0.0)
