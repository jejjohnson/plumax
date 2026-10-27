"""Tests for the inventory prior mean and the per-tile background."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest
import xarray as xr
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

from plumax.errors import GridMismatchError
from plumax.les_fvm.grid import make_grid
from plumax.met import LocalFrame
from plumax.priors import load_inventory, per_tile_background


pytest.importorskip("xrreader")
Geod = pytest.importorskip("pyproj").Geod

SITE = (31.95, -102.15)  # Permian Basin
F_BASIN = 2.0e-9  # kg m⁻² s⁻¹
STEP = 0.1  # degrees, EDGAR's resolution
# The "basin": 2 × 2 inventory cells (0.2° × 0.2°) around the site.
LAT_BOX = (31.9, 32.1)
LON_BOX = (-102.2, -102.0)


def write_inventory(
    path: Path, *, units: str = "kg m-2 s-1", scale: float = 1.0
) -> Path:
    lat = np.arange(31.05, 32.96, STEP)  # centres; edges on whole tenths
    lon = np.arange(-102.95, -101.04, STEP) + 360.0  # 0–360, as some files store it
    lat_c, lon_c = np.meshgrid(lat, lon - 360.0, indexing="ij")
    inside = (
        (lat_c > LAT_BOX[0])
        & (lat_c < LAT_BOX[1])
        & (lon_c > LON_BOX[0])
        & (lon_c < LON_BOX[1])
    )
    flux = np.where(inside, F_BASIN * scale, 0.0)
    xr.Dataset(
        {"fluxes": (("lat", "lon"), flux, {"units": units})},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def frame() -> LocalFrame:
    return LocalFrame.at(*SITE)


@pytest.fixture(scope="module")
def grid():
    # 60 km × 60 km of 1 km cells centred on the site
    return make_grid(
        (-30_000.0, 30_000.0, 60),
        (-30_000.0, 30_000.0, 60),
        (0.0, 100.0, 4),
        dtype=jnp.float64,
    )


def test_basin_total_matches_geodesic_area(tmp_path, frame, grid):
    prior = load_inventory(
        write_inventory(tmp_path / "edgar.nc"), frame=frame, grid=grid
    )
    assert prior.shape == (60, 60)
    total = float(jnp.sum(prior)) * grid.dx * grid.dy  # kg s⁻¹

    lons = [LON_BOX[0], LON_BOX[1], LON_BOX[1], LON_BOX[0]]
    lats = [LAT_BOX[0], LAT_BOX[0], LAT_BOX[1], LAT_BOX[1]]
    area, _ = Geod(ellps="WGS84").polygon_area_perimeter(lons, lats)
    expected = F_BASIN * abs(area)
    # 0.1 % here: the UTM scale factor plus the sub-sampling of the basin edge
    assert total == pytest.approx(expected, rel=3e-3)
    # Cells well inside the basin carry the full flux, cells far outside none.
    assert float(prior[30, 30]) == pytest.approx(F_BASIN, rel=1e-12)
    assert float(prior[0, 0]) == 0.0


def test_gfei_units_are_converted_by_the_reader(tmp_path, frame, grid):
    # 1 Mg km⁻² a⁻¹ = 1e-3 kg m⁻² / 31 557 600 s; write F_BASIN in those units
    per_year = 1.0e3 / 1.0e6 / (365.25 * 86_400.0)
    path = write_inventory(
        tmp_path / "gfei.nc", units="Mg km-2 a-1", scale=1.0 / per_year
    )
    prior = load_inventory(path, frame=frame, grid=grid)
    assert float(prior[30, 30]) == pytest.approx(F_BASIN, rel=1e-9)


def test_grid_beyond_the_raster_raises(tmp_path, frame):
    huge = make_grid(
        (-300_000.0, 300_000.0, 6),
        (-30_000.0, 30_000.0, 4),
        (0.0, 100.0, 4),
        dtype=jnp.float64,
    )
    with pytest.raises(ValueError, match="beyond the inventory raster"):
        load_inventory(write_inventory(tmp_path / "e.nc"), frame=frame, grid=huge)


def test_global_raster_wraps_at_the_antimeridian(tmp_path, grid):
    # centres -180 … 179: edges [-180.5, 179.5], so 179.75°E sits in the
    # cell centred at -180°, past the last edge
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-180.0, 180.0, 1.0)
    flux = np.zeros((lat.size, lon.size))
    flux[:, 0] = F_BASIN  # the cell centred at -180°
    path = tmp_path / "global.nc"
    xr.Dataset(
        {"fluxes": (("lat", "lon"), flux, {"units": "kg m-2 s-1"})},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)
    # a 1 km grid around 179.75°E stays within the -180° cell (179.5 – 180.5)
    small = make_grid(
        (-5_000.0, 5_000.0, 10),
        (-5_000.0, 5_000.0, 10),
        (0.0, 100.0, 4),
        dtype=jnp.float64,
    )
    prior = load_inventory(path, frame=LocalFrame.at(0.0, 179.75), grid=small)
    np.testing.assert_allclose(prior, F_BASIN, rtol=1e-12)


def write_regional_across_antimeridian(path: Path) -> Path:
    # centres 179, 180, 181 °E, which the reader stores as -180, -179, 179
    lat = np.arange(-2.5, 3.0, 1.0)
    lon = np.array([179.0, 180.0, 181.0])
    flux = np.broadcast_to(np.array([1.0, 2.0, 3.0]) * F_BASIN, (lat.size, 3))
    xr.Dataset(
        {"fluxes": (("lat", "lon"), np.array(flux), {"units": "kg m-2 s-1"})},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)
    return path


def test_regional_raster_across_the_antimeridian(tmp_path):
    path = write_regional_across_antimeridian(tmp_path / "pacific.nc")
    small = make_grid(
        (-5_000.0, 5_000.0, 10),
        (-5_000.0, 5_000.0, 10),
        (0.0, 100.0, 4),
        dtype=jnp.float64,
    )
    # inside each of the three cells
    for lon, level in ((179.0, 1.0), (-180.0, 2.0), (-179.0, 3.0)):
        prior = load_inventory(path, frame=LocalFrame.at(0.0, lon), grid=small)
        np.testing.assert_allclose(prior, level * F_BASIN, rtol=1e-12)
    # far outside: the gap between -178.5° and 178.5° is not coverage
    with pytest.raises(ValueError, match="beyond the inventory raster"):
        load_inventory(path, frame=LocalFrame.at(0.0, -100.0), grid=small)


def test_per_tile_background_is_the_tile_median():
    # Tile A (x < 10 km): 1880 ppb with one plume pixel; tile B: 1900 ppb.
    receptors = np.array(
        [
            [1_000.0, 0.0],
            [2_000.0, 0.0],
            [3_000.0, 0.0],
            [12_000.0, 0.0],
            [15_000.0, 0.0],
        ]
    )
    columns = np.array([1880.0, 1880.0, 1960.0, 1900.0, 1900.0])
    background = per_tile_background(columns, receptors, tile_m=10_000.0)
    np.testing.assert_array_equal(background, [1880.0, 1880.0, 1880.0, 1900.0, 1900.0])
    # the jaxtyping hook catches the mismatch first in tests; without it,
    # GridMismatchError does
    with pytest.raises((GridMismatchError, jaxtyping.TypeCheckError)):
        per_tile_background(columns, receptors[:3])


@given(
    positions=hnp.arrays(np.float64, (12, 2), elements=st.floats(-50_000.0, 50_000.0)),
    level=st.floats(1700.0, 2100.0),
    seed=st.integers(0, 2**16),
)
def test_per_tile_background_properties(positions, level, seed):
    # a constant field is its own background
    flat = np.full(12, level)
    np.testing.assert_array_equal(per_tile_background(flat, positions), flat)
    # the result follows the receptors under any reordering
    columns = level + np.arange(12.0)
    order = np.random.default_rng(seed).permutation(12)
    shuffled = per_tile_background(columns[order], positions[order])
    np.testing.assert_array_equal(
        shuffled, per_tile_background(columns, positions)[order]
    )
