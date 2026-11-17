"""Toy GeoArray carriers for the plumax.geo tests, from geotoolz's test helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


rasterio = pytest.importorskip("rasterio")
GeoArray = pytest.importorskip("geoarray").GeoArray
SpectralAxis = pytest.importorskip("geoarray.coords").SpectralAxis

#: dims of a carrier by its rank
DIMS = {2: ("y", "x"), 3: ("band", "y", "x"), 4: ("time", "band", "y", "x")}


#: 10 m UTM grid anchored in zone 29N, arbitrary but stable.
DEFAULT_TRANSFORM = rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)
DEFAULT_CRS = "EPSG:32629"

#: Named toy grids, ``name -> (transform, crs)``: 10 m square pixels, 10 m by
#: 20 m pixels, the 10 m grid rotated by 30 degrees, and a column shear.
TOY_GRIDS: dict[str, tuple[Any, str]] = {
    "utm": (DEFAULT_TRANSFORM, DEFAULT_CRS),
    "non_square": (
        rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -20.0, 4_000_000.0),
        DEFAULT_CRS,
    ),
    "rotated": (
        rasterio.Affine.translation(500_000.0, 4_000_000.0)
        * rasterio.Affine.rotation(30.0)
        * rasterio.Affine.scale(10.0, -10.0),
        DEFAULT_CRS,
    ),
    "sheared": (
        rasterio.Affine(10.0, 5.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        DEFAULT_CRS,
    ),
}


def fill_pixel_mask(shape: tuple[int, ...]) -> np.ndarray:
    """``(H, W)`` mask of the first and the last pixel of the grid."""
    mask = np.zeros(shape[-2:], dtype=bool)
    mask[0, 0] = True
    mask[-1, -1] = True
    return mask


def toy_geoarray(
    values: np.ndarray,
    *,
    fill_value: Any = -9999,
    attrs: dict[str, Any] | None = None,
    band_names: tuple[str, ...] | None = None,
    with_fill_pixels: bool = False,
    grid: str = "utm",
) -> Any:
    """A GeoArray on a :data:`TOY_GRIDS` grid; optionally the corners filled."""
    transform, crs = TOY_GRIDS[grid]
    if with_fill_pixels:
        values = np.array(values, copy=True)
        values[..., fill_pixel_mask(values.shape)] = fill_value
    return GeoArray(
        values,
        transform=transform,
        crs=crs,
        fill_value=fill_value,
        attrs=attrs or {},
        dims=DIMS[np.ndim(values)],
        coords=() if band_names is None else (SpectralAxis(band_names),),
    )


def time_stack(
    shape: tuple[int, int, int, int] = (2, 3, 4, 4), *, seed: int = 0
) -> Any:
    """A ``(T, C, H, W)`` GeoArray time stack with values in ``[0.01, 1)``."""
    band_names = tuple(f"b{i}" for i in range(shape[1]))
    values = np.random.default_rng(seed).uniform(0.01, 1.0, shape)
    return toy_geoarray(values, band_names=band_names)
