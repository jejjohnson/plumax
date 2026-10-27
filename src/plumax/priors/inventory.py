"""Emission-inventory prior mean on the analysis grid.

Every inversion starts from a prior mean emission. A gridded inventory
(EDGAR, GFEI, EPA) gives one, but on a lat/lon raster in its own units and
at its own resolution. :func:`load_inventory` reads it through
:func:`xrreader.open_ch4_inventory` (which converts to a flux ``F`` in
kg m⁻² s⁻¹) and averages it over each analysis T-cell. Inventories differ
by about 2× in well-studied basins, so the file is always an explicit
choice of the caller.

Averaging
---------
The prior for analysis cell ``c`` is its area-mean flux,
``F̄_c = (1/|c|) ∫_c F dA``. Each cell is sampled on an ``s × s`` lattice
of points evenly spaced in frame metres (cell-centred sub-cells), each
point takes the flux of the inventory cell containing it, and ``F̄_c`` is
their mean. That is the area-weighted average up to the sampling error,
which falls as ``1/s`` at inventory-cell edges; the frame is conformal and
locally near-equidistant, so equal frame areas are equal ground areas.

Needs the ``io`` extra (``pip install 'plumax[io]'``).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met.frames import LocalFrame


def load_inventory(
    path: str | Path,
    *,
    frame: LocalFrame,
    grid: PlumeGrid3D,
    variable: str | None = None,
    units: str | None = None,
    time_index: int = 0,
    supersample: int = 8,
) -> Float[Array, "ny nx"]:
    """Area-mean inventory flux over each analysis T-cell.

    Args:
        path: Inventory NetCDF (one sector or a total).
        frame: Frame the analysis grid is laid out in.
        grid: Analysis grid, in ``frame`` metres.
        variable: Flux variable name; inferred by the reader when ``None``.
        units: Override of the file's units attribute.
        time_index: Time step to use when the file has a ``time`` axis.
        supersample: Sub-points per cell edge, ``s``.

    Returns:
        Prior mean flux [kg m⁻² s⁻¹] on the horizontal T-cells,
        ``(ny, nx)``.

    Raises:
        ImportError: If the ``io`` extra (xrreader) is not installed.
        ValueError: If a sub-point falls outside the inventory raster.
    """
    try:
        from xrreader import open_ch4_inventory
    except ImportError as err:
        raise ImportError(
            "load_inventory reads through xrreader; install it with "
            "`pip install 'plumax[io]'`"
        ) from err
    flux = open_ch4_inventory(path, variable=variable, units=units)["ch4_emission_flux"]
    if "time" in flux.dims:
        flux = flux.isel(time=time_index)
    flux = flux.transpose("lat", "lon")
    lat_edges = _edges(flux["lat"].values)
    lon_edges = _edges(flux["lon"].values)
    values = flux.values

    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5  # in cell units
    sub_x = (x[:, None] + offsets[None, :] * grid.dx).ravel()  # (nx·s,)
    sub_y = (y[:, None] + offsets[None, :] * grid.dy).ravel()  # (ny·s,)
    xx, yy = np.meshgrid(sub_x, sub_y)  # (ny·s, nx·s)
    lat, lon = frame.to_latlon(xx, yy)
    j = np.searchsorted(lat_edges, lat, side="right") - 1
    i = np.searchsorted(lon_edges, lon, side="right") - 1
    if (
        np.any(j < 0)
        or np.any(j >= lat_edges.size - 1)
        or np.any(i < 0)
        or np.any(i >= lon_edges.size - 1)
    ):
        raise ValueError(
            "load_inventory: the analysis grid extends beyond the inventory raster"
        )
    samples = values[j, i].reshape(y.size, supersample, x.size, supersample)
    return jnp.asarray(samples.mean(axis=(1, 3)), dtype=grid.x.dtype)


def _edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges of a regular ascending axis from its centres."""
    step = float(centres[1] - centres[0]) if centres.size > 1 else 1.0
    return np.concatenate([centres - 0.5 * step, centres[-1:] + 0.5 * step])
