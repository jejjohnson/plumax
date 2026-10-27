"""Emission-inventory prior mean: a gridded inventory's area-mean flux per T-cell.

Inventories (EDGAR, GFEI, EPA) are lat/lon rasters in their own units and
resolution; they differ by about 2× in well-studied basins, so the file is
always an explicit choice of the caller. The reader converts to kg m⁻² s⁻¹.
"""

from __future__ import annotations

from pathlib import Path

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from plumax._src.errors import DomainError, MissingExtraError
from plumax.les_fvm.grid import PlumeGrid3D
from plumax.met._src.frames import LocalFrame


def load_inventory(
    path: str | Path,
    *,
    frame: LocalFrame,
    grid: PlumeGrid3D,
    variable: str | None = None,
    units: str | None = None,
    time_index: int = 0,
    supersample: int = 8,
) -> Float[Array, "y x"]:
    """Area-mean inventory flux over each analysis T-cell, as a prior mean.

    F̄_c = (1/|c|) ∫_c F dA  ≈  (1/s²) Σ_{k=1}^{s²} F(𝐱_k),

    with ``𝐱_k`` an ``s × s`` lattice of points evenly spaced in frame metres
    over cell ``c``. The sampling error falls as 1/s at inventory-cell edges;
    the frame is conformal and locally near-equidistant, so equal frame areas
    are equal ground areas.

    Algorithm:
        1. Read the inventory with :func:`xrreader.open_ch4_inventory`, which
           converts to kg m⁻² s⁻¹ with ascending lat and lon in [-180, 180).
        2. Lay an ``s × s`` lattice over every analysis cell and map it to
           lat/lon.
        3. Look up the inventory cell holding each point (longitudes taken
           modulo 360° on a global raster, which is periodic) and average
           over each analysis cell's lattice.

    Args:
        path: Inventory NetCDF (one sector or a total).
        frame: Frame the analysis grid is laid out in.
        grid: Analysis grid, in ``frame`` metres.
        variable: Flux variable name; inferred by the reader when ``None``.
        units: Override of the file's units attribute.
        time_index: Time step to use when the file has a ``time`` axis.
        supersample: Sub-points per cell edge, s.

    Returns:
        F̄ [kg m⁻² s⁻¹] on the horizontal T-cells, in the grid's dtype.

    Raises:
        MissingExtraError: If the ``io`` extra (xrreader) is not installed.
        DomainError: If a lattice point falls outside the inventory raster.
    """
    try:
        import xrreader
    except ImportError as err:
        raise MissingExtraError(
            "load_inventory reads through xrreader; install it with "
            "`pip install 'plumax[io]'`"
        ) from err

    # --- read ---
    dataset = xrreader.open_ch4_inventory(path, variable=variable, units=units)
    flux = dataset["ch4_emission_flux"]
    if "time" in flux.dims:
        flux = flux.isel(time=time_index)
    flux = flux.transpose("lat", "lon")
    lat_edges = _edges(flux["lat"].values)
    lon_edges = _edges(flux["lon"].values)
    flux_values = flux.values

    # --- sub-cell lattice ---
    x = np.asarray(grid.x, dtype=float)
    y = np.asarray(grid.y, dtype=float)
    # u_k = (k + 1/2) / s − 1/2, the sub-cell centres in cell units
    unit_offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    # x_{i,k} = x_i + u_k Δx
    sub_x = einx.add("x, k -> (x k)", x, unit_offsets * grid.dx)
    # y_{j,k} = y_j + u_k Δy
    sub_y = einx.add("y, k -> (y k)", y, unit_offsets * grid.dy)
    lattice_x, lattice_y = np.meshgrid(sub_x, sub_y)
    lat, lon = frame.to_latlon(lattice_x, lattice_y)

    # --- look up and average ---
    if np.isclose(lon_edges[-1] - lon_edges[0], 360.0):
        # a global raster is periodic: λ ← λ₀ + ((λ − λ₀) mod 360°), so a point
        # past the last edge lands in the first cell instead of outside
        lon = lon_edges[0] + np.mod(lon - lon_edges[0], 360.0)
    row = np.searchsorted(lat_edges, lat, side="right") - 1
    col = np.searchsorted(lon_edges, lon, side="right") - 1
    outside_rows = (row < 0) | (row >= lat_edges.size - 1)
    outside_cols = (col < 0) | (col >= lon_edges.size - 1)
    if np.any(outside_rows | outside_cols):
        raise DomainError(
            "load_inventory: the analysis grid extends beyond the inventory raster"
        )
    # F(𝐱_k)
    samples = einx.get_at("[lat lon], ys xs, ys xs -> ys xs", flux_values, row, col)
    # F̄_c = mean over the s × s lattice of cell c
    cell_mean = einx.mean(
        "(y [sy]) (x [sx]) -> y x", samples, sy=supersample, sx=supersample
    )
    return jnp.asarray(cell_mean, dtype=grid.x.dtype)


def _edges(centres: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n_edges"]:
    """Cell edges of a regular ascending axis from its centres."""
    # Δ = c₁ − c₀
    step = float(centres[1] - centres[0]) if centres.size > 1 else 1.0
    # e_k = c_k − Δ/2, plus the last edge c_{n−1} + Δ/2
    lower = centres - 0.5 * step
    last = centres[-1:] + 0.5 * step
    return einx.id("n, last -> (n + last)", lower, last)
