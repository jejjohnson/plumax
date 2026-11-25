"""3-D Cartesian Arakawa C-grid wrapper for the Eulerian dispersion model.

The underlying :class:`gridax.CartesianCGrid3D` stores fields with shape
``[Nz, Ny, Nx]`` where every axis carries a one-cell ghost ring, so the
interior indices span ``[1:-1, 1:-1, 1:-1]``.  This module wraps the
construction + coordinate generation so the rest of ``les_fvm`` can treat
the grid as a single pytree leaf with convenient ``x``, ``y``, ``z``
coordinate arrays (interior cell centres only).

Stagger convention (see :class:`gridax.CartesianCGrid3D`)
-----------------------------------------------------------
    T[k, j, i]  cell centre  at  ( i        * dx,  j        * dy,  k * dz )
    U[k, j, i]  east face    at  ((i + 1/2) * dx,  j        * dy,  k * dz )
    V[k, j, i]  north face   at  ( i        * dx, (j + 1/2) * dy,  k * dz )

Within a C-grid stagger, velocities ``u`` and ``v`` live at U- and V-points
respectively; the scalar tracer and vertical velocity ``w`` are treated as
T-point quantities in this package.  Vertical staggering is not used; we
collocate ``w`` at T-points for shape uniformity and pay a small accuracy
cost that is dwarfed by the K-theory parameterisation anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import einx
import equinox as eqx
import jax.numpy as jnp
from gridax import CartesianCGrid3D
from jax.typing import DTypeLike
from jaxtyping import Array, Float


if TYPE_CHECKING:
    import numpy as np


class PlumeGrid3D(eqx.Module):
    """Arakawa C-grid + cached interior coordinate arrays.

    Attributes
    ----------
    grid : CartesianCGrid3D
        Underlying gridax grid.  All field operators (advection,
        diffusion, interpolation, difference) take this as input.
    x : Float[Array, "nx_interior"]
        Interior T-point x-coordinates [m].
    y : Float[Array, "ny_interior"]
        Interior T-point y-coordinates [m].
    z : Float[Array, "nz_interior"]
        Interior T-point z-coordinates [m].
    dx, dy, dz : float
        Cell spacing along each axis [m], static under a JAX transformation.

    Notes
    -----
    ``x``, ``y``, ``z`` are the *interior* cell-centre coordinates — they
    drop the one-cell ghost rings on either side of each axis.  The sizes
    therefore match the interior slice shapes used by the advection and
    diffusion operators, not the raw field-array shapes.
    """

    grid: CartesianCGrid3D
    x: Float[Array, " nx_interior"]
    y: Float[Array, " ny_interior"]
    z: Float[Array, " nz_interior"]
    dx: float = eqx.field(static=True)
    dy: float = eqx.field(static=True)
    dz: float = eqx.field(static=True)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Full field shape ``(Nz, Ny, Nx)`` including ghost rings."""
        return (self.grid.Nz, self.grid.Ny, self.grid.Nx)

    @property
    def interior_shape(self) -> tuple[int, int, int]:
        """Interior shape ``(nz, ny, nx)`` excluding ghost rings."""
        return (int(self.z.size), int(self.y.size), int(self.x.size))


def make_grid(
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    dtype: DTypeLike = jnp.float32,
) -> PlumeGrid3D:
    """Build a :class:`PlumeGrid3D` from ``(x_min, x_max, nx_interior)`` triples.

    Parameters
    ----------
    domain_x : tuple (x_min, x_max, nx_interior)
        Physical x-range + number of interior cells in x.
    domain_y : tuple (y_min, y_max, ny_interior)
        Physical y-range + number of interior cells in y.
    domain_z : tuple (z_min, z_max, nz_interior)
        Physical z-range + number of interior cells in z.  ``z_min`` is
        treated as the ground level (Neumann-wall BC target below).
    dtype : jnp.dtype, default float32
        Coordinate-array dtype.

    Returns
    -------
    PlumeGrid3D

    Raises
    ------
    ValueError
        If any interior cell count is < 4 (WENO5 stencils need at least
        four interior cells in every horizontal direction) or if any
        physical extent is non-positive.
    """
    lower_x, upper_x, num_cells_x = domain_x
    lower_y, upper_y, num_cells_y = domain_y
    lower_z, upper_z, num_cells_z = domain_z

    for label, count in (("x", num_cells_x), ("y", num_cells_y), ("z", num_cells_z)):
        if count < 4:
            raise ValueError(
                f"make_grid: need at least 4 interior cells in {label} "
                f"for WENO5 stencils (got {count})"
            )

    extent_x = float(upper_x - lower_x)
    extent_y = float(upper_y - lower_y)
    extent_z = float(upper_z - lower_z)
    for label, extent, expr in (
        ("x", extent_x, "x_max - x_min"),
        ("y", extent_y, "y_max - y_min"),
        ("z", extent_z, "z_max - z_min"),
    ):
        if extent <= 0.0:
            raise ValueError(
                f"make_grid: {label}-extent must be positive (got {expr} = {extent})"
            )

    grid = CartesianCGrid3D.from_interior(
        nx_interior=int(num_cells_x),
        ny_interior=int(num_cells_y),
        nz_interior=int(num_cells_z),
        Lx=extent_x,
        Ly=extent_y,
        Lz=extent_z,
    )
    # Interior cell centres — offset by half a cell from the lower edge,
    # matching the gridax convention of co-locating T-points at
    # cell centres and drop ghost cells at i = 0 and i = Nx - 1.
    # x_c = x_min + (i + ½) Lx / nx
    cell_index_x = jnp.arange(num_cells_x, dtype=dtype)
    centres_x = jnp.asarray(lower_x + (cell_index_x + 0.5) * (extent_x / num_cells_x))
    # y_c = y_min + (j + ½) Ly / ny
    cell_index_y = jnp.arange(num_cells_y, dtype=dtype)
    centres_y = jnp.asarray(lower_y + (cell_index_y + 0.5) * (extent_y / num_cells_y))
    # z_c = z_min + (k + ½) Lz / nz
    cell_index_z = jnp.arange(num_cells_z, dtype=dtype)
    centres_z = jnp.asarray(lower_z + (cell_index_z + 0.5) * (extent_z / num_cells_z))
    return PlumeGrid3D(
        grid=grid,
        x=centres_x,
        y=centres_y,
        z=centres_z,
        dx=extent_x / int(num_cells_x),
        dy=extent_y / int(num_cells_y),
        dz=extent_z / int(num_cells_z),
    )


def coord_arrays_from_grid(
    plume_grid: PlumeGrid3D,
) -> tuple[
    Float[Array, "nz_interior ny_interior nx_interior"],
    Float[Array, "nz_interior ny_interior nx_interior"],
    Float[Array, "nz_interior ny_interior nx_interior"],
]:
    """Return broadcast 3-D ``(X, Y, Z)`` interior-coordinate arrays."""
    X, Y, Z = jnp.meshgrid(plume_grid.x, plume_grid.y, plume_grid.z, indexing="xy")
    # meshgrid(indexing="xy") returns (Ny, Nx, Nz) — reorder to (Nz, Ny, Nx)
    # to match field layout.
    X = einx.id("y x z -> z y x", X)
    Y = einx.id("y x z -> z y x", Y)
    Z = einx.id("y x z -> z y x", Z)
    return X, Y, Z


def coord_to_numpy(
    plume_grid: PlumeGrid3D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert interior coordinate arrays to NumPy for xarray packaging."""
    import numpy as _np

    return (
        _np.asarray(plume_grid.x),
        _np.asarray(plume_grid.y),
        _np.asarray(plume_grid.z),
    )
