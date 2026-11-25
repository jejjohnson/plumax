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

from plumax.les_fvm._src.grid import (
    PlumeGrid3D,
    coord_arrays_from_grid,
    coord_to_numpy,
    make_grid,
)


__all__ = [
    "PlumeGrid3D",
    "coord_arrays_from_grid",
    "coord_to_numpy",
    "make_grid",
]
