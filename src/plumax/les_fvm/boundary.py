"""Boundary-condition application for 3-D tracer fields.

finitevolX ships 2-D BC atoms (:class:`finitevolx.Dirichlet1D`, etc.)
that expect a ``[Ny, Nx]`` field.  For the Eulerian dispersion solver
we need to update ghost rings on a ``[Nz, Ny, Nx]`` field.  The pattern
used here is:

1. A :class:`HorizontalBC` bundles a :class:`finitevolx.BoundaryConditionSet`
   and applies it per z-slice via ``eqx.filter_vmap`` — keeping the
   per-face semantics of finitevolX but vectorising over the vertical
   axis.
2. A :class:`VerticalBC` holds two ``(bc_type, value)`` pairs for the
   top and bottom faces.  We implement these directly (Dirichlet /
   Neumann / outflow / periodic) because the 2-D BC atoms assume the
   modified axis is horizontal.
"""

from __future__ import annotations

from plumax.les_fvm._src.boundary import (
    LATERAL_FACES as LATERAL_FACES,
    CellDirichlet1D,
    HorizontalBC,
    HorizontalFace as HorizontalFace,
    VerticalBC,
    VerticalBCKind as VerticalBCKind,
    apply_boundary_conditions,
    build_default_concentration_bc,
    periodic_axes,
    scale_dirichlet_faces,
)


__all__ = [
    "CellDirichlet1D",
    "HorizontalBC",
    "VerticalBC",
    "apply_boundary_conditions",
    "build_default_concentration_bc",
    "periodic_axes",
    "scale_dirichlet_faces",
]
