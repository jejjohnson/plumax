"""Boundary-condition application for 3-D tracer fields.

The ghost rings of a ``[Nz, Ny, Nx]`` field are filled in two steps:

1. A :class:`HorizontalBC` holds a :class:`gridax.BC` with the condition of
   each lateral face and fills the x and y ghosts of every z-level with
   :func:`gridax.fill_ghosts`.
2. A :class:`VerticalBC` holds the kind and the value of the bottom and the
   top face (Dirichlet / Neumann / outflow / periodic) and fills the z ghosts
   with the same function.
"""

from __future__ import annotations

from plumax.les_fvm._src.boundary import (
    LATERAL_FACES as LATERAL_FACES,
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
    "HorizontalBC",
    "VerticalBC",
    "apply_boundary_conditions",
    "build_default_concentration_bc",
    "periodic_axes",
    "scale_dirichlet_faces",
]
