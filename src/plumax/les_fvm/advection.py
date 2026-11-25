"""3-D flux-form advection of a passive tracer.

The horizontal contribution ``-∇_h · (u_h C)`` is delegated to
:func:`gridax.advect`, which applies a WENO/TVD reconstruction
at U- and V-faces for every z-level.  The vertical contribution
``-∂_z(w C)`` is :func:`gridax.advect_vertical` with a first-order upwind
reconstruction at k-faces.

Available reconstruction methods for the horizontal term are the schemes of
:mod:`gridax.schemes`, by name: ``upwind1``, ``weno3``, ``weno5``,
``weno7``, ``weno9``, ``minmod``, ``van_leer``, ``superbee``, ``mc``,
``naive``.  The vertical term is always
first-order upwind — vertical resolution is typically too coarse for a
higher-order scheme to pay for itself, and first-order upwind is
monotone by construction, which matters for a non-negative tracer.
"""

from __future__ import annotations

from plumax.les_fvm._src.advection import (
    advection_tendency,
)


__all__ = [
    "advection_tendency",
]
