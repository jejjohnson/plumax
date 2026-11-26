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

import equinox as eqx
import jax.numpy as jnp
from gridax import advect, advect_vertical
from gridax.schemes import TVD, WENO, Centred, Scheme, Upwind
from jaxtyping import Array, Float

from plumax.les_fvm._src.boundary import HorizontalBC
from plumax.les_fvm.grid import PlumeGrid3D


#: The horizontal reconstruction of each ``method`` name.
_SCHEMES: dict[str, Scheme] = {
    "upwind1": Upwind(1),
    "weno3": WENO(3),
    "weno5": WENO(5),
    "weno7": WENO(7),
    "weno9": WENO(9),
    "minmod": TVD("minmod"),
    "van_leer": TVD("van_leer"),
    "superbee": TVD("superbee"),
    "mc": TVD("mc"),
    "naive": Centred(2),
}


def _scheme(method: str) -> Scheme:
    """Return the gridax scheme of a ``method`` name."""
    if method not in _SCHEMES:
        raise ValueError(
            f"Unknown method: {method!r}; choose one of {sorted(_SCHEMES)}."
        )
    return _SCHEMES[method]


@eqx.filter_jit
def advection_tendency(
    concentration: Float[Array, "Nz Ny Nx"],
    u: Float[Array, "Nz Ny Nx"],
    v: Float[Array, "Nz Ny Nx"],
    w: Float[Array, "Nz Ny Nx"],
    plume_grid: PlumeGrid3D,
    method: str = "weno5",
    horizontal_bc: HorizontalBC | None = None,
) -> Float[Array, "Nz Ny Nx"]:
    """Flux-form advective tendency ``-∇·(u C)`` at interior T-points.

    Parameters
    ----------
    concentration : Float[Array, "Nz Ny Nx"]
        Tracer at T-points.
    u : Float[Array, "Nz Ny Nx"]
        x-velocity at U-points (east faces).
    v : Float[Array, "Nz Ny Nx"]
        y-velocity at V-points (north faces).
    w : Float[Array, "Nz Ny Nx"]
        Vertical velocity collocated at T-points (see ``grid.py``).
    plume_grid : PlumeGrid3D
        Grid the fields live on.
    method : str, default ``"weno5"``
        Horizontal reconstruction scheme.  Passed through to
        :func:`gridax.advect`, one of the names listed in the module
        docstring.
    horizontal_bc : HorizontalBC, optional
        The lateral condition of the tracer. gridax fills as many ghost layers
        as ``method`` reads from it, so every interior cell gets the full
        scheme, and on a periodic axis the tracer and the normal wind wrap, so
        the seam is a face like any other. A Dirichlet face is read as a cell
        value (:meth:`~plumax.les_fvm.boundary.HorizontalBC.advection_variant`).
        Without it the lateral faces are open: their fluxes are first-order
        upwind of the ghost ring, which the caller fills.

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Advective tendency at T-points, zero on every ghost face.

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    Notes
    -----
    With ``horizontal_bc`` the horizontal operator is :func:`gridax.advect`
    with ``bc=``; without it, ``wall="open"``, which reads the one ghost ring
    the caller filled. :class:`~plumax.les_fvm.dynamics.EulerianDispersionRHS`
    passes its own conditions.
    """
    scheme = _scheme(method)
    if horizontal_bc is None:
        horizontal = advect(
            concentration, u, v, plume_grid.grid, scheme=scheme, wall="open"
        )
    else:
        conditions = horizontal_bc.advection_variant().bc_set
        horizontal = advect(
            concentration, u, v, plume_grid.grid, scheme=scheme, bc=conditions
        )
    # w_{k+1/2} = ½ (w_k + w_{k+1}) on the top face of each cell; the last one is
    # outside the domain and not read
    w_face = jnp.pad(0.5 * (w[:-1] + w[1:]), ((0, 1), (0, 0), (0, 0)))
    vertical = advect_vertical(
        concentration, w_face, plume_grid.grid, scheme=Upwind(1), wall="open"
    )
    return horizontal + vertical
