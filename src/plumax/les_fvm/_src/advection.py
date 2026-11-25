"""3-D flux-form advection of a passive tracer.

The horizontal contribution ``-∇_h · (u_h C)`` is delegated to
:func:`gridax.advect`, which applies a WENO/TVD reconstruction
at U- and V-faces for every z-level.  The vertical contribution
``-∂_z(w C)`` is assembled by :func:`_vertical_ops.vertical_advection_tendency`
with a first-order upwind reconstruction at k-faces.

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
from gridax import advect
from gridax.schemes import TVD, WENO, Centred, Scheme, Upwind
from jaxtyping import Array, Float

from plumax.les_fvm._src.vertical_ops import vertical_advection_tendency
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
    periodic: tuple[bool, bool] = (False, False),
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
    periodic : tuple[bool, bool], default ``(False, False)``
        ``(x_periodic, y_periodic)``.  On a periodic axis the normal-velocity
        ghost face is *wrapped* (rather than left edge-padded) so the two
        representations of the periodic seam share one velocity and the
        open-wall fluxes cancel — otherwise a spatially-varying normal wind
        breaks exact mass conservation at the seam.  See
        :func:`~plumax.les_fvm.boundary.periodic_axes`.

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
    The horizontal operator runs in gridax ``wall="open"`` mode, so the
    lateral domain-wall face fluxes are assembled from the ghost ring
    (first-order upwind against the wall, high-order ``method`` in the
    interior).  The caller must fill the ghost cells with the desired lateral
    BC (Dirichlet / outflow / periodic) *before* calling — see
    :func:`~plumax.les_fvm.boundary.apply_boundary_conditions`, which
    :class:`~plumax.les_fvm.dynamics.EulerianDispersionRHS` runs every step.
    This is what makes horizontal transport honor ``bc_x`` / ``bc_y``.
    """
    x_periodic, y_periodic = periodic
    if x_periodic:
        # Wrap the x-normal velocity seam: west wall face (u[..., 0]) and east
        # wall face (u[..., -2]) are the same periodic seam and must match.
        u = u.at[:, :, 0].set(u[:, :, -2]).at[:, :, -1].set(u[:, :, 1])
    if y_periodic:
        v = v.at[:, 0, :].set(v[:, -2, :]).at[:, -1, :].set(v[:, 1, :])
    horizontal = advect(
        concentration, u, v, plume_grid.grid, scheme=_scheme(method), wall="open"
    )
    vertical = vertical_advection_tendency(concentration, w, plume_grid.dz)
    return horizontal + vertical
