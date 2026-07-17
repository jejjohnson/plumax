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

from typing import Literal

import equinox as eqx
from finitevolx import (
    BoundaryConditionSet,
    Dirichlet1D,
    Neumann1D,
    Outflow1D,
    Periodic1D,
)
from jaxtyping import Array, Float

from plumax.les_fvm.grid import PlumeGrid3D


VerticalBCKind = Literal["dirichlet", "neumann", "outflow", "periodic"]

HorizontalFace = Literal["south", "north", "west", "east"]


class CellDirichlet1D(eqx.Module):
    """Advective (cell-value) Dirichlet ghost for one horizontal face.

    Writes ``ghost = value`` — the boundary concentration as an exterior
    *cell* value — so a first-order upwind advective flux at an inflow wall
    carries the prescribed ``value``.  This differs from
    :class:`finitevolx.Dirichlet1D`, which writes the *face* reflection
    ``2*value - interior`` (the correct convention for the centred diffusion
    gradient, but wrong as the upwind cell value for advection — see
    jejjohnson/finitevolX#235).

    Parameters
    ----------
    face : {"south", "north", "west", "east"}
        Domain face to update.
    value : float
        Boundary concentration written into the ghost cells.
    """

    face: HorizontalFace = eqx.field(static=True)
    value: float

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one face's ghost set to ``value`` (cell value)."""
        del dx, dy
        if self.face == "south":
            return field.at[0, :].set(self.value)
        if self.face == "north":
            return field.at[-1, :].set(self.value)
        if self.face == "west":
            return field.at[:, 0].set(self.value)
        return field.at[:, -1].set(self.value)


def _to_cell_dirichlet(atom):
    """Swap a face-Dirichlet atom for its cell-value advective counterpart.

    All other atoms (outflow / periodic / Neumann / ``None``) are returned
    unchanged — only Dirichlet needs a different ghost convention for
    advection versus diffusion.
    """
    if isinstance(atom, Dirichlet1D):
        return CellDirichlet1D(face=atom.face, value=atom.value)
    return atom


class HorizontalBC(eqx.Module):
    """Apply a :class:`BoundaryConditionSet` to every z-slice of a 3-D field."""

    bc_set: BoundaryConditionSet

    def __call__(
        self,
        field: Float[Array, "Nz Ny Nx"],
        dx: float,
        dy: float,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Return ``field`` with horizontal ghost rings updated on each slice."""

        def apply_slice(slab: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
            return self.bc_set(slab, dx=dx, dy=dy)

        return eqx.filter_vmap(apply_slice)(field)

    def advection_variant(self) -> HorizontalBC:
        """Return a copy whose Dirichlet faces use a cell-value ghost.

        Open-mode advection reads the ghost as an exterior *cell* value; a
        face-Dirichlet ghost (``2*value - interior``) would make an inflow
        carry the wrong state.  Outflow / periodic / Neumann faces are
        unchanged, so this is a no-op unless a Dirichlet face is present.
        """
        bc_set = self.bc_set
        return HorizontalBC(
            bc_set=BoundaryConditionSet(
                south=_to_cell_dirichlet(bc_set.south),
                north=_to_cell_dirichlet(bc_set.north),
                west=_to_cell_dirichlet(bc_set.west),
                east=_to_cell_dirichlet(bc_set.east),
                mask=bc_set.mask,
            )
        )


class VerticalBC(eqx.Module):
    """Top/bottom ghost-slice update for a 3-D field.

    Parameters
    ----------
    bottom_kind : {"dirichlet", "neumann", "outflow", "periodic"}
        Ground-boundary behaviour.
    bottom_value : float, default 0.0
        For ``dirichlet``: the boundary value.
        For ``neumann``: the coordinate gradient ``∂C/∂z`` along ``+z``
        at the face — a positive value means ``C`` increases with height.
        The ghost cell is set so the finite-difference ``∂C/∂z`` across
        the face equals ``value``. Ignored for ``outflow`` / ``periodic``.
    top_kind, top_value : same, with the Neumann ``value`` again the
        coordinate gradient ``∂C/∂z`` along ``+z`` (same convention on
        both faces; see the sign note in :func:`_apply_vertical_face`).
    """

    bottom_kind: VerticalBCKind = eqx.field(static=True)
    top_kind: VerticalBCKind = eqx.field(static=True)
    bottom_value: float = 0.0
    top_value: float = 0.0
    # When True, a Dirichlet face writes ghost = value (a cell value) instead
    # of the face reflection ``2*value - interior`` — the advection convention.
    dirichlet_cell: bool = eqx.field(static=True, default=False)

    def __call__(
        self,
        field: Float[Array, "Nz Ny Nx"],
        dz: float,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Return ``field`` with top and bottom ghost slices updated.

        ``dz`` is required to translate a Neumann coordinate gradient
        ``∂C/∂z`` into the half-cell ghost offset ``sign · gradient · dz``.
        """
        out = _apply_vertical_face(
            field,
            face="bottom",
            kind=self.bottom_kind,
            value=self.bottom_value,
            dz=dz,
            dirichlet_cell=self.dirichlet_cell,
        )
        out = _apply_vertical_face(
            out,
            face="top",
            kind=self.top_kind,
            value=self.top_value,
            dz=dz,
            dirichlet_cell=self.dirichlet_cell,
        )
        return out

    def advection_variant(self) -> VerticalBC:
        """Return a copy whose Dirichlet faces use a cell-value ghost.

        The vertical advection term also upwinds the ghost slice, so a
        Dirichlet vertical inflow needs the same cell-value convention as the
        horizontal faces.  A no-op unless a vertical Dirichlet face is set.
        """
        return VerticalBC(
            bottom_kind=self.bottom_kind,
            top_kind=self.top_kind,
            bottom_value=self.bottom_value,
            top_value=self.top_value,
            dirichlet_cell=True,
        )


def _apply_vertical_face(
    field: Float[Array, "Nz Ny Nx"],
    face: Literal["bottom", "top"],
    kind: VerticalBCKind,
    value: float,
    dz: float,
    dirichlet_cell: bool = False,
) -> Float[Array, "Nz Ny Nx"]:
    """Update one vertical ghost slice using the requested BC flavour.

    Neumann convention: ``value`` is the coordinate gradient ``∂C/∂z``
    along ``+z``, applied identically on both faces — the ghost cell is
    set so the finite-difference ``∂C/∂z`` across the face equals
    ``value``. The ``outward_sign`` factor (``-1`` at the bottom, ``+1``
    at the top) is what makes the *same* ``value`` mean ``∂C/∂z`` at
    both faces despite the ghost sitting below the interior at the
    bottom and above it at the top; this matches the ghost sign of
    :class:`finitevolx.Neumann1D`.
    """
    if face == "bottom":
        interior_slice = field[1, :, :]
        opposite_slice = field[-2, :, :]
        outward_sign = -1.0
        ghost_index = 0
    else:
        interior_slice = field[-2, :, :]
        opposite_slice = field[1, :, :]
        outward_sign = 1.0
        ghost_index = -1

    if kind == "dirichlet":
        # Advection wants the boundary value as a cell value (ghost = value);
        # diffusion wants the face reflection so the centred gradient sees it.
        ghost = value if dirichlet_cell else (2.0 * value - interior_slice)
    elif kind == "neumann":
        # Ghost value so that the finite-difference coordinate gradient
        # ``∂C/∂z`` (along +z) across the face equals ``value``. The
        # outward_sign flip keeps that meaning identical on both faces.
        ghost = interior_slice + outward_sign * value * dz
    elif kind == "outflow":
        ghost = interior_slice
    elif kind == "periodic":
        ghost = opposite_slice
    else:
        raise ValueError(f"Unknown vertical BC kind: {kind!r}")

    return field.at[ghost_index, :, :].set(ghost)


def apply_boundary_conditions(
    field: Float[Array, "Nz Ny Nx"],
    horizontal_bc: HorizontalBC,
    vertical_bc: VerticalBC,
    plume_grid: PlumeGrid3D,
) -> Float[Array, "Nz Ny Nx"]:
    """Apply horizontal then vertical BCs to a 3-D tracer field."""
    out = horizontal_bc(field, dx=plume_grid.dx, dy=plume_grid.dy)
    out = vertical_bc(out, dz=plume_grid.dz)
    return out


def periodic_axes(horizontal_bc: HorizontalBC) -> tuple[bool, bool]:
    """Return ``(x_periodic, y_periodic)`` for a horizontal BC set.

    A lateral axis is periodic when either of its paired faces carries a
    :class:`finitevolx.Periodic1D` atom.  Used to decide whether the normal
    wind / diffusivity halos must be wrapped (rather than edge-padded) so the
    two representations of a periodic seam face share a coefficient and the
    open-wall fluxes cancel.
    """
    bc_set = horizontal_bc.bc_set
    x_periodic = isinstance(bc_set.west, Periodic1D) or isinstance(
        bc_set.east, Periodic1D
    )
    y_periodic = isinstance(bc_set.south, Periodic1D) or isinstance(
        bc_set.north, Periodic1D
    )
    return x_periodic, y_periodic


def build_default_concentration_bc(
    bc_x: (str | tuple[str, str] | tuple[tuple[str, float], tuple[str, float]]) = (
        "dirichlet",
        "outflow",
    ),
    bc_y: (
        str | tuple[str, str] | tuple[tuple[str, float], tuple[str, float]]
    ) = "periodic",
    bc_z: (str | tuple[str, str] | tuple[tuple[str, float], tuple[str, float]]) = (
        "neumann",
        "neumann",
    ),
) -> tuple[HorizontalBC, VerticalBC]:
    """Build ``(HorizontalBC, VerticalBC)`` from user-facing BC specs.

    Each of ``bc_x``, ``bc_y``, ``bc_z`` can be:

    - ``"periodic"``       — periodic on both faces of that axis.
    - ``(west, east)`` (for ``bc_x``) / ``(south, north)`` (``bc_y``) /
      ``(bottom, top)`` (``bc_z``) where each entry is either a BC-kind
      string (``"dirichlet"``, ``"neumann"``, ``"outflow"``) or a
      ``(kind, value)`` tuple giving the Dirichlet / Neumann target.

    Returns
    -------
    tuple[HorizontalBC, VerticalBC]
    """
    w_bc, e_bc = _split_horizontal_spec(bc_x, faces=("west", "east"))
    s_bc, n_bc = _split_horizontal_spec(bc_y, faces=("south", "north"))

    bc_set = BoundaryConditionSet(south=s_bc, north=n_bc, west=w_bc, east=e_bc)
    bot_kind, bot_val, top_kind, top_val = _split_vertical_spec(bc_z)
    return (
        HorizontalBC(bc_set=bc_set),
        VerticalBC(
            bottom_kind=bot_kind,
            bottom_value=bot_val,
            top_kind=top_kind,
            top_value=top_val,
        ),
    )


def _as_kind_value(
    entry: str | tuple[str, float],
) -> tuple[str, float]:
    if isinstance(entry, str):
        return entry, 0.0
    kind, value = entry
    return str(kind), float(value)


def _split_horizontal_spec(
    spec: (str | tuple[str, str] | tuple[tuple[str, float], tuple[str, float]]),
    faces: tuple[str, str],
):
    """Unpack a horizontal BC spec into two face-specific atoms.

    ``faces`` is a pair of face names ordered ``(lower, upper)``, e.g.
    ``("west", "east")`` or ``("south", "north")``.
    """
    if isinstance(spec, str) and spec.lower() == "periodic":
        return Periodic1D(faces[0]), Periodic1D(faces[1])
    if isinstance(spec, tuple) and len(spec) == 2:
        lower_kind, lower_val = _as_kind_value(spec[0])
        upper_kind, upper_val = _as_kind_value(spec[1])
        return (
            _build_1d_bc(lower_kind, lower_val, faces[0]),
            _build_1d_bc(upper_kind, upper_val, faces[1]),
        )
    raise ValueError(
        "horizontal BC spec must be 'periodic' or a 2-tuple "
        f"(lower, upper); got {spec!r}"
    )


def _build_1d_bc(kind: str, value: float, face: str):
    kind_l = kind.lower()
    if kind_l == "dirichlet":
        return Dirichlet1D(face=face, value=value)
    if kind_l == "neumann":
        return Neumann1D(face=face, value=value)
    if kind_l == "outflow":
        return Outflow1D(face=face)
    if kind_l == "periodic":
        return Periodic1D(face=face)
    raise ValueError(
        f"horizontal BC kind must be one of 'dirichlet', 'neumann', "
        f"'outflow', 'periodic'; got {kind!r}"
    )


def _split_vertical_spec(
    spec: (str | tuple[str, str] | tuple[tuple[str, float], tuple[str, float]]),
) -> tuple[VerticalBCKind, float, VerticalBCKind, float]:
    if isinstance(spec, str) and spec.lower() == "periodic":
        return "periodic", 0.0, "periodic", 0.0
    if isinstance(spec, tuple) and len(spec) == 2:
        bot_kind, bot_val = _as_kind_value(spec[0])
        top_kind, top_val = _as_kind_value(spec[1])
        for kind in (bot_kind, top_kind):
            if kind.lower() not in {"dirichlet", "neumann", "outflow", "periodic"}:
                raise ValueError(
                    f"vertical BC kind must be one of 'dirichlet', 'neumann', "
                    f"'outflow', 'periodic'; got {kind!r}"
                )
        return bot_kind.lower(), bot_val, top_kind.lower(), top_val  # type: ignore[return-value]
    raise ValueError(
        f"vertical BC spec must be 'periodic' or a 2-tuple (bottom, top); got {spec!r}"
    )
