"""Boundary-condition application for 3-D tracer fields.

The ghost rings of a ``[Nz, Ny, Nx]`` field are filled in two steps:

1. A :class:`HorizontalBC` holds a :class:`gridax.BC` with the condition of
   each lateral face and fills the x and y ghosts of every z-level with
   :func:`gridax.fill_ghosts`.
2. A :class:`VerticalBC` holds two ``(bc_type, value)`` pairs for the
   top and bottom faces, filled here (Dirichlet / Neumann / outflow /
   periodic).
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
from gridax import (
    BC,
    Dirichlet,
    Neumann,
    Outflow,
    Periodic,
    Sponge,
    fill_ghosts,
)
from jaxtyping import Array, Float

from plumax.les_fvm.grid import PlumeGrid3D


VerticalBCKind = Literal["dirichlet", "neumann", "outflow", "periodic"]

HorizontalFace = Literal["south", "north", "west", "east"]
#: One face of an axis: a kind, or a (kind, value) pair for a Dirichlet face.
FaceBCSpec = str | tuple[str, float]
#: One axis: a kind for both faces, or the (low, high) faces.
AxisBCSpec = str | tuple[FaceBCSpec, FaceBCSpec]

#: The order of the per-face lateral boundary scaling factors.
LATERAL_FACES: tuple[HorizontalFace, ...] = ("west", "east", "south", "north")

#: The axis of each lateral face, and its side (0 low, 1 high).
_FACE_SIDES: dict[HorizontalFace, tuple[str, int]] = {
    "west": ("x", 0),
    "east": ("x", 1),
    "south": ("y", 0),
    "north": ("y", 1),
}


def _to_cell_dirichlet(atom):
    """Swap a face-Dirichlet condition for its cell-value advective counterpart.

    :class:`gridax.Dirichlet` writes the *face* reflection
    ``2*value - interior``, which is right for the centred diffusion gradient
    and wrong as the upwind cell value of an inflow. A :class:`gridax.Sponge`
    of weight 1 writes ``ghost = value``. All other conditions (outflow /
    periodic / Neumann / ``None``) are returned unchanged.
    """
    if isinstance(atom, Dirichlet):
        return Sponge(background=atom.value, weight=1.0)
    return atom


def _horizontal_bc(
    west: object, east: object, south: object, north: object
) -> HorizontalBC:
    """Return the :class:`HorizontalBC` of four lateral conditions."""
    # the x condition owns the corners, which no tendency reads
    return HorizontalBC(bc_set=BC(x=(west, east), y=(south, north), corners="priority"))


class HorizontalBC(eqx.Module):
    """Fill the lateral ghosts of every z-slice of a 3-D field.

    Parameters
    ----------
    bc_set : gridax.BC
        The condition of each lateral face.
    """

    bc_set: BC

    def __call__(
        self, field: Float[Array, "Nz Ny Nx"], plume_grid: PlumeGrid3D
    ) -> Float[Array, "Nz Ny Nx"]:
        """Return ``field`` with horizontal ghost rings updated on each slice."""
        return fill_ghosts(field, plume_grid.grid, self.bc_set)

    def face(self, name: HorizontalFace) -> object:
        """Return the condition of one lateral face, ``None`` for none."""
        axis, side = _FACE_SIDES[name]
        return self.bc_set.faces.get(axis, (None, None))[side]

    def advection_variant(self) -> HorizontalBC:
        """Return a copy whose Dirichlet faces use a cell-value ghost.

        Open-mode advection reads the ghost as an exterior *cell* value; a
        face-Dirichlet ghost (``2*value - interior``) would make an inflow
        carry the wrong state.  Outflow / periodic / Neumann faces are
        unchanged, so this is a no-op unless a Dirichlet face is present.
        """
        return _horizontal_bc(
            *(_to_cell_dirichlet(self.face(name)) for name in LATERAL_FACES)
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
            spacing_z=dz,
            dirichlet_cell=self.dirichlet_cell,
        )
        out = _apply_vertical_face(
            out,
            face="top",
            kind=self.top_kind,
            value=self.top_value,
            spacing_z=dz,
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
    spacing_z: float,
    dirichlet_cell: bool = False,
) -> Float[Array, "Nz Ny Nx"]:
    """Update one vertical ghost slice using the requested BC flavour.

    Neumann convention: ``value`` is the coordinate gradient ``∂C/∂z``
    along ``+z``, applied identically on both faces — the ghost cell is
    set so the finite-difference ``∂C/∂z`` across the face equals
    ``value``. The ``outward_sign`` factor (``-1`` at the bottom, ``+1``
    at the top) is what makes the *same* ``value`` mean ``∂C/∂z`` at
    both faces despite the ghost sitting below the interior at the
    bottom and above it at the top. The lateral faces follow the same
    convention (see :func:`_build_1d_bc`).
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
        # g = C_int + s ∂C/∂z Δz
        ghost = interior_slice + outward_sign * value * spacing_z
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
    out = horizontal_bc(field, plume_grid)
    out = vertical_bc(out, dz=plume_grid.dz)
    return out


def scale_dirichlet_faces(
    horizontal_bc: HorizontalBC, scale: Float[Array, " 4"]
) -> HorizontalBC:
    """Multiply the prescribed value of each Dirichlet face by its own factor.

    c_face ← α_face c_face, the per-face lateral boundary scaling of the
    4D-Var control. The factors are ordered as :data:`LATERAL_FACES`. Outflow,
    periodic and Neumann faces prescribe no boundary concentration, so their
    factors do nothing, and neither does a factor on a zero-valued Dirichlet face.

    Args:
        horizontal_bc: The boundary conditions to scale.
        scale: One factor α per face.

    Returns:
        The boundary conditions with the scaled values.
    """

    def scaled(atom, factor: Float[Array, ""]):
        if isinstance(atom, Dirichlet):
            return Dirichlet(factor * atom.value)
        return atom

    return _horizontal_bc(
        *(
            scaled(horizontal_bc.face(name), scale[index])
            for index, name in enumerate(LATERAL_FACES)
        )
    )


def periodic_axes(horizontal_bc: HorizontalBC) -> tuple[bool, bool]:
    """Return ``(x_periodic, y_periodic)`` for a horizontal BC set.

    Periodicity is an *axis* property — both paired faces must wrap together —
    so this is used to decide whether the normal wind / diffusivity halos are
    wrapped (rather than edge-padded), keeping the two representations of a
    periodic seam face consistent and the open-wall fluxes cancelling.

    :class:`gridax.BC` refuses an axis with one periodic face, so the west and
    the south face speak for their axes.
    """
    return (
        isinstance(horizontal_bc.face("west"), Periodic),
        isinstance(horizontal_bc.face("south"), Periodic),
    )


def build_default_concentration_bc(
    bc_x: AxisBCSpec = (
        "dirichlet",
        "outflow",
    ),
    bc_y: AxisBCSpec = "periodic",
    bc_z: AxisBCSpec = (
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
      ``(kind, value)`` tuple giving the Dirichlet / Neumann target. A Neumann
      target is the coordinate gradient (``∂C/∂x``, ``∂C/∂y``, ``∂C/∂z``) on
      both faces of its axis.

    Returns
    -------
    tuple[HorizontalBC, VerticalBC]

    Raises
    ------
    ValueError
        If a spec or a kind is unknown, or exactly one lateral face of an axis
        is periodic: the seam couples both faces.
    """
    west_bc, east_bc = _split_horizontal_spec(bc_x, faces=("west", "east"))
    south_bc, north_bc = _split_horizontal_spec(bc_y, faces=("south", "north"))

    bot_kind, bot_val, top_kind, top_val = _split_vertical_spec(bc_z)
    return (
        _horizontal_bc(west_bc, east_bc, south_bc, north_bc),
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
    spec: AxisBCSpec,
    faces: tuple[str, str],
):
    """Unpack a horizontal BC spec into two face-specific atoms.

    ``faces`` is a pair of face names ordered ``(lower, upper)``, e.g.
    ``("west", "east")`` or ``("south", "north")``.
    """
    if isinstance(spec, str) and spec.lower() == "periodic":
        return Periodic(), Periodic()
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
    """Return the gridax condition of one lateral face.

    A Neumann ``value`` is the coordinate gradient, as on the vertical faces.
    :class:`gridax.Neumann` takes the gradient along the outward normal, which
    points down the axis on the low (west, south) faces.
    """
    kind_l = kind.lower()
    if kind_l == "dirichlet":
        return Dirichlet(value)
    if kind_l == "neumann":
        return Neumann(-value if face in ("west", "south") else value)
    if kind_l == "outflow":
        return Outflow()
    if kind_l == "periodic":
        return Periodic()
    raise ValueError(
        f"horizontal BC kind must be one of 'dirichlet', 'neumann', "
        f"'outflow', 'periodic'; got {kind!r}"
    )


def _split_vertical_spec(
    spec: AxisBCSpec,
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
