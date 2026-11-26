"""Prescribed 3-D wind fields for the Eulerian dispersion solver.

The L2 model does **not** resolve the flow — wind is an external input.
A :class:`PrescribedWindField` is a callable pytree that maps a query
time ``t`` to the three velocity components ``(u, v, w)`` on the grid,
with ``u`` at U-points, ``v`` at V-points, and ``w`` at T-points
(the C-grid convention adopted by ``les_fvm``; see ``grid.py``).

Three constructors are provided:

- :func:`uniform_wind_field`        — constant in space *and* time.
- :func:`wind_field_from_schedule`  — spatially uniform, time-varying via
  a ``gauss_puff.wind.WindSchedule``.
- :func:`wind_field_from_callable`  — general-purpose escape hatch for
  user-supplied ``(t, X, Y, Z) -> (u, v, w)`` closures.

All fields return arrays with the same ``[Nz, Ny, Nx]`` layout as the
tracer concentration, so they can be consumed directly by the advection
operators.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from plumax.gauss_puff.wind import WindSchedule
from plumax.les_fvm.grid import PlumeGrid3D, coord_arrays_from_grid


class PrescribedWindField(eqx.Module):
    """Wind velocity evaluator on the full ``[Nz, Ny, Nx]`` field layout.

    The interior field values are populated by ``_interior_fn``; ghost
    rings are padded with the interior boundary values so downstream
    advection operators see a well-defined field everywhere.

    Parameters
    ----------
    plume_grid : PlumeGrid3D
        Grid the field lives on.
    interior_fn : Callable[[float], tuple[Float[Array, 'nz ny nx'], ...]]
        Pure function ``t -> (u_int, v_int, w_int)`` returning interior
        velocity components at the T/U/V stagger points (the ``les_fvm``
        convention).  See :func:`uniform_wind_field` for the simplest
        example.
    """

    plume_grid: PlumeGrid3D
    interior_fn: Callable[[Float[Array, ""]], tuple]

    def __call__(
        self, t: Float[ArrayLike, ""]
    ) -> tuple[
        Float[Array, "Nz Ny Nx"],
        Float[Array, "Nz Ny Nx"],
        Float[Array, "Nz Ny Nx"],
    ]:
        """Return ``(u, v, w)`` on the full ghost-padded grid layout."""
        east_interior, north_interior, up_interior = self.interior_fn(t)
        wind_east = _embed_interior(east_interior, self.plume_grid.shape)
        wind_north = _embed_interior(north_interior, self.plume_grid.shape)
        wind_up = _embed_interior(up_interior, self.plume_grid.shape)
        return wind_east, wind_north, wind_up


def _embed_interior(
    interior: Float[Array, "nz ny nx"], full_shape: tuple[int, int, int]
) -> Float[Array, "Nz Ny Nx"]:
    """Embed an interior-shaped field into the full ghost-padded layout.

    Ghost rings are filled by edge-padding — the interior boundary
    values are copied outward.  This is a safe default because the
    advection/diffusion operators only read from ghost cells via
    first-order stencils, and edge-padding is consistent with both
    ``Outflow`` and a zero-gradient ``Neumann`` BC on the wind.
    """
    del full_shape  # Every axis gets exactly one ghost ring in gridax.
    return jnp.pad(interior, ((1, 1), (1, 1), (1, 1)), mode="edge")


def uniform_wind_field(
    plume_grid: PlumeGrid3D,
    u: float = 0.0,
    v: float = 0.0,
    w: float = 0.0,
) -> PrescribedWindField:
    """Space- and time-constant wind field.

    Parameters
    ----------
    plume_grid : PlumeGrid3D
    u, v, w : float, default 0.0
        Cartesian wind velocity components [m/s].

    Returns
    -------
    PrescribedWindField

    Raises
    ------
    ValueError
        If any of ``u``, ``v``, ``w`` is non-finite — a NaN/inf wind would make
        the CFL bound NaN (disabling the fixed-step guard) and corrupt the solve.
    """
    for name, comp in (("u", u), ("v", v), ("w", w)):
        if not math.isfinite(float(comp)):
            raise ValueError(
                f"uniform_wind_field: {name} must be finite (got {comp!r})."
            )
    num_levels, num_rows, num_cols = plume_grid.interior_shape
    shape = (num_levels, num_rows, num_cols)
    east_field = jnp.full(shape, u, dtype=plume_grid.x.dtype)
    north_field = jnp.full(shape, v, dtype=plume_grid.x.dtype)
    up_field = jnp.full(shape, w, dtype=plume_grid.x.dtype)

    def interior_fn(time):
        del time
        return east_field, north_field, up_field

    return PrescribedWindField(plume_grid=plume_grid, interior_fn=interior_fn)


def wind_field_from_schedule(
    plume_grid: PlumeGrid3D,
    schedule: WindSchedule,
    w: float = 0.0,
) -> PrescribedWindField:
    """Time-varying but spatially uniform wind field.

    ``u(t)`` and ``v(t)`` are interpolated from the ``WindSchedule``
    (shared with ``gauss_puff``); ``w`` remains constant in time and
    space (``0`` by default).

    Parameters
    ----------
    plume_grid : PlumeGrid3D
    schedule : WindSchedule
    w : float, default 0.0
        Constant vertical velocity [m/s].

    Returns
    -------
    PrescribedWindField
    """
    num_levels, num_rows, num_cols = plume_grid.interior_shape
    ones = jnp.ones((num_levels, num_rows, num_cols), dtype=plume_grid.x.dtype)
    up_field = w * ones

    def interior_fn(time):
        east_at_time, north_at_time = schedule.wind_at(time)
        # the grid's dtype: a float64 schedule on a float32 grid stays float32
        east_at_time = jnp.asarray(east_at_time, dtype=ones.dtype)
        north_at_time = jnp.asarray(north_at_time, dtype=ones.dtype)
        return east_at_time * ones, north_at_time * ones, up_field

    return PrescribedWindField(plume_grid=plume_grid, interior_fn=interior_fn)


def wind_field_from_callable(
    plume_grid: PlumeGrid3D,
    fn: Callable[
        [
            Float[Array, ""],
            Float[Array, "nz ny nx"],
            Float[Array, "nz ny nx"],
            Float[Array, "nz ny nx"],
        ],
        tuple[
            Float[Array, "nz ny nx"],
            Float[Array, "nz ny nx"],
            Float[Array, "nz ny nx"],
        ],
    ],
) -> PrescribedWindField:
    """General-purpose wind field from a user-supplied ``(t, X, Y, Z)`` function.

    Each velocity component is sampled at its C-grid stagger point so the
    field the advection operator consumes is correctly located:

    - ``u`` at U-points — T-point coordinates shifted ``+dx/2`` in x,
    - ``v`` at V-points — T-point coordinates shifted ``+dy/2`` in y,
    - ``w`` at T-points (collocated, per the ``les_fvm`` convention).

    ``fn`` is therefore evaluated three times — once per stagger — and
    only the matching component is kept from each call.  A divergence-free
    analytic wind sampled this way stays discretely divergence-free
    (sampling all three at T-points would displace ``u``/``v`` by half a
    cell and introduce spurious tracer sources).

    Parameters
    ----------
    plume_grid : PlumeGrid3D
    fn : callable
        Pure function ``(t, X, Y, Z) -> (u, v, w)`` where each of
        ``X, Y, Z`` is an interior-shaped coordinate array.  Each output
        must have the same interior shape.

    Returns
    -------
    PrescribedWindField
    """
    X, Y, Z = coord_arrays_from_grid(plume_grid)
    half_dx = 0.5 * plume_grid.dx
    half_dy = 0.5 * plume_grid.dy
    # Stagger the sampling coordinates: U-points sit half a cell east of
    # the T-point in x, V-points half a cell north in y.
    east_x = X + half_dx
    north_y = Y + half_dy

    def interior_fn(time):
        wind_east = fn(time, east_x, Y, Z)[0]
        wind_north = fn(time, X, north_y, Z)[1]
        wind_up = fn(time, X, Y, Z)[2]
        return wind_east, wind_north, wind_up

    return PrescribedWindField(plume_grid=plume_grid, interior_fn=interior_fn)
