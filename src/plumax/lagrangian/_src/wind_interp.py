"""The winds of plumax, as velocities for tracerax's particles.

A velocity is tracerax's: a callable ``velocity(t, position)`` that gives one wind
per particle. A wind that is uniform in space is ``tracerax.velocity.uniform``, or
``uniform_in_space`` when it varies in time. The gridded one is
``tracerax.velocity.GriddedVelocity``, and this module builds it from the two
gridded winds of plumax.

The transport tiers share one wind layout (``les_fvm/grid.py``): on a uniform
grid with T-points (cell centres) at ``(x_i, y_j, z_k)``,

    u at U-points (x_i + Δx/2, y_j, z_k),
    v at V-points (x_i, y_j + Δy/2, z_k),
    w at T-points (x_i, y_j, z_k).

A ``GriddedVelocity`` samples each component on its own points, so a particle sees
the wind where the Eulerian solver puts it; reading ``u`` and ``v`` as if they sat
at T-points would shift them by half a cell (the bug fixed in #28). In space the
sample is trilinear; in time it is piecewise-linear between the stored times,
policy (b) of the Tier 0 design page, as ``MetField.wind_at``.

Beyond the outermost sample point of a component, in space or time, the nearest
sample is held. That is the zero-gradient wind ``les_fvm`` puts in its edge-padded
ghost ring. Particles are not stopped or reflected at the lateral walls, which are
open in ``les_fvm``: they leave the domain and drop out of the residence and
footprint grids. The ground and the PBL top still reflect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Num
from tracerax.velocity import GriddedVelocity, Velocity, cgrid_grids, uniform

from plumax._src.wind import wind_components


if TYPE_CHECKING:
    from plumax.les_fvm.grid import PlumeGrid3D
    from plumax.les_fvm.wind import PrescribedWindField
    from plumax.met.field import MetField


def _on_plume_grid(
    plume_grid: PlumeGrid3D,
    times: Float[Array, " time"],
    u: Float[Array, "time z y x"],
    v: Float[Array, "time z y x"],
    w: Float[Array, "time z y x"],
) -> GriddedVelocity:
    """The velocity of interior arrays on the C-grid of an analysis grid."""
    x, y, z = plume_grid.x, plume_grid.y, plume_grid.z
    # u half a cell along x, v half a cell along y, w on the T levels
    x_u, y_v = x + 0.5 * plume_grid.dx, y + 0.5 * plume_grid.dy
    grids = cgrid_grids(times, (x_u, y, z), (x, y_v, z), (x, y, z))
    return GriddedVelocity.fit(grids, (u, v, w))


def gridded_wind_from_met(field: MetField) -> GriddedVelocity:
    """The wind of a :class:`~plumax.met.MetField`, for particles.

    ``MetField`` already stores ``u``, ``v``, ``w`` on the C-grid with a time
    axis, so no value changes.

    Args:
        field: Meteorology on an analysis grid.

    Returns:
        The wind on ``field.plume_grid`` at ``field.times``.
    """
    return _on_plume_grid(field.plume_grid, field.times, field.u, field.v, field.w)


def gridded_wind_from_prescribed(
    wind_field: PrescribedWindField, times: Num[ArrayLike, " time"]
) -> GriddedVelocity:
    """Sample a ``les_fvm`` :class:`~plumax.les_fvm.wind.PrescribedWindField`.

    The field is evaluated at each of ``times``; between them the particles see
    the piecewise-linear interpolant, so ``times`` should resolve the field's
    own time variation.

    Args:
        wind_field: The Eulerian solver's wind.
        times: Times [s] to sample it at, strictly increasing.

    Returns:
        The wind on ``wind_field.plume_grid``.
    """
    times = jnp.asarray(times, dtype=float)
    samples = [wind_field(time) for time in times]
    # interior points: drop the one-cell ghost ring the solver pads with
    wind_east, wind_north, wind_up = (
        jnp.stack([sample[component][1:-1, 1:-1, 1:-1] for sample in samples])
        for component in range(3)
    )
    return _on_plume_grid(wind_field.plume_grid, times, wind_east, wind_north, wind_up)


def wind_from_speed_direction(
    speed: float, direction_deg: float, w: float = 0.0
) -> Velocity:
    """Constant wind from speed + meteorological direction (degrees *from* North).

    ``direction_deg = 270`` is a wind from the west (flowing east), matching the
    convention used by :func:`plumax.gauss_plume.simulate_plume`.

    Args:
        speed: Wind speed [m/s].
        direction_deg: Direction the wind blows from [degrees from north].
        w: Vertical wind [m/s].

    Returns:
        The wind, as a ``tracerax.velocity.uniform`` velocity.

    Examples:
        >>> import jax.numpy as jnp
        >>> wind = wind_from_speed_direction(5.0, 270.0)
        >>> bool(jnp.allclose(wind(0.0, jnp.zeros((1, 3)))[0, 0], 5.0))
        True
    """
    wind_east, wind_north = wind_components(speed, direction_deg)
    return uniform(jnp.array([float(wind_east), float(wind_north), w]))
