"""Gridded wind for particles: the ``les_fvm`` C-grid sampled at each particle.

The transport tiers share one wind layout (``les_fvm/grid.py``): on a uniform
grid with T-points (cell centres) at ``(x_i, y_j, z_k)``,

    u at U-points (x_i + Δx/2, y_j, z_k),
    v at V-points (x_i, y_j + Δy/2, z_k),
    w at T-points (x_i, y_j, z_k).

:class:`GriddedWind` samples each component on its own points, so a particle
sees the wind where the Eulerian solver puts it; reading ``u`` and ``v`` as if
they sat at T-points would shift them by half a cell (the bug fixed in #28).
In space the sample is trilinear; in time it is piecewise-linear between the
stored times, policy (b) of the Tier 0 design page, as ``MetField.wind_at``.

Beyond the outermost sample point of a component, in space or time, the
nearest sample is held. That is the zero-gradient wind ``les_fvm`` puts in its
edge-padded ghost ring. Particles are not stopped or reflected at the lateral
walls, which are open in ``les_fvm``: they leave the domain and drop out of the
residence and footprint grids. The ground and the PBL top still reflect, in
``langevin_step``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


if TYPE_CHECKING:
    from collections.abc import Callable

    from plumax.les_fvm.grid import PlumeGrid3D
    from plumax.les_fvm.wind import PrescribedWindField
    from plumax.met.field import MetField


#: Lower bracket index, upper bracket index and weight of the upper one.
Bracket = tuple[Int[Array, " 2"], Float[Array, ""]]


class GriddedWind(eqx.Module):
    """Mean wind on the ``les_fvm`` C-grid, trilinear in space, linear in time.

    u(p, t) = Σ_corners w_corner u_corner over the 16 samples of ``u`` that
    bracket (t, z, y, x), and likewise for ``v`` and ``w`` on their own points.

    Attributes:
        x: T-point x-coordinates x_i [m], uniformly spaced.
        y: T-point y-coordinates y_j [m], uniformly spaced.
        z: T-point z-coordinates z_k [m], uniformly spaced.
        times: Sample times [s], strictly increasing.
        u: Eastward wind [m/s] at U-points (x_i + Δx/2).
        v: Northward wind [m/s] at V-points (y_j + Δy/2).
        w: Vertical wind [m/s] at T-points.

    Examples:
        A uniform wind is reproduced exactly, anywhere and at any time:

        >>> import jax.numpy as jnp
        >>> axis = jnp.arange(4.0) * 10.0 + 5.0
        >>> ones = jnp.ones((2, 4, 4, 4))
        >>> wind = GriddedWind(
        ...     axis, axis, axis, jnp.array([0.0, 60.0]), 3.0 * ones, -ones, 0 * ones
        ... )
        >>> wind(jnp.array([[12.0, 7.0, 31.0], [-50.0, 80.0, 0.0]]), 30.0).tolist()
        [[3.0, -1.0, 0.0], [3.0, -1.0, 0.0]]

        ``u`` is read on U-points, half a cell east of the T-points, so a
        field linear in x comes back as x itself at the T-points:

        >>> u = jnp.broadcast_to(axis + 5.0, (2, 4, 4, 4))
        >>> sheared = GriddedWind(axis, axis, axis, jnp.array([0.0, 60.0]), u, ones, ones)
        >>> sheared(jnp.array([[15.0, 15.0, 15.0]]), 0.0)[:, 0].tolist()
        [15.0]

        Piecewise-linear in time between the stored times:

        >>> ramp = jnp.stack([ones[0], 3.0 * ones[0]])
        >>> gusty = GriddedWind(axis, axis, axis, jnp.array([0.0, 60.0]), ramp, ones, ones)
        >>> gusty(jnp.array([[15.0, 15.0, 15.0]]), 15.0)[:, 0].tolist()
        [1.5]
    """

    x: Float[Array, " x"]
    y: Float[Array, " y"]
    z: Float[Array, " z"]
    times: Float[Array, " time"]
    u: Float[Array, "time z y x"]
    v: Float[Array, "time z y x"]
    w: Float[Array, "time z y x"]

    def __check_init__(self) -> None:
        for name in ("x", "y", "z"):
            if getattr(self, name).shape[0] < 2:
                raise ValueError(f"GriddedWind: `{name}` needs at least 2 points.")

    @classmethod
    def from_grid(
        cls,
        plume_grid: PlumeGrid3D,
        times: Float[Array, " time"],
        u: Float[Array, "time z y x"],
        v: Float[Array, "time z y x"],
        w: Float[Array, "time z y x"],
    ) -> GriddedWind:
        """Wind on a ``les_fvm`` analysis grid, interior arrays on the C-grid.

        Args:
            plume_grid: The analysis grid; its T-point axes place the samples.
            times: Sample times [s], strictly increasing.
            u: Eastward wind [m/s] at U-points.
            v: Northward wind [m/s] at V-points.
            w: Vertical wind [m/s] at T-points.

        Returns:
            The gridded wind.
        """
        return cls(plume_grid.x, plume_grid.y, plume_grid.z, times, u, v, w)

    def __call__(
        self, position: Float[Array, "particle 3"], t: Float[Array, ""] | float
    ) -> Float[Array, "particle 3"]:
        """Mean wind (u, v, w) [m/s] at each particle position and time t.

        Args:
            position: Particle positions (x, y, z) [m].
            t: Time [s] on the same clock as ``times``.

        Returns:
            The wind at each particle.
        """
        time_bracket = _time_bracket(self.times, jnp.asarray(t, self.times.dtype))
        # Δx, Δy: U-points sit Δx/2 east and V-points Δy/2 north of T-points
        half_dx = 0.5 * _spacing(self.x)
        half_dy = 0.5 * _spacing(self.y)

        def at_particle(point: Float[Array, " 3"]) -> Float[Array, " 3"]:
            px, py, pz = point[0], point[1], point[2]
            z_bracket = _axis_bracket(self.z, pz)
            # T-point brackets serve w, and v in x and u in y
            x_bracket = _axis_bracket(self.x, px)
            y_bracket = _axis_bracket(self.y, py)
            # U-point x = x_i + Δx/2, so p_x on the U axis is p_x − Δx/2 on x_i
            xu_bracket = _axis_bracket(self.x, px - half_dx)
            # V-point y = y_j + Δy/2
            yv_bracket = _axis_bracket(self.y, py - half_dy)
            u = _sample(self.u, time_bracket, z_bracket, y_bracket, xu_bracket)
            v = _sample(self.v, time_bracket, z_bracket, yv_bracket, x_bracket)
            w = _sample(self.w, time_bracket, z_bracket, y_bracket, x_bracket)
            return jnp.stack([u, v, w])

        return jax.vmap(at_particle)(position)


def _spacing(axis: Float[Array, " n"]) -> Float[Array, ""]:
    # Δ = (a_{n-1} − a_0) / (n − 1) on a uniform axis
    return (axis[-1] - axis[0]) / (axis.shape[0] - 1)


def _axis_bracket(axis: Float[Array, " n"], coordinate: Float[Array, ""]) -> Bracket:
    """Bracketing indices and weight on a uniform axis, held beyond its ends."""
    n = axis.shape[0]
    # s = (p − a_0) / Δ, held inside [0, n − 1]
    fractional = (coordinate - axis[0]) / _spacing(axis)
    fractional = jnp.clip(fractional, 0.0, n - 1.0)
    lower = jnp.clip(jnp.floor(fractional).astype(int), 0, n - 2)
    weight = fractional - lower
    return jnp.stack([lower, lower + 1]), weight


def _time_bracket(times: Float[Array, " time"], t: Float[Array, ""]) -> Bracket:
    """Bracketing indices and weight in time, held beyond the first and last."""
    n = times.shape[0]
    if n == 1:
        return jnp.zeros(2, dtype=int), jnp.zeros((), dtype=times.dtype)
    held = jnp.clip(t, times[0], times[-1])
    upper = jnp.clip(jnp.searchsorted(times, held, side="right"), 1, n - 1)
    lower = upper - 1
    # λ = (t − t_lo) / (t_hi − t_lo)
    weight = (held - times[lower]) / (times[upper] - times[lower])
    return jnp.stack([lower, upper]), weight


def _sample(
    field: Float[Array, "time z y x"],
    time_bracket: Bracket,
    z_bracket: Bracket,
    y_bracket: Bracket,
    x_bracket: Bracket,
) -> Float[Array, ""]:
    """Quadrilinear sample from the 2 × 2 × 2 × 2 block of bracketing values."""
    brackets = (time_bracket, z_bracket, y_bracket, x_bracket)
    block = field[jnp.ix_(*(index for index, _ in brackets))]
    # f ← f_0 + λ (f_1 − f_0) along each axis in turn; exact when f_0 = f_1
    for _, weight in brackets:
        block = block[0] + weight * (block[1] - block[0])
    return block


def gridded_wind_from_met(field: MetField) -> GriddedWind:
    """The wind of a :class:`~plumax.met.MetField`, for particles.

    ``MetField`` already stores ``u``, ``v``, ``w`` on the C-grid with a time
    axis, so this is a relabelling; no value changes.

    Args:
        field: Meteorology on an analysis grid.

    Returns:
        The gridded wind on ``field.plume_grid`` at ``field.times``.
    """
    return GriddedWind.from_grid(
        field.plume_grid, field.times, field.u, field.v, field.w
    )


def gridded_wind_from_prescribed(
    wind_field: PrescribedWindField, times: Float[Array, " time"]
) -> GriddedWind:
    """Sample a ``les_fvm`` :class:`~plumax.les_fvm.wind.PrescribedWindField`.

    The field is evaluated at each of ``times``; between them the particles see
    the piecewise-linear interpolant, so ``times`` should resolve the field's
    own time variation.

    Args:
        wind_field: The Eulerian solver's wind.
        times: Times [s] to sample it at, strictly increasing.

    Returns:
        The gridded wind on ``wind_field.plume_grid``.
    """
    times = jnp.asarray(times)
    samples = [wind_field(t) for t in times]
    # interior points: drop the one-cell ghost ring the solver pads with
    u, v, w = (
        jnp.stack([sample[c][1:-1, 1:-1, 1:-1] for sample in samples]) for c in range(3)
    )
    return GriddedWind.from_grid(wind_field.plume_grid, times, u, v, w)


def mean_wind(
    wind: GriddedWind | Callable[[Array], Array],
    position: Float[Array, "particle 3"],
    t: Float[Array, ""] | float,
) -> Float[Array, "particle 3"] | Float[Array, " 3"]:
    """Mean wind for the particle integrators, from either kind of wind.

    A :class:`GriddedWind` is sampled at every particle; any other wind is the
    spatially uniform callable ``t -> (u, v, w)`` the integrators took before.
    """
    if isinstance(wind, GriddedWind):
        return wind(position, t)
    return jnp.asarray(wind(t))
