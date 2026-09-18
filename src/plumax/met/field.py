"""Gridded meteorology container on the ``les_fvm`` analysis grid.

A :class:`MetField` holds the forcing the transport tiers consume — wind,
temperature, pressure, PBL height — already regridded onto a
:class:`~plumax.les_fvm.grid.PlumeGrid3D` and already on the ``les_fvm``
C-grid stagger (``u`` at U-points, ``v`` at V-points, ``w`` and scalars at
T-points; see ``les_fvm/grid.py``).  Every array carries a leading time
axis; ``times`` is seconds since ``t0``.

The container is an :class:`equinox.Module`, so it is a pytree: a met
ensemble is a stack of ``MetField``s under ``jax.vmap`` and needs no
special axis here.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from plumax.les_fvm.grid import PlumeGrid3D
from plumax.les_fvm.wind import PrescribedWindField


class MetField(eqx.Module):
    """Meteorological forcing on an analysis grid with a time axis.

    Attributes:
        plume_grid: Analysis grid every field lives on.
        times: Seconds since ``t0``, strictly increasing, shape ``(n_time,)``.
        u: Eastward wind [m/s] at U-points (T-point ``x + dx/2``),
            interior shape ``(n_time, nz, ny, nx)``.
        v: Northward wind [m/s] at V-points (T-point ``y + dy/2``).
        w: Vertical wind [m/s] at T-points.
        temperature: Air temperature [K] at T-points.
        pressure: Air pressure [Pa] at T-points.
        pbl_height: Boundary-layer height above ground [m] at horizontal
            T-points, shape ``(n_time, ny, nx)``.
        t0: ISO-8601 timestamp that ``times`` counts from (informational;
            static, so it never enters a trace).
    """

    plume_grid: PlumeGrid3D
    times: Float[Array, " n_time"]
    u: Float[Array, "n_time nz ny nx"]
    v: Float[Array, "n_time nz ny nx"]
    w: Float[Array, "n_time nz ny nx"]
    temperature: Float[Array, "n_time nz ny nx"]
    pressure: Float[Array, "n_time nz ny nx"]
    pbl_height: Float[Array, "n_time ny nx"]
    t0: str = eqx.field(static=True, default="")

    @property
    def n_time(self) -> int:
        """Number of met times."""
        return int(self.times.shape[0])

    def wind_at(
        self, t: Float[Array, ""]
    ) -> tuple[
        Float[Array, "nz ny nx"],
        Float[Array, "nz ny nx"],
        Float[Array, "nz ny nx"],
    ]:
        """Interior ``(u, v, w)`` at time ``t``, piecewise-linear in time.

        Met is typically hourly while the transport step is seconds, so the
        wind is interpolated linearly between the two bracketing met times
        (policy (b) of the Tier 0 design page).  Outside ``[times[0],
        times[-1]]`` the nearest end value is held constant.
        """
        return (
            interpolate_in_time(self.times, self.u, t),
            interpolate_in_time(self.times, self.v, t),
            interpolate_in_time(self.times, self.w, t),
        )


def interpolate_in_time(
    times: Float[Array, " n_time"],
    values: Float[Array, "n_time ..."],
    t: Float[Array, ""],
) -> Float[Array, ...]:
    """Linear interpolation of a time-stacked array at scalar time ``t``.

    Clamps ``t`` to the covered interval, so the result is constant beyond
    either end.  Pure JAX; safe under ``jit`` (the branch on ``n_time`` is
    on a static shape).

    Args:
        times: Strictly increasing sample times, shape ``(n_time,)``.
        values: Samples with leading axis ``n_time``.
        t: Query time (same units as ``times``).

    Returns:
        ``values`` interpolated at ``t``; shape ``values.shape[1:]``.
    """
    n_time = int(times.shape[0])
    if n_time == 1:
        return values[0]
    t_c = jnp.clip(jnp.asarray(t, dtype=times.dtype), times[0], times[-1])
    hi = jnp.clip(jnp.searchsorted(times, t_c, side="right"), 1, n_time - 1)
    lo = hi - 1
    t_lo, t_hi = times[lo], times[hi]
    frac = (t_c - t_lo) / (t_hi - t_lo)
    return (1.0 - frac) * values[lo] + frac * values[hi]


def to_prescribed_wind(field: MetField) -> PrescribedWindField:
    """Expose a :class:`MetField` as the wind object the Eulerian solver takes.

    The returned :class:`~plumax.les_fvm.wind.PrescribedWindField` samples
    :meth:`MetField.wind_at` — so ``simulate_eulerian_dispersion(
    wind_field=to_prescribed_wind(field), ...)`` runs the FV transport on
    loaded meteorology with no other change.  The caller's ``domain_*``
    triples must describe the same grid ``field.plume_grid`` was built on.
    """

    def interior_fn(t):
        return field.wind_at(t)

    return PrescribedWindField(plume_grid=field.plume_grid, interior_fn=interior_fn)
