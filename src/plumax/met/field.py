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

from plumax.met._src.field import (
    MetField,
    interpolate_in_time,
    to_prescribed_wind,
)


__all__ = [
    "MetField",
    "interpolate_in_time",
    "to_prescribed_wind",
]
