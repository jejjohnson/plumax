"""pipekit ``Operator`` wrappers over the plumax forward models.

Each forward model in plumax is a pure function with a long, model-specific
signature. These wrappers split that signature the pipekit way: the *static*
grid / domain / scheme configuration is captured at construction time, and the
*dynamic* source / wind parameters flow through ``_apply`` as a single mapping
carrier. The result is an :class:`pipekit_ops.Operator` that composes with ``|``,
supports the config round-trip via :class:`pipekit_ops.ConfigMixin`, and can be
dropped into a :class:`pipekit_ops.Sequential` pipeline.

Example:
    >>> from plumax.operators import GaussianPlume
    >>> plume = GaussianPlume(
    ...     stability_class="D",
    ...     domain_x=(-100.0, 1000.0, 56),
    ...     domain_y=(-300.0, 300.0, 31),
    ...     domain_z=(0.0, 100.0, 11),
    ... )
    >>> ds = plume({
    ...     "emission_rate": 1.0,
    ...     "source_location": (0.0, 0.0, 10.0),
    ...     "wind_speed": 5.0,
    ...     "wind_direction": 270.0,
    ... })
    >>> ds["concentration"].dims
    ('x', 'y', 'z')
"""

from __future__ import annotations

from plumax._src.operators import (
    EulerianDispersion,
    GaussianPlume,
    GaussianPuff,
    LagrangianDispersion,
)


__all__ = [
    "EulerianDispersion",
    "GaussianPlume",
    "GaussianPuff",
    "LagrangianDispersion",
]
