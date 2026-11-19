"""Wind components from a speed and a meteorological direction."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import jax.numpy as jnp


def wind_components(
    speed: Any, direction_deg: Any, *, xp: ModuleType = jnp
) -> tuple[Any, Any]:
    """Eastward and northward wind from a speed and the direction it blows *from*.

    The direction is meteorological: degrees clockwise from north, naming where
    the wind comes from. A wind from the west (270°) blows east, so u > 0; a
    wind from the north (0°) blows south, so v < 0.

        u = −U sin θ,   v = −U cos θ,

    with U the speed and θ the direction in radians.

    Args:
        speed: Wind speed U [m/s], a scalar or an array.
        direction_deg: Direction θ the wind blows from [degrees from north],
            broadcastable against ``speed``.
        xp: The array namespace, ``jax.numpy`` (default) or ``numpy``.

    Returns:
        ``(u, v)`` [m/s], in ``xp`` arrays of the broadcast shape.

    Examples:
        >>> import numpy as np
        >>> u, v = wind_components(5.0, 270.0, xp=np)
        >>> bool(np.isclose(u, 5.0)), bool(np.isclose(v, 0.0, atol=1e-12))
        (True, True)
    """
    # θ = direction in radians
    direction_rad = xp.deg2rad(direction_deg)
    # u = −U sin θ
    wind_east = -speed * xp.sin(direction_rad)
    # v = −U cos θ
    wind_north = -speed * xp.cos(direction_rad)
    return wind_east, wind_north
