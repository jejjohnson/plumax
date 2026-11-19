"""Eager input checks shared by the NUTS inference runners."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jaxtyping import ArrayLike


def check_observations(
    caller: str,
    observations: ArrayLike,
    observation_coords: Sequence[ArrayLike],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Observations and their ``(x, y, z)`` coordinates as arrays, after checking them.

    Args:
        caller: The public function the check guards, for the message.
        observations: Observed values, any shape with at least one element.
        observation_coords: ``(x, y, z)``, each the shape of ``observations``.

    Returns:
        ``(observations, (x, y, z))`` as NumPy arrays.

    Raises:
        ValueError: ``observations`` is empty, ``observation_coords`` does not
            have three axes, or an axis's shape differs from the observations'.

    Examples:
        >>> obs, (x, y, z) = check_observations("f", [1.0, 2.0], ([0, 1], [0, 0], [2, 2]))
        >>> x.shape
        (2,)
        >>> check_observations("f", [1.0], ([0.0], [0.0]))
        Traceback (most recent call last):
        ...
        ValueError: f: `observation_coords` must be (x, y, z); got length 2
    """
    obs = np.asarray(observations)
    if obs.size == 0:
        raise ValueError(f"{caller}: `observations` must contain ≥ 1 point")
    if len(observation_coords) != 3:
        raise ValueError(
            f"{caller}: `observation_coords` must be (x, y, z); "
            f"got length {len(observation_coords)}"
        )
    x, y, z = (np.asarray(coord) for coord in observation_coords)
    for axis_name, axis in zip("xyz", (x, y, z), strict=True):
        if axis.shape != obs.shape:
            raise ValueError(
                f"{caller}: `observation_coords` axis '{axis_name}' "
                f"has shape {axis.shape} ≠ observations shape {obs.shape}"
            )
    return obs, (x, y, z)
