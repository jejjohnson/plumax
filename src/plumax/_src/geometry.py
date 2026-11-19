"""Plane geometry shared by the ``geo`` and ``obs`` packages."""

from __future__ import annotations

import math
from collections.abc import Sequence

import einx
import numpy as np
from jaxtyping import Float

from plumax._src.errors import DomainError


__all__ = ["direction_from_centroid"]


def direction_from_centroid(
    points: Float[np.ndarray, "*pixel 2"],
    weights: Float[np.ndarray, "*pixel"],
    origin: Sequence[float],
    min_distance: float,
    *,
    undefined_message: str,
) -> float:
    """Direction from ``origin`` to the weighted centroid of ``points``.

    The centroid is s̄ = ∑ₙ wₙ sₙ / ∑ₙ wₙ, over a flat list of pixels or an
    ``(H, W)`` raster alike.

    Args:
        points: ``(x, y)`` of each pixel, shape ``(*pixel, 2)``.
        weights: Non-negative weight of each pixel, shape ``(*pixel,)``, with
            a positive sum.
        origin: ``(x, y)`` of the origin s₀.
        min_distance: Smallest ‖s̄ − s₀‖ for which the direction is defined.
        undefined_message: Message of the ``DomainError`` raised when the
            centroid is closer to the origin than ``min_distance``.

    Returns:
        φ₀ = atan2(s̄₂ − s₀₂, s̄₁ − s₀₁), in radians.

    Raises:
        DomainError: If ‖s̄ − s₀‖ < ``min_distance``.
    """
    # ∑ₙ wₙ
    total = float(np.sum(weights))
    flat_points = einx.id("... axis -> (...) axis", points)
    flat_weights = einx.id("... -> (...)", weights)
    # s̄ = ∑ₙ wₙ sₙ / ∑ₙ wₙ
    centroid = einx.dot("pixel axis, pixel -> axis", flat_points, flat_weights) / total
    centroid_x, centroid_y = float(centroid[0]), float(centroid[1])
    # s̄ − s₀
    offset_x = centroid_x - float(origin[0])
    offset_y = centroid_y - float(origin[1])
    if math.hypot(offset_x, offset_y) < min_distance:
        raise DomainError(undefined_message)
    # φ₀ = atan2(s̄₂ − s₀₂, s̄₁ − s₀₁)
    return math.atan2(offset_y, offset_x)
