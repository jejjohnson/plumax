"""Starting values for a plume fit, from a mask and a source point (behind ``[geo]``).

The image fit's cost has several minima in the source position and the
direction. :func:`plume_direction` gives the starting direction, from the
source to the centroid of the plume mask, and :func:`source_offset_grid` the
grid of source positions a multistart solve begins from
(``coupled.solve_coupled_multistart``).
"""

from __future__ import annotations

import math
from typing import Any

import einx
import numpy as np
from jaxtyping import Bool, Float

from plumax._src.errors import DomainError, ParameterError
from plumax._src.tracing import require_positive
from plumax.geo.plume_array import pixel_area, pixel_centers


def plume_direction(
    mask: Bool[np.ndarray, "row col"],
    transform: Any,
    source: tuple[float, float],
    *,
    weights: Float[np.ndarray, "row col"] | None = None,
) -> float:
    """Direction from the source to the centroid of the mask.

    φ₀ = atan2(s̄₂ − s₀₂, s̄₁ − s₀₁), with s̄ = ∑ₙ ωₙ sₙ / ∑ₙ ωₙ over the mask.

    Algorithm:
        1. Place every pixel centre sₙ in the CRS with the geotransform.
        2. Weight the mask pixels by ωₙ (1, or ``weights``) and take the
           centroid s̄.
        3. Take the angle of s̄ − s₀.

    With the enhancement as ``weights`` the centroid is the plume's centre of
    mass, which depends less on where the outline is drawn.

    Args:
        mask: Plume mask; True marks plume pixels.
        transform: Affine geotransform from ``(col, row)`` to the CRS [m].
        source: Source position s₀ [m] in the same CRS.
        weights: Optional weights ωₙ ≥ 0, such as the enhancement.

    Returns:
        φ₀ [rad], counter-clockwise from the CRS x axis (east), in (−π, π].

    Raises:
        DomainError: If the mask is empty, its weights are negative or sum
            to zero, or its centroid is within one pixel of the source.

    Examples:
        A plume of pixels north-east of the source, on a north-up 60 m grid:

        >>> import numpy as np
        >>> from affine import Affine
        >>> transform = Affine(60.0, 0.0, 0.0, 0.0, -60.0, 600.0)
        >>> mask = np.zeros((10, 10), dtype=bool)
        >>> mask[[4, 3, 2], [5, 6, 7]] = True
        >>> angle = plume_direction(mask, transform, (270.0, 270.0))
        >>> round(float(np.degrees(angle)), 6)
        45.0
    """
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        raise DomainError("plume_direction: the mask is empty.")
    if weights is None:
        weight = mask.astype(float)
    else:
        weights = np.asarray(weights, dtype=float)
        if bool(np.any(weights[mask] < 0.0)):
            raise DomainError("plume_direction: `weights` must be ≥ 0 on the mask.")
        weight = np.where(mask, weights, 0.0)
    # ∑ₙ ωₙ
    total = float(einx.sum("[row col]", weight))
    if total <= 0.0:
        raise DomainError("plume_direction: the weights sum to zero on the mask.")

    # --- centroid ---
    # sₙ = (xₙ, yₙ), the pixel centres
    easting, northing = pixel_centers(mask.shape, transform)
    # s̄ = ∑ₙ ωₙ sₙ / ∑ₙ ωₙ
    centroid_x = float(einx.dot("row col, row col ->", weight, easting)) / total
    centroid_y = float(einx.dot("row col, row col ->", weight, northing)) / total

    # --- direction ---
    # s̄ − s₀
    offset_x = centroid_x - source[0]
    offset_y = centroid_y - source[1]
    # g = √|det A|, the pixel side
    pixel_side = math.sqrt(pixel_area(transform))
    if math.hypot(offset_x, offset_y) < pixel_side:
        raise DomainError(
            "plume_direction: the centroid of the mask is within one pixel of the "
            "source, so the direction is undefined."
        )
    # φ₀ = atan2(s̄₂ − s₀₂, s̄₁ − s₀₁)
    return math.atan2(offset_y, offset_x)


def source_offset_grid(
    source: tuple[float, float],
    *,
    pixel_size: float,
    n: int = 3,
    step: int = 1,
) -> Float[np.ndarray, "start axis"]:
    """An n × n grid of source positions around the source, ``step`` pixels apart.

    s_ij = s₀ + step · g · (i, j) for i, j ∈ {−(n − 1)/2, …, (n − 1)/2}.

    The starts run over i (x) first and j (y) second, so the middle start,
    index n² // 2, is the source itself.

    Args:
        source: Source position s₀ [m].
        pixel_size: Pixel side g [m].
        n: Starts along each axis, odd.
        step: Spacing of the starts, in pixels.

    Returns:
        The n² starts, one ``(x, y)`` row each [m].

    Raises:
        ParameterError: If ``n`` is not a positive odd number, or ``step`` or
            ``pixel_size`` is not positive.

    Examples:
        >>> grid = source_offset_grid((100.0, 200.0), pixel_size=60.0)
        >>> grid.shape
        (9, 2)
        >>> grid[4].tolist()
        [100.0, 200.0]
        >>> grid[0].tolist()
        [40.0, 140.0]
    """
    if n < 1 or n % 2 == 0:
        raise ParameterError(f"n must be a positive odd number, got {n}")
    if step < 1:
        raise ParameterError(f"step must be ≥ 1, got {step}")
    require_positive("pixel_size", pixel_size)
    half = (n - 1) // 2
    # step · g · i for i ∈ {−half, …, half}
    offsets = step * pixel_size * np.arange(-half, half + 1, dtype=float)
    offset_x, offset_y = np.meshgrid(offsets, offsets, indexing="ij")
    # (step g i, step g j), i over x first
    grid = einx.id("x y, x y -> (x y) (1 + 1)", offset_x, offset_y)
    # s₀ + step g (i, j)
    starts = einx.add(
        "start axis, axis -> start axis", grid, np.asarray(source, dtype=float)
    )
    return starts


__all__ = ["plume_direction", "source_offset_grid"]
