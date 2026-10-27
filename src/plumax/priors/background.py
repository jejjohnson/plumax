"""Per-tile background for column observations (v1).

An observed column is background plus enhancement, and the background
varies over a scene by more than a weak plume's signal. v1 of the design
(``01_tier1_gaussian.md``: "v1: per-tile; v2: GP") takes the background as
constant within square tiles of the analysis frame and estimates it there
as the median of the observations, which a plume covering less than half
a tile does not move.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Array, Float


def per_tile_background(
    columns: Float[Array, " N"],
    receptors: Float[Array, "N 2"],
    *,
    tile_m: float = 10_000.0,
) -> np.ndarray:
    """Median of the observations in each ``tile_m`` × ``tile_m`` tile.

    Tile ``(⌊x / tile_m⌋, ⌊y / tile_m⌋)`` holds receptor ``(x, y)``; every
    receptor gets its tile's median.

    Args:
        columns: Observed columns, any units, ``(N,)``.
        receptors: Receptor ``(x, y)`` in frame metres, ``(N, 2)``.
        tile_m: Tile edge [m].

    Returns:
        Background per receptor, ``(N,)``, in the units of ``columns``.

    Raises:
        ValueError: If shapes disagree or ``tile_m`` is not positive.
    """
    values = np.asarray(columns, dtype=np.float64)
    xy = np.asarray(receptors, dtype=np.float64)
    if xy.shape != (values.size, 2):
        raise ValueError(
            f"per_tile_background: receptors must be ({values.size}, 2); got {xy.shape}"
        )
    if not tile_m > 0.0:
        raise ValueError("per_tile_background: `tile_m` must be positive")
    tiles = np.floor(xy / tile_m).astype(np.int64)
    _, label = np.unique(tiles, axis=0, return_inverse=True)
    label = label.ravel()
    out = np.empty_like(values)
    for k in np.unique(label):
        members = label == k
        out[members] = np.median(values[members])
    return out
