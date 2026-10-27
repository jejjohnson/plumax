"""Per-tile background for column observations (the design's v1).

An observed column is background plus enhancement, and the background varies
over a scene by more than a weak plume's signal. v1 of the design
(``01_tier1_gaussian.md``: "v1: per-tile; v2: GP") holds it constant within
square tiles of the analysis frame.
"""

from __future__ import annotations

import einx
import numpy as np
from jaxtyping import Float

from plumax._src.errors import GridMismatchError, PlumaxError


def per_tile_background(
    columns: Float[np.ndarray, " obs"],
    receptors: Float[np.ndarray, "obs 2"],
    *,
    tile_m: float = 10_000.0,
) -> Float[np.ndarray, " obs"]:
    """Median of the observations in each square tile, given to every receptor.

    b_n = median{ y_m : tile(𝐱_m) = tile(𝐱_n) },  tile(𝐱) = (⌊x / ℓ⌋, ⌊y / ℓ⌋).

    The median is not moved by a plume covering less than half a tile.

    Args:
        columns: Observed columns y_n, any units.
        receptors: Receptor positions 𝐱_n = (x, y) in frame metres.
        tile_m: Tile edge ℓ [m].

    Returns:
        Background b_n per receptor, in the units of ``columns``.

    Raises:
        GridMismatchError: If ``receptors`` does not match ``columns``.
        PlumaxError: If ``tile_m`` is not positive.

    Examples:
        Two tiles, one with a plume pixel the median ignores:

        >>> import numpy as np
        >>> receptors = np.array(
        ...     [[1e3, 0.0], [2e3, 0.0], [3e3, 0.0], [12e3, 0.0], [15e3, 0.0]]
        ... )
        >>> columns = np.array([1880.0, 1880.0, 1960.0, 1900.0, 1900.0])
        >>> per_tile_background(columns, receptors).tolist()
        [1880.0, 1880.0, 1880.0, 1900.0, 1900.0]

        One tile spanning the scene gives the scene median:

        >>> per_tile_background(columns, receptors, tile_m=1e6).tolist()
        [1900.0, 1900.0, 1900.0, 1900.0, 1900.0]
    """
    values = np.asarray(columns, dtype=float)
    positions = np.asarray(receptors, dtype=float)
    if positions.shape != (values.size, 2):
        raise GridMismatchError(
            f"per_tile_background: receptors must be ({values.size}, 2); "
            f"got {positions.shape}"
        )
    if not tile_m > 0.0:
        raise PlumaxError("per_tile_background: `tile_m` must be positive")
    # tile(𝐱) = (⌊x / ℓ⌋, ⌊y / ℓ⌋)
    tiles = np.floor(positions / tile_m).astype(np.int64)
    tile_x, tile_y = einx.id("obs c -> c obs", tiles)
    # one integer per tile, so a 1-D unique groups receptors by tile
    tile_key = tile_x * (2**32) + tile_y
    _, tile_label = np.unique(tile_key, return_inverse=True)
    background = np.empty_like(values)
    for label in np.unique(tile_label):
        in_tile = tile_label == label
        # b = median of the tile's observations
        background[in_tile] = np.median(values[in_tile])
    return background
