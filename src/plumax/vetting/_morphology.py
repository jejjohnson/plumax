"""Binary mask morphology in JAX for plume-candidate vetting.

Private: `seeds` grows the in-plume set with it. A max-pool needs no SciPy
or geotoolz, so `plumax.vetting` stays an L1 module on arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool


__all__ = ["dilate"]


def dilate(mask: Bool[ArrayLike, "H W"], steps: int = 1) -> Bool[Array, "H W"]:
    """``steps`` binary dilations of ``mask`` by the 3 × 3 square.

    Each step is a 3 × 3 max-pool with zero padding, so a pixel joins when
    any of its eight neighbours is set, and nothing enters from beyond the
    image's edge: ``scipy.ndimage.binary_dilation(mask, np.ones((3, 3)),
    iterations=steps)``. ``steps`` is static (a Python loop).

    Args:
        mask: The mask to grow, ``(H, W)``.
        steps: Number of dilations; ``0`` returns ``mask`` unchanged.

    Returns:
        The grown mask, ``(H, W)``.
    """
    grown = jnp.asarray(mask, dtype=bool)
    for _ in range(steps):
        pooled = jax.lax.reduce_window(
            grown.astype(jnp.int32), jnp.int32(0), jax.lax.max, (3, 3), (1, 1), "SAME"
        )
        grown = pooled > 0
    return grown
