"""Binary mask morphology in JAX for plume-candidate vetting.

Private: `seeds` grows the in-plume set with it. A max-pool needs no SciPy
or geotoolz, so `plumax.vetting` stays an L1 module on arrays.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool
from radtrax import ShapeError


__all__ = ["dilate", "union_masks"]


def union_masks(
    stack: Bool[ArrayLike, "C H W"], shape: tuple[int, int], name: str
) -> Bool[ArrayLike, "H W"]:
    """Union ⋃_c M_c of a ``(C, H, W)`` stack of masks, empty stack included.

    Works on NumPy and JAX arrays alike and returns the same kind.

    Args:
        stack: The masks, as an array (``np.ndarray`` or ``jax.Array``) of
            booleans.
        shape: The ``(H, W)`` the masks must have.
        name: The calling function, prefixed to the error message.

    Returns:
        The union, ``(H, W)``; all ``False`` for ``C = 0``.

    Raises:
        ShapeError: If ``stack`` is not ``(C, H, W)``.
    """
    height, width = shape
    if stack.ndim != 3 or stack.shape[1:] != shape:
        raise ShapeError(
            f"{name}: exclude must be (C, {height}, {width}), got {stack.shape}"
        )
    # einx rejects a zero-size axis; no masks make an empty union
    if stack.shape[0] == 0:
        return np.zeros(shape, dtype=bool)
    # ⋃_c M_c
    return einx.any("[C] H W", stack)


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
