"""A candidate's in-plume seeds and the background pool they are paired from.

Steps 1–2 of Xiang et al.'s vetting, as the reference code runs them
(``pv/src/pv/pv.py``): the seeds are the ``k`` strongest matched-filter
pixels inside the candidate mask, below the mask's 99th percentile, grown by
one pixel inside it; the pool is every pixel near the seeds, outside every
candidate, with a small matched-filter score. Both return boolean masks and
run under ``jax.jit`` with ``k``, ``grow`` and ``box`` static. Nothing here
is differentiable: the outputs are booleans. radtrax's ``pair_pixels``
takes the two sets as flat indices (``np.flatnonzero``).
"""

from __future__ import annotations

from plumax.vetting._src.seeds import (
    background_pool,
    select_seeds,
)


__all__ = [
    "background_pool",
    "select_seeds",
]
