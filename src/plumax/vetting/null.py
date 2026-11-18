"""A per-scene null for vetting: copies of a candidate's mask where there is none.

Step 5 of the vetting (the plume-vetting theory page, "A null from the same
scene"): the honest null is this scene's clutter with no plume, not white
noise. `null_masks` draws rotated and translated copies of a candidate's
mask that overlap no candidate; vetting each copy as the candidate was
vetted gives the null scores, and `null_pvalue` places the candidate's score
among them. The reference package draws 100 such copies
(``EMITPlume.new_random_variation``, ``NUM_PLUME_VARIATIONS``) but avoids
only the plume's own boundary; these avoid every candidate.

The copies are drawn from a ``jax.random`` key the caller passes, so the
same key gives the same copies; the geometry is SciPy's (nearest-neighbour
rotation, and a correlation with the candidates for the free places), off
the gradient path. Nothing here is differentiable.
"""

from __future__ import annotations

from plumax.vetting._src.null import (
    null_masks,
    null_pvalue,
)


__all__ = [
    "null_masks",
    "null_pvalue",
]
