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

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, Bool, Float, PRNGKeyArray
from radtrax import DegenerateInputError, ParameterError, ShapeError


__all__ = ["null_masks", "null_pvalue"]


def null_masks(
    mask: Bool[ArrayLike, "H W"],
    n: int,
    *,
    key: PRNGKeyArray,
    exclude: Bool[ArrayLike, " C H W"] | None = None,
    min_fraction: float = 0.9,
) -> Bool[np.ndarray, "S H W"]:
    """``n`` rotated and translated copies of a mask, off every candidate.

    Each attempt draws an angle ``θ ~ U[0°, 360°)`` and rotates ``mask``
    about the image centre by ``θ`` (nearest neighbour, the image's shape
    kept); a rotation that keeps fewer than ``min_fraction · |M|`` pixels is
    discarded. Among the integer shifts that keep the rotated mask inside
    the image, the free ones are those where

    ``M_s ∩ (M ∪ ⋃_c M_c) = ∅``,

    with ``M_c`` the ``C`` candidate masks in ``exclude``; one free shift is
    drawn uniformly, and an angle with none is discarded. So a copy never
    touches a candidate, this one included, and given its angle its place is
    uniform over the free places (found exactly, by correlating the rotated
    mask with the candidates, rather than by rejecting random shifts, which
    a mask covering a large part of the scene almost never passes). Copies
    may overlap one another. They are returned in the order they were drawn,
    so the first ``m`` of ``n`` copies are the ``m`` copies the same key
    gives.

    Args:
        mask: The candidate ``M``, ``(H, W)``.
        n: Number of copies ``S``; the reference package's 100.
        key: A ``jax.random`` key, the only source of randomness: the same
            key gives the same copies.
        exclude: The scene's ``C`` candidate masks, ``(C, H, W)``; pass
            every candidate (``mask`` itself is always avoided).
        min_fraction: A copy keeps at least this fraction of ``M``'s pixels
            (rotation can push some off the image or merge them), in (0, 1].

    Returns:
        The copies, ``(S, H, W)``, a NumPy array.

    Raises:
        ParameterError: If ``n < 1`` or ``min_fraction`` is outside (0, 1].
        ShapeError: If ``mask`` is not 2-D or ``exclude`` is not
            ``(C, H, W)``.
        DegenerateInputError: If ``mask`` is empty, or ``100 · n`` angles
            place fewer than ``n`` copies (the scene is too small, or too
            crowded with candidates, for copies of this mask).

    Examples:
        >>> import jax
        >>> import numpy as np
        >>> from plumax.vetting import null_masks
        >>> mask = np.zeros((32, 32), bool)
        >>> mask[4:8, 4:12] = True
        >>> copies = null_masks(mask, 5, key=jax.random.key(0), exclude=mask[None])
        >>> copies.shape
        (5, 32, 32)
        >>> bool((copies & mask).any())
        False
        >>> bool((copies.sum(axis=(1, 2)) >= 0.9 * mask.sum()).all())
        True
    """
    if n < 1:
        raise ParameterError(f"null_masks: n must be >= 1, got {n}")
    if not 0.0 < min_fraction <= 1.0:
        raise ParameterError(
            f"null_masks: min_fraction must be in (0, 1], got {min_fraction}"
        )
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ShapeError(f"null_masks: mask must be (H, W), got {mask.shape}")
    height, width = mask.shape
    banned = mask.copy()
    if exclude is not None:
        exclude = np.asarray(exclude, dtype=bool)
        if exclude.ndim != 3 or exclude.shape[1:] != mask.shape:
            raise ShapeError(
                f"null_masks: exclude must be (C, {height}, {width}), got "
                f"{exclude.shape}"
            )
        banned |= np.any(exclude, axis=0)
    size = int(mask.sum())
    if size == 0:
        raise DegenerateInputError("null_masks: the mask is empty")

    from scipy.ndimage import rotate
    from scipy.signal import correlate

    needed = min_fraction * size
    image = mask.astype(np.float64)
    forbidden = banned.astype(np.float64)
    copies: list[np.ndarray] = []
    for _ in range(100 * n):
        key, k_angle, k_place = jax.random.split(key, 3)
        angle = float(jax.random.uniform(k_angle, maxval=360.0))
        rotated = rotate(image, angle, reshape=False, order=0) > 0.5  # (H, W)
        rows, cols = np.nonzero(rotated)
        if rows.size < needed:
            continue
        crop = rotated[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        # Candidate pixels under the crop at each top-left corner that keeps
        # it inside the scene: (H - h + 1, W - w + 1).
        overlap = correlate(forbidden, crop.astype(np.float64), mode="valid")
        free = np.flatnonzero(overlap < 0.5)
        if free.size == 0:
            continue
        pick = free[int(jax.random.randint(k_place, (), 0, free.size))]
        top, left = np.unravel_index(pick, overlap.shape)
        copy = np.zeros_like(mask)
        copy[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
        copies.append(copy)
        if len(copies) == n:
            return np.stack(copies)
    raise DegenerateInputError(
        f"null_masks: {100 * n} angles placed {len(copies)} of {n} copies of a "
        f"{size}-pixel mask in a {height} × {width} scene off every candidate"
    )


def null_pvalue(score: float, null_scores: Float[ArrayLike, " S"]) -> float:
    """The fraction of null copies that score at least as well as the candidate.

    With ``D`` the candidate's score and ``D_s`` the ``S'`` finite scores of
    its null copies (non-finite ones, copies that could not be scored, are
    left out),

    ``p̂ = (1/S') ∑_s 1[D_s ≤ D]``,

    ties counting as at least as good. Lower scores are better, as for
    ``D_norm``; for a score where higher is better (a likelihood ratio),
    pass both negated. ``p̂`` moves in steps of ``1/S'`` and ``p̂ = 0`` says
    only that no copy did as well; the Monte Carlo p-value, never zero and
    valid when the candidate and the copies are exchangeable under the null
    (Phipson and Smyth 2010), is ``(1 + S'·p̂) / (1 + S')``.

    Args:
        score: The candidate's score ``D``, e.g. its ``D_norm``.
        null_scores: The scores of its null copies, ``(S,)``, from
            [`null_masks`][plumax.vetting.null_masks] vetted as
            the candidate was.

    Returns:
        ``p̂``, in [0, 1].

    Raises:
        ShapeError: If ``null_scores`` is not 1-D.
        DegenerateInputError: If ``score`` is not finite or no null score
            is.

    Examples:
        >>> import numpy as np
        >>> from plumax.vetting import null_pvalue
        >>> null_pvalue(0.3, np.array([0.2, 0.3, 0.5, 0.9]))
        0.5
        >>> null_pvalue(0.1, np.array([0.2, np.nan, 0.5]))
        0.0
    """
    null_scores = jnp.asarray(null_scores)
    if null_scores.ndim != 1:
        raise ShapeError(
            f"null_pvalue: null_scores must be (S,), got {null_scores.shape}"
        )
    score = float(score)
    if not np.isfinite(score):
        raise DegenerateInputError(f"null_pvalue: the score is {score}, not finite")
    finite = jnp.isfinite(null_scores)  # (S,)
    n_finite = int(jnp.sum(finite))
    if n_finite == 0:
        raise DegenerateInputError("null_pvalue: no null score is finite")
    at_least_as_good = jnp.sum(finite & (null_scores <= score))
    return float(at_least_as_good) / n_finite
