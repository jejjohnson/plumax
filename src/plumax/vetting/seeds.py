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

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float
from radtrax import DegenerateInputError, ParameterError, ShapeError

from plumax.vetting._morphology import dilate


__all__ = ["background_pool", "select_seeds"]


def select_seeds(
    mf: Float[ArrayLike, "H W"],
    mask: Bool[ArrayLike, "H W"],
    *,
    k: int = 40,
    cap_percentile: Float[ArrayLike, ""] = 99.0,
    grow: int = 1,
) -> Bool[Array, "H W"]:
    """The ``k`` strongest matched-filter pixels of a candidate, grown inside it.

    With ``M`` the mask's pixels of finite score and ``q`` the
    ``cap_percentile``-th percentile of the score over ``M``,

    ``I = grow(top_k{p ∈ M : MF(p) ≤ q}) ∩ M``,

    where each growth step is one 3 × 3 dilation intersected with ``M`` (the
    reference code's Gaussian filter with σ = 1 and threshold 0.05 is the
    same dilation). With ``grow=0``, ``|I| = min(k, |{p ∈ M : MF(p) ≤ q}|)``
    exactly: ties at the ``k``-th score are broken by the lower flat index
    (``jax.lax.top_k``).

    Args:
        mf: Matched-filter score, e.g. ``ΔX`` in mol/m², ``(H, W)``; a
            non-finite pixel is never a seed.
        mask: The candidate, ``(H, W)``.
        k: Number of seeds before growth (static); the reference code's 40.
        cap_percentile: Scores above this percentile of the mask's are not
            seeds, in (0, 100]; the reference code's 99.
        grow: Growth steps inside the mask (static); ``0`` keeps the top
            ``k``.

    Returns:
        The seeds, ``(H, W)``, a subset of ``mask``.

    Raises:
        ParameterError: If ``k < 1``, ``cap_percentile`` is outside
            (0, 100] or ``grow < 0``. ``k`` and ``grow`` are static under
            ``jit``; ``cap_percentile`` may be traced, and then its range
            check runs with ``eqx.error_if``, as the empty-mask check below.
        ShapeError: If ``mf`` is not 2-D or ``mask`` has another shape.
        DegenerateInputError: If the mask holds no pixel with a finite
            score. Under ``jit`` this check runs with ``eqx.error_if`` and
            raises a ``RuntimeError`` instead (``equinox.EquinoxRuntimeError``
            under ``eqx.filter_jit``, ``jax.errors.JaxRuntimeError`` under
            ``jax.jit``).

    Examples:
        >>> import jax.numpy as jnp
        >>> from plumax.vetting import select_seeds
        >>> mf = jnp.arange(25.0).reshape(5, 5)
        >>> mask = jnp.zeros((5, 5), bool).at[1:4, 1:4].set(True)
        >>> print(
        ...     select_seeds(mf, mask, k=2, cap_percentile=90.0, grow=0).astype(int)
        ... )
        [[0 0 0 0 0]
         [0 0 0 0 0]
         [0 0 0 0 0]
         [0 1 1 0 0]
         [0 0 0 0 0]]
        >>> print(select_seeds(mf, mask, k=2, cap_percentile=90.0).astype(int))
        [[0 0 0 0 0]
         [0 0 0 0 0]
         [0 1 1 1 0]
         [0 1 1 1 0]
         [0 0 0 0 0]]
    """
    if k < 1:
        raise ParameterError(f"select_seeds: k must be >= 1, got {k}")
    cap_percentile = _check_percentile(cap_percentile)
    if grow < 0:
        raise ParameterError(f"select_seeds: grow must be >= 0, got {grow}")
    mf = jnp.asarray(mf)
    mask = jnp.asarray(mask, dtype=bool)
    _check_image("select_seeds", mf, mask, "mask")
    inside = _require_any(
        mask & jnp.isfinite(mf),
        "select_seeds: the mask holds no pixel with a finite matched-filter score",
    )
    cap = jnp.nanpercentile(jnp.where(inside, mf, jnp.nan), cap_percentile)
    score = jnp.where(inside & (mf <= cap), mf, -jnp.inf).ravel()
    _, top = jax.lax.top_k(score, min(k, score.size))
    seeds = jnp.zeros(score.shape, dtype=bool).at[top].set(True)
    seeds = (seeds & jnp.isfinite(score)).reshape(mf.shape)
    for _ in range(grow):
        seeds = dilate(seeds) & inside
    return seeds


def background_pool(
    mf: Float[ArrayLike, "H W"],
    seeds: Bool[ArrayLike, "H W"],
    *,
    threshold: Float[ArrayLike, ""],
    exclude: Bool[ArrayLike, " C H W"] | None = None,
    box: int = 200,
) -> Bool[Array, "H W"]:
    """Background pixels near the seeds, outside every candidate, with small scores.

    ``O = box(I) ∖ I ∖ ⋃_c M_c ∩ {p : |MF(p)| < τ}``, where ``box(I)`` is the
    seeds' bounding box widened by ``box`` pixels on every side (clipped to
    the image) and ``M_c`` the ``C`` candidate masks in ``exclude``. The box
    keeps each pair under the same sun, elevation and water vapour; the
    threshold keeps plume pixels out of the pool. Pass every candidate of
    the scene, this one included, as ``exclude``: without it only the seeds
    themselves are kept out.

    Args:
        mf: Matched-filter score, ``(H, W)``, in the units of ``threshold``;
            a non-finite pixel is never in the pool.
        seeds: The in-plume set, ``(H, W)``, from
            [`select_seeds`][plumax.vetting.select_seeds].
        threshold: ``τ``: pixels with ``|MF|`` strictly below it join the
            pool, in mol/m² for a ΔX score
            (radtrax's ``AbsorptionWindow.mf_background_mol_m2``; the
            reference code's 30 ppm·m).
        exclude: The scene's ``C`` candidate masks, ``(C, H, W)``; none of
            their pixels joins the pool.
        box: Margin, in pixels, around the seeds' bounding box (static); the
            reference code's 200.

    Returns:
        The pool, ``(H, W)``, disjoint from ``seeds`` and from every mask in
        ``exclude``.

    Raises:
        ParameterError: If ``box < 0``.
        ShapeError: If ``mf`` is not 2-D, ``seeds`` has another shape, or
            ``exclude`` is not ``(C, H, W)``.
        DegenerateInputError: If the pool is empty (no seeds, or no pixel
            near them passes the cuts). Under ``jit`` this check runs with
            ``eqx.error_if`` and raises a ``RuntimeError`` instead
            (``equinox.EquinoxRuntimeError`` under ``eqx.filter_jit``,
            ``jax.errors.JaxRuntimeError`` under ``jax.jit``).

    Examples:
        >>> import jax.numpy as jnp
        >>> from plumax.vetting import background_pool
        >>> mf = jnp.zeros((6, 6)).at[0, :].set(5.0)
        >>> seeds = jnp.zeros((6, 6), bool).at[2, 2].set(True)
        >>> candidate = jnp.zeros((1, 6, 6), bool).at[0, 2:4, 2:4].set(True)
        >>> pool = background_pool(
        ...     mf, seeds, threshold=1.0, exclude=candidate, box=1
        ... )
        >>> print(pool.astype(int))
        [[0 0 0 0 0 0]
         [0 1 1 1 0 0]
         [0 1 0 0 0 0]
         [0 1 0 0 0 0]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]]
    """
    if box < 0:
        raise ParameterError(f"background_pool: box must be >= 0, got {box}")
    mf = jnp.asarray(mf)
    seeds = jnp.asarray(seeds, dtype=bool)
    _check_image("background_pool", mf, seeds, "seeds")
    height, width = mf.shape
    rows = jnp.any(seeds, axis=1)  # (H,)
    cols = jnp.any(seeds, axis=0)  # (W,)
    r0, r1 = jnp.argmax(rows), height - 1 - jnp.argmax(rows[::-1])
    c0, c1 = jnp.argmax(cols), width - 1 - jnp.argmax(cols[::-1])
    i = jnp.arange(height)[:, None]
    j = jnp.arange(width)[None, :]
    near = (i >= r0 - box) & (i <= r1 + box) & (j >= c0 - box) & (j <= c1 + box)
    pool = near & jnp.any(rows) & ~seeds & (jnp.abs(mf) < threshold)
    if exclude is not None:
        exclude = jnp.asarray(exclude, dtype=bool)
        if exclude.ndim != 3 or exclude.shape[1:] != mf.shape:
            raise ShapeError(
                f"background_pool: exclude must be (C, {height}, {width}), "
                f"got {exclude.shape}"
            )
        pool &= ~jnp.any(exclude, axis=0)
    return _require_any(
        pool,
        "background_pool: the pool is empty; no pixel near the seeds is "
        "outside every candidate with |MF| below the threshold",
    )


def _check_image(
    name: str, mf: Float[Array, "H W"], other: Bool[Array, "H W"], other_name: str
) -> None:
    """Raise unless ``mf`` is 2-D and ``other`` has its shape."""
    if mf.ndim != 2:
        raise ShapeError(f"{name}: mf must be (H, W), got {mf.shape}")
    if other.shape != mf.shape:
        raise ShapeError(
            f"{name}: {other_name} has shape {other.shape}, mf has {mf.shape}"
        )


def _check_percentile(p: Float[ArrayLike, ""]) -> Float[Array, ""]:
    """``p`` as an array, after failing outside (0, 100]: eagerly, or under ``jit``."""
    p = jnp.asarray(p, dtype=float)
    bad = ~((p > 0.0) & (p <= 100.0))
    message = "select_seeds: cap_percentile must be in (0, 100]"
    try:
        out_of_range = bool(bad)
    except (jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError):
        return eqx.error_if(p, bad, message)
    if out_of_range:
        raise ParameterError(f"{message}, got {float(p)}")
    return p


def _require_any(mask: Bool[Array, "H W"], message: str) -> Bool[Array, "H W"]:
    """``mask``, after failing on an empty one: eagerly, or under ``jit``."""
    try:
        empty = not bool(jnp.any(mask))
    except (jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError):
        return eqx.error_if(mask, ~jnp.any(mask), message)
    if empty:
        raise DegenerateInputError(message)
    return mask
