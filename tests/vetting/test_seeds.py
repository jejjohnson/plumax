"""select_seeds and background_pool, a candidate's pixel sets (#154).

The reference for each is NumPy and SciPy written from the issue's
definitions: the top ``k`` of the mask's scores at or below its percentile
cap, a 3 × 3 ``binary_dilation`` intersected with the mask, and the seeds'
bounding box widened by ``box``.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from radtrax import DegenerateInputError, ParameterError, ShapeError

from plumax.vetting import background_pool, select_seeds
from plumax.vetting._morphology import dilate


ndimage = pytest.importorskip("scipy.ndimage")

H, W = 48, 64
SQUARE = np.ones((3, 3), dtype=bool)


@pytest.fixture(scope="module")
def mf() -> np.ndarray:
    """Distinct scores: a plume-like bump on noise, ``(H, W)``."""
    rng = np.random.default_rng(0)
    i, j = np.mgrid[:H, :W]
    bump = 2.0 * np.exp(-((i - 20.0) ** 2 + (j - 30.0) ** 2) / 40.0)
    return bump + 0.1 * rng.standard_normal((H, W))


@pytest.fixture(scope="module")
def mask() -> np.ndarray:
    """An irregular candidate around the bump, ``(H, W)``."""
    i, j = np.mgrid[:H, :W]
    return ((i - 20.0) ** 2 / 64.0 + (j - 30.0) ** 2 / 144.0) <= 1.0


def _top_k_below_cap(
    mf: np.ndarray, mask: np.ndarray, k: int, cap_percentile: float
) -> np.ndarray:
    """The issue's definition with ``grow=0``, in NumPy."""
    cap = np.percentile(mf[mask], cap_percentile)
    flat = np.flatnonzero(mask & (mf <= cap))
    keep = flat[np.argsort(mf.ravel()[flat])[::-1][:k]]
    out = np.zeros(mf.size, dtype=bool)
    out[keep] = True
    return out.reshape(mf.shape)


# ── select_seeds ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("k", "cap_percentile"), [(40, 99.0), (10, 90.0), (1, 100.0), (500, 99.0)]
)
def test_grow_zero_is_the_k_largest_below_the_cap(
    mf: np.ndarray, mask: np.ndarray, k: int, cap_percentile: float
) -> None:
    seeds = select_seeds(mf, mask, k=k, cap_percentile=cap_percentile, grow=0)
    expected = _top_k_below_cap(mf, mask, k, cap_percentile)
    np.testing.assert_array_equal(seeds, expected)
    # |I| = min(k, |{p ∈ M : MF(p) ≤ q}|): k = 500 exceeds the mask.
    eligible = mask & (mf <= np.percentile(mf[mask], cap_percentile))
    assert int(seeds.sum()) == min(k, int(eligible.sum()))


def test_the_cap_drops_the_masks_strongest_pixels(
    mf: np.ndarray, mask: np.ndarray
) -> None:
    seeds = np.asarray(select_seeds(mf, mask, k=40, grow=0))
    cap = np.percentile(mf[mask], 99.0)
    assert mf[seeds].max() <= cap
    assert (mf[mask] > cap).any()  # the cap removed something


@pytest.mark.parametrize("grow", [1, 2, 3])
def test_growth_is_a_3x3_dilation_inside_the_mask(
    mf: np.ndarray, mask: np.ndarray, grow: int
) -> None:
    expected = np.asarray(select_seeds(mf, mask, k=40, grow=0))
    for _ in range(grow):
        expected = ndimage.binary_dilation(expected, SQUARE) & mask
    np.testing.assert_array_equal(select_seeds(mf, mask, k=40, grow=grow), expected)


@pytest.mark.parametrize("steps", [0, 1, 2])
def test_dilate_matches_scipy_up_to_the_edges(steps: int) -> None:
    mask = np.zeros((7, 9), dtype=bool)
    mask[0, 0] = mask[6, 4] = mask[3, 8] = mask[3, 3] = True
    expected = (
        ndimage.binary_dilation(mask, SQUARE, iterations=steps) if steps else mask
    )
    np.testing.assert_array_equal(dilate(mask, steps), expected)


def test_seeds_are_inside_the_mask_and_skip_non_finite_scores(
    mf: np.ndarray, mask: np.ndarray
) -> None:
    bad = mf.copy()
    top = np.unravel_index(np.argmax(np.where(mask, mf, -np.inf)), mf.shape)
    bad[top] = np.nan
    bad[20, 31] = np.inf
    seeds = np.asarray(select_seeds(bad, mask, k=40, grow=2))
    assert not (seeds & ~mask).any()
    assert not (seeds & ~np.isfinite(bad)).any()


# ── background_pool ────────────────────────────────────────────────────────


def _pool_reference(
    mf: np.ndarray,
    seeds: np.ndarray,
    threshold: float,
    exclude: np.ndarray | None,
    box: int,
) -> np.ndarray:
    rows, cols = np.nonzero(seeds)
    near = np.zeros(mf.shape, dtype=bool)
    near[
        max(rows.min() - box, 0) : rows.max() + box + 1,
        max(cols.min() - box, 0) : cols.max() + box + 1,
    ] = True
    pool = near & ~seeds & (np.abs(mf) < threshold)
    if exclude is not None:
        pool &= ~exclude.any(axis=0)
    return pool


@pytest.fixture(scope="module")
def candidates(mask: np.ndarray) -> np.ndarray:
    """This candidate and a second one inside the box, ``(C, H, W)``."""
    other = np.zeros((H, W), dtype=bool)
    other[2:8, 40:50] = True
    return np.stack([mask, other])


@pytest.mark.parametrize("box", [3, 10, 200])
@pytest.mark.parametrize("threshold", [0.05, 0.2])
def test_pool_respects_box_threshold_and_exclusions(
    mf: np.ndarray,
    mask: np.ndarray,
    candidates: np.ndarray,
    box: int,
    threshold: float,
) -> None:
    seeds = np.asarray(select_seeds(mf, mask))
    pool = np.asarray(
        background_pool(mf, seeds, threshold=threshold, exclude=candidates, box=box)
    )
    np.testing.assert_array_equal(
        pool, _pool_reference(mf, seeds, threshold, candidates, box)
    )
    assert not (pool & seeds).any()
    assert not (pool & candidates.any(axis=0)).any()
    assert np.all(np.abs(mf[pool]) < threshold)


def test_without_exclude_only_the_seeds_are_kept_out(
    mf: np.ndarray, mask: np.ndarray
) -> None:
    seeds = np.asarray(select_seeds(mf, mask))
    pool = np.asarray(background_pool(mf, seeds, threshold=0.2, box=5))
    np.testing.assert_array_equal(pool, _pool_reference(mf, seeds, 0.2, None, 5))
    assert (pool & mask).any()  # mask pixels that are not seeds can join


def test_the_threshold_is_strict_and_nan_never_joins() -> None:
    mf = np.zeros((5, 5))
    mf[0, 0], mf[0, 1], mf[0, 2] = 1.0, -1.0, np.nan
    seeds = np.zeros((5, 5), dtype=bool)
    seeds[2, 2] = True
    pool = np.asarray(background_pool(mf, seeds, threshold=1.0, box=5))
    assert not pool[0, :3].any()
    assert pool.sum() == 25 - 1 - 3


# ── jit ────────────────────────────────────────────────────────────────────


def test_both_run_under_jit_with_static_k_grow_box(
    mf: np.ndarray, mask: np.ndarray, candidates: np.ndarray
) -> None:
    seeds_fn = jax.jit(functools.partial(select_seeds, k=25, grow=1))
    pool_fn = jax.jit(functools.partial(background_pool, box=7))
    seeds = seeds_fn(jnp.asarray(mf), jnp.asarray(mask))
    np.testing.assert_array_equal(seeds, select_seeds(mf, mask, k=25, grow=1))
    pool = pool_fn(jnp.asarray(mf), seeds, threshold=0.1, exclude=candidates)
    np.testing.assert_array_equal(
        pool, background_pool(mf, seeds, threshold=0.1, exclude=candidates, box=7)
    )


def test_cap_percentile_may_be_traced_under_jit(
    mf: np.ndarray, mask: np.ndarray
) -> None:
    # Only k and grow are static: a traced percentile must work (radtrax#263 review).
    fn = jax.jit(select_seeds, static_argnames=("k", "grow"))
    for p in (90.0, 99.0, 100.0):
        np.testing.assert_array_equal(
            fn(jnp.asarray(mf), jnp.asarray(mask), k=25, cap_percentile=p, grow=1),
            select_seeds(mf, mask, k=25, cap_percentile=p, grow=1),
        )
    # Out of range when traced: eqx.error_if, as for the empty mask (EquinoxRuntimeError
    # under eqx.filter_jit; jax.jit raises jax.errors.JaxRuntimeError instead).
    traced = eqx.filter_jit(functools.partial(select_seeds, k=25, grow=1))
    with pytest.raises(RuntimeError, match=r"cap_percentile must be in \(0, 100\]"):
        traced(jnp.asarray(mf), jnp.asarray(mask), cap_percentile=jnp.asarray(0.0))


def test_empty_inputs_fail_at_run_time_under_jit() -> None:
    mf = jnp.zeros((4, 4))
    empty = jnp.zeros((4, 4), dtype=bool)
    with pytest.raises(RuntimeError, match="no pixel with a finite"):
        eqx.filter_jit(select_seeds)(mf, empty)
    with pytest.raises(RuntimeError, match="the pool is empty"):
        eqx.filter_jit(functools.partial(background_pool, threshold=1.0))(mf, empty)


# ── errors ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"cap_percentile": 0.0},
        {"cap_percentile": 100.5},
        {"cap_percentile": float("nan")},
        {"grow": -1},
    ],
)
def test_select_seeds_rejects_bad_parameters(
    mf: np.ndarray, mask: np.ndarray, kwargs: dict[str, float]
) -> None:
    with pytest.raises(ParameterError):
        select_seeds(mf, mask, **kwargs)


def test_select_seeds_rejects_an_empty_or_unscored_mask(mf: np.ndarray) -> None:
    with pytest.raises(DegenerateInputError, match="no pixel"):
        select_seeds(mf, np.zeros((H, W), dtype=bool))
    mask = np.zeros((H, W), dtype=bool)
    mask[0, 0] = True
    with pytest.raises(DegenerateInputError, match="no pixel"):
        select_seeds(np.full((H, W), np.nan), mask)


def test_background_pool_rejects_an_empty_pool_and_a_negative_box(
    mf: np.ndarray, mask: np.ndarray
) -> None:
    seeds = np.asarray(select_seeds(mf, mask))
    with pytest.raises(DegenerateInputError, match="the pool is empty"):
        background_pool(mf, seeds, threshold=0.0)
    with pytest.raises(DegenerateInputError, match="the pool is empty"):
        background_pool(mf, np.zeros((H, W), dtype=bool), threshold=1.0)
    with pytest.raises(ParameterError, match="box"):
        background_pool(mf, seeds, threshold=1.0, box=-1)


def test_shapes_are_checked(mf: np.ndarray, mask: np.ndarray) -> None:
    with pytest.raises(ShapeError, match="mask has shape"):
        select_seeds(mf, mask[:-1])
    with pytest.raises(ShapeError, match=r"mf must be \(H, W\)"):
        select_seeds(mf[None], mask[None])
    with pytest.raises(ShapeError, match="seeds has shape"):
        background_pool(mf, mask[:, :-1], threshold=1.0)
    with pytest.raises(ShapeError, match="exclude must be"):
        background_pool(mf, mask, threshold=1.0, exclude=mask)


def test_errors_are_radtrax_classes_and_value_errors(
    mf: np.ndarray, mask: np.ndarray
) -> None:
    """plumax has no hierarchy of its own: vetting raises radtrax's public
    classes, each also a ``ValueError``, so ``except ValueError`` keeps working.
    """
    with pytest.raises(ValueError, match="k must be"):
        select_seeds(mf, mask, k=0)
    with pytest.raises(ValueError, match="no pixel"):
        select_seeds(mf, np.zeros((H, W), dtype=bool))
