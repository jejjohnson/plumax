"""null_masks and null_pvalue, the scene null for vetting (#155).

``null_masks`` is checked on what it promises (inside the scene, off every
candidate, at least ``min_fraction`` of the pixels, reproducible from the
key) and on its distribution: a mask at the rotation centre is placed
uniformly over the free pixels. ``null_pvalue`` is checked on known arrays
and on its null distribution. Gate VT2 (part) runs the whole chain on the
plume-vetting notebook's scene.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from radtrax import DegenerateInputError, ParameterError, ShapeError
from scipy import stats

from plumax.vetting import null_masks, null_pvalue
from tests.vetting._vetting_scene import notebook_scene, vet_d_norm


H, W = 24, 32


@pytest.fixture(scope="module")
def mask() -> np.ndarray:
    """An L-shaped candidate of 24 pixels."""
    m = np.zeros((H, W), bool)
    m[3:9, 4:7] = True
    m[7:9, 7:10] = True
    return m


@pytest.fixture(scope="module")
def exclude(mask: np.ndarray) -> np.ndarray:
    """The scene's candidates: this one and a block across the middle."""
    other = np.zeros((H, W), bool)
    other[10:16, 12:22] = True
    return np.stack([mask, other])


@pytest.fixture(scope="module")
def copies(mask: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    return null_masks(mask, 20, key=jax.random.key(3), exclude=exclude)


# ── what every copy promises ──────────────────────────────────────────────


def test_copies_are_off_every_candidate(
    copies: np.ndarray, exclude: np.ndarray
) -> None:
    assert copies.shape == (20, H, W) and copies.dtype == bool
    assert not (copies & exclude.any(0)).any()


def test_copies_keep_at_least_min_fraction_of_the_pixels(
    mask: np.ndarray, copies: np.ndarray
) -> None:
    assert (copies.sum(axis=(1, 2)) >= 0.9 * mask.sum()).all()
    whole = null_masks(mask, 10, key=jax.random.key(4), min_fraction=1.0)
    assert (whole.sum(axis=(1, 2)) >= mask.sum()).all()


def test_copies_are_rotated_and_moved(mask: np.ndarray, copies: np.ndarray) -> None:
    """Not the mask itself, and not all in one place or at one angle."""
    assert len({c.tobytes() for c in copies}) == len(copies)
    rows = [np.nonzero(c)[0].mean() for c in copies]
    assert np.ptp(rows) > 3.0


def test_the_mask_itself_is_avoided_without_exclude(mask: np.ndarray) -> None:
    out = null_masks(mask, 10, key=jax.random.key(5))
    assert not (out & mask).any()


# ── the key is the only randomness ────────────────────────────────────────


def test_the_same_key_gives_the_same_copies(
    mask: np.ndarray, exclude: np.ndarray, copies: np.ndarray
) -> None:
    again = null_masks(mask, 20, key=jax.random.key(3), exclude=exclude)
    np.testing.assert_array_equal(again, copies)
    # Drawn in order: fewer copies from the same key are a prefix.
    first = null_masks(mask, 5, key=jax.random.key(3), exclude=exclude)
    np.testing.assert_array_equal(first, copies[:5])


def test_different_keys_give_different_copies(
    mask: np.ndarray, exclude: np.ndarray, copies: np.ndarray
) -> None:
    other = null_masks(mask, 20, key=jax.random.key(7), exclude=exclude)
    assert not np.array_equal(other, copies)


def test_a_legacy_uint32_key_is_accepted(mask: np.ndarray) -> None:
    out = null_masks(mask, 3, key=jax.random.PRNGKey(0))
    assert out.shape == (3, H, W)


@pytest.mark.slow  # 1,500 draws
def test_a_copy_is_placed_uniformly_over_the_free_pixels() -> None:
    """A pixel at the rotation centre is moved to a uniform free pixel.

    scipy rotates about ``((H - 1)/2, (W - 1)/2)``, so on a 9 × 9 scene the
    centre pixel is its own rotation at every angle, and each copy is one
    pixel drawn uniformly from the 81 − 1 − 9 = 71 free ones. The bound is
    the chi-square test's own sampling distribution: under uniformity its
    p-value is U(0, 1), so a correct sampler fails it with probability 1e-4.
    """
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    other = np.zeros((9, 9), bool)
    other[0:3, 0:3] = True
    out = null_masks(m, 1500, key=jax.random.key(11), exclude=other[None])
    assert (out.sum(axis=(1, 2)) == 1).all()
    counts = out.sum(axis=0)[~(m | other)]  # (71,)
    assert counts.size == 71 and counts.sum() == 1500
    assert stats.chisquare(counts).pvalue > 1e-4


# ── failures ──────────────────────────────────────────────────────────────


def test_a_scene_too_small_for_the_copies_raises() -> None:
    """A mask over most of the scene has no free place: 100 · n angles, then out."""
    m = np.zeros((10, 10), bool)
    m[1:9, 1:9] = True
    with pytest.raises(DegenerateInputError, match="200 angles placed 0 of 2"):
        null_masks(m, 2, key=jax.random.key(0))


def test_an_empty_mask_raises() -> None:
    with pytest.raises(DegenerateInputError, match="empty"):
        null_masks(np.zeros((5, 5), bool), 3, key=jax.random.key(0))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0}, "n must be >= 1"),
        ({"min_fraction": 0.0}, r"min_fraction must be in \(0, 1\]"),
        ({"min_fraction": 1.5}, r"min_fraction must be in \(0, 1\]"),
    ],
)
def test_bad_parameters_raise(mask: np.ndarray, kwargs: dict, match: str) -> None:
    kwargs = {"n": 3, **kwargs}
    with pytest.raises(ParameterError, match=match):
        null_masks(mask, key=jax.random.key(0), **kwargs)


def test_shapes_are_checked(mask: np.ndarray) -> None:
    with pytest.raises(ShapeError, match=r"mask must be \(H, W\)"):
        null_masks(mask[None], 3, key=jax.random.key(0))
    with pytest.raises(ShapeError, match=r"exclude must be \(C, 24, 32\)"):
        null_masks(mask, 3, key=jax.random.key(0), exclude=mask)
    with pytest.raises(ShapeError, match=r"null_scores must be \(S,\)"):
        null_pvalue(0.5, np.zeros((2, 3)))


# ── null_pvalue ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.1, 0.0), (0.2, 0.25), (0.3, 0.5), (0.4, 0.5), (0.9, 1.0), (2.0, 1.0)],
)
def test_pvalue_on_known_scores_with_ties_counted(
    score: float, expected: float
) -> None:
    null = np.array([0.9, 0.2, 0.5, 0.3])
    assert null_pvalue(score, null) == expected


def test_non_finite_null_scores_are_left_out() -> None:
    null = np.array([0.2, np.nan, 0.5, np.inf, 0.9])
    assert null_pvalue(0.6, null) == pytest.approx(2 / 3)


def test_a_higher_is_better_score_is_passed_negated() -> None:
    """A likelihood ratio Λ: p̂ = mean 1[Λ_s ≥ Λ], from the negated scores."""
    lam = np.array([1.0, 3.0, 5.0, 5.0, 8.0])
    assert null_pvalue(-5.0, -lam) == pytest.approx(np.mean(lam >= 5.0))


def test_the_pvalue_is_uniform_under_the_null() -> None:
    """Candidate and copies exchangeable: ``S · p̂`` uniform on {0, …, S}.

    With ``S`` i.i.d. continuous null scores and a candidate drawn from the
    same law, the candidate's rank is uniform, so ``S · p̂`` is uniform on
    the ``S + 1`` values. Bounded by the chi-square test's own sampling
    distribution, as above (false failure rate 1e-4).
    """
    rng = np.random.default_rng(0)
    s, reps = 9, 1000
    draws = rng.standard_normal((reps, s + 1))
    k = np.array([round(s * null_pvalue(d[0], d[1:])) for d in draws])
    counts = np.bincount(k, minlength=s + 1)
    assert stats.chisquare(counts).pvalue > 1e-4


def test_degenerate_pvalue_inputs_raise() -> None:
    with pytest.raises(DegenerateInputError, match="not finite"):
        null_pvalue(np.nan, np.array([0.1, 0.2]))
    with pytest.raises(DegenerateInputError, match="no null score is finite"):
        null_pvalue(0.5, np.array([np.nan, np.nan]))


# ── gate VT2 (part): the notebook scene ───────────────────────────────────


@pytest.mark.slow  # 200 vettings and the scene's forward model
def test_vt2_scene_null_separates_the_plume_from_the_carbonate() -> None:
    """Gate VT2 (part): ``p̂_A = 0`` and ``p̂_B ≥ 0.05`` with 100 copies each.

    The notebook's cell 23: one key split per candidate, copies off both
    candidates, each vetted as the candidate was (``box=16``).
    """
    scene = notebook_scene()
    key = jax.random.key(0)
    p_hat = {}
    for name, m in zip("AB", scene.masks, strict=True):
        key, sub = jax.random.split(key)
        copies = null_masks(m, 100, key=sub, exclude=scene.masks)
        assert copies.shape == (100, *m.shape)
        assert not (copies & scene.masks.any(0)).any()
        null_scores = np.array([vet_d_norm(scene, c) for c in copies])
        p_hat[name] = null_pvalue(vet_d_norm(scene, m), null_scores)
    assert p_hat["A"] == 0.0
    assert p_hat["B"] >= 0.05
