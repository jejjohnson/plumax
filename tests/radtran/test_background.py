"""Robust background estimators: trimmed mean + low-rank covariance."""

from __future__ import annotations

import numpy as np
import pytest

from plumax.radtran.background import (
    robust_lowrank_covariance,
    trimmed_mean_spectrum,
)


def _synthetic_scene(
    n_bands: int = 2,
    shape: tuple[int, int] = (32, 32),
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    loc = np.linspace(0.3, 0.5, n_bands)
    base = rng.normal(loc=loc, scale=0.02, size=(*shape, n_bands))
    # Rearrange to (bands, ny, nx).
    return np.moveaxis(base, -1, 0)


def test_trimmed_mean_recovers_true_mean():
    scene = _synthetic_scene()
    mu = trimmed_mean_spectrum(scene, trim_frac=0.1)
    assert mu.shape == (2,)
    assert mu[0] == pytest.approx(0.3, abs=0.01)
    assert mu[1] == pytest.approx(0.5, abs=0.01)


def test_trimmed_mean_rejects_spike_outliers():
    scene = _synthetic_scene()
    # Inject a very bright outlier in ~1% of pixels.
    mask = np.random.default_rng(42).random(scene.shape[1:]) < 0.01
    scene[:, mask] += 10.0
    mu = trimmed_mean_spectrum(scene, trim_frac=0.1)
    # Still recovers the clean mean within ~2σ.
    assert mu[0] == pytest.approx(0.3, abs=0.02)
    assert mu[1] == pytest.approx(0.5, abs=0.02)


def test_trimmed_mean_rejects_bad_trim_frac():
    scene = _synthetic_scene()
    with pytest.raises(ValueError, match="trim_frac"):
        trimmed_mean_spectrum(scene, trim_frac=0.5)


def test_covariance_is_symmetric_and_positive_definite():
    scene = _synthetic_scene(n_bands=4, shape=(16, 16))
    Sigma, Sigma_inv = robust_lowrank_covariance(scene, rank=3)
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)
    # Sigma · Sigma_inv == I.
    np.testing.assert_allclose(Sigma @ Sigma_inv, np.eye(4), atol=1e-6)
    # PD check via eigenvalues.
    eigvals = np.linalg.eigvalsh(Sigma)
    assert (eigvals > 0).all()


def test_covariance_rejects_tiny_scene():
    # Only 1 pixel.
    scene = np.array([[[0.3]]], dtype=float)  # shape (1, 1, 1)
    with pytest.raises(ValueError, match="≥ 2 pixels"):
        robust_lowrank_covariance(scene)


def test_covariance_rejects_bad_regularization():
    scene = _synthetic_scene()
    with pytest.raises(ValueError, match="regularization"):
        robust_lowrank_covariance(scene, regularization=0.0)


def test_default_rank_is_reasonable():
    scene = _synthetic_scene(n_bands=10, shape=(16, 16))
    Sigma, _ = robust_lowrank_covariance(scene)  # rank=None → min(n_bands-1, 16)
    # Should still produce a full-rank Σ after the Tikhonov kick.
    assert np.linalg.matrix_rank(Sigma) == 10


def test_noise_matched_floor_far():
    # A low-rank background plus iid sensor noise. Retaining the signal rank and
    # flooring the discarded subspace at the *noise* level (default None) should
    # weight noise directions correctly, giving a tighter false-alarm
    # distribution than the fixed λ=1e-6 floor, which over-weights that subspace.
    rng = np.random.default_rng(0)
    n_bands, k_true, noise_sigma = 60, 5, 0.01
    modes = rng.standard_normal((k_true, n_bands))
    scale = np.array([1.0, 0.7, 0.5, 0.3, 0.2])

    def draw(n):
        amps = rng.standard_normal((n, k_true)) * scale
        return amps @ modes + rng.normal(scale=noise_sigma, size=(n, n_bands))

    train = draw(4000)
    scene = train.T  # (n_bands, n_pixels), band_axis=0
    target = rng.standard_normal(n_bands) * 0.05

    mu = trimmed_mean_spectrum(scene, trim_frac=0.0)
    _, inv_matched = robust_lowrank_covariance(scene, rank=k_true, trim_frac=0.0)
    _, inv_fixed = robust_lowrank_covariance(
        scene, rank=k_true, trim_frac=0.0, regularization=1e-6
    )

    # Fresh noise-only (no target) test pixels; matched-filter statistic.
    test = draw(5000)

    def far_std(inv):
        w = inv @ target
        eps = (test - mu) @ w / float(target @ w)
        return float(eps.std())

    assert far_std(inv_matched) < far_std(inv_fixed)
