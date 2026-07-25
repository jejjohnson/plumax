"""Robust background statistics for matched-filter retrievals.

The matched filter needs a background mean spectrum ``μ`` and an inverse
covariance ``Σ⁻¹`` estimated from the *clean* pixels in the scene. Real
scenes have contaminants — plumes, clouds, shadows, bright targets — that
would bias a naive mean and inflate a naive covariance, so both estimators
here are written to be robust against outlier pixels.

- :func:`trimmed_mean_spectrum` — per-channel trimmed mean. Cheap, robust
  to a few percent of outliers, matches a Gaussian mean in the clean limit.
- :func:`robust_lowrank_covariance` — low-rank plus diagonal-regularisation
  covariance via truncated SVD on the mean-subtracted, trimmed pixel stack.
  Returns ``(Σ, Σ⁻¹)`` ready for the matched filter.

Both operate on a pixel × band array (or an ``xarray.DataArray`` with
``band`` as the last dim). The ``trim_frac`` parameter is a two-sided
fraction in ``[0, 0.5)`` — ``0.1`` removes the brightest and darkest 10%
of pixels per channel before averaging.

Heavily simplified from ``jej_vc_snippets/methane_retrieval/matched_filter_{mean,covariance}.py``
(GMM-based background estimators and dynamic-mode rejection dropped — the
trimmed mean + truncated SVD covers the demo regime with far less tuning).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import trim_mean


def _flatten_pixels(
    radiance: np.ndarray,
    band_axis: int,
) -> np.ndarray:
    """Rearrange to ``(n_pixels, n_bands)``; band_axis is the band dim of ``radiance``."""
    arr = np.asarray(radiance)
    bands_first = np.moveaxis(arr, band_axis, 0)  # (n_bands, ...)
    n_bands = bands_first.shape[0]
    return bands_first.reshape(n_bands, -1).T  # (n_pixels, n_bands)


def _noise_matched_floor(
    singular_values: np.ndarray, rank: int, n_samples: int
) -> float:
    """Diagonal floor ``λ`` = mean of the *discarded* covariance eigenvalues.

    The retained rank captures the signal subspace; the remaining singular
    directions carry the sensor noise, with covariance eigenvalues
    ``S[rank:]² / N``. Setting ``λ`` to their mean makes ``Σ⁻¹`` weight the
    discarded subspace at the true noise level (SNR-optimal) rather than at an
    arbitrary fixed floor. Falls back to a small positive value when no modes
    are discarded (rank-complete) or the estimate is degenerate.
    """
    discarded = np.asarray(singular_values, dtype=float)[rank:]
    if discarded.size > 0:
        lam = float(np.mean(discarded**2) / max(n_samples, 1))
        if lam > 0.0:
            return lam
    return 1e-6


def trimmed_mean_spectrum(
    radiance: np.ndarray,
    *,
    trim_frac: float = 0.1,
    band_axis: int = 0,
) -> np.ndarray:
    """Per-channel trimmed mean ``μ_b``.

    Parameters
    ----------
    radiance : np.ndarray
        Radiance cube, shape ``(n_bands, ny, nx)`` by default (``band_axis=0``),
        or any array where one axis indexes the band.
    trim_frac : float
        Fraction trimmed from *each* tail per channel, in ``[0, 0.5)``.
        Default 0.1 removes the top and bottom 10% per band.
    band_axis : int
        Band axis of ``radiance``. Default 0.

    Returns
    -------
    mu : np.ndarray
        Trimmed mean per band, shape ``(n_bands,)``.
    """
    if not (0.0 <= trim_frac < 0.5):
        raise ValueError(
            f"trimmed_mean_spectrum: `trim_frac` must be in [0, 0.5) (got {trim_frac!r})"
        )
    flat = _flatten_pixels(radiance, band_axis)  # (n_pixels, n_bands)
    return np.asarray(trim_mean(flat, trim_frac, axis=0), dtype=float)


def _estimate_lowrank_model(
    radiance: np.ndarray,
    *,
    rank: int | None,
    trim_frac: float,
    regularization: float | None,
    band_axis: int,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Shared trim + truncated-SVD covariance model (single source of truth).

    Centres the pixel stack with a trimmed mean, drops the extreme-energy
    pixels, truncates the SVD to ``rank``, and picks the diagonal floor. Returns
    ``(mu, U, d, lam)`` such that ``Σ = U diag(d) Uᵀ + lam·I``. Consumed by both
    :func:`robust_lowrank_covariance` (which materialises Σ/Σ⁻¹) and
    :func:`plumax.radtran.gaussx_solve.build_lowrank_covariance_operator` (which
    wraps it in a gaussx operator) so the two paths cannot drift apart — the
    dense-vs-operator agreement tests are only meaningful while they don't.

    Parameters
    ----------
    radiance, band_axis
        Radiance cube and its band axis.
    rank
        Truncation rank; ``None`` → ``min(n_bands - 1, 16)``.
    trim_frac
        Two-sided energy trim fraction in ``[0, 0.5)``.
    regularization
        Diagonal floor; ``None`` → noise-matched (:func:`_noise_matched_floor`).
    context
        Caller name, prepended to validation-error messages.

    Returns
    -------
    mu : np.ndarray, shape ``(n_bands,)``
    U : np.ndarray, shape ``(n_bands, rank)`` — principal directions (columns).
    d : np.ndarray, shape ``(rank,)`` — retained covariance eigenvalues ``S_k²/N``.
    lam : float — diagonal floor.
    """
    flat = _flatten_pixels(radiance, band_axis)  # (n_pixels, n_bands)
    n_pixels, n_bands = flat.shape
    if n_pixels < 2:
        raise ValueError(f"{context}: need ≥ 2 pixels (got {n_pixels})")
    if not (0.0 <= trim_frac < 0.5):
        raise ValueError(
            f"{context}: `trim_frac` must be in [0, 0.5) (got {trim_frac!r})"
        )
    if regularization is not None and regularization <= 0.0:
        raise ValueError(f"{context}: `regularization` must be > 0.")
    if rank is None:
        rank = min(n_bands - 1, 16)
    rank = max(1, min(int(rank), n_bands))

    mu = trimmed_mean_spectrum(radiance, trim_frac=trim_frac, band_axis=band_axis)
    centred = flat - mu[None, :]

    # Drop the top/bottom `trim_frac` pixels by total energy so bright outliers
    # don't dominate the SVD.
    if trim_frac > 0.0:
        energy = np.linalg.norm(centred, axis=1)
        lo, hi = np.quantile(energy, [trim_frac, 1.0 - trim_frac])
        keep = (energy >= lo) & (energy <= hi)
        if keep.sum() < n_bands:
            raise ValueError(
                f"{context}: trimming left fewer pixels than bands; "
                "reduce `trim_frac` or enlarge the scene."
            )
        centred = centred[keep]

    # Deterministic truncated SVD (small scenes; keeps tests reproducible).
    _, S, Vt = np.linalg.svd(centred, full_matrices=False)
    n_kept = centred.shape[0]
    U = Vt[:rank].T  # (n_bands, rank) — principal directions as columns
    d = S[:rank] ** 2 / max(n_kept, 1)  # retained covariance eigenvalues
    lam = (
        _noise_matched_floor(S, rank, n_kept)
        if regularization is None
        else float(regularization)
    )
    return mu, U, d, lam


def robust_lowrank_covariance(
    radiance: np.ndarray,
    *,
    rank: int | None = None,
    trim_frac: float = 0.1,
    regularization: float | None = None,
    band_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust low-rank covariance + inverse ``(Σ, Σ⁻¹)``.

    The pixel stack is centred with a trimmed mean, the extreme 2·``trim_frac``
    pixels (by total energy) are removed, and the covariance is estimated as

        Σ = U_k · S_k² · U_kᵀ / N + λ · I

    where ``U_k`` are the top-``k`` left singular vectors of the centred
    matrix. The added diagonal term ``λ · I`` stabilises the inverse when
    the covariance is ill-conditioned. Returns both ``Σ`` and ``Σ⁻¹`` since
    the matched filter needs the inverse and diagnostics sometimes want the
    forward covariance too.

    Parameters
    ----------
    radiance : np.ndarray
        Shape ``(n_bands, ny, nx)`` by default.
    rank : int or None
        Truncation rank. ``None`` uses ``min(n_bands - 1, 16)`` — the
        practical sweet spot for multispectral data where most of the
        variance is in a handful of modes.
    trim_frac : float
        Per-pixel energy-based trim fraction in ``[0, 0.5)``. Default 0.1.
    regularization : float or None
        Diagonal Tikhonov floor ``λ`` on the discarded subspace. ``None``
        (default) uses the **noise-matched** floor — the mean of the discarded
        covariance eigenvalues ``mean(S[rank:]²)/N`` — so ``Σ⁻¹`` weights that
        subspace at the real sensor-noise level instead of an arbitrary
        constant. A floor too small over-weights the noise subspace (inflating
        false alarms); too large leaks signal into it. Pass a float to override.
    band_axis : int
        Band axis of ``radiance``. Default 0.

    Returns
    -------
    Sigma, Sigma_inv : np.ndarray
        Symmetric PSD matrices, shape ``(n_bands, n_bands)``.
    """
    mu, U, d, lam = _estimate_lowrank_model(
        radiance,
        rank=rank,
        trim_frac=trim_frac,
        regularization=regularization,
        band_axis=band_axis,
        context="robust_lowrank_covariance",
    )
    n_bands = mu.shape[0]
    # Σ = U diag(d) Uᵀ + λ I (d = S_k² / N are the retained covariance eigenvalues).
    Sigma = (U * d) @ U.T + lam * np.eye(n_bands)

    # Woodbury inverse for efficiency: (λI + Uᵀ D U)⁻¹ where
    #   U = V_k (rank, n_bands), D = diag(S_k² / N) (rank, rank).
    # But since n_bands is small for multispectral, direct inversion is fine
    # and keeps the code obvious.
    Sigma_inv = np.linalg.inv(Sigma)
    return Sigma, Sigma_inv
