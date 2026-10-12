"""Background mean and covariance estimators for the matched filter.

Estimation is not on the gradient path — the matched filter treats the
background :math:`(\\mu, \\Sigma)` as known inputs. That lets us lean on
scikit-learn's mature estimators (which have careful numerics, shrinkage,
robust variants, etc.) without any JAX/autodiff constraints. The outputs
are wrapped as :mod:`lineax` / :mod:`gaussx` linear operators so that
:func:`plumax.matched_filter.core.apply_image` can use the same
``gaussx.solve`` dispatch path for all covariance flavours:

============================== ====================================================
Estimator                      Operator returned
============================== ====================================================
``estimate_cov_empirical``     :class:`lineax.MatrixLinearOperator` (dense)
``estimate_cov_shrunk``        :class:`lineax.MatrixLinearOperator` (dense, shrunk)
``estimate_cov_lowrank``       :class:`gaussx.LowRankUpdate` (``λI + UDUᵀ``)
============================== ====================================================

The low-rank path uses scikit-learn's randomised :class:`TruncatedSVD`, which
is ``O(n · k)`` on a ``(n_samples, n_bands)`` matrix and is typically 10–100×
faster than a full SVD for the small ``k`` a matched filter needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Float


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


_RobustMethod = Literal["mean", "median", "trimmed", "huber"]


# ── mean estimators ──────────────────────────────────────────────────────────


def estimate_mean(
    cube: Float[Array, "H W B"] | np.ndarray,
    *,
    method: _RobustMethod = "mean",
    trim_proportion: float = 0.1,
    huber_c: float = 1.345,
) -> np.ndarray:
    """Estimate the per-band background mean ``μ`` from a scene.

    Parameters
    ----------
    cube
        Hyperspectral scene of shape ``(H, W, n_bands)``.
    method
        - ``'mean'``    — arithmetic mean (sensitive to anomalies).
        - ``'median'``  — per-band median (very robust).
        - ``'trimmed'`` — two-sided trimmed mean (``scipy.stats.trim_mean``).
        - ``'huber'``   — Huber M-estimator (sklearn's ``HuberRegressor`` on
          a constant design).
    trim_proportion
        Fraction discarded from each tail for the trimmed mean. Ignored
        otherwise.
    huber_c
        Huber influence threshold in units of the per-band MAD. Ignored
        unless ``method='huber'``.

    Returns
    -------
    np.ndarray
        Mean spectrum of length ``n_bands``.
    """
    cube_np = np.asarray(cube)
    if cube_np.ndim != 3:
        raise ValueError(
            f"estimate_mean: cube must be (H, W, n_bands), got shape {cube_np.shape}."
        )
    X = cube_np.reshape(-1, cube_np.shape[-1])
    if method == "mean":
        return X.mean(axis=0)
    if method == "median":
        return np.median(X, axis=0)
    if method == "trimmed":
        from scipy.stats import trim_mean

        return trim_mean(X, proportiontocut=trim_proportion, axis=0)
    if method == "huber":
        return _huber_mean(X, c=huber_c)
    raise ValueError(f"estimate_mean: unknown method {method!r}.")


def _huber_mean(
    X: np.ndarray, *, c: float, n_iter: int = 20, tol: float = 1e-8
) -> np.ndarray:
    """Per-band Huber M-estimator via iteratively-reweighted least squares."""
    mu = np.median(X, axis=0)
    mad = np.median(np.abs(X - mu), axis=0) + 1e-12
    scale = 1.4826 * mad
    for _ in range(n_iter):
        z = (X - mu) / scale
        w = np.where(np.abs(z) <= c, 1.0, c / np.abs(z).clip(min=1e-12))
        mu_new = (w * X).sum(axis=0) / w.sum(axis=0).clip(min=1e-12)
        if np.max(np.abs(mu_new - mu)) < tol:
            mu = mu_new
            break
        mu = mu_new
    return mu


# ── dense covariance estimators ──────────────────────────────────────────────


def _dense_operator(cov: np.ndarray) -> LinearOperator:
    """Wrap a dense PSD matrix as a lineax operator with the right tags."""
    return lx.MatrixLinearOperator(
        jnp.asarray(cov),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


def estimate_cov_empirical(
    cube: Float[Array, "H W B"] | np.ndarray,
    mean: np.ndarray | None = None,
    *,
    ridge: float = 0.0,
    mask: np.ndarray | None = None,
) -> LinearOperator:
    """Empirical covariance via :class:`sklearn.covariance.EmpiricalCovariance`.

    Returns a dense :class:`lineax.MatrixLinearOperator`. Optionally adds a
    diagonal ``ridge`` for numerical PD-ness when ``n_samples ≲ n_bands``.
    ``mask`` (boolean exclude-pixels) drops flagged pixels — e.g. a detected
    plume — so ``Σ`` is estimated from background only, consistent with a robust
    masked ``μ``.
    """
    from sklearn.covariance import EmpiricalCovariance

    X = _flatten_cube_masked(cube, mask)
    est = EmpiricalCovariance(store_precision=False).fit(X)
    cov = est.covariance_
    if mean is not None:
        # sklearn uses the empirical mean internally; re-centre if the caller
        # supplied a robust mean so (μ, Σ) are consistent.
        Xc = X - mean
        cov = (Xc.T @ Xc) / X.shape[0]
    if ridge > 0.0:
        cov = cov + ridge * np.eye(cov.shape[0])
    return _dense_operator(cov)


def estimate_cov_shrunk(
    cube: Float[Array, "H W B"] | np.ndarray,
    mean: np.ndarray | None = None,
    *,
    method: Literal["ledoit_wolf", "oas"] = "ledoit_wolf",
    mask: np.ndarray | None = None,
) -> LinearOperator:
    """Shrinkage covariance estimator.

    Uses Ledoit–Wolf (``method='ledoit_wolf'``, default) or OAS
    (``'oas'``) — both shrink the sample covariance toward a scaled identity
    and are PD by construction even when ``n_samples < n_bands``. ``mask``
    (boolean exclude-pixels) drops flagged pixels before fitting.
    """
    X = _flatten_cube_masked(cube, mask)
    if mean is not None:
        X = X - mean
        assume_centered = True
    else:
        assume_centered = False
    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        est = LedoitWolf(assume_centered=assume_centered).fit(X)
    elif method == "oas":
        from sklearn.covariance import OAS

        est = OAS(assume_centered=assume_centered).fit(X)
    else:
        raise ValueError(
            f"estimate_cov_shrunk: method must be 'ledoit_wolf' or 'oas'; got {method!r}."
        )
    return _dense_operator(est.covariance_)


# ── low-rank covariance (Woodbury-friendly) ──────────────────────────────────


def estimate_cov_lowrank(
    cube: Float[Array, "H W B"] | np.ndarray,
    mean: np.ndarray | None = None,
    *,
    rank: int,
    tikhonov: float,
    random_state: int | jax.Array | None = 0,
    n_oversamples: int = 10,
    mask: np.ndarray | None = None,
    rank_rtol: float = 1e-10,
) -> LinearOperator:
    """Low-rank + Tikhonov covariance: ``Σ = λI + V D Vᵀ``.

    Uses :func:`gaussx.randomized_svd` (Halko–Martinsson–Tropp, JAX, 5 power
    iterations) to find the top ``rank`` spectral directions of
    the sample covariance. Returned as a :class:`gaussx.LowRankUpdate` so that
    :func:`gaussx.solve` routes through the Woodbury identity — the MF
    precompute cost is ``O(n_bands · rank + rank³)`` instead of
    ``O(n_bands³)``.

    Parameters
    ----------
    cube
        Hyperspectral scene ``(H, W, n_bands)``.
    mean
        Background mean. If ``None``, subtracts the sample mean.
    rank
        Number of leading components to keep. Clamped to
        ``min(rank, n_samples - 1, n_bands - 1)``.
    tikhonov
        Diagonal floor ``λ > 0`` — ensures strict PD.
    random_state
        Seed for the randomised SVD: an ``int`` (converted with
        ``jax.random.key``), a JAX PRNG key, or ``None`` (seed 0).
    n_oversamples
        Extra random directions for the Halko sampling — higher is slower but
        better-conditioned.
    mask
        Boolean exclude-pixels mask; flagged pixels (e.g. a detected plume) are
        dropped before the SVD so ``Σ`` reflects background only.
    rank_rtol
        Components whose covariance eigenvalue ``d_i`` is below
        ``rank_rtol · max(d)`` are dropped before building the operator. On a
        rank-deficient scene (constant regions, duplicated spectra) those
        ``d_i ≈ 0`` would make gaussx's Woodbury capacitance ``diag(1/d) +
        VᵀL⁻¹U`` blow up to inf/NaN; dropping them keeps every score finite.
    """
    if tikhonov <= 0.0:
        raise ValueError("estimate_cov_lowrank: tikhonov must be > 0.")
    X = _flatten_cube_masked(cube, mask)
    mu = X.mean(axis=0) if mean is None else np.asarray(mean)
    Xc = X - mu
    n_samples, n_bands = Xc.shape
    rank = max(1, min(int(rank), n_samples - 1, n_bands - 1))
    key = _as_key(random_state)
    # Same MLE normalization s_i² / n_samples as the empirical / LedoitWolf /
    # OAS paths in this module (using n_samples - 1 here would make
    # matched_filter_snr and detection_threshold depend on the estimator).
    V, d = lowrank_factors(
        jnp.asarray(Xc), rank, n_oversamples=n_oversamples, n_power_iter=5, key=key
    )
    V, d = np.asarray(V), np.asarray(d)
    diag = jnp.asarray(tikhonov * np.ones(n_bands, dtype=float))
    # Rank-deficiency guard: drop near-zero eigen-directions so the Woodbury
    # capacitance stays finite. If none survive (e.g. a constant scene), Σ is
    # just the strictly-PD Tikhonov floor λI.
    d_max = float(d.max()) if d.size else 0.0
    keep = d > rank_rtol * d_max if d_max > 0.0 else np.zeros(d.shape, dtype=bool)
    V, d = V[:, keep], d[keep]
    if d.size == 0:
        return lx.DiagonalLinearOperator(diag)
    # λ I + U diag(d) Uᵀ via the gaussx SVD constructor (auto-inferred
    # symmetric/PSD tags, orthonormal-factor solve path).
    U = jnp.asarray(V)  # (n_bands, n_kept) — spectral directions as columns
    return gx.svd_low_rank_plus_diag(diag, U, jnp.asarray(d), U)


# ── helpers ──────────────────────────────────────────────────────────────────


def _as_key(random_state: int | jax.Array | None) -> jax.Array:
    if random_state is None:
        random_state = 0
    if isinstance(random_state, (int, np.integer)):
        return jax.random.key(int(random_state))
    return random_state


def _lowrank_factors(
    Xc: Float[Array, "N B"],
    rank: int,
    *,
    n_oversamples: int = 10,
    n_power_iter: int = 5,
    key: jax.Array,
) -> tuple[Float[Array, "B r"], Float[Array, " r"]]:
    """Jittable randomised-SVD core: ``(V, d)`` of centred samples ``Xc``.

    ``V`` is ``(n_bands, rank)`` (right singular vectors as columns) and
    ``d = s² / n_samples`` the MLE covariance eigenvalues.
    """
    _, sv, Vt = gx.randomized_svd(
        lx.MatrixLinearOperator(Xc),
        rank,
        oversample=n_oversamples,
        n_power_iter=n_power_iter,
        key=key,
    )
    return Vt.T, sv**2 / Xc.shape[0]


lowrank_factors = jax.jit(
    _lowrank_factors, static_argnames=("rank", "n_oversamples", "n_power_iter")
)


def _flatten_cube(cube: Float[Array, "H W B"] | np.ndarray) -> np.ndarray:
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(
            f"background estimator: cube must be (H, W, n_bands), got shape {arr.shape}."
        )
    return arr.reshape(-1, arr.shape[-1])


def _flatten_cube_masked(
    cube: Float[Array, "H W B"] | np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Flatten a cube to ``(n_kept, n_bands)``, dropping masked-out pixels.

    ``mask`` is a boolean **exclude** mask — ``True`` marks pixels (e.g.
    detected plume / target) to leave out of the background statistics so the
    filter is not attenuated by its own signal. Any shape with one entry per
    pixel is accepted (``(H, W)`` or flat).
    """
    X = _flatten_cube(cube)
    if mask is None:
        return X
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.shape[0] != X.shape[0]:
        raise ValueError(
            f"background estimator: `mask` must have one entry per pixel "
            f"({X.shape[0]}), got {m.size}."
        )
    kept = X[~m]
    if kept.shape[0] < 2:
        raise ValueError(
            "background estimator: `mask` excludes all but "
            f"{kept.shape[0]} pixel(s); need ≥ 2 for a covariance estimate."
        )
    return kept
