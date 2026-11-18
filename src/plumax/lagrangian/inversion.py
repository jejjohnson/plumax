"""Tier II — model-based inversion of the source vector from observations.

Given a source–receptor sensitivity (footprint) matrix ``F`` — the linear map
from a discretised emission field ``q`` to predicted observations
``y_pred = F q`` (see :func:`plumax.lagrangian.compute_footprint`) — this module
recovers a Bayesian posterior over ``q`` from observed column enhancements.

Units follow ``F``'s flux-sensitivity convention (s·m²·kg⁻¹, see
:func:`plumax.lagrangian.compute_footprint`): ``q`` is a per-cell surface
**flux** in kg·m⁻²·s⁻¹, and the prior mean / covariance are expressed in those
same flux units. The estimators are unit-agnostic — they only require ``q`` and
``F`` to be mutually consistent so that ``F q`` lands in the observation units.

Two estimators, both following the Tier II design note
([design](https://github.com/jejjohnson/plumax/blob/main/docs/design/02_tier2_lagrangian.md#tier2-inference)):

- :func:`linear_gaussian_inversion` — the closed-form Gaussian–Gaussian
  (BLUE / Kalman) update for a Gaussian prior ``q ~ N(q_a, B)`` and Gaussian
  likelihood ``y = F q + ε``, ``ε ~ N(0, R)``::

      q* = q_a + B Fᵀ (F B Fᵀ + R)⁻¹ (y − F q_a)
      P* = B − B Fᵀ (F B Fᵀ + R)⁻¹ F B

- :func:`lognormal_inversion` — the sign-constrained ``log q ~ N(log q_a,
  B_log)`` variant, linearised about ``log q_a`` (the design's recommended v1
  prior). With the tangent-linear Jacobian ``F̃ = F · diag(q_a)`` the log-space
  increment is the same BLUE update and ``q* = q_a · exp(δ*) ≥ 0``.

The spatial prior is built with :func:`matern32_covariance` — a Matérn-3/2
kernel with a real correlation length, as the design note insists (a diagonal
``B`` produces single-cell-spike posteriors). The likelihood covariance is
assembled by :func:`observation_covariance`, which adds the **representation
error** to the retrieval error rather than using the retrieval error alone.

Both estimators are vardax analyses (the ``assim`` extra,
``pip install 'plumax[assim]'``): :func:`footprint_operator` wraps
``q ↦ F q`` as a ``vardax.CallableObs``, ``vardax.OptimalInterpolation`` with a
Cholesky solve of the ``(n_obs × n_obs)`` innovation system gives the mean (the
efficient ordering when ``n_obs ≪ n_grid``, the usual overpass regime), and
``vardax.LaplaceCovariance`` the covariance. They are pure JAX, so they compose
under ``jit`` / ``vmap`` and are differentiable through the posterior mean
(e.g. for hyperparameter tuning of ``ℓ`` or ``σ``).
"""

from __future__ import annotations

from plumax.lagrangian._src.inversion import (
    GaussianPosterior,
    LognormalPosterior,
    footprint_operator,
    linear_gaussian_inversion,
    lognormal_inversion,
    matern32_covariance,
    observation_covariance,
)


__all__ = [
    "GaussianPosterior",
    "LognormalPosterior",
    "footprint_operator",
    "linear_gaussian_inversion",
    "lognormal_inversion",
    "matern32_covariance",
    "observation_covariance",
]
