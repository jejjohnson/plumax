"""Scalar event-emission laws for the Tier V per-event payload.

Every tier reduces its posterior over a source field or a control vector to the
event emission rate Q, and recalls the prior that produced it. These helpers
hold the two law families that reduction lands on, with the sampling and the
log-density the population layer needs:

* :class:`GaussianEmission`, Q ~ N(μ, σ²), the law of a linear functional
  Q = wᵀx of a Gaussian vector x ~ N(m, C) (:func:`gaussian_total`).
* :class:`LognormalEmission`, log Q ~ N(μ, σ²). A lognormal Q is exact for a
  single lognormal entry; for a total Q = Σᵢ sᵢ exp(zᵢ) of correlated
  lognormals, :func:`lognormal_total` matches its first two moments.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


_LOG_2PI = float(jnp.log(2.0 * jnp.pi))


class GaussianEmission(eqx.Module):
    """Q ~ N(μ, σ²).

    Args:
        mean: μ [kg/s].
        std: σ > 0 [kg/s].
    """

    mean: Float[Array, ""]
    std: Float[Array, ""]

    def __init__(self, mean: ArrayLike, std: ArrayLike) -> None:
        self.mean = jnp.asarray(mean, dtype=float)
        self.std = jnp.asarray(std, dtype=float)

    def sample(self, key: Array, n: int) -> Float[Array, " member"]:
        """n draws of Q."""
        # ε ~ N(0, 1)
        noise = jax.random.normal(key, (n,), dtype=self.mean.dtype)
        # Q = μ + σ ε
        return self.mean + self.std * noise

    def log_prob(self, q: ArrayLike) -> Float[Array, "*q"]:
        """log N(q; μ, σ²)."""
        # z = (q − μ) / σ
        standardised = (jnp.asarray(q) - self.mean) / self.std
        # log N = −½ z² − log σ − ½ log 2π
        return -0.5 * standardised**2 - jnp.log(self.std) - 0.5 * _LOG_2PI


class LognormalEmission(eqx.Module):
    """log Q ~ N(μ, σ²).

    Args:
        loc: μ, the mean of log Q.
        scale: σ > 0, the standard deviation of log Q.
    """

    loc: Float[Array, ""]
    scale: Float[Array, ""]

    def __init__(self, loc: ArrayLike, scale: ArrayLike) -> None:
        self.loc = jnp.asarray(loc, dtype=float)
        self.scale = jnp.asarray(scale, dtype=float)

    @classmethod
    def from_moments(cls, mean: ArrayLike, std: ArrayLike) -> LognormalEmission:
        """The lognormal with E[Q] = m and sd[Q] = s.

        σ² = log(1 + s²/m²),  μ = log m − σ²/2.
        """
        mean = jnp.asarray(mean, dtype=float)
        std = jnp.asarray(std, dtype=float)
        # σ² = log(1 + (s/m)²)
        log_variance = jnp.log1p((std / mean) ** 2)
        # μ = log m − σ²/2
        loc = jnp.log(mean) - 0.5 * log_variance
        return cls(loc, jnp.sqrt(log_variance))

    def sample(self, key: Array, n: int) -> Float[Array, " member"]:
        """n draws of Q."""
        noise = jax.random.normal(key, (n,), dtype=self.loc.dtype)
        # Q = exp(μ + σ ε)
        return jnp.exp(self.loc + self.scale * noise)

    def log_prob(self, q: ArrayLike) -> Float[Array, "*q"]:
        """log LN(q; μ, σ²), −∞ for q ≤ 0."""
        q = jnp.asarray(q, dtype=self.loc.dtype)
        positive = q > 0.0
        # guard the log; the value is masked to −∞ below
        log_q = jnp.log(jnp.where(positive, q, 1.0))
        standardised = (log_q - self.loc) / self.scale
        # log LN = −½ z² − log σ − ½ log 2π − log q
        density = -0.5 * standardised**2 - jnp.log(self.scale) - 0.5 * _LOG_2PI - log_q
        return jnp.where(positive, density, -jnp.inf)


class EmpiricalEmission(eqx.Module):
    """The empirical law of Q from posterior draws (MCMC samples).

    Args:
        draws: The draws Q⁽ᵉ⁾ [kg/s].
    """

    draws: Float[Array, " member"]

    def __init__(self, draws: ArrayLike) -> None:
        self.draws = jnp.ravel(jnp.asarray(draws, dtype=float))

    def sample(self, key: Array, n: int) -> Float[Array, " member"]:
        """n draws, resampled with replacement."""
        index = jax.random.randint(key, (n,), 0, self.draws.shape[0])
        return self.draws[index]


def gaussian_total(
    mean: Float[Array, " state"],
    covariance: Float[Array, "state state"],
    weights: Float[Array, " state"],
) -> GaussianEmission:
    """The law of Q = wᵀx for x ~ N(m, C): Q ~ N(wᵀm, wᵀCw), exactly.

    Args:
        mean: m.
        covariance: C.
        weights: w.

    Returns:
        The :class:`GaussianEmission` of Q.
    """
    # wᵀ m
    total_mean = jnp.dot(weights, mean)
    # wᵀ C w, clipped against round-off
    total_variance = jnp.dot(weights, jnp.dot(covariance, weights))
    return GaussianEmission(total_mean, jnp.sqrt(jnp.clip(total_variance, 0.0)))


def lognormal_total(
    log_mean: Float[Array, " state"],
    log_covariance: Float[Array, "state state"],
    scale: Float[Array, " state"],
) -> LognormalEmission:
    """Moment-matched lognormal law of Q = Σᵢ sᵢ exp(zᵢ), z ~ N(m, C), sᵢ > 0.

    E[Q] = Σᵢ sᵢ exp(mᵢ + Cᵢᵢ/2) and
    E[Q²] = Σᵢⱼ sᵢ sⱼ exp(mᵢ + mⱼ + (Cᵢᵢ + Cⱼⱼ)/2 + Cᵢⱼ) are exact; the
    lognormal with these two moments is exact for one entry and an
    approximation for a sum of several.

    Args:
        log_mean: m.
        log_covariance: C.
        scale: s.

    Returns:
        The :class:`LognormalEmission` of Q.
    """
    # aᵢ = log sᵢ + mᵢ + Cᵢᵢ/2, so E[Q] = Σᵢ exp(aᵢ)
    log_terms = jnp.log(scale) + log_mean + 0.5 * jnp.diagonal(log_covariance)
    log_first = jax.scipy.special.logsumexp(log_terms)
    # log E[Q²] = logsumexp over (i, j) of aᵢ + aⱼ + Cᵢⱼ
    pair_terms = einx.add("row, col -> row col", log_terms, log_terms) + log_covariance
    log_second = jax.scipy.special.logsumexp(pair_terms)
    # σ² = log E[Q²] − 2 log E[Q]
    log_variance = jnp.clip(log_second - 2.0 * log_first, 0.0)
    # μ = log E[Q] − σ²/2
    loc = log_first - 0.5 * log_variance
    return LognormalEmission(loc, jnp.sqrt(log_variance))


def sample_lognormal_total(
    key: Array,
    n: int,
    log_mean: Float[Array, " state"],
    log_covariance: Float[Array, "state state"],
    scale: Float[Array, " state"],
) -> Float[Array, " member"]:
    """Exact draws of Q = Σᵢ sᵢ exp(zᵢ), z ~ N(m, C).

    Args:
        key: PRNG key.
        n: Number of draws.
        log_mean: m.
        log_covariance: C.
        scale: s.

    Returns:
        n draws of Q.
    """
    # z⁽ᵉ⁾ ~ N(m, C)
    log_draws = jax.random.multivariate_normal(
        key, log_mean, log_covariance, (n,), method="eigh"
    )
    # Q⁽ᵉ⁾ = Σᵢ sᵢ exp(zᵢ⁽ᵉ⁾)
    return jnp.dot(jnp.exp(log_draws), scale)
