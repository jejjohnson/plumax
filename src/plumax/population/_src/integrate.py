"""Deprecated aliases: log-space integration now lives in integrix (plumax#253).

`log_gauss_hermite`, `log_simpson` and `pareto_smoothed_log_weights` were plumax's
own log-space rules for the Tier V mark likelihood. integrix 0.0.5 ships them
(``integrix.expect``), written in JAX, so these names forward there with a
`DeprecationWarning` for one release and then go.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import integrix.expect as ix_expect
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float


def _deprecated(name: str) -> None:
    warnings.warn(
        f"plumax.population.integrate.{name} is deprecated; use "
        f"integrix.expect.{name} (plumax#253).",
        DeprecationWarning,
        stacklevel=3,
    )


def log_gauss_hermite(
    log_f: Callable[[Array], Array],
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    n: int = 32,
) -> Float[Array, ""]:
    """Deprecated alias of `integrix.expect.log_gauss_hermite`.

    log ∫ f(q) LN(q; μ, σ²) dq by Gauss–Hermite quadrature in log space.

    Args:
        log_f: log f, on an array of rates.
        mu: μ, the mean of log Q.
        sigma: σ > 0, the standard deviation of log Q.
        n: Number of Hermite nodes (static).

    Returns:
        The log-integral.

    Examples:
        >>> import jax.numpy as jnp
        >>> # E[Q] = exp(μ + σ²/2) for a lognormal Q
        >>> value = log_gauss_hermite(jnp.log, 0.2, 0.5)
        >>> bool(jnp.allclose(value, 0.2 + 0.5 * 0.5**2))
        True
    """
    _deprecated("log_gauss_hermite")
    return ix_expect.log_gauss_hermite(log_f, mu, sigma, n=n)


def log_simpson(
    log_f: Callable[[Array], Array],
    q_min: float,
    q_max: float,
    *,
    n: int = 2001,
) -> Float[Array, ""]:
    """Deprecated alias of `integrix.expect.log_simpson`.

    log ∫ f(q) dq over [q_min, q_max] by composite Simpson in u = log q.

    Args:
        log_f: log f, on an array of rates.
        q_min: Lower limit, > 0.
        q_max: Upper limit.
        n: Number of nodes (static; made odd).

    Returns:
        The log-integral.

    Examples:
        >>> import jax.numpy as jnp
        >>> # ∫₁^∞ 3 q⁻⁴ dq = 1, truncated at 1e6
        >>> value = log_simpson(lambda q: jnp.log(3.0) - 4.0 * jnp.log(q), 1.0, 1e6)
        >>> bool(jnp.abs(value) < 1e-8)
        True
    """
    _deprecated("log_simpson")
    return ix_expect.log_simpson(log_f, q_min, q_max, n=n)


def pareto_smoothed_log_weights(log_weights: ArrayLike) -> tuple[np.ndarray, float]:
    """Deprecated alias of `integrix.expect.pareto_smoothed_log_weights`.

    Pareto-smoothed importance log-weights and the tail shape k̂, for one set of
    weights; integrix's version also batches over leading axes and runs under
    `jax.jit`. k̂ > 0.7 flags weights too heavy-tailed to trust.

    Args:
        log_weights: Unnormalised log importance weights.

    Returns:
        ``(smoothed, k̂)``: normalised smoothed log-weights (log-sum-exp 0) and
        the fitted shape (NaN when there is no tail to fit).

    Examples:
        >>> import numpy as np
        >>> log_weights = np.random.default_rng(0).normal(0.0, 0.3, 1000)
        >>> smoothed, k_hat = pareto_smoothed_log_weights(log_weights)
        >>> # float32 unless the caller enables x64, hence the tolerance
        >>> bool(np.isclose(np.logaddexp.reduce(smoothed), 0.0, atol=1e-5)), k_hat < 0.7
        (True, True)

    References:
        Vehtari, Simpson, Gelman, Yao & Gabry, Pareto smoothed importance
        sampling, arXiv:1507.02646.
    """
    _deprecated("pareto_smoothed_log_weights")
    flat = jnp.ravel(jnp.asarray(log_weights, dtype=float))
    smoothed, k_hat = ix_expect.pareto_smoothed_log_weights(flat)
    return np.asarray(smoothed), float(k_hat)
