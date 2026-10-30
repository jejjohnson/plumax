"""Log-space integration for the Tier V mark likelihood.

The mark integrals of the TMTPP likelihood mix a saturating POD with a
heavy-tailed size distribution, and in linear Q their integrands underflow or
are unresolved on any one grid. Everything here works on log-integrands and
returns log-integrals:

* :func:`log_gauss_hermite`, log ∫ f(q) LN(q; μ, σ²) dq by Gauss–Hermite
  quadrature in u = log q, for the Gaussian regime;
* :func:`log_simpson`, log ∫ f(q) dq by composite Simpson in u = log q, for
  the thinned rate p̄ = ∫ P_d(q) f(q | θ) dq over the whole support;
* :func:`pareto_smoothed_log_weights`, Pareto-smoothed importance weights and
  their tail-shape diagnostic k̂.

xtremax has no log-space mark quadrature yet (jejjohnson/xtremax#103 is
open), so it lives here; the generalised Pareto quantile of the smoothing
step is xtremax's.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Float

from plumax.population._src._xtremax import require_xtremax


def log_gauss_hermite(
    log_f: Callable[[Array], Array],
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    n: int = 32,
) -> Float[Array, ""]:
    """log ∫ f(q) LN(q; μ, σ²) dq by Gauss–Hermite quadrature in log space.

    With q = exp(μ + √2 σ x), the integral is π^(−½) Σₖ wₖ f(qₖ) for the
    Hermite nodes xₖ and weights wₖ, evaluated as a log-sum-exp so that
    f(qₖ) never leaves log space.

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
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    mu = jnp.asarray(mu, dtype=float)
    sigma = jnp.asarray(sigma, dtype=float)
    # qₖ = exp(μ + √2 σ xₖ)
    rates = jnp.exp(mu + math.sqrt(2.0) * sigma * jnp.asarray(nodes))
    # log(wₖ / √π) + log f(qₖ)
    terms = jnp.log(jnp.asarray(weights)) - 0.5 * math.log(math.pi) + log_f(rates)
    return logsumexp(terms)


def log_simpson(
    log_f: Callable[[Array], Array],
    q_min: float,
    q_max: float,
    *,
    n: int = 2001,
) -> Float[Array, ""]:
    """log ∫ f(q) dq over [q_min, q_max] by composite Simpson in u = log q.

    ∫ f(q) dq = ∫ f(eᵘ) eᵘ du, with Simpson weights on a uniform u grid, as a
    log-sum-exp. A uniform grid in log q resolves a lognormal body and a
    power-law tail alike, which a uniform grid in q cannot.

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
    nodes = n if n % 2 == 1 else n + 1
    grid = np.linspace(math.log(q_min), math.log(q_max), nodes)
    step = grid[1] - grid[0]
    simpson = np.ones(nodes)
    simpson[1:-1:2] = 4.0
    simpson[2:-1:2] = 2.0
    log_weights = jnp.asarray(np.log(simpson * step / 3.0))
    log_grid = jnp.asarray(grid)
    # log[wₖ f(e^uₖ) e^uₖ]
    terms = log_weights + log_f(jnp.exp(log_grid)) + log_grid
    return logsumexp(terms)


def _generalized_pareto_fit(exceedances: np.ndarray) -> tuple[float, float]:
    # Zhang & Stephens (2009) posterior-mean estimator, with the weakly
    # informative prior on the shape that PSIS uses (Vehtari et al.)
    x = np.sort(exceedances)
    n = x.size
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(math.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b = b / (prior_bs * x[int(n / 4 + 0.5) - 1]) + 1.0 / x[-1]
    k = np.mean(np.log1p(-b[:, None] * x[None, :]), axis=1)
    profile = n * (np.log(-b / k) - k - 1.0)
    weights = 1.0 / np.sum(np.exp(profile[None, :] - profile[:, None]), axis=1)
    weights = weights / weights.sum()
    b_mean = float(np.sum(b * weights))
    shape = float(np.mean(np.log1p(-b_mean * x)))
    scale = -shape / b_mean
    # shrink the shape towards ½
    shape = (n * shape + prior_k * 0.5) / (n + prior_k)
    return shape, scale


def pareto_smoothed_log_weights(log_weights: ArrayLike) -> tuple[np.ndarray, float]:
    """Pareto-smoothed importance log-weights and the tail shape k̂.

    The largest M = ⌈min(S/5, 3√S)⌉ weights are replaced by the expected
    order statistics of a generalised Pareto fitted to their exceedances, then
    the weights are normalised. k̂ > 0.7 flags weights too heavy-tailed for
    the importance estimate to be trusted. A host-side diagnostic (NumPy, not
    traced): the fit sorts the weights.

    Args:
        log_weights: Unnormalised log importance weights.

    Returns:
        ``(smoothed, k̂)``: normalised smoothed log-weights (log-sum-exp 0) and
        the fitted shape (NaN when fewer than five weights exceed the cutoff,
        e.g. equal weights, which need no smoothing).

    Raises:
        ImportError: Without the ``population`` extra.

    Examples:
        >>> import numpy as np
        >>> log_weights = np.random.default_rng(0).normal(0.0, 0.3, 1000)
        >>> smoothed, k_hat = pareto_smoothed_log_weights(log_weights)
        >>> bool(np.isclose(np.logaddexp.reduce(smoothed), 0.0)), k_hat < 0.7
        (True, True)

    References:
        Vehtari, Simpson, Gelman, Yao & Gabry, Pareto smoothed importance
        sampling, arXiv:1507.02646. Zhang & Stephens (2009), Technometrics.
    """
    require_xtremax("pareto_smoothed_log_weights")
    import xtremax.primitives as extremes

    log_w = np.asarray(log_weights, dtype=float).ravel()
    log_w = log_w - log_w.max()
    size = log_w.size
    tail_length = math.ceil(min(0.2 * size, 3.0 * math.sqrt(size)))
    if tail_length >= size:
        # too few weights to separate a tail: normalise, no diagnostic
        return log_w - np.logaddexp.reduce(log_w), math.nan
    order = np.argsort(log_w)
    cutoff = max(log_w[order[-tail_length - 1]], math.log(np.finfo(float).tiny))
    tail = order[log_w[order] > cutoff]
    k_hat = math.nan
    if tail.size > 4:
        exceedances = np.exp(log_w[tail]) - math.exp(cutoff)
        k_hat, scale = _generalized_pareto_fit(exceedances)
        if math.isfinite(k_hat):
            # expected order statistics, F⁻¹((j − ½)/M)
            levels = (np.arange(tail.size) + 0.5) / tail.size
            quantiles = np.asarray(extremes.gpd_icdf(levels, scale, k_hat))
            # the largest weight stays the largest raw weight, at most 1
            log_w[tail] = np.minimum(np.log(quantiles + math.exp(cutoff)), 0.0)
    smoothed = log_w - np.logaddexp.reduce(log_w)
    return smoothed, float(k_hat)
