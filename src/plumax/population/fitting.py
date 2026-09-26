"""Tier V.B — POD-modified power-law fitting via NumPyro NUTS.

A Bayesian fit of the observed methane plume flux distribution to the
model::

    p_power(x) = C₁ · x⁻ᵅ,          x_min ≤ x ≤ x_max
    q(x)       = Φ_LN(x; x₀, σ)     (lognormal CDF = POD)
    p_obs(x)   = C₂ · x⁻ᵅ · q(x)    (detected distribution)

with parameters ``(α, x₀, σ)`` inferred by NUTS. Data loading, plotting and
per-satellite CSV plumbing stay in the notebooks; this module exposes only
the pure-computation pieces:

* :func:`lognorm_cdf` — POD curve evaluation on a grid.
* :func:`power_law` — un-normalised power-law PDF.
* :func:`pod_powerlaw_model` — the NumPyro generative model.
* :func:`run_mcmc` — NUTS runner returning a posterior DataFrame.

This module imports NumPyro at module level; :mod:`plumax.population`
binds its names lazily so importing the package stays cheap. pandas is
imported only inside :func:`run_mcmc`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import scipy.special
from jax.scipy.special import erfc
from jax.scipy.stats import norm as jax_norm
from numpy.typing import NDArray
from numpyro.infer import MCMC, NUTS


if TYPE_CHECKING:  # pandas is imported lazily inside run_mcmc
    import pandas as pd


__all__ = [
    "X_MAX_DEFAULT",
    "X_MIN_DEFAULT",
    "lognorm_cdf",
    "pod_powerlaw_model",
    "power_law",
    "run_mcmc",
]


# ── Constants ────────────────────────────────────────────────────────────────

X_MIN_DEFAULT = 10.0  # kg/hr — lower bound of power-law domain
X_MAX_DEFAULT = 1e6  # kg/hr — upper bound of power-law domain


# ── Utility functions (NumPy) ────────────────────────────────────────────────


def lognorm_cdf(x: NDArray, x50: float, s: float) -> NDArray:
    """Lognormal CDF used as the probability-of-detection curve.

    ``POD(x) = 1 − ½ · erfc((ln x − ln x₅₀) / (√2 · σ))``.

    Args:
        x: Emission rate [kg/hr]. Must be strictly positive. Shape ``(N,)``.
        x50: Median detection rate [kg/hr]. Must be strictly positive.
        s: Lognormal width [dimensionless]. Must be strictly positive.

    Returns:
        Probability of detection ∈ [0, 1]. Shape ``(N,)``.

    Raises:
        ValueError: If any of ``x``, ``x50``, ``s`` are non-positive.
    """
    x_arr = np.asarray(x)
    if np.any(x_arr <= 0.0):
        raise ValueError("lognorm_cdf: all entries of `x` must be > 0")
    if not (x50 > 0.0):
        raise ValueError(f"lognorm_cdf: `x50` must be > 0 (got {x50!r})")
    if not (s > 0.0):
        raise ValueError(f"lognorm_cdf: `s` must be > 0 (got {s!r})")
    return 1.0 - 0.5 * scipy.special.erfc(
        (np.log(x_arr) - np.log(x50)) / (np.sqrt(2.0) * s)
    )


def power_law(x: NDArray, alpha: float) -> NDArray:
    """Un-normalised power-law PDF: x⁻ᵅ."""
    return np.power(x, -alpha)


# ── NumPyro model ────────────────────────────────────────────────────────────


def pod_powerlaw_model(
    x_obs: jnp.ndarray,
    x_min: float,
    x_max: float,
    num_integration_points: int = 5000,
) -> None:
    """NumPyro model for a POD-modified power law on observed fluxes.

    Sampled parameters:
        * ``alpha`` — power-law exponent, ``Uniform(1.1, 4.5)`` [dimensionless].
        * ``x0`` — lognormal CDF median (POD 50 %), ``Uniform(1, 20000)`` [kg/hr].
        * ``sk`` — lognormal CDF width, ``Uniform(0.1, 1.5)`` [dimensionless].

    Log-likelihood (numerical integration in log-space for ``I(θ)``)::

        ln L = −α · Σᵢ ln xᵢ + Σᵢ ln q(xᵢ) − N · ln I(θ)
        I(θ) = ∫_{x_min}^{x_max} q(x) · x⁻ᵅ dx

    Args:
        x_obs: Observed emission rates [kg/hr]. Shape ``(N,)``.
        x_min: Lower integration bound [kg/hr].
        x_max: Upper integration bound [kg/hr].
        num_integration_points: Trapezoidal grid size for I(θ). Default 5000.
    """
    N = len(x_obs)

    alpha = numpyro.sample("alpha", dist.Uniform(1.1, 4.5))
    x0 = numpyro.sample("x0", dist.Uniform(1.0, 20000.0))
    sk = numpyro.sample("sk", dist.Uniform(0.1, 1.5))

    sum_log_x = jnp.sum(jnp.log(x_obs))
    # q(x) = Φ((ln x − ln x0)/sk) as a Normal CDF; log-evaluate via
    # `norm.logcdf` so the far-left tail stays finite instead of underflowing
    # to −inf for parameter draws with large x0 / small sk (which would
    # otherwise break NUTS with NaN gradients).
    u = (jnp.log(x_obs) - jnp.log(x0)) / sk
    sum_log_q = jnp.sum(jax_norm.logcdf(u))

    xi_values = jnp.linspace(jnp.log(x_min), jnp.log(x_max), num_integration_points)
    pod_on_grid = 1.0 - 0.5 * erfc((xi_values - jnp.log(x0)) / (sk * jnp.sqrt(2.0)))
    integrand = pod_on_grid * jnp.exp(-alpha * xi_values + xi_values)
    I_theta = jnp.trapezoid(integrand, xi_values)
    I_theta = jnp.clip(I_theta, min=1e-300)

    logL = -alpha * sum_log_x + sum_log_q - N * jnp.log(I_theta)
    numpyro.factor("log_likelihood", logL)


# ── MCMC runner ──────────────────────────────────────────────────────────────


def run_mcmc(
    data_values: NDArray,
    *,
    x_min: float = X_MIN_DEFAULT,
    x_max: float = X_MAX_DEFAULT,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
    seed: int = 20,
    print_summary: bool = False,
) -> pd.DataFrame:
    """Run NumPyro NUTS on the POD-modified power-law model.

    Args:
        data_values: Observed Q values [kg/hr]. Must be strictly positive and within the
            [x_min, x_max] domain of the power law. Shape ``(N,)``.
        x_min: Lower power-law support bound [kg/hr].
        x_max: Upper power-law support bound [kg/hr].
        num_warmup: NUTS warmup iterations per chain. Defaults are tuned for a
            notebook-grade fit (~seconds on CPU); bump to 2000-4000 for a
            publication-grade run.
        num_samples: Posterior samples per chain after warmup.
        num_chains: Number of independent MCMC chains.
        seed: PRNG seed.
        print_summary: If True, call ``mcmc.print_summary()`` after the run. Default False
            to keep notebook output tidy.

    Returns:
        A pandas ``DataFrame`` of shape ``(num_samples * num_chains, 3)``
        with columns ``x0`` [kg/hr], ``sk`` [dimensionless] and ``alpha``
        [dimensionless].

    Raises:
        ValueError: If ``data_values`` is empty, contains non-positive values, lies outside
            ``[x_min, x_max]``, or if the support bounds are invalid.
    """
    import pandas as pd  # lazy — only `run_mcmc` requires pandas

    data_values = np.asarray(data_values)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or not (x_min < x_max):
        raise ValueError(
            f"run_mcmc: expected finite bounds with x_min < x_max, got "
            f"x_min={x_min!r}, x_max={x_max!r}"
        )
    if data_values.size == 0:
        raise ValueError(
            "run_mcmc: `data_values` must contain at least one observation"
        )
    if np.any(~np.isfinite(data_values)):
        raise ValueError("run_mcmc: `data_values` must contain only finite values")
    if np.any(data_values <= 0.0):
        raise ValueError("run_mcmc: `data_values` must be strictly positive")
    if np.any((data_values < x_min) | (data_values > x_max)):
        raise ValueError(
            "run_mcmc: all `data_values` must lie within the closed support "
            f"[x_min, x_max] = [{x_min}, {x_max}]"
        )

    kernel = NUTS(pod_powerlaw_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, x_obs=data_values, x_min=x_min, x_max=x_max)
    if print_summary:
        mcmc.print_summary()

    samples = mcmc.get_samples()
    return pd.DataFrame(
        {
            "x0": np.asarray(samples["x0"]),
            "sk": np.asarray(samples["sk"]),
            "alpha": np.asarray(samples["alpha"]),
        }
    )
