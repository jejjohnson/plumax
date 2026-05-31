"""Tier V.B — spatio-temporal point process (event rate).

The tractable parametric core of the TMTPP event-rate model:

* :func:`fit_poisson_rate` — homogeneous Poisson rate ``lambda`` (events
  per unit area per unit time) via the closed-form Gamma-Poisson conjugate
  posterior.  Pure NumPy / SciPy, no NumPyro.
* :func:`fit_inhomogeneous_intensity` — inhomogeneous log-linear intensity
  ``log lambda = beta0 + beta . covariates`` fit with NUTS.  NumPyro is
  imported lazily.

A full log-Gaussian Cox process (LGCP) is out of scope for v1; it is the
natural next kernel when clustering is environmentally driven (see
``docs/design/06b_point_process.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from plumax.lagrangian.inversion import _is_traced


if TYPE_CHECKING:
    from plumax.population.catalog import EmissionCatalog

__all__ = [
    "InhomogeneousIntensityPosterior",
    "PoissonRatePosterior",
    "fit_inhomogeneous_intensity",
    "fit_poisson_rate",
]


@dataclass(frozen=True)
class PoissonRatePosterior:
    """Gamma posterior over a homogeneous Poisson rate ``lambda``.

    The conjugate update for ``n`` events observed over an exposure
    ``area * duration`` with a ``Gamma(alpha0, beta0)`` prior on
    ``lambda`` is ``Gamma(alpha0 + n, beta0 + area * duration)``.

    Attributes:
        alpha: Posterior Gamma shape.
        beta: Posterior Gamma rate (inverse scale); has units of exposure.
        n_events: Number of events observed.
        exposure: The space-time exposure ``area * duration`` used.
    """

    alpha: float
    beta: float
    n_events: int
    exposure: float

    @property
    def rate_mean(self) -> float:
        """Posterior-mean rate ``lambda`` [events per unit area per unit time]."""
        return self.alpha / self.beta

    @property
    def rate_std(self) -> float:
        """Posterior standard deviation of ``lambda``."""
        return float(np.sqrt(self.alpha) / self.beta)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed posterior credible interval for ``lambda``.

        Args:
            level: Central probability mass, e.g. ``0.95``.

        Returns:
            The ``(low, high)`` interval bounds.
        """
        tail = (1.0 - level) / 2.0
        lo, hi = stats.gamma.ppf(
            [tail, 1.0 - tail], a=self.alpha, scale=1.0 / self.beta
        )
        return float(lo), float(hi)


@dataclass(frozen=True)
class InhomogeneousIntensityPosterior:
    """Posterior over a log-linear inhomogeneous intensity model.

    The model is ``log lambda_i = beta0 + covariates_i . beta`` evaluated at
    the detection covariates, treated as an inhomogeneous-Poisson
    log-intensity.

    Attributes:
        beta0_samples: Posterior samples of the intercept ``(n_samples,)``.
        beta_samples: Posterior samples of the coefficients
            ``(n_samples, n_covariates)``.
    """

    beta0_samples: np.ndarray
    beta_samples: np.ndarray

    @property
    def beta0_mean(self) -> float:
        """Posterior-mean intercept."""
        return float(np.mean(self.beta0_samples))

    @property
    def beta_mean(self) -> np.ndarray:
        """Posterior-mean coefficients ``(n_covariates,)``."""
        return np.mean(self.beta_samples, axis=0)


def fit_poisson_rate(
    catalog: EmissionCatalog,
    *,
    area: float,
    duration: float,
    prior_shape: float = 1e-3,
    prior_rate: float = 1e-3,
) -> PoissonRatePosterior:
    """Closed-form Gamma-Poisson posterior for a homogeneous event rate.

    Counts the events in ``catalog`` and forms the conjugate
    ``Gamma(prior_shape + n, prior_rate + area * duration)`` posterior on
    the rate ``lambda`` (events per unit area per unit time).  No NumPyro
    required.

    Args:
        catalog: The detected-event catalog.
        area: Spatial exposure (area observed) [m^2 or chosen unit].
        duration: Temporal exposure (window length) [s or chosen unit].
        prior_shape: Gamma prior shape ``alpha0`` (default near-flat).
        prior_rate: Gamma prior rate ``beta0`` (default near-flat).

    Returns:
        The :class:`PoissonRatePosterior`.

    Raises:
        ValueError: If ``area`` or ``duration`` is non-positive (skipped
            for traced values).
    """
    if not _is_traced(area) and float(area) <= 0.0:
        msg = f"area must be positive, got {area!r}"
        raise ValueError(msg)
    if not _is_traced(duration) and float(duration) <= 0.0:
        msg = f"duration must be positive, got {duration!r}"
        raise ValueError(msg)

    n = catalog.n_events
    exposure = float(area) * float(duration)
    return PoissonRatePosterior(
        alpha=float(prior_shape) + n,
        beta=float(prior_rate) + exposure,
        n_events=n,
        exposure=exposure,
    )


def fit_inhomogeneous_intensity(
    covariates: np.ndarray,
    *,
    quadrature_covariates: np.ndarray,
    quadrature_weights: np.ndarray,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    beta_prior_scale: float = 5.0,
    seed: int = 0,
) -> InhomogeneousIntensityPosterior:
    """Log-linear inhomogeneous Poisson-process intensity fit via NUTS.

    Fits ``log lambda(x) = beta0 + x . beta`` as the intensity of an
    inhomogeneous Poisson process. The point-process log-likelihood has the
    standard two terms — the log-intensity summed over detected events minus
    the **integrated intensity** over the observation domain
    ([daleyVereJones2008]):

        log L(beta)  =  sum_i [beta0 + x_i . beta]  -  integral lambda(x) dx,

    where the integral is approximated by quadrature over caller-supplied
    background points,
    ``integral lambda(x) dx ~= sum_q w_q * exp(beta0 + x_q . beta)``.

    The exposure (integrated-intensity) term is what makes the rate
    identifiable: without it the likelihood is maximised by ``lambda -> 1`` at
    every observed covariate and the fit cannot estimate an event rate over a
    region / time window. The quadrature weights carry the domain measure
    (area * duration), so the recovered ``beta0`` is a true log-rate density.
    NumPyro is imported lazily.

    Args:
        covariates: Detected-event covariate matrix ``(n_events, n_covariates)``.
        quadrature_covariates: Background / integration covariate points
            ``(n_quad, n_covariates)`` spanning the observation domain over
            which the intensity is integrated.
        quadrature_weights: Per-point quadrature weights ``(n_quad,)`` — the
            domain measure (e.g. area * duration per cell); must be strictly
            positive.
        num_warmup: NUTS warmup iterations.
        num_samples: Posterior samples (per chain).
        num_chains: Number of NUTS chains.
        beta_prior_scale: Std of the ``Normal(0, .)`` prior on the
            intercept and coefficients.
        seed: PRNG seed for the sampler.

    Returns:
        The :class:`InhomogeneousIntensityPosterior`.

    Raises:
        ValueError: If ``covariates`` / ``quadrature_covariates`` are not 2-D
            with a matching covariate dimension, if either has zero rows, or if
            the quadrature weights are mis-shaped or non-positive.
    """
    cov = np.asarray(covariates, dtype=float)
    if cov.ndim != 2 or cov.shape[0] == 0:
        msg = f"covariates must be a (n_events, n_covariates) array, got shape {cov.shape}"
        raise ValueError(msg)
    quad = np.asarray(quadrature_covariates, dtype=float)
    weights = np.asarray(quadrature_weights, dtype=float)
    if quad.ndim != 2 or quad.shape[0] == 0:
        msg = (
            "quadrature_covariates must be a (n_quad, n_covariates) array, "
            f"got shape {quad.shape}"
        )
        raise ValueError(msg)
    if quad.shape[1] != cov.shape[1]:
        msg = (
            "quadrature_covariates and covariates must share the covariate "
            f"dimension, got {quad.shape[1]} vs {cov.shape[1]}"
        )
        raise ValueError(msg)
    if weights.shape != (quad.shape[0],):
        msg = (
            "quadrature_weights must have shape (n_quad,) matching "
            f"quadrature_covariates, got {weights.shape} vs ({quad.shape[0]},)"
        )
        raise ValueError(msg)
    if np.any(weights <= 0.0):
        msg = "quadrature_weights must be strictly positive"
        raise ValueError(msg)

    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    n_cov = cov.shape[1]
    cov_j = jax.numpy.asarray(cov)
    quad_j = jax.numpy.asarray(quad)
    weights_j = jax.numpy.asarray(weights)

    def model(x: Any, xq: Any, wq: Any) -> None:
        beta0 = numpyro.sample("beta0", dist.Normal(0.0, beta_prior_scale))
        beta = numpyro.sample(
            "beta", dist.Normal(0.0, beta_prior_scale).expand([n_cov]).to_event(1)
        )
        # Inhomogeneous-Poisson log-likelihood: sum_i log lambda(x_i) minus the
        # integrated intensity int lambda(x) dx, the integral approximated by
        # sum_q w_q lambda(x_q). Added via numpyro.factor because it is not a
        # per-observation density.
        log_lambda_events = beta0 + x @ beta
        integrated = jax.numpy.sum(wq * jax.numpy.exp(beta0 + xq @ beta))
        numpyro.factor("point_process", jax.numpy.sum(log_lambda_events) - integrated)

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), x=cov_j, xq=quad_j, wq=weights_j)
    samples = mcmc.get_samples()
    return InhomogeneousIntensityPosterior(
        beta0_samples=np.asarray(samples["beta0"]),
        beta_samples=np.asarray(samples["beta"]),
    )
