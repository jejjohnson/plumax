"""Tier V.A — instantaneous emission size distribution.

Hierarchical-Bayes fit of the population size distribution ``p(Q)`` from
the *per-event posteriors* in an :class:`~plumax.population.catalog.EmissionCatalog`
— not from point estimates.  This is the design's core commitment: each
event's posterior uncertainty must propagate into the population
parameters.

The model is a hierarchical lognormal:

* population parameters ``theta = (mu, sigma)`` with weakly-informative
  hyperpriors;
* per-event latent log-rate ``log Q_i ~ Normal(mu, sigma)``;
* per-event soft observation: the catalog's lognormal summary
  ``(log_mean_i, log_std_i)`` enters as a Gaussian likelihood on the
  latent ``log Q_i``.

Because each event contributes a *Gaussian* (with its own ``log_std_i``)
rather than a delta at the point estimate, poorly-constrained events pull
less weight and widen the posterior on ``(mu, sigma)`` — the
uncertainty-propagation property the design requires.

NumPyro is a heavy, optional dependency: it is imported lazily inside
:func:`fit_lognormal_size_distribution`, so ``import
plumax.population.size_distribution`` works without NumPyro installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from plumax.population.catalog import EmissionCatalog

__all__ = [
    "SizeDistributionPosterior",
    "fit_lognormal_size_distribution",
]


@dataclass(frozen=True)
class SizeDistributionPosterior:
    """Posterior over the lognormal population size-distribution parameters.

    Attributes:
        mu_samples: Posterior samples of the log-scale location ``mu``
            ``(n_samples,)``.
        sigma_samples: Posterior samples of the log-scale spread ``sigma``
            ``(n_samples,)``.
    """

    mu_samples: np.ndarray
    sigma_samples: np.ndarray

    @property
    def mu_mean(self) -> float:
        """Posterior-mean log-scale location ``mu``."""
        return float(np.mean(self.mu_samples))

    @property
    def sigma_mean(self) -> float:
        """Posterior-mean log-scale spread ``sigma``."""
        return float(np.mean(self.sigma_samples))

    def summary(self) -> dict[str, dict[str, float]]:
        """Posterior mean / std / 95% credible interval for ``mu`` and ``sigma``.

        Returns:
            A nested dict keyed by ``"mu"`` / ``"sigma"`` with ``mean``,
            ``std``, ``q2.5`` and ``q97.5`` entries.
        """
        out: dict[str, dict[str, float]] = {}
        for name, samples in (("mu", self.mu_samples), ("sigma", self.sigma_samples)):
            lo, hi = np.quantile(samples, [0.025, 0.975])
            out[name] = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "q2.5": float(lo),
                "q97.5": float(hi),
            }
        return out

    def fitted_pdf(self, q: np.ndarray) -> np.ndarray:
        """Posterior-mean fitted lognormal density evaluated at ``q``.

        Evaluates the lognormal pdf at the posterior-mean ``(mu, sigma)``.

        Args:
            q: Emission rates [kg/s] at which to evaluate the density;
                must be strictly positive.

        Returns:
            The density values, same shape as ``q``.
        """
        q = np.asarray(q, dtype=float)
        return stats.lognorm.pdf(q, s=self.sigma_mean, scale=np.exp(self.mu_mean))

    def quantile(self, p: float | np.ndarray) -> np.ndarray:
        """Posterior-mean fitted-distribution quantile(s) of ``Q``.

        Args:
            p: Probability level(s) in ``(0, 1)``.

        Returns:
            The corresponding emission-rate quantile(s) [kg/s].
        """
        return stats.lognorm.ppf(p, s=self.sigma_mean, scale=np.exp(self.mu_mean))


def fit_lognormal_size_distribution(
    catalog: EmissionCatalog,
    *,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    mu_prior_scale: float = 5.0,
    sigma_prior_scale: float = 2.0,
    seed: int = 0,
) -> SizeDistributionPosterior:
    """Hierarchical-Bayes fit of the lognormal population size distribution.

    Fits ``mu`` and ``sigma`` of the population size distribution
    ``Q ~ LogNormal(mu, sigma^2)`` from the per-event posteriors, with each
    event's latent ``log Q_i`` marginalised analytically: because both the
    population prior on ``log Q_i`` and the per-event soft observation are
    Gaussian, the marginal likelihood of the observed ``log_mean_i`` is

    .. math::

        \\text{log\\_mean}_i \\sim \\text{Normal}(\\mu, \\sigma^2 + \\text{log\\_std}_i^2).

    The per-event ``log_std_i`` widens the effective likelihood variance,
    so poorly-constrained events propagate their uncertainty into the
    ``(mu, sigma)`` posterior (a catalog of wide events yields a wider
    ``theta`` posterior than naively fitting the point estimates).

    NumPyro is imported lazily here so the module import stays NumPyro-free.

    Args:
        catalog: The per-event posterior catalog (≥ 1 event).
        num_warmup: NUTS warmup iterations.
        num_samples: Posterior samples to draw (per chain).
        num_chains: Number of NUTS chains.
        mu_prior_scale: Std of the ``Normal(0, .)`` hyperprior on ``mu``.
        sigma_prior_scale: Scale of the ``HalfNormal`` hyperprior on
            ``sigma``.
        seed: PRNG seed for the sampler.

    Returns:
        The fitted :class:`SizeDistributionPosterior`.

    Raises:
        ValueError: If the catalog is empty.
    """
    if catalog.n_events == 0:
        msg = "cannot fit a size distribution to an empty catalog"
        raise ValueError(msg)

    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    log_mean, log_std = catalog.log_moments()
    log_mean_j = jax.numpy.asarray(log_mean)
    log_std_j = jax.numpy.asarray(log_std)

    def model(obs_mean: Any, obs_std: Any) -> None:
        mu = numpyro.sample("mu", dist.Normal(0.0, mu_prior_scale))
        sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior_scale))
        eff_std = jax.numpy.sqrt(sigma**2 + obs_std**2)
        numpyro.sample("obs", dist.Normal(mu, eff_std), obs=obs_mean)

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), obs_mean=log_mean_j, obs_std=log_std_j)
    samples = mcmc.get_samples()
    return SizeDistributionPosterior(
        mu_samples=np.asarray(samples["mu"]),
        sigma_samples=np.asarray(samples["sigma"]),
    )
