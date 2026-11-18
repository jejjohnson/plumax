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

from plumax.population._src.size_distribution import (
    SizeDistributionPosterior,
    fit_lognormal_size_distribution,
)


__all__ = [
    "SizeDistributionPosterior",
    "fit_lognormal_size_distribution",
]
