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

The importance-corrected models over per-event posteriors (Tier V's
canonical likelihood, 06a / 06b) are re-exported here from
``plumax.population._src.fitting_iw``: :func:`mark_iw_model`,
:func:`tmtpp_iw_model` and :func:`fit_size_distribution_iw`.

This module imports NumPyro at module level; :mod:`plumax.population`
binds its names lazily so importing the package stays cheap. pandas is
imported only inside :func:`run_mcmc`.
"""

from __future__ import annotations

from plumax.population._src.fitting import (
    X_MAX_DEFAULT,
    X_MIN_DEFAULT,
    MarkPosterior,
    TMTPPPosterior,
    fit_size_distribution_iw,
    log_thinned_rate,
    lognorm_cdf,
    mark_iw_model,
    pod_powerlaw_model,
    power_law,
    run_mcmc,
    tmtpp_iw_model,
)


__all__ = [
    "X_MAX_DEFAULT",
    "X_MIN_DEFAULT",
    "MarkPosterior",
    "TMTPPPosterior",
    "fit_size_distribution_iw",
    "log_thinned_rate",
    "lognorm_cdf",
    "mark_iw_model",
    "pod_powerlaw_model",
    "power_law",
    "run_mcmc",
    "tmtpp_iw_model",
]
