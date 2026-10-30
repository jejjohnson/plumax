"""Importance-corrected population fits: the mark-only fit and the full TMTPP.

Both score detection i by the importance-corrected mark term of
:mod:`~plumax.population.importance`,

    mᵢ(θ) = log Σₛ exp(ℓᵢₛ + log P_d(qᵢₛ) + log f(qᵢₛ | θ)),

with the thinned rate p̄(θ) = ∫ P_d(q) f(q | θ) dq in log space
(:func:`~plumax.population.integrate.log_simpson`; p̄ = 1 without a POD).

* :func:`mark_iw_model`: the marks given the detections,
  Σᵢ mᵢ(θ) − N log p̄(θ), the λ-free part of the likelihood.
* :func:`tmtpp_iw_model`: the full TMTPP,
  Σᵢ [log λ(tᵢ) + mᵢ(θ)] − Λ(T) p̄(θ), with the ground intensity from the
  intensity registry on xtremax (:func:`~plumax.population.processes.make_process`).
* :func:`fit_size_distribution_iw`: NUTS on :func:`mark_iw_model`, with the
  per-event ESS and Pareto k̂ at the posterior-mean θ and a collapse warning.

The POD is a fixed callable here; its calibration prior is #110.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from plumax.population._src.importance import (
    EventEvidence,
    Measure,
    Regime,
    catalog_evidence,
    effective_sample_size,
    warn_on_collapse,
)
from plumax.population._src.integrate import log_simpson, pareto_smoothed_log_weights
from plumax.population._src.marks import MARKS, Mark


if TYPE_CHECKING:
    from plumax.population.catalog import EmissionCatalog


#: Decades of rate either side of the catalog's draws that p̄ integrates over.
SUPPORT_DECADES = 4.0


def _log_integrand(
    mark: Mark, theta: dict[str, Any], log_pod: Callable[[Array], Array] | None
) -> Callable[[Array], Array]:
    def log_integrand(q: Array) -> Array:
        # log f(q | θ) + log P_d(q)
        value = mark.log_prob(q, **theta)
        if log_pod is not None:
            value = value + log_pod(q)
        return value

    return log_integrand


def log_thinned_rate(
    mark: Mark,
    theta: dict[str, Any],
    log_pod: Callable[[Array], Array] | None,
    support: tuple[float, float],
    *,
    n_nodes: int = 2001,
) -> Float[Array, ""]:
    """log p̄(θ) = log ∫ P_d(q) f(q | θ) dq, 0 without a POD.

    Log-space Simpson on [q_lo, q_hi], plus the mass above q_hi at the POD's
    value there, P_d(q_hi) P(Q > q_hi | θ), so a heavy tail beyond the grid is
    not dropped. The mass below q_lo is left out: a detection curve is
    nondecreasing in Q, and q_lo sits far below every detection.

    Args:
        mark: The mark family.
        theta: Its parameters.
        log_pod: log P_d, or ``None`` for certain detection.
        support: ``(q_lo, q_hi)`` of the log-space Simpson rule [kg/s].
        n_nodes: Simpson nodes.

    Returns:
        log p̄.
    """
    if log_pod is None:
        # a normalised f integrates to one
        return jnp.zeros(())
    integrand = _log_integrand(mark, theta, log_pod)
    inside = log_simpson(integrand, support[0], support[1], n=n_nodes)
    # P_d(q_hi) P(Q > q_hi), the tail beyond the last node
    upper = jnp.asarray(support[1], dtype=float)
    beyond = log_pod(upper) + mark.log_survival(upper, **theta)
    return jnp.logaddexp(inside, beyond)


def default_support(evidence: EventEvidence) -> tuple[float, float]:
    """``SUPPORT_DECADES`` decades either side of the evidence nodes."""
    nodes = np.asarray(evidence.q_samples)
    positive = nodes[nodes > 0.0]
    lower = float(positive.min()) * 10.0 ** (-SUPPORT_DECADES)
    upper = float(positive.max()) * 10.0**SUPPORT_DECADES
    return lower, upper


def mark_iw_model(
    evidence: EventEvidence,
    *,
    mark: Mark,
    log_pod: Callable[[Array], Array] | None = None,
    support: tuple[float, float] | None = None,
) -> None:
    """NumPyro model of the marks given the detections, importance-corrected.

    log L(θ) = Σᵢ mᵢ(θ) − N log p̄(θ), θ under ``mark.numpyro_prior("mark")``.

    Args:
        evidence: Stacked per-event evidence (:func:`catalog_evidence`).
        mark: The mark family.
        log_pod: log P_d on rates in the marks' units, or ``None``.
        support: Rate range of the p̄ integral; default from the evidence.
    """
    import numpyro

    theta = mark.numpyro_prior("mark")
    integrand = _log_integrand(mark, theta, log_pod)
    # mᵢ(θ) = log Σₛ exp(ℓᵢₛ + g(qᵢₛ))
    mark_terms = evidence.log_mark_term(integrand)
    support = support or default_support(evidence)
    log_rate = log_thinned_rate(mark, theta, log_pod, support)
    n_events = mark_terms.shape[0]
    # Σᵢ mᵢ − N log p̄
    numpyro.factor("marks", jnp.sum(mark_terms) - n_events * log_rate)


def tmtpp_iw_model(
    event_times: Float[Array, " event"],
    mask: Bool[Array, " event"],
    evidence: EventEvidence,
    *,
    mark: Mark,
    intensity: str,
    window: tuple[float, float],
    log_pod: Callable[[Array], Array] | None = None,
    support: tuple[float, float] | None = None,
) -> None:
    """NumPyro model of the full importance-corrected TMTPP likelihood.

    log L = Σᵢ [log λ(tᵢ) + mᵢ(θ)] − Λ(T) p̄(θ), with λ a registry intensity
    under its ``sample_priors`` on the xtremax Poisson process of the window.

    Args:
        event_times: Detection times since t₀ in xtremax's padded buffer
            (:func:`~plumax.population.processes.padded_event_buffer`) [days].
        mask: Its mask.
        evidence: Stacked per-event evidence, one row per detection in the
            order of the buffer's real entries.
        mark: The mark family.
        intensity: A registry name of a Poisson intensity.
        window: ``(t₀, t₁)`` [days].
        log_pod: log P_d on rates in the marks' units, or ``None``.
        support: Rate range of the p̄ integral; default from the evidence.

    Raises:
        ImportError: Without the ``population`` extra.
    """
    import numpyro

    from plumax.population._src.processes import make_process
    from plumax.population.intensity import INTENSITY_REGISTRY

    theta = mark.numpyro_prior("mark")
    kernel = INTENSITY_REGISTRY[intensity].sample_priors()
    process = make_process(kernel, window)
    # Σᵢ log λ(tᵢ) over the real events
    log_rates = process.log_intensity_fn(event_times)
    ground = jnp.sum(jnp.where(mask, log_rates, 0.0))
    # Λ(T)
    compensator = process.effective_integrated_intensity()
    integrand = _log_integrand(mark, theta, log_pod)
    mark_terms = evidence.log_mark_term(integrand)
    support = support or default_support(evidence)
    log_rate = log_thinned_rate(mark, theta, log_pod, support)
    # Σᵢ [log λ(tᵢ) + mᵢ] − Λ(T) p̄
    log_likelihood = ground + jnp.sum(mark_terms) - compensator * jnp.exp(log_rate)
    numpyro.factor("tmtpp", log_likelihood)


@dataclass(frozen=True)
class MarkPosterior:
    """Posterior over a mark family's parameters, with importance diagnostics.

    Attributes:
        mark: The registry name of the family.
        regime: How the events entered.
        samples: Posterior draws of θ by parameter name.
        ess: Per-event ESS of the importance weights at the posterior-mean θ.
        pareto_k: Per-event Pareto k̂ (FULL regime; NaN otherwise).
    """

    mark: str
    regime: Regime
    samples: dict[str, np.ndarray] = field(repr=False)
    ess: np.ndarray = field(repr=False)
    pareto_k: np.ndarray = field(repr=False)

    def posterior_mean(self) -> dict[str, np.ndarray]:
        """θ̄, the posterior mean of every parameter."""
        return {name: np.mean(draws, axis=0) for name, draws in self.samples.items()}


def fit_size_distribution_iw(
    catalog: EmissionCatalog,
    *,
    mark: str = "lognormal",
    mark_fields: dict[str, Any] | None = None,
    regime: Regime | str = Regime.FULL,
    log_pod: Callable[[Array], Array] | None = None,
    support: tuple[float, float] | None = None,
    n_nodes: int = 32,
    measure: Measure = "lognormal",
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    seed: int = 0,
) -> MarkPosterior:
    """Importance-corrected NUTS fit of the population size distribution.

    Runs :func:`mark_iw_model`, then reports the per-event ESS (and, in the
    FULL regime, the Pareto k̂) of the weights at the posterior-mean θ,
    warning in every regime when an event collapses (ESS < 0.1 S, or
    k̂ > 0.7).

    Args:
        catalog: Events with the payload-v2 fields (``prior_logpdf``, and
            ``q_samples`` for FULL).
        mark: A registry name (:data:`~plumax.population.marks.MARKS`).
        mark_fields: The family's fields (support, prior settings).
        regime: POINT, GAUSSIAN or FULL.
        log_pod: log P_d on rates in the marks' units, or ``None``.
        support: Rate range of the p̄ integral; default from the evidence.
        n_nodes: Gauss–Hermite nodes of the GAUSSIAN regime.
        measure: The GAUSSIAN-regime summary, ``"lognormal"`` or ``"normal"``
            (see :func:`~plumax.population.importance.event_evidence`).
        num_warmup: NUTS warm-up steps.
        num_samples: Posterior draws per chain.
        num_chains: NUTS chains.
        seed: PRNG seed.

    Returns:
        The :class:`MarkPosterior`.

    Raises:
        PayloadError: When the events lack the payload the regime needs.
    """
    from numpyro.infer import MCMC, NUTS

    regime = Regime(regime)
    family = MARKS[mark](**(mark_fields or {}))
    evidence = catalog_evidence(
        catalog, regime=regime, n_nodes=n_nodes, measure=measure
    )
    mcmc = MCMC(
        NUTS(mark_iw_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        evidence,
        mark=family,
        log_pod=log_pod,
        support=support,
    )
    draws = mcmc.get_samples()
    prefix = "mark."
    samples = {
        name[len(prefix) :]: np.asarray(value)
        for name, value in draws.items()
        if name.startswith(prefix) and not name.startswith(f"{prefix}raw_")
    }
    theta = {
        name: jnp.asarray(np.mean(value, axis=0)) for name, value in samples.items()
    }
    log_weights = np.asarray(evidence.log_terms(_log_integrand(family, theta, log_pod)))
    ess = np.asarray(effective_sample_size(jnp.asarray(log_weights)))
    pareto_k = np.full(ess.shape, math.nan)
    if regime is Regime.FULL:
        pareto_k = np.array(
            [pareto_smoothed_log_weights(row)[1] for row in log_weights]
        )
    warn_on_collapse(ess, log_weights.shape[-1], pareto_k)
    return MarkPosterior(
        mark=mark, regime=regime, samples=samples, ess=ess, pareto_k=pareto_k
    )
