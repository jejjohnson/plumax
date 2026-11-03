"""Importance-corrected per-event evidence for the Tier V mark likelihood.

Detection i enters the TMTPP likelihood through its mark integral against the
per-event likelihood L_i, not the posterior. With draws Q⁽ˢ⁾ from the
per-event posterior made under the prior π_i,

    ∫ P_d(q) L_i(q) f(q | θ) dq ∝ (1/S) Σₛ P_d(Q⁽ˢ⁾) f(Q⁽ˢ⁾ | θ) / π_i(Q⁽ˢ⁾),

so dividing by π_i removes the per-event prior that the posterior already
contains. :func:`event_evidence` turns an event into nodes qₛ and log-weights
ℓₛ such that the mark term is log Σₛ exp(ℓₛ + log P_d(qₛ) + log f(qₛ | θ)), in
one of three regimes (06a):

* ``FULL``: the posterior draws, ℓₛ = −log π_i(qₛ) − log S;
* ``GAUSSIAN``: Gauss–Hermite nodes of the event's lognormal summary,
  ℓₖ = log(wₖ/√π) − log π_i(qₖ), exact for integrands smooth in log Q;
* ``POINT``: the posterior mean alone, ℓ = −log π_i(Q̂), for tight posteriors.

Design validation checklist (06a / 06b), each a test in
``tests/population/test_importance.py``:

* prior-swap invariance: per-event priors LogNormal(0, 1.5) and
  LogNormal(0, 0.5) give the same likelihood in θ up to importance noise;
* regime consistency: POINT, GAUSSIAN and FULL agree on tight posteriors;
* reduction: with π_i = f(· | θ₀) the corrected term at θ₀ is the plain
  posterior average of P_d;
* log-space integration: :func:`~plumax.population.integrate.log_simpson`
  matches Pareto importance sampling for a Hill POD against a lognormal-Pareto
  mark where linear quadrature does not;
* ESS: S for equal weights, near 1 when one weight dominates, and a warning
  when ESS < 0.1 S or the Pareto k̂ > 0.7.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from plumax._src.errors import PayloadError
from plumax._src.scalar_emission import GaussianEmission, LognormalEmission
from plumax.population._src.integrate import pareto_smoothed_log_weights


if TYPE_CHECKING:
    from plumax.population.catalog import EmissionCatalog, EmissionEvent


#: ESS below this fraction of the draws flags a collapsed event (06a).
ESS_WARNING_FRACTION = 0.1
#: Pareto shape above which importance weights are unreliable (PSIS).
PARETO_K_WARNING = 0.7


#: Nats by which the summary likelihood p̃/π may rise at an outermost
#: Hermite node above its interior maximum before the GAUSSIAN regime is
#: rejected (equal summary and prior give a flat likelihood, 0 nats).
TAIL_TOLERANCE = 1.0
#: The GAUSSIAN regime's two-moment summary of a posterior.
Measure = Literal["lognormal", "normal"]


class Regime(StrEnum):
    """How an event's per-event posterior enters the mark integral."""

    POINT = "point"
    GAUSSIAN = "gaussian"
    FULL = "full"


class EventEvidence(eqx.Module):
    """Nodes and log-weights of one event's (or a stack of events') mark integral.

    The mark term is log Σₛ exp(ℓₛ + g(qₛ)) for g = log P_d + log f(· | θ).

    Attributes:
        log_weights: ℓₛ, including −log π_i(qₛ).
        q_samples: The nodes qₛ [kg/s].
    """

    log_weights: Float[Array, "*event member"]
    q_samples: Float[Array, "*event member"]

    def log_terms(
        self, log_integrand: Callable[[Array], Array]
    ) -> Float[Array, "*event member"]:
        """ℓₛ + g(qₛ), with nodes at Q ≤ 0 excluded (−∞).

        g is evaluated on a positive stand-in at excluded nodes, so a POD or
        mark that is undefined there cannot turn the sum into NaN.
        """
        positive = self.q_samples > 0.0
        safe_nodes = jnp.where(positive, self.q_samples, 1.0)
        terms = self.log_weights + log_integrand(safe_nodes)
        return jnp.where(positive, terms, -jnp.inf)

    def log_mark_term(
        self, log_integrand: Callable[[Array], Array]
    ) -> Float[Array, "*event"]:
        """log Σₛ exp(ℓₛ + g(qₛ)), per event."""
        return logsumexp(self.log_terms(log_integrand), axis=-1)

    def effective_sample_size(
        self, log_integrand: Callable[[Array], Array] | None = None
    ) -> Float[Array, "*event"]:
        """ESS = (Σₛ wₛ)² / Σₛ wₛ² of wₛ = exp(ℓₛ + g(qₛ)), per event.

        Without ``log_integrand`` the weights are the prior correction alone.
        """
        if log_integrand is None:
            return effective_sample_size(self.log_weights)
        return effective_sample_size(self.log_terms(log_integrand))


def effective_sample_size(log_weights: Float[Array, "*event member"]) -> Array:
    """ESS = (Σₛ wₛ)² / Σₛ wₛ² from log-weights, along the last axis.

    Examples:
        >>> import jax.numpy as jnp
        >>> bool(jnp.allclose(effective_sample_size(jnp.zeros(8)), 8.0))
        True
    """
    # 2 log Σ w − log Σ w²
    log_ess = 2.0 * logsumexp(log_weights, axis=-1) - logsumexp(
        2.0 * log_weights, axis=-1
    )
    return jnp.exp(log_ess)


def _prior_logpdf(event: EmissionEvent) -> Callable[[Array], Array]:
    if event.prior_logpdf is None:
        raise PayloadError(
            "event_evidence: the event carries no per-event prior; build it with "
            "event_from_posterior from a payload-v2 posterior."
        )
    return event.prior_logpdf


def event_evidence(
    event: EmissionEvent,
    *,
    regime: Regime | str = Regime.FULL,
    n_nodes: int = 32,
    measure: Measure = "lognormal",
) -> EventEvidence:
    """Nodes and log-weights of one event's importance-corrected mark integral.

    In the GAUSSIAN regime the posterior is replaced by its two-moment summary,
    a lognormal (the 06a default) or a normal in Q (``measure="normal"``, the
    family of the Gaussian Tier II–IV posteriors). The prior is divided out
    exactly, so the summary's tails must not be heavier than the prior's: when
    the summary's likelihood p̃/π rises at an outermost Hermite node more than
    ``TAIL_TOLERANCE`` nats above its interior maximum, the approximation is
    rejected, since the quadrature would then be driven by its extreme nodes.
    Nodes at Q ≤ 0 carry no weight (every registry mark lives on Q > 0).

    Args:
        event: The event, with ``prior_logpdf`` (and ``q_samples`` for FULL).
        regime: POINT, GAUSSIAN or FULL.
        n_nodes: Gauss–Hermite nodes for the GAUSSIAN regime.
        measure: ``"lognormal"`` or ``"normal"``, the GAUSSIAN-regime summary.

    Returns:
        The :class:`EventEvidence`.

    Raises:
        PayloadError: Without the per-event prior, without draws for FULL, or
            when the GAUSSIAN summary's tails outweigh the prior's.
        ValueError: For a non-positive emission rate outside FULL, or an
            unknown measure.
    """
    regime = Regime(regime)
    prior_logpdf = _prior_logpdf(event)
    if regime is Regime.FULL:
        if event.q_samples is None:
            raise PayloadError(
                "event_evidence: the FULL regime needs posterior draws; pass a key "
                "to event_from_posterior."
            )
        nodes = jnp.ravel(jnp.asarray(event.q_samples, dtype=float))
        # ℓₛ = −log π(qₛ) − log S
        log_weights = -prior_logpdf(nodes) - math.log(nodes.shape[0])
        return EventEvidence(
            log_weights=_positive_only(log_weights, nodes), q_samples=nodes
        )
    if not event.emission_rate > 0.0:
        raise ValueError(
            f"event_evidence: the {regime.value} regime needs a positive "
            f"emission rate, got {event.emission_rate!r}."
        )
    if regime is Regime.POINT:
        nodes = jnp.asarray([event.emission_rate], dtype=float)
        return EventEvidence(log_weights=-prior_logpdf(nodes), q_samples=nodes)
    hermite, weights = np.polynomial.hermite.hermgauss(n_nodes)
    if measure == "lognormal":
        # lognormal summary moment-matched to (mean, sd), as catalog.log_moments
        summary = LognormalEmission.from_moments(
            event.emission_rate, event.emission_std
        )
        # qₖ = exp(μ + √2 σ xₖ)
        nodes = jnp.exp(summary.loc + math.sqrt(2.0) * summary.scale * hermite)
    elif measure == "normal":
        summary = GaussianEmission(event.emission_rate, event.emission_std)
        # qₖ = m + √2 s xₖ
        nodes = summary.mean + math.sqrt(2.0) * summary.std * jnp.asarray(hermite)
    else:
        raise ValueError(f"event_evidence: unknown measure {measure!r}.")
    log_prior = prior_logpdf(nodes)
    # ℓₖ = log(wₖ/√π) − log π(qₖ)
    quadrature = jnp.log(jnp.asarray(weights)) - 0.5 * math.log(math.pi)
    log_weights = _positive_only(quadrature - log_prior, nodes)
    # the summary's likelihood log p̃(q) − log π(q) must not climb towards
    # the outermost nodes, or they would dominate the corrected integral
    log_likelihood = np.asarray(
        _positive_only(summary.log_prob(nodes) - log_prior, nodes)
    )
    edge = max(log_likelihood[0], log_likelihood[-1])
    interior = np.max(log_likelihood[1:-1])
    if edge > interior + TAIL_TOLERANCE:
        raise PayloadError(
            f"event_evidence: the {measure} summary has heavier tails than the "
            "per-event prior, so dividing the prior out is dominated by the "
            "outermost Hermite node; use the FULL regime or the measure that "
            "matches the posterior's family."
        )
    return EventEvidence(log_weights=log_weights, q_samples=nodes)


def _positive_only(
    log_weights: Float[Array, " member"], nodes: Float[Array, " member"]
) -> Float[Array, " member"]:
    # Q ≤ 0 lies outside every mark's support; drop it before −log π can be +∞
    return jnp.where(nodes > 0.0, log_weights, -jnp.inf)


def catalog_evidence(
    catalog: EmissionCatalog,
    *,
    regime: Regime | str = Regime.FULL,
    n_nodes: int = 32,
    measure: Measure = "lognormal",
) -> EventEvidence:
    """:func:`event_evidence` for every event, stacked along a leading axis.

    Raises:
        ValueError: For an empty catalog, or FULL draws of unequal length.
    """
    if catalog.n_events == 0:
        raise ValueError("catalog_evidence: the catalog is empty.")
    per_event = [
        event_evidence(event, regime=regime, n_nodes=n_nodes, measure=measure)
        for event in catalog.events
    ]
    lengths = {evidence.q_samples.shape[0] for evidence in per_event}
    if len(lengths) != 1:
        raise ValueError(
            f"catalog_evidence: events carry different numbers of draws {lengths}."
        )
    return EventEvidence(
        log_weights=jnp.stack([evidence.log_weights for evidence in per_event]),
        q_samples=jnp.stack([evidence.q_samples for evidence in per_event]),
    )


def ess_report(
    catalog: EmissionCatalog,
    *,
    log_integrand: Callable[[Array], Array],
    regime: Regime | str = Regime.FULL,
    measure: Measure = "lognormal",
    warn: bool = True,
) -> np.ndarray:
    """Per-event ESS of the importance weights at one θ, with a collapse warning.

    wₛ = P_d(qₛ) f(qₛ | θ) / π_i(qₛ) for g = log P_d + log f(· | θ). Warns (does
    not fail) when an event's ESS < 0.1 S or, in the FULL regime, its Pareto
    k̂ > 0.7.

    Args:
        catalog: The events.
        log_integrand: g, e.g. ``lambda q: mark.log_prob(q, **theta)``.
        regime: How the events enter.
        measure: The GAUSSIAN-regime summary (see :func:`event_evidence`).
        warn: Emit the collapse warning.

    Returns:
        ESS per event.
    """
    evidence = catalog_evidence(catalog, regime=regime, measure=measure)
    log_w = np.asarray(evidence.log_terms(log_integrand))
    ess = np.asarray(effective_sample_size(jnp.asarray(log_w)))
    if warn:
        pareto_k = None
        if Regime(regime) is Regime.FULL:
            pareto_k = np.array([pareto_smoothed_log_weights(row)[1] for row in log_w])
        warn_on_collapse(ess, log_w.shape[-1], pareto_k)
    return ess


def warn_on_collapse(
    ess: np.ndarray, n_draws: int, pareto_k: np.ndarray | None = None
) -> None:
    """Warn when an event's ESS < 0.1 S or its Pareto k̂ > 0.7."""
    low = np.flatnonzero(ess < ESS_WARNING_FRACTION * n_draws)
    heavy = np.array([], dtype=int)
    if pareto_k is not None:
        heavy = np.flatnonzero(np.asarray(pareto_k) > PARETO_K_WARNING)
    if low.size or heavy.size:
        warnings.warn(
            "importance weights collapsed: ESS < "
            f"{ESS_WARNING_FRACTION:g}·S for events {low.tolist()}, Pareto k̂ > "
            f"{PARETO_K_WARNING:g} for events {heavy.tolist()}; the per-event "
            "posteriors sit far from the population mark distribution there.",
            RuntimeWarning,
            stacklevel=3,
        )
