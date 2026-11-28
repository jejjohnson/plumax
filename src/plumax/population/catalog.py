"""Cross-tier posterior catalog — the Tier V load-bearing interface.

Tiers II-IV each produce per-event posteriors over the emission rate
``Q``.  This module materialises those posteriors as a tier-agnostic
:class:`EmissionCatalog` of :class:`EmissionEvent` records that the
population fits (V.A size distribution, V.B point process) consume.

The catalog is deliberately thin: it carries the per-event posterior mean
``emission_rate`` and standard deviation ``emission_std`` together with the
source location and detection time, plus provenance (``tier``,
``instrument``).  It does *not* re-derive any transport physics — that is
inherited from the per-event posteriors.

The adapter reads the scalar ``(Q_mean, Q_std)`` that every tier's
posterior exposes as ``emission_rate`` and ``emission_std``.  The grid-vector
posteriors of Tier II compute it themselves:
:class:`plumax.lagrangian.inversion.GaussianPosterior` gives the field total
``Q = 1ᵀ mean`` with ``Q_std = sqrt(1ᵀ Σ 1)``, and
:class:`plumax.lagrangian.inversion.LognormalPosterior` gives the same total
with the delta-method ``Q_std = sqrt(qᵀ (exp(Σ_log) − 1) q)``.  For genuine
multi-source overpasses, build one event per source upstream.

Payload v2 adds what the importance-corrected mark likelihood needs: every
tier's posterior also exposes ``samples(key, n)`` (draws of Q) and
``prior_logpdf(q)`` (the log-density of the per-event prior on Q the
posterior was made under), and :func:`event_from_posterior` copies them onto
the event as ``q_samples`` and ``prior_logpdf`` when they are available. The
Gaussian summary is unchanged, so v1 consumers see no difference.
"""

from __future__ import annotations

from plumax.population._src.catalog import (
    EmissionCatalog,
    EmissionEvent,
    PerEventPosterior,
    event_from_posterior,
)


__all__ = [
    "EmissionCatalog",
    "EmissionEvent",
    "PerEventPosterior",
    "event_from_posterior",
]
