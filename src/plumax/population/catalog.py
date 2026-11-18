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

The adapter reads a scalar ``(Q_mean, Q_std)`` from each tier's posterior
type, normalising their heterogeneous field layouts:

* :class:`plumax.coupled.fusion.FusionPosterior` (Tier IV) already exposes
  scalar ``emission_rate`` / ``emission_std``; they are read directly.
* :class:`plumax.lagrangian.inversion.GaussianPosterior` (Tier II) is a
  *grid-vector* source-field posterior (``mean`` ``(n_grid,)``,
  ``covariance`` ``(n_grid, n_grid)``).  The event-scale ``Q`` is the
  total over the field: ``Q = 1ᵀ mean`` with
  ``Q_std = sqrt(1ᵀ Σ 1)``.
* :class:`plumax.lagrangian.inversion.LognormalPosterior` (Tier II) is the
  sign-constrained grid-vector posterior (``mean`` ``(n_grid,)``,
  ``log_covariance``).  The total ``Q = 1ᵀ mean`` and its std is
  propagated from the log-covariance via the delta method:
  ``Var[Q] ≈ (mean ⊙ ...) ... `` — concretely
  ``Var[q_i, q_j] ≈ q_i q_j (exp(Σ_log[i, j]) − 1)`` so
  ``Q_std = sqrt(qᵀ (exp(Σ_log) − 1) q)``.

For a single-source overpass these reductions are exact / near-exact; for
genuine multi-source overpasses, build one event per source upstream.

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
