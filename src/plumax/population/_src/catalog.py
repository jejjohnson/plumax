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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import einx
import jax.numpy as jnp
import numpy as np

from plumax._src.tracing import _is_traced, require_positive


if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax

__all__ = [
    "EmissionCatalog",
    "EmissionEvent",
    "PerEventPosterior",
    "event_from_posterior",
]


@runtime_checkable
class PerEventPosterior(Protocol):
    """Structural type for a per-event ``Q`` posterior (payload v2).

    A posterior mean and standard deviation of the emission rate ``Q``, plus
    draws of ``Q`` and the log-density of the per-event prior on ``Q`` that
    the importance correction divides out. Every tier implements it:
    ``gauss_plume.EmissionRateSamples`` (Tier I),
    ``lagrangian.GaussianPosterior`` / ``LognormalPosterior`` (Tier II, for
    the field total), ``les_fvm.FourDVarResult`` (Tier III, for the time
    mean), ``coupled.FusionPosterior`` and ``coupled.CoupledResult``
    (Tier IV, for a scalar ``Q``).
    """

    @property
    def emission_rate(self) -> jax.Array:
        """Posterior-mean emission rate ``Q`` [kg/s]."""
        ...

    @property
    def emission_std(self) -> jax.Array:
        """Posterior standard deviation of ``Q`` [kg/s]."""
        ...

    def samples(self, key: jax.Array, n: int) -> jax.Array:
        """``n`` posterior draws of ``Q`` [kg/s]."""
        ...

    def prior_logpdf(self, q: jax.Array) -> jax.Array:
        """Log-density of the per-event prior on ``Q`` at ``q``."""
        ...


@dataclass(frozen=True)
class EmissionEvent:
    """A single detected emission event with its per-event posterior summary.

    Attributes:
        emission_rate: Posterior-mean emission rate ``Q`` [kg/s].
        emission_std: Posterior standard deviation of ``Q`` [kg/s];
            must be non-negative.
        x: Source east-west location [m].
        y: Source north-south location [m].
        time: Detection / overpass time [s] (UTC seconds).
        tier: Originating tier label, e.g. ``"II"`` / ``"III"`` / ``"IV"``.
        instrument: Instrument identifier for per-satellite POD dispatch,
            or ``None`` when unknown.
        q_samples: Posterior draws of ``Q`` [kg/s], or ``None`` (payload v2).
        prior_logpdf: Log-density of the per-event prior on ``Q``, or
            ``None`` (payload v2). Both are left out of equality.
    """

    emission_rate: float
    emission_std: float
    x: float
    y: float
    time: float
    tier: str
    instrument: str | None = None
    q_samples: jax.Array | None = field(default=None, compare=False)
    prior_logpdf: Callable[[jax.Array], jax.Array] | None = field(
        default=None, compare=False
    )

    def __post_init__(self) -> None:
        """Validate that ``emission_std`` is non-negative.

        The check is skipped when ``emission_std`` is a traced JAX value
        (under ``jit`` / ``grad`` / ``vmap``) because ``bool(...)`` on a
        tracer raises ``TracerError``.
        """
        if _is_traced(self.emission_std):
            return
        require_positive("emission_std", self.emission_std, allow_zero=True)


def _scalar_q_summary(posterior: object) -> tuple[jax.Array, jax.Array]:
    """Read the scalar ``(Q_mean, Q_std)`` of any Tier I-IV posterior.

    Every plumax posterior exposes ``emission_rate`` and ``emission_std``
    (the grid-vector Tier II posteriors as the total over their field), so
    they are read directly.

    Args:
        posterior: The per-event posterior.

    Returns:
        ``(Q_mean, Q_std)`` as 0-d JAX arrays.

    Raises:
        TypeError: If the posterior does not expose ``emission_rate`` and
            ``emission_std``.
    """
    rate_attr = getattr(posterior, "emission_rate", None)
    std_attr = getattr(posterior, "emission_std", None)
    if rate_attr is not None and std_attr is not None:
        rate = einx.id("... -> (...)", jnp.asarray(rate_attr))
        std = einx.id("... -> (...)", jnp.asarray(std_attr))
        return rate[0], std[0]

    msg = (
        "event_from_posterior: posterior must expose "
        "(emission_rate, emission_std); "
        f"got {type(posterior).__name__}."
    )
    raise TypeError(msg)


def event_from_posterior(
    posterior: object,
    *,
    x: float,
    y: float,
    time: float,
    tier: str,
    instrument: str | None = None,
    n_samples: int = 512,
    key: jax.Array | None = None,
) -> EmissionEvent:
    """Build an :class:`EmissionEvent` from any Tier I-IV posterior.

    The posterior is duck-typed: a scalar ``(Q_mean, Q_std)`` is extracted
    by :func:`_scalar_q_summary`, so the same adapter works for
    :class:`~plumax.coupled.fusion.FusionPosterior` (read directly) and the
    grid-vector :class:`~plumax.lagrangian.inversion.GaussianPosterior` /
    :class:`~plumax.lagrangian.inversion.LognormalPosterior` (which report
    their field total).  For genuine multi-source overpasses, build one event per
    source upstream.

    Args:
        posterior: A per-event posterior.
        x: Source east-west location [m].
        y: Source north-south location [m].
        time: Detection / overpass time [s].
        tier: Originating tier label.
        instrument: Instrument identifier, or ``None``.
        n_samples: Number of draws of ``Q`` to store when ``key`` is given.
        key: PRNG key for the draws; without it ``q_samples`` stays ``None``.

    Returns:
        The corresponding :class:`EmissionEvent`, with ``q_samples`` and
        ``prior_logpdf`` filled when the posterior exposes them (payload v2)
        and ``None`` otherwise.
    """
    rate, std = _scalar_q_summary(posterior)
    draw = getattr(posterior, "samples", None)
    q_samples = None
    if key is not None and callable(draw):
        q_samples = jnp.asarray(draw(key, n_samples))
    prior_logpdf = getattr(posterior, "prior_logpdf", None)
    return EmissionEvent(
        emission_rate=float(rate),
        emission_std=float(std),
        x=float(x),
        y=float(y),
        time=float(time),
        tier=tier,
        instrument=instrument,
        q_samples=q_samples,
        prior_logpdf=prior_logpdf if callable(prior_logpdf) else None,
    )


@dataclass(frozen=True)
class EmissionCatalog:
    """A population of detected emission events.

    Attributes:
        events: The detected events, one per (de-duplicated) detection.
    """

    events: tuple[EmissionEvent, ...] = field(default_factory=tuple)

    @classmethod
    def from_events(cls, events: Sequence[EmissionEvent]) -> EmissionCatalog:
        """Build a catalog from any sequence of events.

        Args:
            events: The events to wrap (copied into a tuple).

        Returns:
            The catalog.
        """
        return cls(events=tuple(events))

    @property
    def n_events(self) -> int:
        """Number of events in the catalog."""
        return len(self.events)

    @property
    def emission_rate(self) -> np.ndarray:
        """Per-event posterior-mean emission rates ``(n_events,)`` [kg/s]."""
        return np.asarray([event.emission_rate for event in self.events], dtype=float)

    @property
    def emission_std(self) -> np.ndarray:
        """Per-event posterior standard deviations ``(n_events,)`` [kg/s]."""
        return np.asarray([event.emission_std for event in self.events], dtype=float)

    @property
    def x(self) -> np.ndarray:
        """Per-event source east-west locations ``(n_events,)`` [m]."""
        return np.asarray([event.x for event in self.events], dtype=float)

    @property
    def y(self) -> np.ndarray:
        """Per-event source north-south locations ``(n_events,)`` [m]."""
        return np.asarray([event.y for event in self.events], dtype=float)

    @property
    def time(self) -> np.ndarray:
        """Per-event detection times ``(n_events,)`` [s]."""
        return np.asarray([event.time for event in self.events], dtype=float)

    def log_moments(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Per-event lognormal sufficient statistic ``(log_mean, log_std)``.

        Each per-event posterior is summarised by a Gaussian mean ``m`` and
        standard deviation ``s`` on ``Q``.  V.A needs the matching moments
        in log-space, so we moment-match ``Q ~ Normal(m, s^2)`` to a
        lognormal ``Q ~ LogNormal(log_mean, log_std^2)`` by equating the
        first two moments:

        .. math::

            \log\_std^2 = \ln\!\left(1 + (s / m)^2\right), \qquad
            \log\_mean = \ln(m) - \tfrac12 \log\_std^2.

        This is the standard lognormal moment match.  It is exact in the
        small-:math:`\sigma` limit (coefficient of variation ``s / m``
        small) where the Gaussian and lognormal coincide, and degrades for
        very poorly-constrained events (CV ≳ 1), which the V.A regime rule
        flags for the full-sample path.  Events with non-positive mean are
        not representable in log-space and raise.

        Returns:
            A tuple ``(log_mean, log_std)`` of arrays, each ``(n_events,)``.
        """
        rate = self.emission_rate
        std = self.emission_std
        if np.any(rate <= 0.0):
            msg = "log_moments requires strictly positive emission_rate for every event"
            raise ValueError(msg)
        # c² = (s / m)²
        squared_cv = (std / rate) ** 2
        # σ_log² = ln(1 + c²)
        log_var = np.log1p(squared_cv)
        # σ_log = √σ_log²
        log_std = np.sqrt(log_var)
        # μ_log = ln(m) − ½ σ_log²
        log_mean = np.log(rate) - 0.5 * log_var
        return log_mean, log_std
