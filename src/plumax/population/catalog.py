"""Cross-tier posterior catalog — the Tier V load-bearing interface.

Tiers II-IV each produce per-event posteriors over the instantaneous
emission rate ``Q`` that all expose a uniform ``emission_rate`` /
``emission_std`` interface (see
:mod:`plumax.lagrangian.inversion` and :mod:`plumax.coupled.fusion`).
This module materialises those posteriors as a tier-agnostic
:class:`EmissionCatalog` of :class:`EmissionEvent` records that the
population fits (V.A size distribution, V.B point process) consume.

The catalog is deliberately thin: it carries the per-event posterior mean
``emission_rate`` and standard deviation ``emission_std`` together with the
source location and detection time, plus provenance (``tier``,
``instrument``).  It does *not* re-derive any transport physics — that is
inherited from the per-event posteriors.

Note:
    The full TMTPP cross-tier contract (importance-weighted mark
    likelihood, per-event prior recall, full posterior samples) is future
    work; v1 consumes the Gaussian-summary representation
    ``(emission_rate, emission_std)`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import jax.numpy as jnp
import numpy as np

from plumax.lagrangian.inversion import _is_traced

if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax

__all__ = [
    "EmissionCatalog",
    "EmissionEvent",
    "PerEventPosterior",
    "event_from_posterior",
]


class PerEventPosterior(Protocol):
    """Structural type for any Tier II-IV per-event posterior.

    Every per-event posterior consumed by Tier V exposes a scalar
    posterior mean and standard deviation of the emission rate ``Q``.
    :class:`plumax.lagrangian.inversion.GaussianPosterior`,
    :class:`plumax.lagrangian.inversion.LognormalPosterior` and
    :class:`plumax.coupled.fusion.FusionPosterior` all satisfy this.
    """

    @property
    def emission_rate(self) -> jax.Array:
        """Posterior-mean emission rate ``Q`` [kg/s]."""
        ...

    @property
    def emission_std(self) -> jax.Array:
        """Posterior standard deviation of ``Q`` [kg/s]."""
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
    """

    emission_rate: float
    emission_std: float
    x: float
    y: float
    time: float
    tier: str
    instrument: str | None = None

    def __post_init__(self) -> None:
        """Validate that ``emission_std`` is non-negative.

        The check is skipped when ``emission_std`` is a traced JAX value
        (under ``jit`` / ``grad`` / ``vmap``) because ``bool(...)`` on a
        tracer raises ``TracerError``.
        """
        if _is_traced(self.emission_std):
            return
        if float(self.emission_std) < 0.0:
            msg = f"emission_std must be non-negative, got {self.emission_std!r}"
            raise ValueError(msg)


def event_from_posterior(
    posterior: PerEventPosterior,
    *,
    x: float,
    y: float,
    time: float,
    tier: str,
    instrument: str | None = None,
) -> EmissionEvent:
    """Build an :class:`EmissionEvent` from any Tier II-IV posterior.

    The posterior is duck-typed: only its ``emission_rate`` and
    ``emission_std`` attributes are read, so the same adapter works for
    :class:`~plumax.lagrangian.inversion.GaussianPosterior`,
    :class:`~plumax.lagrangian.inversion.LognormalPosterior` and
    :class:`~plumax.coupled.fusion.FusionPosterior`.  Vector-valued
    posteriors (``n_src > 1``) are reduced to a scalar by taking the
    first source; build one event per source upstream for multi-source
    overpasses.

    Args:
        posterior: A per-event posterior exposing ``emission_rate`` and
            ``emission_std``.
        x: Source east-west location [m].
        y: Source north-south location [m].
        time: Detection / overpass time [s].
        tier: Originating tier label.
        instrument: Instrument identifier, or ``None``.

    Returns:
        The corresponding :class:`EmissionEvent`.
    """
    rate = jnp.asarray(posterior.emission_rate).reshape(-1)[0]
    std = jnp.asarray(posterior.emission_std).reshape(-1)[0]
    return EmissionEvent(
        emission_rate=float(rate),
        emission_std=float(std),
        x=float(x),
        y=float(y),
        time=float(time),
        tier=tier,
        instrument=instrument,
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
        return np.asarray([e.emission_rate for e in self.events], dtype=float)

    @property
    def emission_std(self) -> np.ndarray:
        """Per-event posterior standard deviations ``(n_events,)`` [kg/s]."""
        return np.asarray([e.emission_std for e in self.events], dtype=float)

    @property
    def x(self) -> np.ndarray:
        """Per-event source east-west locations ``(n_events,)`` [m]."""
        return np.asarray([e.x for e in self.events], dtype=float)

    @property
    def y(self) -> np.ndarray:
        """Per-event source north-south locations ``(n_events,)`` [m]."""
        return np.asarray([e.y for e in self.events], dtype=float)

    @property
    def time(self) -> np.ndarray:
        """Per-event detection times ``(n_events,)`` [s]."""
        return np.asarray([e.time for e in self.events], dtype=float)

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
        m = self.emission_rate
        s = self.emission_std
        if np.any(m <= 0.0):
            msg = "log_moments requires strictly positive emission_rate for every event"
            raise ValueError(msg)
        cv2 = (s / m) ** 2
        log_var = np.log1p(cv2)
        log_std = np.sqrt(log_var)
        log_mean = np.log(m) - 0.5 * log_var
        return log_mean, log_std
