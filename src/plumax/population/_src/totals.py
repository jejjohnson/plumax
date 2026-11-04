"""Basin total emission from a TMTPP posterior, split by detection.

The inventory product of Tier V (06d): the un-thinned total mass over the observed
windows, and how much of it the satellites are expected to have missed,

    M_total = Λ E_f[Q] D,   M_missing = Λ E_f[Q (1 − P_d(Q))] D,

with Λ = ∫ λ(t) dt the expected number of events and D the mean event duration. Each
is evaluated per posterior draw, so the output is posterior samples.

Mass and mass rate. A flux Q times a duration D is the mass of one event, so
the mass is in kg whenever D is in the reciprocal of Q's time unit: hours for
kg hr⁻¹, seconds for kg s⁻¹. The rate is the mass over T_eff, the length of the union
of the observed windows. Λ integrates over that same union, so the rate is the mean
emission rate while observed, not an extrapolation to an unobserved part of the year.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from plumax.population._src.fitting_iw import TMTPPPosterior
from plumax.population._src.integrate import log_simpson
from plumax.population._src.processes import (
    NODES_PER_DAY,
    LogIntensity,
    _quadrature_points,
)


SECONDS_PER_DAY = 86_400.0
#: The Julian year [s], for rates in Tg yr⁻¹.
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
KG_PER_TG = 1e9


@dataclass(frozen=True)
class TotalEmission:
    """Posterior samples of a total emission, split into detected and missing mass.

    ``samples_kg = detected_kg + missing_kg`` draw by draw.

    Attributes:
        samples_kg: M_total, the un-thinned total mass [kg].
        detected_kg: Λ E_f[Q P_d(Q)] D, the mass the model expects to be detected [kg].
        missing_kg: Λ E_f[Q (1 − P_d(Q))] D, the inferred missing mass [kg].
        t_effective_s: T_eff, the length of the union of the observed windows [s].
    """

    samples_kg: Float[Array, " member"]
    detected_kg: Float[Array, " member"]
    missing_kg: Float[Array, " member"]
    t_effective_s: float

    def rate_tg_per_yr(self) -> Float[Array, " member"]:
        """Mean emission rate while observed, M_total / T_eff [Tg yr⁻¹].

        Returns:
            Posterior samples of the rate, over a Julian year.
        """
        # M_total / T_eff [kg s⁻¹]
        rate_kg_per_s = self.samples_kg / self.t_effective_s
        return rate_kg_per_s * SECONDS_PER_YEAR / KG_PER_TG


def _merge_windows(
    horizon: tuple[float, float] | Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    # a lone (t₀, t₁) of numbers, or a sequence of them
    windows = list(horizon) if isinstance(horizon[0], Sequence) else [horizon]
    merged: list[tuple[float, float]] = []
    for start, stop in sorted((float(a), float(b)) for a, b in windows):
        if not stop > start:
            raise ValueError(
                f"total_emission: a window needs t₁ > t₀, got ({start}, {stop})."
            )
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _stratum_emission(
    posterior: TMTPPPosterior,
    *,
    windows: list[tuple[float, float]],
    event_duration: float,
    log_pod: Callable[[Array], Array] | None,
    q_min: float,
    q_max: float,
    n_nodes: int,
    nodes_per_day: int,
) -> tuple[Float[Array, " member"], Float[Array, " member"]]:
    def member(draw: dict[str, Array]) -> tuple[Array, Array]:
        kernel = posterior.kernel(draw)
        theta = posterior.theta(draw)

        # --- ground process ---
        # Λ = Σ_w ∫_w λ(t) dt
        expected_events = sum(
            LogIntensity(
                kernel,
                n_integration_points=_quadrature_points(stop - start, nodes_per_day),
            ).integrate(start, stop)
            for start, stop in windows
        )

        # --- marks ---
        # E_f[Q] = ∫ q f(q | θ) dq, a partial moment on [q_min, q_max]
        mean_rate = posterior.mark.truncated_mean(q_min, q_max, **theta)
        if log_pod is None:
            missing_rate = jnp.zeros_like(mean_rate)
        else:

            def log_integrand(q: Array) -> Array:
                # log P_d is near 0 where detection is certain, so 1 − P_d needs expm1
                log_miss = jnp.log(-jnp.expm1(log_pod(q)))
                # log q + log f(q | θ) + log(1 − P_d(q))
                return jnp.log(q) + posterior.mark.log_prob(q, **theta) + log_miss

            # E_f[Q (1 − P_d(Q))], in log space: a heavy tail underflows in linear Q
            missing_rate = jnp.exp(log_simpson(log_integrand, q_min, q_max, n=n_nodes))

        # M_total = Λ E_f[Q] D
        total = expected_events * mean_rate * event_duration
        # M_missing = Λ E_f[Q (1 − P_d(Q))] D
        missing = expected_events * missing_rate * event_duration
        return total, missing

    draws = {name: jnp.asarray(value) for name, value in posterior.samples.items()}
    return jax.vmap(member)(draws)


def total_emission(
    posterior: TMTPPPosterior | Mapping[str, TMTPPPosterior],
    *,
    horizon: tuple[float, float] | Sequence[tuple[float, float]],
    event_duration: float,
    log_pod: Callable[[Array], Array] | None,
    q_min: float,
    q_max: float,
    n_nodes: int = 2001,
    nodes_per_day: int = NODES_PER_DAY,
) -> TotalEmission:
    """Posterior samples of the basin total, with the inferred missing mass.

    M_total⁽ᵉ⁾ = Λ⁽ᵉ⁾ E_f⁽ᵉ⁾[Q] D,  M_missing⁽ᵉ⁾ = Λ⁽ᵉ⁾ E_f⁽ᵉ⁾[Q (1 − P_d(Q))] D.

    E_f[Q] is the mark's ``truncated_mean``, the partial first moment over
    [q_min, q_max], not a mean conditional on that range. See the module docstring for
    the mass and mass-rate convention.

    Algorithm:
        1. Merge ``horizon`` into disjoint observed windows; T_eff is their length.
        2. For each draw, Λ over those windows, from the kernel's closed form or
           Simpson's rule.
        3. For each draw, E_f[Q] in closed form and E_f[Q (1 − P_d)] by log-space
           Simpson over [q_min, q_max].
        4. Detected mass is the difference, floored at zero against round-off.
        5. Sum strata draw by draw: their posteriors are independent, so any pairing
           of their draws is a draw of the sum.

    Args:
        posterior: A TMTPP posterior, or one per stratum (the design's v1 spatial
            shortcut: facility classes that sum to the basin).
        horizon: The observed window ``(t₀, t₁)`` or a sequence of them [days]; they
            may overlap.
        event_duration: Mean event duration D > 0, in the reciprocal of Q's time unit.
        log_pod: log P_d on rates in the marks' units, as
            :func:`~plumax.population.fitting.tmtpp_iw_model` takes it, or ``None``
            for certain detection.
        q_min: Lower limit of the mark integrals, > 0.
        q_max: Upper limit of the mark integrals, finite and > ``q_min``.
        n_nodes: Simpson nodes of the missing-mass integral.
        nodes_per_day: Quadrature density of Λ for kernels without a closed form.
            Lower it for long windows when the kernel is smooth.

    Returns:
        The :class:`TotalEmission`.

    Raises:
        ValueError: For an empty window, invalid mark limits or duration, no strata,
            or strata with different numbers of draws.

    Examples:
        Two draws of a constant rate, where a tenth of the events are detected:

        >>> import math
        >>> import jax.numpy as jnp
        >>> from plumax.population.fitting import TMTPPPosterior
        >>> from plumax.population.marks import LognormalMark
        >>> samples = {
        ...     "lambda_0": jnp.array([1.0, 2.0]),
        ...     "mark.mu": jnp.zeros(2),
        ...     "mark.sigma": jnp.ones(2),
        ... }
        >>> posterior = TMTPPPosterior("constant", LognormalMark(), samples)
        >>> total = total_emission(
        ...     posterior,
        ...     horizon=(0.0, 10.0),
        ...     event_duration=2.0,
        ...     log_pod=lambda q: math.log(0.1) + jnp.zeros_like(q),
        ...     q_min=1e-6,
        ...     q_max=1e6,
        ... )
        >>> # M_total = λ T e^{1/2} D
        >>> expected = jnp.array([1.0, 2.0]) * 10.0 * math.exp(0.5) * 2.0
        >>> bool(jnp.allclose(total.samples_kg, expected))
        True
        >>> bool(jnp.allclose(total.detected_kg, 0.1 * expected, rtol=1e-4))
        True
        >>> total.t_effective_s
        864000.0
    """
    strata = (
        [posterior]
        if isinstance(posterior, TMTPPPosterior)
        else list(posterior.values())
    )
    if not strata:
        raise ValueError("total_emission: no strata given.")
    if len({stratum.num_samples for stratum in strata}) != 1:
        raise ValueError(
            "total_emission: strata need the same number of draws to sum, got "
            f"{[stratum.num_samples for stratum in strata]}."
        )
    if not 0.0 < q_min < q_max < float("inf"):
        raise ValueError(
            f"total_emission: need 0 < q_min < q_max < inf, got ({q_min}, {q_max})."
        )
    if not event_duration > 0.0:
        raise ValueError(
            f"total_emission: event_duration must be > 0, got {event_duration}."
        )
    windows = _merge_windows(horizon)
    parts = [
        _stratum_emission(
            stratum,
            windows=windows,
            event_duration=event_duration,
            log_pod=log_pod,
            q_min=q_min,
            q_max=q_max,
            n_nodes=n_nodes,
            nodes_per_day=nodes_per_day,
        )
        for stratum in strata
    ]
    # M_total = Σ_strata M_total,stratum
    samples_kg = sum(total for total, _ in parts)
    missing_kg = sum(missing for _, missing in parts)
    # T_eff = |∪_w w|
    effective_days = sum(stop - start for start, stop in windows)
    return TotalEmission(
        samples_kg=samples_kg,
        detected_kg=jnp.maximum(samples_kg - missing_kg, 0.0),
        missing_kg=missing_kg,
        t_effective_s=effective_days * SECONDS_PER_DAY,
    )
