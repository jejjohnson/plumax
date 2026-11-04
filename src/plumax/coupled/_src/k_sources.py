"""Masked-K sources in the coupled inversion: up to K_max plumes across instruments.

The forward of instrument i sums the active slots,

    y_i = Σ_k a_k Q_k r_i(x_k, y_k) + c_bg + ε_i,

with r_i the unit-emission column response of the slot's location through the
instrument's averaging kernel. The prior, the Gibbs-within-NUTS runner and the
posterior carrier are Tier I's (:mod:`plumax.gauss_plume.multi_source`); only the
forward and the per-instrument likelihood differ.

The split follows #186. The posterior on the active count K stays a NumPyro model
here, since vardax leaves MCMC to user code. For a given active set the MAP and the
Laplace posterior are ``vardax`` through :func:`~plumax.coupled.solve_coupled`, on the
problem from :func:`build_masked_k_problem`.

Promotion to RJMCMC (06, Tier IV design): masked-K caps K at K_max. When more than 5 %
of overpasses saturate, :func:`saturation_rate` above 0.05, the cap is binding and the
trans-dimensional model is due.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from plumax.coupled._src.inversion import (
    CoupledPrior,
    CoupledProblem,
    plume_response,
)
from plumax.coupled.backends import TransportBackend
from plumax.coupled.forward import CoupledForward, PlumeSource
from plumax.coupled.instrument import Instrument
from plumax.gauss_plume.multi_source import MultiSourcePosterior


@dataclasses.dataclass(frozen=True)
class MaskedKPrior:
    """The slot prior: how many slots, how likely each is on, and its Q_k.

    a_k ~ Bernoulli(p_active), Q_k ~ LogNormal moment-matched to the real-scale mean
    and standard deviation, as :func:`plumax.gauss_plume.inference.multi_source_plume_model`.

    Attributes:
        k_max: K_max ≥ 1 (the design's v1 cap is 10).
        prior_active: p_active in (0, 1]; 1 keeps every slot on, with no discrete
            sites.
        emission_mean: Real-scale prior mean of each Q_k [kg/s].
        emission_std: Real-scale prior standard deviation of each Q_k [kg/s].
        background_std: HalfNormal scale of the shared background column.

    Raises:
        ValueError: On a parameter out of range.
    """

    k_max: int = 10
    prior_active: float = 0.3
    emission_mean: float = 0.1
    emission_std: float = 0.05
    background_std: float = 1e-6

    def __post_init__(self) -> None:
        if self.k_max < 1:
            raise ValueError(f"MaskedKPrior: k_max must be ≥ 1, got {self.k_max}.")
        if not 0.0 < self.prior_active <= 1.0:
            raise ValueError(
                f"MaskedKPrior: prior_active must be in (0, 1], got {self.prior_active}."
            )
        if not (
            self.emission_mean > 0.0
            and self.emission_std > 0.0
            and self.background_std > 0.0
        ):
            raise ValueError("MaskedKPrior: the prior scales must be > 0.")


def slot_responses(
    transport: PlumeSource | TransportBackend,
    instruments: Sequence[Instrument],
    candidate_locations: Float[Array, "source 2"] | None = None,
) -> list[Array]:
    """Unit-emission column of every slot, as one design matrix per instrument.

    R_i[n, k] = r_i(x_k, y_k) at receptor n. The Tier I plume is rerun at each
    candidate. Any other backend has its emission parameters as the slots already:
    its ``column_response`` must return the ``(obs, source)`` matrix.

    Args:
        transport: The transport backend.
        instruments: The observing instruments.
        candidate_locations: ``(x, y)`` per slot [m], for a ``PlumeSource``.

    Returns:
        One ``(obs, source)`` matrix per instrument.

    Raises:
        ValueError: For a ``PlumeSource`` without candidates, or a backend with
            candidates, or whose response is not a matrix.
    """
    if isinstance(transport, PlumeSource):
        if candidate_locations is None:
            raise ValueError(
                "slot_responses: a PlumeSource needs candidate_locations (x, y)."
            )
        locations = jnp.asarray(candidate_locations, dtype=float)
        wind = jnp.asarray([transport.wind_speed, transport.wind_direction])

        def design(instrument: Instrument) -> Array:
            response = jax.vmap(plume_response, (None, None, 0, None), 1)
            return response(transport, instrument, locations, wind)

        return [design(instrument) for instrument in instruments]
    if candidate_locations is not None:
        raise ValueError(
            "slot_responses: only a PlumeSource reruns at candidate_locations; another "
            "backend's emission parameters are its slots."
        )
    matrices = [transport.column_response(instrument) for instrument in instruments]
    if any(jnp.ndim(matrix) != 2 for matrix in matrices):
        raise ValueError(
            "slot_responses: the backend must return one column per slot, (obs, source)."
        )
    return [jnp.asarray(matrix, dtype=float) for matrix in matrices]


def masked_k_coupled_model(
    observations: Mapping[str, Array] | None,
    *,
    instruments: Sequence[Instrument],
    transport: PlumeSource | TransportBackend,
    k_max: int,
    candidate_locations: Float[Array, "source 2"] | None = None,
    source_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    prior_active: float = 0.3,
    emission_mean: float = 0.1,
    emission_std: float = 0.05,
    background_std: float = 1e-6,
    fixed_responses: Sequence[Array] | None = None,
) -> None:
    """NumPyro masked-K model over every instrument's columns.

    y_i ~ Normal(Σ_k a_k Q_k r_i(x_k, y_k) + c_bg, Σ_ε,i), with Σ_ε,i and the mask of
    each :class:`~plumax.coupled.Instrument`, under Tier I's slot prior. With
    ``source_bounds`` the slots move, so a ``PlumeSource`` is rerun at each draw.

    Args:
        observations: Column observations by instrument, or ``None`` to predict.
        instruments: The observing instruments.
        transport: The transport backend.
        k_max: K_max.
        candidate_locations: Fixed ``(x, y)`` per slot [m].
        source_bounds: Box of free, x-ordered slots; needs a ``PlumeSource``.
        prior_active: p_active.
        emission_mean: Real-scale prior mean of each Q_k [kg/s].
        emission_std: Real-scale prior standard deviation of each Q_k [kg/s].
        background_std: HalfNormal scale of the background column.
        fixed_responses: :func:`slot_responses`, when the slots are fixed; passed in so
            they are computed once rather than at every model evaluation.
    """
    import numpyro
    import numpyro.distributions as dist

    from plumax.gauss_plume.inference import _masked_source_prior

    sources, background = _masked_source_prior(
        k_max=k_max,
        candidate_locations=candidate_locations,
        source_bounds=source_bounds,
        prior_emission_rate_mean=emission_mean,
        prior_emission_rate_std=emission_std,
        prior_active=prior_active,
        background_prior_std=background_std,
    )
    # a_k Q_k
    active_rates = sources.active * sources.emission_rate
    for index, instrument in enumerate(instruments):
        if fixed_responses is not None:
            design = fixed_responses[index]
        else:
            locations = jnp.stack([sources.x, sources.y], axis=-1)
            design = slot_responses(transport, [instrument], locations)[0]
        # Σ_k a_k Q_k r_i(x_k, y_k) + c_bg
        predicted = design @ active_rates + background
        noise_std = jnp.sqrt(instrument.observation_variance())
        keep = instrument.observation_mask() > 0.0
        likelihood = dist.Normal(predicted, noise_std).mask(keep)
        observed = None
        if observations is not None:
            # masked pixels may hold anything, and the mask drops them
            observed = jnp.where(keep, jnp.asarray(observations[instrument.name]), 0.0)
        numpyro.sample(f"obs_{instrument.name}", likelihood.to_event(1), obs=observed)


def infer_masked_k_sources(
    observations: Mapping[str, Array],
    *,
    instruments: Sequence[Instrument],
    transport: PlumeSource | TransportBackend,
    prior: MaskedKPrior,
    candidate_locations: Float[Array, "source 2"] | None = None,
    source_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    num_warmup: int = 500,
    num_samples: int = 1000,
    seed: int = 0,
) -> MultiSourcePosterior:
    """Posterior over the active count and the per-slot Q_k, from every instrument.

    Tier I's runner on :func:`masked_k_coupled_model`: NUTS for the continuous sites,
    a Gibbs step for the a_k.

    Args:
        observations: Column observations by instrument.
        instruments: The observing instruments.
        transport: The transport backend.
        prior: The slot prior.
        candidate_locations: Fixed ``(x, y)`` per slot [m].
        source_bounds: Box of free slots, instead of candidates.
        num_warmup: NUTS warm-up steps.
        num_samples: Posterior draws.
        seed: PRNG seed.

    Returns:
        The posterior: ``count_probabilities()`` is P(K | y), and ``per_source(k)``
        the slot's summary, which ``population.event_from_posterior`` accepts.

    Raises:
        ValueError: On missing observations or an invalid slot layout.
    """
    from plumax.gauss_plume.inference import _check_multi_source_layout, _run_masked_k

    names = {instrument.name for instrument in instruments}
    if set(observations) != names:
        raise ValueError(
            f"infer_masked_k_sources: observations for {sorted(observations)} but "
            f"the instruments are {sorted(names)}."
        )
    fixed_slots = source_bounds is None
    is_plume = isinstance(transport, PlumeSource)
    if is_plume or not fixed_slots:
        _check_multi_source_layout(
            prior.k_max, candidate_locations, source_bounds, prior.prior_active
        )
    if not fixed_slots and not is_plume:
        raise ValueError(
            "infer_masked_k_sources: free slots rerun the Tier I plume and need a "
            "PlumeSource."
        )
    fixed_responses = None
    if fixed_slots:
        fixed_responses = slot_responses(transport, instruments, candidate_locations)
        widths = {matrix.shape[-1] for matrix in fixed_responses}
        if widths != {prior.k_max}:
            raise ValueError(
                f"infer_masked_k_sources: the responses have {widths} slots but "
                f"k_max is {prior.k_max}."
            )
    layout = candidate_locations
    if layout is None and fixed_slots:
        # a backend's slots have no coordinates of their own
        layout = jnp.full((prior.k_max, 2), jnp.nan)
    return _run_masked_k(
        masked_k_coupled_model,
        observations,
        k_max=prior.k_max,
        candidate_locations=layout,
        prior_active=prior.prior_active,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        seed=seed,
        progress_bar=False,
        instruments=instruments,
        transport=transport,
        source_bounds=source_bounds,
        emission_mean=prior.emission_mean,
        emission_std=prior.emission_std,
        background_std=prior.background_std,
        fixed_responses=fixed_responses,
    )


def build_masked_k_problem(
    instruments: Sequence[Instrument],
    *,
    transport: PlumeSource | TransportBackend,
    prior: MaskedKPrior,
    observations: Mapping[str, Array],
    active: Bool[Array, " source"],
    candidate_locations: Float[Array, "source 2"] | None = None,
) -> CoupledProblem:
    """The coupled problem of one active set, for the MAP and Laplace path.

    y_i = Σ_{k active} Q_k r_i(x_k, y_k) + c_bg + ε_i over log Q_k, under the same
    lognormal prior as the slots. Hand it to :func:`~plumax.coupled.solve_coupled`,
    which solves on vardax. This is the operational path; the active set itself comes
    from :func:`infer_masked_k_sources` or from the caller.

    Args:
        instruments: The observing instruments.
        transport: The transport backend.
        prior: The slot prior.
        observations: Observations by instrument.
        active: a_k, with at least one slot on.
        candidate_locations: Fixed ``(x, y)`` per slot [m], for a ``PlumeSource``.

    Returns:
        The problem, with the ``log_emission_rate`` block over the active slots.

    Raises:
        ValueError: When ``active`` does not match the slots or has none on.
    """
    mask = np.asarray(active, dtype=bool)
    matrices = slot_responses(transport, instruments, candidate_locations)
    if mask.shape != (prior.k_max,) or not mask.any():
        raise ValueError(
            f"build_masked_k_problem: active must have shape ({prior.k_max},) with "
            f"at least one slot on, got {mask.tolist()}."
        )
    from plumax.gauss_plume.inference import _lognormal_from_moments

    log_mean, log_std = _lognormal_from_moments(prior.emission_mean, prior.emission_std)
    n_active = int(mask.sum())
    forward = CoupledForward(
        source=transport,
        instruments=tuple(instruments),
        responses=tuple(matrix[:, mask] for matrix in matrices),
    )
    coupled_prior = CoupledPrior.from_std(
        {"log_emission_rate": log_mean * jnp.ones(n_active)},
        {"log_emission_rate": log_std * jnp.ones(n_active)},
    )
    return CoupledProblem(
        forward=forward, prior=coupled_prior, observations=dict(observations)
    )


def saturation_rate(posteriors: Sequence[MultiSourcePosterior]) -> float:
    """Fraction of overpasses whose most probable active count is K_max.

    K̂ = argmax_k P(K = k | y), rate = #{K̂ = K_max} / N. Above 0.05 the cap is binding
    and the RJMCMC model is due (see the module docstring).

    Args:
        posteriors: One masked-K posterior per overpass.

    Returns:
        The saturated fraction.

    Raises:
        ValueError: With no posteriors.

    Examples:
        One overpass at the cap, one below it:

        >>> import numpy as np
        >>> def posterior(active):
        ...     shape = np.shape(active)
        ...     return MultiSourcePosterior(
        ...         emission_rate=np.ones(shape),
        ...         x=np.zeros(shape),
        ...         y=np.zeros(shape),
        ...         active=np.asarray(active),
        ...         background=np.zeros(shape[0]),
        ...     )
        >>> full = posterior([[True, True], [True, True]])
        >>> partial = posterior([[True, False], [True, False]])
        >>> saturation_rate([full, partial])
        0.5
    """
    if not posteriors:
        raise ValueError("saturation_rate: no posteriors given.")
    saturated = [
        int(np.argmax(posterior.count_probabilities())) == posterior.k_max
        for posterior in posteriors
    ]
    return float(np.mean(saturated))
