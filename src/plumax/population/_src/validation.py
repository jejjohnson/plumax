"""Calibration harnesses: simulation-based calibration and the per-event-prior swap.

* :func:`run_sbc` is simulation-based calibration for any prior, simulator and fit:
  draw θ* from the prior, simulate, fit, and rank θ* among the posterior draws. When the
  fit is right the ranks are uniform (Talts et al. 2018). It knows nothing of the
  population model, so other tiers can reuse it.
* :func:`prior_swap_test` is 06's per-event-prior swap-out: the population posterior
  should not move beyond importance-sampling noise when the per-event prior changes.
* :func:`identifiability_stress` is 06b's stress test of what the model cannot
  disentangle: corr(λ₀, Q₅₀) in a confounded and a separated regime.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from scipy.stats import kstest


if TYPE_CHECKING:
    from plumax.population.catalog import EmissionCatalog


@dataclass(frozen=True)
class SBCResult:
    """Ranks of the true parameters among their posterior draws.

    Attributes:
        ranks: r = #{s : θ⁽ˢ⁾ < θ*}, in 0…L, per simulation and parameter.
        ks_pvalues: KS p-value of the ranks against uniform, per parameter.
        param_names: The parameters, in the order of the columns.
        n_draws: L, the posterior draws per fit.
    """

    ranks: Int[Array, "sample param"] = field(repr=False)
    ks_pvalues: Float[Array, " param"]
    param_names: tuple[str, ...]
    n_draws: int


def rank_uniformity_pvalues(
    ranks: Int[Array, "sample param"], n_draws: int, *, seed: int = 0
) -> Float[Array, " param"]:
    """KS p-value of SBC ranks against uniform, with ties broken at random.

    u = (r + U) / (L + 1), U ~ Uniform(0, 1), is exactly uniform when r is uniform on
    0…L, so the discreteness of the ranks costs the KS test nothing.

    Args:
        ranks: r per simulation and parameter.
        n_draws: L.
        seed: Seed of the tie-breaking U, for a reproducible p-value.

    Returns:
        The p-value per parameter.

    Examples:
        Evenly spread ranks pass, and a pile at one end does not:

        >>> import jax.numpy as jnp
        >>> even = jnp.arange(200).reshape(-1, 1) % 100
        >>> bool(rank_uniformity_pvalues(even, 99)[0] > 0.01)
        True
        >>> bool(rank_uniformity_pvalues(jnp.zeros((200, 1), dtype=int), 99)[0] < 0.01)
        True
    """
    host_ranks = np.asarray(ranks, dtype=float)
    jitter = np.random.default_rng(seed).uniform(size=host_ranks.shape)
    # u = (r + U) / (L + 1)
    positions = (host_ranks + jitter) / (n_draws + 1)
    pvalues = [kstest(column, "uniform").pvalue for column in positions.T]
    return jnp.asarray(pvalues)


def run_sbc(
    *,
    prior_sample: Callable[[Array], Mapping[str, Array]],
    simulate: Callable[[Array, Mapping[str, Array]], Any],
    fit: Callable[[Array, Any], Mapping[str, Array]],
    n_sims: int,
    key: Array,
    batch_size: int | None = None,
) -> SBCResult:
    """Simulation-based calibration of a fit, compiled once.

    θ* ~ p(θ), y ~ p(y | θ*), θ⁽ˢ⁾ ~ p(θ | y), r = #{s : θ⁽ˢ⁾ < θ*}, uniform on 0…L.

    The three callables run under ``jax.lax.map``, so they must be traceable, and a
    catalog must keep one shape from simulation to simulation (a padded buffer). Only
    the ranks are kept, so memory does not grow with the number of draws. The draws
    should be close to independent: correlated draws bias r even for a correct fit.

    Args:
        prior_sample: ``key → {name: θ*}``, scalar parameters by name.
        simulate: ``(key, θ*) → data``.
        fit: ``(key, data) → {name: θ⁽ˢ⁾}``, with the prior's names and L draws each.
        n_sims: Simulations.
        key: PRNG key.
        batch_size: ``jax.lax.map`` vmaps this many simulations together. The default
            runs them one at a time, which suits NUTS: under vmap every simulation
            pays for the deepest tree of any in the batch at every step.

    Returns:
        The :class:`SBCResult`.

    Raises:
        ValueError: For ``n_sims`` below 1.

    Examples:
        A conjugate Gaussian fit is calibrated by construction:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> def prior_sample(key):
        ...     return {"mean": jax.random.normal(key)}
        >>> def simulate(key, truth):
        ...     return truth["mean"] + jax.random.normal(key)
        >>> def fit(key, y):
        ...     return {"mean": 0.5 * y + jnp.sqrt(0.5) * jax.random.normal(key, (99,))}
        >>> result = run_sbc(
        ...     prior_sample=prior_sample,
        ...     simulate=simulate,
        ...     fit=fit,
        ...     n_sims=300,
        ...     key=jax.random.PRNGKey(0),
        ...     batch_size=300,
        ... )
        >>> result.param_names, result.n_draws
        (('mean',), 99)
        >>> bool(result.ks_pvalues[0] > 0.01)
        True
    """
    if n_sims < 1:
        raise ValueError(f"run_sbc: n_sims must be ≥ 1, got {n_sims}.")
    prior_key, data_key, fit_key = jax.random.split(jax.random.fold_in(key, 0), 3)
    template = jax.eval_shape(prior_sample, prior_key)
    names = tuple(sorted(template))
    data_shape = jax.eval_shape(simulate, data_key, prior_sample(prior_key))
    draws_shape = jax.eval_shape(fit, fit_key, data_shape)
    n_draws = int(draws_shape[names[0]].shape[0])

    def one(simulation_key: Array) -> Int[Array, " param"]:
        prior_key, data_key, fit_key = jax.random.split(simulation_key, 3)
        truth = prior_sample(prior_key)
        data = simulate(data_key, truth)
        draws = fit(fit_key, data)
        # r = #{s : θ⁽ˢ⁾ < θ*}
        return jnp.stack([jnp.sum(draws[name] < truth[name]) for name in names])

    ranks = jax.lax.map(one, jax.random.split(key, n_sims), batch_size=batch_size)
    return SBCResult(
        ranks=ranks,
        ks_pvalues=rank_uniformity_pvalues(ranks, n_draws),
        param_names=names,
        n_draws=n_draws,
    )


def _swap_prior(
    catalog: EmissionCatalog,
    log_prior: Callable[[Array], Array],
    rng: np.random.Generator,
) -> EmissionCatalog:
    from plumax._src.errors import PayloadError
    from plumax.population.catalog import EmissionCatalog

    events = []
    for event in catalog.events:
        if event.q_samples is None or event.prior_logpdf is None:
            raise PayloadError(
                "prior_swap_test: every event needs q_samples and prior_logpdf."
            )
        draws = np.ravel(np.asarray(event.q_samples, dtype=float))
        # w ∝ π_new(q) / π_old(q): posterior draws under the new prior, without the tier
        log_ratio = log_prior(jnp.asarray(draws)) - event.prior_logpdf(
            jnp.asarray(draws)
        )
        log_ratio = np.asarray(log_ratio)
        weights = np.exp(log_ratio - log_ratio.max())
        chosen = rng.choice(draws.size, size=draws.size, p=weights / weights.sum())
        swapped = draws[chosen]
        # the summaries too: the GAUSSIAN and POINT regimes read them, not the draws
        events.append(
            dataclasses.replace(
                event,
                emission_rate=float(swapped.mean()),
                emission_std=float(swapped.std()),
                q_samples=jnp.asarray(swapped),
                prior_logpdf=log_prior,
            )
        )
    return EmissionCatalog.from_events(events)


def prior_swap_test(
    catalog: EmissionCatalog,
    *,
    priors: tuple[Callable[[Array], Array], Callable[[Array], Array]],
    fit: Callable[..., Any] | None = None,
    n_replicates: int = 8,
    seed: int = 0,
    **fit_kwargs: Any,
) -> float:
    """How far the population posterior moves between per-event priors, in IS σ.

    z = max_θ |m̄_A(θ) − m̄_B(θ)| / √((s²_A(θ) + s²_B(θ)) / R), with m̄ and s the mean and
    standard deviation of the posterior mean over R replicate fits. A replicate
    resamples each event's draws by π_new / π_old, the same posterior as rerunning the
    tier under π_new, then refits; the spread s is therefore the importance-sampling
    noise, with the sampler's. A correct importance correction divides the prior out, so
    z stays below about 3.

    Args:
        catalog: Events with ``q_samples`` and ``prior_logpdf``.
        priors: Two per-event log-priors log π(q).
        fit: ``(catalog, seed=…, **fit_kwargs) → posterior`` with a
            ``posterior_mean()``; default
            :func:`~plumax.population.fitting.fit_size_distribution_iw`.
        n_replicates: R ≥ 2 per prior.
        seed: Seed of the resampling and of the fits.
        **fit_kwargs: Passed to ``fit``.

    Returns:
        z, the largest over the fit's parameters.

    Raises:
        PayloadError: When an event lacks its draws or prior.
        ValueError: For fewer than 2 replicates.
    """
    if n_replicates < 2:
        raise ValueError(f"prior_swap_test: need R ≥ 2, got {n_replicates}.")
    if fit is None:
        from plumax.population.fitting import fit_size_distribution_iw as fit
    rng = np.random.default_rng(seed)
    means: list[dict[str, list[float]]] = []
    for prior_index, log_prior in enumerate(priors):
        per_parameter: dict[str, list[float]] = {}
        for replicate in range(n_replicates):
            swapped = _swap_prior(catalog, log_prior, rng)
            replicate_seed = seed + 1 + prior_index * n_replicates + replicate
            posterior = fit(swapped, seed=replicate_seed, **fit_kwargs)
            for name, value in posterior.posterior_mean().items():
                per_parameter.setdefault(name, []).append(float(value))
        means.append(per_parameter)
    statistics = []
    for name in means[0]:
        first, second = (np.asarray(means[index][name]) for index in range(2))
        # √((s²_A + s²_B) / R)
        noise = np.sqrt((first.var(ddof=1) + second.var(ddof=1)) / n_replicates)
        statistics.append(abs(first.mean() - second.mean()) / noise)
    return float(max(statistics))


def _regime_correlation(
    regime: tuple[float, float],
    *,
    key: Array,
    window: tuple[float, float],
    mu: float,
    sigma: float,
    slope: float,
    q50_std: float,
    max_ground: int,
    num_warmup: int,
    num_samples: int,
) -> float:
    from numpyro.infer import MCMC, NUTS

    from plumax.population._src.fitting_iw import tmtpp_iw_model
    from plumax.population._src.importance import EventEvidence
    from plumax.population._src.marks import LognormalMark
    from plumax.population._src.pod_union import InstrumentPOD, log_pod_union
    from plumax.population._src.processes import padded_event_buffer

    rate, q50 = regime
    pod = InstrumentPOD("stress", "log_logistic", q50, q50_std, slope)
    count_key, mark_key, keep_key, mcmc_key = jax.random.split(key, 4)
    span = window[1] - window[0]
    n_ground = int(jax.random.poisson(count_key, rate * span))
    if n_ground > max_ground:
        raise ValueError(
            f"identifiability_stress: {n_ground} ground events exceed {max_ground}."
        )
    # Q ~ LogNormal(μ, σ²)
    marks = jnp.exp(mu + sigma * jax.random.normal(mark_key, (n_ground,)))
    # keep with probability P_d(Q), at the true Q₅₀
    log_uniform = jnp.log(jax.random.uniform(keep_key, (n_ground,)))
    detected = marks[log_uniform < log_pod_union([pod])(marks)]
    n_detected = int(detected.shape[0])
    # a constant λ's likelihood depends on the times only through their count
    times = np.linspace(window[0], window[1], n_detected + 2)[1:-1]
    event_times, mask = padded_event_buffer(times, window)
    mcmc = MCMC(
        NUTS(tmtpp_iw_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    mcmc.run(
        mcmc_key,
        event_times,
        mask,
        EventEvidence(jnp.zeros((n_detected, 1)), detected[:, None]),
        mark=LognormalMark(),
        intensity="constant",
        window=window,
        pods=[pod],
        support=(1e-6, 1e6),
    )
    samples = mcmc.get_samples()
    draws = jnp.stack([samples["lambda_0"], samples["pod.stress.q50"]])
    # corr(λ₀, Q₅₀)
    return float(jnp.corrcoef(draws)[0, 1])


def identifiability_stress(
    *,
    lam_high_pod_low: tuple[float, float],
    lam_low_pod_high: tuple[float, float],
    key: Array,
    window: tuple[float, float] = (0.0, 100.0),
    mu: float = 0.0,
    sigma: float = 1.0,
    slope: float = 2.0,
    q50_std: float = 0.5,
    max_ground: int = 5000,
    num_warmup: int = 500,
    num_samples: int = 500,
) -> tuple[float, float]:
    """Posterior corr(λ₀, Q₅₀) in a confounded and a separated regime.

    With a high rate seen through a high Q₅₀, more events and a higher threshold explain
    the same catalog, and the correlation is strong. With a low Q₅₀ almost every event
    is detected, N pins λ₀ alone, and the correlation is weak. 06b's targets are above
    0.5 and below 0.2.

    Args:
        lam_high_pod_low: (λ₀ [day⁻¹], Q₅₀) of the confounded regime, with Q₅₀ above
            most of the marks.
        lam_low_pod_high: (λ₀, Q₅₀) of the separated regime, with Q₅₀ below them.
        key: PRNG key.
        window: The observed window [days].
        mu: Log-median μ of the lognormal marks.
        sigma: Log-scale σ of the marks.
        slope: k of the Hill curve, fixed.
        q50_std: Log-scale spread of the Q₅₀ prior, centred on the truth.
        max_ground: Refuses a regime with more ground events than this.
        num_warmup: NUTS warm-up steps.
        num_samples: NUTS draws.

    Returns:
        ``(confounded, separated)``, each corr(λ₀, Q₅₀).

    Raises:
        ImportError: Without the ``population`` extra.
        ValueError: When a regime would simulate more than ``max_ground`` events.
    """
    confounded_key, separated_key = jax.random.split(key)
    settings: dict[str, Any] = {
        "window": window,
        "mu": mu,
        "sigma": sigma,
        "slope": slope,
        "q50_std": q50_std,
        "max_ground": max_ground,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
    }
    return (
        _regime_correlation(lam_high_pod_low, key=confounded_key, **settings),
        _regime_correlation(lam_low_pod_high, key=separated_key, **settings),
    )
