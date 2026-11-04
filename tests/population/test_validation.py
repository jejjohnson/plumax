"""Tests for the population calibration harnesses."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax._src.errors import PayloadError
from plumax._src.scalar_emission import LognormalEmission
from plumax.gauss_plume.dispersion import get_dispersion_params
from plumax.gauss_plume.plume import plume_concentration_vmap
from plumax.population.catalog import EmissionCatalog, EmissionEvent
from plumax.population.importance import EventEvidence
from plumax.population.marks import LognormalMark
from plumax.population.pod import InstrumentPOD, log_pod_union
from plumax.population.validation import (
    identifiability_stress,
    prior_swap_test,
    rank_uniformity_pvalues,
    run_sbc,
)


pytest.importorskip("numpyro")
pytest.importorskip("xtremax")

from numpyro.infer import MCMC, NUTS

from plumax.population.fitting import tmtpp_iw_model


# --- the SBC designs, fixed before their first run ---
# KS p-values above SBC_MIN_PVALUE for every parameter is the issue's criterion.
SBC_MIN_PVALUE = 0.01
# The point regime runs the issue's 1000 simulations. The soft regime's draws cost
# several times more per simulation, so CI runs 200 of them.
SBC_POINT_SIMS = 1000
SBC_SOFT_SIMS = 200
SBC_CONTROL_SIMS = 100
WINDOW = (0.0, 30.0)
# Above the prior predictive ground count: λ₀ ~ HalfNormal(3) over 30 days
MAX_GROUND = 600
# Above the prior predictive detections, from the same ground count
MAX_DETECTED = 250
FIXED_POD = [InstrumentPOD("only", "log_logistic", 1.0, 0.0, 2.0)]
SBC_MARK = LognormalMark(mu_prior_scale=1.0, sigma_prior_scale=1.0)
SUPPORT = (1e-3, 1e3)
N_NODES = 401
# Thinned, because correlated draws bias the ranks of a correct fit
NUM_WARMUP = 200
NUM_DRAWS = 100
THINNING = 3

# --- the soft-observation layer ---
PER_EVENT_PRIOR = LognormalEmission(0.0, 2.0)
N_EVENT_DRAWS = 32
RECEPTORS_X = jnp.linspace(200.0, 1000.0, 12)
RECEPTORS_Y = jnp.linspace(-40.0, 40.0, 12)
NOISE_STD = 2e-6
LOG_Q_GRID = jnp.linspace(-7.0, 7.0, 512)


def _prior_sample(key: jax.Array) -> dict[str, jax.Array]:
    rate_key, mu_key, sigma_key = jax.random.split(key, 3)
    return {
        "lambda_0": 3.0 * jnp.abs(jax.random.normal(rate_key)),
        "mark.mu": jax.random.normal(mu_key),
        "mark.sigma": jnp.abs(jax.random.normal(sigma_key)),
    }


def _detected_marks(key: jax.Array, truth) -> tuple[jax.Array, jax.Array]:
    count_key, mark_key, keep_key = jax.random.split(key, 3)
    span = WINDOW[1] - WINDOW[0]
    n_ground = jax.random.poisson(count_key, truth["lambda_0"] * span)
    noise = jax.random.normal(mark_key, (MAX_GROUND,))
    marks = jnp.exp(truth["mark.mu"] + truth["mark.sigma"] * noise)
    log_uniform = jnp.log(jax.random.uniform(keep_key, (MAX_GROUND,)))
    detected = (jnp.arange(MAX_GROUND) < n_ground) & (
        log_uniform < log_pod_union(FIXED_POD)(marks)
    )
    order = jnp.argsort(~detected)[:MAX_DETECTED]
    return marks[order], detected[order]


def _buffer(mask: jax.Array) -> jax.Array:
    # a constant λ's likelihood depends on the times only through their count
    span = WINDOW[1] - WINDOW[0]
    return jnp.where(mask, 0.5 * span, span)


def _simulate_point(key: jax.Array, truth):
    marks, mask = _detected_marks(key, truth)
    return _buffer(mask), mask, jnp.where(mask, marks, 1.0)[:, None]


def _unit_response() -> jax.Array:
    return plume_concentration_vmap(
        RECEPTORS_X,
        RECEPTORS_Y,
        jnp.zeros_like(RECEPTORS_X),
        0.0,
        0.0,
        5.0,
        5.0,
        0.0,
        1.0,
        get_dispersion_params("C"),
    )


def _event_draws(key: jax.Array, rate: jax.Array, response: jax.Array) -> jax.Array:
    """Exact draws of Q | y for y = Q c₁ + ε under the per-event lognormal prior."""
    noise_key, draw_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, response.shape)
    obs = rate * response + NOISE_STD * noise
    log_posterior = _grid_log_posterior(obs, response)
    chosen = jax.random.categorical(draw_key, log_posterior, shape=(N_EVENT_DRAWS,))
    return jnp.exp(LOG_Q_GRID)[chosen]


@pytest.mark.slow
def test_event_draws_match_the_gauss_plume_posterior():
    # the SBC's per-event layer against Tier I's own NUTS, on one event
    from plumax.gauss_plume.inference import infer_emission_rate

    response = _unit_response()
    rate = jnp.asarray(1.5)
    key = jax.random.PRNGKey(7)
    noise_key, _ = jax.random.split(key)
    obs = rate * response + NOISE_STD * jax.random.normal(noise_key, response.shape)
    mean = math.exp(0.0 + 0.5 * 2.0**2)
    std = mean * math.sqrt(math.expm1(2.0**2))
    posterior = infer_emission_rate(
        np.asarray(obs),
        (
            np.asarray(RECEPTORS_X),
            np.asarray(RECEPTORS_Y),
            np.zeros(RECEPTORS_X.shape),
        ),
        source_location=(0.0, 0.0, 5.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="C",
        prior_mean=mean,
        prior_std=std,
        obs_noise_std=NOISE_STD,
        background_prior_std=1e-12,
        num_warmup=300,
        num_samples=1000,
    )
    nuts = np.log(posterior["emission_rate"])
    # the same posterior, up to Monte Carlo error
    assert abs(_grid_mean(obs, response) - nuts.mean()) < 0.25 * nuts.std()
    np.testing.assert_allclose(_grid_std(obs, response), nuts.std(), rtol=0.2)


def _grid_log_posterior(obs: jax.Array, response: jax.Array) -> jax.Array:
    grid_rates = jnp.exp(LOG_Q_GRID)
    # log p(log Q | y) = log N(y; Q c₁, σ²) + log π(Q) + log Q on a uniform log Q grid
    residuals = obs[None, :] - grid_rates[:, None] * response[None, :]
    log_likelihood = -0.5 * jnp.sum(residuals**2, axis=1) / NOISE_STD**2
    return log_likelihood + PER_EVENT_PRIOR.log_prob(grid_rates) + LOG_Q_GRID


def _grid_posterior(obs: jax.Array, response: jax.Array) -> jax.Array:
    return jax.nn.softmax(_grid_log_posterior(obs, response))


def _grid_mean(obs: jax.Array, response: jax.Array) -> float:
    return float(jnp.sum(_grid_posterior(obs, response) * LOG_Q_GRID))


def _grid_std(obs: jax.Array, response: jax.Array) -> float:
    weights = _grid_posterior(obs, response)
    center = jnp.sum(weights * LOG_Q_GRID)
    return float(jnp.sqrt(jnp.sum(weights * (LOG_Q_GRID - center) ** 2)))


def _simulate_soft(key: jax.Array, truth):
    marks_key, event_key = jax.random.split(key)
    marks, mask = _detected_marks(marks_key, truth)
    response = _unit_response()
    keys = jax.random.split(event_key, MAX_DETECTED)
    draws = jax.vmap(_event_draws, (0, 0, None))(keys, marks, response)
    return _buffer(mask), mask, jnp.where(mask[:, None], draws, 1.0)


def _fit(key: jax.Array, data, log_weights: jax.Array) -> dict[str, jax.Array]:
    times, mask, q_samples = data
    mcmc = MCMC(
        NUTS(tmtpp_iw_model),
        num_warmup=NUM_WARMUP,
        num_samples=NUM_DRAWS * THINNING,
        thinning=THINNING,
        progress_bar=False,
    )
    mcmc.run(
        key,
        times,
        mask,
        EventEvidence(log_weights, q_samples),
        mark=SBC_MARK,
        intensity="constant",
        window=WINDOW,
        log_pod=log_pod_union(FIXED_POD),
        support=SUPPORT,
        n_nodes=N_NODES,
    )
    samples = mcmc.get_samples()
    return {name: samples[name] for name in ("lambda_0", "mark.mu", "mark.sigma")}


def _fit_point(key: jax.Array, data) -> dict[str, jax.Array]:
    return _fit(key, data, jnp.zeros_like(data[2]))


def _fit_soft(key: jax.Array, data) -> dict[str, jax.Array]:
    # ℓₛ = −log π(qₛ) − log S
    log_weights = -PER_EVENT_PRIOR.log_prob(data[2]) - math.log(N_EVENT_DRAWS)
    return _fit(key, data, log_weights)


@pytest.mark.slow
def test_sbc_point_regime():
    result = run_sbc(
        prior_sample=_prior_sample,
        simulate=_simulate_point,
        fit=_fit_point,
        n_sims=SBC_POINT_SIMS,
        key=jax.random.PRNGKey(114),
    )
    assert result.n_draws == NUM_DRAWS
    assert jnp.all(result.ks_pvalues > SBC_MIN_PVALUE), result


@pytest.mark.slow
def test_sbc_soft_observation():
    result = run_sbc(
        prior_sample=_prior_sample,
        simulate=_simulate_soft,
        fit=_fit_soft,
        n_sims=SBC_SOFT_SIMS,
        key=jax.random.PRNGKey(115),
    )
    assert jnp.all(result.ks_pvalues > SBC_MIN_PVALUE), result


@pytest.mark.slow
def test_sbc_flags_a_fit_that_ignores_the_pod():
    # the negative control: thinned catalogs fitted as if every event were seen
    def fit_without_pod(key, data):
        times, mask, q_samples = data
        mcmc = MCMC(
            NUTS(tmtpp_iw_model),
            num_warmup=NUM_WARMUP,
            num_samples=NUM_DRAWS * THINNING,
            thinning=THINNING,
            progress_bar=False,
        )
        mcmc.run(
            key,
            times,
            mask,
            EventEvidence(jnp.zeros_like(q_samples), q_samples),
            mark=SBC_MARK,
            intensity="constant",
            window=WINDOW,
            support=SUPPORT,
        )
        samples = mcmc.get_samples()
        return {name: samples[name] for name in ("lambda_0", "mark.mu", "mark.sigma")}

    result = run_sbc(
        prior_sample=_prior_sample,
        simulate=_simulate_point,
        fit=fit_without_pod,
        n_sims=SBC_CONTROL_SIMS,
        key=jax.random.PRNGKey(116),
    )
    assert jnp.min(result.ks_pvalues) < SBC_MIN_PVALUE


def test_run_sbc_passes_a_conjugate_fit_and_flags_a_biased_one():
    def prior_sample(key):
        return {"mean": jax.random.normal(key)}

    def simulate(key, truth):
        return truth["mean"] + jax.random.normal(key)

    def exact(key, y):
        # N(y/2, 1/2) is the posterior of N(0, 1) under one unit-noise observation
        return {"mean": 0.5 * y + math.sqrt(0.5) * jax.random.normal(key, (99,))}

    def biased(key, y):
        return {"mean": 0.5 + exact(key, y)["mean"]}

    arguments = {
        "prior_sample": prior_sample,
        "simulate": simulate,
        "n_sims": 500,
        "key": jax.random.PRNGKey(0),
        "batch_size": 500,
    }
    assert run_sbc(fit=exact, **arguments).ks_pvalues[0] > SBC_MIN_PVALUE
    assert run_sbc(fit=biased, **arguments).ks_pvalues[0] < SBC_MIN_PVALUE


def test_ranks_count_the_draws_below_the_truth():
    result = run_sbc(
        prior_sample=lambda key: {"x": jnp.asarray(0.5)},
        simulate=lambda key, truth: None,
        fit=lambda key, data: {"x": jnp.array([0.0, 0.25, 0.75, 1.0])},
        n_sims=3,
        key=jax.random.PRNGKey(0),
    )
    assert result.ranks.tolist() == [[2], [2], [2]]
    assert result.param_names == ("x",)
    assert result.n_draws == 4


def test_rank_uniformity_rejects_a_pile_at_one_end():
    ranks = jnp.zeros((200, 1), dtype=int)
    assert rank_uniformity_pvalues(ranks, 99)[0] < SBC_MIN_PVALUE


def test_run_sbc_needs_a_simulation():
    with pytest.raises(ValueError, match="n_sims"):
        run_sbc(
            prior_sample=lambda key: {"x": jnp.zeros(())},
            simulate=lambda key, truth: None,
            fit=lambda key, data: {"x": jnp.zeros(2)},
            n_sims=0,
            key=jax.random.PRNGKey(0),
        )


# --- the per-event-prior swap ---
SWAP_TRUTH = {"mu": 0.5, "sigma": 0.8}
SWAP_REPLICATES = 4
SWAP_FIT = {"num_warmup": 300, "num_samples": 300, "support": (1e-6, 1e6)}
# Design page 06's alternative prior against a narrower one
SWAP_PRIORS = (
    LognormalEmission(0.0, 2.0).log_prob,
    LognormalEmission(0.0, 1.0).log_prob,
)


def _swap_catalog(*, noise=0.4, n_events=30, n_draws=1000, seed=0) -> EmissionCatalog:
    # conjugate posterior on log Q under the catalog's own broad prior N(0, 3²)
    prior_scale = 3.0
    rng = np.random.default_rng(seed)
    log_q = rng.normal(SWAP_TRUTH["mu"], SWAP_TRUTH["sigma"], n_events)
    measurements = log_q + rng.normal(0.0, noise, n_events)
    precision = 1.0 / noise**2 + 1.0 / prior_scale**2
    post_mean = (measurements / noise**2) / precision
    post_std = 1.0 / math.sqrt(precision)
    prior = LognormalEmission(0.0, prior_scale)
    events = []
    for center in post_mean:
        draws = np.exp(center + post_std * rng.standard_normal(n_draws))
        events.append(
            EmissionEvent(
                emission_rate=float(draws.mean()),
                emission_std=float(draws.std()),
                x=0.0,
                y=0.0,
                time=0.0,
                tier="I",
                q_samples=jnp.asarray(draws),
                prior_logpdf=prior.log_prob,
            )
        )
    return EmissionCatalog.from_events(events)


@pytest.mark.slow
def test_prior_swap_statistic_is_within_is_noise():
    statistic = prior_swap_test(
        _swap_catalog(), priors=SWAP_PRIORS, n_replicates=SWAP_REPLICATES, **SWAP_FIT
    )
    assert statistic < 3.0


@pytest.mark.slow
def test_prior_swap_flags_a_missing_correction():
    # the negative control: a fit that does not divide the per-event prior out
    from plumax.population.fitting import fit_size_distribution_iw

    def uncorrected(catalog, **kwargs):
        events = [
            dataclasses.replace(event, prior_logpdf=jnp.zeros_like)
            for event in catalog.events
        ]
        return fit_size_distribution_iw(EmissionCatalog.from_events(events), **kwargs)

    statistic = prior_swap_test(
        _swap_catalog(),
        priors=SWAP_PRIORS,
        fit=uncorrected,
        n_replicates=SWAP_REPLICATES,
        **SWAP_FIT,
    )
    assert statistic > 3.0


def test_prior_swap_needs_draws_and_priors():
    event = EmissionEvent(
        emission_rate=1.0, emission_std=0.1, x=0.0, y=0.0, time=0.0, tier="I"
    )
    catalog = EmissionCatalog.from_events([event])
    with pytest.raises(PayloadError, match="q_samples and prior_logpdf"):
        prior_swap_test(catalog, priors=SWAP_PRIORS)
    with pytest.raises(ValueError, match="R ≥ 2"):
        prior_swap_test(_swap_catalog(), priors=SWAP_PRIORS, n_replicates=1)


@pytest.mark.slow
def test_identifiability_stress():
    # (λ₀, Q₅₀) of each regime, chosen before the first run: marks have median 1, so
    # Q₅₀ = 5 sees only the upper tail of a busy basin and Q₅₀ = 0.02 sees nearly all
    # of a quiet one. 06b's targets are above 0.5 and below 0.2.
    confounded, separated = identifiability_stress(
        lam_high_pod_low=(8.0, 5.0),
        lam_low_pod_high=(1.0, 0.02),
        key=jax.random.PRNGKey(0),
    )
    assert confounded > 0.5
    assert abs(separated) < 0.2
