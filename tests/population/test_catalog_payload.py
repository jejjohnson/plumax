"""Per-event posterior payload v2: draws of Q and prior recall from every tier.

Toy problems only. Each tier's posterior must (i) give draws whose mean and
standard deviation match its ``(emission_rate, emission_std)`` summary within
Monte-Carlo error, and (ii) expose a per-event prior log-density on Q that
integrates to one.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax._src.scalar_emission import (
    GaussianEmission,
    LognormalEmission,
    gaussian_total,
    lognormal_total,
    sample_lognormal_total,
)
from plumax.errors import PayloadError
from plumax.lagrangian.inversion import (
    GaussianPosterior,
    LognormalPosterior,
    linear_gaussian_inversion,
    lognormal_inversion,
    matern32_covariance,
)
from plumax.population.catalog import EmissionCatalog, event_from_posterior


pytest.importorskip("vardax")

N_DRAWS = 20_000


def _assert_samples_match_summary(posterior, *, seed=0):
    draws = np.asarray(posterior.samples(jax.random.key(seed), N_DRAWS))
    mean = float(posterior.emission_rate)
    std = float(posterior.emission_std)
    assert draws.shape == (N_DRAWS,)
    # the sample mean has sd σ/√S; 5 of those is a generous band
    assert abs(draws.mean() - mean) < 5.0 * std / np.sqrt(N_DRAWS)
    np.testing.assert_allclose(draws.std(), std, rtol=0.03)


def _prior_mass(posterior, lower, upper):
    grid = np.linspace(lower, upper, 400_001)
    density = np.exp(np.asarray(posterior.prior_logpdf(jnp.asarray(grid))))
    return np.trapezoid(density, grid)


# ── shared scalar laws ───────────────────────────────────────────────────────


def test_lognormal_from_moments_round_trip():
    law = LognormalEmission.from_moments(2.0, 0.5)
    draws = law.sample(jax.random.key(0), 200_000)
    np.testing.assert_allclose(draws.mean(), 2.0, rtol=5e-3)
    np.testing.assert_allclose(draws.std(), 0.5, rtol=1e-2)
    assert float(law.log_prob(0.0)) == -np.inf


def test_lognormal_total_is_exact_for_one_entry_and_matches_moments():
    single = lognormal_total(jnp.array([0.3]), jnp.array([[0.04]]), jnp.array([2.0]))
    np.testing.assert_allclose(single.loc, np.log(2.0) + 0.3)
    np.testing.assert_allclose(single.scale, 0.2)
    # a correlated sum: the matched lognormal has the exact first two moments
    log_mean = jnp.array([0.0, -0.5, 0.2])
    log_cov = matern32_covariance(jnp.arange(3.0), variance=0.09, length_scale=2.0)
    scale = jnp.array([1.0, 2.0, 0.5])
    matched = lognormal_total(log_mean, log_cov, scale)
    draws = sample_lognormal_total(jax.random.key(1), 400_000, log_mean, log_cov, scale)
    matched_mean = np.exp(matched.loc + 0.5 * matched.scale**2)
    matched_var = (np.exp(matched.scale**2) - 1.0) * matched_mean**2
    np.testing.assert_allclose(draws.mean(), matched_mean, rtol=2e-3)
    np.testing.assert_allclose(draws.var(), matched_var, rtol=2e-2)


def test_gaussian_total_is_the_linear_functional():
    covariance = jnp.array([[1.0, 0.3], [0.3, 2.0]])
    law = gaussian_total(jnp.array([1.0, 2.0]), covariance, jnp.array([0.5, 0.5]))
    np.testing.assert_allclose(law.mean, 1.5)
    np.testing.assert_allclose(law.std, np.sqrt(0.25 * (1.0 + 0.6 + 2.0)))
    assert isinstance(law, GaussianEmission)


# ── Tier I ───────────────────────────────────────────────────────────────────


def test_tier_one_payload():
    from plumax.gauss_plume.inference import EmissionRateSamples

    rng = np.random.default_rng(0)
    draws = rng.lognormal(np.log(0.1), 0.2, size=4000)
    posterior = EmissionRateSamples(
        {"emission_rate": draws, "background": np.zeros(4000)},
        prior_mean=0.1,
        prior_std=0.05,
    )
    # the dict view is unchanged
    assert posterior["emission_rate"] is draws
    _assert_samples_match_summary(posterior)
    np.testing.assert_allclose(_prior_mass(posterior, 1e-9, 2.0), 1.0, atol=1e-6)


def test_tier_one_runner_returns_the_payload():
    from plumax.gauss_plume.inference import EmissionRateSamples, infer_emission_rate

    x = np.array([300.0, 400.0, 500.0, 600.0])
    coords = (x, np.zeros(4), np.full(4, 2.0))
    posterior = infer_emission_rate(
        np.full(4, 1e-6),
        coords,
        (0.0, 0.0, 2.0),
        wind_speed=5.0,
        wind_direction=270.0,
        num_warmup=50,
        num_samples=100,
        prior_mean=0.2,
        prior_std=0.1,
    )
    assert isinstance(posterior, EmissionRateSamples)
    expected = LognormalEmission.from_moments(0.2, 0.1).log_prob(0.3)
    np.testing.assert_allclose(posterior.prior_logpdf(0.3), expected)


# ── Tier II ──────────────────────────────────────────────────────────────────


def _tier_two_problem():
    footprint = jnp.array([[1.0, 0.5, 0.1], [0.2, 1.0, 0.4], [0.1, 0.3, 1.0]])
    truth = jnp.array([1.0, 2.0, 0.5])
    prior_cov = matern32_covariance(jnp.arange(3.0), variance=0.5, length_scale=1.5)
    return footprint, footprint @ truth, prior_cov


def test_tier_two_gaussian_payload():
    footprint, observation, prior_cov = _tier_two_problem()
    posterior = linear_gaussian_inversion(
        footprint,
        observation,
        prior_mean=jnp.ones(3),
        prior_cov=prior_cov,
        obs_variance=jnp.full(3, 0.05),
    )
    _assert_samples_match_summary(posterior)
    prior_std = np.sqrt(float(jnp.sum(prior_cov)))
    mass = _prior_mass(posterior, 3.0 - 12 * prior_std, 3.0 + 12 * prior_std)
    np.testing.assert_allclose(mass, 1.0, atol=1e-6)


def test_tier_two_lognormal_payload():
    footprint, observation, _ = _tier_two_problem()
    log_cov = matern32_covariance(jnp.arange(3.0), variance=0.04, length_scale=1.5)
    posterior = lognormal_inversion(
        footprint,
        observation,
        prior_mean=jnp.ones(3),
        prior_log_cov=log_cov,
        obs_variance=jnp.full(3, 1e-4),
    )
    # emission_rate / emission_std are the delta-method summary of the total;
    # with a tight posterior the exact draws agree with it
    _assert_samples_match_summary(posterior)
    np.testing.assert_allclose(_prior_mass(posterior, 1e-6, 20.0), 1.0, atol=1e-6)


def test_hand_built_posteriors_report_the_missing_prior():
    posterior = GaussianPosterior(mean=jnp.ones(2), covariance=jnp.eye(2))
    with pytest.raises(PayloadError, match="no prior"):
        posterior.prior_logpdf(1.0)
    lognormal = LognormalPosterior(
        mean=jnp.ones(2), log_increment=jnp.zeros(2), log_covariance=jnp.eye(2)
    )
    with pytest.raises(PayloadError, match="no prior"):
        lognormal.prior_logpdf(1.0)


# ── Tier III ─────────────────────────────────────────────────────────────────


def test_tier_three_payload_on_a_laplace_result():
    from plumax.les_fvm.fourdvar import FourDVarResult, PosteriorCovariance

    times = jnp.linspace(0.0, 30.0, 4)
    source_cov = matern32_covariance(times, variance=0.04, length_scale=15.0)
    prior_cov = matern32_covariance(times, variance=1.0, length_scale=15.0)
    result = FourDVarResult(
        source=jnp.array([0.4, 0.9, 1.1, 0.8]),
        whitened=jnp.zeros(4),
        cost=0.0,
        n_steps=1,
        posterior=PosteriorCovariance(
            whitened_covariance=jnp.eye(4), source_covariance=source_cov
        ),
        prior_mean=jnp.full(4, 0.5),
        prior_covariance=prior_cov,
    )
    np.testing.assert_allclose(result.emission_rate, 0.8)
    _assert_samples_match_summary(result)
    prior_std = np.sqrt(float(jnp.sum(prior_cov))) / 4
    mass = _prior_mass(result, 0.5 - 12 * prior_std, 0.5 + 12 * prior_std)
    np.testing.assert_allclose(mass, 1.0, atol=1e-6)
    without = dataclasses.replace(result, posterior=None)
    assert bool(jnp.isnan(without.emission_std))
    with pytest.raises(PayloadError, match="compute_posterior"):
        without.samples(jax.random.key(0), 10)


@pytest.mark.slow
def test_tier_three_payload():
    from plumax.les_fvm.fourdvar import build_forward, build_problem, solve_4dvar

    n_t = 3
    save_times = jnp.linspace(0.0, 30.0, n_t)
    forward = build_forward(
        domain_x=(0.0, 300.0, 12),
        domain_y=(-100.0, 100.0, 6),
        domain_z=(0.0, 80.0, 4),
        save_times=save_times,
        source_location=(50.0, 0.0, 20.0),
        uniform_wind=(4.0, 0.0, 0.0),
        eddy_diffusivity=2.0,
        advection_scheme="upwind1",
    )
    observations = forward.predict(jnp.array([0.5, 1.0, 1.0]))
    prior_cov = matern32_covariance(save_times, variance=1.0, length_scale=20.0)
    problem = build_problem(
        forward=forward,
        observations=observations,
        prior_mean=jnp.full(n_t, 0.5),
        prior_covariance=prior_cov,
        obs_variance=(float(observations.max()) * 1e-2) ** 2,
    )
    result = solve_4dvar(problem, max_steps=15, compute_posterior=True)
    _assert_samples_match_summary(result)
    prior_std = np.sqrt(float(jnp.sum(prior_cov))) / n_t
    mass = _prior_mass(result, 0.5 - 12 * prior_std, 0.5 + 12 * prior_std)
    np.testing.assert_allclose(mass, 1.0, atol=1e-6)


# ── Tier IV ──────────────────────────────────────────────────────────────────


def _coupled_forward():
    from plumax.coupled import Instrument, PlumeSource, build_coupled_forward

    source = PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=jnp.linspace(0.0, 200.0, 11),
        background=0.0,
    )
    xs, ys = np.meshgrid([200.0, 400.0, 700.0], [-60.0, 0.0, 60.0], indexing="ij")
    pixels = np.column_stack([xs.ravel(), ys.ravel()])
    instrument = Instrument("TROPOMI", pixels, retrieval_variance=1.0)
    forward = build_coupled_forward(source, [instrument])
    clean = forward.predict(2.0, [0.0])[0]
    scale = float(jnp.max(clean))
    instrument = Instrument("TROPOMI", pixels, retrieval_variance=(0.2 * scale) ** 2)
    forward = build_coupled_forward(source, [instrument])
    noise = 0.2 * scale * jax.random.normal(jax.random.key(3), (9,))
    return forward, clean + noise, scale


def test_tier_four_fusion_payload():
    from plumax.coupled import fuse_observations

    forward, observation, scale = _coupled_forward()
    posterior = fuse_observations(
        forward,
        [observation],
        prior_mean=jnp.array([1.0, 0.0]),
        prior_covariance=jnp.diag(jnp.array([4.0, (0.1 * scale) ** 2])),
    )
    _assert_samples_match_summary(posterior)
    np.testing.assert_allclose(_prior_mass(posterior, -23.0, 25.0), 1.0, atol=1e-6)


@pytest.mark.parametrize("block", ["emission_rate", "log_emission_rate"])
def test_tier_four_coupled_payload(block):
    from plumax.coupled import CoupledPrior, CoupledProblem, solve_coupled

    forward, observation, scale = _coupled_forward()
    center = 0.0 if block == "log_emission_rate" else 1.0
    spread = 1.0 if block == "log_emission_rate" else 2.0
    prior = CoupledPrior.from_std(
        {block: center, "biases": jnp.zeros(1)},
        {block: spread, "biases": 0.1 * scale},
    )
    result = solve_coupled(
        CoupledProblem(
            forward=forward, prior=prior, observations={"TROPOMI": observation}
        )
    )
    _assert_samples_match_summary(result)
    if block == "log_emission_rate":
        mass = _prior_mass(result, 1e-8, 400.0)
        np.testing.assert_allclose(mass, 1.0, atol=1e-4)
    else:
        mass = _prior_mass(result, -23.0, 25.0)
        np.testing.assert_allclose(mass, 1.0, atol=1e-6)


# ── the catalog ──────────────────────────────────────────────────────────────


def test_event_from_posterior_fills_the_payload():
    footprint, observation, prior_cov = _tier_two_problem()
    posterior = linear_gaussian_inversion(
        footprint,
        observation,
        prior_mean=jnp.ones(3),
        prior_cov=prior_cov,
        obs_variance=jnp.full(3, 0.05),
    )
    event = event_from_posterior(
        posterior, x=0.0, y=0.0, time=0.0, tier="II", key=jax.random.key(0)
    )
    assert event.q_samples.shape == (512,)
    np.testing.assert_allclose(
        event.prior_logpdf(jnp.asarray(3.0)), posterior.prior_logpdf(3.0)
    )
    # without a key the draws stay empty; the prior is still recalled
    no_draws = event_from_posterior(posterior, x=0.0, y=0.0, time=0.0, tier="II")
    assert no_draws.q_samples is None
    assert no_draws.prior_logpdf is not None
    # the summary and equality are unchanged by the payload
    assert event == no_draws
    assert EmissionCatalog.from_events([event]).n_events == 1


def test_v1_posterior_leaves_the_payload_empty():
    class SummaryOnly:
        emission_rate = jnp.asarray(2.0)
        emission_std = jnp.asarray(0.5)

    event = event_from_posterior(
        SummaryOnly(), x=0.0, y=0.0, time=0.0, tier="IV", key=jax.random.key(0)
    )
    assert event.q_samples is None
    assert event.prior_logpdf is None
    assert event.emission_rate == 2.0
