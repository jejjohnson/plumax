"""The total-emission estimator of 06d: closed forms and the missing-mass truth."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from plumax.population.fitting import TMTPPPosterior, tmtpp_iw_model
from plumax.population.importance import EventEvidence
from plumax.population.marks import LognormalMark, PowerLawMark
from plumax.population.paradox import (
    FacilityConfig,
    compute_E_Pd,
    logistic_pod,
    simulate_paradox,
)
from plumax.population.processes import padded_event_buffer
from plumax.population.totals import TotalEmission, total_emission


Q_RANGE = {"q_min": 1e-4, "q_max": 1e6}


def _posterior(rates, mus, sigmas) -> TMTPPPosterior:
    samples = {
        "lambda_0": jnp.asarray(rates, dtype=float),
        "mark.mu": jnp.asarray(mus, dtype=float),
        "mark.sigma": jnp.asarray(sigmas, dtype=float),
    }
    return TMTPPPosterior("constant", LognormalMark(), samples)


def _logistic(q_50: float, k: float):
    return lambda q: jax.nn.log_sigmoid(k * (q - q_50))


def test_no_pod_is_the_closed_form_and_misses_nothing():
    posterior = _posterior([2.0, 3.0], [1.0, 0.5], [0.5, 0.8])
    total = total_emission(
        posterior, horizon=(5.0, 15.0), event_duration=2.0, log_pod=None, **Q_RANGE
    )
    # M_total = λ T exp(μ + σ²/2) D
    expected = [
        2.0 * 10.0 * math.exp(1.0 + 0.125) * 2.0,
        3.0 * 10.0 * math.exp(0.5 + 0.32) * 2.0,
    ]
    np.testing.assert_allclose(total.samples_kg, expected, rtol=1e-9)
    assert np.all(np.asarray(total.missing_kg) == 0.0)
    np.testing.assert_allclose(total.detected_kg, total.samples_kg, rtol=1e-15)


def test_perfect_pod_no_missing_mass():
    posterior = _posterior([2.0, 3.0], [1.0, 0.5], [0.5, 0.8])
    # log P_d ≡ 0, as a curve rather than the certain-detection shortcut
    total = total_emission(
        posterior,
        horizon=(0.0, 10.0),
        event_duration=2.0,
        log_pod=jnp.zeros_like,
        **Q_RANGE,
    )
    assert np.all(np.asarray(total.missing_kg) == 0.0)
    np.testing.assert_allclose(total.detected_kg, total.samples_kg, rtol=1e-15)


def test_blind_pod_misses_everything():
    posterior = _posterior([2.0], [1.0], [0.5])
    total = total_emission(
        posterior,
        horizon=(0.0, 10.0),
        event_duration=2.0,
        log_pod=lambda q: jnp.full_like(q, -jnp.inf),
        **Q_RANGE,
    )
    np.testing.assert_allclose(total.missing_kg, total.samples_kg, rtol=1e-8)
    np.testing.assert_allclose(total.detected_kg, 0.0, atol=1e-5)


def test_missing_mass_matches_independent_quadrature():
    mu, sigma, q_50, k = 2.0, 1.5, 100.0, 0.02
    posterior = _posterior([1.5], [mu], [sigma])
    total = total_emission(
        posterior,
        horizon=(0.0, 4.0),
        event_duration=3.0,
        log_pod=_logistic(q_50, k),
        **Q_RANGE,
    )

    def integrand(u: float) -> float:
        # E_f[Q (1 − P_d)] in u = log q
        density = math.exp(-0.5 * ((u - mu) / sigma) ** 2) / (
            sigma * math.sqrt(2.0 * math.pi)
        )
        return math.exp(u) * (1.0 - float(expit(k * (math.exp(u) - q_50)))) * density

    rate, _ = quad(integrand, math.log(1e-4), math.log(1e6), limit=400, points=[4.6])
    np.testing.assert_allclose(total.missing_kg, [1.5 * 4.0 * rate * 3.0], rtol=1e-8)


def test_step_pod_splits_the_partial_moments():
    # P_d = 1 above q_c gives missing = Λ D ∫_{q_min}^{q_c} q f(q) dq, also on a
    # heavy-tailed family with a kink at its q_min
    q_c = 10.0
    mark = PowerLawMark(q_min=1.0)
    samples = {"lambda_0": jnp.array([1.0]), "mark.alpha": jnp.array([2.5])}
    posterior = TMTPPPosterior("constant", mark, samples)
    total = total_emission(
        posterior,
        horizon=(0.0, 1.0),
        event_duration=1.0,
        log_pod=lambda q: jnp.where(q >= q_c, 0.0, -jnp.inf),
        q_min=mark.q_min,
        q_max=1e6,
        n_nodes=20_001,
    )
    below = mark.truncated_mean(mark.q_min, q_c, alpha=2.5)
    # the step sits inside one Simpson cell, so this is first order in the cell size
    np.testing.assert_allclose(total.missing_kg, [below], rtol=2e-3)


def test_q_range_is_a_partial_moment():
    # a narrow range keeps only the mass inside it: no renormalisation to a mean
    posterior = _posterior([1.0], [0.0], [1.0])
    total = total_emission(
        posterior,
        horizon=(0.0, 1.0),
        event_duration=1.0,
        log_pod=None,
        q_min=1.0,
        q_max=2.0,
    )
    partial = LognormalMark().truncated_mean(1.0, 2.0, mu=0.0, sigma=1.0)
    np.testing.assert_allclose(total.samples_kg, [partial], rtol=1e-12)


def test_detected_and_missing_add_to_the_total():
    posterior = _posterior([2.0, 3.0], [1.0, 2.0], [0.5, 1.0])
    total = total_emission(
        posterior,
        horizon=(0.0, 30.0),
        event_duration=2.0,
        log_pod=_logistic(50.0, 0.1),
        **Q_RANGE,
    )
    np.testing.assert_allclose(
        total.detected_kg + total.missing_kg, total.samples_kg, rtol=1e-12
    )
    assert np.all(np.asarray(total.detected_kg) > 0.0)
    assert np.all(np.asarray(total.missing_kg) > 0.0)


def test_non_constant_ground_process_integrates_the_kernel():
    samples = {
        "lambda_0": jnp.array([2.0]),
        "amplitude": jnp.array([1.0]),
        "phase_hours": jnp.array([8.0]),
        "mark.mu": jnp.array([0.0]),
        "mark.sigma": jnp.array([1.0]),
    }
    posterior = TMTPPPosterior("diurnal", LognormalMark(), samples)
    total = total_emission(
        posterior,
        horizon=(0.0, 3.0),
        event_duration=1.0,
        log_pod=None,
        nodes_per_day=96,
        **Q_RANGE,
    )
    # whole days integrate the sinusoid to λ₀ T
    np.testing.assert_allclose(total.samples_kg, [6.0 * math.exp(0.5)], rtol=1e-9)


def test_union_of_windows_sets_lambda_and_the_effective_time():
    posterior = _posterior([2.0], [0.0], [1.0])
    # overlapping and out of order: the union is [0, 3] ∪ [5, 6], 4 days
    total = total_emission(
        posterior,
        horizon=[(5.0, 6.0), (1.0, 3.0), (0.0, 2.0)],
        event_duration=1.0,
        log_pod=None,
        **Q_RANGE,
    )
    assert total.t_effective_s == 4.0 * 86_400.0
    np.testing.assert_allclose(total.samples_kg, [2.0 * 4.0 * math.exp(0.5)], rtol=1e-9)


def test_rate_in_tg_per_year():
    samples = jnp.array([1e9, 2e9])
    # 1 Tg over half a Julian year is 2 Tg yr⁻¹
    half_year = 0.5 * 365.25 * 86_400.0
    total = TotalEmission(
        samples_kg=samples,
        detected_kg=samples,
        missing_kg=jnp.zeros(2),
        t_effective_s=half_year,
    )
    np.testing.assert_allclose(total.rate_tg_per_yr(), [2.0, 4.0], rtol=1e-12)


def test_strata_sum_draw_by_draw():
    wells = _posterior([1.0, 2.0], [0.0, 0.0], [1.0, 1.0])
    compressors = _posterior([3.0, 4.0], [1.0, 1.0], [0.5, 0.5])
    arguments = {
        "horizon": (0.0, 10.0),
        "event_duration": 2.0,
        "log_pod": _logistic(5.0, 0.5),
        **Q_RANGE,
    }
    both = total_emission({"wells": wells, "compressors": compressors}, **arguments)
    first = total_emission(wells, **arguments)
    second = total_emission(compressors, **arguments)
    np.testing.assert_allclose(
        both.samples_kg, first.samples_kg + second.samples_kg, rtol=1e-12
    )
    np.testing.assert_allclose(
        both.missing_kg, first.missing_kg + second.missing_kg, rtol=1e-12
    )
    assert both.t_effective_s == first.t_effective_s


def test_missing_mass_gradient_matches_central_difference():
    def missing(mu):
        posterior = _posterior(jnp.ones(1), mu[None], jnp.ones(1))
        total = total_emission(
            posterior,
            horizon=(0.0, 1.0),
            event_duration=1.0,
            log_pod=_logistic(10.0, 0.5),
            **Q_RANGE,
        )
        return total.missing_kg[0]

    step = 1e-6
    point = jnp.asarray(1.0)
    central = (missing(point + step) - missing(point - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(missing)(point), central, rtol=1e-6)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"q_min": 0.0}, "0 < q_min"),
        ({"q_max": jnp.inf}, "q_max < inf"),
        ({"q_min": 2.0, "q_max": 1.0}, "q_min < q_max"),
        ({"event_duration": 0.0}, "event_duration"),
        ({"horizon": (3.0, 3.0)}, "t₁ > t₀"),
    ],
)
def test_invalid_arguments_are_rejected(arguments, message):
    base = {
        "horizon": (0.0, 1.0),
        "event_duration": 1.0,
        "log_pod": None,
        "q_min": 1.0,
        "q_max": 2.0,
    }
    with pytest.raises(ValueError, match=message):
        total_emission(_posterior([1.0], [0.0], [1.0]), **(base | arguments))


def test_strata_need_equal_draws_and_at_least_one():
    one = _posterior([1.0], [0.0], [1.0])
    two = _posterior([1.0, 1.0], [0.0, 0.0], [1.0, 1.0])
    arguments = {
        "horizon": (0.0, 1.0),
        "event_duration": 1.0,
        "log_pod": None,
        **Q_RANGE,
    }
    with pytest.raises(ValueError, match="same number of draws"):
        total_emission({"a": one, "b": two}, **arguments)
    with pytest.raises(ValueError, match="no strata"):
        total_emission({}, **arguments)


def _expected_detected_fraction(config: FacilityConfig) -> float:
    # E_f[Q P_d(Q)] / E_f[Q] by quadrature in u = log q, at the true parameters
    grid = np.linspace(config.mu - 12.0, config.mu + 12.0, 200_001)
    q = np.exp(grid)
    density = np.exp(-0.5 * ((grid - config.mu) / config.sigma) ** 2) / (
        config.sigma * math.sqrt(2.0 * math.pi)
    )
    detected = np.trapezoid(q * logistic_pod(q, config.Q_50, config.k) * density, grid)
    return float(detected / config.E_Q_true)


def test_detected_sum_is_biased_low_by_the_expected_factor():
    config = FacilityConfig()
    factor = _expected_detected_fraction(config)
    assert factor < 0.6
    e_pd = compute_E_Pd(config.mu, config.sigma, config.Q_50, config.k)
    results = [
        simulate_paradox(dataclasses.replace(config, seed=seed), E_Pd=e_pd)
        for seed in range(400)
    ]
    # the ensemble mean, since one heavy-tailed realisation is noisy
    observed = np.array([result.M_obs_mc for result in results])
    true = np.array([result.M_true_mc for result in results])
    np.testing.assert_allclose(observed.mean() / true.mean(), factor, rtol=0.03)


@pytest.mark.slow
def test_recovers_simulated_total():
    # 06d's truth test: a population from the missing-mass simulator, fitted through
    # the TMTPP model with the marks observed exactly and the POD known
    config = FacilityConfig()
    result = simulate_paradox(config)
    window = (0.0, config.observation_window)
    # a constant λ's likelihood depends on the times only through their count
    times = np.sort(np.random.default_rng(0).uniform(*window, result.N_obs))
    event_times, mask = padded_event_buffer(times, window)
    marks = jnp.asarray(result.marks_obs)
    evidence = EventEvidence(
        log_weights=jnp.zeros((result.N_obs, 1)), q_samples=marks[:, None]
    )
    log_pod = _logistic(config.Q_50, config.k)
    mark = LognormalMark()
    from numpyro.infer import MCMC, NUTS

    mcmc = MCMC(
        NUTS(tmtpp_iw_model), num_warmup=500, num_samples=500, progress_bar=False
    )
    mcmc.run(
        jax.random.PRNGKey(0),
        event_times,
        mask,
        evidence,
        mark=mark,
        intensity="constant",
        window=window,
        log_pod=log_pod,
        support=(1e-3, 1e5),
    )
    posterior = TMTPPPosterior("constant", mark, mcmc.get_samples())
    total = total_emission(
        posterior,
        horizon=window,
        event_duration=config.duration,
        log_pod=log_pod,
        q_min=1e-3,
        q_max=1e5,
    )
    low, high = np.quantile(total.samples_kg, [0.025, 0.975])
    # the realised total of this catalog, and the expected total
    assert low < result.M_true_mc < high
    assert low < config.M_true_analytic < high
    # the detected-only sum sits below the whole interval
    assert result.M_obs_mc < low
    # the split reproduces the expected factor at the true parameters
    fraction = total.detected_kg / total.samples_kg
    factor = _expected_detected_fraction(config)
    fraction_low, fraction_high = np.quantile(fraction, [0.025, 0.975])
    assert fraction_low < factor < fraction_high
    assert total.t_effective_s == config.observation_window * 86_400.0
