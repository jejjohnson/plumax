"""The persistency metrics of 06c: closed forms, thinning, posterior coverage."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.population.fitting import TMTPPPosterior
from plumax.population.intensity import (
    INTENSITY_REGISTRY,
    ConstantIntensity,
    DiurnalSinusoidalIntensity,
    OperationalScheduleIntensity,
)
from plumax.population.marks import LognormalMark
from plumax.population.persistency import (
    expected_count,
    expected_wait_time,
    occurrence_probability,
    posterior_metric,
)
from plumax.population.point_process import (
    InhomogeneousIntensityPosterior,
    PoissonRatePosterior,
)
from plumax.population.processes import make_process


EPS = float(jnp.finfo(float).eps)
# λ(t) = λ₀ + A sin(2πt − φ) peaks 6 h after phase_hours: 14:00 here, trough 02:00
DIURNAL = {"lambda_0": 2.0, "amplitude": 1.6, "phase_hours": 8.0}


def _dense_wait(kernel, t: float, *, days: float, per_day: int) -> float:
    # independent reference: a uniform trapezoid on S(s), over a much finer grid
    lags = np.arange(int(days * per_day) + 1) / per_day
    rates = np.asarray(kernel(jnp.asarray(t + lags)))
    steps = 0.5 * (rates[1:] + rates[:-1]) / per_day
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    survival = np.exp(-cumulative)
    tail = survival[-1] / (cumulative[-1] / lags[-1])
    return float(np.trapezoid(survival, lags) + tail)


@pytest.mark.parametrize("rate", [1e-3, 0.5, 2.0, 1e3])
@pytest.mark.parametrize("t", [0.0, 17.25])
def test_homogeneous_closed_form(rate, t):
    horizon = 1.75
    wait = expected_wait_time("constant", t, lambda_0=rate)
    probability = occurrence_probability("constant", t, horizon, lambda_0=rate)
    count = expected_count("constant", t, horizon, lambda_0=rate)
    np.testing.assert_allclose(wait, 1.0 / rate, rtol=2 * EPS)
    np.testing.assert_allclose(probability, -math.expm1(-rate * horizon), rtol=4 * EPS)
    np.testing.assert_allclose(count, rate * horizon, rtol=4 * EPS)


def test_quadrature_path_recovers_the_homogeneous_limit():
    # a flat diurnal has no closed form to dispatch to, so this is the quadrature
    flat = DiurnalSinusoidalIntensity(lambda_0=2.0, amplitude=0.0, phase_hours=14.0)
    count = expected_count(flat, 0.3, 1.5, nodes_per_day=24)
    probability = occurrence_probability(flat, 0.3, 1.5, nodes_per_day=24)
    # Simpson is exact on a constant
    np.testing.assert_allclose(count, 3.0, rtol=1e-13)
    np.testing.assert_allclose(probability, -math.expm1(-3.0), rtol=1e-13)


@pytest.mark.slow
def test_wait_time_quadrature_converges_at_second_order():
    # the trapezoid on the log-spaced grid: second order in the cell ratio
    flat = DiurnalSinusoidalIntensity(lambda_0=2.0, amplitude=0.0, phase_hours=14.0)
    wait = expected_wait_time(flat, 0.3)
    np.testing.assert_allclose(wait, 0.5, rtol=5e-5)
    finer = expected_wait_time(flat, 0.3, n_cells=4096)
    assert abs(float(finer) - 0.5) < abs(float(wait) - 0.5) / 3.0


def test_wait_time_tail_closure_covers_waits_beyond_the_grid():
    # E[Δt] = 2 × 10⁵ days, far beyond the grid's last node at 10⁴ days
    flat = DiurnalSinusoidalIntensity(lambda_0=5e-6, amplitude=0.0, phase_hours=0.0)
    np.testing.assert_allclose(expected_wait_time(flat, 0.0), 2e5, rtol=5e-5)


def test_expected_count_derivative_in_the_horizon_is_the_intensity():
    # d/dh Λ(t, t + h) = λ(t + h), with Simpson nodes dense enough for a sinusoid
    kernel = DiurnalSinusoidalIntensity(**DIURNAL)
    t, horizon, step = 0.2, 0.4, 1e-4

    def count(h):
        return expected_count(kernel, t, h, nodes_per_day=20_000)

    central = (count(horizon + step) - count(horizon - step)) / (2 * step)
    np.testing.assert_allclose(central, kernel(jnp.asarray(t + horizon)), rtol=1e-6)


def test_counts_over_adjoining_horizons_add():
    kernel = DiurnalSinusoidalIntensity(**DIURNAL)
    nodes = 288
    whole = expected_count(kernel, 0.1, 1.0, nodes_per_day=nodes)
    first = expected_count(kernel, 0.1, 0.35, nodes_per_day=nodes)
    second = expected_count(kernel, 0.45, 0.65, nodes_per_day=nodes)
    parts = first + second
    np.testing.assert_allclose(whole, parts, rtol=1e-10)


def test_diurnal_sanity():
    # the 06c check: peak λ at 14:00 means a much shorter wait than at 02:00
    kernel = DiurnalSinusoidalIntensity(lambda_0=8.0, amplitude=7.2, phase_hours=8.0)
    at_peak = float(expected_wait_time(kernel, 14.0 / 24.0))
    at_trough = float(expected_wait_time(kernel, 2.0 / 24.0))
    assert at_peak < 0.5 * at_trough
    peak_probability = occurrence_probability(kernel, 12.0 / 24.0, 4.0 / 24.0)
    trough_probability = occurrence_probability(kernel, 0.0, 4.0 / 24.0)
    assert float(peak_probability) > 2.0 * float(trough_probability)


@pytest.mark.parametrize(
    ("kernel", "t", "rtol"),
    [
        (DiurnalSinusoidalIntensity(**DIURNAL), 0.3, 1e-4),
        (DiurnalSinusoidalIntensity(0.1, 0.8, 14.0), 0.0, 1e-4),
        # sharp on/off switching at a low rate, the hardest case for the grid
        (OperationalScheduleIntensity(0.15, 0.0025, 8.0, 17.0), 0.3, 5e-3),
    ],
)
def test_wait_time_matches_dense_integration(kernel, t, rtol):
    dense = _dense_wait(kernel, t, days=1500.0, per_day=1440)
    np.testing.assert_allclose(expected_wait_time(kernel, t), dense, rtol=rtol)


def test_metrics_jit_and_vmap_match_a_python_loop():
    kernel = DiurnalSinusoidalIntensity(**DIURNAL)
    times = jnp.array([0.0, 0.3, 0.7])
    loop = jnp.stack([expected_wait_time(kernel, t) for t in times])
    vmapped = jax.vmap(lambda t: expected_wait_time(kernel, t))(times)
    jitted = jax.jit(lambda t: expected_wait_time(kernel, t))(times[1])
    np.testing.assert_allclose(vmapped, loop, rtol=1e-12)
    np.testing.assert_allclose(jitted, loop[1], rtol=1e-12)


def test_wait_time_gradient_matches_central_difference():
    def wait(rate):
        return expected_wait_time(DiurnalSinusoidalIntensity(rate, 0.8, 14.0), 0.3)

    step = 1e-5
    central = (wait(2.0 + step) - wait(2.0 - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(wait)(2.0), central, rtol=1e-6)


def test_renewal_hazard_is_rejected():
    with pytest.raises(ValueError, match="renewal process"):
        expected_wait_time("weibull_renewal", 0.0, scale=1.0, shape=1.5)
    with pytest.raises(ValueError, match="renewal process"):
        occurrence_probability("weibull_renewal", 0.0, 1.0, scale=1.0, shape=1.5)


def test_parameters_with_an_instance_are_rejected():
    with pytest.raises(ValueError, match="only with a registry name"):
        expected_wait_time(ConstantIntensity(1.0), 0.0, lambda_0=2.0)


def test_posterior_metric_on_a_gamma_posterior():
    posterior = PoissonRatePosterior(
        alpha=401.0, beta=200.0, n_events=400, exposure=200.0
    )
    result = posterior_metric(
        expected_wait_time,
        posterior,
        0.0,
        key=jax.random.PRNGKey(0),
        num_samples=20_000,
    )
    assert result["samples"].shape == (20_000,)
    # 1/λ under Gamma(α, β) is inverse-gamma, with mean β / (α − 1)
    np.testing.assert_allclose(result["samples"].mean(), 200.0 / 400.0, rtol=2e-3)
    np.testing.assert_allclose(
        result["quantiles"],
        np.quantile(result["samples"], result["levels"]),
        rtol=1e-12,
    )


def test_posterior_metric_scales_a_rate_per_unit_area():
    posterior = PoissonRatePosterior(alpha=11.0, beta=10.0, n_events=10, exposure=10.0)
    key = jax.random.PRNGKey(1)
    base = posterior_metric(expected_wait_time, posterior, 0.0, key=key)
    scaled = posterior_metric(
        expected_wait_time, posterior, 0.0, key=key, rate_scale=4.0
    )
    np.testing.assert_allclose(scaled["samples"], base["samples"] / 4.0, rtol=1e-12)


def test_gamma_posterior_needs_a_key():
    posterior = PoissonRatePosterior(alpha=2.0, beta=1.0, n_events=1, exposure=1.0)
    with pytest.raises(ValueError, match="needs a key"):
        posterior_metric(expected_wait_time, posterior, 0.0)


def test_posterior_metric_on_a_log_linear_posterior():
    beta0 = np.array([0.0, math.log(2.0)])
    beta = np.array([[1.0], [-3.0]])

    def covariates(t):
        # z(t) ≡ 0, so λ⁽ᵉ⁾ = c exp(β₀⁽ᵉ⁾)
        return jnp.zeros((*jnp.shape(t), 1))

    posterior = InhomogeneousIntensityPosterior(beta0_samples=beta0, beta_samples=beta)
    result = posterior_metric(
        expected_count, posterior, 0.0, 1.0, covariates=covariates, rate_scale=3.0
    )
    np.testing.assert_allclose(result["samples"], [3.0, 6.0], rtol=1e-10)
    with pytest.raises(ValueError, match="needs covariates"):
        posterior_metric(expected_count, posterior, 0.0, 1.0)


def test_log_linear_covariates_shape_the_intensity_in_time():
    # log λ(t) = β t, so Λ(0, 1) = (e^β − 1) / β
    posterior = InhomogeneousIntensityPosterior(
        beta0_samples=np.zeros(1), beta_samples=np.array([[0.7]])
    )
    result = posterior_metric(
        expected_count, posterior, 0.0, 1.0, covariates=lambda t: jnp.expand_dims(t, -1)
    )
    np.testing.assert_allclose(result["samples"], [math.expm1(0.7) / 0.7], rtol=1e-10)


def test_posterior_metric_on_tmtpp_draws():
    samples = {
        "lambda_0": jnp.array([1.0, 2.0]),
        "amplitude": jnp.array([0.5, 0.2]),
        "phase_hours": jnp.array([14.0, 3.0]),
        "mark.mu": jnp.zeros(2),
        "mark.sigma": jnp.ones(2),
    }
    posterior = TMTPPPosterior("diurnal", LognormalMark(), samples)
    result = posterior_metric(expected_count, posterior, 0.0, 1.0)
    # a whole day integrates the sinusoid away
    np.testing.assert_allclose(result["samples"], [1.0, 2.0], rtol=1e-10)
    assert posterior.num_samples == 2
    theta = posterior.theta({k: v[0] for k, v in samples.items()})
    assert set(theta) == {"mu", "sigma"}


@pytest.mark.parametrize("name", sorted(set(INTENSITY_REGISTRY) - {"weibull_renewal"}))
def test_tmtpp_kernel_maps_every_prior_site_to_its_field(name):
    # a kernel's site names can differ from its fields, so round-trip the prior
    from numpyro import handlers

    cls = INTENSITY_REGISTRY[name]
    trace = handlers.trace(handlers.seed(cls.sample_priors, 3)).get_trace()
    draw = {site: value["value"] for site, value in trace.items()}
    prior_kernel = handlers.seed(cls.sample_priors, 3)()
    posterior = TMTPPPosterior(name, LognormalMark(), {})
    rebuilt = posterior.kernel(draw)
    times = jnp.linspace(0.0, 400.0, 41)
    np.testing.assert_allclose(rebuilt(times), prior_kernel(times), rtol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("start", [14.0 / 24.0, 2.0 / 24.0])
def test_mc_self_consistency(start):
    # thinning draws of a known λ(t), against the metrics at the same start time
    n_replicates = 40_000
    # S(10 days) ~ e⁻²⁰, so truncating at the window leaves no measurable bias
    window = 10.0
    process = make_process("diurnal", (start, start + window), **DIURNAL)
    keys = jax.random.split(jax.random.PRNGKey(0), n_replicates)
    draw = jax.jit(jax.vmap(lambda key: process.sample(key, max_candidates=128)))
    times, mask, _ = draw(keys)
    times, mask = np.asarray(times), np.asarray(mask)
    assert mask[:, 0].all()
    first = times[:, 0]

    wait = float(expected_wait_time("diurnal", start, **DIURNAL))
    standard_error = first.std() / math.sqrt(n_replicates)
    assert abs(first.mean() - wait) < 4.0 * standard_error

    horizon = 0.25
    frequency = float((first < horizon).mean())
    probability = float(occurrence_probability("diurnal", start, horizon, **DIURNAL))
    standard_error = math.sqrt(probability * (1 - probability) / n_replicates)
    assert abs(frequency - probability) < 4.0 * standard_error

    counts = (mask & (times < horizon)).sum(-1)
    count = float(expected_count("diurnal", start, horizon, **DIURNAL))
    # N is Poisson, so its variance is its mean
    assert abs(counts.mean() - count) < 4.0 * math.sqrt(count / n_replicates)


@pytest.mark.slow
def test_posterior_coverage():
    # conjugate in the mean: λ(t) = λ₀ + A sin(2πt − φ) over whole days integrates
    # to λ₀ T, so N ~ Poisson(λ₀ T) and λ₀ | N ~ Gamma(α₀ + N, β₀ + T)
    true_rate = 1.5
    days = 20.0
    prior_shape, prior_rate = 1e-3, 1e-3
    n_replicates = 400
    n_draws = 400
    start = 0.3
    amplitude, phase = 0.8, 14.0
    true_kernel = DiurnalSinusoidalIntensity(true_rate, amplitude, phase)
    truth = expected_wait_time(true_kernel, start, n_cells=512, nodes_per_cell=5)
    count_key, draw_key = jax.random.split(jax.random.PRNGKey(0))
    counts = jax.random.poisson(count_key, true_rate * days, (n_replicates,))

    def interval(count, key):
        posterior = PoissonRatePosterior(
            alpha=prior_shape + count,
            beta=prior_rate + days,
            n_events=count,
            exposure=days,
        )

        def diurnal_wait(kernel, t):
            scaled = DiurnalSinusoidalIntensity(kernel.lambda_0, amplitude, phase)
            return expected_wait_time(scaled, t, n_cells=512, nodes_per_cell=5)

        result = posterior_metric(
            diurnal_wait,
            posterior,
            start,
            quantiles=(0.025, 0.975),
            num_samples=n_draws,
            key=key,
        )
        return result["quantiles"]

    keys = jax.random.split(draw_key, n_replicates)
    bounds = jax.jit(jax.vmap(interval))(counts, keys)
    inside = (bounds[:, 0] <= truth) & (truth <= bounds[:, 1])
    coverage = float(inside.mean())
    # four binomial standard errors around the nominal 95 %
    margin = 4.0 * math.sqrt(0.95 * 0.05 / n_replicates)
    assert abs(coverage - 0.95) < margin, coverage
