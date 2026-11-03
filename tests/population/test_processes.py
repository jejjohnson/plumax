"""The intensity and POD registries as xtremax point processes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from plumax.population.intensity import (
    INTENSITY_REGISTRY,
    ConstantIntensity,
    DiurnalSinusoidalIntensity,
    WeibullRenewalIntensity,
)
from plumax.population.pod import LogLogisticPOD
from plumax.population.processes import (
    LogIntensity,
    PODRetention,
    detection_process,
    make_process,
    padded_event_buffer,
)


pytest.importorskip("xtremax")

POISSON_KERNELS = sorted(
    name for name in INTENSITY_REGISTRY if name != "weibull_renewal"
)


def _kernel(name: str, seed: int = 1):
    # one prior draw: a valid, non-trivial parameter set for every kernel
    return numpyro.handlers.seed(
        INTENSITY_REGISTRY[name].sample_priors, rng_seed=seed
    )()


@pytest.mark.parametrize("name", POISSON_KERNELS)
def test_analytic_compensator_matches_quadrature(name):
    kernel = _kernel(name)
    window = (100.0, 102.0)
    process = make_process(kernel, window)
    compensator = float(process.effective_integrated_intensity())
    # dense trapezoid on calendar time, independent of the adapter
    time = np.linspace(*window, 1_000_001)
    rate = np.asarray(kernel(jnp.asarray(time)))
    reference = np.trapezoid(rate, time)
    np.testing.assert_allclose(compensator, reference, rtol=1e-6)


@pytest.mark.parametrize("name", POISSON_KERNELS)
def test_max_intensity_bounds_kernel(name):
    kernel = _kernel(name, seed=2)
    time = jnp.linspace(0.0, 365.25, 200_001)
    assert float(jnp.max(kernel(time))) <= float(kernel.max_intensity()) + 1e-12


def test_log_prob_is_the_poisson_likelihood():
    kernel = DiurnalSinusoidalIntensity(lambda_0=3.0, amplitude=1.0)
    window = (10.0, 12.0)
    process = make_process(kernel, window)
    event_times, mask = padded_event_buffer([10.3, 11.6, 10.9], window, max_events=5)
    # Σ log λ(tᵢ) − Λ(T); Λ in closed form for an unclipped sinusoid
    phase = 2.0 * np.pi * 14.0 / 24.0
    times = np.array([10.3, 10.9, 11.6])
    log_rates = np.log(3.0 + np.sin(2.0 * np.pi * times - phase))

    def antiderivative(t):
        return 3.0 * t - np.cos(2.0 * np.pi * t - phase) / (2.0 * np.pi)

    expected = log_rates.sum() - (antiderivative(12.0) - antiderivative(10.0))
    np.testing.assert_allclose(float(process.log_prob(event_times, mask)), expected)


def test_log_prob_gradient_matches_finite_differences():
    window = (0.0, 3.0)
    event_times, mask = padded_event_buffer([0.4, 1.2, 2.9], window)

    def log_likelihood(rate):
        kernel = DiurnalSinusoidalIntensity(lambda_0=rate, amplitude=0.5)
        return make_process(kernel, window).log_prob(event_times, mask)

    step = 1e-6
    central = (log_likelihood(2.0 + step) - log_likelihood(2.0 - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(log_likelihood)(2.0), central, rtol=1e-6)


def test_sampler_count_matches_compensator():
    process = make_process("constant", (0.0, 5.0), lambda_0=2.0)
    keys = jax.random.split(jax.random.key(0), 2000)
    counts = jax.vmap(lambda key: process.sample(key, 64)[1].sum())(keys)
    # E[N(T)] = λ T = 10, Monte-Carlo sd ≈ √10 / √2000 ≈ 0.07
    np.testing.assert_allclose(float(counts.mean()), 10.0, atol=0.3)


def _hill(q50: float, steepness: float) -> LogLogisticPOD:
    # P_d = 1 / (1 + (Q₅₀/Q)ⁿ) is a logistic in log Q with β₁ = n, β₀ = −n log Q₅₀
    return LogLogisticPOD(beta_0=-steepness * np.log(q50), beta_1=steepness)


def test_hill_retention_in_thinning_process():
    rate, horizon = 2.0, 10.0
    marks_law = dist.LogNormal(0.0, 1.0)
    retention = PODRetention(_hill(q50=1.0, steepness=2.0))
    ground = make_process("constant", (0.0, horizon), lambda_0=rate)
    detected = detection_process(ground, marks_law, retention, n_integration_points=11)
    event_times, mask = padded_event_buffer([1.0, 4.5, 8.0], (0.0, horizon))
    marks = jnp.array([0.5, 2.0, 1.5])
    log_prob = detected.log_prob(
        event_times,
        mask,
        marks,
        mark_sample_key=jax.random.key(1),
        n_mark_samples=4096,
    )
    # p̄ = ∫ P_d f dQ; the Hill curve at Q₅₀ = 1 is symmetric in log Q about the
    # lognormal's median, so p̄ = ½ exactly
    pod = 1.0 / (1.0 + (1.0 / marks) ** 2)
    expected = (
        3 * np.log(rate)
        + float(marks_law.log_prob(marks).sum())
        + float(jnp.log(pod).sum())
        - rate * horizon * 0.5
    )
    # Monte-Carlo mark average in the compensator: sd ≈ λ T · 0.5 / √(11 · 4096)
    np.testing.assert_allclose(float(log_prob), expected, atol=0.2)


def test_detected_count_matches_thinned_rate():
    rate, horizon = 2.0, 10.0
    retention = PODRetention(_hill(q50=1.0, steepness=2.0))
    ground = make_process("constant", (0.0, horizon), lambda_0=rate)
    detected = detection_process(ground, dist.LogNormal(0.0, 1.0), retention)
    keys = jax.random.split(jax.random.key(2), 500)
    counts = jax.vmap(lambda key: detected.sample(key, 64)[1].sum())(keys)
    # E[N_det] = λ T p̄ = 10; sd of the mean ≈ √10 / √500 ≈ 0.14
    np.testing.assert_allclose(float(counts.mean()), 10.0, atol=0.6)


def test_retention_needs_the_mark():
    retention = PODRetention(_hill(q50=1.0, steepness=2.0))
    with pytest.raises(ValueError, match="depends on the mark"):
        retention(0.0, None, None)


def test_retention_scales_marks_to_pod_units():
    retention = PODRetention(_hill(q50=3600.0, steepness=2.0), mark_scale=3600.0)
    np.testing.assert_allclose(float(retention(0.0, None, jnp.asarray(1.0))), 0.5)


def test_padded_event_buffer_invariant():
    event_times, mask = padded_event_buffer([7.0, 5.5], (5.0, 8.0), max_events=4)
    np.testing.assert_allclose(event_times, [0.5, 2.0, 3.0, 3.0])
    assert mask.tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="window"):
        padded_event_buffer([9.0], (5.0, 8.0))
    with pytest.raises(ValueError, match="max_events"):
        padded_event_buffer([5.5, 6.0], (5.0, 8.0), max_events=1)


def test_make_process_rejects_renewal_and_bad_arguments():
    with pytest.raises(ValueError, match="renewal"):
        make_process(WeibullRenewalIntensity(scale=1.0, shape=2.0), (0.0, 1.0))
    with pytest.raises(ValueError, match="registry name"):
        make_process(ConstantIntensity(lambda_0=1.0), (0.0, 1.0), lambda_0=2.0)
    with pytest.raises(ValueError, match="t₁ > t₀"):
        make_process("constant", (1.0, 1.0), lambda_0=1.0)


def test_log_intensity_is_window_relative():
    kernel = DiurnalSinusoidalIntensity(lambda_0=3.0, amplitude=1.0)
    shifted = LogIntensity(kernel, origin=7.25)
    np.testing.assert_allclose(
        shifted(jnp.array([0.5])), jnp.log(kernel(jnp.array([7.75])))
    )
