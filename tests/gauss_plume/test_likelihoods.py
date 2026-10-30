"""Tests for the Tier I observation-noise models (gauss_plume.likelihoods)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from hypothesis import given, settings, strategies as st
from jax.scipy.special import gammaln
from numpyro import handlers

from plumax._src.errors import ParameterError
from plumax.gauss_plume.dispersion import BRIGGS_DISPERSION_PARAMS
from plumax.gauss_plume.inference import gaussian_plume_model, infer_emission_rate
from plumax.gauss_plume.likelihoods import (
    GaussianNoise,
    HeteroscedasticNoise,
    StudentTNoise,
)
from plumax.gauss_plume.plume import plume_concentration
from plumax.gauss_puff import make_release_times, release_durations
from plumax.gauss_puff.inference import gaussian_puff_model, gaussian_puff_rw_model
from plumax.gauss_puff.wind import WindSchedule


NOISES = (
    GaussianNoise(std=0.7),
    StudentTNoise(scale=0.7, df=5.0),
    HeteroscedasticNoise(std=jnp.array([0.5, 1.0, 2.0])),
)


# ── closed-form oracles ──────────────────────────────────────────────────────


def test_gaussian_log_prob_closed_form():
    obs = jnp.array([0.3, -1.2, 2.0])
    mean = jnp.array([0.0, 0.5, 1.0])
    std = 0.7
    expected = (
        -0.5 * ((obs - mean) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
    )
    value = GaussianNoise(std=std).log_prob(obs, mean)
    np.testing.assert_allclose(value, expected, rtol=1e-12)


def test_student_t_log_prob_closed_form():
    obs = jnp.array([0.3, -1.2, 8.0])
    mean = jnp.array([0.0, 0.5, 1.0])
    scale, df = 0.7, 5.0
    z = (obs - mean) / scale
    expected = (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * jnp.log(df * jnp.pi)
        - jnp.log(scale)
        - (df + 1) / 2 * jnp.log1p(z**2 / df)
    )
    value = StudentTNoise(scale=scale, df=df).log_prob(obs, mean)
    np.testing.assert_allclose(value, expected, rtol=1e-12)


def test_heteroscedastic_log_prob_closed_form():
    obs = jnp.array([0.3, -1.2, 2.0])
    mean = jnp.zeros(3)
    std = jnp.array([0.5, 1.0, 2.0])
    expected = -0.5 * (obs / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
    value = HeteroscedasticNoise(std=std).log_prob(obs, mean)
    np.testing.assert_allclose(value, expected, rtol=1e-12)


@pytest.mark.parametrize("noise", NOISES, ids=lambda n: type(n).__name__)
def test_log_prob_matches_numpyro_distribution(noise):
    obs = jnp.array([0.3, -1.2, 2.0])
    mean = jnp.array([0.1, 0.5, -1.0])
    expected = noise.distribution(mean).log_prob(obs)
    np.testing.assert_allclose(noise.log_prob(obs, mean), expected, rtol=1e-12)


@pytest.mark.parametrize("noise", NOISES, ids=lambda n: type(n).__name__)
def test_call_observes_the_named_site(noise):
    obs = jnp.array([0.3, -1.2, 2.0])
    mean = jnp.array([0.1, 0.5, -1.0])
    trace = handlers.trace(lambda: noise("obs", mean, obs)).get_trace()
    site = trace["obs"]
    assert site["is_observed"]
    np.testing.assert_allclose(
        site["fn"].log_prob(obs), noise.log_prob(obs, mean), rtol=1e-12
    )


# ── properties, gradients, transforms ────────────────────────────────────────


@settings(max_examples=50, deadline=None)
@given(
    residual=st.floats(0.0, 50.0),
    extra=st.floats(0.0, 50.0),
    scale=st.floats(0.1, 10.0),
    df=st.floats(1.0, 30.0),
)
def test_student_t_symmetric_and_decreasing(residual, extra, scale, df):
    noise = StudentTNoise(scale=scale, df=df)
    mean = jnp.zeros(1)
    near = noise.log_prob(jnp.array([residual]), mean)
    mirrored = noise.log_prob(jnp.array([-residual]), mean)
    far = noise.log_prob(jnp.array([residual + extra]), mean)
    np.testing.assert_allclose(near, mirrored, rtol=1e-12)
    assert float(far[0]) <= float(near[0]) + 1e-12


@pytest.mark.parametrize("noise", NOISES, ids=lambda n: type(n).__name__)
def test_grad_matches_central_differences(noise):
    obs = jnp.array([0.3, -1.2, 2.0])
    mean = jnp.array([0.1, 0.5, -1.0])

    def total(m):
        return jnp.sum(noise.log_prob(obs, m))

    grad = jax.grad(total)(mean)
    step = 1e-6
    for i in range(3):
        bump = jnp.zeros(3).at[i].set(step)
        fd = (total(mean + bump) - total(mean - bump)) / (2 * step)
        np.testing.assert_allclose(grad[i], fd, rtol=1e-6)


@pytest.mark.parametrize("noise", NOISES, ids=lambda n: type(n).__name__)
def test_jit_and_vmap_match_loop(noise):
    obs = jnp.array([[0.3, -1.2, 2.0], [1.0, 0.0, -0.5]])
    mean = jnp.array([[0.1, 0.5, -1.0], [0.0, 0.2, 0.3]])
    loop = jnp.stack([noise.log_prob(o, m) for o, m in zip(obs, mean, strict=True)])
    jitted = jax.jit(lambda o, m: noise.log_prob(o, m))(obs[0], mean[0])
    vmapped = jax.vmap(noise.log_prob)(obs, mean)
    np.testing.assert_allclose(jitted, loop[0], rtol=1e-12)
    np.testing.assert_allclose(vmapped, loop, rtol=1e-12)


# ── validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "build",
    [
        lambda: GaussianNoise(std=0.0),
        lambda: GaussianNoise(std=jnp.array([1.0, -1.0])),
        lambda: StudentTNoise(scale=-1.0),
        lambda: StudentTNoise(scale=1.0, df=0.0),
        lambda: HeteroscedasticNoise(std=jnp.array([1.0, 0.0])),
    ],
)
def test_nonpositive_parameters_rejected(build):
    with pytest.raises(ParameterError, match="must be > 0"):
        build()


def test_heteroscedastic_shape_mismatch_rejected():
    noise = HeteroscedasticNoise(std=jnp.ones(3))
    with pytest.raises(ParameterError, match="shape"):
        noise.log_prob(jnp.zeros(4), jnp.zeros(4))
    with pytest.raises(ParameterError, match="shape"):
        noise("obs", jnp.zeros(4), jnp.zeros(4))


def test_noise_built_under_jit():
    # A traced scale skips the eager check rather than raising on a tracer.
    value = jax.jit(lambda s: GaussianNoise(std=s).log_prob(jnp.zeros(1), jnp.ones(1)))(
        2.0
    )
    assert value.shape == (1,)


# ── defaults keep today's numbers ────────────────────────────────────────────


def _plume_inputs():
    x = jnp.linspace(200.0, 2000.0, 12)
    coords = (x, jnp.zeros_like(x), jnp.ones_like(x))
    source = (0.0, 0.0, 2.0)
    clean = plume_concentration(
        *coords, *source, 5.0, 0.0, 0.15, BRIGGS_DISPERSION_PARAMS["D"]
    )
    return coords, source, clean


def _obs_log_prob(model, **kwargs):
    values = {"emission_rate": jnp.array(0.12), "background": jnp.array(1e-7)}
    substituted = handlers.substitute(model, data=values)
    trace = handlers.trace(handlers.seed(substituted, 0)).get_trace(**kwargs)
    site = trace["obs"]
    return site["fn"].log_prob(site["value"]), site["fn"]


def test_plume_default_is_previous_gaussian_likelihood():
    coords, source, clean = _plume_inputs()
    obs = clean + 3e-8
    kwargs = {
        "observations": obs,
        "receptor_coords": coords,
        "source_location": source,
        "wind_u": 5.0,
        "wind_v": 0.0,
        "stability_class": "D",
        "obs_noise_std": 4e-8,
    }
    value, fn = _obs_log_prob(gaussian_plume_model, **kwargs)
    # The likelihood before this module: Normal(f_plume + background, σ_obs).
    predicted = (
        plume_concentration(
            *coords, *source, 5.0, 0.0, 0.12, BRIGGS_DISPERSION_PARAMS["D"]
        )
        + 1e-7
    )
    expected = dist.Normal(predicted, 4e-8).log_prob(obs)
    assert isinstance(fn, dist.Normal)
    np.testing.assert_allclose(value, expected, rtol=1e-12)


def test_plume_noise_override_reaches_the_site():
    coords, source, clean = _plume_inputs()
    noise = StudentTNoise(scale=4e-8)
    _, fn = _obs_log_prob(
        gaussian_plume_model,
        observations=clean,
        receptor_coords=coords,
        source_location=source,
        wind_u=5.0,
        wind_v=0.0,
        noise=noise,
    )
    assert isinstance(fn, dist.StudentT)
    np.testing.assert_allclose(fn.scale, 4e-8)


def _puff_inputs():
    schedule = WindSchedule.from_speed_direction(
        jnp.array([0.0, 600.0]), jnp.full(2, 4.0), jnp.full(2, 270.0)
    )
    release_times = make_release_times(0.0, 300.0, 0.05)
    x = jnp.linspace(100.0, 600.0, 5)
    return {
        "receptor_coords": (x, jnp.zeros_like(x), jnp.ones_like(x)),
        "observation_times": jnp.full(5, 300.0),
        "source_location": (0.0, 0.0, 2.0),
        "schedule": schedule,
        "release_times": release_times,
        "release_interval": 20.0,
        "puff_durations": release_durations(release_times, 300.0),
        "obs_noise_std": 4e-8,
    }


@pytest.mark.parametrize("model", [gaussian_puff_model, gaussian_puff_rw_model])
def test_puff_default_is_previous_gaussian_likelihood(model):
    inputs = _puff_inputs()
    obs = jnp.full(5, 1e-7)
    values = {
        "emission_rate": jnp.array(0.12),
        "background": jnp.array(1e-7),
        "innovations": jnp.zeros(inputs["release_times"].shape[0]),
    }
    substituted = handlers.substitute(model, data=values)
    trace = handlers.trace(handlers.seed(substituted, 0)).get_trace(
        observations=obs, **inputs
    )
    site = trace["obs"]
    assert isinstance(site["fn"], dist.Normal)
    np.testing.assert_allclose(site["fn"].scale, 4e-8)
    explicit = handlers.trace(
        handlers.seed(handlers.substitute(model, data=values), 0)
    ).get_trace(observations=obs, noise=GaussianNoise(std=4e-8), **inputs)
    np.testing.assert_allclose(
        site["fn"].log_prob(obs), explicit["obs"]["fn"].log_prob(obs), rtol=1e-12
    )


# ── robustness (the issue's Definition of Done) ──────────────────────────────


@pytest.mark.slow
def test_student_t_robust_to_outliers():
    """5 % positive outliers: Student-t recovers Q within σ_post, Gaussian does not."""
    params = BRIGGS_DISPERSION_PARAMS["D"]
    source = (0.0, 0.0, 2.0)
    x_obs = jnp.linspace(200.0, 2000.0, 40)
    y_obs = jnp.zeros_like(x_obs)
    z_obs = jnp.ones_like(x_obs)
    true_q = 0.15
    clean = np.asarray(
        plume_concentration(x_obs, y_obs, z_obs, *source, 5.0, 0.0, true_q, params)
    )
    rng = np.random.default_rng(0)
    noise_std = 2e-8
    noisy = clean + rng.normal(0.0, noise_std, size=clean.shape)
    # 5 % of the pixels (2 of 40) carry a large positive retrieval artefact.
    noisy[[3, 7]] += 0.5 * clean[[3, 7]]

    def posterior(noise):
        samples = infer_emission_rate(
            noisy,
            (np.asarray(x_obs), np.asarray(y_obs), np.asarray(z_obs)),
            source_location=source,
            wind_speed=5.0,
            wind_direction=270.0,
            stability_class="D",
            prior_mean=0.1,
            prior_std=0.08,
            num_warmup=300,
            num_samples=500,
            seed=0,
            noise=noise,
        )
        q = samples["emission_rate"]
        return float(q.mean()), float(q.std())

    student_mean, student_std = posterior(StudentTNoise(scale=noise_std, df=5.0))
    gaussian_mean, gaussian_std = posterior(GaussianNoise(std=noise_std))
    assert abs(student_mean - true_q) < student_std
    assert abs(gaussian_mean - true_q) > gaussian_std
