"""Tests for masked-K multi-source plumes and their inference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from jaxtyping import TypeCheckError
from numpyro.infer import Predictive

from plumax._src.errors import ParameterError
from plumax.gauss_plume.dispersion import BRIGGS_DISPERSION_PARAMS
from plumax.gauss_plume.inference import (
    infer_emission_rate,
    infer_multi_source,
    multi_source_plume_model,
)
from plumax.gauss_plume.multi_source import MaskedSources, MultiSourcePosterior
from plumax.gauss_plume.plume import plume_concentration
from plumax.population import event_from_posterior


PARAMS = BRIGGS_DISPERSION_PARAMS["D"]


def _receptor_grid(n_x=8, n_y=19):
    x, y = jnp.meshgrid(
        jnp.linspace(200.0, 2000.0, n_x), jnp.linspace(-900.0, 900.0, n_y)
    )
    x = x.ravel()
    return x, y.ravel(), jnp.ones_like(x)


# ── MaskedSources ────────────────────────────────────────────────────────────


def test_concentration_is_the_sum_of_active_plumes():
    receptors = _receptor_grid(4, 5)
    sources = MaskedSources(
        emission_rate=jnp.array([0.1, 0.3, 0.2]),
        x=jnp.array([0.0, 10.0, -20.0]),
        y=jnp.array([0.0, 200.0, -200.0]),
        active=jnp.array([1.0, 0.0, 1.0]),
    )
    field = sources.concentration(
        receptors, source_z=2.0, wind_u=5.0, wind_v=0.0, dispersion_params=PARAMS
    )
    expected = sum(
        float(a) * plume_concentration(*receptors, sx, sy, 2.0, 5.0, 0.0, q, PARAMS)
        for q, sx, sy, a in zip(
            sources.emission_rate, sources.x, sources.y, sources.active, strict=True
        )
    )
    np.testing.assert_allclose(field, expected, rtol=1e-12)
    assert float(sources.active_count()) == 2.0
    assert sources.k_max == 3


@settings(max_examples=25, deadline=None)
@given(scale=st.floats(0.0, 10.0), rate=st.floats(0.0, 5.0))
def test_inactive_slots_never_contribute(scale, rate):
    receptors = _receptor_grid(3, 3)
    on = MaskedSources(
        emission_rate=jnp.array([0.1]),
        x=jnp.zeros(1),
        y=jnp.zeros(1),
        active=jnp.ones(1),
    )
    with_off = MaskedSources(
        emission_rate=jnp.array([0.1, rate]),
        x=jnp.array([0.0, scale]),
        y=jnp.array([0.0, -scale]),
        active=jnp.array([1.0, 0.0]),
    )
    kwargs = {
        "source_z": 2.0,
        "wind_u": 5.0,
        "wind_v": 0.0,
        "dispersion_params": PARAMS,
    }
    np.testing.assert_allclose(
        with_off.concentration(receptors, **kwargs),
        on.concentration(receptors, **kwargs),
        rtol=1e-12,
    )


def test_concentration_is_linear_in_rates_with_exact_gradient():
    receptors = _receptor_grid(3, 3)
    kwargs = {
        "source_z": 2.0,
        "wind_u": 5.0,
        "wind_v": 0.0,
        "dispersion_params": PARAMS,
    }

    def total(rates):
        sources = MaskedSources(
            emission_rate=rates,
            x=jnp.array([0.0, 0.0]),
            y=jnp.array([0.0, 300.0]),
            active=jnp.array([1.0, 1.0]),
        )
        return jnp.sum(sources.concentration(receptors, **kwargs))

    rates = jnp.array([0.1, 0.2])
    grad = jax.grad(total)(rates)
    step = 1e-6
    for k in range(2):
        bump = jnp.zeros(2).at[k].set(step)
        fd = (total(rates + bump) - total(rates - bump)) / (2 * step)
        np.testing.assert_allclose(grad[k], fd, rtol=1e-6)


def test_jit_matches_eager():
    receptors = _receptor_grid(3, 3)
    sources = MaskedSources(
        emission_rate=jnp.array([0.1, 0.2]),
        x=jnp.array([0.0, 0.0]),
        y=jnp.array([0.0, 300.0]),
        active=jnp.array([1.0, 1.0]),
    )

    def run(s):
        return s.concentration(
            receptors, source_z=2.0, wind_u=5.0, wind_v=0.0, dispersion_params=PARAMS
        )

    np.testing.assert_allclose(jax.jit(run)(sources), run(sources), rtol=1e-12)


def test_mismatched_slots_rejected():
    # the jaxtyping hook catches it first in tests; plain installs get ours
    with pytest.raises((ParameterError, TypeCheckError)):
        MaskedSources(
            emission_rate=jnp.ones(2),
            x=jnp.ones(3),
            y=jnp.ones(2),
            active=jnp.ones(2),
        )


# ── MultiSourcePosterior ─────────────────────────────────────────────────────


def _posterior():
    return MultiSourcePosterior(
        emission_rate=np.array([[0.1, 0.5, 9.0], [0.3, 0.7, 9.0]]),
        x=np.array([[0.0, 10.0, 0.0], [2.0, 14.0, 0.0]]),
        y=np.zeros((2, 3)),
        active=np.array([[True, True, False], [True, False, False]]),
        background=np.zeros(2),
    )


def test_count_probabilities_and_saturation():
    posterior = _posterior()
    np.testing.assert_allclose(posterior.count_probabilities(), [0.0, 0.5, 0.5, 0.0])
    assert posterior.saturated_fraction() == 0.0
    np.testing.assert_array_equal(posterior.active_count(), [2, 1])


def test_per_source_conditions_on_activity():
    posterior = _posterior()
    first = posterior.per_source(0)
    assert first.probability_active == 1.0
    np.testing.assert_allclose(first.emission_rate, 0.2)
    np.testing.assert_allclose(first.emission_std, 0.1)
    np.testing.assert_allclose(first.x, 1.0)
    second = posterior.per_source(1)
    assert second.probability_active == 0.5
    np.testing.assert_allclose(second.emission_rate, 0.5)
    never = posterior.per_source(2)
    assert never.probability_active == 0.0
    assert np.isnan(never.emission_rate)
    with pytest.raises(ParameterError, match="slot"):
        posterior.per_source(3)


def test_per_source_feeds_the_population_catalogue():
    summary = _posterior().per_source(0)
    event = event_from_posterior(summary, x=summary.x, y=summary.y, time=0.0, tier="I")
    np.testing.assert_allclose(event.emission_rate, 0.2)
    np.testing.assert_allclose(event.emission_std, 0.1)


# ── the model ────────────────────────────────────────────────────────────────


def test_free_slots_are_ordered_west_to_east():
    receptors = _receptor_grid(3, 3)
    draws = Predictive(multi_source_plume_model, num_samples=200)(
        jax.random.PRNGKey(0),
        receptor_coords=receptors,
        wind_u=5.0,
        wind_v=0.0,
        k_max=4,
        source_bounds=((-500.0, 500.0), (-300.0, 300.0)),
    )
    source_x = np.asarray(draws["source_x"])
    assert source_x.shape == (200, 4)
    assert np.all(np.diff(source_x, axis=1) >= 0.0)
    assert np.all((source_x > -500.0) & (source_x < 500.0))
    assert set(np.unique(np.asarray(draws["active"]))) <= {0.0, 1.0}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "candidate_locations": jnp.zeros((2, 2)),
            "source_bounds": ((0.0, 1.0), (0.0, 1.0)),
        },
        {"candidate_locations": jnp.zeros((3, 2))},
        {"source_bounds": ((1.0, 0.0), (0.0, 1.0))},
        {"source_bounds": ((0.0, 1.0), (0.0, 1.0)), "prior_active": 0.0},
    ],
)
def test_invalid_layouts_rejected(kwargs):
    with pytest.raises(ValueError, match="multi_source_plume_model"):
        multi_source_plume_model(
            receptor_coords=_receptor_grid(2, 2),
            wind_u=5.0,
            wind_v=0.0,
            k_max=2,
            **kwargs,
        )


def test_k1_matches_single_source():
    """One always-on slot at the known source is the single-source posterior."""
    x_obs = jnp.linspace(200.0, 2000.0, 24)
    coords = (np.asarray(x_obs), np.zeros(24), np.ones(24))
    clean = plume_concentration(
        x_obs, jnp.zeros(24), jnp.ones(24), 0.0, 0.0, 2.0, 5.0, 0.0, 0.15, PARAMS
    )
    rng = np.random.default_rng(0)
    noisy = np.asarray(clean) + rng.normal(0.0, 2e-8, size=24)
    common = {
        "stability_class": "D",
        "prior_mean": 0.1,
        "prior_std": 0.08,
        "obs_noise_std": 2e-8,
        "num_warmup": 300,
        "num_samples": 1000,
        "seed": 0,
    }
    single = infer_emission_rate(
        noisy,
        coords,
        source_location=(0.0, 0.0, 2.0),
        wind_speed=5.0,
        wind_direction=270.0,
        **common,
    )
    multi = infer_multi_source(
        noisy,
        coords,
        wind_speed=5.0,
        wind_direction=270.0,
        k_max=1,
        candidate_locations=np.array([[0.0, 0.0]]),
        source_height=2.0,
        prior_active=1.0,
        **common,
    )
    q_single = single["emission_rate"]
    q_multi = multi.emission_rate[:, 0]
    # Same model, different chains: the means agree to Monte Carlo error, which
    # for a few hundred effective draws is under a quarter of a posterior σ.
    assert abs(q_multi.mean() - q_single.mean()) < 0.25 * q_single.std()
    np.testing.assert_allclose(q_multi.std(), q_single.std(), rtol=0.2)
    np.testing.assert_array_equal(multi.count_probabilities(), [0.0, 1.0])


@pytest.mark.slow
def test_three_sources_recovered():
    """Five candidate facilities, three emitting: K = 3 and each Q_k within σ_post."""
    receptors = _receptor_grid()
    candidates = np.array([[0.0, y] for y in (-600.0, -300.0, 0.0, 300.0, 600.0)])
    true_active = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    true_rates = np.array([0.1, 0.10, 0.20, 0.15, 0.1])
    truth = MaskedSources(
        emission_rate=jnp.asarray(true_rates),
        x=jnp.asarray(candidates[:, 0]),
        y=jnp.asarray(candidates[:, 1]),
        active=jnp.asarray(true_active),
    )
    clean = truth.concentration(
        receptors, source_z=2.0, wind_u=5.0, wind_v=0.0, dispersion_params=PARAMS
    )
    rng = np.random.default_rng(1)
    noisy = np.asarray(clean) + rng.normal(0.0, 2e-8, size=clean.shape)

    posterior = infer_multi_source(
        noisy,
        tuple(np.asarray(c) for c in receptors),
        wind_speed=5.0,
        wind_direction=270.0,
        k_max=5,
        candidate_locations=candidates,
        source_height=2.0,
        stability_class="D",
        prior_mean=0.1,
        prior_std=0.08,
        obs_noise_std=2e-8,
        num_warmup=500,
        num_samples=1000,
        seed=0,
    )
    assert posterior.count_probabilities()[3] >= 0.9
    for k in (1, 2, 3):
        summary = posterior.per_source(k)
        assert summary.probability_active > 0.9
        assert abs(summary.emission_rate - true_rates[k]) < summary.emission_std
    for k in (0, 4):
        assert posterior.per_source(k).probability_active < 0.1
