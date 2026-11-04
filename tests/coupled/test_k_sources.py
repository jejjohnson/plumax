"""Masked-K multi-source handling in the coupled inversion (#104).

The geometry and rates were fixed before the first run, following Tier I's test:
five candidate facilities, three emitting, seen by two instruments.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.coupled import (
    CoupledPrior,
    CoupledProblem,
    Instrument,
    MaskedKPrior,
    PlumeSource,
    build_coupled_forward,
    build_masked_k_problem,
    infer_masked_k_sources,
    saturation_rate,
    slot_responses,
    solve_coupled,
)
from plumax.coupled.inversion import plume_response
from plumax.gauss_plume.dispersion import get_dispersion_params
from plumax.gauss_plume.inference import _lognormal_from_moments
from plumax.gauss_plume.multi_source import MaskedSources, MultiSourcePosterior


pytest.importorskip("vardax")
pytest.importorskip("numpyro")

SOURCE = PlumeSource(
    location=(0.0, 0.0, 2.0),
    wind_speed=5.0,
    wind_direction=270.0,
    stability_class="D",
    column_z=jnp.linspace(0.0, 200.0, 11),
    background=0.0,
)
CANDIDATES = jnp.array([[0.0, y] for y in (-600.0, -300.0, 0.0, 300.0, 600.0)])
TRUE_ACTIVE = jnp.array([False, True, True, True, False])
TRUE_RATES = jnp.array([0.1, 0.10, 0.20, 0.15, 0.1])
NOISE_STD = 2e-6


def _grid(xs, ys):
    grid_x, grid_y = np.meshgrid(np.asarray(xs), np.asarray(ys), indexing="ij")
    return np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])


def _instruments():
    fine = _grid(np.linspace(200.0, 1400.0, 5), np.linspace(-800.0, 800.0, 17))
    coarse = _grid(np.linspace(400.0, 2000.0, 3), np.linspace(-800.0, 800.0, 9))
    return [
        Instrument("fine", fine, retrieval_variance=NOISE_STD**2 * np.ones(len(fine))),
        Instrument(
            "coarse", coarse, retrieval_variance=NOISE_STD**2 * np.ones(len(coarse))
        ),
    ]


def _observations(instruments, active=TRUE_ACTIVE, rates=TRUE_RATES, seed=0):
    rng = np.random.default_rng(seed)
    observations = {}
    matrices = slot_responses(SOURCE, instruments, CANDIDATES)
    for instrument, matrix in zip(instruments, matrices, strict=True):
        clean = matrix @ (active * rates)
        noise = rng.normal(0.0, NOISE_STD, instrument.n_obs)
        observations[instrument.name] = clean + noise
    return observations


def test_slot_responses_are_the_single_source_columns():
    instruments = _instruments()
    matrices = slot_responses(SOURCE, instruments, CANDIDATES)
    wind = jnp.asarray([SOURCE.wind_speed, SOURCE.wind_direction])
    for instrument, matrix in zip(instruments, matrices, strict=True):
        assert matrix.shape == (instrument.n_obs, 5)
        for slot, location in enumerate(CANDIDATES):
            single = plume_response(SOURCE, instrument, location, wind)
            np.testing.assert_allclose(matrix[:, slot], single, rtol=1e-12)


def test_forward_is_linear_in_the_active_slots():
    # Σ_k a_k Q_k r(x_k, y_k) is Tier I's MaskedSources sum through the column
    instrument = _instruments()[0]
    matrix = slot_responses(SOURCE, [instrument], CANDIDATES)[0]
    levels = SOURCE.column_z
    sources = MaskedSources(
        emission_rate=TRUE_RATES,
        x=CANDIDATES[:, 0],
        y=CANDIDATES[:, 1],
        active=TRUE_ACTIVE.astype(float),
    )
    receptor = instrument.receptors[12]
    profile = sources.concentration(
        (
            jnp.full_like(levels, receptor[0]),
            jnp.full_like(levels, receptor[1]),
            levels,
        ),
        source_z=2.0,
        wind_u=5.0,
        wind_v=0.0,
        dispersion_params=get_dispersion_params("D"),
    )
    column = jnp.trapezoid(profile, levels)
    np.testing.assert_allclose(
        (matrix @ (TRUE_ACTIVE * TRUE_RATES))[12], column, rtol=1e-10
    )


@pytest.mark.slow
def test_k1_matches_single_source():
    # one active slot in the masked problem is the single-source coupled problem
    instruments = _instruments()
    location = CANDIDATES[2]
    only = TRUE_ACTIVE & jnp.array([False, False, True, False, False])
    observations = _observations(instruments, active=only)
    prior = MaskedKPrior(k_max=5, emission_mean=0.1, emission_std=0.08)
    masked = solve_coupled(
        build_masked_k_problem(
            instruments,
            transport=SOURCE,
            prior=prior,
            observations=observations,
            active=only,
            candidate_locations=CANDIDATES,
        )
    )
    single_source = PlumeSource(
        location=(float(location[0]), float(location[1]), 2.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=SOURCE.column_z,
        background=0.0,
    )
    log_mean, log_std = _lognormal_from_moments(0.1, 0.08)
    single = solve_coupled(
        CoupledProblem(
            forward=build_coupled_forward(single_source, instruments),
            prior=CoupledPrior.from_std(
                {"log_emission_rate": log_mean}, {"log_emission_rate": log_std}
            ),
            observations=observations,
        )
    )
    np.testing.assert_allclose(
        masked.map_estimate["log_emission_rate"][0],
        single.map_estimate["log_emission_rate"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        masked.std["log_emission_rate"][0],
        single.std["log_emission_rate"],
        rtol=1e-6,
    )


@pytest.mark.slow
def test_k1_gibbs_path_matches_the_laplace_posterior():
    # the NumPyro path with one always-on slot against vardax's Laplace posterior
    instruments = _instruments()
    location = CANDIDATES[2:3]
    only = jnp.array([True])
    rate = jnp.array([0.2])
    matrices = slot_responses(SOURCE, instruments, location)
    rng = np.random.default_rng(3)
    observations = {
        instrument.name: matrix @ rate + rng.normal(0.0, NOISE_STD, instrument.n_obs)
        for instrument, matrix in zip(instruments, matrices, strict=True)
    }
    prior = MaskedKPrior(
        k_max=1,
        prior_active=1.0,
        emission_mean=0.1,
        emission_std=0.08,
        background_std=1e-12,
    )
    posterior = infer_masked_k_sources(
        observations,
        instruments=instruments,
        transport=SOURCE,
        prior=prior,
        candidate_locations=location,
    )
    laplace = solve_coupled(
        build_masked_k_problem(
            instruments,
            transport=SOURCE,
            prior=prior,
            observations=observations,
            active=only,
            candidate_locations=location,
        )
    )
    draws = np.log(posterior.emission_rate[:, 0])
    center = float(laplace.map_estimate["log_emission_rate"][0])
    spread = float(laplace.std["log_emission_rate"][0])
    # the same posterior, up to the chain's Monte Carlo error
    assert abs(draws.mean() - center) < 0.25 * spread
    np.testing.assert_allclose(draws.std(), spread, rtol=0.2)
    np.testing.assert_array_equal(posterior.count_probabilities(), [0.0, 1.0])


@pytest.mark.slow
def test_three_sources_two_instruments():
    instruments = _instruments()
    observations = _observations(instruments, seed=1)
    posterior = infer_masked_k_sources(
        observations,
        instruments=instruments,
        transport=SOURCE,
        prior=MaskedKPrior(k_max=5, emission_mean=0.1, emission_std=0.08),
        candidate_locations=CANDIDATES,
        seed=0,
    )
    assert posterior.count_probabilities()[3] >= 0.9
    for slot in (1, 2, 3):
        summary = posterior.per_source(slot)
        assert summary.probability_active > 0.9
        rate = float(TRUE_RATES[slot])
        assert abs(summary.emission_rate - rate) < summary.emission_std
    for slot in (0, 4):
        assert posterior.per_source(slot).probability_active < 0.1


def test_masked_pixels_do_not_enter_the_likelihood():
    from numpyro import handlers
    from numpyro.infer.util import log_density

    from plumax.coupled import masked_k_coupled_model

    instruments = _instruments()
    observations = _observations(instruments)
    flagged = np.ones(instruments[0].n_obs)
    flagged[:10] = 0.0
    masked = [dataclasses.replace(instruments[0], mask=flagged), instruments[1]]
    corrupted = dict(observations)
    corrupted["fine"] = observations["fine"].at[:10].set(jnp.nan)
    arguments = {
        "transport": SOURCE,
        "k_max": 5,
        "candidate_locations": CANDIDATES,
        "prior_active": 1.0,
        "fixed_responses": slot_responses(SOURCE, masked, CANDIDATES),
    }
    params = {"emission_rate": TRUE_RATES, "background": 1e-7}
    model = handlers.substitute(masked_k_coupled_model, data=params)
    clean, _ = log_density(
        model, (observations,), {"instruments": masked, **arguments}, {}
    )
    dirty, _ = log_density(
        model, (corrupted,), {"instruments": masked, **arguments}, {}
    )
    np.testing.assert_allclose(dirty, clean, rtol=1e-12)


def test_saturation_rate():
    def posterior(active):
        shape = np.shape(active)
        return MultiSourcePosterior(
            emission_rate=np.ones(shape),
            x=np.zeros(shape),
            y=np.zeros(shape),
            active=np.asarray(active),
            background=np.zeros(shape[0]),
        )

    full = posterior([[True, True], [True, True], [True, False]])
    partial = posterior([[True, False], [True, False], [True, True]])
    assert saturation_rate([full, partial, partial, partial]) == 0.25
    with pytest.raises(ValueError, match="no posteriors"):
        saturation_rate([])


def test_invalid_arguments_are_rejected():
    instruments = _instruments()
    with pytest.raises(ValueError, match="k_max"):
        MaskedKPrior(k_max=0)
    with pytest.raises(ValueError, match="prior_active"):
        MaskedKPrior(prior_active=0.0)
    with pytest.raises(ValueError, match="candidate_locations"):
        slot_responses(SOURCE, instruments, None)
    with pytest.raises(ValueError, match="observations"):
        infer_masked_k_sources(
            {"fine": jnp.zeros(1)},
            instruments=instruments,
            transport=SOURCE,
            prior=MaskedKPrior(k_max=5),
            candidate_locations=CANDIDATES,
        )
    with pytest.raises(ValueError, match="at least one slot"):
        build_masked_k_problem(
            instruments,
            transport=SOURCE,
            prior=MaskedKPrior(k_max=5),
            observations=_observations(instruments),
            active=jnp.zeros(5, dtype=bool),
            candidate_locations=CANDIDATES,
        )


@pytest.mark.slow
def test_free_slots_rerun_the_plume_at_each_draw():
    from numpyro import handlers
    from numpyro.infer.util import log_density

    from plumax.coupled import masked_k_coupled_model

    instruments = _instruments()
    wind = jnp.asarray([SOURCE.wind_speed, SOURCE.wind_direction])
    location = jnp.array([0.0, 100.0])
    observations = {
        instrument.name: 0.2 * plume_response(SOURCE, instrument, location, wind)
        for instrument in instruments
    }

    def density(source_y):
        # x = x_low + (x_high − x_low) w₁ = −100 + 200 · ½ = 0
        params = {
            "x_spacing": jnp.array([0.5, 0.5]),
            "source_y": jnp.array([source_y]),
            "emission_rate": jnp.array([0.2]),
            "background": 1e-9,
        }
        model = handlers.substitute(masked_k_coupled_model, data=params)
        arguments = {
            "instruments": instruments,
            "transport": SOURCE,
            "k_max": 1,
            "source_bounds": ((-100.0, 100.0), (-500.0, 500.0)),
            "prior_active": 1.0,
        }
        value, _ = log_density(model, (observations,), arguments, {})
        return float(value)

    assert density(100.0) > density(150.0)
    assert density(100.0) > density(50.0)
