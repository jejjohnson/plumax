"""The nonlinear end-to-end coupled inversion on vardax (#102).

Toy problems only: a Tier I plume seen by a column instrument and a radiance
instrument on a handful of pixels.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.coupled import (
    CoupledPrior,
    CoupledProblem,
    Instrument,
    LinearisedRadianceOperator,
    PlumeSource,
    QualityFlag,
    RadianceObservationOperator,
    apply_quality,
    build_coupled_forward,
    column_response,
    fuse_observations,
    solve_coupled,
)
from plumax.coupled.inversion import plume_response
from plumax.population.catalog import event_from_posterior


pytest.importorskip("vardax")

SOURCE = PlumeSource(
    location=(0.0, 0.0, 10.0),
    wind_speed=5.0,
    wind_direction=270.0,
    stability_class="D",
    column_z=jnp.linspace(0.0, 200.0, 11),
    background=0.0,
)


def _grid(xs, ys):
    gx, gy = np.meshgrid(np.asarray(xs), np.asarray(ys), indexing="ij")
    return np.column_stack([gx.reshape(-1), gy.reshape(-1)])


def _column_instrument(variance, name="TROPOMI"):
    return Instrument(
        name,
        _grid([200.0, 400.0, 700.0], [-60.0, 0.0, 60.0]),
        retrieval_variance=variance,
    )


def _radiance_instrument(name="EMIT"):
    return Instrument(name, _grid([150.0, 300.0, 500.0], [-30.0, 30.0]))


class _BeerLambert(eqx.Module):
    """Toy JAX radiance operator: L_c = exp(−k_c m) per channel."""

    absorption: jax.Array

    def predict(self, column_enhancement):
        optical_depth = column_enhancement[:, None] * self.absorption[None, :]
        return jnp.exp(-optical_depth)


def _scale():
    # a typical column of the truth, to set noise and absorption levels
    return float(jnp.max(column_response(SOURCE, _column_instrument(1.0))))


# ── linear limit ─────────────────────────────────────────────────────────────


def test_linear_limit_equals_fusion():
    scale = _scale()
    variance = (0.1 * scale) ** 2
    instruments = [_column_instrument(variance), _column_instrument(4 * variance, "B")]
    forward = build_coupled_forward(SOURCE, instruments)
    key = jax.random.PRNGKey(0)
    clean = forward.predict(2.0, [0.05 * scale, -0.05 * scale])
    noise = jax.random.normal(key, (18,))
    observations = [
        clean[0] + 0.1 * scale * noise[:9],
        clean[1] + 0.2 * scale * noise[9:],
    ]
    prior_std = {"emission_rate": 3.0, "biases": 0.2 * scale}
    prior = CoupledPrior.from_std(
        {"emission_rate": 1.0, "biases": jnp.zeros(2)}, prior_std
    )
    result = solve_coupled(
        CoupledProblem(
            forward=forward,
            prior=prior,
            observations={"TROPOMI": observations[0], "B": observations[1]},
        )
    )
    closed_form = fuse_observations(
        forward,
        observations,
        prior_mean=jnp.array([1.0, 0.0, 0.0]),
        prior_covariance=jnp.diag(
            jnp.array([9.0, (0.2 * scale) ** 2, (0.2 * scale) ** 2])
        ),
    )
    np.testing.assert_allclose(
        result.map_estimate["emission_rate"], closed_form.emission_rate, rtol=1e-6
    )
    np.testing.assert_allclose(
        result.map_estimate["biases"], closed_form.biases, rtol=1e-6, atol=1e-6 * scale
    )
    # layout is sorted by block name: [biases₁, biases₂, emission_rate]
    order = jnp.array([2, 0, 1])
    np.testing.assert_allclose(
        result.covariance[jnp.ix_(order, order)], closed_form.covariance, rtol=1e-6
    )
    assert result.emission_rate == result.map_estimate["emission_rate"]
    assert result.pixels_used == {"TROPOMI": 9, "B": 9}


def test_linear_radiance_laplace_equals_closed_form_gaussian():
    scale = _scale()
    gain = -jnp.array([0.2, 0.5]) / scale
    operator = LinearisedRadianceOperator(baseline=jnp.ones(2), gain=gain)
    instruments = [_column_instrument((0.1 * scale) ** 2), _radiance_instrument()]
    forward = build_coupled_forward(SOURCE, instruments)
    mean = {"emission_rate": 1.0, "background": 0.0, "biases": jnp.zeros(2)}
    # a dense prior: the second bias and Q correlate (ρ = 0.3)
    covariance = jnp.array(
        [
            [(0.1 * scale) ** 2, 0.0, 0.0, 0.0],
            [0.0, (0.1 * scale) ** 2, 0.0, 0.0],
            [0.0, 0.0, (0.2 * scale) ** 2, 0.3 * 0.2 * scale * 3.0],
            [0.0, 0.0, 0.3 * 0.2 * scale * 3.0, 9.0],
        ]
    )
    prior = CoupledPrior(
        mean={k: jnp.asarray(v) for k, v in mean.items()}, covariance=covariance
    )
    assert prior.layout == (("background", 1), ("biases", 2), ("emission_rate", 1))
    truth = {
        "emission_rate": 2.5,
        "background": 0.05 * scale,
        "biases": jnp.array([0.02, -0.03]) * scale,
    }
    problem_truth = CoupledProblem(
        forward=forward,
        prior=prior,
        observations={"TROPOMI": jnp.zeros(9), "EMIT": jnp.zeros((6, 2))},
        radiance_ops={"EMIT": operator},
        radiance_variance={"EMIT": 1e-4},
    )
    clean = problem_truth.predict({k: jnp.asarray(v) for k, v in truth.items()})
    observations = {
        "TROPOMI": clean["TROPOMI"] + 0.1 * scale * jnp.linspace(-1.0, 1.0, 9),
        "EMIT": clean["EMIT"] + 1e-2 * jnp.cos(jnp.arange(12.0)).reshape(6, 2),
    }
    problem = CoupledProblem(
        forward=forward,
        prior=prior,
        observations=observations,
        radiance_ops={"EMIT": operator},
        radiance_variance={"EMIT": 1e-4},
    )
    result = solve_coupled(problem)

    # closed form: the forward is affine in θ, h(θ) = h(μ) + G (θ − μ)
    prior_mean, unflatten = prior.flatten()
    design = jax.jacfwd(lambda theta: problem.predict_stacked(unflatten(theta)))(
        prior_mean
    )
    stacked = problem.stacked_observations()
    precision_obs = jnp.diag(1.0 / stacked.variance)
    precision = jnp.linalg.inv(covariance) + design.T @ precision_obs @ design
    posterior_covariance = jnp.linalg.inv(precision)
    innovation = stacked.values - problem.predict_stacked(unflatten(prior_mean))
    posterior_mean = (
        prior_mean + posterior_covariance @ design.T @ precision_obs @ innovation
    )
    np.testing.assert_allclose(
        result.mean, posterior_mean, rtol=1e-6, atol=1e-9 * scale
    )
    np.testing.assert_allclose(
        result.covariance, posterior_covariance, rtol=1e-6, atol=1e-12
    )


# ── nonlinear recovery ───────────────────────────────────────────────────────


def _nonlinear_problem(instrument_names, *, seed=1):
    scale = _scale()
    instruments = {
        "TROPOMI": _column_instrument((0.05 * scale) ** 2),
        "EMIT": _radiance_instrument(),
    }
    forward = build_coupled_forward(SOURCE, [instruments[n] for n in instrument_names])
    operators = {"EMIT": _BeerLambert(absorption=jnp.array([0.3, 0.8]) / scale)}
    radiance_ops = {n: operators[n] for n in instrument_names if n in operators}
    variances = {n: 1e-4 for n in radiance_ops}
    n_inst = len(instrument_names)
    prior = CoupledPrior.from_std(
        {
            "log_emission_rate": jnp.log(1.0),
            "location": jnp.zeros(2),
            "wind": jnp.array([5.0, 270.0]),
            "biases": jnp.zeros(n_inst),
        },
        {
            "log_emission_rate": 1.5,
            "location": 30.0,
            "wind": jnp.array([1.0, 8.0]),
            "biases": 0.05 * scale,
        },
    )
    truth = {
        "log_emission_rate": jnp.log(3.0),
        "location": jnp.array([15.0, -10.0]),
        "wind": jnp.array([5.5, 265.0]),
        "biases": jnp.full(n_inst, 0.02 * scale),
    }
    placeholder = {
        n: jnp.zeros((6, 2)) if n in radiance_ops else jnp.zeros(9)
        for n in instrument_names
    }
    clean = CoupledProblem(
        forward=forward,
        prior=prior,
        observations=placeholder,
        radiance_ops=radiance_ops,
        radiance_variance=variances,
    ).predict(truth)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_inst)
    observations = {}
    for key, name in zip(keys, instrument_names, strict=True):
        sigma = 1e-2 if name in radiance_ops else 0.05 * scale
        observations[name] = clean[name] + sigma * jax.random.normal(
            key, clean[name].shape
        )
    problem = CoupledProblem(
        forward=forward,
        prior=prior,
        observations=observations,
        radiance_ops=radiance_ops,
        radiance_variance=variances,
    )
    return problem, truth


def test_nonlinear_recovery_within_posterior_uncertainty():
    widths = {}
    for names in (("TROPOMI",), ("TROPOMI", "EMIT")):
        problem, truth = _nonlinear_problem(names)
        result = solve_coupled(problem)
        assert result.std is not None
        for block in ("log_emission_rate", "location", "wind"):
            z = jnp.abs(result.map_estimate[block] - truth[block]) / result.std[block]
            assert bool(jnp.all(z < 3.0)), (names, block, z)
        widths[len(names)] = float(result.std["log_emission_rate"])
        assert bool(jnp.isfinite(result.emission_std))
    # the second instrument tightens log Q
    assert widths[2] < widths[1]


def test_lognormal_q_feeds_tier_v():
    problem, _ = _nonlinear_problem(("TROPOMI",))
    result = solve_coupled(problem)
    log_mean = result.map_estimate["log_emission_rate"]
    log_std = result.std["log_emission_rate"]
    np.testing.assert_allclose(
        result.emission_rate, jnp.exp(log_mean + 0.5 * log_std**2), rtol=1e-12
    )
    event = event_from_posterior(result, x=0.0, y=0.0, time=0.0, tier="IV")
    assert event.emission_rate == pytest.approx(float(result.emission_rate))
    assert event.emission_std == pytest.approx(float(result.emission_std))


# ── masks from coupled.quality ───────────────────────────────────────────────


def test_masked_pixels_match_masked_fusion():
    scale = _scale()
    variance = (0.1 * scale) ** 2
    flags = jnp.zeros(9, dtype=int).at[jnp.array([1, 4])].set(QualityFlag.CLOUD)
    instruments = [
        apply_quality(_column_instrument(variance), flags),
        apply_quality(
            _column_instrument(variance, "B"), jnp.full(9, QualityFlag.CLOUD)
        ),
    ]
    forward = build_coupled_forward(SOURCE, instruments)
    clean = forward.predict(2.0)
    observations = [clean[0].at[1].set(jnp.nan), jnp.full(9, jnp.nan)]
    prior = CoupledPrior.from_std(
        {"emission_rate": 1.0, "biases": jnp.zeros(2)},
        {"emission_rate": 3.0, "biases": 0.2 * scale},
    )
    result = solve_coupled(
        CoupledProblem(
            forward=forward,
            prior=prior,
            observations={"TROPOMI": observations[0], "B": observations[1]},
        )
    )
    closed_form = fuse_observations(
        forward,
        observations,
        prior_mean=jnp.array([1.0, 0.0, 0.0]),
        prior_covariance=jnp.diag(
            jnp.array([9.0, (0.2 * scale) ** 2, (0.2 * scale) ** 2])
        ),
    )
    np.testing.assert_allclose(
        result.map_estimate["emission_rate"], closed_form.emission_rate, rtol=1e-6
    )
    assert result.pixels_used == {"TROPOMI": 7, "B": 0}
    assert bool(jnp.isnan(result.per_instrument_residuals["B"]).all())
    assert int(jnp.isnan(result.per_instrument_residuals["TROPOMI"]).sum()) == 2


# ── plumbing ─────────────────────────────────────────────────────────────────


def test_plume_response_matches_column_response():
    instrument = _column_instrument(1.0)
    traced = plume_response(SOURCE, instrument, jnp.zeros(2), jnp.array([5.0, 270.0]))
    np.testing.assert_allclose(traced, column_response(SOURCE, instrument), rtol=1e-12)


class _MatrixSource:
    background = 0.0

    def column_response(self, instrument):
        return jnp.ones(instrument.n_obs)


def test_problem_validation():
    forward = build_coupled_forward(SOURCE, [_column_instrument(1.0)])
    prior = CoupledPrior.from_std({"emission_rate": 1.0}, {"emission_rate": 1.0})
    observations = {"TROPOMI": jnp.zeros(9)}
    with pytest.raises(NotImplementedError, match="vardax#72"):
        solve_coupled(
            CoupledProblem(forward=forward, prior=prior, observations=observations),
            method="gauss_newton",
        )
    numpy_operator = RadianceObservationOperator(
        ds=None,
        nu_obs=np.ones(2),
        T_K=1.0,
        p_atm=1.0,
        vmr_background=1.0,
        path_length_cm=1.0,
        amf=1.0,
    )
    with pytest.raises(TypeError, match="NumPy"):
        CoupledProblem(
            forward=forward,
            prior=prior,
            observations=observations,
            radiance_ops={"TROPOMI": numpy_operator},
            radiance_variance={"TROPOMI": 1.0},
        )
    with pytest.raises(ValueError, match="observations for"):
        CoupledProblem(forward=forward, prior=prior, observations={})
    with pytest.raises(ValueError, match="exactly one"):
        CoupledPrior.from_std({"biases": jnp.zeros(1)}, {"biases": 1.0})
    with pytest.raises(ValueError, match="covariance"):
        CoupledPrior(mean={"emission_rate": jnp.asarray(1.0)}, covariance=jnp.ones(2))
    other = build_coupled_forward(_MatrixSource(), [_column_instrument(1.0)])
    with pytest.raises(ValueError, match="PlumeSource"):
        CoupledProblem(
            forward=other,
            prior=CoupledPrior.from_std(
                {"emission_rate": 1.0, "wind": jnp.array([5.0, 270.0])},
                {"emission_rate": 1.0, "wind": 1.0},
            ),
            observations=observations,
        )


@pytest.mark.slow
def test_holdout_instrument():
    scale = _scale()
    variance = (0.05 * scale) ** 2
    held_out = Instrument(
        "HELD", _grid([250.0, 450.0, 650.0], [-40.0, 40.0]), retrieval_variance=variance
    )
    used = _column_instrument(variance)
    truth = {"log_emission_rate": jnp.log(3.0), "location": jnp.array([10.0, 5.0])}
    full = build_coupled_forward(SOURCE, [used, held_out])
    prior = CoupledPrior.from_std(
        {"log_emission_rate": 0.0, "location": jnp.zeros(2)},
        {"log_emission_rate": 1.5, "location": 30.0},
    )
    placeholder = {"TROPOMI": jnp.zeros(9), "HELD": jnp.zeros(6)}
    clean = CoupledProblem(forward=full, prior=prior, observations=placeholder).predict(
        truth
    )
    key_used, key_held = jax.random.split(jax.random.PRNGKey(3))
    observed = clean["TROPOMI"] + 0.05 * scale * jax.random.normal(key_used, (9,))
    held = clean["HELD"] + 0.05 * scale * jax.random.normal(key_held, (6,))
    result = solve_coupled(
        CoupledProblem(
            forward=build_coupled_forward(SOURCE, [used]),
            prior=prior,
            observations={"TROPOMI": observed},
        )
    )
    predicted = CoupledProblem(
        forward=full, prior=prior, observations=placeholder
    ).predict(result.map_estimate)["HELD"]
    rmse = float(jnp.sqrt(jnp.mean((predicted - held) ** 2)))
    assert rmse < 2.0 * float(jnp.sqrt(variance)), rmse
