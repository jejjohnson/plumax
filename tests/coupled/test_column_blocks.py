"""Backend-declared control blocks and the column plume's line density (#268).

Toy scenes only: a column plume on a few hundred 60 m pixels.
"""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from plumax._src.errors import ParameterError
from plumax.coupled import (
    AbstractControlledSource,
    CoupledPrior,
    CoupledProblem,
    CoupledResult,
    Instrument,
    PlumeSource,
    build_coupled_forward,
    fuse_observations,
    solve_coupled,
)
from plumax.errors import ControlBlockError, PayloadError
from plumax.gauss_plume import ColumnPlume, ColumnPlumeSource, PlumeGeometry


PIXEL = 60.0
# w₀ = g / √12
INITIAL_WIDTH = PIXEL / math.sqrt(12.0)
LINE_DENSITY = 0.1
NOISE = 2e-4
BLOCKS = ("location", "direction", "log_reference_width", "width_exponent")


def _receptors(length=1800.0, half_width=600.0):
    xs = np.arange(-2 * PIXEL, length + 1.0, PIXEL)
    ys = np.arange(-half_width, half_width + 1.0, PIXEL)
    gx, gy = jnp.meshgrid(jnp.asarray(xs), jnp.asarray(ys), indexing="ij")
    return einx.rearrange("x y, x y -> (x y) (1 + 1)", gx, gy)


def _plume(reference_distance=900.0):
    return ColumnPlume(
        reference_distance=reference_distance,
        initial_width=INITIAL_WIDTH,
        upwind_scale=INITIAL_WIDTH,
        pixel_size=PIXEL,
    )


def _geometry(reference_width=120.0, width_exponent=0.8):
    return PlumeGeometry(
        location=jnp.zeros(2),
        direction=0.1,
        reference_width=reference_width,
        width_exponent=width_exponent,
    )


def _source(plume=None, geometry=None):
    return ColumnPlumeSource(
        plume=_plume() if plume is None else plume,
        geometry=_geometry() if geometry is None else geometry,
    )


def _instrument(receptors=None):
    receptors = _receptors() if receptors is None else receptors
    return Instrument("EMIT", receptors, retrieval_variance=NOISE**2)


def _prior_blocks(geometry):
    mean = {
        "log_line_density": jnp.log(LINE_DENSITY),
        "log_reference_width": jnp.log(geometry.reference_width),
        "width_exponent": geometry.width_exponent,
        "direction": geometry.direction,
        "location": geometry.location,
        "background": 0.0,
    }
    std = {
        "log_line_density": 2.0,
        "log_reference_width": 1.0,
        "width_exponent": 0.5,
        "direction": 0.3,
        "location": 60.0,
        "background": 1e-4,
    }
    return mean, std


def _problem(source, instrument, observation, prior):
    forward = build_coupled_forward(source, [instrument])
    return CoupledProblem(
        forward=forward, prior=prior, observations={instrument.name: observation}
    )


def _scene(key, source, instrument):
    # y = q r + ε
    clean = LINE_DENSITY * source.column_response(instrument)
    noise = NOISE * jax.random.normal(key, clean.shape)
    return clean + noise


def _result(**fields):
    # a hand-built log-line-density result, as solve_coupled returns one
    blocks = {
        "map_estimate": {"log_line_density": jnp.log(LINE_DENSITY)},
        "std": {"log_line_density": jnp.asarray(0.15)},
        "mean": jnp.log(LINE_DENSITY)[None],
        "covariance": None,
        "emission_rate": jnp.asarray(jnp.nan),
        "emission_std": jnp.asarray(jnp.nan),
        "converged": True,
        "n_steps": 1,
        "cost": 0.0,
        "per_instrument_residuals": {},
        "pixels_used": {},
    }
    # E[q] = exp(μ + σ²/2),  sd[q] = E[q] √(exp(σ²) − 1)
    line_density = LINE_DENSITY * math.exp(0.5 * 0.15**2)
    line_density_std = line_density * math.sqrt(math.expm1(0.15**2))
    blocks["line_density"] = jnp.asarray(line_density)
    blocks["line_density_std"] = jnp.asarray(line_density_std)
    blocks.update(fields)
    return CoupledResult(**blocks)


# ── declarations and validation ──────────────────────────────────────────────


def test_backends_declare_their_blocks():
    column = _source()
    plume = PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=jnp.linspace(0.0, 200.0, 11),
    )
    assert isinstance(column, AbstractControlledSource)
    assert isinstance(plume, AbstractControlledSource)
    assert dict(column.control_blocks) == {
        "location": (2,),
        "direction": (),
        "log_reference_width": (),
        "width_exponent": (),
    }
    assert column.emission_blocks == ("line_density", "log_line_density")
    assert dict(plume.control_blocks) == {"location": (2,), "wind": (2,)}
    assert plume.emission_blocks == ("emission_rate", "log_emission_rate")


def test_prior_must_match_the_declarations():
    source = _source()
    instrument = _instrument()
    observation = jnp.zeros(instrument.n_obs)
    mean, std = _prior_blocks(source.geometry)

    # Q and U are not separable in a column: emission_rate is refused
    rate_mean = {**mean, "log_emission_rate": 0.0}
    rate_std = {**std, "log_emission_rate": 1.0}
    del rate_mean["log_line_density"], rate_std["log_line_density"]
    with pytest.raises(ControlBlockError, match="log_line_density"):
        _problem(
            source, instrument, observation, CoupledPrior.from_std(rate_mean, rate_std)
        )

    # an undeclared block: the message lists the declared ones
    wind_prior = CoupledPrior.from_std(
        {**mean, "wind": jnp.array([3.5, 270.0])}, {**std, "wind": 1.0}
    )
    with pytest.raises(ControlBlockError, match="'direction', 'log_reference_width'"):
        _problem(source, instrument, observation, wind_prior)

    # a declared block of the wrong shape
    bent_prior = CoupledPrior.from_std(
        {**mean, "direction": jnp.zeros(2)}, {**std, "direction": 0.3}
    )
    with pytest.raises(ControlBlockError, match="`direction` must have shape"):
        _problem(source, instrument, observation, bent_prior)

    # both emission blocks
    both_prior = CoupledPrior.from_std(
        {**mean, "line_density": 0.1}, {**std, "line_density": 1.0}
    )
    with pytest.raises(ControlBlockError, match="exactly one"):
        _problem(source, instrument, observation, both_prior)


def test_with_wind_validation():
    result = _result()
    with pytest.raises(ParameterError, match="speed must be > 0"):
        result.with_wind(speed=0.0, speed_std=1.0)
    with pytest.raises(ParameterError, match="speed_std must be ≥ 0"):
        result.with_wind(speed=3.5, speed_std=-1.0)
    rate_only = _result(line_density=None, line_density_std=None)
    with pytest.raises(PayloadError, match="retrieved Q"):
        rate_only.with_wind(speed=3.5, speed_std=1.0)
    # the Tier V payload of a line density needs the wind first
    with pytest.raises(PayloadError, match="with_wind"):
        result.samples(jax.random.PRNGKey(0), 10)


# ── response_at properties ───────────────────────────────────────────────────


@settings(max_examples=25, deadline=None)
@given(
    direction=st.floats(-math.pi, math.pi),
    reference_width=st.floats(30.0, 400.0),
    width_exponent=st.floats(0.3, 1.2),
    x0=st.floats(-200.0, 200.0),
    y0=st.floats(-200.0, 200.0),
)
def test_response_at_properties(direction, reference_width, width_exponent, x0, y0):
    geometry = PlumeGeometry(
        location=jnp.array([x0, y0]),
        direction=direction,
        reference_width=reference_width,
        width_exponent=width_exponent,
    )
    source = _source(geometry=geometry)
    instrument = _instrument(_receptors(length=600.0, half_width=300.0))
    response = source.column_response(instrument)

    # an empty control is the source's own response
    np.testing.assert_array_equal(source.response_at(instrument, {}), response)

    # a control that repeats the geometry gives the same response
    repeated = {
        "location": geometry.location,
        "direction": geometry.direction,
        "log_reference_width": jnp.log(geometry.reference_width),
        "width_exponent": geometry.width_exponent,
    }
    np.testing.assert_allclose(
        source.response_at(instrument, repeated), response, rtol=1e-10, atol=1e-16
    )

    # h = q r + Ω_bg: scaling q scales the enhancement
    prior = CoupledPrior.from_std(
        {"line_density": 1.0, "background": 0.0},
        {"line_density": 1.0, "background": 1.0},
    )
    problem = _problem(source, instrument, jnp.zeros(instrument.n_obs), prior)
    control = {"line_density": jnp.asarray(0.1), "background": jnp.asarray(2e-3)}
    tripled = {**control, "line_density": jnp.asarray(0.3)}
    enhancement = problem.predict(control)["EMIT"] - 2e-3
    tripled_enhancement = problem.predict(tripled)["EMIT"] - 2e-3
    np.testing.assert_allclose(
        tripled_enhancement, 3.0 * enhancement, rtol=1e-10, atol=1e-16
    )


def _control(geometry):
    return {
        "location": jnp.array([15.0, -10.0]),
        "direction": jnp.asarray(0.15),
        "log_reference_width": jnp.log(110.0),
        "width_exponent": jnp.asarray(0.75),
    }


@pytest.mark.parametrize("block", BLOCKS)
def test_response_at_gradient_matches_central_differences(block):
    source = _source()
    instrument = _instrument(_receptors(length=600.0, half_width=300.0))
    control = _control(source.geometry)

    def response(value):
        return source.response_at(instrument, {**control, block: value})

    value = control[block]
    jacobian = jax.jacrev(response)(value)
    if block == "location":
        step = 1e-3
        # (r(v + h eₖ) − r(v − h eₖ)) / 2h, one per axis
        differences = [
            (response(value + step * unit) - response(value - step * unit)) / (2 * step)
            for unit in jnp.eye(2)
        ]
        expected = einx.rearrange("axis obs -> obs axis", jnp.stack(differences))
    else:
        step = 1e-6
        # (r(v + h) − r(v − h)) / 2h
        expected = (response(value + step) - response(value - step)) / (2 * step)
    scale = float(jnp.max(jnp.abs(expected)))
    np.testing.assert_allclose(jacobian, expected, rtol=1e-5, atol=1e-5 * scale)


def test_response_at_eager_jit_vmap_agree():
    source = _source()
    instrument = _instrument(_receptors(length=600.0, half_width=300.0))
    base = _control(source.geometry)
    offsets = jnp.linspace(-0.1, 0.1, 4)
    batch = {
        "location": einx.add(
            "member, axis -> member axis", 20 * offsets, base["location"]
        ),
        "direction": base["direction"] + offsets,
        "log_reference_width": base["log_reference_width"] + offsets,
        "width_exponent": base["width_exponent"] + offsets,
    }

    def response(control):
        return source.response_at(instrument, control)

    looped = jnp.stack(
        [
            response({name: value[member] for name, value in batch.items()})
            for member in range(offsets.shape[0])
        ]
    )
    jitted = jnp.stack(
        [
            jax.jit(response)({name: value[member] for name, value in batch.items()})
            for member in range(offsets.shape[0])
        ]
    )
    mapped = jax.vmap(response)(batch)
    np.testing.assert_allclose(jitted, looped, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(mapped, looped, rtol=1e-12, atol=1e-18)


def test_float32_in_gives_float32_out():
    geometry = PlumeGeometry(
        location=jnp.zeros(2, dtype=jnp.float32),
        direction=jnp.float32(0.1),
        reference_width=jnp.float32(120.0),
        width_exponent=jnp.float32(0.8),
    )
    source = _source(geometry=geometry)
    receptors = jnp.asarray(_receptors(length=600.0, half_width=300.0), jnp.float32)
    instrument = _instrument(receptors)
    control = {
        "direction": jnp.float32(0.2),
        "log_reference_width": jnp.log(jnp.float32(100.0)),
    }
    assert source.response_at(instrument, control).dtype == jnp.float32
    assert source.column_response(instrument).dtype == jnp.float32


# ── oracles ──────────────────────────────────────────────────────────────────


def test_fisher_of_rate_and_wind_has_rank_one():
    # gate G4: h = exp(log Q − log U) r + Ω_bg constrains only log Q − log U
    source = _source()
    instrument = _instrument()
    response = source.column_response(instrument)
    wind_speed = 3.5

    def predict(log_pair):
        # h = exp(log Q − log U) r
        return jnp.exp(log_pair[0] - log_pair[1]) * response

    jacobian = jax.jacfwd(predict)(
        jnp.log(jnp.array([LINE_DENSITY * wind_speed, wind_speed]))
    )
    # F = Jᵀ Σ_ε⁻¹ J
    fisher = einx.dot("obs a, obs b -> a b", jacobian, jacobian) / NOISE**2
    singular_values = jnp.linalg.svd(fisher, compute_uv=False)
    assert singular_values[1] < 1e-10 * singular_values[0]


@pytest.mark.slow
def test_linear_line_density_equals_fusion():
    # gate G5: with only q and one bias free, the problem is linear
    source = _source()
    instrument = _instrument(_receptors(length=900.0, half_width=300.0))
    response = source.column_response(instrument)
    bias = 2e-4
    noise = NOISE * jax.random.normal(jax.random.PRNGKey(3), response.shape)
    observation = LINE_DENSITY * response + bias + noise
    prior = CoupledPrior.from_std(
        {"line_density": 0.05, "biases": jnp.zeros(1)},
        {"line_density": 1.0, "biases": 1e-3},
    )
    problem = _problem(source, instrument, observation, prior)
    result = solve_coupled(problem, rtol=1e-12, atol=1e-14, max_steps=500)
    closed_form = fuse_observations(
        problem.forward,
        [observation],
        prior_mean=jnp.array([0.05, 0.0]),
        prior_covariance=jnp.diag(jnp.array([1.0, 1e-6])),
    )
    assert result.converged
    np.testing.assert_allclose(
        result.map_estimate["line_density"], closed_form.emission_rate, rtol=1e-8
    )
    np.testing.assert_allclose(
        result.map_estimate["biases"], closed_form.biases, rtol=1e-8, atol=1e-14
    )
    # a column solve leaves Q to the wind
    assert jnp.isnan(result.emission_rate)
    np.testing.assert_allclose(result.line_density, result.map_estimate["line_density"])


def test_with_wind_moments_match_monte_carlo():
    result = _result()
    speed, speed_std = 3.5, 1.0
    rate = result.with_wind(speed=speed, speed_std=speed_std)
    assert rate.wind == (speed, speed_std)

    n = 1_000_000
    line_key, wind_key = jax.random.split(jax.random.PRNGKey(7))
    # log q ~ N(μ_q, σ_q²)
    log_line_density = jnp.log(LINE_DENSITY) + 0.15 * jax.random.normal(line_key, (n,))
    # σ_U² = log(1 + s_U² / U²),  μ_U = log U − σ_U² / 2
    log_variance = math.log1p((speed_std / speed) ** 2)
    log_speed = math.log(speed) - 0.5 * log_variance
    wind = jnp.exp(
        log_speed + math.sqrt(log_variance) * jax.random.normal(wind_key, (n,))
    )
    # Q = q U
    draws = jnp.exp(log_line_density) * wind
    mean = jnp.mean(draws)
    variance = jnp.var(draws)
    mean_error = jnp.std(draws) / math.sqrt(n)
    variance_error = jnp.std(jnp.square(draws - mean)) / math.sqrt(n)
    assert abs(rate.emission_rate - mean) < 3 * mean_error
    assert abs(rate.emission_std**2 - variance) < 3 * variance_error

    # the payload draws log Q ~ N(μ_q + μ_U, σ_q² + σ_U²)
    samples = rate.samples(jax.random.PRNGKey(8), n)
    assert abs(jnp.mean(samples) - rate.emission_rate) < 3 * mean_error


def test_with_wind_without_spread_is_a_product():
    result = _result()
    rate = result.with_wind(speed=3.5, speed_std=0.0)
    np.testing.assert_allclose(
        rate.emission_rate, result.line_density * 3.5, rtol=1e-10
    )
    np.testing.assert_allclose(
        rate.emission_std, result.line_density_std * 3.5, rtol=1e-10
    )


def test_prior_logpdf_adds_the_wind():
    mean, std = {"log_line_density": jnp.log(LINE_DENSITY)}, {"log_line_density": 2.0}
    result = _result(prior=CoupledPrior.from_std(mean, std))
    rate = result.with_wind(speed=3.5, speed_std=1.0)
    emission = jnp.array([0.1, 0.35, 2.0])
    log_variance = math.log1p((1.0 / 3.5) ** 2)
    # log Q ~ N(μ_q,0 + μ_U, σ²_q,0 + σ_U²)
    center = math.log(LINE_DENSITY) + math.log(3.5) - 0.5 * log_variance
    variance = 4.0 + log_variance
    expected = (
        -jnp.log(emission)
        - 0.5 * jnp.log(2 * math.pi * variance)
        - jnp.square(jnp.log(emission) - center) / (2 * variance)
    )
    np.testing.assert_allclose(rate.prior_logpdf(emission), expected, rtol=1e-10)


# ── the full image fit ───────────────────────────────────────────────────────


def _full_problem(key, reference_distance=900.0, length=1800.0, geometry=None):
    geometry = _geometry() if geometry is None else geometry
    source = _source(plume=_plume(reference_distance), geometry=geometry)
    instrument = _instrument(_receptors(length=length))
    observation = _scene(key, source, instrument)
    mean, std = _prior_blocks(geometry)
    prior = CoupledPrior.from_std(mean, std)
    return _problem(source, instrument, observation, prior)


@pytest.mark.slow
def test_full_fit_recovers_the_geometry():
    problem = _full_problem(jax.random.PRNGKey(11))
    result = solve_coupled(problem)
    geometry = _geometry()
    truth = {
        "log_line_density": jnp.log(LINE_DENSITY),
        "log_reference_width": jnp.log(geometry.reference_width),
        "width_exponent": geometry.width_exponent,
        "direction": geometry.direction,
        "location": geometry.location,
    }
    assert result.converged
    for name, value in truth.items():
        error = jnp.abs(result.map_estimate[name] - value)
        assert bool(jnp.all(error < 4 * result.std[name])), name
    rate = result.with_wind(speed=3.5, speed_std=1.0)
    assert jnp.isfinite(rate.emission_rate)


def _width_correlation(reference_distance, length):
    # the same widths w(d) as the default, written at another reference distance
    base = _geometry()
    # w_ref' = w_ref (d_ref' / d_ref)^b
    reference_width = (
        base.reference_width * (reference_distance / 900.0) ** base.width_exponent
    )
    geometry = _geometry(reference_width=reference_width)
    problem = _full_problem(
        jax.random.PRNGKey(5),
        reference_distance=reference_distance,
        length=length,
        geometry=geometry,
    )
    result = solve_coupled(problem)
    offsets = {}
    offset = 0
    for name, size in problem.prior.layout:
        offsets[name] = offset
        offset += size
    covariance = result.covariance
    i, j = offsets["log_reference_width"], offsets["width_exponent"]
    # ρ = Σ_ij / √(Σ_ii Σ_jj)
    return float(covariance[i, j] / jnp.sqrt(covariance[i, i] * covariance[j, j]))


@pytest.mark.slow
def test_width_law_correlation_drops_at_half_the_plume_length():
    # a 600 m plume: d_ref = 1000 m lies beyond it, d_ref = 300 m is its middle
    outside = _width_correlation(1000.0, length=600.0)
    middle = _width_correlation(300.0, length=600.0)
    assert abs(middle) < abs(outside), (
        f"posterior correlation of log w_ref and b: {outside:.3f} at d_ref = 1000 m, "
        f"{middle:.3f} at d_ref = 300 m"
    )


@pytest.mark.validation
def test_line_density_interval_is_calibrated():
    # gate G3: the 1σ Laplace interval of log q covers the truth 68.3 % of the time
    n_scenes = 200
    hits = 0
    for scene in range(n_scenes):
        problem = _full_problem(jax.random.PRNGKey(1000 + scene))
        result = solve_coupled(problem)
        error = abs(
            float(result.map_estimate["log_line_density"]) - math.log(LINE_DENSITY)
        )
        hits += int(error < float(result.std["log_line_density"]))
        # each solve keeps ~340 memory maps of compiled code; 200 of them pass
        # the kernel's default limit of 65,530 and XLA aborts, while clearing
        # after every solve recompiles everything (16 s against 2 s warm)
        if scene % 50 == 49:
            jax.clear_caches()
    coverage = hits / n_scenes
    # 3 binomial standard errors around 0.683
    tolerance = 3 * math.sqrt(0.683 * 0.317 / n_scenes)
    assert abs(coverage - 0.683) < tolerance, f"coverage {coverage:.3f}"
