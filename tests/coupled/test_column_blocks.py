"""Backend-declared control blocks, the line density (#268) and the centreline (#270).

Toy scenes only: a column plume on a few hundred to a few thousand 60 m pixels.
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
from plumax.gauss_plume import (
    CentrelinePrior,
    ColumnPlume,
    ColumnPlumeSource,
    PlumeGeometry,
    centreline_offset,
    plume_frame,
)


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
    return einx.id("x y, x y -> (x y) (1 + 1)", gx, gy)


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


@pytest.mark.slow
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
    with pytest.raises(ParameterError, match="`speed` must be > 0"):
        result.with_wind(speed=0.0, speed_std=1.0)
    with pytest.raises(ParameterError, match="`speed_std` must be ≥ 0"):
        result.with_wind(speed=3.5, speed_std=-1.0)
    rate_only = _result(line_density=None, line_density_std=None)
    with pytest.raises(PayloadError, match="retrieved Q"):
        rate_only.with_wind(speed=3.5, speed_std=1.0)
    # the Tier V payload of a line density needs the wind first
    with pytest.raises(PayloadError, match="with_wind"):
        result.samples(jax.random.PRNGKey(0), 10)


# ── response_at properties ───────────────────────────────────────────────────


@pytest.mark.slow
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


@pytest.mark.slow
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
        expected = einx.id("axis obs -> obs axis", jnp.stack(differences))
    else:
        step = 1e-6
        # (r(v + h) − r(v − h)) / 2h
        expected = (response(value + step) - response(value - step)) / (2 * step)
    scale = float(jnp.max(jnp.abs(expected)))
    np.testing.assert_allclose(jacobian, expected, rtol=1e-5, atol=1e-5 * scale)


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


class _FluxSource(AbstractControlledSource):
    """A controlled backend with its own emission names and no geometry."""

    scale: float = 0.01
    background: float = 0.0

    @property
    def control_blocks(self):
        return {}

    @property
    def emission_blocks(self):
        return ("mass_flux", "log_mass_flux")

    def column_response(self, instrument):
        return self.response_at(instrument, {})

    def response_at(self, instrument, control):
        # r = s 1
        return jnp.full(instrument.n_obs, self.scale)


@pytest.mark.slow
def test_payload_follows_the_backend_emission_block():
    source = _FluxSource()
    instrument = _instrument(_receptors(length=120.0, half_width=60.0))
    # y = exp(log F) r + ε with F = 2
    observation = 2.0 * source.column_response(instrument)
    prior = CoupledPrior.from_std({"log_mass_flux": 0.0}, {"log_mass_flux": 2.0})
    result = solve_coupled(_problem(source, instrument, observation, prior))
    assert result.emission_block == "log_mass_flux"
    draws = result.samples(jax.random.PRNGKey(0), 1000)
    # lognormal draws: positive, around exp(log F*)
    assert bool(jnp.all(draws > 0.0))
    np.testing.assert_allclose(jnp.median(draws), 2.0, rtol=0.05)
    # log π(F) of log F ~ N(0, 4)
    expected = (
        -jnp.log(2.0) - 0.5 * jnp.log(2 * math.pi * 4.0) - jnp.log(2.0) ** 2 / 8.0
    )
    np.testing.assert_allclose(
        result.prior_logpdf(jnp.asarray(2.0)), expected, rtol=1e-10
    )


# ── the centreline block (#270) ──────────────────────────────────────────────

SPACING = 180.0


def _bent_source(n_knots=4, centreline=None, length=None):
    plume = ColumnPlume(
        reference_distance=900.0,
        initial_width=INITIAL_WIDTH,
        upwind_scale=INITIAL_WIDTH,
        pixel_size=PIXEL,
        knot_spacing=SPACING,
    )
    base = _geometry()
    centreline = jnp.zeros(n_knots) if centreline is None else centreline
    geometry = PlumeGeometry(
        location=base.location,
        direction=base.direction,
        reference_width=base.reference_width,
        width_exponent=base.width_exponent,
        centreline=centreline,
    )
    return ColumnPlumeSource(plume=plume, geometry=geometry)


def _knot_distance(n_knots):
    # d_k = k Δ_d
    return SPACING * jnp.arange(1, n_knots + 1)


@pytest.mark.slow
def test_centreline_block_is_declared_with_the_geometry():
    assert "centreline" not in _source().control_blocks
    assert _bent_source(n_knots=7).control_blocks["centreline"] == (7,)
    instrument = _instrument(_receptors(length=600.0, half_width=300.0))
    observation = jnp.zeros(instrument.n_obs)
    mean = {"log_line_density": 0.0, "centreline": jnp.zeros(3)}
    std = {"log_line_density": 1.0, "centreline": 100.0}
    prior = CoupledPrior.from_std(mean, std)
    with pytest.raises(ControlBlockError, match="does not declare"):
        _problem(_source(), instrument, observation, prior)
    with pytest.raises(ControlBlockError, match="shape"):
        _problem(_bent_source(n_knots=4), instrument, observation, prior)
    _problem(_bent_source(n_knots=3), instrument, observation, prior)


@settings(max_examples=25, deadline=None)
@given(
    std_scalar=st.floats(1e-3, 1e3),
    std_vector=st.lists(st.floats(1e-3, 1e3), min_size=3, max_size=3),
)
def test_from_blocks_with_deviations_equals_from_std(std_scalar, std_vector):
    mean = {"background": 0.0, "location": jnp.zeros(2), "centreline": jnp.zeros(3)}
    std = {
        "background": std_scalar,
        "location": std_scalar,
        "centreline": jnp.asarray(std_vector),
    }
    blocks = CoupledPrior.from_blocks(mean, std)
    diagonal = CoupledPrior.from_std(mean, std)
    np.testing.assert_allclose(blocks.covariance, diagonal.covariance, rtol=1e-15)
    assert blocks.layout == diagonal.layout


@pytest.mark.slow
def test_from_blocks_places_a_dense_block_in_layout_order():
    knot_covariance = CentrelinePrior()(_knot_distance(3))
    mean = {"background": 0.0, "centreline": jnp.zeros(3), "width_exponent": 0.8}
    covariance = {
        "background": 1e-4,
        "centreline": knot_covariance,
        "width_exponent": 0.5,
    }
    prior = CoupledPrior.from_blocks(mean, covariance)
    # layout (background, centreline, width_exponent) = (1, 3, 1)
    expected = jnp.zeros((5, 5))
    expected = expected.at[0, 0].set(1e-8)
    expected = expected.at[1:4, 1:4].set(knot_covariance)
    expected = expected.at[4, 4].set(0.25)
    np.testing.assert_allclose(prior.covariance, expected, rtol=1e-15)


def test_from_blocks_validation():
    with pytest.raises(ControlBlockError, match="keys differ"):
        CoupledPrior.from_blocks({"background": 0.0}, {"biases": 1.0})
    with pytest.raises(ControlBlockError, match=r"\(3, 3\) matrix"):
        CoupledPrior.from_blocks(
            {"centreline": jnp.zeros(3)}, {"centreline": jnp.eye(2)}
        )
    with pytest.raises(ControlBlockError, match="> 0"):
        CoupledPrior.from_blocks({"background": 0.0}, {"background": 0.0})


def _bent_prior(source, n_knots, centreline_mean=None):
    geometry = source.geometry
    mean = {
        "log_line_density": jnp.log(LINE_DENSITY),
        "log_reference_width": jnp.log(geometry.reference_width),
        "width_exponent": geometry.width_exponent,
        "location": geometry.location,
        "background": 0.0,
        "centreline": jnp.zeros(n_knots)
        if centreline_mean is None
        else centreline_mean,
    }
    covariance = {
        "log_line_density": 2.0,
        "log_reference_width": 1.0,
        "width_exponent": 0.5,
        "location": 60.0,
        "background": 1e-4,
        "centreline": CentrelinePrior()(_knot_distance(n_knots)),
    }
    return CoupledPrior.from_blocks(mean, covariance)


@pytest.mark.slow
def test_masked_scene_leaves_the_knots_at_their_prior():
    n_knots = 4
    source = _bent_source(n_knots=n_knots)
    receptors = _receptors(length=600.0, half_width=300.0)
    instrument = Instrument(
        "EMIT",
        receptors,
        retrieval_variance=NOISE**2,
        mask=jnp.zeros(receptors.shape[0]),
    )
    observation = LINE_DENSITY * source.column_response(instrument)
    prior = _bent_prior(source, n_knots)
    result = solve_coupled(_problem(source, instrument, observation, prior))
    prior_std = jnp.sqrt(jnp.diagonal(CentrelinePrior()(_knot_distance(n_knots))))
    np.testing.assert_allclose(result.std["centreline"], prior_std, rtol=1e-6)


# a sinusoidal centreline, 200 m amplitude and 3 km wavelength, on a 3 km image
# with knots 180 m apart that run 600 m past its end
BENT_LENGTH = 3000.0
BENT_KNOTS = 20


def _bent_truth():
    knot_distance = _knot_distance(BENT_KNOTS)
    # m_k = 200 sin(2π d_k / 3 km)
    return 200.0 * jnp.sin(2.0 * jnp.pi * knot_distance / 3000.0)


def _two_stage_fit(observation, instrument):
    """Stage 1 straight, stage 2 bent with the direction held (#266)."""
    straight = _full_problem_for(observation, instrument)
    first = solve_coupled(straight)
    estimate = first.map_estimate
    geometry = PlumeGeometry(
        location=estimate["location"],
        direction=estimate["direction"],
        reference_width=jnp.exp(estimate["log_reference_width"]),
        width_exponent=estimate["width_exponent"],
        centreline=jnp.zeros(BENT_KNOTS),
    )
    source = ColumnPlumeSource(plume=_bent_source().plume, geometry=geometry)
    mean = {name: value for name, value in estimate.items() if name != "direction"}
    mean["centreline"] = jnp.zeros(BENT_KNOTS)
    covariance = {
        "log_line_density": 2.0,
        "log_reference_width": 1.0,
        "width_exponent": 0.5,
        "location": 60.0,
        "background": 1e-4,
        "centreline": CentrelinePrior()(_knot_distance(BENT_KNOTS)),
    }
    prior = CoupledPrior.from_blocks(mean, covariance)
    second = solve_coupled(
        _problem(source, instrument, observation, prior), max_steps=1000
    )
    return first, second


def _full_problem_for(observation, instrument):
    mean, std = _prior_blocks(_geometry())
    prior = CoupledPrior.from_std(mean, std)
    return _problem(_source(), instrument, observation, prior)


def _ridge_error(first, second):
    """Crosswind error of each fitted knot in the true plume frame [m].

    Stage 2 bends the ridge in the frame of the stage-1 direction, so each
    fitted knot is placed in the plane and measured against the true ridge.
    """
    estimate = second.map_estimate
    direction = first.map_estimate["direction"]
    # e_∥, e_⊥ of the held direction
    along = jnp.stack([jnp.cos(direction), jnp.sin(direction)])
    across = jnp.stack([-jnp.sin(direction), jnp.cos(direction)])
    knot_distance = _knot_distance(BENT_KNOTS)
    # s_k = s₀ + d_k e_∥ + m_k e_⊥
    downwind_part = einx.multiply("knot, axis -> knot axis", knot_distance, along)
    crosswind_part = einx.multiply(
        "knot, axis -> knot axis", estimate["centreline"], across
    )
    points = einx.add(
        "axis, knot axis -> knot axis",
        estimate["location"],
        downwind_part + crosswind_part,
    )
    truth = _geometry()
    downwind, crosswind = plume_frame(points, truth.location, truth.direction)
    # c_k − m_true(d_k)
    true_offset = centreline_offset(downwind, _bent_truth(), knot_spacing=SPACING)
    return crosswind - true_offset


@pytest.fixture(scope="module")
def bent_scene():
    source = _bent_source(centreline=_bent_truth())
    receptors = _receptors(length=BENT_LENGTH, half_width=1200.0)
    clean = LINE_DENSITY * source.column_response(_instrument(receptors))
    # peak signal-to-noise ratio of 10
    noise = float(jnp.max(clean)) / 10.0
    instrument = Instrument("EMIT", receptors, retrieval_variance=noise**2)
    observation = clean + noise * jax.random.normal(jax.random.PRNGKey(3), clean.shape)
    return clean, observation, instrument


@pytest.mark.slow
def test_noise_free_bent_plume_is_recovered(bent_scene):
    clean, _, instrument = bent_scene
    first, second = _two_stage_fit(clean, instrument)
    assert second.converged
    inside = _knot_distance(BENT_KNOTS) <= BENT_LENGTH
    error = _ridge_error(first, second)[inside]
    rmse = float(jnp.sqrt(jnp.mean(jnp.square(error))))
    assert rmse < PIXEL / 6, f"noise-free centreline RMSE {rmse:.1f} m"


@pytest.mark.slow
def test_noisy_bent_plume_is_recovered_within_its_posterior(bent_scene):
    _, observation, instrument = bent_scene
    first, second = _two_stage_fit(observation, instrument)
    assert second.converged
    knot_distance = _knot_distance(BENT_KNOTS)
    inside = knot_distance <= BENT_LENGTH
    std = second.std["centreline"]

    # the knot errors are of the size the posterior gives them
    error = _ridge_error(first, second)
    standardised = error[inside] / std[inside]
    rms_standardised = float(jnp.sqrt(jnp.mean(jnp.square(standardised))))
    assert rms_standardised < 2.0, f"RMS knot error {rms_standardised:.2f} σ"

    # log q is inside its 2σ interval
    log_error = second.map_estimate["log_line_density"] - jnp.log(LINE_DENSITY)
    assert bool(jnp.abs(log_error) < 2.0 * second.std["log_line_density"])

    # the bent fit is closer to the truth than the straight one
    straight_error = jnp.abs(
        first.map_estimate["log_line_density"] - jnp.log(LINE_DENSITY)
    )
    assert bool(jnp.abs(log_error) < straight_error), (
        f"q straight {float(jnp.exp(first.map_estimate['log_line_density'])):.4f}, "
        f"bent {float(jnp.exp(second.map_estimate['log_line_density'])):.4f}, "
        f"truth {LINE_DENSITY}"
    )

    # a knot past the image is less determined than any knot inside it
    assert float(jnp.min(std[~inside])) > float(jnp.max(std[inside]))
