"""Tests for the closed-form column plume (gauss_plume.column)."""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from plumax._src.errors import ParameterError
from plumax.coupled import Instrument, TransportBackend, build_coupled_forward
from plumax.gauss_plume.column import (
    ColumnPlume,
    ColumnPlumeSource,
    PlumeGeometry,
    plume_frame,
    power_law_width,
    unit_column,
)
from plumax.gauss_plume.dispersion import (
    calculate_briggs_dispersion,
    get_dispersion_params,
)
from plumax.gauss_plume.plume import plume_concentration_vmap


PIXEL = 60.0
INITIAL_WIDTH = PIXEL / math.sqrt(12.0)


def _geometry(direction=0.0, location=(0.0, 0.0), width=150.0, exponent=0.9):
    return PlumeGeometry(
        location=jnp.asarray(location),
        direction=direction,
        reference_width=width,
        width_exponent=exponent,
    )


def _grid(n=101, spacing=PIXEL, origin=(0.0, 0.0)):
    """Receptors on a square grid centred on the origin."""
    offsets = (jnp.arange(n) - n // 2) * spacing
    x_axis = origin[0] + offsets
    y_axis = origin[1] + offsets
    xs = einx.id("col -> row col", x_axis, row=n)
    ys = einx.id("row -> row col", y_axis, col=n)
    return einx.id("row col, row col -> (row col) (1 + 1)", xs, ys)


# ── closed-form oracles ──────────────────────────────────────────────────────


@pytest.mark.parametrize("stability_class", ["A", "D", "F"])
@pytest.mark.parametrize("release_height", [0.0, 50.0])
def test_column_matches_trapezoid_of_the_reflected_plume(
    stability_class, release_height
):
    """Gate G1: the closed form is the full column of the 3-D plume."""
    wind_speed = 5.0
    emission_rate = 1.0
    downwind = jnp.array([500.0, 1000.0, 2000.0])
    crosswind_scale = jnp.array([0.0, 0.5, 1.5])
    params = get_dispersion_params(stability_class)
    sigma_y, _ = calculate_briggs_dispersion(downwind, params)
    # receptors along the wind (+x), across it at multiples of σ_y
    xs = einx.id("dist -> dist off", downwind, off=3)
    ys = einx.multiply("dist, off -> dist off", sigma_y, crosswind_scale)
    levels = jnp.linspace(0.0, 5000.0, 2000)
    x = einx.id("dist off -> (dist off level)", xs, level=2000)
    y = einx.id("dist off -> (dist off level)", ys, level=2000)
    z = einx.id("level -> (dist off level)", levels, dist=3, off=3)
    concentration = plume_concentration_vmap(
        x, y, z, 0.0, 0.0, release_height, wind_speed, 0.0, emission_rate, params
    )
    profiles = einx.id(
        "(dist off level) -> dist off level", concentration, off=3, level=2000
    )
    # trapezoid weights on the uniform levels: h, with h / 2 at the two ends
    spacing = levels[1] - levels[0]
    weights = jnp.full_like(levels, spacing).at[0].set(0.5 * spacing)
    weights = weights.at[-1].set(0.5 * spacing)
    trapezoid = einx.dot("dist off level, level -> dist off", profiles, weights)

    width = einx.id("dist -> dist off", sigma_y, off=3)
    closed_form = unit_column(xs, ys, width) * emission_rate / wind_speed
    np.testing.assert_allclose(trapezoid, closed_form, rtol=1e-4)


def test_pixel_average_conserves_mass_at_every_distance():
    """Gate G2, exact form: a crosswind row of pixel averages sums to q."""
    plume = ColumnPlume(initial_width=INITIAL_WIDTH, pixel_size=PIXEL)
    geometry = _geometry()
    offsets = (jnp.arange(-200, 201)) * PIXEL
    for distance in (30.0, 60.0, 300.0, 3000.0):
        downwind = jnp.full_like(offsets, distance)
        receptors = einx.id("obs, obs -> obs (1 + 1)", downwind, offsets)
        response = plume(receptors, geometry)
        mass = einx.sum("[obs]", response) * PIXEL
        np.testing.assert_allclose(mass, 1.0, rtol=1e-10)


def test_point_evaluation_conserves_mass_once_the_plume_is_a_pixel_wide():
    """Gate G2, point form: the sampled sum is exact up to aliasing.

    A Gaussian of width w sampled every g sums to 1 / g up to a relative
    error of about 2 exp(−2π² w² / g²), which is below 1e-8 once w ≥ g.
    """
    plume = ColumnPlume(initial_width=INITIAL_WIDTH)
    geometry = _geometry()
    offsets = (jnp.arange(-200, 201)) * PIXEL
    distances = jnp.array([500.0, 1000.0, 3000.0, 6000.0])
    widths = power_law_width(
        distances,
        reference_width=150.0,
        width_exponent=0.9,
        reference_distance=1000.0,
        initial_width=INITIAL_WIDTH,
    )
    assert bool(jnp.all(widths >= PIXEL))
    for distance in distances:
        downwind = jnp.full_like(offsets, distance)
        receptors = einx.id("obs, obs -> obs (1 + 1)", downwind, offsets)
        mass = einx.sum("[obs]", plume(receptors, geometry)) * PIXEL
        np.testing.assert_allclose(mass, 1.0, rtol=1e-3)


def test_peak_on_the_centreline():
    """ΔΩ(d, 0) = q / (√(2π) w(d)); 0.1 kg/m at w = 100 m is 3.99e-4 kg/m²."""
    response = unit_column(jnp.array([800.0]), jnp.array([0.0]), jnp.array([100.0]))
    np.testing.assert_allclose(0.1 * response, 3.989422804e-4, rtol=1e-9)


def test_width_law_reduces_to_the_pure_power_law():
    """With w₀ = 0 the width is SPIIRE's σ_y = a (d / d_ref)^b."""
    distance = jnp.array([10.0, 250.0, 1000.0, 8000.0])
    width = power_law_width(
        distance, reference_width=120.0, width_exponent=0.8, reference_distance=1000.0
    )
    np.testing.assert_allclose(width, 120.0 * (distance / 1000.0) ** 0.8, rtol=1e-12)


def test_upwind_is_zero_with_the_hard_cut_and_ramped_otherwise():
    geometry = _geometry()
    receptors = jnp.array([[-500.0, 0.0], [-1.0, 0.0], [3 * 17.0, 0.0]])
    hard = ColumnPlume(initial_width=17.0)(receptors, geometry)
    np.testing.assert_array_equal(hard[:2], 0.0)
    soft = ColumnPlume(initial_width=17.0, upwind_scale=17.0)
    ramped = soft(receptors, geometry)
    # sigmoid(3) = 0.9526
    np.testing.assert_allclose(ramped[2] / hard[2], 1.0 / (1.0 + math.exp(-3.0)))
    assert float(ramped[0]) < 1e-10


# ── properties ───────────────────────────────────────────────────────────────


@settings(max_examples=40, deadline=None)
@given(
    direction=st.floats(-math.pi, math.pi),
    width=st.floats(20.0, 400.0),
    exponent=st.floats(0.5, 1.2),
)
def test_response_is_symmetric_across_the_plume(direction, width, exponent):
    geometry = _geometry(direction=direction, width=width, exponent=exponent)
    downwind = jnp.array([300.0, 1500.0])
    crosswind = jnp.array([120.0, 40.0])
    along = jnp.array([math.cos(direction), math.sin(direction)])
    across = jnp.array([-math.sin(direction), math.cos(direction)])
    left = einx.add(
        "obs axis, obs axis -> obs axis",
        einx.multiply("obs, axis -> obs axis", downwind, along),
        einx.multiply("obs, axis -> obs axis", crosswind, across),
    )
    right = einx.subtract(
        "obs axis, obs axis -> obs axis",
        einx.multiply("obs, axis -> obs axis", downwind, along),
        einx.multiply("obs, axis -> obs axis", crosswind, across),
    )
    plume = ColumnPlume(initial_width=INITIAL_WIDTH)
    np.testing.assert_allclose(
        plume(left, geometry), plume(right, geometry), rtol=1e-10
    )


@settings(max_examples=40, deadline=None)
@given(
    angle=st.floats(-math.pi, math.pi),
    shift_x=st.floats(-2000.0, 2000.0),
    shift_y=st.floats(-2000.0, 2000.0),
)
def test_response_is_invariant_to_a_rigid_motion(angle, shift_x, shift_y):
    """Rotating and shifting receptors, source and direction together."""
    receptors = jnp.array([[400.0, 30.0], [1200.0, -250.0], [-100.0, 10.0]])
    geometry = _geometry(direction=0.4, location=(50.0, -20.0))
    rotation = jnp.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    shift = jnp.array([shift_x, shift_y])

    def move(points):
        rotated = einx.dot("... col, row col -> ... row", points, rotation)
        return einx.add("... axis, axis -> ... axis", rotated, shift)

    moved = _geometry(direction=0.4 + angle, location=tuple(move(geometry.location)))
    plume = ColumnPlume(initial_width=INITIAL_WIDTH, upwind_scale=INITIAL_WIDTH)
    np.testing.assert_allclose(
        plume(move(receptors), moved),
        plume(receptors, geometry),
        rtol=1e-8,
        atol=1e-14,
    )


@settings(max_examples=30, deadline=None)
@given(line_density=st.floats(1e-4, 10.0))
def test_source_response_is_linear_in_line_density(line_density):
    """ΔΩ = q r: the backend's unit response scales the observation."""
    receptors = _grid(n=11)
    instrument = Instrument(name="EMIT", receptors=receptors)
    source = ColumnPlumeSource(
        plume=ColumnPlume(initial_width=INITIAL_WIDTH), geometry=_geometry()
    )
    response = source.column_response(instrument)
    observed = line_density * response
    np.testing.assert_allclose(observed / line_density, response, rtol=1e-12)


def test_frame_preserves_distances():
    receptors = jnp.array([[300.0, 400.0], [-120.0, 50.0]])
    downwind, crosswind = plume_frame(receptors, jnp.zeros(2), jnp.asarray(1.1))
    radius = jnp.sqrt(jnp.square(downwind) + jnp.square(crosswind))
    expected = jnp.sqrt(einx.sum("obs [axis]", jnp.square(receptors)))
    np.testing.assert_allclose(radius, expected, rtol=1e-12)


# ── gradients ────────────────────────────────────────────────────────────────


def _flat_response(theta, receptors, plume):
    geometry = PlumeGeometry(
        location=theta[0:2],
        direction=theta[2],
        reference_width=theta[3],
        width_exponent=theta[4],
    )
    return plume(receptors, geometry)


def test_gradient_matches_central_differences():
    plume = ColumnPlume(initial_width=INITIAL_WIDTH, upwind_scale=INITIAL_WIDTH)
    receptors = jnp.array(
        [[0.0, 0.0], [10.0, 5.0], [600.0, 80.0], [2500.0, -300.0], [-60.0, 20.0]]
    )
    theta = jnp.array([3.0, -2.0, 0.15, 140.0, 0.85])
    jacobian = jax.jacrev(_flat_response)(theta, receptors, plume)
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    step = 1e-6 * jnp.maximum(jnp.abs(theta), 1.0)
    for index in range(theta.shape[0]):
        bump = jnp.zeros_like(theta).at[index].set(step[index])
        forward = _flat_response(theta + bump, receptors, plume)
        backward = _flat_response(theta - bump, receptors, plume)
        central = (forward - backward) / (2.0 * step[index])
        np.testing.assert_allclose(jacobian[:, index], central, rtol=1e-5, atol=1e-12)


def test_gradient_is_finite_at_the_source_with_the_hard_cut():
    plume = ColumnPlume(initial_width=INITIAL_WIDTH)
    receptors = jnp.array([[0.0, 0.0], [-30.0, 0.0], [30.0, 0.0]])
    theta = jnp.array([0.0, 0.0, 0.0, 150.0, 0.9])
    jacobian = jax.jacrev(_flat_response)(theta, receptors, plume)
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_gradient_with_pixel_average_matches_central_differences():
    plume = ColumnPlume(
        initial_width=INITIAL_WIDTH, upwind_scale=INITIAL_WIDTH, pixel_size=PIXEL
    )
    receptors = jnp.array([[600.0, 80.0], [2500.0, -300.0], [90.0, 0.0]])
    theta = jnp.array([3.0, -2.0, 0.15, 140.0, 0.85])
    jacobian = jax.jacrev(_flat_response)(theta, receptors, plume)
    step = 1e-6 * jnp.maximum(jnp.abs(theta), 1.0)
    for index in range(theta.shape[0]):
        bump = jnp.zeros_like(theta).at[index].set(step[index])
        central = (
            _flat_response(theta + bump, receptors, plume)
            - _flat_response(theta - bump, receptors, plume)
        ) / (2.0 * step[index])
        np.testing.assert_allclose(jacobian[:, index], central, rtol=1e-5, atol=1e-12)


# ── transforms ───────────────────────────────────────────────────────────────


def test_jit_and_vmap_agree_with_a_loop():
    plume = ColumnPlume(initial_width=INITIAL_WIDTH, upwind_scale=INITIAL_WIDTH)
    receptors = _grid(n=21)
    directions = jnp.array([0.0, 0.5, 1.0, 2.0])
    widths = jnp.array([100.0, 150.0, 200.0, 250.0])

    def evaluate(direction, width):
        geometry = PlumeGeometry(
            location=jnp.zeros(2),
            direction=direction,
            reference_width=width,
            width_exponent=0.9,
        )
        return plume(receptors, geometry)

    looped = jnp.stack(
        [evaluate(d, w) for d, w in zip(directions, widths, strict=True)]
    )
    jitted = jnp.stack(
        [jax.jit(evaluate)(d, w) for d, w in zip(directions, widths, strict=True)]
    )
    mapped = jax.vmap(evaluate)(directions, widths)
    np.testing.assert_allclose(jitted, looped, rtol=1e-12)
    np.testing.assert_allclose(mapped, looped, rtol=1e-12)


# ── dtype, validation, backend, config ───────────────────────────────────────


def test_float32_in_gives_float32_out():
    receptors = jnp.asarray(_grid(n=5), dtype=jnp.float32)
    geometry = PlumeGeometry(
        location=jnp.zeros(2, dtype=jnp.float32),
        direction=jnp.float32(0.2),
        reference_width=jnp.float32(150.0),
        width_exponent=jnp.float32(0.9),
    )
    for plume in (
        ColumnPlume(initial_width=INITIAL_WIDTH),
        ColumnPlume(initial_width=INITIAL_WIDTH, upwind_scale=10.0, pixel_size=PIXEL),
    ):
        assert plume(receptors, geometry).dtype == jnp.float32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_distance": 0.0},
        {"initial_width": -1.0},
        {"upwind_scale": 0.0},
        {"pixel_size": -60.0},
    ],
)
def test_invalid_fields_raise(kwargs):
    with pytest.raises(ParameterError):
        ColumnPlume(**kwargs)


def test_source_is_a_transport_backend_and_builds_a_forward():
    receptors = _grid(n=11)
    kernel = jnp.linspace(0.5, 1.5, receptors.shape[0])
    instrument = Instrument(name="EMIT", receptors=receptors, averaging_kernel=kernel)
    plume = ColumnPlume(initial_width=INITIAL_WIDTH)
    source = ColumnPlumeSource(plume=plume, geometry=_geometry())
    assert isinstance(source, TransportBackend)
    forward = build_coupled_forward(source, [instrument])
    expected = kernel * plume(receptors, _geometry())
    np.testing.assert_allclose(forward.responses[0], expected, rtol=1e-12)


def test_config_round_trips():
    plume = ColumnPlume(
        reference_distance=2500.0,
        initial_width=INITIAL_WIDTH,
        upwind_scale=INITIAL_WIDTH,
        pixel_size=PIXEL,
    )
    assert ColumnPlume.from_config(plume.get_config()) == plume
