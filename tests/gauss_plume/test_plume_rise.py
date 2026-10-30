"""Tests for Briggs plume rise (gauss_plume.plume_rise) and its use in the plume."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from plumax._src.errors import ParameterError
from plumax.gauss_plume.dispersion import BRIGGS_DISPERSION_PARAMS
from plumax.gauss_plume.plume import plume_concentration, simulate_plume
from plumax.gauss_plume.plume_rise import (
    BriggsPlumeRise,
    briggs_plume_rise,
    effective_height,
)


G = 9.80665


def _stack(**overrides):
    kwargs = {
        "exit_velocity": 10.0,
        "exit_temperature": 400.0,
        "stack_diameter": 2.0,
        "ambient_temperature": 290.0,
        "stability_class": "D",
    }
    kwargs.update(overrides)
    return kwargs


def _buoyancy_flux(v, d, ts, ta):
    return G * v * d**2 * (ts - ta) / (4.0 * ts)


# ── closed-form limits ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("diameter", "exit_temperature"),
    [(2.0, 400.0), (5.0, 450.0)],
    ids=["F_below_55", "F_above_55"],
)
def test_textbook_values(diameter, exit_temperature):
    """Neutral buoyant rise: the 2/3 law before x_f, the final rise past it."""
    stack = _stack(stack_diameter=diameter, exit_temperature=exit_temperature)
    u = 5.0
    flux = _buoyancy_flux(10.0, diameter, exit_temperature, 290.0)
    below = flux < 55.0
    x_final = 49.0 * flux ** (5.0 / 8.0) if below else 119.0 * flux ** (2.0 / 5.0)
    final = 1.6 * flux ** (1.0 / 3.0) * x_final ** (2.0 / 3.0) / u
    x = jnp.array([0.3 * x_final, 0.6 * x_final, 2.0 * x_final, 50.0 * x_final])
    rise = briggs_plume_rise(x, wind_speed=u, **stack)
    two_thirds = 1.6 * flux ** (1.0 / 3.0) * np.asarray(x) ** (2.0 / 3.0) / u
    np.testing.assert_allclose(rise[:2], two_thirds[:2], rtol=1e-12)
    np.testing.assert_allclose(rise[2:], final, rtol=1e-12)
    # these stacks are buoyancy-dominated: the jet rise stays below
    assert float(final) > 3.0 * diameter * 10.0 / u


@pytest.mark.parametrize(("stability_class", "gradient"), [("E", 0.020), ("F", 0.035)])
def test_stable_final_rise(stability_class, gradient):
    stack = _stack(stability_class=stability_class)
    u = 4.0
    flux = _buoyancy_flux(10.0, 2.0, 400.0, 290.0)
    s = G * gradient / 290.0
    final = 2.6 * (flux / (u * s)) ** (1.0 / 3.0)
    rise = briggs_plume_rise(jnp.array([1.0e5]), wind_speed=u, **stack)
    np.testing.assert_allclose(rise, final, rtol=1e-12)
    # the 2/3 law meets the stable final rise at x_f = (2.6/1.6)^{3/2} u / √s
    x_final = (2.6 / 1.6) ** 1.5 * u / np.sqrt(s)
    before = briggs_plume_rise(jnp.array([0.99 * x_final]), wind_speed=u, **stack)
    np.testing.assert_allclose(
        before, 1.6 * flux ** (1 / 3) * (0.99 * x_final) ** (2 / 3) / u, rtol=1e-12
    )


def test_momentum_only_neutral():
    """T_s = T_a: no buoyancy, so the jet law up to 3 d v_s / u."""
    stack = _stack(exit_temperature=290.0, exit_velocity=20.0)
    u, d, v = 4.0, 2.0, 20.0
    final = 3.0 * d * v / u
    x_final = d * (final / (1.44 * d * (v / u) ** (2 / 3))) ** 3
    x = jnp.array([0.1 * x_final, 10.0 * x_final])
    rise = briggs_plume_rise(x, wind_speed=u, **stack)
    gradual = 1.44 * d * (v / u) ** (2 / 3) * (np.asarray(x[0]) / d) ** (1 / 3)
    np.testing.assert_allclose(rise[0], gradual, rtol=1e-12)
    np.testing.assert_allclose(rise[1], final, rtol=1e-12)


def test_momentum_only_stable_takes_the_lower_final_rise():
    stack = _stack(exit_temperature=290.0, exit_velocity=20.0, stability_class="F")
    u, d, v = 4.0, 2.0, 20.0
    s = G * 0.035 / 290.0
    momentum_flux = v**2 * d**2 * 290.0 / (4.0 * 290.0)
    stable = 1.5 * (momentum_flux / u) ** (1 / 3) * s ** (-1 / 6)
    expected = min(3.0 * d * v / u, stable)
    rise = briggs_plume_rise(jnp.array([1.0e5]), wind_speed=u, **stack)
    np.testing.assert_allclose(rise, expected, rtol=1e-12)


def test_no_rise_without_flux_and_upwind():
    still = _stack(exit_temperature=290.0, exit_velocity=0.0)
    np.testing.assert_array_equal(
        briggs_plume_rise(jnp.array([10.0, 1000.0]), wind_speed=5.0, **still), 0.0
    )
    hot = _stack()
    np.testing.assert_array_equal(
        briggs_plume_rise(jnp.array([-10.0, 0.0]), wind_speed=5.0, **hot), 0.0
    )


def test_effective_height_adds_stack_height():
    x = jnp.array([50.0, 500.0, 5000.0])
    stack = _stack()
    rise = briggs_plume_rise(x, wind_speed=5.0, **stack)
    height = effective_height(x, stack_height=30.0, wind_speed=5.0, **stack)
    np.testing.assert_allclose(height, 30.0 + rise, rtol=1e-12)


# ── properties ───────────────────────────────────────────────────────────────


@settings(max_examples=50, deadline=None)
@given(
    x=st.floats(1.0, 2.0e4),
    step=st.floats(0.0, 2.0e4),
    u=st.floats(1.0, 15.0),
    du=st.floats(0.0, 10.0),
    excess=st.floats(0.0, 300.0),
    stability_class=st.sampled_from("ABCDEF"),
)
def test_rise_grows_with_distance_and_falls_with_wind(
    x, step, u, du, excess, stability_class
):
    stack = _stack(exit_temperature=290.0 + excess, stability_class=stability_class)
    near = briggs_plume_rise(jnp.array([x]), wind_speed=u, **stack)
    far = briggs_plume_rise(jnp.array([x + step]), wind_speed=u, **stack)
    windier = briggs_plume_rise(jnp.array([x]), wind_speed=u + du, **stack)
    assert float(far[0]) >= float(near[0]) - 1e-9
    assert float(windier[0]) <= float(near[0]) + 1e-9


# ── gradients and transforms ─────────────────────────────────────────────────


@pytest.mark.parametrize("stability_class", ["D", "F"])
def test_grad_matches_fd(stability_class):
    """∂Δh/∂T_s is finite and matches central differences, before and past x_f."""
    x = jnp.array([200.0, 1.0e5])

    def total(exit_temperature):
        stack = _stack(
            exit_temperature=exit_temperature, stability_class=stability_class
        )
        return jnp.sum(briggs_plume_rise(x, wind_speed=5.0, **stack))

    grad = jax.grad(total)(400.0)
    step = 1e-4
    fd = (total(400.0 + step) - total(400.0 - step)) / (2 * step)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=1e-6)


def test_grad_finite_at_zero_buoyancy():
    def total(exit_temperature):
        stack = _stack(exit_temperature=exit_temperature, exit_velocity=0.0)
        return jnp.sum(briggs_plume_rise(jnp.array([500.0]), wind_speed=5.0, **stack))

    assert np.isfinite(jax.grad(total)(290.0))


def test_jit_and_vmap_match_loop():
    rise = BriggsPlumeRise(**_stack())
    x = jnp.array([[50.0, 500.0], [5000.0, 50000.0]])
    winds = jnp.array([3.0, 7.0])
    loop = jnp.stack([rise(row, wind_speed=w) for row, w in zip(x, winds, strict=True)])
    jitted = jax.jit(lambda r, row, w: r(row, wind_speed=w))(rise, x[0], winds[0])
    vmapped = jax.vmap(lambda row, w: rise(row, wind_speed=w))(x, winds)
    np.testing.assert_allclose(jitted, loop[0], rtol=1e-12)
    np.testing.assert_allclose(vmapped, loop, rtol=1e-12)


# ── validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"stability_class": "G"},
        {"stack_diameter": 0.0},
        {"exit_velocity": -1.0},
        {"exit_temperature": 0.0},
        {"ambient_temperature": -5.0},
    ],
)
def test_invalid_parameters_rejected(overrides):
    with pytest.raises(ParameterError):
        BriggsPlumeRise(**_stack(**overrides))


# ── plume integration ────────────────────────────────────────────────────────


def _receptors():
    x = jnp.array([150.0, 800.0, 3000.0])
    return x, jnp.array([5.0, -20.0, 40.0]), jnp.array([1.0, 30.0, 80.0])


def test_plume_concentration_default_unchanged_by_zero_rise():
    """A stack with no buoyancy or momentum gives the scalar-H numbers exactly."""
    x, y, z = _receptors()
    params = BRIGGS_DISPERSION_PARAMS["D"]
    still = BriggsPlumeRise(**_stack(exit_temperature=290.0, exit_velocity=0.0))
    base = plume_concentration(x, y, z, 0.0, 0.0, 20.0, 5.0, 0.0, 0.1, params)
    risen = plume_concentration(
        x, y, z, 0.0, 0.0, 20.0, 5.0, 0.0, 0.1, params, plume_rise=still
    )
    np.testing.assert_array_equal(risen, base)


def test_plume_concentration_uses_the_effective_height_per_receptor():
    x, y, z = _receptors()
    params = BRIGGS_DISPERSION_PARAMS["D"]
    rise = BriggsPlumeRise(**_stack())
    risen = plume_concentration(
        x, y, z, 0.0, 0.0, 20.0, 5.0, 0.0, 0.1, params, plume_rise=rise
    )
    heights = effective_height(x, stack_height=20.0, wind_speed=5.0, **_stack())
    for i in range(3):
        expected = plume_concentration(
            x[i], y[i], z[i], 0.0, 0.0, heights[i], 5.0, 0.0, 0.1, params
        )
        np.testing.assert_allclose(risen[i], expected, rtol=1e-12)


def test_plume_concentration_grad_wrt_exit_temperature():
    x, y, z = _receptors()
    params = BRIGGS_DISPERSION_PARAMS["D"]

    def total(exit_temperature):
        rise = BriggsPlumeRise(**_stack(exit_temperature=exit_temperature))
        conc = plume_concentration(
            x, y, z, 0.0, 0.0, 20.0, 5.0, 0.0, 0.1, params, plume_rise=rise
        )
        return jnp.sum(conc)

    grad = jax.grad(total)(400.0)
    step = 1e-3
    fd = (total(400.0 + step) - total(400.0 - step)) / (2 * step)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


def test_simulate_plume_with_rise_lifts_the_plume():
    common = {
        "emission_rate": 0.1,
        "source_location": (0.0, 0.0, 20.0),
        "wind_speed": 5.0,
        "wind_direction": 270.0,
        "stability_class": "D",
        "domain_x": (100.0, 2000.0, 8),
        "domain_y": (-200.0, 200.0, 5),
        "domain_z": (0.0, 400.0, 21),
    }
    flat = simulate_plume(**common)
    risen = simulate_plume(**common, plume_rise=BriggsPlumeRise(**_stack()))
    weights = flat["z"]
    flat_height = (flat["concentration"] * weights).sum() / flat["concentration"].sum()
    risen_height = (risen["concentration"] * weights).sum() / risen[
        "concentration"
    ].sum()
    assert float(risen_height) > float(flat_height)
    assert risen.attrs == flat.attrs
