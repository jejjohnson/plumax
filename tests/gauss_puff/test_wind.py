"""Tests for the wind schedule and its cumulative integrals."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.gauss_puff.wind import (
    WindSchedule,
    cumulative_wind_integrals,
)


@pytest.mark.parametrize(
    ("speed", "direction", "expected_u", "expected_v"),
    [
        # Wind "from west" (270°) should give +u, ≈0 v.
        pytest.param(5.0, 270.0, 5.0, 0.0, id="west_wind"),
        # Wind "from north" (0°): flows south → v negative.
        pytest.param(3.0, 0.0, 0.0, -3.0, id="north_wind"),
    ],
)
def test_from_speed_direction(speed, direction, expected_u, expected_v):
    times = np.linspace(0, 10, 5)
    sched = WindSchedule.from_speed_direction(
        times, np.full(5, speed), np.full(5, direction)
    )
    np.testing.assert_allclose(sched.u_wind, expected_u, atol=1e-5)
    np.testing.assert_allclose(sched.v_wind, expected_v, atol=1e-5)


@pytest.mark.slow
def test_constant_wind_cumulative_integrals():
    # Constant 5 m/s from west → I_u(t) = 5t, I_v(t) = 0, S(t) = 5t.
    times = np.linspace(0.0, 60.0, 7)
    sched = WindSchedule.from_speed_direction(times, np.full(7, 5.0), np.full(7, 270.0))
    save_at = jnp.array([0.0, 15.0, 30.0, 60.0])
    I_u, I_v, S = cumulative_wind_integrals(sched, save_at)
    np.testing.assert_allclose(I_u, 5.0 * save_at, atol=1e-3)
    np.testing.assert_allclose(I_v, jnp.zeros_like(save_at), atol=1e-3)
    np.testing.assert_allclose(S, 5.0 * save_at, atol=1e-3)


@pytest.mark.slow
def test_piecewise_linear_wind_integrals():
    # u(t) is 5 for [0,10], linearly 5→3 for [10,20], 3 for [20,30].
    # Analytical cumulative:
    #   F(10) = 50, F(20) = 50 + (5+3)/2 * 10 = 90, F(30) = 120
    # At t=15: 50 + trapezoid(5, (5+3)/2 ≈ 4) * 5 = 50 + (5+4)/2*5 = 72.5
    times = jnp.array([0.0, 10.0, 20.0, 30.0])
    u_vals = jnp.array([5.0, 5.0, 3.0, 3.0])
    sched = WindSchedule(times=times, u_wind=u_vals, v_wind=jnp.zeros(4))
    save_at = jnp.array([0.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    I_u, I_v, S = cumulative_wind_integrals(sched, save_at)
    expected_I_u = jnp.array([0.0, 50.0, 72.5, 90.0, 105.0, 120.0])
    np.testing.assert_allclose(I_u, expected_I_u, atol=0.05)
    # Since v = 0, S = |u|·integral = I_u here.
    np.testing.assert_allclose(S, expected_I_u, atol=0.05)
    np.testing.assert_allclose(I_v, jnp.zeros_like(save_at), atol=1e-4)


@pytest.mark.slow
def test_wind_speed_accumulation_tracks_vector_magnitude():
    # Wind blowing diagonally at 5 m/s (45° from west → u=v=5/√2 each).
    # Not a meteorological "from" convention case — direct (u, v) construction.
    s = 5.0 / np.sqrt(2.0)
    times = jnp.array([0.0, 30.0, 60.0])
    sched = WindSchedule(
        times=times,
        u_wind=jnp.full(3, s),
        v_wind=jnp.full(3, s),
    )
    save_at = jnp.array([0.0, 30.0, 60.0])
    I_u, I_v, S = cumulative_wind_integrals(sched, save_at)
    # |V| = 5, so S(t) = 5t.
    np.testing.assert_allclose(S, 5.0 * save_at, atol=1e-3)
    np.testing.assert_allclose(I_u, s * save_at, atol=1e-3)
    np.testing.assert_allclose(I_v, s * save_at, atol=1e-3)


def test_wind_at_interpolates_linearly():
    times = jnp.array([0.0, 10.0, 20.0])
    u_vals = jnp.array([5.0, 10.0, 5.0])
    sched = WindSchedule(times=times, u_wind=u_vals, v_wind=jnp.zeros(3))
    u, v = sched.wind_at(jnp.asarray(5.0))
    np.testing.assert_allclose(float(u), 7.5, atol=1e-5)
    np.testing.assert_allclose(float(v), 0.0, atol=1e-5)


def _hourly_day() -> WindSchedule:
    """A 24 h hourly wind near 5 m/s whose direction drifts (#419), in float64."""
    rng = np.random.default_rng(0)
    speed = np.clip(5.0 + np.cumsum(rng.normal(0.0, 0.6, 25)), 1.0, 10.0)
    heading = 240.0 + np.cumsum(rng.normal(0.0, 8.0, 25))
    low = WindSchedule.from_speed_direction(np.arange(25) * 3600.0, speed, heading)
    times, u_wind, v_wind = (
        jnp.asarray(array, dtype=jnp.float64)
        for array in (low.times, low.u_wind, low.v_wind)
    )
    return WindSchedule(times=times, u_wind=u_wind, v_wind=v_wind)


def _gauss_legendre_reference(schedule: WindSchedule, save_at: np.ndarray):
    """(I_u, I_v, S) by 400-point Gauss–Legendre on every node interval, in NumPy."""
    times = np.asarray(schedule.times)
    u_wind, v_wind = np.asarray(schedule.u_wind), np.asarray(schedule.v_wind)
    nodes = np.unique(np.concatenate([times, save_at]))
    unit_nodes, unit_weights = np.polynomial.legendre.leggauss(400)
    middle = 0.5 * (nodes[1:] + nodes[:-1])
    half = 0.5 * np.diff(nodes)
    # τ_{n,q} = m_n + h_n ξ_q
    points = middle[:, None] + half[:, None] * unit_nodes
    u_at = np.interp(points, times, u_wind)
    v_at = np.interp(points, times, v_wind)

    def running(values):
        # ∫ over each interval, h_n Σ_q w_q f(τ_{n,q}), summed from t_0
        pieces = (values @ unit_weights) * half
        return np.interp(save_at, nodes, np.concatenate([[0.0], np.cumsum(pieces)]))

    return running(u_at), running(v_at), running(np.hypot(u_at, v_at))


@pytest.mark.slow
def test_day_of_hourly_wind_matches_quadrature_reference():
    # #419: the diffrax solve was 13–30 m off over this day
    schedule = _hourly_day()
    save_at = np.linspace(0.0, 24 * 3600.0, 601)
    integrals = cumulative_wind_integrals(schedule, jnp.asarray(save_at))
    reference = _gauss_legendre_reference(schedule, save_at)
    for value, expected in zip(integrals, reference, strict=True):
        np.testing.assert_allclose(np.asarray(value), expected, rtol=0.0, atol=1e-6)


@pytest.mark.slow
def test_travel_gradient_matches_central_difference():
    # #419: reverse mode raised, because the solve had max_steps=None
    schedule = _hourly_day()
    times, v_wind = schedule.times[:7], schedule.v_wind[:7]
    save_at = jnp.linspace(0.0, 6 * 3600.0, 13)

    def final_travel(u_wind):
        integrals = cumulative_wind_integrals(
            WindSchedule(times=times, u_wind=u_wind, v_wind=v_wind), save_at
        )
        return integrals[2][-1]

    u_wind = schedule.u_wind[:7]
    gradient = jax.grad(final_travel)(u_wind)
    step = 1e-4
    basis = jnp.eye(7)
    central = jnp.stack(
        [
            (final_travel(u_wind + step * row) - final_travel(u_wind - step * row))
            / (2.0 * step)
            for row in basis
        ]
    )
    assert bool(jnp.isfinite(gradient).all())
    np.testing.assert_allclose(gradient, central, rtol=1e-6)


def test_speed_at_has_a_zero_gradient_in_calm_air():
    # #419: √(u² + v²) at zero gave a NaN gradient
    times = jnp.array([0.0, 10.0, 20.0])

    def speed(u_wind):
        schedule = WindSchedule(times=times, u_wind=u_wind, v_wind=jnp.zeros(3))
        return schedule.speed_at(jnp.asarray(0.0))

    gradient = jax.grad(speed)(jnp.zeros(3))
    np.testing.assert_array_equal(gradient, jnp.zeros(3))


def test_save_time_before_first_knot_is_refused():
    schedule = WindSchedule(
        times=jnp.array([10.0, 20.0]), u_wind=jnp.ones(2), v_wind=jnp.zeros(2)
    )
    with pytest.raises(ValueError, match="before the wind schedule's first knot"):
        cumulative_wind_integrals(schedule, jnp.array([5.0, 15.0]))


def test_wind_at_extrapolates_past_both_ends():
    times = jnp.array([0.0, 10.0, 30.0])
    schedule = WindSchedule(
        times=times, u_wind=jnp.array([1.0, -2.0, 4.0]), v_wind=jnp.zeros(3)
    )
    # u(−10) continues the first segment, u(40) the last
    before, _ = schedule.wind_at(jnp.asarray(-10.0))
    after, _ = schedule.wind_at(jnp.asarray(40.0))
    np.testing.assert_allclose([float(before), float(after)], [4.0, 7.0], atol=1e-12)


def test_one_knot_is_a_constant_wind():
    schedule = WindSchedule(
        times=jnp.array([0.0]), u_wind=jnp.array([2.0]), v_wind=jnp.array([0.0])
    )
    east, north, travel = cumulative_wind_integrals(schedule, jnp.array([0.0, 5.0]))
    np.testing.assert_allclose(east, [0.0, 10.0], atol=1e-12)
    np.testing.assert_allclose(north, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(travel, [0.0, 10.0], atol=1e-12)
