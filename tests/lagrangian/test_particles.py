"""Tests for the Markov-1 Langevin particle integrator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.lagrangian.particles import (
    ParticleState,
    integrate_particles,
    langevin_step,
    n_steps_for_horizon,
    step_durations,
    uniform_wind,
    wind_from_speed_direction,
)
from plumax.lagrangian.turbulence import HomogeneousTurbulence


@pytest.mark.parametrize(
    "horizon,dt,expected",
    [
        (10.0, 1.0, 10),  # exact
        (10.1, 1.0, 11),  # genuine partial step
        (0.5, 1.0, 1),  # sub-step horizon
        (0.0, 1.0, 0),  # empty
        (2.1, 0.15, 14),  # float round-up trap: 2.1/0.15 = 14.000…02, not 15
        (1.0, 0.1, 10),  # 1.0/0.1 = 9.999… in float, must snap to 10
    ],
)
def test_n_steps_for_horizon_is_float_robust(horizon, dt, expected):
    assert n_steps_for_horizon(horizon, dt) == expected


def test_n_steps_no_zero_length_final_step_on_rounded_multiple():
    # The P3 case: 2.1 is an exact multiple of 0.15 (×14) but the float quotient
    # rounds just above 14, so a naive ceil would add a 15th, zero-length step.
    n = n_steps_for_horizon(2.1, 0.15)
    dts = np.asarray(step_durations(2.1, 0.15, n))
    assert dts.size == 14
    assert np.all(dts > 0.0)  # no degenerate zero-length step
    np.testing.assert_allclose(dts, 0.15, atol=1e-9)
    assert dts.sum() == pytest.approx(2.1)


@pytest.mark.parametrize(
    "horizon,dt,n",
    [(10.0, 1.0, 10), (10.1, 1.0, 11), (10.5, 2.0, 6), (0.5, 1.0, 1)],
)
def test_step_durations_sum_to_horizon(horizon, dt, n):
    dts = np.asarray(step_durations(horizon, dt, n))
    assert dts.size == n
    assert dts.sum() == pytest.approx(horizon)
    # All but the last are the full dt; the last is the (≤ dt) remainder.
    np.testing.assert_allclose(dts[:-1], dt)
    assert 0.0 < dts[-1] <= dt + 1e-9


def test_step_durations_empty_when_zero_steps():
    assert np.asarray(step_durations(0.0, 1.0, 0)).size == 0


def test_integrate_particles_stops_at_t1_for_nondivisible_horizon():
    # Deterministic wind: with t1 - t0 not a multiple of dt, the ensemble must
    # advance to exactly t1 (x = u * (t1 - t0)), not overshoot to t0 + n*dt.
    turb = HomogeneousTurbulence(0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
    state = ParticleState(
        position=jnp.array([[0.0, 0.0, 100.0]]),
        velocity=jnp.zeros((1, 3)),
    )
    wind = uniform_wind(5.0, 0.0, 0.0)
    final, _ = integrate_particles(
        state, wind, turb, t0=0.0, t1=1.5, dt=1.0, key=jax.random.PRNGKey(0)
    )
    # x = u * 1.5 = 7.5 (overshoot to t=2 would give 10.0).
    np.testing.assert_allclose(
        np.asarray(final.position[0]), [7.5, 0.0, 100.0], atol=1e-6
    )


def _calm_turbulence():
    # σ = 0 → deterministic; τ kept positive (required).
    return HomogeneousTurbulence(0.0, 0.0, 0.0, 100.0, 100.0, 100.0)


def test_zero_turbulence_follows_streamline():
    """With σ → 0 a particle must follow the mean-wind streamline exactly."""
    turb = _calm_turbulence()
    state = ParticleState(
        position=jnp.array([[0.0, 0.0, 100.0]]),
        velocity=jnp.zeros((1, 3)),
    )
    wind = uniform_wind(5.0, 0.0, 0.0)
    final, _ = integrate_particles(
        state, wind, turb, t0=0.0, t1=100.0, dt=1.0, key=jax.random.PRNGKey(0)
    )
    # x = u·t = 500, y and z unchanged.
    np.testing.assert_allclose(
        np.asarray(final.position[0]), [500.0, 0.0, 100.0], atol=1e-6
    )


def test_well_mixed_velocity_variance_is_stationary():
    """Homogeneous OU started at stationarity keeps Var[v] ≈ σ² (well mixed)."""
    turb = HomogeneousTurbulence.isotropic(sigma=1.0, tau=20.0)
    n = 20000
    key = jax.random.PRNGKey(1)
    key, vkey = jax.random.split(key)
    state = ParticleState(
        position=jnp.full((n, 3), 200.0),  # high up, away from the ground lid
        velocity=jax.random.normal(vkey, (n, 3)) * turb.sigma,
    )
    wind = uniform_wind(0.0, 0.0, 0.0)
    final, _ = integrate_particles(state, wind, turb, t0=0.0, t1=100.0, dt=0.5, key=key)
    var = np.var(np.asarray(final.velocity), axis=0)
    np.testing.assert_allclose(var, [1.0, 1.0, 1.0], rtol=0.05)


def test_velocity_variance_grows_from_rest_to_sigma_squared():
    """Starting at v=0 the variance approaches σ²(1 - e^{-2t/τ})."""
    turb = HomogeneousTurbulence.isotropic(sigma=1.0, tau=10.0)
    n = 20000
    state = ParticleState(
        position=jnp.full((n, 3), 500.0),
        velocity=jnp.zeros((n, 3)),
    )
    wind = uniform_wind(0.0, 0.0, 0.0)
    t = 10.0
    final, _ = integrate_particles(
        state, wind, turb, t0=0.0, t1=t, dt=0.25, key=jax.random.PRNGKey(2)
    )
    expected = 1.0 * (1.0 - np.exp(-2.0 * t / 10.0))
    var_u = float(np.var(np.asarray(final.velocity)[:, 0]))
    assert abs(var_u - expected) < 0.05


def test_ground_reflection_keeps_particles_above_zero():
    turb = HomogeneousTurbulence.isotropic(sigma=2.0, tau=10.0)
    n = 5000
    key = jax.random.PRNGKey(3)
    key, vkey = jax.random.split(key)
    state = ParticleState(
        position=jnp.full((n, 3), 10.0),  # start near the ground
        velocity=jax.random.normal(vkey, (n, 3)) * turb.sigma,
    )
    wind = uniform_wind(1.0, 0.0, 0.0)
    final, _ = integrate_particles(
        state, wind, turb, t0=0.0, t1=200.0, dt=0.5, key=key, pbl_height=500.0
    )
    z = np.asarray(final.position[:, 2])
    assert np.all(z >= 0.0)
    assert np.all(z <= 500.0)


def test_wind_from_speed_direction_west_wind_flows_east():
    # 270° = wind FROM the west, i.e. blowing toward +x.
    wind = wind_from_speed_direction(5.0, 270.0)
    vec = np.asarray(wind(jnp.array(0.0)))
    np.testing.assert_allclose(vec, [5.0, 0.0, 0.0], atol=1e-6)


def test_single_step_matches_manual_ou():
    turb = HomogeneousTurbulence.isotropic(sigma=1.0, tau=10.0)
    state = ParticleState(
        position=jnp.array([[0.0, 0.0, 300.0]]),
        velocity=jnp.array([[0.5, -0.3, 0.2]]),
    )
    dt = 1.0
    key = jax.random.PRNGKey(7)
    nxt = langevin_step(state, jnp.array([2.0, 0.0, 0.0]), turb, dt, key)
    # Position update uses the *updated* velocity: x += (u + v_new)·dt.
    decay = np.exp(-dt / 10.0)
    noise = np.asarray(jax.random.normal(key, (1, 3)))
    v_new = decay * np.asarray(state.velocity) + 1.0 * np.sqrt(1 - decay**2) * noise
    expected_pos = np.asarray(state.position) + (np.array([2.0, 0.0, 0.0]) + v_new) * dt
    np.testing.assert_allclose(np.asarray(nxt.position), expected_pos, rtol=1e-5)
