"""Tests for the Markov-1 Langevin particle integrator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from plumax.lagrangian.particles import (
    ParticleState,
    integrate_particles,
    langevin_step,
    uniform_wind,
    wind_from_speed_direction,
)
from plumax.lagrangian.turbulence import HomogeneousTurbulence


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
