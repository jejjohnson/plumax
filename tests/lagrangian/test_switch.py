"""The particle model on tracerax: the old values, its closures, its walls."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tracerax.advect import Langevin, Walls, integrate, langevin_step
from tracerax.particles import at_point, with_turbulent_velocity
from tracerax.velocity import uniform, uniform_in_space

from plumax.lagrangian import (
    HannaTurbulence,
    HomogeneousTurbulence,
    compute_footprint,
    gridded_wind_from_prescribed,
    simulate_lagrangian,
    wind_from_speed_direction,
)
from plumax.les_fvm import make_grid, wind_field_from_callable


# --- the values of plumax 96449d2, before the switch --------------------------

TURBULENCE = HomogeneousTurbulence(1.0, 1.0, 0.6, 30.0, 30.0, 20.0)
HANNA = HannaTurbulence(u_star=0.4, pbl_height=800.0, obukhov_length=-50.0, w_star=1.2)
STABLE = HannaTurbulence(0.3, 500.0, 100.0)
STEADY = uniform(jnp.array([4.0, 1.0, 0.0]))
GUST = uniform_in_space(
    lambda t: jnp.stack([4.0 + 0.02 * t, 0.5 * jnp.sin(0.05 * t), 0.0 * t])
)


def _gridded():
    grid = make_grid((-500.0, 500.0, 20), (-400.0, 400.0, 16), (0.0, 200.0, 8))
    field = wind_field_from_callable(
        grid,
        lambda t, x, y, z: (
            3.0 + 0.004 * x + 0.002 * z + 0.01 * t,
            -1.0 + 0.003 * y,
            0.0 * x,
        ),
    )
    return gridded_wind_from_prescribed(field, jnp.array([0.0, 20.0, 60.0, 120.0]))


def _stored(name: str) -> np.ndarray:
    path = Path(__file__).parent / "data" / "plumax_96449d2.npz"
    with np.load(path) as stored:
        return stored[name]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("speed", {"wind_speed": 5.0, "wind_direction": 270.0}),
        ("gust", {"wind": GUST, "pbl_height": 150.0, "dt": 2.5, "t_end": 121.0}),
        ("grid", {"wind": "gridded", "pbl_height": 150.0, "dt": 5.0}),
        ("hanna", {"wind": STEADY, "turbulence": HANNA, "pbl_height": 800.0}),
        (
            "stable",
            {"wind": STEADY, "turbulence": STABLE, "pbl_height": 500.0, "seed": 3},
        ),
    ],
)
def test_the_column_equals_the_one_before_the_switch(name, changes):
    settings = {
        "emission_rate": 0.1,
        "source_location": (0.0, 0.0, 20.0),
        "turbulence": TURBULENCE,
        "domain_x": (-50.0, 600.0, 65),
        "domain_y": (-150.0, 150.0, 30),
        "domain_z": (0.0, 200.0, 20),
        "n_particles": 1500,
        "t_end": 120.0,
        "dt": 1.0,
        "seed": 0,
        **changes,
    }
    if settings.get("wind") == "gridded":
        settings["wind"] = _gridded()
    dataset = simulate_lagrangian(**settings)
    expected = _stored(f"col_{name}")
    np.testing.assert_allclose(
        dataset["column_concentration"].values,
        expected,
        rtol=0.0,
        atol=1e-12 * expected.max(),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("speed", {"wind": "west"}),
        (
            "gust",
            {
                "wind": GUST,
                "receptor_time": 300.0,
                "t_back": 61.0,
                "dt": 2.5,
                "pbl_height": 150.0,
            },
        ),
        (
            "grid",
            {
                "wind": "gridded",
                "receptor_time": 110.0,
                "dt": 5.0,
                "t_back": 100.0,
                "pbl_height": 150.0,
            },
        ),
        ("hanna", {"wind": STEADY, "turbulence": HANNA, "pbl_height": 800.0}),
        (
            "stable",
            {
                "wind": STEADY,
                "turbulence": STABLE,
                "pbl_height": 500.0,
                "seed": 7,
                "pbl_fraction": 0.3,
            },
        ),
    ],
)
def test_the_footprint_equals_the_one_before_the_switch(name, changes):
    settings = {
        "receptor_location": (100.0, 0.0, 20.0),
        "turbulence": TURBULENCE,
        "domain_x": (-600.0, 200.0, 40),
        "domain_y": (-200.0, 200.0, 20),
        "n_particles": 1500,
        "t_back": 120.0,
        "dt": 1.0,
        "seed": 0,
        **changes,
    }
    winds = {"gridded": _gridded, "west": lambda: wind_from_speed_direction(5.0, 270.0)}
    if isinstance(settings["wind"], str):
        settings["wind"] = winds[settings["wind"]]()
    footprint, _, _ = compute_footprint(**settings)
    expected = _stored(f"f_{name}")
    np.testing.assert_allclose(
        footprint, expected, rtol=0.0, atol=1e-12 * expected.max()
    )


def test_a_negative_horizon_is_refused():
    with pytest.raises(ValueError, match="t_end"):
        simulate_lagrangian(
            1.0,
            (0.0, 0.0, 10.0),
            TURBULENCE,
            (-10.0, 10.0, 4),
            (-10.0, 10.0, 4),
            (0.0, 20.0, 2),
            wind=STEADY,
            t_end=-60.0,
        )


# --- the closures -------------------------------------------------------------


@pytest.mark.validation
def test_well_mixed_condition_hanna():
    """An initially uniform slab stays uniform under inhomogeneous Hanna turbulence.

    This is the well-mixed criterion (Thomson 1987): with height-dependent
    ``σ_w(z)`` the drift ``½(1+v_w²/σ_w²)∂σ_w²/∂z`` is what stops particles from
    piling up where the turbulence is weak. The closure is plumax's and the step
    tracerax's, so the pair is tested here.
    """
    turb = HannaTurbulence(
        u_star=0.5, pbl_height=1000.0, obukhov_length=-50.0, w_star=2.0
    )  # convective → σ_w varies strongly with height
    h = 1000.0
    n = 8000
    key, zkey, vkey = jax.random.split(jax.random.PRNGKey(0), 3)
    z0 = jax.random.uniform(zkey, (n,), minval=0.0, maxval=h)
    pos = jnp.stack([jnp.zeros(n), jnp.zeros(n), z0], axis=1)
    state = eqx.tree_at(lambda s: s.position, at_point(jnp.zeros(3), n), pos)
    # velocities drawn from the *local* stationary variance
    state = with_turbulent_velocity(state, turb, key=vkey)
    result = integrate(
        state,
        Langevin(uniform(jnp.zeros(3)), turb, Walls(0.0, h)),
        t0=0.0,
        duration=400.0,
        step_size=1.0,
        key=key,
    )
    zf = np.asarray(result.state.position[:, 2])
    assert np.all((zf >= 0.0) & (zf <= h))
    # Still uniform: per-bin occupancy within ~20 % of N/nbins, and the mean
    # near the slab centre (no collapse toward the low-σ_w upper layer).
    counts, _ = np.histogram(zf, bins=5, range=(0.0, h))
    expected = n / 5
    assert np.max(np.abs(counts - expected)) / expected < 0.2
    assert abs(float(zf.mean()) - 0.5 * h) < 0.1 * h


@pytest.mark.slow
def test_hanna_ground_level_drift_is_finite():
    # The convective Hanna σ_w² carries a (z/h)^{2/3} factor whose ∂/∂z is
    # singular at z = 0, so a ground-level particle (or one reflected exactly to
    # the ground) would give inf/NaN vertical drift. HannaTurbulence.at floors z,
    # so the step must stay finite.
    turb = HannaTurbulence(
        u_star=0.5, pbl_height=1000.0, obukhov_length=-50.0, w_star=2.0
    )
    state = at_point(jnp.zeros(3), 1)
    noise = jax.random.normal(jax.random.PRNGKey(0), (1, 3))
    moved = langevin_step(
        state, jnp.zeros((1, 3)), turb, jnp.ones(1), noise, walls=Walls(0.0, 1000.0)
    )
    assert np.isfinite(np.asarray(moved.position)).all()
    assert np.isfinite(np.asarray(moved.velocity)).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    "turbulence",
    [
        HomogeneousTurbulence(0.5, 0.4, 0.3, 30.0, 30.0, 20.0),
        HannaTurbulence(0.4, 800.0, -50.0, w_star=1.2),
    ],
    ids=["homogeneous", "hanna"],
)
def test_gradients_reach_the_parameters_of_a_closure(turbulence):
    """The spread of a cloud, in every parameter, against central differences."""
    leaves, structure = jax.tree.flatten(turbulence)
    parameters = jnp.array(leaves)
    noise = jax.random.normal(jax.random.PRNGKey(0), (4, 50, 3))

    def spread(parameters):
        closure = jax.tree.unflatten(structure, list(parameters))
        result = integrate(
            at_point(jnp.array([0.0, 0.0, 50.0]), 50),
            Langevin(uniform(jnp.zeros(3)), closure, Walls(0.0, 800.0)),
            t0=0.0,
            duration=8.0,
            step_size=2.0,
            noise=noise,
        )
        return jnp.var(result.state.position, axis=0).sum()

    gradient = np.asarray(jax.grad(spread)(parameters))
    for index, value in enumerate(leaves):
        step = 1e-5 * max(abs(value), 1e-3)
        offset = jnp.zeros_like(parameters).at[index].set(step)
        difference = (spread(parameters + offset) - spread(parameters - offset)) / (
            2.0 * step
        )
        np.testing.assert_allclose(
            gradient[index], difference, rtol=1e-4, atol=1e-6 * np.abs(gradient).max()
        )
    assert np.abs(gradient).max() > 0.0


# --- the walls (#392) ---------------------------------------------------------


@pytest.mark.slow
def test_no_particle_leaves_a_shallow_layer_at_a_long_step():
    """One reflection per wall left 6.9 % of these particles below the ground."""
    turbulence = HomogeneousTurbulence(1.5, 1.5, 1.5, 100.0, 100.0, 100.0)
    height_key, velocity_key, run_key = jax.random.split(jax.random.key(0), 3)
    n = 20_000
    height = 50.0 * jax.random.uniform(height_key, (n,))
    position = jnp.stack([jnp.zeros(n), jnp.zeros(n), height], axis=1)
    state = eqx.tree_at(lambda s: s.position, at_point(jnp.zeros(3), n), position)
    state = with_turbulent_velocity(state, turbulence, key=velocity_key)
    result = integrate(
        state,
        Langevin(uniform(jnp.zeros(3)), turbulence, Walls(0.0, 50.0)),
        t0=0.0,
        duration=1_200.0,
        step_size=30.0,
        key=run_key,
    )
    z = np.asarray(result.state.position[:, 2])
    assert ((z >= 0.0) & (z <= 50.0)).all()
    # uniform still: the share below half the layer, to four standard errors
    assert abs((z < 25.0).mean() - 0.5) < 4.0 * 0.5 / np.sqrt(n)
