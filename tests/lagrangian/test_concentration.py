"""Tests for the forward Lagrangian concentration field."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from plumax.lagrangian.concentration import (
    _step_durations,
    bin_positions,
    simulate_lagrangian,
)
from plumax.lagrangian.turbulence import HomogeneousTurbulence


DOMAIN = dict(
    domain_x=(-50.0, 600.0, 65),
    domain_y=(-150.0, 150.0, 30),
    domain_z=(0.0, 200.0, 20),
)


def _turb():
    return HomogeneousTurbulence(1.0, 1.0, 0.6, 30.0, 30.0, 20.0)


@pytest.mark.parametrize(
    "horizon,dt,n",
    [(10.0, 1.0, 10), (10.1, 1.0, 11), (10.5, 2.0, 6), (0.5, 1.0, 1)],
)
def test_step_durations_sum_to_horizon(horizon, dt, n):
    dts = np.asarray(_step_durations(horizon, dt, n))
    assert dts.size == n
    assert dts.sum() == pytest.approx(horizon)
    # All but the last are the full dt; the last is the (≤ dt) remainder.
    np.testing.assert_allclose(dts[:-1], dt)
    assert 0.0 < dts[-1] <= dt + 1e-9


def test_step_durations_empty_when_zero_steps():
    assert np.asarray(_step_durations(0.0, 1.0, 0)).size == 0


def test_total_residence_tracks_horizon_for_partial_step():
    # With all particles confined to the domain, the total residence time
    # Σ conc · V_cell / Q equals the integration horizon. A non-divisible
    # horizon (t_end=10.5, dt=1) must give 10.5 s, not the 11 s a ceil'd
    # full-dt accumulation would produce.
    big_domain = dict(
        domain_x=(-500.0, 500.0, 20),
        domain_y=(-500.0, 500.0, 20),
        domain_z=(0.0, 400.0, 16),
    )
    # Calm, isotropic turbulence + zero wind keeps the cloud near the source
    # and inside the box over the short horizon.
    turb = HomogeneousTurbulence.isotropic(sigma=0.5, tau=30.0)
    common = dict(
        emission_rate=1.0,
        source_location=(0.0, 0.0, 200.0),
        turbulence=turb,
        wind=lambda t: jnp.zeros(3),
        n_particles=3000,
        dt=1.0,
        seed=0,
        **big_domain,
    )
    dx = 1000.0 / 20
    dy = 1000.0 / 20
    dz = 400.0 / 16
    cell_v = dx * dy * dz
    ds = simulate_lagrangian(t_end=10.5, **common)
    total_residence = float(ds["concentration"].sum()) * cell_v  # /Q, Q=1
    assert total_residence == pytest.approx(10.5, rel=1e-3)


def test_bin_positions_counts():
    pos = jnp.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [9.9, 9.9, 9.9]])
    edges = jnp.linspace(0.0, 10.0, 11)
    hist = bin_positions(pos, edges, edges, edges)
    assert hist.sum() == 3
    assert hist[0, 0, 0] == 2


def test_simulate_returns_dataset_with_positive_plume():
    ds = simulate_lagrangian(
        emission_rate=1.0,
        source_location=(0.0, 0.0, 20.0),
        turbulence=_turb(),
        wind_speed=5.0,
        wind_direction=270.0,  # from west -> flows east (+x)
        n_particles=3000,
        t_end=180.0,
        dt=1.0,
        seed=0,
        **DOMAIN,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds["concentration"].dims == ("x", "y", "z")
    assert float(ds["concentration"].max()) > 0.0


def test_plume_transported_downwind():
    ds = simulate_lagrangian(
        emission_rate=1.0,
        source_location=(0.0, 0.0, 20.0),
        turbulence=_turb(),
        wind_speed=5.0,
        wind_direction=270.0,
        n_particles=4000,
        t_end=180.0,
        dt=1.0,
        seed=1,
        **DOMAIN,
    )
    col = ds["column_concentration"]
    # Mass centroid in x must lie downwind (east) of the source at x = 0.
    weights = col.sum("y")
    x = ds["x"].values
    x_centroid = float((x * weights).sum() / weights.sum())
    assert x_centroid > 20.0


def test_concentration_scales_linearly_with_emission():
    common = dict(
        source_location=(0.0, 0.0, 20.0),
        turbulence=_turb(),
        wind_speed=5.0,
        wind_direction=270.0,
        n_particles=3000,
        t_end=150.0,
        dt=1.0,
        seed=5,
        **DOMAIN,
    )
    ds1 = simulate_lagrangian(emission_rate=1.0, **common)
    ds2 = simulate_lagrangian(emission_rate=3.0, **common)
    # Same particles/seed → field scales exactly with Q.
    np.testing.assert_allclose(
        np.asarray(ds2["concentration"].values),
        3.0 * np.asarray(ds1["concentration"].values),
        rtol=1e-6,
    )


def test_background_is_added():
    common = dict(
        emission_rate=1.0,
        source_location=(0.0, 0.0, 20.0),
        turbulence=_turb(),
        wind_speed=5.0,
        wind_direction=270.0,
        n_particles=2000,
        t_end=120.0,
        dt=1.0,
        seed=3,
        **DOMAIN,
    )
    ds0 = simulate_lagrangian(background_conc=0.0, **common)
    dsb = simulate_lagrangian(background_conc=0.01, **common)
    diff = np.asarray(dsb["concentration"].values) - np.asarray(
        ds0["concentration"].values
    )
    np.testing.assert_allclose(diff, 0.01, atol=1e-9)


def test_invalid_emission_and_wind_args_raise():
    with pytest.raises(ValueError, match="emission_rate"):
        simulate_lagrangian(
            emission_rate=0.0,
            source_location=(0.0, 0.0, 10.0),
            turbulence=_turb(),
            wind_speed=5.0,
            wind_direction=270.0,
            **DOMAIN,
        )
    with pytest.raises(ValueError, match="wind"):
        simulate_lagrangian(
            emission_rate=1.0,
            source_location=(0.0, 0.0, 10.0),
            turbulence=_turb(),
            **DOMAIN,
        )
