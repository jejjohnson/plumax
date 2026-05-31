"""Tests for the forward Lagrangian concentration field."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from plumax.lagrangian.concentration import bin_positions, simulate_lagrangian
from plumax.lagrangian.turbulence import HomogeneousTurbulence


DOMAIN = dict(
    domain_x=(-50.0, 600.0, 65),
    domain_y=(-150.0, 150.0, 30),
    domain_z=(0.0, 200.0, 20),
)


def _turb():
    return HomogeneousTurbulence(1.0, 1.0, 0.6, 30.0, 30.0, 20.0)


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
