"""Tests for the backward Lagrangian footprint."""

from __future__ import annotations

import numpy as np

from plumax.lagrangian.footprint import compute_footprint
from plumax.lagrangian.particles import wind_from_speed_direction
from plumax.lagrangian.turbulence import HomogeneousTurbulence


def _turb():
    return HomogeneousTurbulence(1.0, 1.0, 0.6, 30.0, 30.0, 20.0)


def test_footprint_shape_and_nonnegative():
    wind = wind_from_speed_direction(5.0, 270.0)
    fp, x, y = compute_footprint(
        (400.0, 0.0, 20.0),
        _turb(),
        domain_x=(-50.0, 500.0, 55),
        domain_y=(-150.0, 150.0, 30),
        wind=wind,
        n_particles=3000,
        t_back=180.0,
        dt=1.0,
        seed=0,
    )
    assert fp.shape == (55, 30)
    assert x.shape == (55,)
    assert y.shape == (30,)
    assert np.all(fp >= 0.0)
    assert fp.sum() > 0.0


def test_footprint_lies_upwind_of_receptor():
    # Wind from west (flows east); backward particles travel west, so the
    # surface influence of a receptor is upwind (smaller x).
    wind = wind_from_speed_direction(5.0, 270.0)
    receptor_x = 400.0
    fp, x, y = compute_footprint(
        (receptor_x, 0.0, 20.0),
        _turb(),
        domain_x=(-50.0, 500.0, 55),
        domain_y=(-150.0, 150.0, 30),
        wind=wind,
        n_particles=4000,
        t_back=180.0,
        dt=1.0,
        seed=1,
    )
    weights = fp.sum(axis=1)
    x_centroid = float((x * weights).sum() / weights.sum())
    assert x_centroid < receptor_x


def test_footprint_scales_inversely_with_air_density():
    wind = wind_from_speed_direction(5.0, 270.0)
    kw = dict(
        receptor_location=(300.0, 0.0, 20.0),
        turbulence=_turb(),
        domain_x=(-50.0, 400.0, 45),
        domain_y=(-120.0, 120.0, 24),
        wind=wind,
        n_particles=2000,
        t_back=150.0,
        dt=1.0,
        seed=2,
    )
    fp1, _, _ = compute_footprint(air_density=1.0, **kw)
    fp2, _, _ = compute_footprint(air_density=2.0, **kw)
    np.testing.assert_allclose(fp2, 0.5 * fp1, rtol=1e-6)
