"""The formulas plumax now takes from geophysix equal the ones it wrote inline (plumax#389)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from geophysix.columns import column_from_mixing_ratio
from geophysix.thermo import dry_air_density, potential_temperature, vertical_velocity

from plumax._src.constants import (
    AVOGADRO,
    CP_DRY,
    GRAVITY,
    M_CH4,
    P_REFERENCE,
    R_DRY,
    T_STANDARD,
)
from plumax._src.wind import wind_components
from plumax.obs._src.ingest import ppb_to_column_mass


RNG = np.random.default_rng(389)
P = RNG.uniform(2e4, 1.05e5, 64)
T = RNG.uniform(200.0, 310.0, 64)
OMEGA = RNG.normal(0.0, 2.0, 64)


def test_potential_temperature_is_the_pbl_formula() -> None:
    old = T * (P_REFERENCE / P) ** (R_DRY / CP_DRY)
    new = potential_temperature(T, P, reference_pressure=P_REFERENCE)
    np.testing.assert_array_equal(new, old)


def test_density_and_vertical_velocity_are_the_era5_formulas() -> None:
    density = P / (R_DRY * T)
    np.testing.assert_array_equal(dry_air_density(P, T), density)
    np.testing.assert_array_equal(
        vertical_velocity(OMEGA, P, T), -OMEGA / (density * GRAVITY)
    )


def test_ppb_to_column_mass_is_the_old_formula() -> None:
    delta = RNG.uniform(-50.0, 50.0, 64)
    dry_air = RNG.uniform(1e29, 2.2e29, 64)
    old = delta * (1.0e-9 / AVOGADRO) * dry_air * M_CH4
    np.testing.assert_allclose(ppb_to_column_mass(delta, dry_air), old, rtol=1e-14)
    np.testing.assert_allclose(
        ppb_to_column_mass(delta, dry_air),
        M_CH4 * column_from_mixing_ratio(delta * 1e-9, dry_air / AVOGADRO),
        rtol=1e-15,
    )


def test_wind_components_are_the_old_formula() -> None:
    speed = np.abs(RNG.normal(5.0, 3.0, 64))
    direction = RNG.uniform(0.0, 360.0, 64)
    u, v = wind_components(speed, direction, xp=np)
    np.testing.assert_array_equal(u, -speed * np.sin(np.deg2rad(direction)))
    np.testing.assert_array_equal(v, -speed * np.cos(np.deg2rad(direction)))


def test_wind_components_keep_the_jax_default() -> None:
    import jax

    u, v = wind_components(5.0, 270.0)
    assert isinstance(u, jax.Array)
    assert isinstance(v, jax.Array)


def test_standard_temperature_is_isa() -> None:
    assert T_STANDARD == 288.15


def test_standard_molar_volume_is_deprecated() -> None:
    from plumax.geo import plume_array

    with pytest.warns(DeprecationWarning, match="geophysix.columns.convert"):
        assert plume_array.STANDARD_MOLAR_VOLUME_M3_PER_MOL == 0.024465
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "STANDARD_MOLAR_VOLUME_M3_PER_MOL" not in plume_array.__all__
