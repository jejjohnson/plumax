"""Tests for Monin–Obukhov similarity, the PBL diagnostic and the Hanna factory."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from plumax.les_fvm.grid import make_grid
from plumax.met import (
    MetField,
    MoninObukhov,
    hanna_turbulence_from,
    pbl_height_bulk_richardson,
)
from plumax.met.surface_layer import (
    CP_DRY,
    GRAVITY,
    KARMAN,
    R_DRY,
    friction_velocity,
    phi_h,
    phi_m,
    psi_h,
    psi_m,
)


GRID = make_grid((0.0, 400.0, 4), (0.0, 300.0, 4), (0.0, 1000.0, 20), dtype=jnp.float64)
Z0 = 0.1  # m
PBLH = 800.0  # m


def heights() -> np.ndarray:
    z = np.asarray(GRID.z)
    return z - (z[0] - 0.5 * GRID.dz)


def make_field(
    *, u_of_z, temperature_of_z, pressure: float = 1.0e5, n_time: int = 2
) -> MetField:
    z = heights()
    nz, ny, nx = z.size, GRID.y.size, GRID.x.size
    shape = (n_time, nz, ny, nx)
    column = lambda f: np.broadcast_to(f(z)[None, :, None, None], shape)
    return MetField(
        plume_grid=GRID,
        times=jnp.arange(n_time, dtype=jnp.float64) * 3600.0,
        u=jnp.asarray(column(u_of_z)),
        v=jnp.zeros(shape),
        w=jnp.zeros(shape),
        temperature=jnp.asarray(column(temperature_of_z)),
        pressure=jnp.full(shape, pressure),
        pbl_height=jnp.full((n_time, ny, nx), PBLH),
    )


WIND = make_field(
    u_of_z=lambda z: 5.0 + 0.0 * z, temperature_of_z=lambda z: 300.0 + 0.0 * z
)


# ---- stability functions --------------------------------------------------


def test_phi_equals_one_minus_zeta_dpsi():
    zeta = jnp.concatenate([jnp.linspace(-5.0, -1e-3, 40), jnp.linspace(1e-3, 2.0, 40)])
    for phi, psi in ((phi_m, psi_m), (phi_h, psi_h)):
        dpsi = jax.vmap(jax.grad(psi))(zeta)
        np.testing.assert_allclose(phi(zeta), 1.0 - zeta * dpsi, rtol=1e-10)


def test_neutral_limits_and_continuity():
    for fn, at_zero in ((phi_m, 1.0), (phi_h, 1.0), (psi_m, 0.0), (psi_h, 0.0)):
        for zeta in (-1e-9, 0.0, 1e-9):
            assert float(fn(jnp.asarray(zeta))) == pytest.approx(at_zero, abs=1e-7)


# ---- u*, L, w* ------------------------------------------------------------


def test_neutral_is_the_log_law():
    layer = MoninObukhov()(WIND, z0=Z0, sensible_heat_flux=0.0)
    z_ref = 0.5 * GRID.dz
    np.testing.assert_allclose(
        layer.u_star, KARMAN * 5.0 / np.log(z_ref / Z0), rtol=1e-12
    )
    assert np.all(np.asarray(layer.inverse_obukhov_length) == 0.0)
    assert np.all(np.asarray(layer.w_star) == 0.0)


@pytest.mark.parametrize("heat", [200.0, -20.0])
def test_fixed_point_satisfies_both_similarity_equations(heat):
    layer = MoninObukhov()(WIND, z0=Z0, sensible_heat_flux=heat)
    u_star = np.asarray(layer.u_star)
    inv_l = np.asarray(layer.inverse_obukhov_length)
    rho = 1.0e5 / (R_DRY * 300.0)
    # 1/L = −κ g H / (ρ c_p T u*³)
    np.testing.assert_allclose(
        inv_l, -KARMAN * GRAVITY * heat / (rho * CP_DRY * 300.0 * u_star**3), rtol=1e-6
    )
    np.testing.assert_allclose(
        u_star,
        friction_velocity(jnp.asarray(5.0), 0.5 * GRID.dz, Z0, jnp.asarray(inv_l)),
        rtol=1e-6,
    )
    assert np.all(np.sign(inv_l) == -np.sign(heat))  # heating → unstable


def test_convective_velocity_scale():
    heat = 200.0
    layer = MoninObukhov()(WIND, z0=Z0, sensible_heat_flux=heat)
    rho = 1.0e5 / (R_DRY * 300.0)
    expected = (GRAVITY * heat * PBLH / (rho * CP_DRY * 300.0)) ** (1.0 / 3.0)
    np.testing.assert_allclose(layer.w_star, expected, rtol=1e-12)


@pytest.mark.parametrize("heat", [0.0, -20.0, 200.0])
def test_gradients_are_finite(heat):
    def total(h):
        layer = MoninObukhov()(WIND, z0=Z0, sensible_heat_flux=h)
        return jnp.sum(layer.u_star + layer.inverse_obukhov_length + layer.w_star)

    assert np.isfinite(float(jax.grad(total)(jnp.asarray(heat))))


# ---- Hanna factory ----------------------------------------------------------


def test_hanna_factory_regimes():
    neutral = MoninObukhov()(WIND, z0=Z0, sensible_heat_flux=0.0)
    turb = hanna_turbulence_from(WIND, neutral, it=0, iy=1, ix=2)
    assert turb.obukhov_length == 0.0  # Hanna's neutral branch
    assert turb.pbl_height == PBLH

    convective = MoninObukhov()(WIND, z0=Z0, sensible_heat_flux=200.0)
    turb = hanna_turbulence_from(WIND, convective, it=1, iy=0, ix=0)
    assert turb.obukhov_length < 0.0 and turb.w_star > 0.0
    sigma, tau = turb.at(jnp.array([10.0, 200.0, 600.0]))
    assert np.all(np.isfinite(np.asarray(sigma))) and np.all(np.asarray(tau) > 0.0)


# ---- bulk Richardson PBL height --------------------------------------------


def test_bulk_richardson_matches_direct_profile():
    pytest.importorskip("xrtoolz")
    inversion = 400.0

    def theta(z):
        return 300.0 + 0.01 * np.maximum(z - inversion, 0.0)

    def u(z):
        return 2.0 + 0.005 * z

    field = make_field(u_of_z=u, temperature_of_z=theta)  # p = p_ref, so θ = T
    h = pbl_height_bulk_richardson(field, ri_crit=0.25)

    # Ri_b(z) = g (θ(z) − θ_s)(z − z_s) / (θ_s (u(z) − u_s)²), first crossing
    z = heights()
    ri = np.zeros_like(z)
    ri[1:] = GRAVITY * (theta(z[1:]) - theta(z[0])) * (z[1:] - z[0])
    ri[1:] /= theta(z[0]) * (u(z[1:]) - u(z[0])) ** 2
    k = int(np.argmax(ri >= 0.25))
    frac = (0.25 - ri[k - 1]) / (ri[k] - ri[k - 1])
    expected = z[k - 1] + frac * (z[k] - z[k - 1])
    # the crossing is interpolated from the last level below the inversion
    assert inversion - GRID.dz < expected < z[-1]
    np.testing.assert_allclose(h, expected, rtol=1e-9)


def test_buoyancy_uses_actual_temperature_off_reference_pressure():
    """At p ≠ p_ref the θ/T factor must cancel: 1/L and w* carry T, not θ."""
    p, t, heat = 85_000.0, 290.0, 150.0
    field = make_field(
        u_of_z=lambda z: 5.0 + 0.0 * z,
        temperature_of_z=lambda z: t + 0.0 * z,
        pressure=p,
    )
    layer = MoninObukhov()(field, z0=Z0, sensible_heat_flux=heat)
    rho = p / (R_DRY * t)
    u_star = np.asarray(layer.u_star)
    np.testing.assert_allclose(
        layer.inverse_obukhov_length,
        -KARMAN * GRAVITY * heat / (rho * CP_DRY * t * u_star**3),
        rtol=1e-6,
    )
    expected_w = (GRAVITY * heat * PBLH / (rho * CP_DRY * t)) ** (1.0 / 3.0)
    np.testing.assert_allclose(layer.w_star, expected_w, rtol=1e-12)


@pytest.mark.parametrize("heat", [0.0, 100.0, -20.0])
def test_calm_columns_stay_finite(heat):
    calm = make_field(
        u_of_z=lambda z: 0.0 * z, temperature_of_z=lambda z: 300.0 + 0.0 * z
    )
    layer = MoninObukhov()(calm, z0=Z0, sensible_heat_flux=heat)
    for value in (layer.u_star, layer.inverse_obukhov_length, layer.w_star):
        assert np.all(np.isfinite(np.asarray(value)))
    np.testing.assert_array_equal(layer.u_star, 0.0)
    if heat == 0.0:  # calm and neutral: the neutral limit
        np.testing.assert_array_equal(layer.inverse_obukhov_length, 0.0)
        turb = hanna_turbulence_from(calm, layer, it=0, iy=0, ix=0)
        sigma, tau = turb.at(jnp.array([10.0, 100.0]))
        assert np.all(np.isfinite(np.asarray(sigma))) and np.all(
            np.isfinite(np.asarray(tau))
        )


@pytest.mark.parametrize("heat", [0.0, 100.0, -20.0])
def test_calm_columns_have_finite_wind_gradients(heat):
    calm = make_field(
        u_of_z=lambda z: 0.0 * z, temperature_of_z=lambda z: 300.0 + 0.0 * z
    )

    def total_u_star(u, v):
        field = eqx.tree_at(lambda f: (f.u, f.v), calm, (u, v))
        return jnp.sum(MoninObukhov()(field, z0=Z0, sensible_heat_flux=heat).u_star)

    grad_u, grad_v = jax.grad(total_u_star, argnums=(0, 1))(calm.u, calm.v)
    assert np.all(np.isfinite(np.asarray(grad_u)))
    assert np.all(np.isfinite(np.asarray(grad_v)))


# ---- house-style quartet for the forward: property, gradient, transforms ----


@settings(max_examples=100, deadline=None)
@given(zeta=st.floats(-10.0, 5.0).filter(lambda z: abs(z) > 1e-6))
def test_phi_identity_property(zeta):
    for phi, psi in ((phi_m, psi_m), (phi_h, psi_h)):
        z = jnp.asarray(zeta)
        dpsi = jax.grad(psi)(z)
        np.testing.assert_allclose(phi(z), 1.0 - z * dpsi, rtol=1e-9)


@pytest.mark.parametrize("heat", [-20.0, 50.0, 200.0])
def test_gradient_matches_central_differences(heat):
    def friction(h):
        layer = MoninObukhov()(WIND, z0=Z0, sensible_heat_flux=h)
        return layer.u_star[0, 1, 1]

    step = 1e-3
    analytic = jax.jacrev(friction)(jnp.asarray(heat))
    numeric = (friction(heat + step) - friction(heat - step)) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


def test_eager_jit_and_vmap_agree_with_a_loop():
    heats = jnp.array([-20.0, 0.0, 50.0, 200.0])
    diagnose = lambda h: MoninObukhov()(WIND, z0=Z0, sensible_heat_flux=h).u_star
    looped = jnp.stack([diagnose(h) for h in heats])
    np.testing.assert_allclose(jax.jit(diagnose)(heats[2]), looped[2], rtol=1e-12)
    np.testing.assert_allclose(jax.vmap(diagnose)(heats), looped, rtol=1e-12)


def test_iteration_count_is_validated():
    from plumax.errors import PlumaxError

    with pytest.raises(PlumaxError, match="n_iterations"):
        MoninObukhov(n_iterations=0)
    assert MoninObukhov().with_n_iterations(5).n_iterations == 5
