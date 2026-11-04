"""Tests for the Tier II turbulence parameterisations."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.lagrangian.turbulence import HomogeneousTurbulence, hanna_profiles


def test_homogeneous_at_is_constant_in_height():
    turb = HomogeneousTurbulence(1.0, 0.8, 0.5, 100.0, 100.0, 50.0)
    z = jnp.array([0.0, 10.0, 500.0])
    sigma, tau = turb.at(z)
    assert sigma.shape == (3, 3)
    assert tau.shape == (3, 3)
    # Same row repeated for every height.
    np.testing.assert_allclose(np.asarray(sigma[0]), [1.0, 0.8, 0.5])
    np.testing.assert_allclose(np.asarray(sigma[2]), [1.0, 0.8, 0.5])


def test_isotropic_constructor():
    turb = HomogeneousTurbulence.isotropic(sigma=0.7, tau=30.0)
    np.testing.assert_allclose(np.asarray(turb.sigma), [0.7, 0.7, 0.7])
    np.testing.assert_allclose(np.asarray(turb.tau), [30.0, 30.0, 30.0])


def test_homogeneous_validation():
    with pytest.raises(ValueError, match="sigma_u"):
        HomogeneousTurbulence(-1.0, 1.0, 1.0, 10.0, 10.0, 10.0)
    with pytest.raises(ValueError, match="tau_w"):
        HomogeneousTurbulence(1.0, 1.0, 1.0, 10.0, 10.0, 0.0)


@pytest.mark.parametrize("L", [-50.0, 200.0, 1e9])  # unstable, stable, neutral
def test_hanna_profiles_positive_and_shaped(L):
    z = jnp.linspace(1.0, 900.0, 25)
    sigma, tau = hanna_profiles(
        z, u_star=0.4, pbl_height=1000.0, obukhov_length=L, w_star=1.5
    )
    assert sigma.shape == (25, 3)
    assert tau.shape == (25, 3)
    assert np.all(np.asarray(sigma) >= 0.0)
    assert np.all(np.asarray(tau) >= 1.0)


def test_hanna_unstable_has_larger_vertical_mixing_than_stable():
    z = jnp.array([100.0])
    sig_unstable, _ = hanna_profiles(
        z, u_star=0.4, pbl_height=1000.0, obukhov_length=-50.0, w_star=2.0
    )
    sig_stable, _ = hanna_profiles(
        z, u_star=0.4, pbl_height=1000.0, obukhov_length=50.0
    )
    # Convective conditions mix more vigorously in the vertical.
    assert float(sig_unstable[0, 2]) > float(sig_stable[0, 2])


# ---- Hanna (1982) against Stohl et al. (2005), eqs. 26–33 -------------------
# Hand-computed at z = 100 m, h = 500 m, u_* = 0.3 m/s, so z/h = 0.2.

Z, H, U_STAR = 100.0, 500.0, 0.3


def _hanna(obukhov_length, **kwargs):
    sigma, tau = hanna_profiles(
        jnp.array([Z]),
        u_star=U_STAR,
        pbl_height=H,
        obukhov_length=obukhov_length,
        **kwargs,
    )
    return np.asarray(sigma[0]), np.asarray(tau[0])


def test_hanna_stable_sigmas_match_eqs_29_30():
    # h/L = 2: stable. σ_u = 2.0 u_*(1 − z/h) = 0.48, σ_v = σ_w = 1.3 u_*(1 − z/h)
    sigma, _ = _hanna(250.0)
    np.testing.assert_allclose(sigma, [0.48, 0.312, 0.312], rtol=1e-6)


def test_hanna_stable_timescales_match_eqs_31_33():
    # τ_L,u = 0.15 (h/σ_u)(z/h)^0.5, τ_L,v = 0.07 (h/σ_v)(z/h)^0.5,
    # τ_L,w = 0.1 (h/σ_w)(z/h)^0.8 (Hanna's and hanna.f90's exponent)
    _, tau = _hanna(250.0)
    expected = [
        0.15 * 500.0 / 0.48 * 0.2**0.5,
        0.07 * 500.0 / 0.312 * 0.2**0.5,
        0.1 * 500.0 / 0.312 * 0.2**0.8,
    ]
    np.testing.assert_allclose(tau, expected, rtol=1e-6)
    np.testing.assert_allclose(tau, [69.877124, 50.168192, 44.222105], rtol=1e-6)


def test_hanna_neutral_profiles_match_eqs_26_28():
    # σ_u = 2.0 u_* exp(−3fz/u_*), σ_v = σ_w = 1.3 u_* exp(−2fz/u_*),
    # τ_L = 0.5 z/σ_w / (1 + 15fz/u_*), with f = 1e-4 s⁻¹
    sigma, tau = _hanna(0.0)
    f = 1e-4
    sigma_w = 1.3 * U_STAR * np.exp(-2.0 * f * Z / U_STAR)
    expected_sigma = [2.0 * U_STAR * np.exp(-3.0 * f * Z / U_STAR), sigma_w, sigma_w]
    np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-6)
    np.testing.assert_allclose(sigma, [0.542902, 0.364848, 0.364848], rtol=1e-5)
    expected_tau = 0.5 * Z / sigma_w / (1.0 + 15.0 * f * Z / U_STAR)
    np.testing.assert_allclose(tau, [expected_tau] * 3, rtol=1e-6)


def test_hanna_neutral_uses_the_coriolis_parameter():
    # Southern-hemisphere f < 0 gives the same profiles as |f|.
    f = 1.4e-4
    sigma, _ = _hanna(0.0, coriolis_parameter=-f)
    expected = 2.0 * U_STAR * np.exp(-3.0 * f * Z / U_STAR)
    np.testing.assert_allclose(sigma[0], expected, rtol=1e-6)
    default, _ = _hanna(0.0)
    assert sigma[0] < default[0]


@pytest.mark.parametrize("length", [600.0, -600.0, 1e9, -1e9])
def test_hanna_weak_stability_is_neutral(length):
    # hanna.f90 takes the neutral branch whenever h/|L| < 1, either sign of L.
    np.testing.assert_allclose(_hanna(length)[0], _hanna(0.0)[0], rtol=1e-12)
    np.testing.assert_allclose(_hanna(length)[1], _hanna(0.0)[1], rtol=1e-12)


def test_hanna_regime_threshold_at_h_over_abs_l_equal_one():
    # h/|L| = 1 is no longer neutral: L = h is stable, L = −h unstable.
    stable_sigma, _ = _hanna(H)
    np.testing.assert_allclose(stable_sigma[1], 1.3 * U_STAR * 0.8, rtol=1e-6)
    unstable_sigma, _ = _hanna(-H, w_star=1.0)
    expected_u = U_STAR * (12.0 + 0.5) ** (1.0 / 3.0)
    np.testing.assert_allclose(unstable_sigma[0], expected_u, rtol=1e-6)
