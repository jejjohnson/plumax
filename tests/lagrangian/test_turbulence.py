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
