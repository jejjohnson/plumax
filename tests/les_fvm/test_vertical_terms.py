"""Tests for the vertical terms of les_fvm's advection and diffusion."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.les_fvm.advection import advection_tendency
from plumax.les_fvm.diffusion import EddyDiffusivity, diffusion_tendency
from plumax.les_fvm.grid import make_grid


def _grid():
    # Δz = 0.5
    return make_grid((0.0, 4.0, 4), (0.0, 4.0, 4), (0.0, 2.0, 4))


def _profile(g, power: int):
    """Return C = z^power on every column, ghosts included."""
    z = jnp.arange(g.shape[0], dtype=jnp.float32) * g.dz
    return jnp.broadcast_to((z**power)[:, None, None], g.shape)


def _advect(g, concentration, w_value):
    calm = jnp.zeros(g.shape, dtype=jnp.float32)
    w = jnp.full(g.shape, w_value, dtype=jnp.float32)
    return np.asarray(advection_tendency(concentration, calm, calm, w, g))


def _diffuse(g, concentration, kappa_z):
    eddy = EddyDiffusivity(horizontal=0.0, vertical=kappa_z)
    return np.asarray(diffusion_tendency(concentration, eddy, g))


@pytest.mark.parametrize("w_value", [0.0, 2.0, -3.0])
def test_vertical_advection_of_a_linear_profile(w_value):
    # C = z: the upwind flux takes the cell below for w > 0 and above for
    # w < 0, and either way −∂z(w C) = −w on every interior cell
    g = _grid()
    tendency = _advect(g, _profile(g, 1), w_value)
    np.testing.assert_allclose(tendency[1:-1, 1:-1, 1:-1], -w_value, atol=1e-5)


def test_vertical_advection_is_upwind():
    # one full cell: an updraft moves it to the cell above, and only there
    g = _grid()
    concentration = jnp.zeros(g.shape, dtype=jnp.float32).at[2, 2, 2].set(1.0)
    tendency = _advect(g, concentration, 1.0)
    np.testing.assert_allclose(
        tendency[:, 2, 2], [0.0, 0.0, -1.0 / g.dz, 1.0 / g.dz, 0.0, 0.0], atol=1e-6
    )


def test_vertical_advection_reads_the_ghost_levels():
    # an updraft of 1 m/s through the floor carries the ghost value in
    g = _grid()
    concentration = jnp.zeros(g.shape, dtype=jnp.float32).at[0].set(3.0)
    tendency = _advect(g, concentration, 1.0)
    np.testing.assert_allclose(tendency[1, 2, 2], 3.0 / g.dz, rtol=1e-6)


def test_vertical_advection_averages_w_to_the_faces():
    # w = k, so w_{k+1/2} = k + ½, and a uniform C has −∂z(w C) = −C / Δz
    g = _grid()
    calm = jnp.zeros(g.shape, dtype=jnp.float32)
    level = jnp.arange(g.shape[0], dtype=jnp.float32)
    w = jnp.broadcast_to(level[:, None, None], g.shape)
    tendency = advection_tendency(jnp.ones(g.shape), calm, calm, w, g)
    np.testing.assert_allclose(
        np.asarray(tendency)[1:-1, 1:-1, 1:-1], -1.0 / g.dz, rtol=1e-5
    )


def test_vertical_diffusion_of_a_constant_profile_is_zero():
    g = _grid()
    np.testing.assert_allclose(_diffuse(g, jnp.ones(g.shape), 1.5), 0.0, atol=1e-6)


def test_vertical_diffusion_of_a_quadratic_profile():
    # ∂z(K ∂z z²) = 2 K, which the central difference gives exactly
    g = _grid()
    tendency = _diffuse(g, _profile(g, 2), 2.0)
    np.testing.assert_allclose(tendency[1:-1, 1:-1, 1:-1], 4.0, atol=1e-4)


def test_vertical_diffusion_averages_a_field_kappa():
    # K = z on the T-points (edge-padded ghosts), C = z: ∂z(K ∂z C) = 1 away
    # from the walls, where the padded K is constant over half a cell
    g = _grid()
    z = (jnp.arange(4, dtype=jnp.float32) + 1.0) * g.dz
    kappa = jnp.broadcast_to(z[:, None, None], g.interior_shape)
    tendency = _diffuse(g, _profile(g, 1), kappa)
    np.testing.assert_allclose(tendency[2:-2, 1:-1, 1:-1], 1.0, atol=1e-5)


def test_tendencies_are_zero_on_the_ghost_ring():
    g = _grid()
    rng = np.random.default_rng(0)
    concentration = jnp.asarray(rng.random(g.shape), dtype=jnp.float32)
    ring = np.ones(g.shape, dtype=bool)
    ring[1:-1, 1:-1, 1:-1] = False
    for tendency in (_advect(g, concentration, 1.0), _diffuse(g, concentration, 1.0)):
        assert (tendency[ring] == 0.0).all()
        assert (tendency[~ring] != 0.0).any()
