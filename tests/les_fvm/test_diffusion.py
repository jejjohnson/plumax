"""Tests for les_fvm.diffusion — K-theory eddy diffusivity."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from plumax.gauss_puff.dispersion import calculate_pg_dispersion, get_pg_params
from plumax.les_fvm.diffusion import (
    EddyDiffusivity,
    diffusion_tendency,
    make_eddy_diffusivity,
    mo_eddy_diffusivity,
    pg_eddy_diffusivity,
    smagorinsky_k_h,
)
from plumax.les_fvm.grid import make_grid
from plumax.les_fvm.simulate import _max_diffusivity
from plumax.les_fvm.wind import uniform_wind_field, wind_field_from_callable
from plumax.met.surface_layer import KARMAN, SurfaceLayer


def _build_grid():
    return make_grid(
        domain_x=(0.0, 100.0, 8),
        domain_y=(0.0, 100.0, 8),
        domain_z=(0.0, 40.0, 4),
    )


def test_constant_field_has_zero_tendency():
    g = _build_grid()
    c = jnp.ones(g.shape)
    kappa = EddyDiffusivity(horizontal=1.5, vertical=0.4)
    t = diffusion_tendency(c, kappa, g)
    np.testing.assert_allclose(np.asarray(t), 0.0, atol=1e-6)


def test_quadratic_z_field_matches_expected_tendency():
    # For C(z) = z² and uniform K_z, vertical Laplacian = 2 * K_z.
    g = _build_grid()
    z = jnp.arange(g.shape[0], dtype=jnp.float32) * g.dz
    c = jnp.broadcast_to(z[:, None, None] ** 2, g.shape)
    kappa = EddyDiffusivity(horizontal=0.0, vertical=2.0)
    t = diffusion_tendency(c, kappa, g)
    # Interior cells carry 2 * K_z = 4.0; ghost rings are zero.
    t_int = np.asarray(t)[1:-1, 1:-1, 1:-1]
    np.testing.assert_allclose(t_int, 4.0, atol=1e-3)


def test_pg_eddy_diffusivity_positive_values():
    eddy = pg_eddy_diffusivity(
        stability_class="C", wind_speed=5.0, reference_distance=200.0
    )
    assert eddy.horizontal > 0.0
    assert eddy.vertical > 0.0
    # Under neutral class C the vertical diffusivity is smaller than the
    # lateral, consistent with σ_z < σ_y.
    assert eddy.vertical < eddy.horizontal


def test_pg_eddy_diffusivity_rejects_zero_wind_speed():
    with pytest.raises(ValueError, match=r"wind_speed must be > 0"):
        pg_eddy_diffusivity(stability_class="C", wind_speed=0.0)


def test_pg_eddy_diffusivity_rejects_zero_reference_distance():
    with pytest.raises(ValueError, match=r"reference_distance must be > 0"):
        pg_eddy_diffusivity(stability_class="C", wind_speed=5.0, reference_distance=0.0)


def test_make_eddy_diffusivity_from_scalar():
    e = make_eddy_diffusivity(0.7)
    assert isinstance(e, EddyDiffusivity)
    assert e.horizontal == 0.7
    assert e.vertical == 0.7


def test_make_eddy_diffusivity_from_tuple():
    e = make_eddy_diffusivity((1.5, 0.3))
    assert e.horizontal == 1.5
    assert e.vertical == 0.3


def test_make_eddy_diffusivity_pg_requires_class_and_speed():
    with pytest.raises(ValueError, match=r"requires both"):
        make_eddy_diffusivity("pg")


def test_make_eddy_diffusivity_unknown_string_rejected():
    with pytest.raises(ValueError, match=r"unknown string spec"):
        make_eddy_diffusivity("smagorinsky")


def test_dirichlet_wall_diffusive_flux():
    """A lateral Dirichlet ghost now drives a diffusive flux through the wall
    (issue #26 — horizontal diffusion previously ignored all lateral BCs)."""
    from plumax.les_fvm.boundary import (
        apply_boundary_conditions,
        build_default_concentration_bc,
    )

    g = _build_grid()
    hbc, vbc = build_default_concentration_bc(
        bc_x=(("dirichlet", 0.0), "outflow"),  # west wall value 0, east zero-grad
        bc_y=(("neumann", 0.0), ("neumann", 0.0)),  # side walls no-flux
        bc_z=("neumann", "neumann"),
    )
    # Uniform interior => zero interior gradient; only the west Dirichlet wall
    # (ghost = -interior) produces a flux.
    c = apply_boundary_conditions(jnp.ones(g.shape), hbc, vbc, g)
    kappa = EddyDiffusivity(horizontal=0.5, vertical=0.0)
    tend = diffusion_tendency(c, kappa, g)
    # West edge column (i=1) diffuses toward the lower wall value -> nonzero.
    assert float(jnp.max(jnp.abs(tend[1:-1, 1:-1, 1]))) > 1e-3
    # Everywhere else (no active wall flux, uniform field) stays ~0.
    np.testing.assert_allclose(np.asarray(tend[1:-1, 1:-1, 2:-1]), 0.0, atol=1e-5)


# --- Monin–Obukhov K_z and Smagorinsky K_h ---


def _surface_layer(u_star: float, inverse_length: float, shape=(1, 4, 4)):
    return SurfaceLayer(
        u_star=jnp.full(shape, u_star),
        inverse_obukhov_length=jnp.full(shape, inverse_length),
        w_star=jnp.zeros(shape),
    )


def _z_grid(z_min: float = 0.0):
    # Δz = 40 m: T-levels at 20, 60, 100, ... m above ground
    return make_grid(
        domain_x=(0.0, 400.0, 4),
        domain_y=(0.0, 400.0, 4),
        domain_z=(z_min, z_min + 400.0, 10),
        dtype=jnp.float64,
    )


def _sigma_z_d(distance: float) -> float:
    _, _, sigma_z = calculate_pg_dispersion(
        jnp.asarray(distance, dtype=jnp.float64), get_pg_params("D")
    )
    return float(sigma_z)


def test_mo_neutral_limit():
    # PG class D is defined on the 10 m wind, over rural z₀ = 0.1 m: neutral log law
    wind_speed = 5.0
    u_star = KARMAN * wind_speed / math.log(10.0 / 0.1)
    pbl_height = 200.0
    eddy = mo_eddy_diffusivity(
        _z_grid(), _surface_layer(u_star, 0.0), pbl_height, it=0, k_h=1.0
    )
    # T-level k = 2 sits at z = 100 m = z_i / 2
    k_mo = float(eddy.vertical[2, 0, 0])
    # Taylor's K = σ²/(2t) is calibrated where the plume has reached that height:
    # the travel distance with σ_z = z_i / 2
    distance = brentq(lambda x: _sigma_z_d(x) - 0.5 * pbl_height, 1e2, 1e5)
    pg = pg_eddy_diffusivity("D", wind_speed, reference_distance=distance)
    np.testing.assert_allclose(k_mo, pg.vertical, rtol=0.3)


def test_mo_neutral_closed_form_on_every_level():
    grid = _z_grid()
    eddy = mo_eddy_diffusivity(grid, _surface_layer(0.4, 0.0), 300.0, it=0, k_h=2.0)
    z = (np.arange(10) + 0.5) * 40.0
    expected = KARMAN * 0.4 * z * np.clip(1.0 - z / 300.0, 0.0, None) ** 2
    np.testing.assert_allclose(eddy.vertical[:, 0, 0], expected, rtol=1e-12)
    assert eddy.horizontal == 2.0


def test_mo_height_is_above_ground_not_absolute():
    # a grid whose surface sits at 1500 m has the same profile
    low = mo_eddy_diffusivity(_z_grid(), _surface_layer(0.4, 0.0), 300.0, it=0, k_h=1.0)
    high = mo_eddy_diffusivity(
        _z_grid(1500.0), _surface_layer(0.4, 0.0), 300.0, it=0, k_h=1.0
    )
    np.testing.assert_allclose(high.vertical, low.vertical, rtol=1e-12)


def test_mo_does_not_mix_above_the_boundary_layer():
    eddy = mo_eddy_diffusivity(
        _z_grid(), _surface_layer(0.4, 0.0), 150.0, it=0, k_h=1.0
    )
    # z = 180 m and above: without the clip (1 − z/z_i)² would grow again
    assert np.all(np.asarray(eddy.vertical[4:]) == 0.0)
    assert np.all(np.asarray(eddy.vertical[:4]) > 0.0)


def test_mo_stability_ordering_follows_phi_h():
    grid = _z_grid()
    neutral = mo_eddy_diffusivity(grid, _surface_layer(0.4, 0.0), 400.0, it=0, k_h=1.0)
    stable = mo_eddy_diffusivity(grid, _surface_layer(0.4, 0.01), 400.0, it=0, k_h=1.0)
    unstable = mo_eddy_diffusivity(
        grid, _surface_layer(0.4, -0.01), 400.0, it=0, k_h=1.0
    )
    assert np.all(np.asarray(stable.vertical) < np.asarray(neutral.vertical))
    assert np.all(np.asarray(unstable.vertical) > np.asarray(neutral.vertical))
    # ζ = 100 × 0.01 = 1 at k = 2: φ_h = 1 + 5ζ = 6 stable, (1 + 16)^{−1/2} unstable
    base = float(neutral.vertical[2, 0, 0])
    np.testing.assert_allclose(stable.vertical[2, 0, 0], base / 6.0, rtol=1e-12)
    np.testing.assert_allclose(
        unstable.vertical[2, 0, 0], base * math.sqrt(17.0), rtol=1e-12
    )


def test_mo_indexes_time_in_the_layer_and_the_pbl_field():
    grid = _z_grid()
    shape = (2, 4, 4)
    layer = SurfaceLayer(
        u_star=jnp.stack([jnp.full((4, 4), 0.2), jnp.full((4, 4), 0.6)]),
        inverse_obukhov_length=jnp.zeros(shape),
        w_star=jnp.zeros(shape),
    )
    depth = jnp.stack([jnp.full((4, 4), 100.0), jnp.full((4, 4), 300.0)])
    field = mo_eddy_diffusivity(grid, layer, depth, it=1, k_h=1.0)
    scalar = mo_eddy_diffusivity(grid, _surface_layer(0.6, 0.0), 300.0, it=0, k_h=1.0)
    np.testing.assert_allclose(field.vertical, scalar.vertical, rtol=1e-12)


def test_mo_gradient_matches_central_difference():
    grid = _z_grid()

    def profile_sum(inverse_length):
        layer = _surface_layer(0.4, inverse_length)
        return mo_eddy_diffusivity(grid, layer, 400.0, it=0, k_h=1.0).vertical.sum()

    for point in (-0.01, 0.01):
        step = 1e-6
        central = (profile_sum(point + step) - profile_sum(point - step)) / (2 * step)
        np.testing.assert_allclose(jax.grad(profile_sum)(point), central, rtol=1e-6)


def test_mo_jit_matches_eager():
    grid = _z_grid()
    layer = _surface_layer(0.4, -0.01)

    def vertical(depth):
        return mo_eddy_diffusivity(grid, layer, depth, it=0, k_h=1.0).vertical

    np.testing.assert_allclose(jax.jit(vertical)(300.0), vertical(300.0), rtol=1e-12)


def test_mo_rejects_a_negative_pbl_height():
    grid = _z_grid()
    with pytest.raises(eqx.EquinoxRuntimeError, match="pbl_height"):
        mo_eddy_diffusivity(grid, _surface_layer(0.4, 0.0), -1.0, it=0, k_h=1.0)


def _smag_grid(dx: float = 10.0, dy: float = 10.0):
    return make_grid(
        domain_x=(0.0, 8.0 * dx, 8),
        domain_y=(0.0, 8.0 * dy, 8),
        domain_z=(0.0, 40.0, 4),
        dtype=jnp.float64,
    )


def _wind(grid, fn):
    return wind_field_from_callable(grid, fn)


def _k_h_interior(wind, grid, **kwargs):
    # one cell in from the domain edge, where the edge-padded ghost ring is unseen
    return np.asarray(smagorinsky_k_h(wind, grid, t=0.0, **kwargs))[:, 1:-1, 1:-1]


def test_smagorinsky_shear():
    grid = _smag_grid()
    uniform = smagorinsky_k_h(uniform_wind_field(grid, u=5.0, v=-2.0), grid, t=0.0)
    assert uniform.shape == grid.interior_shape
    assert np.all(np.asarray(uniform) == 0.0)
    shear = _wind(grid, lambda t, x, y, z: (0.1 * y, jnp.zeros_like(y), 0.0 * y))
    k_h = smagorinsky_k_h(shear, grid, t=0.0)
    assert np.all(np.asarray(k_h) > 0.0)
    # D_S = ∂u/∂y = S: K_h = (c_s Δ)² S
    np.testing.assert_allclose(k_h[:, 1:-1, 1:-1], (0.2 * 10.0) ** 2 * 0.1, rtol=1e-9)


def test_smagorinsky_strain_flow_is_tension():
    # u = S x, v = −S y: D_T = 2 S, D_S = 0
    grid = _smag_grid()
    wind = _wind(grid, lambda t, x, y, z: (0.05 * x, -0.05 * y, 0.0 * x))
    k_h = _k_h_interior(wind, grid, c_s=0.25)
    np.testing.assert_allclose(k_h, (0.25 * 10.0) ** 2 * 2.0 * 0.05, rtol=1e-9)


def test_smagorinsky_vanishes_without_deformation():
    grid = _smag_grid()
    # solid-body rotation: ∂v/∂x = Ω, ∂u/∂y = −Ω, D_S = 0
    rotation = _wind(grid, lambda t, x, y, z: (-0.1 * y, 0.1 * x, 0.0 * x))
    np.testing.assert_allclose(_k_h_interior(rotation, grid), 0.0, atol=1e-9)
    # isotropic horizontal divergence: D_T = 0
    divergence = _wind(grid, lambda t, x, y, z: (0.1 * x, 0.1 * y, 0.0 * x))
    np.testing.assert_allclose(_k_h_interior(divergence, grid), 0.0, atol=1e-9)


def test_smagorinsky_ignores_vertical_shear():
    grid = _smag_grid()
    wind = _wind(grid, lambda t, x, y, z: (0.5 * z, 0.0 * z, 0.0 * z))
    assert np.all(np.asarray(smagorinsky_k_h(wind, grid, t=0.0)) == 0.0)


def test_smagorinsky_filter_scale_on_an_anisotropic_grid():
    grid = _smag_grid(dx=10.0, dy=40.0)
    shear = _wind(grid, lambda t, x, y, z: (0.1 * y, 0.0 * y, 0.0 * y))
    # Δ² = Δx Δy
    expected = 0.2**2 * (10.0 * 40.0) * 0.1
    np.testing.assert_allclose(_k_h_interior(shear, grid), expected, rtol=1e-9)


def test_smagorinsky_follows_time():
    grid = _smag_grid()
    wind = _wind(grid, lambda t, x, y, z: (t * y, 0.0 * y, 0.0 * y))
    k_h = np.asarray(smagorinsky_k_h(wind, grid, t=2.0))[:, 1:-1, 1:-1]
    np.testing.assert_allclose(k_h, (0.2 * 10.0) ** 2 * 2.0, rtol=1e-9)


def test_smagorinsky_gradient_is_finite_at_rest():
    # a uniform wind is the double-where branch: an adjoint must stay finite
    grid = _smag_grid()

    def total(speed):
        wind = _wind(grid, lambda t, x, y, z: (speed + 0.0 * y, 0.0 * y, 0.0 * y))
        return smagorinsky_k_h(wind, grid, t=0.0).sum()

    assert float(jax.grad(total)(3.0)) == 0.0


def test_smagorinsky_gradient_matches_central_difference():
    grid = _smag_grid()

    def total(rate):
        wind = _wind(grid, lambda t, x, y, z: (rate * y, 0.3 * x, 0.0 * y))
        return smagorinsky_k_h(wind, grid, t=0.0).sum()

    step = 1e-6
    central = (total(0.1 + step) - total(0.1 - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(total)(0.1), central, rtol=1e-6)


def test_smagorinsky_rejects_a_non_positive_constant():
    grid = _smag_grid()
    with pytest.raises(ValueError, match="c_s must be > 0"):
        smagorinsky_k_h(uniform_wind_field(grid), grid, c_s=0.0, t=0.0)


def test_field_diffusivities_reach_the_stability_guard():
    grid = _smag_grid()
    k_h = jnp.full(grid.interior_shape, 1.0).at[1, 2, 3].set(7.0)
    k_z = jnp.full(grid.interior_shape, 0.5).at[2, 3, 4].set(3.0)
    eddy = EddyDiffusivity(horizontal=k_h, vertical=k_z)
    assert _max_diffusivity(eddy) == (7.0, 3.0)
    with pytest.raises(ValueError, match="non-negative"):
        _max_diffusivity(
            EddyDiffusivity(horizontal=k_h.at[0, 0, 0].set(-1.0), vertical=k_z)
        )
    with pytest.raises(ValueError, match="finite"):
        _max_diffusivity(
            EddyDiffusivity(horizontal=k_h, vertical=k_z.at[0, 0, 0].set(jnp.nan))
        )


def test_mo_and_smagorinsky_fields_drive_the_tendency():
    # both closures on one grid, through the operator: a linear tracer in z under
    # a K_z(z) profile has the tendency ∂_z K_z · ∂_z C in the column interior
    grid = make_grid(
        domain_x=(0.0, 80.0, 8),
        domain_y=(0.0, 80.0, 8),
        domain_z=(0.0, 400.0, 10),
        dtype=jnp.float64,
    )
    shear = _wind(grid, lambda t, x, y, z: (0.1 * y, 0.0 * y, 0.0 * y))
    k_h = smagorinsky_k_h(shear, grid, t=0.0)
    layer = _surface_layer(0.4, 0.0, shape=(1, 8, 8))
    eddy = mo_eddy_diffusivity(grid, layer, 300.0, it=0, k_h=k_h)
    gradient = 0.01
    levels = jnp.arange(grid.shape[0], dtype=jnp.float64) * grid.dz
    concentration = jnp.broadcast_to(levels[:, None, None] * gradient, grid.shape)
    tendency = np.asarray(diffusion_tendency(concentration, eddy, grid))
    k_z = np.asarray(eddy.vertical)[:, 0, 0]
    # ∂_z(K_z ∂_z C) with the face-mean K_z, levels with both faces in the interior
    expected = gradient * (k_z[2:] - k_z[:-2]) / (2.0 * grid.dz)
    np.testing.assert_allclose(tendency[2:-2, 1, 1], expected, rtol=1e-9, atol=1e-12)
