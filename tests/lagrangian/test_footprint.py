"""Tests for the backward Lagrangian footprint."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.lagrangian.footprint import compute_footprint
from plumax.lagrangian.particles import (
    ParticleState,
    integrate_particles,
    wind_from_speed_direction,
)
from plumax.lagrangian.turbulence import HomogeneousTurbulence


def _turb():
    return HomogeneousTurbulence(1.0, 1.0, 0.6, 30.0, 30.0, 20.0)


def test_footprint_integrates_requested_horizon_for_partial_step():
    # Flux-sensitivity convention (units s·m²·kg⁻¹): Σ footprint · (ρ · mix_height)
    # = total surface residence time — no cell-area factor — which equals t_back
    # when every particle stays below the mixing height and inside the domain. A
    # non-divisible horizon (t_back=60.5, dt=1) must give 60.5 s, not the 61 s a
    # ceil'd full-dt accumulation would.
    rho, t_back = 1.2, 60.5
    pbl_height, pbl_fraction = 2000.0, 0.5
    mix_height = pbl_fraction * pbl_height
    fp, _, _ = compute_footprint(
        (0.0, 0.0, 20.0),
        HomogeneousTurbulence.isotropic(sigma=0.5, tau=30.0),
        domain_x=(-500.0, 500.0, 40),
        domain_y=(-500.0, 500.0, 40),
        wind=lambda t: jnp.zeros(3),  # calm → cloud stays near the receptor
        n_particles=3000,
        t_back=t_back,
        dt=1.0,
        pbl_height=pbl_height,
        pbl_fraction=pbl_fraction,
        air_density=rho,
        seed=0,
    )
    total_residence = float(fp.sum()) * rho * mix_height
    assert total_residence == pytest.approx(t_back, rel=1e-3)


def test_footprint_units_convention():
    # Pin the flux-sensitivity convention dimensionally. Halving either the air
    # density or the mixing-layer depth doubles the footprint (F ∝ 1/(ρ·f·h)),
    # and there is NO dependence on the surface cell area — the distinguishing
    # property of s·m²·kg⁻¹ (flux) vs s·kg⁻¹ (per-cell rate, which would scale
    # with 1/A_cell). Same calm setup, so total residence is conserved.
    turb = HomogeneousTurbulence.isotropic(sigma=0.5, tau=30.0)
    base = dict(
        receptor_location=(0.0, 0.0, 20.0),
        turbulence=turb,
        wind=lambda t: jnp.zeros(3),
        n_particles=3000,
        t_back=60.0,
        dt=1.0,
        pbl_height=2000.0,
        pbl_fraction=0.5,
        air_density=1.2,
        seed=0,
    )
    fp_ref, _, _ = compute_footprint(
        domain_x=(-500.0, 500.0, 40), domain_y=(-500.0, 500.0, 40), **base
    )
    # Coarser grid → 4× larger cells. Flux sensitivity total is invariant to the
    # cell size (up to which cells particles land in); a per-cell-rate footprint
    # would instead scale each cell by 1/A_cell and change the sum.
    fp_coarse, _, _ = compute_footprint(
        domain_x=(-500.0, 500.0, 20), domain_y=(-500.0, 500.0, 20), **base
    )
    np.testing.assert_allclose(fp_ref.sum(), fp_coarse.sum(), rtol=1e-6)

    # Halving ρ or the mixing depth doubles F.
    fp_half_rho, _, _ = compute_footprint(
        domain_x=(-500.0, 500.0, 40),
        domain_y=(-500.0, 500.0, 40),
        **{**base, "air_density": 0.6},
    )
    np.testing.assert_allclose(fp_half_rho, 2.0 * fp_ref, rtol=1e-6)


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
    fp, x, _ = compute_footprint(
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


def test_time_varying_wind_clock():
    # For deterministic advection the backward trajectory from a receptor at
    # physical time T is the exact time-reverse of a forward trajectory that
    # arrives there at T. With a strongly time-varying (accelerating) wind this
    # only holds if the backward step samples the wind on the receptor clock,
    # -wind(T - τ). We forward-integrate a particle to fix the receptor, then
    # check the backward footprint retraces the forward path — and that the old
    # forward-clock behaviour (receptor_time=0) does not.
    calm = HomogeneousTurbulence(0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
    t_back = 60.0

    def wind(t):
        # Accelerating along +x: u sweeps 1 → 7 m/s over [0, T].
        return jnp.array([1.0 + 0.1 * t, 0.0, 0.0])

    # Forward reference: a particle from the source lands at the receptor at T.
    src = (0.0, 0.0, 20.0)
    state0 = ParticleState(position=jnp.array([[*src]]), velocity=jnp.zeros((1, 3)))
    final, traj = integrate_particles(
        state0,
        wind,
        calm,
        t0=0.0,
        t1=t_back,
        dt=1.0,
        key=jax.random.PRNGKey(0),
        save_trajectory=True,
    )
    receptor_x = float(final.position[0, 0])
    # The backward run bins the position after each step, visiting (in reverse)
    # the forward positions p₀…pₙ₋₁ — i.e. the source up to just-before-receptor.
    # Compare against that dt-weighted x-centroid of the forward path.
    x_ref = float(np.asarray(traj[:-1, 0, 0]).mean())

    common = dict(
        receptor_location=(receptor_x, 0.0, 20.0),
        turbulence=calm,
        domain_x=(-100.0, 400.0, 100),
        domain_y=(-50.0, 50.0, 10),
        wind=wind,
        n_particles=200,
        t_back=t_back,
        dt=1.0,
        pbl_height=2000.0,
        seed=0,
    )
    fp_ok, x_c, _ = compute_footprint(receptor_time=t_back, **common)
    fp_bad, _, _ = compute_footprint(receptor_time=0.0, **common)

    def centroid(fp):
        w = fp.sum(axis=1)
        return float((x_c * w).sum() / w.sum())

    x_ok, x_bad = centroid(fp_ok), centroid(fp_bad)
    # Correct clock is the exact discrete reverse of the forward run, so its
    # footprint centroid matches the forward path to binning resolution; a wrong
    # receptor clock samples the wind at the wrong physical times and lands a
    # materially different centroid.
    np.testing.assert_allclose(x_ok, x_ref, rtol=0.02)
    assert abs(x_ok - x_bad) > 20.0


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
