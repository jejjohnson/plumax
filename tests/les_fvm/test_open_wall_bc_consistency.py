"""Open-wall BC-consistency tests (issue #60).

Open-mode advection/diffusion read the ghost ring, so every per-axis BC must
be reflected consistently across the tracer *and* the coefficient/velocity
halos:

1. advection needs a **cell-value** Dirichlet ghost (``ghost = value``), while
   diffusion needs the **face** reflection (``2*value - interior``);
2. on a periodic axis the normal **wind** halo must be wrapped, not
   edge-padded, or a spatially-varying wind breaks seam conservation;
3. on a periodic axis a **field** ``K_h`` halo must be wrapped for the same
   reason.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from plumax.les_fvm.advection import advection_tendency
from plumax.les_fvm.boundary import (
    apply_boundary_conditions,
    build_default_concentration_bc,
    periodic_axes,
)
from plumax.les_fvm.diffusion import EddyDiffusivity, diffusion_tendency
from plumax.les_fvm.dynamics import EulerianDispersionRHS
from plumax.les_fvm.grid import make_grid
from plumax.les_fvm.source import GaussianSource
from plumax.les_fvm.wind import uniform_wind_field


def _build_grid():
    return make_grid(
        domain_x=(0.0, 160.0, 16),
        domain_y=(0.0, 160.0, 16),
        domain_z=(0.0, 40.0, 4),
    )


def _wrap_xy(f):
    f = f.at[:, :, 0].set(f[:, :, -2]).at[:, :, -1].set(f[:, :, 1])
    f = f.at[:, 0, :].set(f[:, -2, :]).at[:, -1, :].set(f[:, 1, :])
    return f


def _edge_pad_interior(interior):
    return jnp.asarray(
        np.pad(np.asarray(interior), ((1, 1), (1, 1), (1, 1)), mode="edge")
    )


# ── 1. cell-Dirichlet advective ghost ───────────────────────────────────────
def test_cell_dirichlet_advective_ghost_vs_face_reflection():
    g = _build_grid()
    hbc, vbc = build_default_concentration_bc(
        bc_x=(("dirichlet", 3.0), "outflow"),
        bc_y="periodic",
        bc_z=(("dirichlet", 1.0), "neumann"),
    )
    c = jnp.full(g.shape, 5.0)
    c_diff = apply_boundary_conditions(c, hbc, vbc, g)
    c_adv = apply_boundary_conditions(
        c, hbc.advection_variant(), vbc.advection_variant(), g
    )
    # West horizontal Dirichlet: diffusion 2*3-5=1 (face); advection 3 (cell).
    np.testing.assert_allclose(float(c_diff[2, 2, 0]), 1.0)
    np.testing.assert_allclose(float(c_adv[2, 2, 0]), 3.0)
    # Bottom vertical Dirichlet: diffusion 2*1-5=-3 (face); advection 1 (cell).
    np.testing.assert_allclose(float(c_diff[0, 3, 3]), 2.0 * 1.0 - 5.0)
    np.testing.assert_allclose(float(c_adv[0, 3, 3]), 1.0)


def test_advection_variant_is_noop_without_dirichlet():
    g = _build_grid()
    hbc, vbc = build_default_concentration_bc(
        bc_x="periodic", bc_y=("outflow", "outflow"), bc_z=("neumann", "neumann")
    )
    rng = np.random.default_rng(0)
    c = jnp.asarray(rng.normal(size=g.shape).astype(np.float32))
    base = apply_boundary_conditions(c, hbc, vbc, g)
    adv = apply_boundary_conditions(
        c, hbc.advection_variant(), vbc.advection_variant(), g
    )
    np.testing.assert_array_equal(np.asarray(base), np.asarray(adv))


def test_zero_dirichlet_inflow_injects_no_negative_tracer():
    """A zero-Dirichlet inflow wall must not create negative concentration.

    With the face reflection (ghost = -interior) an upwind inflow reads a
    negative ghost and can drive the wall-adjacent cell negative; the
    cell-value ghost (ghost = 0) keeps the monotone scheme non-negative.
    """
    g = _build_grid()
    hbc, vbc = build_default_concentration_bc(
        bc_x=(("dirichlet", 0.0), "outflow"),
        bc_y=(("dirichlet", 0.0), ("dirichlet", 0.0)),
        bc_z=("neumann", "neumann"),
    )
    hbc_a, vbc_a = hbc.advection_variant(), vbc.advection_variant()
    # Tracer sitting on the west interior edge (i=1), +x inflow wind.
    c = jnp.zeros(g.shape).at[2, 8, 1].set(1.0)
    u = jnp.ones(g.shape)
    v = jnp.zeros(g.shape)
    w = jnp.zeros(g.shape)
    for _ in range(30):
        c = apply_boundary_conditions(c, hbc_a, vbc_a, g)
        c = c + 0.1 * advection_tendency(c, u, v, w, g, method="upwind1")
    assert float(jnp.min(c[1:-1, 1:-1, 1:-1])) >= -1e-6


# ── 2. periodic wind seam ────────────────────────────────────────────────────
def test_periodic_wind_seam_conserves_with_nonuniform_wind():
    g = _build_grid()
    rng = np.random.default_rng(1)
    c = _wrap_xy(jnp.asarray(rng.normal(size=g.shape).astype(np.float64)))
    # Spatially-varying normal winds; edge-padded halos break the seam match.
    u = _edge_pad_interior(rng.uniform(0.5, 2.0, size=g.interior_shape))
    v = _edge_pad_interior(rng.uniform(-1.5, 1.5, size=g.interior_shape))
    w = jnp.zeros(g.shape)
    wrapped = advection_tendency(c, u, v, w, g, method="upwind1", periodic=(True, True))
    unwrapped = advection_tendency(
        c, u, v, w, g, method="upwind1", periodic=(False, False)
    )
    # Fully periodic: matched seams => total interior tendency vanishes.
    np.testing.assert_allclose(
        float(jnp.sum(wrapped[1:-1, 1:-1, 1:-1])), 0.0, atol=1e-5
    )
    # Without wrapping the seam mismatch injects/removes mass.
    assert abs(float(jnp.sum(unwrapped[1:-1, 1:-1, 1:-1]))) > 1e-3


def test_periodic_wind_seam_wrap_only_flagged_axis():
    """Wrapping is applied per-axis: x-periodic wraps u, leaves v alone."""
    g = _build_grid()
    rng = np.random.default_rng(2)
    c = jnp.asarray(rng.normal(size=g.shape).astype(np.float64))
    u = _edge_pad_interior(rng.uniform(0.5, 2.0, size=g.interior_shape))
    v = jnp.zeros(g.shape)
    w = jnp.zeros(g.shape)
    only_x = advection_tendency(c, u, v, w, g, method="upwind1", periodic=(True, False))
    none = advection_tendency(c, u, v, w, g, method="upwind1", periodic=(False, False))
    # The x-wrap changes the west/east wall fluxes, so the result differs.
    assert not np.allclose(np.asarray(only_x), np.asarray(none))


# ── 3. periodic field-diffusivity seam ───────────────────────────────────────
def test_periodic_field_kappa_seam_conserves():
    g = _build_grid()
    rng = np.random.default_rng(3)
    c = _wrap_xy(jnp.asarray(rng.normal(size=g.shape).astype(np.float64)))
    kh = jnp.asarray(rng.uniform(0.3, 1.0, size=g.interior_shape).astype(np.float64))
    eddy = EddyDiffusivity(horizontal=kh, vertical=0.0)
    wrapped = diffusion_tendency(c, eddy, g, periodic=(True, True))
    unwrapped = diffusion_tendency(c, eddy, g, periodic=(False, False))
    np.testing.assert_allclose(
        float(jnp.sum(wrapped[1:-1, 1:-1, 1:-1])), 0.0, atol=1e-5
    )
    assert abs(float(jnp.sum(unwrapped[1:-1, 1:-1, 1:-1]))) > 1e-4


def test_scalar_kappa_conserves_periodic_without_wrapping():
    """A scalar K_h is uniform, so the seam already matches regardless."""
    g = _build_grid()
    rng = np.random.default_rng(4)
    c = _wrap_xy(jnp.asarray(rng.normal(size=g.shape).astype(np.float64)))
    eddy = EddyDiffusivity(horizontal=0.5, vertical=0.0)
    t = diffusion_tendency(c, eddy, g, periodic=(True, True))
    np.testing.assert_allclose(float(jnp.sum(t[1:-1, 1:-1, 1:-1])), 0.0, atol=1e-5)


# ── RHS integration: the dual BC fill + periodicity are wired through ─────────
def _zero_source(g):
    return GaussianSource(
        emission_fn=lambda t: jnp.asarray(0.0, dtype=jnp.float32),
        density=jnp.zeros(g.interior_shape, dtype=jnp.float32),
    )


def test_rhs_runs_and_is_interior_only():
    g = _build_grid()
    hbc, vbc = build_default_concentration_bc(
        bc_x=(("dirichlet", 0.0), "outflow"),
        bc_y="periodic",
        bc_z=("neumann", "neumann"),
    )
    rhs = EulerianDispersionRHS(
        plume_grid=g,
        wind_field=uniform_wind_field(g, u=2.0, v=0.0, w=0.0),
        eddy_diffusivity=EddyDiffusivity(horizontal=0.5, vertical=0.1),
        source=_zero_source(g),
        horizontal_bc=hbc,
        vertical_bc=vbc,
        advection_scheme="upwind1",
    )
    c = jnp.zeros(g.shape).at[2, 8, 8].set(1.0)
    out = rhs(jnp.asarray(0.0), c)
    assert jnp.all(jnp.isfinite(out))
    # Ghost ring of the tendency stays zero (interior-only integration).
    assert float(jnp.max(jnp.abs(out[0, :, :]))) == 0.0
    assert float(jnp.max(jnp.abs(out[:, 0, :]))) == 0.0
    assert float(jnp.max(jnp.abs(out[:, :, 0]))) == 0.0


def test_periodic_axes_rejects_half_periodic_axis():
    import pytest

    # west periodic, east Dirichlet — a half-periodic x-axis is ill-defined.
    hbc, _ = build_default_concentration_bc(
        bc_x=(("periodic", 0.0), ("dirichlet", 0.0)),
        bc_y=("neumann", "neumann"),
    )
    with pytest.raises(ValueError, match=r"only one x-face"):
        periodic_axes(hbc)


def test_periodic_axes_detection():
    hbc, _ = build_default_concentration_bc(
        bc_x="periodic", bc_y=("dirichlet", "outflow")
    )
    assert periodic_axes(hbc) == (True, False)
    hbc2, _ = build_default_concentration_bc(
        bc_x=("outflow", "outflow"), bc_y="periodic"
    )
    assert periodic_axes(hbc2) == (False, True)
