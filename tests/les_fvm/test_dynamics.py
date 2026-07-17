"""Tests that horizontal lateral BCs actually drive transport (issues #25/#26).

Since finitevolX gained the ``wall="open"`` lateral-boundary mode, the
``les_fvm`` horizontal advection / diffusion operators read the BC-updated
ghost ring at the domain walls.  These integration tests exercise the real
code path used by :class:`~plumax.les_fvm.dynamics.EulerianDispersionRHS`:
fill the ghost ring with :func:`apply_boundary_conditions`, then step the
advective tendency forward.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np

from plumax.les_fvm.advection import advection_tendency
from plumax.les_fvm.boundary import (
    apply_boundary_conditions,
    build_default_concentration_bc,
)
from plumax.les_fvm.grid import make_grid


def _build_grid():
    return make_grid(
        domain_x=(0.0, 160.0, 16),
        domain_y=(0.0, 160.0, 16),
        domain_z=(0.0, 40.0, 4),
    )


def _interior_mass(c):
    return float(jnp.sum(c[1:-1, 1:-1, 1:-1]))


def test_periodic_y_wraps_mass():
    """A blob advecting in +y past the north wall re-enters at the south,
    conserving interior mass (was frozen/lost before wall='open')."""
    g = _build_grid()
    hbc, vbc = build_default_concentration_bc(
        bc_x="periodic",
        bc_y="periodic",
        bc_z=("neumann", "neumann"),
    )
    # Blob near the north interior edge.
    c = jnp.zeros(g.shape).at[2, -2, 8].set(1.0)
    u = jnp.zeros(g.shape)
    v = jnp.ones(g.shape) * 2.0  # +y wind
    w = jnp.zeros(g.shape)
    dt = 0.1
    mass0 = _interior_mass(c)
    for _ in range(120):
        c = apply_boundary_conditions(c, hbc, vbc, g)
        c = c + dt * advection_tendency(c, u, v, w, g, method="upwind1")
    # Mass conserved (blob wrapped, not lost off the north edge).
    np.testing.assert_allclose(_interior_mass(c), mass0, rtol=1e-4)
    # Some mass has crossed into the south half of the domain (it wrapped).
    south_half = float(jnp.sum(c[2, 1 : g.shape[1] // 2, :]))
    assert south_half > 1e-2


def test_outflow_mass_budget():
    """Outflow-east + zero-Dirichlet elsewhere: mass leaves ONLY through the
    open east boundary (monotone loss), while a fully-periodic control run
    conserves mass exactly."""
    g = _build_grid()
    # +x wind strong enough to advect the blob across the domain and out the
    # east wall within the step budget (CFL = u·dt/dx = 4·0.2/10 = 0.08/step;
    # ~200 steps ≈ 16 cells >> the ~14-cell interior width).
    u = jnp.ones(g.shape) * 4.0
    v = jnp.zeros(g.shape)
    w = jnp.zeros(g.shape)
    dt = 0.2
    n_steps = 200

    # --- outflow east, closed (zero-Dirichlet) elsewhere ---
    hbc, vbc = build_default_concentration_bc(
        bc_x=(("dirichlet", 0.0), "outflow"),
        bc_y=(("dirichlet", 0.0), ("dirichlet", 0.0)),
        bc_z=("neumann", "neumann"),
    )
    c = jnp.zeros(g.shape).at[2, 8, 3].set(1.0)  # start near the west side
    masses = []
    for _ in range(n_steps):
        c = apply_boundary_conditions(c, hbc, vbc, g)
        c = c + dt * advection_tendency(c, u, v, w, g, method="upwind1")
        masses.append(_interior_mass(c))
    # Never gains mass, and most of it has exited through the east boundary.
    assert all(b <= a + 1e-6 for a, b in itertools.pairwise(masses))
    assert masses[-1] < 0.5 * masses[0]

    # --- fully-periodic control: identical wind, mass conserved ---
    hbc_p, vbc_p = build_default_concentration_bc(
        bc_x="periodic", bc_y="periodic", bc_z=("neumann", "neumann")
    )
    cp = jnp.zeros(g.shape).at[2, 8, 3].set(1.0)
    mass0 = _interior_mass(cp)
    for _ in range(n_steps):
        cp = apply_boundary_conditions(cp, hbc_p, vbc_p, g)
        cp = cp + dt * advection_tendency(cp, u, v, w, g, method="upwind1")
    np.testing.assert_allclose(_interior_mass(cp), mass0, rtol=1e-4)


def test_wall_ring_is_no_longer_frozen():
    """The wall-adjacent interior ring now receives a nonzero advective
    tendency (regression for the frozen-ring bug in #25)."""
    g = _build_grid()
    hbc, vbc = build_default_concentration_bc(
        bc_x="periodic", bc_y="periodic", bc_z=("neumann", "neumann")
    )
    # Uniform gradient in x so the wall-adjacent ring has a real flux.
    x = jnp.arange(g.shape[2], dtype=jnp.float32)
    c = jnp.broadcast_to(x[None, None, :], g.shape)
    c = apply_boundary_conditions(c, hbc, vbc, g)
    u = jnp.ones(g.shape) * 1.5
    v = jnp.zeros(g.shape)
    w = jnp.zeros(g.shape)
    tend = advection_tendency(c, u, v, w, g, method="upwind1")
    # Wall-adjacent interior columns i=1 and i=Nx-2 are nonzero (not frozen).
    assert float(jnp.max(jnp.abs(tend[1:-1, 1:-1, 1]))) > 1e-6
    assert float(jnp.max(jnp.abs(tend[1:-1, 1:-1, -2]))) > 1e-6
