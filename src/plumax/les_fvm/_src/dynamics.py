"""diffrax-compatible RHS for the Eulerian dispersion ODE system.

The full tendency ``∂_t C = RHS(t, C)`` is assembled by composing the
advection, diffusion, source, and boundary-application modules from this
sub-package.  Wrapping the RHS in an :class:`equinox.Module` keeps it a
single pytree leaf that diffrax can JIT as ``term = ODETerm(rhs)``.
"""

from __future__ import annotations

import equinox as eqx
from gridax import embed, interior
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from plumax.les_fvm.advection import advection_tendency
from plumax.les_fvm.boundary import (
    HorizontalBC,
    VerticalBC,
    apply_boundary_conditions,
    periodic_axes,
)
from plumax.les_fvm.diffusion import EddyDiffusivity, diffusion_tendency
from plumax.les_fvm.grid import PlumeGrid3D
from plumax.les_fvm.source import GaussianSource, SurfaceFluxSource
from plumax.les_fvm.wind import PrescribedWindField


class EulerianDispersionRHS(eqx.Module):
    """Full tracer-transport tendency for diffrax.

    Parameters
    ----------
    plume_grid : PlumeGrid3D
    wind_field : PrescribedWindField
        Time-queryable prescribed wind.
    eddy_diffusivity : EddyDiffusivity
        ``(K_h, K_z)`` eddy diffusivity.
    source : GaussianSource or SurfaceFluxSource
        Methane source term.
    horizontal_bc : HorizontalBC
    vertical_bc : VerticalBC
    advection_scheme : str, default ``"weno5"``
        Horizontal reconstruction scheme for
        :func:`advection.advection_tendency`.
    """

    plume_grid: PlumeGrid3D
    wind_field: PrescribedWindField
    eddy_diffusivity: EddyDiffusivity
    source: GaussianSource | SurfaceFluxSource
    horizontal_bc: HorizontalBC
    vertical_bc: VerticalBC
    advection_scheme: str = eqx.field(static=True, default="weno5")

    def __call__(
        self,
        t: Float[ArrayLike, ""],
        concentration: Float[Array, "Nz Ny Nx"],
        args: object = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Return ``dC/dt`` at time ``t`` for tracer field ``concentration``."""
        del args
        # Enforce BCs before reading neighbours in the advection/diffusion
        # stencils so ghost cells reflect the current physical BC state. The
        # horizontal operators run in gridax ``wall="open"`` mode, so these
        # ghost values drive the lateral wall fluxes; the vertical operators
        # read the top/bottom ghost slices directly.
        #
        # Advection and diffusion need *different* Dirichlet ghosts: advection
        # upwinds the ghost as an exterior cell value (ghost = value), while
        # diffusion needs the face reflection (2*value - interior) so the
        # centred gradient sees the boundary value. Outflow / periodic /
        # Neumann ghosts are identical, so the two fills differ only at
        # Dirichlet faces.
        diffusion_concentration = apply_boundary_conditions(
            concentration,
            horizontal_bc=self.horizontal_bc,
            vertical_bc=self.vertical_bc,
            plume_grid=self.plume_grid,
        )
        advection_concentration = apply_boundary_conditions(
            concentration,
            horizontal_bc=self.horizontal_bc.advection_variant(),
            vertical_bc=self.vertical_bc.advection_variant(),
            plume_grid=self.plume_grid,
        )
        wind_east, wind_north, wind_up = self.wind_field(t)
        # Wrap the normal wind / field-diffusivity halos on periodic axes so
        # the paired seam faces stay conservative under a spatially-varying
        # wind or K_h (see periodic_axes / the tendency `periodic=` argument).
        periodic = periodic_axes(self.horizontal_bc)
        adv = advection_tendency(
            advection_concentration,
            wind_east,
            wind_north,
            wind_up,
            self.plume_grid,
            method=self.advection_scheme,
            periodic=periodic,
        )
        diff = diffusion_tendency(
            diffusion_concentration,
            self.eddy_diffusivity,
            self.plume_grid,
            periodic=periodic,
        )
        src = self.source(t)
        # Keep ghost-cell entries of the tendency zero: the integrator writes
        # only to the interior, and the BC pass above already updated the ghost
        # ring of the state.
        grid = self.plume_grid.grid
        return embed(interior(adv + diff + src, grid), grid)
