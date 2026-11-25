"""K-theory eddy diffusion for the Eulerian dispersion solver.

Under the K-theory closure the turbulent flux is modelled as a gradient
term

    u' C'  ≈  -K · ∇C,

giving a diffusion contribution ``∂_t C ⊇ ∇·(K ∇C)`` that closes the
mean-flow tracer equation.  Here ``K = diag(K_h, K_h, K_z)`` is an
anisotropic diagonal diffusivity — horizontal eddies are taken isotropic
in (x, y) while the vertical component is smaller in a stable boundary
layer.

Horizontal diffusion is assembled via :func:`gridax.diffuse`
(one 2-D Laplacian per z-level) and vertical diffusion by
:func:`gridax.diffuse_vertical`.

The :func:`pg_eddy_diffusivity` helper converts Pasquill-Gifford σ rates
into an effective ``(K_h, K_z)`` using the standard Taylor relation

    K = (1/2) d(σ²)/dt  ≈  σ² / (2 t),

evaluated at a user-supplied reference travel distance.  This is a
rough calibration and is exposed primarily as a convenience for
cross-checks against ``gauss_puff`` at the reference distance.

:func:`mo_eddy_diffusivity` follows the diagnosed boundary layer instead: a
K-profile ``K_z(z)`` from the :mod:`plumax.met.surface_layer` scales. And
:func:`smagorinsky_k_h` gives a field-valued ``K_h`` from the resolved horizontal
deformation of a spatially varying wind.
"""

from __future__ import annotations

from plumax.les_fvm._src.diffusion import (
    EddyDiffusivity,
    diffusion_tendency,
    make_eddy_diffusivity,
    mo_eddy_diffusivity,
    pg_eddy_diffusivity,
    smagorinsky_k_h,
)


__all__ = [
    "EddyDiffusivity",
    "diffusion_tendency",
    "make_eddy_diffusivity",
    "mo_eddy_diffusivity",
    "pg_eddy_diffusivity",
    "smagorinsky_k_h",
]
