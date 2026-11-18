"""Methane source terms for the Eulerian dispersion solver.

The source enters the tracer equation as

    ∂C/∂t ⊇  S(x, t) = q(t) · ρ_s(x) / (Δx Δy Δz)

where ``q(t) [kg/s]`` is the instantaneous emission rate and ``ρ_s`` is a
unit-mass spatial profile (integrates to 1 over the interior volume).  We
use a Gaussian profile centred at the release point with a small
isotropic radius; this is the standard way to regularise a Dirac point
source on a finite grid, and the radius should be comparable to the
grid spacing.

:class:`SurfaceFluxSource` is the distributed alternative, a per-cell surface
flux map, which is the control of the spatial 4D-Var inversion.
"""

from __future__ import annotations

from plumax.les_fvm._src.source import (
    GaussianSource,
    SurfaceFluxSource,
    make_gaussian_source,
)


__all__ = [
    "GaussianSource",
    "SurfaceFluxSource",
    "make_gaussian_source",
]
