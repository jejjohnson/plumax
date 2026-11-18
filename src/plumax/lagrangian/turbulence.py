"""Turbulence parameterisations for the Lagrangian particle model.

The Markov-1 Langevin model (see :mod:`plumax.lagrangian.particles`) needs two
fields at every particle location: the turbulent velocity standard deviations
``σ = (σ_u, σ_v, σ_w)`` [m/s] and the Lagrangian decorrelation timescales
``τ_L = (τ_u, τ_v, τ_w)`` [s]. Together they define the Ornstein–Uhlenbeck
velocity process whose stationary variance is ``σ²`` and whose autocorrelation
decays as ``exp(-Δt / τ_L)``.

Two parameterisations are provided:

- :class:`HomogeneousTurbulence` — constant ``σ`` and ``τ_L`` everywhere. This
  is the well-mixed-exact baseline used by the validation suite (a constant-σ
  OU process is exactly well mixed, so no density-gradient drift correction is
  needed) and a reasonable first approximation inside a well-mixed boundary
  layer.
- :func:`hanna_profiles` — Monin–Obukhov-similarity profiles after Hanna
  (1982), giving height-dependent ``σ(z)`` and ``τ_L(z)`` from surface-layer
  scales ``(u_*, L, w_*, h)``. This is the operational parameterisation; height
  dependence makes the turbulence inhomogeneous, which the integrator handles
  with the well-mixed drift correction.

References:
    Hanna, S. R. (1982). Applications in air pollution modeling. In
    *Atmospheric Turbulence and Air Pollution Modelling* (pp. 275–310).
"""

from __future__ import annotations

from plumax.lagrangian._src.turbulence import (
    DEFAULT_CORIOLIS_PARAMETER as DEFAULT_CORIOLIS_PARAMETER,
    HannaTurbulence,
    HomogeneousTurbulence,
    hanna_profiles,
)


__all__ = [
    "HannaTurbulence",
    "HomogeneousTurbulence",
    "hanna_profiles",
]
