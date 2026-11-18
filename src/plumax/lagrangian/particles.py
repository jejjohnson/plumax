"""Markov-1 Langevin particle integrator (the Tier II forward dynamics).

Operational Lagrangian particle dispersion models (FLEXPART, STILT, HYSPLIT)
integrate a **Markov-1** process: a Langevin equation on the turbulent
*velocity* perturbation, with position advected by mean wind plus that
perturbation,

    dv = a(x, v, t) dt + b(x, t) dW,     dx = (u(x, t) + v) dt.

For Gaussian turbulence the well-mixed drift (Thomson 1987) is

    a_i = -v_i / τ_i  +  ½ (1 + v_i² / σ_i²) ∂σ_i²/∂z   (vertical inhomogeneity),
    b_i = sqrt(2 σ_i² / τ_i).

This module integrates the *linear* (OU) part with its **exact** transition —
``v ← e^{-Δt/τ} v + σ√(1 - e^{-2Δt/τ}) ξ`` — so a homogeneous run is exactly
well mixed (stationary velocity variance ``σ²`` with no time-step bias), and
adds the inhomogeneous vertical drift correction with an Euler step. The
correction's ``∂σ_w²/∂z`` is obtained by autodiff of the turbulence profile, so
it is zero for :class:`~plumax.lagrangian.turbulence.HomogeneousTurbulence` and
nonzero for :class:`~plumax.lagrangian.turbulence.HannaTurbulence` (the
``at``-exposing adapter over
:func:`~plumax.lagrangian.turbulence.hanna_profiles`) automatically.

Vertical motion reflects off the ground (``z = 0``) and, when a boundary-layer
height is supplied, off the PBL top — flipping both ``z`` and ``v_w``.
"""

from __future__ import annotations

from plumax.lagrangian._src.particles import (
    ParticleState,
    TurbulenceModel,
    integrate_particles,
    langevin_step,
    n_steps_for_horizon as n_steps_for_horizon,
    step_durations as step_durations,
    uniform_wind,
    wind_from_speed_direction,
)


__all__ = [
    "ParticleState",
    "TurbulenceModel",
    "integrate_particles",
    "langevin_step",
    "uniform_wind",
    "wind_from_speed_direction",
]
