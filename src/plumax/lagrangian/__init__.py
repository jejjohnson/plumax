"""Tier II — Lagrangian particle dispersion (JAX).

A Markov-1 Langevin particle model: stochastic trajectories driven by mean wind
plus an Ornstein–Uhlenbeck turbulent-velocity process. The bridge between the
analytical Tier I (no real wind variability) and the Eulerian Tier III. See the
[Tier II design doc](https://github.com/jejjohnson/plumax/blob/main/docs/design/02_tier2_lagrangian.md).

This v1 implements **Step 1 (the forward model)**:

- ``turbulence`` — ``HomogeneousTurbulence`` (well-mixed-exact) and Hanna (1982)
  Monin–Obukhov-similarity profiles.
- ``particles`` — the Markov-1 integrator with the Thomson (1987) well-mixed
  vertical drift correction and ground / PBL reflection.
- ``concentration`` — forward steady concentration field via residence-time
  binning (``simulate_lagrangian``).
- ``footprint`` — backward source–receptor sensitivity (``compute_footprint``).

Model-based inference, the footprint emulator and the amortized predictor
(Steps 2–5) are not yet implemented.
"""

from __future__ import annotations

from plumax.lagrangian import concentration, footprint, particles, turbulence
from plumax.lagrangian.concentration import bin_positions, simulate_lagrangian
from plumax.lagrangian.footprint import compute_footprint
from plumax.lagrangian.particles import (
    ParticleState,
    integrate_particles,
    langevin_step,
    uniform_wind,
    wind_from_speed_direction,
)
from plumax.lagrangian.turbulence import HomogeneousTurbulence, hanna_profiles


__all__ = [
    "HomogeneousTurbulence",
    "ParticleState",
    "bin_positions",
    "compute_footprint",
    "concentration",
    "footprint",
    "hanna_profiles",
    "integrate_particles",
    "langevin_step",
    "particles",
    "simulate_lagrangian",
    "turbulence",
    "uniform_wind",
    "wind_from_speed_direction",
]
