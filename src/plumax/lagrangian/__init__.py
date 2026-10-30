"""Tier II — Lagrangian particle dispersion (JAX).

A Markov-1 Langevin particle model: stochastic trajectories driven by mean wind
plus an Ornstein–Uhlenbeck turbulent-velocity process. The bridge between the
analytical Tier I (no real wind variability) and the Eulerian Tier III. See the
[Tier II design doc](https://github.com/jejjohnson/plumax/blob/main/docs/design/02_tier2_lagrangian.md).

This implements **Step 1 (the forward model)**, **Step 2 (model-based
inference)** and **Steps 3–4 (the footprint emulator and inference on it)**:

- ``turbulence`` — ``HomogeneousTurbulence`` (well-mixed-exact) and Hanna (1982)
  Monin–Obukhov-similarity profiles.
- ``particles`` — the Markov-1 integrator with the Thomson (1987) well-mixed
  vertical drift correction and ground / PBL reflection.
- ``concentration`` — forward steady concentration field via residence-time
  binning (``simulate_lagrangian``).
- ``footprint`` — backward source–receptor sensitivity (``compute_footprint``).
- ``wind_interp`` — ``GriddedWind``, the ``les_fvm`` C-grid wind (from a
  ``MetField`` or a ``PrescribedWindField``) sampled at each particle; accepted
  wherever a wind callable is.
- ``met_ensemble`` — one footprint per met-ensemble member
  (``ensemble_footprints``) and the Gaussian inversion that adds their spread
  as a transport-error covariance (``ensemble_inversion``).
- ``inversion`` — Gaussian and sign-constrained lognormal Bayesian inversion of
  the source vector from observations, with a Matérn-3/2 spatial prior
  (``linear_gaussian_inversion`` / ``lognormal_inversion``): vardax
  ``OptimalInterpolation`` over ``footprint_operator`` (the ``assim`` extra).

- ``emulator`` — ``FootprintEmulator``, an MLP from met context and receptor to
  a footprint row, trained on particle-model footprints
  (``generate_training_set``, ``train_footprint_emulator``) and calibrated
  against them (``emulator_relative_error``, ``assert_emulator_calibrated``).
  ``emulated_footprint`` feeds the inversions unchanged (Step 4).

The amortized predictor (Step 5) is not yet implemented.
"""

from __future__ import annotations

from plumax.lagrangian import (
    concentration,
    emulator,
    footprint,
    inversion,
    met_ensemble,
    particles,
    turbulence,
    wind_interp,
)
from plumax.lagrangian.concentration import bin_positions, simulate_lagrangian
from plumax.lagrangian.emulator import (
    FootprintEmulator,
    FootprintTrainingSet,
    assert_emulator_calibrated,
    emulated_footprint,
    emulator_relative_error,
    generate_training_set,
    init_footprint_emulator,
    train_footprint_emulator,
)
from plumax.lagrangian.footprint import compute_footprint
from plumax.lagrangian.inversion import (
    GaussianPosterior,
    LognormalPosterior,
    footprint_operator,
    linear_gaussian_inversion,
    lognormal_inversion,
    matern32_covariance,
    observation_covariance,
)
from plumax.lagrangian.met_ensemble import (
    ensemble_footprints,
    ensemble_inversion,
    transport_covariance,
)
from plumax.lagrangian.particles import (
    ParticleState,
    integrate_particles,
    langevin_step,
    n_steps_for_horizon,
    step_durations,
    uniform_wind,
    wind_from_speed_direction,
)
from plumax.lagrangian.turbulence import (
    HannaTurbulence,
    HomogeneousTurbulence,
    hanna_profiles,
)
from plumax.lagrangian.wind_interp import (
    GriddedWind,
    gridded_wind_from_met,
    gridded_wind_from_prescribed,
)


__all__ = [
    "FootprintEmulator",
    "FootprintTrainingSet",
    "GaussianPosterior",
    "GriddedWind",
    "HannaTurbulence",
    "HomogeneousTurbulence",
    "LognormalPosterior",
    "ParticleState",
    "assert_emulator_calibrated",
    "bin_positions",
    "compute_footprint",
    "concentration",
    "emulated_footprint",
    "emulator",
    "emulator_relative_error",
    "ensemble_footprints",
    "ensemble_inversion",
    "footprint",
    "footprint_operator",
    "generate_training_set",
    "gridded_wind_from_met",
    "gridded_wind_from_prescribed",
    "hanna_profiles",
    "init_footprint_emulator",
    "integrate_particles",
    "inversion",
    "langevin_step",
    "linear_gaussian_inversion",
    "lognormal_inversion",
    "matern32_covariance",
    "met_ensemble",
    "n_steps_for_horizon",
    "observation_covariance",
    "particles",
    "simulate_lagrangian",
    "step_durations",
    "train_footprint_emulator",
    "transport_covariance",
    "turbulence",
    "uniform_wind",
    "wind_from_speed_direction",
    "wind_interp",
]
