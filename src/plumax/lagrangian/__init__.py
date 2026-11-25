"""Tier II — Lagrangian particle dispersion (JAX).

A Markov-1 Langevin particle model: stochastic trajectories driven by mean wind
plus an Ornstein–Uhlenbeck turbulent-velocity process. The particles, the loop that
moves them, the gridded wind, the deposits and the backward run are tracerax's; the
closures, the plume and the inversions are here. The bridge between the
analytical Tier I (no real wind variability) and the Eulerian Tier III. See the
[Tier II design doc](https://github.com/jejjohnson/plumax/blob/main/docs/design/02_tier2_lagrangian.md).

This implements **Step 1 (the forward model)**, **Step 2 (model-based
inference)** and **Steps 3–4 (the footprint emulator and inference on it)**:

- ``turbulence`` — ``HomogeneousTurbulence`` (well-mixed-exact) and Hanna (1982)
  Monin–Obukhov-similarity profiles.
- ``concentration`` — forward steady concentration field via residence-time
  binning (``simulate_lagrangian``).
- ``footprint`` — backward source–receptor sensitivity (``compute_footprint``).
- ``wind_interp`` — the winds as ``tracerax.velocity`` velocities: the ``les_fvm``
  C-grid wind of a ``MetField`` or a ``PrescribedWindField``, sampled at each
  particle, and a uniform one from its speed and direction.
- ``met_ensemble`` — one footprint per met-ensemble member
  (``ensemble_footprints``) and the Gaussian inversion that adds their spread
  as a transport-error covariance (``ensemble_inversion``).
- ``inversion`` — Gaussian and sign-constrained lognormal Bayesian inversion of
  the source vector from observations, with a Matérn-3/2 spatial prior
  (``linear_gaussian_inversion`` / ``lognormal_inversion``): vardax
  ``OptimalInterpolation`` over ``footprint_operator`` (the ``assim`` extra).

- ``emulator`` — ``FootprintEmulator``, an MLP from met context and receptor to
  a footprint row, trained on particle-model footprints with pipekit-train
  (``footprint_dataset``, ``train_footprint_emulator``; the ``emulate`` extra)
  and calibrated
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
    turbulence,
    wind_interp,
)
from plumax.lagrangian.concentration import simulate_lagrangian
from plumax.lagrangian.emulator import (
    EmulatedFootprint,
    FootprintEmulator,
    FootprintSimulator,
    MetContextPrior,
    assert_emulator_calibrated,
    emulated_footprint,
    emulator_relative_error,
    footprint_dataset,
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
from plumax.lagrangian.turbulence import (
    HannaTurbulence,
    HomogeneousTurbulence,
    hanna_profiles,
)
from plumax.lagrangian.wind_interp import (
    gridded_wind_from_met,
    gridded_wind_from_prescribed,
    wind_from_speed_direction,
)


__all__ = [
    "EmulatedFootprint",
    "FootprintEmulator",
    "FootprintSimulator",
    "GaussianPosterior",
    "HannaTurbulence",
    "HomogeneousTurbulence",
    "LognormalPosterior",
    "MetContextPrior",
    "assert_emulator_calibrated",
    "compute_footprint",
    "concentration",
    "emulated_footprint",
    "emulator",
    "emulator_relative_error",
    "ensemble_footprints",
    "ensemble_inversion",
    "footprint",
    "footprint_dataset",
    "footprint_operator",
    "gridded_wind_from_met",
    "gridded_wind_from_prescribed",
    "hanna_profiles",
    "init_footprint_emulator",
    "inversion",
    "linear_gaussian_inversion",
    "lognormal_inversion",
    "matern32_covariance",
    "met_ensemble",
    "observation_covariance",
    "simulate_lagrangian",
    "train_footprint_emulator",
    "transport_covariance",
    "turbulence",
    "wind_from_speed_direction",
    "wind_interp",
]
