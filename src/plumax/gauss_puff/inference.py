"""Bayesian inference for the Gaussian puff forward model (NumPyro).

The forward model evaluates puff concentrations at observation coordinates
for a given emission rate ``Q`` (constant or time-varying) under a specified
wind schedule. This module provides:

* :func:`gaussian_puff_model` — a NumPyro model over ``Q``, a half-normal
  background, and observation noise (a
  :class:`plumax.gauss_plume.likelihoods.Noise`, Gaussian by default).
* :func:`infer_emission_rate` — NUTS runner for a **constant** Q, wrapping
  the model with input validation and posterior unpacking.
* :func:`infer_emission_timeseries` — NUTS runner for a **time-varying**
  ``Q_i`` under a random-walk prior, for state-estimation workflows.

The log-space prior parameters are derived from real-scale ``(mean, std)``
via :meth:`plumax._src.scalar_emission.LognormalEmission.from_moments`,
shared with the plume inference helpers.
"""

from __future__ import annotations

from plumax.gauss_puff._src.inference import (
    gaussian_puff_model,
    gaussian_puff_rw_model,
    infer_emission_rate,
    infer_emission_timeseries,
)


__all__ = [
    "gaussian_puff_model",
    "gaussian_puff_rw_model",
    "infer_emission_rate",
    "infer_emission_timeseries",
]
