"""Tier IV — end-to-end coupled, multi-instrument source inversion.

Tier IV is **assembly + multi-instrument fusion, not new physics** (design
[Tier IV](https://github.com/jejjohnson/plumax/blob/main/docs/design/05_tier4_coupled.md)):
it composes a transport tier with the observation operator (averaging kernel /
RTM) and fuses several satellites' observations into one coherent posterior.

This v1 implements the design's build-order **step 1** — Tier I (Gaussian
plume) + averaging kernel + multi-instrument L2 fusion for a static emission
rate, with the per-instrument bias as a first-class state element:

- ``instrument`` — :class:`Instrument`: per-satellite receptor geometry,
  averaging kernel, and ``R = R_retr + R_repr + R_align`` error budget.
- ``forward`` — the coupled Tier I → column → AK forward, kept per-instrument
  at native resolution (``build_coupled_forward`` / ``CoupledForward``); linear
  in the emission rate ``Q``.
- ``fusion`` — the closed-form joint posterior over ``(Q, bias_inst)``
  (``fuse_observations``), exploiting that linearity (the design's
  linear-conditional-Gaussian limit).
- ``rtm`` — *additive* RTM-based observation operators that map the plume
  column enhancement → gas ``ΔVMR`` → band-integrated normalised radiance via
  the :mod:`plumax.radtran` Beer–Lambert + SRF stack, as the build-order step
  toward L1-radiance fusion. Two complementary, well-known models:
  the exact nonlinear ``RadianceObservationOperator`` / ``radiance_response``
  (``L_norm = exp(-Δτ)``), and its tangent-linear counterpart
  ``LinearisedRadianceOperator`` / ``radiance_response_linear`` (via
  ``linearise``), which is linear in ``Q`` and pure-JAX so it feeds gradient
  -based and closed-form inversion. The scalar-AK path above is unchanged.

Later build-order steps (Tier II/III transport + RTM, the ``Q(t)`` stochastic
process, trans-dimensional source count, coupled emulator, operational
predictor) are future work; the fusion harness and likelihood structure here
are designed to stay the same as those blocks are swapped in.
"""

from __future__ import annotations

from plumax.coupled import forward, fusion, instrument, rtm
from plumax.coupled.forward import (
    CoupledForward,
    PlumeSource,
    build_coupled_forward,
    column_response,
    predict_observation,
)
from plumax.coupled.fusion import FusionPosterior, default_prior, fuse_observations
from plumax.coupled.instrument import Instrument
from plumax.coupled.rtm import (
    LinearisedRadianceOperator,
    RadianceObservationOperator,
    column_mass_to_delta_vmr,
    linearise,
    radiance_response,
    radiance_response_linear,
)


__all__ = [
    "CoupledForward",
    "FusionPosterior",
    "Instrument",
    "LinearisedRadianceOperator",
    "PlumeSource",
    "RadianceObservationOperator",
    "build_coupled_forward",
    "column_mass_to_delta_vmr",
    "column_response",
    "default_prior",
    "forward",
    "fuse_observations",
    "fusion",
    "instrument",
    "linearise",
    "predict_observation",
    "radiance_response",
    "radiance_response_linear",
    "rtm",
]
