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

Later build-order steps (Tier II/III transport + RTM, the ``Q(t)`` stochastic
process, trans-dimensional source count, coupled emulator, operational
predictor) are future work; the fusion harness and likelihood structure here
are designed to stay the same as those blocks are swapped in.
"""

from __future__ import annotations

from plumax.coupled import forward, fusion, instrument
from plumax.coupled.forward import (
    CoupledForward,
    PlumeSource,
    build_coupled_forward,
    column_response,
    predict_observation,
)
from plumax.coupled.fusion import FusionPosterior, default_prior, fuse_observations
from plumax.coupled.instrument import Instrument


__all__ = [
    "CoupledForward",
    "FusionPosterior",
    "Instrument",
    "PlumeSource",
    "build_coupled_forward",
    "column_response",
    "default_prior",
    "forward",
    "fuse_observations",
    "fusion",
    "instrument",
    "predict_observation",
]
