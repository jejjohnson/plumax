"""Multi-instrument fusion: closed-form joint inversion of ``(Q, bias_inst)``.

This is the Tier IV v1 inference (design build-order step 1 + the
[linear-conditional-Gaussian limit](https://github.com/jejjohnson/plumax/blob/main/docs/design/05_tier4_coupled.md#tier4-validation)).
Conditioned on the source location, wind and stability, the Gaussian-plume
forward is **linear in the emission rate ``Q``**, and the per-instrument bias
enters additively, so the joint state

    θ = (Q, bias₁, …, bias_M)

has a closed-form Gaussian posterior given a Gaussian prior and Gaussian
observation errors. Stacking every instrument's receptors, the forward is the
affine map

    y_stacked = G θ + c_bg,     G[:, 0] = responses,  G[:, 1+i] = 1 on instrument i,

and the BLUE / Kalman update gives ``θ*`` and its covariance ``P*``. Per the
design, the per-instrument **bias is a first-class state element** (documented
inter-instrument biases are O(±10 ppb); ignoring them double-counts agreement),
which is why it is solved jointly with ``Q`` here rather than pre-subtracted.

plumax builds the design matrix, the background, the prior and the instrument
error budget; the solve is a ``vardax.OptimalInterpolation`` analysis in its
state-space (information) form (the ``assim`` extra, ``pip install
'plumax[assim]'``). The estimator stays pure JAX and differentiable; it reuses
:func:`plumax.lagrangian.inversion.observation_covariance` semantics implicitly
via the per-instrument :meth:`Instrument.observation_variance`.
"""

from __future__ import annotations

from plumax.coupled._src.fusion import (
    FusionPosterior,
    default_prior,
    fuse_observations,
)


__all__ = [
    "FusionPosterior",
    "default_prior",
    "fuse_observations",
]
