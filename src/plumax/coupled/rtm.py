"""RTM-based observation operator for the Tier IV coupled forward.

The Tier IV v1 forward (:mod:`plumax.coupled.forward`) maps a plume column
enhancement to an observation through a **scalar averaging kernel**
(``y_inst = A_inst · col_z(...)``). The design's build-order step toward
L1-radiance fusion ([Tier IV §per-instrument-forward](https://github.com/jejjohnson/plumax/blob/main/docs/design/05_tier4_coupled.md#tier4-per-instrument-forward),
[RTM stack](https://github.com/jejjohnson/plumax/blob/main/docs/design/04_rtm_stack.md))
replaces that scalar AK with a *physically real* radiative-transfer operator:
the same column enhancement is mapped to a gas column ``ΔVMR`` and then, via
radtrax's single-slab Beer–Lambert model and its band instrument, to the
band-integrated normalised radiance each instrument would actually observe.

This module provides :class:`RadianceObservationOperator`, which maps a
*per-receptor column-enhancement field* (what Tier I gives) to per-receptor
band-integrated radiances (:class:`plumax.adapters.RadtranObservationOperator`
is its one-receptor case in ``ΔVMR``), and :func:`radiance_response`, the
drop-in parallel to :func:`plumax.coupled.forward.column_response` for the
radiance forward. The existing scalar-AK path is untouched.

Differentiability
-----------------
The table lookup ``a(ν) = σ(ν, T, p) n L AMF`` goes through ``xarray`` and
happens once, when the operator is built
(:meth:`radtrax.retrieval.var3d.RadianceObservationModel.from_lut`). Every
step after it is JAX: the column→ΔVMR conversion
(:func:`column_mass_to_delta_vmr`), ``exp(−a ΔVMR)`` and the band
integration, so :class:`RadianceObservationOperator`,
:class:`LayeredRadianceOperator` and :class:`LinearisedRadianceOperator` all
run under ``jax.jit``, ``jax.vmap`` and ``jax.grad``.

Covariance / gaussx
-------------------
This operator deliberately returns a *band-integrated mean radiance* response
and does **not** roll its own spectral covariance / whitening: when a spectral
error covariance over bands is needed (matched-filter target construction,
whitened radiance residuals), route it through the gaussx-backed
background estimators in :mod:`radtrax.retrieval.matched_filter.background` and
:func:`radtrax.retrieval.matched_filter.core.apply_pixel` (``gx.LowRankUpdate`` + Woodbury
``gx.solve``) rather than a hand-rolled dense inverse.
"""

from __future__ import annotations

from plumax.coupled._src.rtm import (
    CH4_MOLAR_MASS_KG_PER_MOL,
    LayeredRadianceOperator,
    LinearisedRadianceOperator,
    RadianceObservationOperator,
    column_mass_to_delta_vmr,
    linearise,
    radiance_response,
    radiance_response_linear,
)


__all__ = [
    "CH4_MOLAR_MASS_KG_PER_MOL",
    "LayeredRadianceOperator",
    "LinearisedRadianceOperator",
    "RadianceObservationOperator",
    "column_mass_to_delta_vmr",
    "linearise",
    "radiance_response",
    "radiance_response_linear",
]
