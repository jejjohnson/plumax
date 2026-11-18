"""RTM-based observation operator for the Tier IV coupled forward.

The Tier IV v1 forward (:mod:`plumax.coupled.forward`) maps a plume column
enhancement to an observation through a **scalar averaging kernel**
(``y_inst = A_inst · col_z(...)``). The design's build-order step toward
L1-radiance fusion ([Tier IV §per-instrument-forward](https://github.com/jejjohnson/plumax/blob/main/docs/design/05_tier4_coupled.md#tier4-per-instrument-forward),
[RTM stack](https://github.com/jejjohnson/plumax/blob/main/docs/design/04_rtm_stack.md))
replaces that scalar AK with a *physically real* radiative-transfer operator:
the same column enhancement is mapped to a gas column ``ΔVMR`` and then, via the
Beer–Lambert + spectral-response-function stack in :mod:`radtrax.slab`, to the
band-integrated normalised radiance each instrument would actually observe.

This module provides :class:`RadianceObservationOperator` — the RTM analogue of
:class:`plumax.adapters.RadtranObservationOperator`, but mapping a *per-receptor
column-enhancement field* (what Tier I gives) to per-receptor band-integrated
radiances, and :func:`radiance_response`, the drop-in parallel to
:func:`plumax.coupled.forward.column_response` for the radiance forward. The
existing scalar-AK path is untouched — the change is purely additive.

Differentiability
-----------------
The :mod:`radtrax.slab` Beer–Lambert forward and the
:class:`~radtrax.instrument.SpectralResponseFunction` are **NumPy-side** (the
LUT interpolation goes through ``xarray``), so :class:`RadianceObservationOperator`
is *not* JAX-traceable end-to-end: it consumes concrete arrays and returns
NumPy arrays, exactly like :class:`plumax.adapters.RadtranObservationOperator`.
The analytic Jacobian ``d(L_norm)/d(ΔVMR)`` returned by the radtrax slab forward is
the tangent-linear used in :meth:`RadianceObservationOperator.linearize`. The
*Tier I plume* column response (:func:`plumax.coupled.forward.column_response`)
that feeds it remains differentiable; the column→ΔVMR conversion
(:func:`column_mass_to_delta_vmr`) is a pure JAX function and is differentiable.

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
    AVOGADRO_PER_MOL,
    CH4_MOLAR_MASS_KG_PER_MOL,
    CM2_PER_M2,
    LayeredRadianceOperator,
    LinearisedRadianceOperator,
    RadianceObservationOperator,
    column_mass_to_delta_vmr,
    linearise,
    radiance_response,
    radiance_response_linear,
)


__all__ = [
    "AVOGADRO_PER_MOL",
    "CH4_MOLAR_MASS_KG_PER_MOL",
    "CM2_PER_M2",
    "LayeredRadianceOperator",
    "LinearisedRadianceOperator",
    "RadianceObservationOperator",
    "column_mass_to_delta_vmr",
    "linearise",
    "radiance_response",
    "radiance_response_linear",
]
