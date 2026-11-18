"""Tier V.B — methane source intensity functions as Equinox modules.

Physically motivated intensity functions ``λ(t)`` [events day⁻¹] for CH₄
emission sources observed by satellite remote sensing, packaged as
``equinox.Module`` dataclasses. Each module bundles:

* deterministic parameters as frozen fields;
* ``__call__(t)`` — ``λ(t)`` evaluation, compatible with jit / vmap / grad;
* ``sample_priors(cls)`` — a NumPyro prior factory returning a configured
  instance for MCMC inference.

Conventions:

* Time ``t`` is always in **days**; fractional days encode sub-daily
  resolution.
* Intensity ``λ(t)`` is always in **events day⁻¹**.
* All modules are immutable pytrees (equinox).

NumPyro is only imported inside ``sample_priors``, so evaluating an
intensity never pulls it in.
"""

from __future__ import annotations

from plumax.population._src.intensity import (
    INTENSITY_REGISTRY,
    CoalMineVentilationIntensity,
    ConstantIntensity,
    DiurnalSeasonalIntensity,
    DiurnalSinusoidalIntensity,
    LandfillIntensity,
    LivestockFeedlotIntensity,
    OffshorePlatformIntensity,
    OperationalScheduleIntensity,
    PeriodicBatchIntensity,
    SeasonallyModulatedDiurnalIntensity,
    SeasonalSinusoidalIntensity,
    WeibullRenewalIntensity,
    WetlandPermafrostIntensity,
)


__all__ = [
    "INTENSITY_REGISTRY",
    "CoalMineVentilationIntensity",
    "ConstantIntensity",
    "DiurnalSeasonalIntensity",
    "DiurnalSinusoidalIntensity",
    "LandfillIntensity",
    "LivestockFeedlotIntensity",
    "OffshorePlatformIntensity",
    "OperationalScheduleIntensity",
    "PeriodicBatchIntensity",
    "SeasonalSinusoidalIntensity",
    "SeasonallyModulatedDiurnalIntensity",
    "WeibullRenewalIntensity",
    "WetlandPermafrostIntensity",
]
