"""Tier V.B — probability-of-detection (POD) models as Equinox modules.

Here "POD" means **probability of detection** — the chance that a satellite
detects a methane plume of a given size under given conditions. (It is
unrelated to the proper orthogonal decomposition used for reduced-order
modelling elsewhere in the docs.)

A family of GLM-style POD models ``P_d(x) ∈ [0, 1]`` for methane plume
detectability from satellite remote sensing, packaged as ``eqx.Module``.
Each module bundles:

* covariate transformation + link function in ``__call__``;
* ``sample_priors(cls)`` — a NumPyro prior factory for MCMC.

The progression mirrors :mod:`plumax.population.intensity`: simple →
complex, each model relaxing an assumption of the last.

Detection is not a deterministic threshold. It is a probabilistic event
governed by the interplay of source strength, atmospheric dilution,
instrument optics and radiometric conditions, modelled as a Bernoulli GLM
``Y ~ Bernoulli(P_d(x))`` where ``Y ∈ {0, 1}`` (detected / not detected)
and ``P_d`` is evaluated at covariates ``x``.

Conventions:

* Flux ``Q`` in [kg hr⁻¹]; wind speed ``U`` in [m s⁻¹]; pixel size ``p``
  in [m]; albedo ``A`` in [–], ``∈ [0, 1]``.
* Solar zenith angle ``θ`` in [°]; models use ``cos(θ) ∈ [0.26, 1.0]``.
* Spectral bands ``n_bands`` [dimensionless count]; noise ``σ_noise`` in
  [ppb] or instrument-specific units.
* All modules are immutable pytrees, jit / vmap / grad compatible.

NumPyro is only imported inside ``sample_priors``, so evaluating a POD
curve never pulls it in.
"""

from __future__ import annotations

from plumax.population._src.pod import (
    POD_REGISTRY,
    TOD_BINS,
    AdditiveMultiCovariatePOD,
    CloglogPOD,
    ConcentrationProxyPOD,
    FullVaryingCoefficientPOD,
    InstrumentPOD,
    LogisticPOD,
    LogLogisticPOD,
    ProbitPOD,
    SNRBasedPOD,
    SpectralAwarePOD,
    VaryingCoefficientPOD,
    calibration_from_draw,
    calibration_prior,
    cos_zenith,
    load_pod_calibrations,
    log_concentration_proxy,
    log_flux,
    log_flux_wind_ratio,
    log_pod_union,
    pod_union,
    snr_proxy,
    tod_bin_index,
)


__all__ = [
    "POD_REGISTRY",
    "TOD_BINS",
    "AdditiveMultiCovariatePOD",
    "CloglogPOD",
    "ConcentrationProxyPOD",
    "FullVaryingCoefficientPOD",
    "InstrumentPOD",
    "LogLogisticPOD",
    "LogisticPOD",
    "ProbitPOD",
    "SNRBasedPOD",
    "SpectralAwarePOD",
    "VaryingCoefficientPOD",
    "calibration_from_draw",
    "calibration_prior",
    "cos_zenith",
    "load_pod_calibrations",
    "log_concentration_proxy",
    "log_flux",
    "log_flux_wind_ratio",
    "log_pod_union",
    "pod_union",
    "snr_proxy",
    "tod_bin_index",
]
