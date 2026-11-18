"""Bayesian inference for the Gaussian plume forward model (NumPyro).

Infers the emission rate Q from downwind concentration observations, and
optionally the atmospheric stability class as a discrete latent. The forward
model is :func:`plumax.gauss_plume.plume.plume_concentration`;
observations are modelled as noisy around the forward prediction plus a
half-normal background. The noise is a :class:`plumax.gauss_plume.likelihoods.Noise`
(Gaussian by default; Student-t or heteroscedastic on request).

Identifiability target (Tier I design): the inference is good enough when
three overpasses with varied wind direction give CV(Q) ≤ 50 % and a single
overpass gives CV(Q) ≤ 100 % under a tight met wind prior.

Public surface
--------------
- ``gaussian_plume_model`` : NumPyro probabilistic model
- ``infer_emission_rate``   : NUTS runner returning an
  :class:`EmissionRateSamples`, the dict of posterior arrays that also carries
  the Tier V per-event payload (``samples`` and ``prior_logpdf``)
- ``multi_source_plume_model`` : masked-K model, up to K_max sources with an
  on/off indicator each
- ``infer_multi_source``    : NUTS + Gibbs runner returning a
  :class:`~plumax.gauss_plume.multi_source.MultiSourcePosterior`
"""

from __future__ import annotations

from plumax.gauss_plume._src.inference import (
    EmissionRateSamples,
    gaussian_plume_model,
    infer_emission_rate,
    infer_multi_source,
    multi_source_plume_model,
)


__all__ = [
    "EmissionRateSamples",
    "gaussian_plume_model",
    "infer_emission_rate",
    "infer_multi_source",
    "multi_source_plume_model",
]
