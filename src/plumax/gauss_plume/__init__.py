"""Steady-state Gaussian plume model (JAX + NumPyro).

Submodules:
  - dispersion    — Briggs σ_y(x), σ_z(x) coefficients (A-F stability classes)
  - plume         — coordinate rotation + JIT forward model + xarray wrapper
  - plume_rise    — Briggs plume rise Δh(x) and the effective release height
  - inference     — NumPyro Bayesian inference for Q (and optional stability)
  - observation   — column + averaging-kernel observation shared by every tier
  - likelihoods   — Gaussian, Student-t and heteroscedastic observation noise
  - multi_source  — masked-K source slots and their posterior

The ``inference`` submodule is imported lazily on first attribute access so
that ``from plumax import gauss_plume`` does not pull in NumPyro
for consumers of only the forward model.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from plumax.gauss_plume import (
    dispersion,
    likelihoods,
    multi_source,
    observation,
    plume,
    plume_rise,
)
from plumax.gauss_plume.dispersion import (
    BRIGGS_DISPERSION_PARAMS,
    STABILITY_CLASSES,
    calculate_briggs_dispersion,
    get_dispersion_params,
)
from plumax.gauss_plume.likelihoods import (
    GaussianNoise,
    HeteroscedasticNoise,
    Noise,
    StudentTNoise,
)
from plumax.gauss_plume.multi_source import (
    MaskedSources,
    MultiSourcePosterior,
    SourceSummary,
)
from plumax.gauss_plume.observation import ColumnObservation
from plumax.gauss_plume.plume import (
    MIN_WIND_SPEED,
    plume_concentration,
    plume_concentration_vmap,
    rotate_to_wind_frame,
    simulate_plume,
)
from plumax.gauss_plume.plume_rise import (
    BriggsPlumeRise,
    briggs_plume_rise,
    effective_height,
)


_LAZY_INFERENCE_SYMBOLS = {
    "gaussian_plume_model",
    "infer_emission_rate",
    "infer_multi_source",
    "multi_source_plume_model",
}


def __getattr__(name: str):  # PEP 562 lazy module-level attribute
    if name == "inference" or name in _LAZY_INFERENCE_SYMBOLS:
        module = importlib.import_module("plumax.gauss_plume.inference")
        if name == "inference":
            return module
        return getattr(module, name)
    raise AttributeError(f"module 'plumax.gauss_plume' has no attribute {name!r}")


if TYPE_CHECKING:  # make the lazy names visible to type checkers / IDEs
    from plumax.gauss_plume import inference
    from plumax.gauss_plume.inference import (
        gaussian_plume_model,
        infer_emission_rate,
        infer_multi_source,
        multi_source_plume_model,
    )


__all__ = [
    "BRIGGS_DISPERSION_PARAMS",
    "MIN_WIND_SPEED",
    "STABILITY_CLASSES",
    "BriggsPlumeRise",
    "ColumnObservation",
    "GaussianNoise",
    "HeteroscedasticNoise",
    "MaskedSources",
    "MultiSourcePosterior",
    "Noise",
    "SourceSummary",
    "StudentTNoise",
    "briggs_plume_rise",
    "calculate_briggs_dispersion",
    "dispersion",
    "effective_height",
    "gaussian_plume_model",
    "get_dispersion_params",
    "infer_emission_rate",
    "infer_multi_source",
    "inference",
    "likelihoods",
    "multi_source",
    "multi_source_plume_model",
    "observation",
    "plume",
    "plume_concentration",
    "plume_concentration_vmap",
    "plume_rise",
    "rotate_to_wind_frame",
    "simulate_plume",
]
