"""Tier V — source population & forecasting.

Assembly of per-event Tier II-IV posteriors into population-scale
products:

* :mod:`~plumax.population.catalog` — the cross-tier posterior catalog
  (the load-bearing interface).
* :mod:`~plumax.population.size_distribution` — V.A instantaneous
  emission size distribution (hierarchical lognormal fit).
* :mod:`~plumax.population.point_process` — V.B spatio-temporal event-rate
  models (homogeneous Gamma-Poisson + log-linear inhomogeneous intensity).
* :mod:`~plumax.population.intensity` — V.B registry of 13 physically
  motivated source intensities ``λ(t)`` (equinox modules).
* :mod:`~plumax.population.pod` — V.B registry of 10 probability-of-detection
  models ``P_d(·)`` (equinox modules). "POD" here is *probability of
  detection*, not proper orthogonal decomposition.
* :mod:`~plumax.population.paradox` — V.D missing-mass paradox Monte Carlo.
* :mod:`~plumax.population.fitting` — POD-modified power-law NUTS fitter.

Importing this package is cheap: the NumPyro-dependent fit functions and
result types are bound lazily (PEP 562), so ``import plumax.population``
never pulls in NumPyro.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from plumax.population.catalog import (
    EmissionCatalog,
    EmissionEvent,
    PerEventPosterior,
    event_from_posterior,
)
from plumax.population.intensity import (
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
from plumax.population.paradox import (
    FacilityConfig,
    ParadoxResult,
    build_canonical_scenarios,
    compute_E_Pd,
    logistic_pod,
    lognormal_pdf,
    run_scenario_grid,
    simulate_paradox,
)
from plumax.population.pod import (
    POD_REGISTRY,
    AdditiveMultiCovariatePOD,
    CloglogPOD,
    ConcentrationProxyPOD,
    FullVaryingCoefficientPOD,
    LogisticPOD,
    LogLogisticPOD,
    ProbitPOD,
    SNRBasedPOD,
    SpectralAwarePOD,
    VaryingCoefficientPOD,
)
from plumax.population.point_process import (
    PoissonRatePosterior,
    fit_poisson_rate,
)


if TYPE_CHECKING:
    from plumax.population.fitting import (
        X_MAX_DEFAULT,
        X_MIN_DEFAULT,
        lognorm_cdf,
        pod_powerlaw_model,
        power_law,
        run_mcmc,
    )
    from plumax.population.point_process import (
        InhomogeneousIntensityPosterior,
        fit_inhomogeneous_intensity,
    )
    from plumax.population.size_distribution import (
        SizeDistributionPosterior,
        fit_lognormal_size_distribution,
    )

__all__ = [
    "INTENSITY_REGISTRY",
    "POD_REGISTRY",
    "X_MAX_DEFAULT",
    "X_MIN_DEFAULT",
    "AdditiveMultiCovariatePOD",
    "CloglogPOD",
    "CoalMineVentilationIntensity",
    "ConcentrationProxyPOD",
    "ConstantIntensity",
    "DiurnalSeasonalIntensity",
    "DiurnalSinusoidalIntensity",
    "EmissionCatalog",
    "EmissionEvent",
    "FacilityConfig",
    "FullVaryingCoefficientPOD",
    "InhomogeneousIntensityPosterior",
    "LandfillIntensity",
    "LivestockFeedlotIntensity",
    "LogLogisticPOD",
    "LogisticPOD",
    "OffshorePlatformIntensity",
    "OperationalScheduleIntensity",
    "ParadoxResult",
    "PerEventPosterior",
    "PeriodicBatchIntensity",
    "PoissonRatePosterior",
    "ProbitPOD",
    "SNRBasedPOD",
    "SeasonalSinusoidalIntensity",
    "SeasonallyModulatedDiurnalIntensity",
    "SizeDistributionPosterior",
    "SpectralAwarePOD",
    "VaryingCoefficientPOD",
    "WeibullRenewalIntensity",
    "WetlandPermafrostIntensity",
    "build_canonical_scenarios",
    "compute_E_Pd",
    "event_from_posterior",
    "fit_inhomogeneous_intensity",
    "fit_lognormal_size_distribution",
    "fit_poisson_rate",
    "logistic_pod",
    "lognorm_cdf",
    "lognormal_pdf",
    "pod_powerlaw_model",
    "power_law",
    "run_mcmc",
    "run_scenario_grid",
    "simulate_paradox",
]

_LAZY = {
    "SizeDistributionPosterior": "plumax.population.size_distribution",
    "fit_lognormal_size_distribution": "plumax.population.size_distribution",
    "InhomogeneousIntensityPosterior": "plumax.population.point_process",
    "fit_inhomogeneous_intensity": "plumax.population.point_process",
    "X_MAX_DEFAULT": "plumax.population.fitting",
    "X_MIN_DEFAULT": "plumax.population.fitting",
    "lognorm_cdf": "plumax.population.fitting",
    "pod_powerlaw_model": "plumax.population.fitting",
    "power_law": "plumax.population.fitting",
    "run_mcmc": "plumax.population.fitting",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve NumPyro-dependent symbols on first access (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is not None:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
