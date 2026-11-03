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
* :mod:`~plumax.population.marks` — V.B mark registry, the size distribution
  ``f(Q | θ)`` (power law, lognormal, lognormal-Pareto, mixture of lognormals).
* :mod:`~plumax.population.processes` — the intensity and POD registries as
  ``xtremax.point_processes`` Poisson, marked and detection-thinned processes
  (the ``population`` extra).
* :mod:`~plumax.population.paradox` — V.D missing-mass paradox Monte Carlo.
* :mod:`~plumax.population.importance` — importance-corrected per-event
  evidence (POINT / GAUSSIAN / FULL regimes) and the ESS diagnostic.
* :mod:`~plumax.population.integrate` — log-space Gauss–Hermite and Simpson
  quadrature, Pareto-smoothed importance weights.
* :mod:`~plumax.population.fitting` — POD-modified power-law NUTS fitter and
  the importance-corrected mark and TMTPP models.

Importing this package is cheap: the NumPyro-dependent fit functions and
result types are bound lazily (PEP 562), so ``import plumax.population``
never pulls in NumPyro. xtremax (the ``population`` extra) is imported only
when a process is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from plumax.population.catalog import (
    EmissionCatalog,
    EmissionEvent,
    PerEventPosterior,
    event_from_posterior,
)
from plumax.population.importance import (
    EventEvidence,
    Regime,
    catalog_evidence,
    effective_sample_size,
    ess_report,
    event_evidence,
)
from plumax.population.integrate import (
    log_gauss_hermite,
    log_simpson,
    pareto_smoothed_log_weights,
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
from plumax.population.marks import (
    MARKS,
    LognormalMark,
    LognormalParetoMark,
    Mark,
    MixtureLognormalMark,
    PowerLawMark,
    make_mark,
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
from plumax.population.processes import (
    LogIntensity,
    PODRetention,
    detection_process,
    make_process,
    padded_event_buffer,
)


if TYPE_CHECKING:
    from plumax.population.fitting import (
        X_MAX_DEFAULT,
        X_MIN_DEFAULT,
        MarkPosterior,
        fit_size_distribution_iw,
        lognorm_cdf,
        mark_iw_model,
        pod_powerlaw_model,
        power_law,
        run_mcmc,
        tmtpp_iw_model,
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
    "MARKS",
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
    "EventEvidence",
    "FacilityConfig",
    "FullVaryingCoefficientPOD",
    "InhomogeneousIntensityPosterior",
    "LandfillIntensity",
    "LivestockFeedlotIntensity",
    "LogIntensity",
    "LogLogisticPOD",
    "LogisticPOD",
    "LognormalMark",
    "LognormalParetoMark",
    "Mark",
    "MarkPosterior",
    "MixtureLognormalMark",
    "OffshorePlatformIntensity",
    "OperationalScheduleIntensity",
    "PODRetention",
    "ParadoxResult",
    "PerEventPosterior",
    "PeriodicBatchIntensity",
    "PoissonRatePosterior",
    "PowerLawMark",
    "ProbitPOD",
    "Regime",
    "SNRBasedPOD",
    "SeasonalSinusoidalIntensity",
    "SeasonallyModulatedDiurnalIntensity",
    "SizeDistributionPosterior",
    "SpectralAwarePOD",
    "VaryingCoefficientPOD",
    "WeibullRenewalIntensity",
    "WetlandPermafrostIntensity",
    "build_canonical_scenarios",
    "catalog_evidence",
    "compute_E_Pd",
    "detection_process",
    "effective_sample_size",
    "ess_report",
    "event_evidence",
    "event_from_posterior",
    "fit_inhomogeneous_intensity",
    "fit_lognormal_size_distribution",
    "fit_poisson_rate",
    "fit_size_distribution_iw",
    "log_gauss_hermite",
    "log_simpson",
    "logistic_pod",
    "lognorm_cdf",
    "lognormal_pdf",
    "make_mark",
    "make_process",
    "mark_iw_model",
    "padded_event_buffer",
    "pareto_smoothed_log_weights",
    "pod_powerlaw_model",
    "power_law",
    "run_mcmc",
    "run_scenario_grid",
    "simulate_paradox",
    "tmtpp_iw_model",
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
    "MarkPosterior": "plumax.population.fitting",
    "fit_size_distribution_iw": "plumax.population.fitting",
    "mark_iw_model": "plumax.population.fitting",
    "tmtpp_iw_model": "plumax.population.fitting",
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
