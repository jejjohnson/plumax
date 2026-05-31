"""Tier V — source population & forecasting.

Assembly of per-event Tier II-IV posteriors into population-scale
products:

* :mod:`~plumax.population.catalog` — the cross-tier posterior catalog
  (the load-bearing interface).
* :mod:`~plumax.population.size_distribution` — V.A instantaneous
  emission size distribution (hierarchical lognormal fit).
* :mod:`~plumax.population.point_process` — V.B spatio-temporal event-rate
  models (homogeneous Gamma-Poisson + log-linear inhomogeneous intensity).

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
from plumax.population.point_process import (
    PoissonRatePosterior,
    fit_poisson_rate,
)


if TYPE_CHECKING:
    from plumax.population.point_process import (
        InhomogeneousIntensityPosterior,
        fit_inhomogeneous_intensity,
    )
    from plumax.population.size_distribution import (
        SizeDistributionPosterior,
        fit_lognormal_size_distribution,
    )

__all__ = [
    "EmissionCatalog",
    "EmissionEvent",
    "InhomogeneousIntensityPosterior",
    "PerEventPosterior",
    "PoissonRatePosterior",
    "SizeDistributionPosterior",
    "event_from_posterior",
    "fit_inhomogeneous_intensity",
    "fit_lognormal_size_distribution",
    "fit_poisson_rate",
]

_LAZY = {
    "SizeDistributionPosterior": "plumax.population.size_distribution",
    "fit_lognormal_size_distribution": "plumax.population.size_distribution",
    "InhomogeneousIntensityPosterior": "plumax.population.point_process",
    "fit_inhomogeneous_intensity": "plumax.population.point_process",
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
