"""Tier V.D — missing-mass paradox Monte Carlo (no plotting, no I/O).

A pedagogical numerical proof of the satellite "missing mass paradox": a
satellite alerting system simultaneously **overestimates** the average
emission rate of a facility while strictly **underestimating** the total
emitted mass.

The paradox emerges from the thinned marked temporal point process (TMTPP)
framework: a size-dependent probability of detection preferentially
destroys small events, biasing sample means upward and sample sums
downward.

The simulator has three strictly separated layers:

1. Event generator — a temporal point process ``λ(t)`` [events day⁻¹].
2. Mark generator — a lognormal mark distribution ``f(Q)`` over
   ``Q`` [kg hr⁻¹].
3. Atmospheric filter — a parametric logistic POD ``P_d(Q)``.

Units: ``t`` in days, ``λ`` in events day⁻¹, ``Q`` in kg hr⁻¹, ``D`` in
hr event⁻¹, ``M`` in kg.

Pure NumPy / SciPy (CPU); the JAX / equinox intensity and POD models live
in :mod:`plumax.population.intensity` and :mod:`plumax.population.pod`.
"""

from __future__ import annotations

from plumax.population._src.paradox import (
    FacilityConfig,
    ParadoxResult,
    build_canonical_scenarios,
    compute_E_Pd,
    logistic_pod,
    lognormal_pdf,
    run_scenario_grid,
    simulate_paradox,
)


__all__ = [
    "FacilityConfig",
    "ParadoxResult",
    "build_canonical_scenarios",
    "compute_E_Pd",
    "logistic_pod",
    "lognormal_pdf",
    "run_scenario_grid",
    "simulate_paradox",
]
