"""High-level diffrax runner returning an xarray ``Dataset``.

This module wires every other ``les_fvm`` piece together behind a single
user-facing entry point, :func:`simulate_eulerian_dispersion`, modelled
after ``gauss_puff.simulate_puff`` for cross-model consistency.
"""

from __future__ import annotations

from plumax.les_fvm._src.simulate import (
    AdvectionScheme as AdvectionScheme,
    SolverName as SolverName,
    simulate_eulerian_dispersion,
    stable_step_bound,
)


__all__ = [
    "simulate_eulerian_dispersion",
    "stable_step_bound",
]
