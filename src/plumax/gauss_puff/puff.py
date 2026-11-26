"""Gaussian puff forward model (JAX).

A continuous source releasing at rate ``Q(t)`` is represented by a sequence
of instantaneous puffs at release times ``t_r^{(i)}`` carrying mass
``m_i = Q(t_r^{(i)}) · Δt_release``. Each puff is advected by the (possibly
time-varying) wind and spreads as a 3-D Gaussian:

    C_i(x, y, z; t) = m_i / [(2π)^{3/2} σ_x σ_y σ_z]
                      · exp[−(x − x_i(t))² / (2 σ_x²)]
                      · exp[−(y − y_i(t))² / (2 σ_y²)]
                      · [exp(−(z − z_s)² / (2 σ_z²))
                         + exp(−(z + z_s)² / (2 σ_z²))]          (ground reflection)

The total field is the superposition ``C = Σ_i C_i`` over all active puffs.
Dispersion coefficients are evaluated as functions of puff travel distance
``s_i(t) = ∫_{t_r^{(i)}}^{t} |V(τ)| dτ``.

Public surface
--------------
- ``release_interval_to_frequency`` / ``frequency_to_release_interval`` : helpers
- ``make_release_times``                 : build evenly spaced release times
- ``PuffState``                           : eqx.Module bundling puff positions/mass
- ``puff_concentration`` / ``puff_concentration_vmap`` : single-puff kernel
- ``evolve_puffs``                        : release_times + wind → PuffState via the wind integrals
- ``simulate_puff_field``                : JIT orchestrator, field at one time
- ``simulate_puff``                       : xarray wrapper with ``time`` axis
"""

from __future__ import annotations

from plumax.gauss_puff._src.puff import (
    PuffState,
    evolve_puffs,
    frequency_to_release_interval,
    make_release_times,
    puff_concentration,
    puff_concentration_vmap as puff_concentration_vmap,
    release_durations,
    release_interval_to_frequency,
    simulate_puff,
    simulate_puff_field,
)


__all__ = [
    "PuffState",
    "evolve_puffs",
    "frequency_to_release_interval",
    "make_release_times",
    "puff_concentration",
    "release_durations",
    "release_interval_to_frequency",
    "simulate_puff",
    "simulate_puff_field",
]
