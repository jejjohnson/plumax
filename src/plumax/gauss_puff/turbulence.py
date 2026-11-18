"""Ornstein-Uhlenbeck turbulence for Gaussian-puff position disturbances.

The base :mod:`plumax.gauss_puff.puff` model advects every puff by
the cumulative wind integral

    x_i(t) = x_src + ∫_{t_r^{(i)}}^{t} u(τ) dτ ,          y_i analogous,

so two puffs released at the same instant sit on top of each other and puffs
released close together are near-collinear along the mean wind. Real plumes
show sub-grid meander and turbulence that this model cannot resolve. Following
the Gorroño-calibrated approach used by Orbio's Project Eucalyptus we add a
per-puff offset drawn from a 2-D Ornstein-Uhlenbeck (OU) process indexed by
release time:

    dp(τ) = -p(τ)/τ_c · dτ + σ_f · dW_τ                    (SDE)

where τ_c is the OU correlation time [s] and σ_f is the fluctuation
amplitude [m/√s]. Under the exact transition

    p(τ + Δτ) = p(τ) · exp(-Δτ / τ_c)
              + σ_f · √(τ_c/2 · (1 - exp(-2 Δτ / τ_c))) · ξ,   ξ ~ 𝒩(0, I)

the steady-state variance is Var[p_∞] = σ_f² · τ_c / 2 (per component). We
evaluate the process at the puff release times and add the result to each
puff's advected centre as a *time-of-release* offset — one sample per puff,
held constant over the puff's lifetime. This reproduces the observed meander
structure (neighbouring puffs share similar offsets, distant-in-time puffs are
independent) without introducing a second SDE into :func:`evolve_puffs`.

Public surface
--------------
- :class:`OUTurbulence`  — dataclass bundling (σ_f, τ_c).
- :func:`sample_ou_offsets` — draw per-release-time (Δx, Δy) from the SDE.

Design choices
~~~~~~~~~~~~~~
* The sampler returns NumPy arrays (not JAX): OU samples are consumed by the
  JIT-compiled :func:`evolve_puffs` as static per-puff offsets, so staying on
  the CPU side avoids a spurious recompilation per key and keeps the API
  friendly to callers who prefer :mod:`numpy.random` for reproducibility.
* ``seed`` (an ``int`` or :class:`numpy.random.Generator`) is an explicit
  argument rather than a thread-local — tests and notebooks pin it for
  reproducibility.
"""

from __future__ import annotations

from plumax.gauss_puff._src.turbulence import (
    OUTurbulence,
    sample_ou_offsets,
)


__all__ = [
    "OUTurbulence",
    "sample_ou_offsets",
]
