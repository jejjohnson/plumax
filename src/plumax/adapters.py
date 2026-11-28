"""Data-assimilation adapters for the ``pipekit_cycle`` protocols.

``pipekit_cycle`` decomposes data assimilation into three runtime-checkable
protocols — ``ForwardModel`` (predict), ``ObservationOperator`` (compare),
``AnalysisStep`` (update). plumax ships adapters that satisfy them
**structurally**: the class below exposes exactly the right methods /
properties, so ``isinstance(obj, ObservationOperator)`` succeeds, *without*
plumax importing ``pipekit_cycle`` at runtime. This keeps the dependency arrow
pointing the right way — assimilation drivers depend on plumax, not the
reverse.

- :class:`RadtranObservationOperator` — wraps the normalised Beer–Lambert
  forward model as an ``ObservationOperator`` mapping a scalar plume
  enhancement (ΔVMR) to a normalised radiance spectrum, with its ``jax.jacfwd``
  as the tangent-linear.

A ``ForwardModel`` over an :class:`~plumax.les_fvm.EulerianDispersionRHS` is
``integrix.march.DiffraxForwardModel``, which takes any diffrax solver; with
``diffrax.Euler`` and ``dt0 = dt`` one ``step`` is one explicit-Euler step,
``c ← c + Δt · RHS(0, c)``::

    import diffrax as dfx
    from integrix.march import DiffraxForwardModel

    forward = DiffraxForwardModel(rhs, solver=dfx.Euler(), dt0=1.0, dt=1.0)
    # (Nz, Ny, Nx), integrated over [0, Δt]
    next_c = forward.step(c, 1.0)
    # (Nz, Ny, Nx), over [t₀, t₀ + Δt] at the true time
    next_c = forward.as_dynamics()(c, t0, t0 + 1.0)
"""

from __future__ import annotations

from plumax._src.adapters import RadtranObservationOperator


__all__ = [
    "RadtranObservationOperator",
]
