"""Data-assimilation adapters for the ``pipekit_cycle`` protocols.

``pipekit_cycle`` decomposes data assimilation into three runtime-checkable
protocols — ``ForwardModel`` (predict), ``ObservationOperator`` (compare),
``AnalysisStep`` (update). plumax ships adapters that satisfy them
**structurally**: the classes below expose exactly the right methods /
properties, so ``isinstance(obj, ForwardModel)`` succeeds, *without* plumax
importing ``pipekit_cycle`` at runtime. This keeps the dependency arrow
pointing the right way — assimilation drivers depend on plumax, not the
reverse.

- :class:`EulerianForwardModel` — wraps an
  :class:`~plumax.les_fvm.EulerianDispersionRHS` as a ``ForwardModel``,
  advancing a 3-D concentration field one explicit-Euler step per call.
- :class:`RadtranObservationOperator` — wraps the normalised Beer–Lambert
  forward model as an ``ObservationOperator`` mapping a scalar plume
  enhancement (ΔVMR) to a normalised radiance spectrum, with the analytic
  Jacobian as its tangent-linear.
"""

from __future__ import annotations

from plumax._src.adapters import (
    EulerianForwardModel,
    RadtranObservationOperator,
)


__all__ = [
    "EulerianForwardModel",
    "RadtranObservationOperator",
]
