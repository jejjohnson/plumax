"""Tier III — strong-constraint 4D-Var inversion over the Eulerian transport model.

Recovers a time-resolved emission signal from a sequence of column observations
by directly minimising the strong-constraint 4D-Var cost
([design](https://github.com/jejjohnson/plumax/blob/main/docs/design/03_tier3_eulerian.md#tier3-cost)):

    J(S) = ½‖S − S_b‖²_B  +  ½ Σ_t ‖H_t c(S, t) − y_t‖²_{R_t}

where ``c(S, t)`` is the 3-D tracer field produced by the finite-volume solver
:func:`~plumax.les_fvm.simulate_eulerian_dispersion` for emission signal ``S``,
``H_t`` is the column observation operator (vertical integral → optional spatial
sampling), ``S_b`` / ``B`` the source prior, and ``R_t`` the observation-error
covariance at overpass ``t``. By default the initial field is pinned at zero and
only the source is inverted; a :class:`BackgroundControl` adds the IC / background
term ``½‖c_0 − c_b‖²_{B_c}`` of the three-term design cost and per-face lateral
boundary scaling factors to the control.

The solve is a **vardax analysis** (the ``assim`` extra,
``pip install 'plumax[assim]'``). The control is a source, not an initial
state, so the cost above is ``vardax.ThreeDVar`` with the source–receptor map
``𝒢(S) = vec H c(S, ·)`` as its observation operator
(:meth:`FourDVarProblem.source_receptor_operator`), minimised by L-BFGS. The key
enabler (design §adjoint) is that **JAX reverse-mode autodiff through the diffrax
FV solve is the exact discrete adjoint** — there is no hand-written adjoint
model. The diffrax solve defines only a VJP, so the operator is linearised in
reverse mode (``CallableObs(jac="bwd", materialise=True)``). The incremental
variant (design §incremental), Gauss–Newton outer and CG inner iterations, is
``vardax.IncrementalFourDVar`` over the same map
(:meth:`FourDVarProblem.incremental_analysis`).

The control vector is either the **time-resolved scalar emission rate** ``S = q``
at a known source location (:class:`TemporalControl`, or the ``prior_mean`` and
``prior_covariance`` arguments of :func:`build_problem`), or a **per-cell surface
flux map** with a Matérn-3/2 prior (:class:`SpatialControl`). The analysis is
written against a flat control vector, so both share one solver.

Control-variable transform (design §control-transform): optimisation runs in
**whitened space** ``χ = L⁻¹ (S − S_b)`` where ``B = L Lᵀ``, through
``vardax.LinearParameterization(linear_basis(L, variance=1))``. The prior term
is then the identity ``½‖χ‖²``, which is what makes the problem well-conditioned;
the Matérn-3/2 temporal prior from :mod:`plumax.lagrangian.inversion` supplies
``B``. The observations are whitened too (``y / σ`` with ``R = I``), so the
tiny concentration-scale variances never enter a linear solve. The posterior is
the Gauss–Newton Laplace approximation of that whitened control, an argminax
``Laplace``.
"""

from __future__ import annotations

from plumax.les_fvm._src.fourdvar import (
    BackgroundAnalysis,
    BackgroundControl,
    ColumnObservationOperator,
    EulerianForward4DVar,
    FourDVarProblem,
    FourDVarResult,
    PosteriorCovariance,
    SpatialControl,
    TemporalControl,
    build_forward,
    build_problem,
    laplace_sample,
    posterior_covariance,
    solve_4dvar,
)


__all__ = [
    "BackgroundAnalysis",
    "BackgroundControl",
    "ColumnObservationOperator",
    "EulerianForward4DVar",
    "FourDVarProblem",
    "FourDVarResult",
    "PosteriorCovariance",
    "SpatialControl",
    "TemporalControl",
    "build_forward",
    "build_problem",
    "laplace_sample",
    "posterior_covariance",
    "solve_4dvar",
]
