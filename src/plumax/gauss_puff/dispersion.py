"""Pasquill-Gifford dispersion coefficients for the Gaussian puff model (JAX).

The Pasquill-Gifford (PG) parameterisation expresses the lateral and vertical
dispersion coefficients of a Gaussian puff as log-quadratic functions of the
puff travel distance:

    σ_y(s) = exp(a_y + b_y · ln(s) + c_y · (ln s)²)       [m]
    σ_z(s) = exp(a_z + b_z · ln(s) + c_z · (ln s)²)       [m]

The along-wind dispersion coefficient is taken equal to the crosswind one
(``σ_x = σ_y``), reflecting isotropic horizontal spread of an instantaneous
release.

Coefficient values come from the classic Pasquill-Gifford (1961/1976) curves
re-expressed in log-quadratic form by several later authors (Turner, 1994;
Beychok, 2005). Parameters are stored per stability class A-F.

Public surface
--------------
- ``PG_DISPERSION_PARAMS`` : mapping {stability class → jnp.ndarray(6,)}
- ``STABILITY_CLASSES``     : canonical ordering ('A', ..., 'F')
- ``calculate_pg_dispersion`` : JIT-compiled σ_x, σ_y, σ_z evaluator
- ``get_pg_params``         : validated dict lookup
- ``calculate_briggs_dispersion_xyz`` : Briggs wrapper exposing σ_x = σ_y
"""

from __future__ import annotations

from plumax.gauss_puff._src.dispersion import (
    DISPERSION_SCHEMES as DISPERSION_SCHEMES,
    PG_DISPERSION_PARAMS as PG_DISPERSION_PARAMS,
    STABILITY_CLASSES as STABILITY_CLASSES,
    DispersionFn as DispersionFn,
    calculate_briggs_dispersion_xyz,
    calculate_pg_dispersion,
    get_dispersion_scheme,
    get_pg_params,
)


__all__ = [
    "calculate_briggs_dispersion_xyz",
    "calculate_pg_dispersion",
    "get_dispersion_scheme",
    "get_pg_params",
]
