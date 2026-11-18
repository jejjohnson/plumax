"""Briggs-McElroy-Pooler dispersion coefficients (JAX).

The Briggs parameterisation expresses the lateral and vertical dispersion
coefficients of a steady-state Gaussian plume as

    σ_y(x) = a_y · x · (1 + b_y · x)^{c_y}       [m]
    σ_z(x) = a_z · x · (1 + b_z · x)^{c_z}       [m]

with one parameter set per Pasquill-Gifford stability class A-F.

Public surface
--------------
- ``BRIGGS_DISPERSION_PARAMS`` : mapping {stability class → jnp.ndarray(6,)}
- ``STABILITY_CLASSES``        : canonical ordering ('A', ..., 'F')
- ``calculate_briggs_dispersion`` : JIT-compiled σ_y, σ_z evaluator
- ``get_dispersion_params``    : validated dict lookup
"""

from __future__ import annotations

from plumax.gauss_plume._src.dispersion import (
    BRIGGS_DISPERSION_PARAMS as BRIGGS_DISPERSION_PARAMS,
    STABILITY_CLASSES as STABILITY_CLASSES,
    calculate_briggs_dispersion,
    get_dispersion_params,
)


__all__ = [
    "calculate_briggs_dispersion",
    "get_dispersion_params",
]
