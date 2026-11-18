"""Steady-state Gaussian plume forward model (JAX).

Implements the analytical solution to the steady-state advection-diffusion
equation with a ground-reflection boundary condition, in wind-aligned
coordinates (x', y', z):

    C(x', y', z) = (Q / (2π u σ_y σ_z))
                   · exp(-y'² / (2 σ_y²))
                   · [exp(-(z - H)² / (2 σ_z²)) + exp(-(z + H)² / (2 σ_z²))]

where Q is the emission rate [kg/s], u = √(u² + v²) the wind speed [m/s],
σ_y(x'), σ_z(x') the Briggs dispersion coefficients, and H the source height.

Public surface
--------------
- ``rotate_to_wind_frame``       : JIT-compiled (x, y) → (x', y') rotation
- ``plume_concentration``        : JIT-compiled forward model
- ``plume_concentration_vmap``   : vmapped over receptor locations
- ``simulate_plume``              : xarray wrapper building a 3-D field
"""

from __future__ import annotations

from plumax.gauss_plume._src.plume import (
    MIN_WIND_SPEED as MIN_WIND_SPEED,
    plume_concentration,
    plume_concentration_vmap as plume_concentration_vmap,
    rotate_to_wind_frame,
    simulate_plume,
)


__all__ = [
    "plume_concentration",
    "rotate_to_wind_frame",
    "simulate_plume",
]
