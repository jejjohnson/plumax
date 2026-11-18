"""Tier V.B — spatio-temporal point process (event rate).

The tractable parametric core of the TMTPP event-rate model:

* :func:`fit_poisson_rate` — homogeneous Poisson rate ``lambda`` (events
  per unit area per unit time) via the closed-form Gamma-Poisson conjugate
  posterior.  Pure NumPy / SciPy, no NumPyro.
* :func:`fit_inhomogeneous_intensity` — inhomogeneous log-linear intensity
  ``log lambda = beta0 + beta . covariates`` fit with NUTS.  NumPyro is
  imported lazily.

A full log-Gaussian Cox process (LGCP) is out of scope for v1; it is the
natural next kernel when clustering is environmentally driven (see
``docs/design/06_tier5b_point_process.md``).
"""

from __future__ import annotations

from plumax.population._src.point_process import (
    InhomogeneousIntensityPosterior,
    PoissonRatePosterior,
    fit_inhomogeneous_intensity,
    fit_poisson_rate,
)


__all__ = [
    "InhomogeneousIntensityPosterior",
    "PoissonRatePosterior",
    "fit_inhomogeneous_intensity",
    "fit_poisson_rate",
]
