"""plumax — forward models for atmospheric plume dispersion.

Sub-packages
------------
- ``gauss_plume``  : steady-state Gaussian plume (JAX + NumPyro).
- ``gauss_puff``   : time-resolved Gaussian puff (JAX + diffrax + NumPyro),
                     with optional Ornstein-Uhlenbeck sub-grid turbulence.
- ``les_fvm``      : Eulerian 3-D advection-diffusion on an Arakawa C-grid
                     (JAX + diffrax + finitevolX) for spatially-varying
                     wind fields and K-theory eddy diffusivity.
- ``met``          : gridded meteorology (``MetField``) on the ``les_fvm``
                     analysis grid, with a ``wrfout`` loader and the adapter
                     that feeds it to the Eulerian solver as wind.
- ``obs``          : satellite L2 methane products (TROPOMI, EMIT, GHGSat)
                     as ``coupled.Instrument``s with observed enhancements.
- ``priors``       : inventory prior mean on the analysis grid and the
                     per-tile observation background.
- ``vetting``      : plume-candidate vetting on arrays (seeds, background
                     pool, scene null) on radtrax's absorption-shape test.

The ``operators`` and ``adapters`` modules expose the forward models as
``pipekit`` operators and ``pipekit_cycle`` protocol adapters respectively.

Additional dispersion models (resolved-flow LES, etc.) may be added as
sibling sub-packages in future ports.
"""

from __future__ import annotations

from plumax import (
    coupled,
    gauss_plume,
    gauss_puff,
    les_fvm,
    met,
    obs,
    population,
    priors,
    vetting,
)


__version__ = "0.2.0"  # x-release-please-version

__all__ = [
    "coupled",
    "gauss_plume",
    "gauss_puff",
    "les_fvm",
    "met",
    "obs",
    "population",
    "priors",
    "vetting",
]
