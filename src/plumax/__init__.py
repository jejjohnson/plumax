"""plumax — forward models for atmospheric plume dispersion.

Sub-packages
------------
- ``coupled``      : multi-instrument coupled source inversion that composes
                     a transport tier with the observation operator (Tier IV).
- ``gauss_plume``  : steady-state Gaussian plume (JAX + NumPyro).
- ``gauss_puff``   : time-resolved Gaussian puff (JAX + diffrax + NumPyro),
                     with optional Ornstein-Uhlenbeck sub-grid turbulence.
- ``geo``          : plume operators on GeoTensors ported from geotoolz
                     (masks, footprints, IME, cross-sectional flux), behind
                     the ``[geo]`` extra; loaded on first use.
- ``les_fvm``      : Eulerian 3-D advection-diffusion on an Arakawa C-grid
                     (JAX + diffrax + finitevolX) for spatially-varying
                     wind fields and K-theory eddy diffusivity.
- ``lagrangian``   : Markov-1 Langevin particle dispersion with forward
                     concentrations, backward footprints and the Gaussian /
                     lognormal source inversions (Tier II).
- ``met``          : gridded meteorology (``MetField``) on the ``les_fvm``
                     analysis grid, with a ``wrfout`` loader and the adapter
                     that feeds it to the Eulerian solver as wind.
- ``obs``          : satellite L2 methane products (TROPOMI, EMIT, GHGSat)
                     as ``coupled.Instrument``s with observed enhancements.
- ``population``   : source-population products assembled from per-event
                     posteriors: catalogs, point processes, size
                     distributions (Tier V).
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
    geo,
    lagrangian,
    les_fvm,
    met,
    obs,
    population,
    priors,
    vetting,
)


__version__ = "0.2.3"  # x-release-please-version

__all__ = [
    "coupled",
    "gauss_plume",
    "gauss_puff",
    "geo",
    "lagrangian",
    "les_fvm",
    "met",
    "obs",
    "population",
    "priors",
    "vetting",
]
