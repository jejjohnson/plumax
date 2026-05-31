"""plumax — forward models for atmospheric plume dispersion.

Sub-packages
------------
- ``gauss_plume``  : steady-state Gaussian plume (JAX + NumPyro).
- ``gauss_puff``   : time-resolved Gaussian puff (JAX + diffrax + NumPyro),
                     with optional Ornstein-Uhlenbeck sub-grid turbulence.
- ``les_fvm``      : Eulerian 3-D advection-diffusion on an Arakawa C-grid
                     (JAX + diffrax + finitevolX) for spatially-varying
                     wind fields and K-theory eddy diffusivity.
- ``hapi_lut``     : HITRAN line-by-line absorption cross-section LUTs
                     (HAPI + xarray) plus a Beer-Lambert forward model and
                     the differential-ratio form for plume-enhancement
                     retrievals. ``hitran-api`` is imported lazily.
- ``radtran``      : Band-integrated Beer-Lambert forward model, normalised
                     -brightness LUT, and matched-filter retrieval
                     (multispectral / hyperspectral).
- ``matched_filter``: hyperspectral matched-filter detection pipeline.
- ``assimilation`` : 3D/4D-Var cost / control / solve scaffolding
                     (``optimistix``); imported on demand.

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
    hapi_lut,
    les_fvm,
    population,
    radtran,
)


__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "coupled",
    "gauss_plume",
    "gauss_puff",
    "hapi_lut",
    "les_fvm",
    "population",
    "radtran",
]
