"""Satellite L2 observations for the coupled inversion (Tier 0 prerequisites).

Loaders that turn one L2 methane product into a
:class:`~plumax.coupled.Instrument` (receptors on a
:class:`~plumax.met.LocalFrame`, a scalar averaging kernel, the retrieval
variance) plus the observed column-mass enhancement
:func:`~plumax.coupled.fuse_observations` takes:

- :func:`load_tropomi_l2` — TROPOMI CH₄ (bias-corrected XCH₄, column
  averaging kernel, QA screening).
- :func:`load_emit_l2b` — EMIT L2B CH₄ enhancement (ppm·m).
- :func:`load_ghgsat_l2` — GHGSat per-plume XCH₄.

For the image fit, one plume rather than a whole scene:

- :func:`load_emit_plume_complex` — one catalogued plume of an EMIT scene as
  a :class:`PlumeScene`: the window around its outline with the sensitivity
  as the averaging kernel, a per-pixel noise, the quality mask, the
  pseudo-origin and a starting direction.
- :func:`matched_filter_scene` — the same observation from radtrax's score,
  noise and sensitivity maps.

The files are read with xrproducts's openers; needs the ``io`` extra.
"""

from __future__ import annotations

from plumax.obs.ingest import (
    ColumnObservation,
    load_emit_l2b,
    load_ghgsat_l2,
    load_tropomi_l2,
    ppb_to_column_mass,
    ppm_m_to_column_mass,
)
from plumax.obs.plume_scene import (
    PlumeScene,
    load_emit_plume_complex,
    matched_filter_scene,
)


__all__ = [
    "ColumnObservation",
    "PlumeScene",
    "load_emit_l2b",
    "load_emit_plume_complex",
    "load_ghgsat_l2",
    "load_tropomi_l2",
    "matched_filter_scene",
    "ppb_to_column_mass",
    "ppm_m_to_column_mass",
]
