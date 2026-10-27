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

The files are read with xrreader's openers; needs the ``io`` extra.
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


__all__ = [
    "ColumnObservation",
    "load_emit_l2b",
    "load_ghgsat_l2",
    "load_tropomi_l2",
    "ppb_to_column_mass",
    "ppm_m_to_column_mass",
]
