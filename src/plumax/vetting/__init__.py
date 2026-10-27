"""Plume-candidate vetting: is a matched-filter candidate the gas, or clutter?

Candidate-level vetting (Xiang et al. 2024; Růžička et al. 2026) as L1
functions on arrays: no carrier, no geotoolz. A candidate is a boolean mask
on a radiance cube and its matched-filter map; this package chooses which of
its pixels count as in (the seeds) and which as out (the background pool),
and runs radtrax's plume-agnostic absorption-shape test on them
(``radtrax.retrieval.absorption``: the pairing, the ratio spectrum, the fit
to the gas's transmittance and ``D_norm``). Theory: the plume-vetting page.

- `seeds`: `select_seeds` and `background_pool`, a candidate's in-plume
  seeds and the background pool they are paired from (#154).
"""

from __future__ import annotations

from plumax.vetting.seeds import background_pool, select_seeds


__all__ = [
    "background_pool",
    "select_seeds",
]
