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
- `null`: `null_masks` and `null_pvalue`, rotated and translated copies of
  a candidate's mask off every candidate, and the candidate's p-value
  among their scores (#155).
- `vet`: `vet_candidate` and `VettingResult`, one candidate vetted in one
  call: seeds, pool, radtrax's pairs, ratio, fit and ``D_norm`` (#156).
"""

from __future__ import annotations

from plumax.vetting.null import null_masks, null_pvalue
from plumax.vetting.seeds import background_pool, select_seeds
from plumax.vetting.vet import VettingResult, vet_candidate


__all__ = [
    "VettingResult",
    "background_pool",
    "null_masks",
    "null_pvalue",
    "select_seeds",
    "vet_candidate",
]
