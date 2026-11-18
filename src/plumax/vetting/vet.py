"""One candidate, vetted: seeds, pool, pairs, ratio, fit and score in one call.

`vet_candidate` runs Xiang et al.'s steps in their fixed order with the
window's band selection: plumax's `select_seeds` and `background_pool`, then
radtrax's absorption-shape test on the two sets (``pair_pixels`` on the
window's pairing bands, ``ratio_spectrum`` on its fitting bands,
``fit_transmittance`` and ``d_norm``, from ``radtrax.retrieval.absorption``).
It returns a `VettingResult`, the scores and the spectra an analyst plots
(``R / P`` against ``E``). Theory: the plume-vetting page.

The result carries scores, not a verdict: the decision rule (a fixed
``D_norm`` cut, the scene null's ``p̂``, or a likelihood ratio between the
fits with and without the gas) is #158's. For the last,
``residual_ss_no_gas`` is the fit's residual at ``ΔX = 0`` beside the
``fit.residual_ss`` at ``ΔX̂``; for ``p̂``, vet each of
`null_masks`' copies the same way and pass the scores to `null_pvalue`.

``ΔX̂`` is the column of the *mean* transmittance over the pairs, not the
mean column (``E`` is convex in ``ΔX``), so it sits below the mean of the
truth over the seeds: a score, not a retrieval.
"""

from __future__ import annotations

from plumax.vetting._src.vet import (
    VettingResult,
    vet_candidate,
)


__all__ = [
    "VettingResult",
    "vet_candidate",
]
