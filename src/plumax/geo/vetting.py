"""Plume vetting on GeoArrays (behind the ``[geo]`` extra): one Dataset per scene.

`PlumeVetting` is the L2 operator over the array-level vetting in
:mod:`plumax.vetting` (#157): it reads a radiance GeoArray, its
matched-filter GeoArray and a stack of candidate masks, runs
`vet_candidate` on each candidate and the scene null (`null_masks`,
`null_pvalue`) beside it, and returns the scores as one ``xarray.Dataset``.
It returns scores, not a verdict: the decision rule is #158's.
"""

from __future__ import annotations

from plumax.geo._src.vetting import (
    PlumeVetting,
)


__all__ = [
    "PlumeVetting",
]
