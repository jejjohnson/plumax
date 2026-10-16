"""``plumax.radtran`` moved to radtrax (plumax#133).

The old path raises rather than re-exporting, so no name has two homes.
"""

raise ImportError(
    "plumax.radtran moved to radtrax: the slab forward models, unit constants and nB-LUT are in radtrax.slab; SpectralResponseFunction, PointSpreadFunction and GroundSamplingDistance are in radtrax.instrument."
)
