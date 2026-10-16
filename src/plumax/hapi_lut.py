"""``plumax.hapi_lut`` moved to radtrax (plumax#133).

The old path raises rather than re-exporting, so no name has two homes.
"""

raise ImportError(
    "plumax.hapi_lut moved to radtrax: the HITRAN table builder is radtrax.spectroscopy.hitran (install radtrax[hitran]) and Beer-Lambert on a table is radtrax.spectroscopy.tables."
)
