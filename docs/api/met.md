# Meteorology

Gridded meteorology (`MetField`) on the `les_fvm` analysis grid, the `wrfout` and ERA5 loaders, and the adapter that feeds it to the Eulerian solver as wind; the shared lat/lon frame (`LocalFrame`) and clock (`TimeAxis`) every loader places data with; Monin–Obukhov surface-layer scales (`surface_layer_from_met`), a bulk-Richardson PBL height and the Hanna turbulence factory. **Tier 0.** The loaders and `LocalFrame`'s transforms need `pip install 'plumax[io]'`.

::: plumax.met
