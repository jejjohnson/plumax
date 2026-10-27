# Meteorology

Gridded meteorology (`MetField`) on the `les_fvm` analysis grid, the `wrfout` loader, and the adapter that feeds it to the Eulerian solver as wind; the shared lat/lon frame (`LocalFrame`) and clock (`TimeAxis`) every loader places data with. **Tier 0.** The loaders and `LocalFrame`'s transforms need `pip install 'plumax[io]'`.

::: plumax.met
