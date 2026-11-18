# Observations

Satellite L2 methane products as `coupled.Instrument`s on a `LocalFrame`, with the observed column-mass enhancement `fuse_observations` takes: TROPOMI, EMIT and GHGSat, read through xrproducts. **Tier 0.** Needs `pip install 'plumax[io]'`.

::: plumax.obs

## `plume_scene`: one plume for the image fit

`load_emit_plume_complex` loads one catalogued plume of an EMIT scene as a `PlumeScene`: the window around its outline (plus a margin), with the sensitivity as the averaging kernel, a per-pixel noise, the quality mask (cloud, cirrus and water from the L2A mask, and the saturation mask from the band mask, through `geoops.qa`), the pseudo-origin and a starting direction. `matched_filter_scene` builds the same observation from radtrax's score, noise and sensitivity maps. The enhancement is never divided by the sensitivity; the model is multiplied by it instead.

::: plumax.obs.plume_scene
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - PlumeScene
        - load_emit_plume_complex
        - matched_filter_scene
