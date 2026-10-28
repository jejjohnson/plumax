# API Reference

Each forward model is its own sub-package, importable directly:

```python
from plumax.gauss_plume import simulate_plume
from plumax.les_fvm import simulate_eulerian_dispersion
```

The pages follow the modelling tiers of the
[roadmap](https://jejjohnson.github.io/plumax/roadmap/). NumPyro inference
submodules are imported lazily, so importing a forward model never pulls in
NumPyro. Radiative transfer, the matched filter and the 3D-Var retrieval are in [radtrax](https://github.com/jejjohnson/radtrax).

## Modules

| Tier | Module | Contents |
|---|---|---|
| 0 | [Meteorology](met.md) | `MetField`, `load_wrf`, `load_era5`, `LocalFrame`, `TimeAxis`, `surface_layer_from_met`, `to_prescribed_wind` |
| 0 | [Observations](obs.md) | `load_tropomi_l2`, `load_emit_l2b`, `load_ghgsat_l2`, `ColumnObservation` |
| 0 | [Priors](priors.md) | `load_inventory`, `per_tile_background` |
| I | [Gaussian plume](gauss_plume.md) | steady-state plume + emission-rate inference |
| I | [Gaussian puff](gauss_puff.md) | time-resolved puffs, OU turbulence |
| II | [Lagrangian](lagrangian.md) | particle dispersion |
| III | [Eulerian FV](les_fvm.md) | advection–diffusion on a C-grid |
| IV | [Coupled](coupled.md) | multi-instrument source inversion |
| V | [Population](population.md) | source populations and forecasting |
| — | [Plume vetting](vetting.md) | `vet_candidate`, `select_seeds`, `background_pool`, `null_masks`, `null_pvalue` on radtrax's absorption-shape test |
| — | [Operators](operators.md) · [Adapters](adapters.md) | `pipekit` integration |

## Package overview

::: plumax
    options:
      members: false
      show_root_heading: false
      show_source: false
