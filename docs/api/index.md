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
| 0 | [Meteorology](met.md) | `MetField`, `load_wrf`, `to_prescribed_wind` |
| I | [Gaussian plume](gauss_plume.md) | steady-state plume + emission-rate inference |
| I | [Gaussian puff](gauss_puff.md) | time-resolved puffs, OU turbulence |
| II | [Lagrangian](lagrangian.md) | particle dispersion |
| III | [Eulerian FV](les_fvm.md) | advection–diffusion on a C-grid |
| IV | [Coupled](coupled.md) | multi-instrument source inversion |
| V | [Population](population.md) | source populations and forecasting |
| — | [Operators](operators.md) · [Adapters](adapters.md) | `pipekit` integration |

## Package overview

::: plumax
    options:
      members: false
      show_root_heading: false
      show_source: false
