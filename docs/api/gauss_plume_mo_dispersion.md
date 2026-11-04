# Monin–Obukhov dispersion

Pasquill–Gifford dispersion driven by Monin–Obukhov scales: Golder's (1972) class boundaries map the inverse Obukhov length 1/L and the roughness length z₀ to a PG class, whose Briggs σ_y and σ_z curves then give the plume spread. `GolderDispersion` is built from a column of `plumax.met.SurfaceLayer` and passed as `dispersion_params` to `plume_concentration` or as `stability_class` to `simulate_plume`. The hard class lookup follows Golder; the optional `interpolate=True` blends neighbouring classes linearly in 1/L so σ is differentiable in L, which is a plumax modelling choice. PG classes given by letter remain the default. **Tier I.**

::: plumax.gauss_plume.mo_dispersion
