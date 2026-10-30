# Gridded wind

`GriddedWind` samples the `les_fvm` C-grid wind (U at x + Δx/2, V at y + Δy/2, W at T-points) at each particle: trilinear in space, piecewise-linear in time, edge value held beyond the walls. Build it from a `MetField` (`gridded_wind_from_met`) or a `PrescribedWindField` (`gridded_wind_from_prescribed`) and pass it as `wind` to `integrate_particles`, `simulate_lagrangian` or `compute_footprint`. **Tier II.**

::: plumax.lagrangian.wind_interp
