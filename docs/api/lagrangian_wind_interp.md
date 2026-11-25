# Gridded wind

A wind is a tracerax velocity, `wind(t, position)`, with one velocity per particle. `wind_from_speed_direction` makes a steady, uniform one. `gridded_wind_from_met` and `gridded_wind_from_prescribed` put a `MetField` or a `PrescribedWindField` on the `les_fvm` C-grid (U at x + Δx/2, V at y + Δy/2, W at T-points) as a `tracerax.velocity.GriddedVelocity`: trilinear in space, piecewise-linear in time, edge value held beyond the walls. `tracerax.velocity.uniform` and `uniform_in_space` make the others. Pass a wind as `wind` to `simulate_lagrangian` or `compute_footprint`, or inside a `tracerax.advect.Langevin` to `tracerax.advect.integrate`. **Tier II.**

::: plumax.lagrangian.wind_interp
