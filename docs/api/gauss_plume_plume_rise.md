# Plume rise

Briggs plume rise for buoyant and momentum sources: `BriggsPlumeRise` returns Δh(x) from a stack's exit velocity, exit temperature and diameter, and `effective_height` gives H_eff = H_stack + Δh(x). Pass a `BriggsPlumeRise` as `plume_rise=` to `plume_concentration` or `simulate_plume` to release the plume at its effective height, differentiably in the stack parameters. **Tier I.**

::: plumax.gauss_plume.plume_rise
