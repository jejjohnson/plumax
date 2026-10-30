# Multi-source

Masked-K multi-source plumes: up to K_max source slots, each with an emission rate, a location and an on/off indicator. `MaskedSources` evaluates the summed field and `MultiSourcePosterior` turns posterior draws into a posterior over the active count and per-source summaries for the population catalogue. The NumPyro model and runner are `multi_source_plume_model` and `infer_multi_source` in [Gaussian plume](gauss_plume.md). **Tier I**, reused by the Tier IV coupled version.

::: plumax.gauss_plume.multi_source
