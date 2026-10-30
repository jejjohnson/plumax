# Quality flags

The per-instrument quality-flag aggregator. A `QualityPolicy` names the flags that reject a pixel and whether a rejected pixel is masked (the default) or kept with its variance inflated; `apply_quality` writes that into an `Instrument`, `contribution_report` counts each instrument's full-weight pixels, and `stack_observations` hands the observations, mask and diagonal R to vardax. **Tier IV.**

::: plumax.coupled.quality
