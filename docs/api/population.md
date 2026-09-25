# Source population

Source population and forecasting: catalogues, size distributions, and point-process models. **Tier V.**

::: plumax.population
    options:
      members:
        - EmissionCatalog
        - EmissionEvent
        - PerEventPosterior
        - event_from_posterior
        - SizeDistributionPosterior
        - fit_lognormal_size_distribution
        - PoissonRatePosterior
        - fit_poisson_rate
        - InhomogeneousIntensityPosterior
        - fit_inhomogeneous_intensity

The package also re-exports the intensity and probability-of-detection model classes, the missing-mass simulator, and (lazily) the NumPyro fitter. They are documented below, from the submodules that define them.

## `intensity` — source intensities λ(t)

::: plumax.population.intensity
    options:
      show_root_heading: false
      heading_level: 3

## `pod` — probability of detection P_d(·)

"POD" here is **probability of detection**, not proper orthogonal decomposition.

::: plumax.population.pod
    options:
      show_root_heading: false
      heading_level: 3

## `paradox` — missing-mass Monte Carlo

::: plumax.population.paradox
    options:
      show_root_heading: false
      heading_level: 3

## `fitting` — POD-modified power-law NUTS fit

::: plumax.population.fitting
    options:
      show_root_heading: false
      heading_level: 3
