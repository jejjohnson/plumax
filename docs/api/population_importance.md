# Importance correction

The importance-corrected TMTPP mark likelihood: per-event evidence in the POINT, GAUSSIAN and FULL regimes, which divides each event's per-event prior out of its posterior; the ESS and Pareto-k̂ diagnostics; log-space quadrature for the mark integrals and the thinned rate; and the NumPyro models and fit that use them (`mark_iw_model`, `tmtpp_iw_model`, `fit_size_distribution_iw`, in `plumax.population.fitting`). Needs the `population` extra. **Tier V.**

## `importance` — per-event evidence and ESS

::: plumax.population.importance
    options:
      show_root_heading: false
      heading_level: 3

## `integrate` — log-space quadrature

::: plumax.population.integrate
    options:
      show_root_heading: false
      heading_level: 3

## Importance-corrected models

::: plumax.population._src.fitting_iw
    options:
      show_root_heading: false
      heading_level: 3
