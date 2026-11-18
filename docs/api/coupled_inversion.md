# Nonlinear inversion

The end-to-end coupled inversion over the emission rate (Gaussian or lognormal), whichever blocks the transport backend declares (the source position and wind of the Tier I plume, the geometry of the column plume), the background and per-instrument biases, through each instrument's averaging kernel or a JAX radiance operator. plumax builds the forward, the prior and the quality masks; vardax solves it (`ThreeDVar` with L-BFGS in the whitened control) and gives the Laplace posterior (`LaplaceCovariance`). A column-plume solve retrieves the line density $q = Q / U$, and `CoupledResult.with_wind` applies the wind after it. Needs the `assim` extra. **Tier IV.**

::: plumax.coupled.inversion

## Multistart

The image-fit cost has several minima in the source position, so `solve_coupled_multistart` runs `solve_coupled(problem, initial=...)` from each of several starts, with the same prior, and keeps the converged start with the lowest cost.

::: plumax.coupled.multistart
    options:
      show_root_heading: false
      heading_level: 3

## Fit diagnostics

`fit_diagnostics` reports what the data determined in a solved problem: the Fisher information at the MAP, the prior-free Cramér–Rao bound of each parameter, the share of each parameter's prior variance the data removed, the degrees of freedom for signal, how far the prior was pulled, and each instrument's residual statistics.

::: plumax.coupled.diagnostics
    options:
      show_root_heading: false
      heading_level: 3
