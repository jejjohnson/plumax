# Nonlinear inversion

The end-to-end coupled inversion over the emission rate (Gaussian or lognormal), whichever blocks the transport backend declares (the source position and wind of the Tier I plume, the geometry of the column plume), the background and per-instrument biases, through each instrument's averaging kernel or a JAX radiance operator. plumax builds the forward, the prior and the quality masks; vardax solves it (`ThreeDVar` with L-BFGS in the whitened control) and gives the Laplace posterior (`LaplaceCovariance`). A column-plume solve retrieves the line density $q = Q / U$, and `CoupledResult.with_wind` applies the wind after it. Needs the `assim` extra. **Tier IV.**

::: plumax.coupled.inversion
