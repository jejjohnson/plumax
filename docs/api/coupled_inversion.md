# Nonlinear inversion

The end-to-end coupled inversion over the emission rate (Gaussian or lognormal), source position, wind, background and per-instrument biases, through each instrument's averaging kernel or a JAX radiance operator. plumax builds the forward, the prior and the quality masks; vardax solves it (`ThreeDVar` with L-BFGS in the whitened control) and gives the Laplace posterior (`LaplaceCovariance`). Needs the `assim` extra. **Tier IV.**

::: plumax.coupled.inversion
