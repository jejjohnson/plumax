# Footprint emulator

A trained network that maps a met context and a receptor position to the receptor's footprint row (`FootprintEmulator`), the particle-model training set it learns from (`generate_training_set`), and its calibration against the particle model: the relative Frobenius error (`emulator_relative_error`) and the probed source–receptor check through vardax (`assert_emulator_calibrated`, the `assim` extra). `emulated_footprint` returns the `(obs, cell)` matrix that `linear_gaussian_inversion` takes, so emulator-based inference is that inversion with the emulated footprint. **Tier II, Steps 3–4.**

::: plumax.lagrangian.emulator
