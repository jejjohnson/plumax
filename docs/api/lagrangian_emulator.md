# Footprint emulator

A trained network that maps a met context and a receptor position to the receptor's footprint row (`FootprintEmulator`), its training on pipekit-train (the `emulate` extra: `FootprintSimulator` and `MetContextPrior` feed a cached `SimulationDataset` built by `footprint_dataset`, and `train_footprint_emulator` runs a `TrainingLoop` with the `MSE` loss and `EarlyStopping`), and its calibration against the particle model: the relative Frobenius error (`emulator_relative_error`) and the probed source–receptor check through vardax (`assert_emulator_calibrated`, the `assim` extra). `emulated_footprint` returns the `(obs, cell)` matrix that `linear_gaussian_inversion` takes, so emulator-based inference is that inversion with the emulated footprint. **Tier II, Steps 3–4.**

::: plumax.lagrangian.emulator
