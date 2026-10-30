"""Footprint emulator and its calibration (implementation in ``plumax.lagrangian._src.emulator``)."""

from __future__ import annotations

from plumax.lagrangian._src.emulator import (
    FootprintEmulator,
    FootprintTrainingSet,
    assert_emulator_calibrated,
    emulated_footprint,
    emulator_relative_error,
    generate_training_set,
    init_footprint_emulator,
    train_footprint_emulator,
)


__all__ = [
    "FootprintEmulator",
    "FootprintTrainingSet",
    "assert_emulator_calibrated",
    "emulated_footprint",
    "emulator_relative_error",
    "generate_training_set",
    "init_footprint_emulator",
    "train_footprint_emulator",
]
