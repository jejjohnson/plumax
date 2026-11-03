"""Footprint emulator and its calibration (implementation in ``plumax.lagrangian._src.emulator``)."""

from __future__ import annotations

from plumax.lagrangian._src.emulator import (
    EmulatedFootprint,
    FootprintEmulator,
    FootprintSimulator,
    MetContextPrior,
    assert_emulator_calibrated,
    emulated_footprint,
    emulator_relative_error,
    footprint_dataset,
    init_footprint_emulator,
    train_footprint_emulator,
)


__all__ = [
    "EmulatedFootprint",
    "FootprintEmulator",
    "FootprintSimulator",
    "MetContextPrior",
    "assert_emulator_calibrated",
    "emulated_footprint",
    "emulator_relative_error",
    "footprint_dataset",
    "init_footprint_emulator",
    "train_footprint_emulator",
]
