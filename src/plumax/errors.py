"""plumax's error hierarchy (see :mod:`plumax._src.errors`)."""

from __future__ import annotations

from plumax._src.errors import (
    ControlBlockError,
    ConvergenceError,
    DomainError,
    GridMismatchError,
    MissingExtraError,
    MissingVariableError,
    PayloadError,
    PlumaxError,
    ScreeningError,
    TimeAxisError,
    UnitsError,
)


__all__ = [
    "ControlBlockError",
    "ConvergenceError",
    "DomainError",
    "GridMismatchError",
    "MissingExtraError",
    "MissingVariableError",
    "PayloadError",
    "PlumaxError",
    "ScreeningError",
    "TimeAxisError",
    "UnitsError",
]
