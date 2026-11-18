"""plumax's error hierarchy (see :mod:`plumax._src.errors`)."""

from __future__ import annotations

from plumax._src.errors import (
    ControlBlockError,
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
