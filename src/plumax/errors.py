"""plumax's error hierarchy (see :mod:`plumax._src.errors`)."""

from __future__ import annotations

from plumax._src.errors import (
    DomainError,
    GridMismatchError,
    MissingExtraError,
    MissingVariableError,
    PlumaxError,
    ScreeningError,
    TimeAxisError,
    UnitsError,
)


__all__ = [
    "DomainError",
    "GridMismatchError",
    "MissingExtraError",
    "MissingVariableError",
    "PlumaxError",
    "ScreeningError",
    "TimeAxisError",
    "UnitsError",
]
