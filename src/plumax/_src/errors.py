"""plumax's error hierarchy: one base class, subclasses named by what failed.

Each subclass also inherits the builtin it refines (``ValueError``,
``KeyError``, ``ImportError``), so callers catching the builtin keep working.
"""

from __future__ import annotations


class PlumaxError(Exception):
    """Base class of every error plumax raises on purpose."""


class MissingExtraError(PlumaxError, ImportError):
    """An optional dependency is missing; the message names the extra."""


class DomainError(PlumaxError, ValueError):
    """A point falls outside the grid, raster or swath it must be sampled from."""


class GridMismatchError(PlumaxError, ValueError):
    """Two inputs that must share a grid or a time axis do not."""


class TimeAxisError(PlumaxError, ValueError):
    """Timestamps are empty, unsorted or select nothing."""


class MissingVariableError(PlumaxError, KeyError):
    """A file lacks a variable the reader needs."""


class ScreeningError(PlumaxError, ValueError):
    """No observation survives quality, NaN and region screening."""


class UnitsError(PlumaxError, ValueError):
    """A quantity is missing or in units the reader cannot convert."""
