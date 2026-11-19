"""The shared ``sample_priors`` of the intensity and POD classes.

A class lists its priors as ``_PRIORS``, one row per constructor field, in the
order the NumPyro sites are drawn: ``field: (site name, factory)``. The factory
takes ``numpyro.distributions`` and returns the prior, so NumPyro is imported
only when a prior is sampled.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar, TypeVar


__all__ = ["PriorTable", "PriorTableMixin"]

_T = TypeVar("_T", bound="PriorTableMixin")

#: ``field -> (NumPyro site name, factory(numpyro.distributions) -> distribution)``.
PriorTable = dict[str, tuple[str, Callable[[ModuleType], Any]]]


class PriorTableMixin:
    """Gives a module ``sample_priors`` from its ``_PRIORS`` table."""

    _PRIORS: ClassVar[PriorTable]

    @classmethod
    def sample_priors(cls: type[_T]) -> _T:
        """Draw the class's priors and build an instance from them.

        Draws one NumPyro site per entry of the class's ``_PRIORS`` table, in table
        order; the class docstring lists them.

        Returns:
            The class built from the sampled values.
        """
        import numpyro
        import numpyro.distributions as dist

        return cls(
            **{
                field: numpyro.sample(site, factory(dist))
                for field, (site, factory) in cls._PRIORS.items()
            }
        )
