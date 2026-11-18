"""The base of a transport backend whose response depends on fitted parameters.

It lives outside ``plumax.coupled`` because the backends that subclass it sit
in the tiers (``gauss_plume.ColumnPlumeSource``) that ``coupled`` itself
imports; ``plumax.coupled.backends`` re-exports it.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING

import equinox as eqx
from jaxtyping import Array, Float


if TYPE_CHECKING:
    from plumax.coupled.instrument import Instrument


class AbstractControlledSource(eqx.Module):
    """A transport backend whose response depends on fitted parameters.

    r_inst(θ) = A_inst ⊙ r(s_inst; θ), with θ the blocks this backend declares.

    The control arrives as an argument and is never stored: the fields hold
    the values the response takes when a block is absent from the control.
    The inversion fits the declared blocks, one of the emission blocks, the
    background and the per-instrument biases.

    A subclass gives ``background`` as a field: ``eqx.AbstractVar`` cannot
    be read from the postponed annotations every plumax module uses, so the
    base declares it as an abstract property.
    """

    @property
    @abc.abstractmethod
    def background(self) -> float:
        """Additive background column Ω_bg, in the observation's units."""

    @property
    @abc.abstractmethod
    def control_blocks(self) -> Mapping[str, tuple[int, ...]]:
        """Block name → shape, for every parameter this backend can fit."""

    @property
    @abc.abstractmethod
    def emission_blocks(self) -> tuple[str, ...]:
        """Names of the blocks that scale the response; a prior holds one.

        A name that starts with ``log_`` is the logarithm of the scale, so the
        inversion multiplies the response by its exponential.
        """

    @abc.abstractmethod
    def column_response(self, instrument: Instrument) -> Float[Array, " obs"]:
        """The unit response at this backend's own parameters.

        Args:
            instrument: The observing instrument (receptors and kernel).

        Returns:
            The response r_inst, one value per receptor.
        """

    @abc.abstractmethod
    def response_at(
        self,
        instrument: Instrument,
        control: Mapping[str, Float[Array, "*shape"]],
    ) -> Float[Array, " obs"]:
        """The unit response with the declared blocks taken from ``control``.

        Blocks missing from ``control`` keep this backend's own values, and
        names it does not declare are ignored.

        Args:
            instrument: The observing instrument (receptors and kernel).
            control: The control, by block.

        Returns:
            The response r_inst(θ), one value per receptor.
        """


__all__ = ["AbstractControlledSource"]
