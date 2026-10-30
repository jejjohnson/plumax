"""Per-instrument quality flags turned into the mask and the inflated R of the fusion.

Each pixel carries bit flags (cloud, aerosol, failed convergence, …). A
:class:`QualityPolicy` names the flags it rejects and what to do with a
rejected pixel:

- **mask** (the design's default): the pixel drops out of the likelihood,
  m_i = 0, so it contributes ½ m_i (y_i − h_i)² / R_ii = 0;
- **inflate**: the pixel stays with its variance scaled, R_ii ← κ R_ii with
  κ = ``inflate_factor``, which tends to the mask as κ → ∞.

:func:`apply_quality` writes the policy into an ``Instrument`` (its ``mask``
and ``variance_inflation``), :func:`contribution_report` counts the pixels
each instrument contributes at full weight, and :func:`stack_observations`
hands every instrument's observations, mask and variance to vardax as one
``Batch1D`` and one diagonal R. Masked observations are filled with zeros:
vardax multiplies the residual by the mask, and a NaN there would still
poison the cost (jejjohnson/vardax#147).
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING

import einx
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from plumax._src.errors import ParameterError
from plumax._vardax import require_vardax


if TYPE_CHECKING:
    import vardax

    from plumax.coupled.instrument import Instrument


class QualityFlag(enum.IntFlag):
    """Per-pixel quality bits, ``0`` = clear.

    The bits of the planned radtrax retrieval flags (jejjohnson/radtrax#110,
    which supersedes plumax#97). Any integer bitmask works; these name the
    common ones.
    """

    CLEAR = 0
    CLOUD = 1
    AOD = 2
    GEOMETRY = 4
    CHI2 = 8
    NOT_CONVERGED = 16
    SATURATED = 32


class FlagPolicy(enum.StrEnum):
    """What a rejected pixel becomes: dropped, or kept with an inflated variance."""

    MASK = "mask"
    INFLATE = "inflate"


@dataclasses.dataclass(frozen=True)
class QualityPolicy:
    """Which flags reject a pixel, and how a rejected pixel enters the fusion.

    Attributes:
        policy: ``MASK`` drops rejected pixels; ``INFLATE`` scales their
            variance by ``inflate_factor``.
        inflate_factor: Variance factor κ ≥ 1 for ``INFLATE``.
        reject: Bitmask of the flags that reject a pixel.

    Raises:
        ParameterError: If ``inflate_factor`` is below 1 or ``reject`` is
            negative.
    """

    policy: FlagPolicy = FlagPolicy.MASK
    inflate_factor: float = 10.0
    reject: int = QualityFlag.CLOUD | QualityFlag.NOT_CONVERGED

    def __post_init__(self) -> None:
        if not self.inflate_factor >= 1.0:
            raise ParameterError(
                f"QualityPolicy: `inflate_factor` must be ≥ 1, got {self.inflate_factor}."
            )
        if int(self.reject) < 0:
            raise ParameterError(
                f"QualityPolicy: `reject` must be a non-negative bitmask, got "
                f"{self.reject}."
            )

    def rejected(self, flags: Int[Array, " obs"]) -> Bool[Array, " obs"]:
        """Whether each pixel carries a rejecting flag, (flags & reject) ≠ 0."""
        hits = jnp.bitwise_and(jnp.asarray(flags), int(self.reject))
        return hits != 0


def apply_quality(
    instrument: Instrument,
    flags: Int[Array, " obs"] | None = None,
    policy: QualityPolicy | None = None,
) -> Instrument:
    """The instrument with its flags and the policy's mask or inflation set.

    m_i = 0 for a rejected pixel under ``MASK``; R_ii ← κ R_ii under
    ``INFLATE``. The geometry, kernel and error budget are unchanged, so the
    coupled forward's responses stay valid. The result depends only on the
    flags and the policy, so applying it twice is the same as once.

    Args:
        instrument: The instrument to screen.
        flags: Per-pixel bit flags. Default: the instrument's own ``flags``
            (all clear when it has none).
        policy: The quality policy. Default ``QualityPolicy()`` (mask cloud
            and failed convergence).

    Returns:
        A copy of ``instrument`` with ``flags``, ``mask`` and
        ``variance_inflation`` set.

    Examples:
        Mask the cloudy pixel of three:

        >>> import jax.numpy as jnp
        >>> from plumax.coupled import Instrument
        >>> instrument = Instrument("A", jnp.zeros((3, 2)))
        >>> flags = jnp.array([0, QualityFlag.CLOUD, QualityFlag.AOD])
        >>> apply_quality(instrument, flags).observation_mask().tolist()
        [1.0, 0.0, 1.0]

        Inflate instead, by 100:

        >>> inflate = QualityPolicy(FlagPolicy.INFLATE, inflate_factor=100.0)
        >>> apply_quality(instrument, flags, inflate).observation_variance().tolist()
        [1.0, 100.0, 1.0]
    """
    policy = QualityPolicy() if policy is None else policy
    if flags is None:
        flags = instrument.flags
    if flags is None:
        flags = jnp.zeros(instrument.n_obs, dtype=int)
    rejected = policy.rejected(flags)
    if policy.policy is FlagPolicy.MASK:
        # m_i = 0 for a rejected pixel
        mask = jnp.where(rejected, 0.0, 1.0)
        return dataclasses.replace(
            instrument, flags=jnp.asarray(flags), mask=mask, variance_inflation=None
        )
    # κ_i = κ for a rejected pixel, 1 otherwise
    inflation = jnp.where(rejected, policy.inflate_factor, 1.0)
    return dataclasses.replace(
        instrument, flags=jnp.asarray(flags), mask=None, variance_inflation=inflation
    )


def full_weight_pixels(instrument: Instrument) -> Int[Array, ""]:
    """Pixels used at full weight: unmasked and not inflated."""
    mask = instrument.observation_mask()
    if instrument.variance_inflation is not None:
        inflation = jnp.asarray(instrument.variance_inflation, dtype=float)
        mask = jnp.where(inflation > 1.0, 0.0, mask)
    return jnp.sum(mask > 0.0)


def contribution_report(instruments: Sequence[Instrument]) -> dict[str, int]:
    """Pixels each instrument contributes at full weight, by name.

    Masked pixels and inflated pixels both count as not contributing in full,
    so the report is the effective per-instrument observation count the design
    asks to keep auditable.

    Args:
        instruments: The instruments of a fusion.

    Returns:
        ``{name: pixel count}``.

    Raises:
        ValueError: If two instruments share a name.

    Examples:
        >>> import jax.numpy as jnp
        >>> from plumax.coupled import Instrument
        >>> clear = Instrument("A", jnp.zeros((2, 2)))
        >>> cloudy = apply_quality(
        ...     Instrument("B", jnp.zeros((2, 2))), jnp.full(2, QualityFlag.CLOUD)
        ... )
        >>> contribution_report([clear, cloudy])
        {'A': 2, 'B': 0}
    """
    names = [instrument.name for instrument in instruments]
    if len(set(names)) != len(names):
        raise ValueError(f"contribution_report: instrument names repeat: {names}.")
    return {
        instrument.name: int(full_weight_pixels(instrument))
        for instrument in instruments
    }


class StackedObservations(eqx.Module):
    """Every instrument's observations, mask and variance on one stacked axis.

    The form vardax takes: y with masked pixels filled by zeros, the mask m and
    the diagonal of R (inflation included), instrument after instrument.

    Attributes:
        values: Observations y, zero where masked.
        mask: Observation mask m, ``1`` = used.
        variance: Diagonal of the observation-error covariance R.
    """

    values: Float[Array, " obs"]
    mask: Float[Array, " obs"]
    variance: Float[Array, " obs"]

    def batch(self) -> vardax.Batch1D:
        """One-window ``vardax.Batch1D`` of shape ``(1, 1, D_y)``.

        Raises:
            ImportError: Without the ``assim`` extra.
        """
        vdx = require_vardax("StackedObservations.batch")
        return vdx.Batch1D(
            input=einx.rearrange("obs -> 1 1 obs", self.values),
            mask=einx.rearrange("obs -> 1 1 obs", self.mask),
        )

    def obs_cov_op(self) -> lx.DiagonalLinearOperator:
        """R as a diagonal operator on a ``(1, D_y)`` observation row.

        A diagonal operator rather than a scaled identity, which vardax
        mis-solves (jejjohnson/vardax#146).
        """
        return lx.DiagonalLinearOperator(einx.rearrange("obs -> 1 obs", self.variance))


def stack_observations(
    instruments: Sequence[Instrument],
    observations: Sequence[Float[Array, " _"]],
) -> StackedObservations:
    """Stack the instruments' observations, masks and variances for vardax.

    Args:
        instruments: The instruments, with any quality policy already applied.
        observations: One observation vector per instrument, aligned with
            ``instruments``. Masked entries may hold anything, NaN included.

    Returns:
        The stacked observations, masked entries set to zero.

    Raises:
        ValueError: On a count or shape mismatch.

    Examples:
        A NaN under the mask becomes a zero:

        >>> import jax.numpy as jnp
        >>> from plumax.coupled import Instrument
        >>> instrument = Instrument("A", jnp.zeros((2, 2)), mask=jnp.array([1.0, 0.0]))
        >>> stacked = stack_observations([instrument], [jnp.array([3.0, jnp.nan])])
        >>> stacked.values.tolist(), stacked.mask.tolist()
        ([3.0, 0.0], [1.0, 0.0])
    """
    if len(observations) != len(instruments):
        raise ValueError(
            f"stack_observations: got {len(observations)} observation vectors for "
            f"{len(instruments)} instruments."
        )
    values, masks, variances = [], [], []
    for instrument, observation in zip(instruments, observations, strict=True):
        if np.shape(observation) != (instrument.n_obs,):
            raise ValueError(
                f"stack_observations: observation for {instrument.name!r} must be "
                f"({instrument.n_obs},); got {np.shape(observation)}."
            )
        mask = instrument.observation_mask()
        # y_i = 0 where m_i = 0, so a NaN never reaches the cost
        filled = jnp.where(mask > 0.0, jnp.asarray(observation, dtype=float), 0.0)
        values.append(filled)
        masks.append(mask)
        variances.append(instrument.observation_variance())
    return StackedObservations(
        values=jnp.concatenate(values),
        mask=jnp.concatenate(masks),
        variance=jnp.concatenate(variances),
    )


__all__ = [
    "FlagPolicy",
    "QualityFlag",
    "QualityPolicy",
    "StackedObservations",
    "apply_quality",
    "contribution_report",
    "full_weight_pixels",
    "stack_observations",
]
