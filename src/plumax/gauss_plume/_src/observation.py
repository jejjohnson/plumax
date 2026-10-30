"""The column + averaging-kernel observation shared by every transport tier.

A satellite sees a column, not a 3-D field. Every tier maps its concentration
to the observation with the same weighted column sum,

    y_i = A_i Σ_k h_k c_k(x_i, y_i),

with ``c_k`` the concentration on level ``k`` at receptor ``i``, ``h_k`` the
level weights and ``A_i`` the receptor's scalar averaging kernel (the L2
default; see ``coupled.Instrument``). :class:`ColumnObservation` holds
``(h, A, receptors)`` and applies it to a gridded field (Tiers II and III) or to
per-receptor profiles (the Tier I plume, through
``coupled.forward.column_response``).

Units, once for all tiers: with ``c`` in kg m⁻³ and ``h`` the layer
thicknesses in m, ``y`` is the column-mass enhancement ΔΩ in kg m⁻². That is
the unit ``plumax.obs`` converts every L2 product to (ppb and ppm·m included),
so model and observation meet there and no tier converts to ppb. Per unit
emission (the Tier I response) ``c`` is in s m⁻³ and ``y`` in s m⁻². The
averaging kernel is dimensionless. Normalised weights (``Σ_k h_k = 1``, e.g.
pressure weights ``Δp_k / p_s``) give the weighted column average instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from plumax._src.errors import DomainError


if TYPE_CHECKING:
    from plumax.coupled.instrument import Instrument
    from plumax.les_fvm.grid import PlumeGrid3D


class ColumnObservation(eqx.Module):
    """Averaging-kernel-weighted column at a set of receptors.

    y_i = A_i Σ_k h_k c_{k,i}.

    Attributes:
        layer_weights: Level weights h_k, the layer thicknesses [m] for a
            column integral.
        averaging_kernel: Per-receptor scalar averaging kernel A_i. ``None``
            is the identity kernel.
        receptor_index: ``(iy, ix)`` interior-cell index of each receptor on
            the field :meth:`from_field` takes. ``None`` observes every
            column, flattened row-major over ``(y, x)``.

    Examples:
        Two levels 10 m thick, every column of a uniform field:

        >>> import jax.numpy as jnp
        >>> observation = ColumnObservation(layer_weights=jnp.full(2, 10.0))
        >>> observation.from_field(jnp.ones((2, 3, 4))).shape
        (12,)

        Two receptors with an averaging kernel:

        >>> observation = ColumnObservation(
        ...     layer_weights=jnp.full(2, 10.0),
        ...     averaging_kernel=jnp.array([1.0, 0.5]),
        ...     receptor_index=jnp.array([[0, 0], [2, 3]]),
        ... )
        >>> observation.from_field(jnp.ones((2, 3, 4))).tolist()
        [20.0, 10.0]

        Trapezoid weights for profiles on irregular levels:

        >>> levels = jnp.array([0.0, 10.0, 30.0])
        >>> ColumnObservation.trapezoid(levels).layer_weights.tolist()
        [5.0, 15.0, 10.0]
    """

    layer_weights: Float[Array, " level"]
    averaging_kernel: Float[Array, " obs"] | None = None
    receptor_index: Int[Array, "obs 2"] | None = None

    @classmethod
    def trapezoid(
        cls,
        levels: Float[Array, " level"],
        averaging_kernel: Float[Array, " obs"] | None = None,
    ) -> ColumnObservation:
        """Column over profiles on strictly increasing levels, trapezoid rule.

        h_0 = Δz_0 / 2,  h_k = (Δz_{k-1} + Δz_k) / 2,  h_{K-1} = Δz_{K-2} / 2,

        so Σ_k h_k c_k is ``jnp.trapezoid(c, x=levels)``.

        Args:
            levels: Level heights z_k [m].
            averaging_kernel: Per-receptor scalar averaging kernel A_i.

        Returns:
            The observation, with no receptor index (it takes profiles).
        """
        # Δz_k = z_{k+1} − z_k
        spacing = jnp.diff(levels)
        # h_k = (Δz_{k-1} + Δz_k) / 2, with Δz_{-1} = Δz_{K-1} = 0
        layer_weights = 0.5 * (jnp.pad(spacing, (0, 1)) + jnp.pad(spacing, (1, 0)))
        return cls(layer_weights=layer_weights, averaging_kernel=averaging_kernel)

    @classmethod
    def for_grid(
        cls,
        plume_grid: PlumeGrid3D,
        *,
        instrument: Instrument | None = None,
        receptor_index: Int[Array, "obs 2"] | None = None,
    ) -> ColumnObservation:
        """Column over a ``les_fvm`` analysis grid, midpoint rule (h_k = Δz).

        An instrument places each receptor in the T-cell that contains it and
        brings its averaging kernel. Without one, ``receptor_index`` (or every
        column) is observed with the identity kernel.

        Args:
            plume_grid: The analysis grid the field lives on.
            instrument: Receptors ``(x, y)`` [m] and their averaging kernel.
            receptor_index: ``(iy, ix)`` interior-cell indices, used when no
                instrument is given.

        Returns:
            The observation for interior fields of shape ``(nz, ny, nx)``.

        Raises:
            DomainError: If a receptor lies outside the grid.
            ValueError: If both ``instrument`` and ``receptor_index`` are given.
        """
        nz = plume_grid.interior_shape[0]
        layer_weights = jnp.full(nz, plume_grid.dz, dtype=plume_grid.z.dtype)
        if instrument is None:
            return cls(layer_weights=layer_weights, receptor_index=receptor_index)
        if receptor_index is not None:
            raise ValueError(
                "ColumnObservation.for_grid: pass `instrument` or "
                "`receptor_index`, not both."
            )
        receptors = np.asarray(instrument.receptors, dtype=float)
        ix = _containing_cell(receptors[:, 0], np.asarray(plume_grid.x), "x")
        iy = _containing_cell(receptors[:, 1], np.asarray(plume_grid.y), "y")
        index = jnp.asarray(np.column_stack([iy, ix]))
        return cls(
            layer_weights=layer_weights,
            averaging_kernel=instrument.ak,
            receptor_index=index,
        )

    def column(self, profiles: Float[Array, "obs level"]) -> Float[Array, " obs"]:
        """Apply the observation to per-receptor profiles c_{k,i}.

        Args:
            profiles: Concentration at each receptor and level.

        Returns:
            The observation y_i.
        """
        # Σ_k h_k c_{k,i}
        column = einx.dot("obs level, level -> obs", profiles, self.layer_weights)
        if self.averaging_kernel is None:
            return column
        # y_i = A_i Σ_k h_k c_{k,i}
        return self.averaging_kernel * column

    def from_field(
        self, concentration: Float[Array, "level y x"]
    ) -> Float[Array, " obs"]:
        """Apply the observation to an interior field on the analysis grid.

        Args:
            concentration: Interior concentration c, ghost rings stripped.

        Returns:
            The observation y_i at each receptor.
        """
        if self.receptor_index is None:
            profiles = einx.rearrange("level y x -> (y x) level", concentration)
        else:
            iy = self.receptor_index[:, 0]
            ix = self.receptor_index[:, 1]
            # c_{k,i} = c[k, iy_i, ix_i]
            gathered = concentration[:, iy, ix]
            profiles = einx.rearrange("level obs -> obs level", gathered)
        return self.column(profiles)


def _containing_cell(
    coordinate: Float[np.ndarray, " obs"],
    centres: Float[np.ndarray, " cell"],
    axis: str,
) -> Int[np.ndarray, " obs"]:
    """Index of the uniform cell (centres ``centres``) holding each coordinate."""
    spacing = (centres[-1] - centres[0]) / max(centres.size - 1, 1)
    lower = centres[0] - 0.5 * spacing
    index = np.floor((coordinate - lower) / spacing).astype(int)
    outside = (index < 0) | (index >= centres.size)
    if outside.any():
        raise DomainError(
            f"ColumnObservation.for_grid: {int(outside.sum())} receptor(s) lie "
            f"outside the grid in {axis}."
        )
    return index
