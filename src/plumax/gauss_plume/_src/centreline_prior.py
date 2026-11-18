"""Gaussian-process prior on a plume's centreline, pinned to zero at the source.

The centreline offsets m = (m₁, …, m_K) at the knots d_k = k Δ_d of a
:class:`~plumax.gauss_plume.ColumnPlume` get the prior m ~ N(0, K_DD), with
[K_DD]_jk = k₀(d_j, d_k) a kernel conditioned on m(0) = 0. The default kernel
is the Matérn-3/2; a random walk from the source is the one-parameter
alternative.
"""

from __future__ import annotations

import math
from typing import Literal

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from plumax._src.errors import ParameterError


#: √3, the scale of the Matérn-3/2 kernel.
_SQRT_THREE = math.sqrt(3.0)

_KINDS = ("matern32", "random_walk")


class CentrelinePrior(eqx.Module):
    """Prior covariance of the centreline knots.

    [K_DD]_jk = k₀(d_j, d_k), with k₀(d, d') = k(d, d') − k(d, 0) k(0, d') / k(0, 0)
    and k the Matérn-3/2 kernel, k(d, d') = σ_m² (1 + √3 r / ℓ) exp(−√3 r / ℓ),
    r = |d − d'|. The marginal variance is σ_m² (1 − ρ(d)²), with
    ρ(d) = (1 + √3 d / ℓ) exp(−√3 d / ℓ): zero at the source and σ_m² far
    downwind. The random walk k(d, d') = κ² min(d, d'), κ² = σ_m² / ℓ, is zero
    at the source already and needs no conditioning.

    Algorithm:
        1. Evaluate the kernel between the knots, and between the knots
           and the source at d = 0.
        2. Subtract the rank-one term that conditions on m(0) = 0.
        3. Add a jitter of ``jitter · amplitude²`` to the diagonal.

    Attributes:
        length_scale: Length scale ℓ [m].
        amplitude: Standard deviation σ_m [m] far from the source.
        kind: ``"matern32"``, or ``"random_walk"`` for k(d, d') = κ² min(d, d')
            with κ² = σ_m² / ℓ.
        jitter: Relative diagonal jitter.

    Raises:
        ParameterError: If ``length_scale`` or ``amplitude`` is not positive,
            ``jitter`` is negative, or ``kind`` is unknown.

    Examples:
        The standard deviation grows from the source and levels off:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> knot_index = jnp.arange(1, 34)
        >>> knot_distance = 180.0 * knot_index
        >>> prior = CentrelinePrior(length_scale=1000.0, amplitude=300.0)
        >>> covariance = prior(knot_distance)
        >>> covariance.shape
        (33, 33)
        >>> std = jnp.sqrt(jnp.diagonal(covariance))
        >>> bool(jnp.allclose(std[0], 84.0, atol=1.0))
        True
        >>> bool(std[-1] > 290.0)
        True

        The matrix is symmetric:

        >>> transposed = einx.id("row col -> col row", covariance)
        >>> bool(jnp.allclose(covariance, transposed))
        True

        A random walk's variance grows as κ² d without limit:

        >>> walk = CentrelinePrior(length_scale=1000.0, kind="random_walk")
        >>> variance = jnp.diagonal(walk(jnp.array([1000.0, 4000.0])))
        >>> bool(jnp.allclose(variance, jnp.array([300.0, 1200.0]) * 300.0))
        True

    References:
        Rasmussen and Williams (2006), Gaussian Processes for Machine Learning,
        ch. 4.
    """

    length_scale: float = 1000.0
    amplitude: float = 300.0
    kind: Literal["matern32", "random_walk"] = "matern32"
    jitter: float = 1e-8

    def __check_init__(self) -> None:
        if self.length_scale <= 0.0:
            raise ParameterError(f"length_scale must be > 0, got {self.length_scale}")
        if self.amplitude <= 0.0:
            raise ParameterError(f"amplitude must be > 0, got {self.amplitude}")
        if self.jitter < 0.0:
            raise ParameterError(f"jitter must be ≥ 0, got {self.jitter}")
        if self.kind not in _KINDS:
            raise ParameterError(
                f"kind must be one of {list(_KINDS)}, got {self.kind!r}"
            )

    def __call__(
        self, knot_distance: Float[Array, " knot"]
    ) -> Float[Array, "knot knot"]:
        knot_distance = jnp.asarray(knot_distance)
        # σ_m²
        source_variance = self.amplitude**2
        if self.kind == "random_walk":
            # κ² = σ_m² / ℓ
            increment = source_variance / self.length_scale
            # min(d_j, d_k)
            shared = einx.minimum("row, col -> row col", knot_distance, knot_distance)
            # k_rw(d_j, d_k) = κ² min(d_j, d_k), already zero at the source
            conditioned = increment * shared
        else:
            # --- kernel between the knots ---
            # r_jk = |d_j − d_k|
            difference = einx.subtract(
                "row, col -> row col", knot_distance, knot_distance
            )
            separation = jnp.abs(difference)
            # k(d_j, d_k) = σ_m² (1 + √3 r / ℓ) exp(−√3 r / ℓ)
            gram_knots = self._matern(separation)

            # --- condition on m(0) = 0 ---
            # k(d_j, 0)
            gram_source = self._matern(jnp.abs(knot_distance))
            # k(d_j, 0) k(0, d_k) / k(0, 0)
            outer = einx.multiply("row, col -> row col", gram_source, gram_source)
            correction = outer / source_variance
            # k₀ = k − k(·, 0) k(0, ·) / k(0, 0)
            conditioned = gram_knots - correction

        # --- jitter ---
        n_knots = knot_distance.shape[0]
        identity = jnp.eye(n_knots, dtype=conditioned.dtype)
        # K_DD = k₀ + jitter σ_m² I
        covariance = conditioned + self.jitter * source_variance * identity
        return covariance

    def _matern(self, separation: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        # √3 r / ℓ
        scaled = _SQRT_THREE * separation / self.length_scale
        # σ_m² (1 + √3 r / ℓ) exp(−√3 r / ℓ)
        return self.amplitude**2 * (1.0 + scaled) * jnp.exp(-scaled)


__all__ = ["CentrelinePrior"]
