"""Monin–Obukhov-driven Pasquill–Gifford dispersion via Golder's (1972) classes.

Golder (1972) related the Pasquill–Gifford (PG) stability class to the Obukhov
length L and the roughness length z₀. Each boundary between neighbouring
classes is a curve L_k(z₀) = c_k / (ln z₀ − d_k), with z₀ in metres:

    boundary   c_k      d_k
    A/B        70.0     4.350
    B/C        85.2     0.502
    C/D        245.0    0.050
    D/E       −327.0    0.627
    E/F        −70.0    0.295

valid for 0.01 m ≤ z₀ ≤ 0.5 m (z₀ is clamped to that range). These are the
coefficients of ``pg_golder`` in EPA's MMIF v4.1.1 (``pasquill_gifford.f90``),
which follows the ``LSTAB`` routine of CTDMPLUS that CALPUFF also uses. In
terms of the inverse length, each boundary is linear in ln z₀,

    (1/L)_k = (ln z₀ − d_k) / c_k,

so the class is read off 1/L, which is finite (zero) at neutral, as
:class:`plumax.met.SurfaceLayer` stores it. A value on a boundary belongs to the
less stable class, as in MMIF. The class then picks plumax's existing
Briggs–PG curves (:data:`~plumax.gauss_plume.BRIGGS_DISPERSION_PARAMS`), so
σ is ordered stable ≤ neutral ≤ unstable by construction.

A class lookup has zero gradient in L. :func:`golder_stability_index` offers a
continuous class index instead, linear in 1/L between the boundaries; this is
a modelling choice of plumax, not part of Golder (1972) (see its docstring).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from plumax._src.errors import ParameterError
from plumax.gauss_plume.dispersion import (
    BRIGGS_DISPERSION_PARAMS,
    STABILITY_CLASSES,
    calculate_briggs_dispersion,
)
from plumax.met._src.surface_layer import SurfaceLayer


#: Roughness lengths [m] over which Golder's class boundaries are defined;
#: z₀ outside this range is clamped to it (as MMIF's ``pg_golder`` does).
GOLDER_Z0_RANGE: tuple[float, float] = (0.01, 0.5)

#: c_k [m] of the boundaries L_k = c_k / (ln z₀ − d_k), A/B … E/F.
GOLDER_BOUNDARY_SCALE: tuple[float, ...] = (70.0, 85.2, 245.0, -327.0, -70.0)

#: d_k [–] of the boundaries L_k = c_k / (ln z₀ − d_k), A/B … E/F.
GOLDER_BOUNDARY_OFFSET: tuple[float, ...] = (4.350, 0.502, 0.050, 0.627, 0.295)

# Continuous class index at the knots of golder_stability_index: the outer
# ends, the five boundaries (half-integers) and neutral (1/L = 0, class D).
_INDEX_KNOTS = (0.0, 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.0)

# Index of class D in STABILITY_CLASSES.
_NEUTRAL_INDEX = 3


def golder_class_boundaries(z0: float) -> Float[Array, " boundary"]:
    """Inverse Obukhov lengths at Golder's five PG class boundaries.

    (1/L)_k = (ln z₀ − d_k) / c_k for the A/B, B/C, C/D, D/E and E/F
    boundaries, with z₀ clamped to :data:`GOLDER_Z0_RANGE`.

    Args:
        z0: Roughness length z₀ > 0 [m].

    Returns:
        The five boundaries [m⁻¹], increasing from A/B (most negative) to E/F.

    Examples:
        At z₀ = 0.1 m the unstable boundaries sit at L ≈ −10.5, −30.4 and
        −104 m, the stable ones at L ≈ 112 and 26.9 m:

        >>> import jax.numpy as jnp
        >>> boundaries = golder_class_boundaries(0.1)
        >>> [round(float(1.0 / b), 1) for b in boundaries]
        [-10.5, -30.4, -104.1, 111.6, 26.9]

        z₀ beyond 0.5 m is clamped:

        >>> bool(jnp.allclose(golder_class_boundaries(2.0), golder_class_boundaries(0.5)))
        True
    """
    roughness = jnp.asarray(z0)
    roughness = eqx.error_if(roughness, roughness <= 0.0, "z0 must be positive")
    clamped = jnp.clip(roughness, *GOLDER_Z0_RANGE)
    log_z0 = jnp.log(clamped)
    scale = jnp.asarray(GOLDER_BOUNDARY_SCALE, dtype=log_z0.dtype)
    offset = jnp.asarray(GOLDER_BOUNDARY_OFFSET, dtype=log_z0.dtype)
    # (1/L)_k = (ln z₀ − d_k) / c_k
    return (log_z0 - offset) / scale


def golder_stability_class(
    inverse_obukhov_length: Float[Array, "*shape"] | float, z0: float
) -> Int[Array, "*shape"]:
    """Golder's (1972) PG class for 1/L and z₀, as an index 0 (A) … 5 (F).

    The class is the number of boundaries (1/L)_k strictly below 1/L, so a
    value on a boundary goes to the less stable class and 1/L = 0 is D.

    Args:
        inverse_obukhov_length: 1/L [m⁻¹]; zero is neutral, negative unstable.
        z0: Roughness length z₀ > 0 [m].

    Returns:
        The class index into :data:`~plumax.gauss_plume.STABILITY_CLASSES`.

    Examples:
        Over z₀ = 0.1 m, L = −20 m is B, neutral is D and L = 50 m is E:

        >>> import jax.numpy as jnp
        >>> inverse = jnp.array([-1.0 / 20.0, 0.0, 1.0 / 50.0])
        >>> golder_stability_class(inverse, 0.1).tolist()
        [1, 3, 4]
    """
    inverse_obukhov_length = jnp.asarray(inverse_obukhov_length)
    boundaries = golder_class_boundaries(z0)
    above = inverse_obukhov_length[..., None] > boundaries
    return jnp.sum(above, axis=-1)


def golder_stability_index(
    inverse_obukhov_length: Float[Array, "*shape"] | float, z0: float
) -> Float[Array, "*shape"]:
    """Continuous PG class index from 1/L and z₀, piecewise linear in 1/L.

    s(1/L) interpolates linearly through the knots (1/L, s) =
    ((1/L)_AB − Δ_A, 0), ((1/L)_AB, 0.5), ((1/L)_BC, 1.5), ((1/L)_CD, 2.5),
    (0, 3), ((1/L)_DE, 3.5), ((1/L)_EF, 4.5), ((1/L)_EF + Δ_F, 5), and is
    clamped to [0, 5] outside them, with Δ_A = ((1/L)_BC − (1/L)_AB)/2 and
    Δ_F = ((1/L)_EF − (1/L)_DE)/2 (the half-width of the neighbouring class).

    This is a plumax modelling choice, not part of Golder (1972): Golder gives
    only the class boundaries. Each boundary maps to the half-integer between
    its two classes, so ``round`` of the index (ties to the less stable class)
    is the hard class of :func:`golder_stability_class`, and neutral maps to
    exactly 3 (class D). Its derivative in 1/L is non-zero everywhere except
    deep inside A (1/L < (1/L)_AB − Δ_A) and F (1/L > (1/L)_EF + Δ_F).

    Args:
        inverse_obukhov_length: 1/L [m⁻¹]; zero is neutral, negative unstable.
        z0: Roughness length z₀ > 0 [m] (a scalar).

    Returns:
        The class index s ∈ [0, 5], 0 for A and 5 for F.

    Examples:
        Boundaries land on half-integers and neutral on D:

        >>> import jax.numpy as jnp
        >>> boundaries = golder_class_boundaries(0.1)
        >>> golder_stability_index(boundaries, 0.1).tolist()
        [0.5, 1.5, 2.5, 3.5, 4.5]
        >>> float(golder_stability_index(0.0, 0.1))
        3.0
    """
    inverse_obukhov_length = jnp.asarray(inverse_obukhov_length)
    boundaries = golder_class_boundaries(z0)
    # Δ_A = ((1/L)_BC − (1/L)_AB) / 2
    half_width_a = 0.5 * (boundaries[1] - boundaries[0])
    # Δ_F = ((1/L)_EF − (1/L)_DE) / 2
    half_width_f = 0.5 * (boundaries[4] - boundaries[3])
    neutral = jnp.zeros((1,), dtype=boundaries.dtype)
    knots = jnp.concatenate(
        [
            (boundaries[0] - half_width_a)[None],
            boundaries[:3],
            neutral,
            boundaries[3:],
            (boundaries[4] + half_width_f)[None],
        ]
    )
    index_knots = jnp.asarray(_INDEX_KNOTS, dtype=boundaries.dtype)
    return jnp.interp(inverse_obukhov_length, knots, index_knots)


class GolderDispersion(eqx.Module):
    """PG σ_y, σ_z with the class chosen from Monin–Obukhov scales (Golder 1972).

    σ_i(x) = σ_i^PG(x; class(1/L, z₀)), i ∈ {y, z}, with the Briggs–PG curves
    of :func:`~plumax.gauss_plume.calculate_briggs_dispersion`. With
    ``interpolate=True`` the class index s = :func:`golder_stability_index` is
    continuous and σ_i(x) = (1 − w) σ_i^PG(x; ⌊s⌋) + w σ_i^PG(x; ⌊s⌋ + 1),
    w = s − ⌊s⌋, which is differentiable in L (a plumax modelling choice, not
    part of Golder). Pass it as ``dispersion_params`` to
    :func:`~plumax.gauss_plume.plume_concentration` or as ``stability_class``
    to :func:`~plumax.gauss_plume.simulate_plume`.

    Attributes:
        inverse_obukhov_length: 1/L [m⁻¹] of the column; zero is neutral.
        z0: Roughness length z₀ > 0 [m]; clamped to :data:`GOLDER_Z0_RANGE`.
        interpolate: Blend neighbouring PG classes linearly in 1/L (default
            ``False``: the hard class lookup, as in Golder and MMIF).

    Raises:
        ParameterError: If ``z0`` is not positive.

    Examples:
        Neutral air reproduces PG class D exactly:

        >>> import jax.numpy as jnp
        >>> from plumax.gauss_plume import calculate_briggs_dispersion
        >>> from plumax.gauss_plume import get_dispersion_params
        >>> distance = jnp.array([100.0, 1000.0])
        >>> neutral = GolderDispersion(inverse_obukhov_length=0.0, z0=0.1)
        >>> sigma_y, sigma_z = neutral(distance)
        >>> pg_y, pg_z = calculate_briggs_dispersion(distance, get_dispersion_params("D"))
        >>> bool(jnp.allclose(sigma_y, pg_y) & jnp.allclose(sigma_z, pg_z))
        True

        The interpolated σ moves smoothly with L inside a class:

        >>> import jax
        >>> def sigma_z_at_1km(inverse_length):
        ...     dispersion = GolderDispersion(inverse_length, z0=0.1, interpolate=True)
        ...     return dispersion(jnp.array(1000.0))[1]
        >>> bool(jax.grad(sigma_z_at_1km)(-1.0 / 50.0) < 0.0)
        True

        Build it from a met column's surface-layer diagnosis:

        >>> from plumax.les_fvm.grid import make_grid
        >>> from plumax.met import MetField, MoninObukhov
        >>> grid = make_grid(
        ...     (0.0, 400.0, 4), (0.0, 400.0, 4), (0.0, 200.0, 4), dtype=jnp.float64
        ... )
        >>> shape = (1, 4, 4, 4)
        >>> field = MetField(
        ...     plume_grid=grid, times=jnp.zeros(1), u=jnp.full(shape, 5.0),
        ...     v=jnp.zeros(shape), w=jnp.zeros(shape),
        ...     temperature=jnp.full(shape, 300.0), pressure=jnp.full(shape, 1.0e5),
        ...     pbl_height=jnp.full((1, 4, 4), 800.0),
        ... )
        >>> layer = MoninObukhov()(field, z0=0.1, sensible_heat_flux=0.0)
        >>> GolderDispersion.from_surface_layer(layer, z0=0.1, it=0, iy=1, ix=2).z0
        0.1

    References:
        Golder, D. (1972). Relations among stability parameters in the surface
        layer. *Boundary-Layer Meteorology* 3, 47–58.
        US EPA (2024). MMIF v4.1.1, ``pasquill_gifford.f90`` (``pg_golder``).
    """

    inverse_obukhov_length: Float[Array, ""]
    z0: float = eqx.field(static=True)
    interpolate: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        inverse_obukhov_length: Float[Array, ""] | float,
        z0: float,
        interpolate: bool = False,
    ) -> None:
        self.inverse_obukhov_length = jnp.asarray(inverse_obukhov_length)
        self.z0 = float(z0)
        self.interpolate = bool(interpolate)

    def __check_init__(self) -> None:
        if not (self.z0 > 0.0):
            raise ParameterError(f"z0 must be > 0, got {self.z0!r}")

    @classmethod
    def from_surface_layer(
        cls,
        layer: SurfaceLayer,
        *,
        z0: float,
        it: int,
        iy: int,
        ix: int,
        interpolate: bool = False,
    ) -> GolderDispersion:
        """Golder dispersion at one column of a :class:`plumax.met.SurfaceLayer`.

        Args:
            layer: Surface-layer scales from :class:`plumax.met.MoninObukhov`;
                only its 1/L is used.
            z0: The roughness length z₀ [m] the layer was diagnosed with.
            it: Met time index.
            iy: Analysis y index.
            ix: Analysis x index.
            interpolate: See the class attribute.

        Returns:
            The dispersion model for that column.
        """
        inverse_length = layer.inverse_obukhov_length[it, iy, ix]
        return cls(inverse_length, z0=z0, interpolate=interpolate)

    def __call__(
        self, distance: Float[Array, "*shape"]
    ) -> tuple[Float[Array, "*shape"], Float[Array, "*shape"]]:
        # σ^PG(x; k) for every class k, stacked along a leading class axis
        params = jnp.stack([BRIGGS_DISPERSION_PARAMS[c] for c in STABILITY_CLASSES])
        sigma_y_all, sigma_z_all = jax.vmap(
            calculate_briggs_dispersion, in_axes=(None, 0)
        )(distance, params)
        if not self.interpolate:
            stability_class = golder_stability_class(
                self.inverse_obukhov_length, self.z0
            )
            return sigma_y_all[stability_class], sigma_z_all[stability_class]
        index = golder_stability_index(self.inverse_obukhov_length, self.z0)
        lower = jnp.clip(jnp.floor(index), 0, len(STABILITY_CLASSES) - 2)
        lower = lower.astype(jnp.int32)
        # w = s − ⌊s⌋
        weight = index - lower
        # σ = (1 − w) σ^PG(⌊s⌋) + w σ^PG(⌊s⌋ + 1)
        sigma_y = (1.0 - weight) * sigma_y_all[lower] + weight * sigma_y_all[lower + 1]
        sigma_z = (1.0 - weight) * sigma_z_all[lower] + weight * sigma_z_all[lower + 1]
        return sigma_y, sigma_z


__all__ = [
    "GOLDER_BOUNDARY_OFFSET",
    "GOLDER_BOUNDARY_SCALE",
    "GOLDER_Z0_RANGE",
    "GolderDispersion",
    "golder_class_boundaries",
    "golder_stability_class",
    "golder_stability_index",
]
