"""Closed-form column of a steady Gaussian plume, and its transport backend.

A satellite sees the vertical column. For a steady Gaussian plume with ground
reflection the column does not depend on the vertical spread or the release
height: the two vertical Gaussians integrate to √(2π) σ_z over z ≥ 0, so

    ΔΩ(d, c) = q / (√(2π) w(d)) exp(−(c − m(d))² / 2w(d)²)   for d > 0,

with d and c the downwind and crosswind distances, q = Q / U the line density
[kg m⁻¹], w(d) the crosswind width and m(d) a centreline offset. The width
follows a power law above an initial width,

    w(d)² = w₀² + w_ref² (d / d_ref)^(2b).

:class:`ColumnPlume` evaluates the column per unit line density at a set of
receptors for a :class:`PlumeGeometry`; :class:`ColumnPlumeSource` wraps the
two as a ``coupled.TransportBackend``. The design is on the image-fit page of
the docs (``design/01_tier1_image_fit.md``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from plumax._src.errors import ParameterError


if TYPE_CHECKING:
    from plumax.coupled.instrument import Instrument


#: √(2π), the normaliser of a unit Gaussian.
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)

#: Smallest downwind distance [m] the width law is evaluated at, so that
#: log(d / d_ref) and its gradient in the exponent stay finite at the source.
_MIN_DISTANCE = 1e-3


def plume_frame(
    receptors: Float[Array, "obs axis"],
    location: Float[ArrayLike, " axis"],
    direction: Float[ArrayLike, ""],
) -> tuple[Float[Array, " obs"], Float[Array, " obs"]]:
    """Downwind and crosswind distance of each receptor from the source.

    d = (s − s₀)ᵀ e_∥ and c = (s − s₀)ᵀ e_⊥, with e_∥ = (cos φ, sin φ) and
    e_⊥ = (−sin φ, cos φ).

    Args:
        receptors: Receptor positions s [m], one ``(x, y)`` row each.
        location: Source position s₀ [m].
        direction: Plume direction φ [rad], counter-clockwise from the x axis,
            pointing downwind.

    Returns:
        The downwind distance d and the crosswind distance c [m].

    Examples:
        A plume heading along y: a receptor 100 m north is 100 m downwind.

        >>> import jax.numpy as jnp
        >>> receptors = jnp.array([[0.0, 100.0], [50.0, 0.0]])
        >>> downwind, crosswind = plume_frame(
        ...     receptors, jnp.zeros(2), jnp.asarray(jnp.pi / 2)
        ... )
        >>> bool(jnp.allclose(downwind, jnp.array([100.0, 0.0]), atol=1e-4))
        True
        >>> bool(jnp.allclose(crosswind, jnp.array([0.0, -50.0]), atol=1e-4))
        True
    """
    direction = jnp.asarray(direction)
    cos_direction = jnp.cos(direction)
    sin_direction = jnp.sin(direction)
    # e_∥ = (cos φ, sin φ)
    along = jnp.stack([cos_direction, sin_direction])
    # e_⊥ = (−sin φ, cos φ)
    across = jnp.stack([-sin_direction, cos_direction])
    # Δs = s − s₀
    offset = einx.subtract("obs axis, axis -> obs axis", receptors, location)
    # d = Δsᵀ e_∥
    downwind = einx.dot("obs axis, axis -> obs", offset, along)
    # c = Δsᵀ e_⊥
    crosswind = einx.dot("obs axis, axis -> obs", offset, across)
    return downwind, crosswind


def power_law_width(
    downwind: Float[Array, "*obs"],
    *,
    reference_width: Float[ArrayLike, ""],
    width_exponent: Float[ArrayLike, ""],
    reference_distance: float,
    initial_width: float = 0.0,
) -> Float[Array, "*obs"]:
    """Crosswind standard deviation of the plume at each downwind distance.

    w(d) = √(w₀² + w_ref² (d / d_ref)^(2b)), with d clipped below at 1 mm.

    Args:
        downwind: Downwind distance d [m]; upwind values are clipped.
        reference_width: Width w_ref [m] at the reference distance.
        width_exponent: Growth exponent b.
        reference_distance: Reference distance d_ref [m].
        initial_width: Width w₀ [m] at the source.

    Returns:
        The width w(d) [m].

    Examples:
        At the reference distance the width is the reference width:

        >>> import jax.numpy as jnp
        >>> width = power_law_width(
        ...     jnp.array([1000.0, 4000.0]),
        ...     reference_width=150.0,
        ...     width_exponent=0.5,
        ...     reference_distance=1000.0,
        ... )
        >>> bool(jnp.allclose(width, jnp.array([150.0, 300.0])))
        True

        The initial width adds in quadrature:

        >>> near = power_law_width(
        ...     jnp.array([0.0]),
        ...     reference_width=150.0,
        ...     width_exponent=0.9,
        ...     reference_distance=1000.0,
        ...     initial_width=17.0,
        ... )
        >>> bool(jnp.allclose(near, 17.0, rtol=1e-6))
        True
    """
    # d ← max(d, 1 mm): the power law is evaluated in logs
    distance = jnp.maximum(downwind, _MIN_DISTANCE)
    # log(d / d_ref)
    log_ratio = jnp.log(distance / reference_distance)
    # (d / d_ref)^(2b) = exp(2b log(d / d_ref))
    growth = jnp.exp(2.0 * width_exponent * log_ratio)
    # w² = w₀² + w_ref² (d / d_ref)^(2b)
    variance = initial_width**2 + jnp.square(reference_width) * growth
    width = jnp.sqrt(variance)
    return width


def unit_column(
    downwind: Float[Array, "*obs"],
    crosswind: Float[Array, "*obs"],
    width: Float[Array, "*obs"],
    *,
    centreline: Float[Array, "*obs"] | None = None,
    upwind_scale: float | None = None,
    pixel_size: float | None = None,
) -> Float[Array, "*obs"]:
    """Column per unit line density, the Gaussian ridge across the plume.

    r = exp(−(c − m)² / 2w²) / (√(2π) w) for d > 0, else 0 [m⁻¹]. With a
    pixel side g the ridge is averaged across the pixel,
    r = (erf((c − m + g/2) / √2 w) − erf((c − m − g/2) / √2 w)) / 2g. With an
    upwind scale ℓ_u the cut at the source becomes the ramp 1 / (1 + e^(−d/ℓ_u)).

    Algorithm:
        1. Shift the crosswind distance by the centreline: c − m.
        2. Evaluate the Gaussian at the point, or average it across the pixel.
        3. Apply the upwind cut, hard or as a logistic ramp.

    Args:
        downwind: Downwind distance d [m].
        crosswind: Crosswind distance c [m].
        width: Crosswind width w [m], positive.
        centreline: Centreline offset m [m] at each receptor; ``None`` is zero.
        upwind_scale: Ramp length ℓ_u [m]; ``None`` is the hard cut.
        pixel_size: Pixel side g [m]; ``None`` is point evaluation.

    Returns:
        The column per unit line density r [m⁻¹]; ΔΩ = q r.

    Examples:
        On the centreline the response is 1 / (√(2π) w):

        >>> import einx
        >>> import jax.numpy as jnp
        >>> downwind = jnp.array([500.0, -500.0])
        >>> crosswind = jnp.zeros(2)
        >>> width = jnp.full(2, 100.0)
        >>> response = unit_column(downwind, crosswind, width)
        >>> bool(jnp.allclose(response[0], 1.0 / (2.5066282746 * 100.0)))
        True
        >>> float(response[1])
        0.0

        The pixel average integrates to one across a row of pixels:

        >>> offsets = jnp.arange(-50.0, 51.0) * 60.0
        >>> row = unit_column(
        ...     jnp.full_like(offsets, 500.0),
        ...     offsets,
        ...     jnp.full_like(offsets, 20.0),
        ...     pixel_size=60.0,
        ... )
        >>> bool(jnp.allclose(einx.sum("[obs]", row) * 60.0, 1.0, rtol=1e-10))
        True
    """
    # c − m
    shifted = crosswind if centreline is None else crosswind - centreline
    # the masked upwind branch still gets a gradient, so keep its width finite
    downstream = downwind > 0.0
    safe_width = jnp.where(downstream, width, 1.0) if upwind_scale is None else width

    # --- ridge ---
    if pixel_size is None:
        # (c − m) / w
        standardised = shifted / safe_width
        # exp(−(c − m)² / 2w²)
        kernel = jnp.exp(-0.5 * jnp.square(standardised))
        # r = exp(−(c − m)² / 2w²) / (√(2π) w)
        response = kernel / (_SQRT_TWO_PI * safe_width)
    else:
        # √2 w
        scale = math.sqrt(2.0) * safe_width
        # erf((c − m + g/2) / √2 w)
        upper = jax.scipy.special.erf((shifted + 0.5 * pixel_size) / scale)
        # erf((c − m − g/2) / √2 w)
        lower = jax.scipy.special.erf((shifted - 0.5 * pixel_size) / scale)
        # r = (erf(·) − erf(·)) / 2g
        response = (upper - lower) / (2.0 * pixel_size)

    # --- upwind cut ---
    if upwind_scale is None:
        # r = 0 for d ≤ 0
        return jnp.where(downstream, response, 0.0)
    # r ← r / (1 + exp(−d / ℓ_u))
    ramp = jax.nn.sigmoid(downwind / upwind_scale)
    return response * ramp


class PlumeGeometry(eqx.Module):
    """The fitted variables of a column plume: where it starts, heads and widens.

    Attributes:
        location: Source position s₀ [m].
        direction: Plume direction φ [rad], counter-clockwise from the x axis,
            pointing downwind.
        reference_width: Width w_ref [m] at the reference distance.
        width_exponent: Growth exponent b of the width law.

    Examples:
        Scalars are stored as arrays, so the geometry is a pytree of leaves:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> geometry = PlumeGeometry(
        ...     location=jnp.zeros(2),
        ...     direction=0.3,
        ...     reference_width=150.0,
        ...     width_exponent=0.9,
        ... )
        >>> len(jax.tree_util.tree_leaves(geometry))
        4
    """

    location: Float[Array, " axis"] = eqx.field(converter=jnp.asarray)
    direction: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    reference_width: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    width_exponent: Float[Array, ""] = eqx.field(converter=jnp.asarray)


class ColumnPlume(eqx.Module):
    """Column of a steady Gaussian plume per unit line density.

    r(s; θ) = exp(−c² / 2w²) / (√(2π) w) for d > 0, else 0, with
    w² = w₀² + w_ref² (d / d_ref)^(2b), so that ΔΩ = q r for q = Q / U.

    Algorithm:
        1. Rotate the receptors into the plume frame: d = (s − s₀)ᵀ e_∥,
           c = (s − s₀)ᵀ e_⊥.
        2. Evaluate the width w(d) on d clipped at the source.
        3. Evaluate the Gaussian ridge, or its average across a pixel.
        4. Apply the upwind cut, hard or as a logistic ramp.

    Attributes:
        reference_distance: Distance d_ref [m] at which the width is w_ref.
        initial_width: Width w₀ [m] at the source; g / √12 stands in for a
            pixel of side g.
        upwind_scale: Length ℓ_u [m] of the logistic ramp at the source;
            ``None`` is the hard cut.
        pixel_size: Pixel side g [m] for the crosswind average; ``None`` is
            point evaluation.

    Raises:
        ParameterError: If a length is not positive, or ``initial_width`` is
            negative.

    Examples:
        The crosswind integral of the response is 1 at every distance:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> crosswind = jnp.linspace(-3000.0, 3000.0, 6001)
        >>> downwind = jnp.full_like(crosswind, 2000.0)
        >>> receptors = einx.rearrange(
        ...     "obs, obs -> obs (1 + 1)", downwind, crosswind
        ... )
        >>> geometry = PlumeGeometry(
        ...     location=jnp.zeros(2),
        ...     direction=0.0,
        ...     reference_width=150.0,
        ...     width_exponent=0.9,
        ... )
        >>> response = ColumnPlume()(receptors, geometry)
        >>> response.shape
        (6001,)
        >>> spacing = 1.0
        >>> integral = einx.sum("[obs]", response) * spacing
        >>> bool(jnp.allclose(integral, 1.0, rtol=1e-4))
        True

        Upwind of the source the response is zero:

        >>> upwind = jnp.array([[-500.0, 0.0]])
        >>> float(ColumnPlume()(upwind, geometry)[0])
        0.0

        The configuration round-trips:

        >>> plume = ColumnPlume(initial_width=17.3, upwind_scale=17.3)
        >>> ColumnPlume.from_config(plume.get_config()) == plume
        True

    References:
        Seinfeld and Pandis, Atmospheric Chemistry and Physics, ch. 18.
    """

    reference_distance: float = 1000.0
    initial_width: float = 0.0
    upwind_scale: float | None = None
    pixel_size: float | None = None

    def __check_init__(self) -> None:
        if self.reference_distance <= 0.0:
            raise ParameterError(
                f"reference_distance must be > 0, got {self.reference_distance}"
            )
        if self.initial_width < 0.0:
            raise ParameterError(f"initial_width must be ≥ 0, got {self.initial_width}")
        if self.upwind_scale is not None and self.upwind_scale <= 0.0:
            raise ParameterError(f"upwind_scale must be > 0, got {self.upwind_scale}")
        if self.pixel_size is not None and self.pixel_size <= 0.0:
            raise ParameterError(f"pixel_size must be > 0, got {self.pixel_size}")

    def __call__(
        self,
        receptors: Float[Array, "obs axis"],
        geometry: PlumeGeometry,
    ) -> Float[Array, " obs"]:
        # --- plume frame ---
        # d = (s − s₀)ᵀ e_∥,  c = (s − s₀)ᵀ e_⊥
        downwind, crosswind = plume_frame(
            receptors, geometry.location, geometry.direction
        )

        # --- width ---
        # w² = w₀² + w_ref² (d / d_ref)^(2b)
        width = power_law_width(
            downwind,
            reference_width=geometry.reference_width,
            width_exponent=geometry.width_exponent,
            reference_distance=self.reference_distance,
            initial_width=self.initial_width,
        )

        # --- column per unit line density ---
        # r = exp(−c² / 2w²) / (√(2π) w)
        response = unit_column(
            downwind,
            crosswind,
            width,
            upwind_scale=self.upwind_scale,
            pixel_size=self.pixel_size,
        )
        return response

    def get_config(self) -> dict[str, Any]:
        """The four fixed choices, as plain floats."""
        return {
            "reference_distance": self.reference_distance,
            "initial_width": self.initial_width,
            "upwind_scale": self.upwind_scale,
            "pixel_size": self.pixel_size,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ColumnPlume:
        """The plume a :meth:`get_config` dictionary describes."""
        return cls(**config)


class ColumnPlumeSource(eqx.Module):
    """A column plume as a ``coupled.TransportBackend``.

    r_inst = A_inst ⊙ r(s_inst; θ), the column per unit line density at an
    instrument's receptors, through its averaging kernel. The coupled forward
    then predicts y = q r_inst + Ω_bg + bias.

    Attributes:
        plume: The column model and its fixed choices.
        geometry: The geometry the response is evaluated at.
        background: Additive background column Ω_bg [kg m⁻²].

    Examples:
        The response through a unit averaging kernel is the plume's own:

        >>> import jax.numpy as jnp
        >>> from plumax.coupled import Instrument
        >>> receptors = jnp.array([[1000.0, 0.0], [1000.0, 300.0]])
        >>> instrument = Instrument(name="EMIT", receptors=receptors)
        >>> geometry = PlumeGeometry(
        ...     location=jnp.zeros(2),
        ...     direction=0.0,
        ...     reference_width=150.0,
        ...     width_exponent=0.9,
        ... )
        >>> source = ColumnPlumeSource(plume=ColumnPlume(), geometry=geometry)
        >>> response = source.column_response(instrument)
        >>> bool(jnp.allclose(response, ColumnPlume()(receptors, geometry)))
        True
    """

    plume: ColumnPlume
    geometry: PlumeGeometry
    background: float = 0.0

    def column_response(self, instrument: Instrument) -> Float[Array, " obs"]:
        """Column per unit line density at the instrument's receptors [m⁻¹].

        r_inst = A_inst ⊙ r(s_inst; θ).

        Args:
            instrument: The observing instrument (receptors and kernel).

        Returns:
            The unit response r_inst.
        """
        receptors = jnp.asarray(instrument.receptors)
        # r = exp(−c² / 2w²) / (√(2π) w)
        response = self.plume(receptors, self.geometry)
        # r_inst = A ⊙ r
        observed = instrument.ak * response
        return observed


__all__ = [
    "ColumnPlume",
    "ColumnPlumeSource",
    "PlumeGeometry",
    "plume_frame",
    "power_law_width",
    "unit_column",
]
