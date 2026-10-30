"""Briggs plume rise for buoyant and momentum sources: Δh(x) and the effective height.

A heated or fast stack plume keeps rising after it leaves the stack, so the
Gaussian plume sees an effective release height H_eff = H_stack + Δh(x)
rather than the stack height. :class:`BriggsPlumeRise` holds the stack's exit
conditions and returns Δh(x) at downwind distance x for a wind speed u, pure
JAX, so H_eff is differentiable in the source parameters.

Buoyancy and momentum fluxes (Briggs 1969, Eq. 4.19b; Briggs 1975):

    F_b = g v_s d² (T_s − T_a) / (4 T_s)        [m⁴ s⁻³]
    F_m = v_s² d² T_a / (4 T_s)                  [m⁴ s⁻²]

Buoyant rise follows the 2/3 law up to the final rise,

    Δh_b(x) = min(1.6 F_b^{1/3} x^{2/3} / u, Δh_b,f),

with, for classes A–D, the final rise reached at x_f = 3.5 x*,

    x_f = 49 F_b^{5/8} (F_b < 55),  119 F_b^{2/5} (F_b ≥ 55),
    Δh_b,f = 1.6 F_b^{1/3} x_f^{2/3} / u,

and, for the stable classes E–F with s = (g / T_a) ∂θ/∂z,

    Δh_b,f = 2.6 (F_b / (u s))^{1/3}.

Momentum (jet) rise in neutral wind grows as x^{1/3} up to its final value
(Briggs 1969, Eqs. 4.33 and 5.2), and in stable air is capped at the
stable jet rise (Briggs 1969, Eq. 4.28):

    Δh_m(x) = min(1.44 d (v_s/u)^{2/3} (x/d)^{1/3}, Δh_m,f),
    Δh_m,f = 3 d v_s / u                                       (A–D),
    Δh_m,f = min(3 d v_s / u, 1.5 (F_m / u)^{1/3} s^{−1/6})     (E–F).

The plume rises by the larger of the two, Δh = max(Δh_b, Δh_m).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from plumax._src.errors import ParameterError
from plumax.met._src.surface_layer import GRAVITY


#: Default potential-temperature gradient ∂θ/∂z [K m⁻¹] of the stable classes
#: (ISC3 user's guide, Vol. II, §1.1.4.6).
STABLE_THETA_GRADIENT: dict[str, float] = {"E": 0.020, "F": 0.035}

#: The Pasquill–Gifford classes Briggs treats as unstable or neutral.
UNSTABLE_NEUTRAL_CLASSES: tuple[str, ...] = ("A", "B", "C", "D")

# Buoyancy flux [m⁴ s⁻³] separating the two A–D final-rise distances.
_BUOYANCY_FLUX_BREAK = 55.0


def _require_positive(name: str, value: object, *, allow_zero: bool = False) -> None:
    # Traced values skip the eager check; concrete ones fail at construction.
    if isinstance(value, jax.core.Tracer):
        return
    array = np.asarray(value)
    ok = np.all(array >= 0.0) if allow_zero else np.all(array > 0.0)
    if not bool(ok):
        bound = "≥ 0" if allow_zero else "> 0"
        raise ParameterError(f"{name} must be {bound}, got {value!r}")


def _safe_power(
    base: float | Float[Array, "*shape"], exponent: float
) -> Float[Array, "*shape"]:
    # b^p with b ≤ 0 mapped to 0 and a finite gradient there: fractional
    # powers have an infinite slope at 0, so take the power of 1 on that side
    # (the double-where trick)
    base = jnp.asarray(base)
    is_positive = base > 0.0
    safe_base = jnp.where(is_positive, base, 1.0)
    return jnp.where(is_positive, safe_base**exponent, 0.0)


class BriggsPlumeRise(eqx.Module):
    """Briggs plume rise Δh(x) of a stack plume, from its buoyancy and momentum.

    Δh(x) = max(Δh_b(x), Δh_m(x)), with the buoyant and momentum rises of the
    module docstring.

    Algorithm:
        1. Fluxes F_b = g v_s d² (T_s − T_a)/(4 T_s), floored at 0, and
           F_m = v_s² d² T_a/(4 T_s).
        2. Buoyant rise: the 2/3 law 1.6 F_b^{1/3} x^{2/3}/u, capped at the
           class's final rise.
        3. Momentum rise: the jet law 1.44 d (v_s/u)^{2/3} (x/d)^{1/3}, capped
           at 3 d v_s/u, and in stable air also at 1.5 (F_m/u)^{1/3} s^{−1/6}.
        4. Take the larger of the two.

    Attributes:
        exit_velocity: Stack exit velocity v_s ≥ 0 [m s⁻¹].
        exit_temperature: Stack gas temperature T_s > 0 [K].
        stack_diameter: Inside stack-top diameter d > 0 [m].
        ambient_temperature: Ambient air temperature T_a > 0 [K].
        stability_class: Pasquill–Gifford class ``'A'`` … ``'F'``; E and F use
            the stable formulas with ∂θ/∂z from :data:`STABLE_THETA_GRADIENT`.

    Raises:
        ParameterError: If the class is not A–F or a parameter is out of range.

    Examples:
        A hot stack in neutral air: past x_f the rise is the final rise.

        >>> import jax.numpy as jnp
        >>> rise = BriggsPlumeRise(
        ...     exit_velocity=10.0, exit_temperature=400.0, stack_diameter=2.0,
        ...     ambient_temperature=290.0, stability_class="D",
        ... )
        >>> heights = rise(jnp.array([100.0, 5000.0, 20000.0]), wind_speed=5.0)
        >>> heights.shape
        (3,)
        >>> bool(heights[1] == heights[2])
        True

        A cold, still stack does not rise:

        >>> still = BriggsPlumeRise(
        ...     exit_velocity=0.0, exit_temperature=290.0, stack_diameter=2.0,
        ...     ambient_temperature=290.0,
        ... )
        >>> still(jnp.array([500.0]), wind_speed=5.0).tolist()
        [0.0]

        Classes outside A–F are rejected:

        >>> BriggsPlumeRise(
        ...     exit_velocity=1.0, exit_temperature=300.0, stack_diameter=1.0,
        ...     ambient_temperature=290.0, stability_class="G",
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ParameterError: stability_class must be one of A-F, got 'G'

    References:
        Briggs, G. A. (1969). *Plume Rise*. USAEC TID-25075 (Eqs. 4.19b,
        4.28, 4.33, 5.2). U.S. EPA, *User's Guide for the Industrial Source
        Complex (ISC3) Dispersion Models*, Vol. II, §1.1.4 (after Briggs 1971,
        1975).
    """

    exit_velocity: float | Float[Array, ""]
    exit_temperature: float | Float[Array, ""]
    stack_diameter: float | Float[Array, ""]
    ambient_temperature: float | Float[Array, ""]
    stability_class: str = eqx.field(static=True, default="D")

    def __check_init__(self) -> None:
        valid = (*UNSTABLE_NEUTRAL_CLASSES, *STABLE_THETA_GRADIENT)
        if self.stability_class not in valid:
            raise ParameterError(
                f"stability_class must be one of A-F, got {self.stability_class!r}"
            )
        _require_positive("exit_velocity", self.exit_velocity, allow_zero=True)
        _require_positive("exit_temperature", self.exit_temperature)
        _require_positive("stack_diameter", self.stack_diameter)
        _require_positive("ambient_temperature", self.ambient_temperature)

    @property
    def is_stable(self) -> bool:
        """Whether the class is E or F (the stable formulas)."""
        return self.stability_class in STABLE_THETA_GRADIENT

    def buoyancy_flux(self) -> Float[Array, ""]:
        """F_b = g v_s d² (T_s − T_a)/(4 T_s) [m⁴ s⁻³], floored at 0 (no sinking)."""
        # T_s − T_a
        excess = self.exit_temperature - self.ambient_temperature
        # g v_s d² (T_s − T_a) / (4 T_s)
        flux = (
            GRAVITY
            * self.exit_velocity
            * self.stack_diameter**2
            * excess
            / (4.0 * self.exit_temperature)
        )
        return jnp.maximum(flux, 0.0)

    def momentum_flux(self) -> Float[Array, ""]:
        """F_m = v_s² d² T_a/(4 T_s) [m⁴ s⁻²]."""
        # v_s² d² T_a / (4 T_s)
        return jnp.asarray(
            self.exit_velocity**2
            * self.stack_diameter**2
            * self.ambient_temperature
            / (4.0 * self.exit_temperature)
        )

    def stability_parameter(self) -> Float[Array, ""]:
        """s = (g / T_a) ∂θ/∂z [s⁻²] of the stable classes; 0 for A–D."""
        gradient = STABLE_THETA_GRADIENT.get(self.stability_class, 0.0)
        # s = g ∂θ/∂z / T_a
        return jnp.asarray(GRAVITY * gradient / self.ambient_temperature)

    def __call__(
        self, x: Float[Array, "*obs"], *, wind_speed: float | Float[Array, ""]
    ) -> Float[Array, "*obs"]:
        """Plume rise Δh(x) [m] at downwind distance x [m] (Δh = 0 for x ≤ 0)."""
        distance = jnp.maximum(x, 0.0)
        buoyancy_flux = self.buoyancy_flux()
        momentum_flux = self.momentum_flux()
        stability = self.stability_parameter()

        # --- buoyant rise ---
        # F_b^{1/3}
        buoyancy_cube_root = _safe_power(buoyancy_flux, 1.0 / 3.0)
        # 1.6 F_b^{1/3} x^{2/3} / u
        gradual_buoyant = (
            1.6 * buoyancy_cube_root * _safe_power(distance, 2.0 / 3.0) / wind_speed
        )
        if self.is_stable:
            # 2.6 (F_b / (u s))^{1/3}
            final_buoyant = 2.6 * _safe_power(
                buoyancy_flux / (wind_speed * stability), 1.0 / 3.0
            )
        else:
            # x_f = 49 F_b^{5/8} below the break, 119 F_b^{2/5} above
            final_distance = jnp.where(
                buoyancy_flux < _BUOYANCY_FLUX_BREAK,
                49.0 * _safe_power(buoyancy_flux, 5.0 / 8.0),
                119.0 * _safe_power(buoyancy_flux, 2.0 / 5.0),
            )
            # 1.6 F_b^{1/3} x_f^{2/3} / u
            final_buoyant = (
                1.6
                * buoyancy_cube_root
                * _safe_power(final_distance, 2.0 / 3.0)
                / wind_speed
            )
        buoyant_rise = jnp.minimum(gradual_buoyant, final_buoyant)

        # --- momentum rise ---
        # v_s / u
        velocity_ratio = self.exit_velocity / wind_speed
        # 1.44 d (v_s/u)^{2/3} (x/d)^{1/3}
        gradual_jet = (
            1.44
            * self.stack_diameter
            * _safe_power(velocity_ratio, 2.0 / 3.0)
            * _safe_power(distance / self.stack_diameter, 1.0 / 3.0)
        )
        # 3 d v_s / u
        final_jet = 3.0 * self.stack_diameter * velocity_ratio
        if self.is_stable:
            # 1.5 (F_m / u)^{1/3} s^{−1/6}
            stable_jet = (
                1.5
                * _safe_power(momentum_flux / wind_speed, 1.0 / 3.0)
                * stability ** (-1.0 / 6.0)
            )
            final_jet = jnp.minimum(final_jet, stable_jet)
        momentum_rise = jnp.minimum(gradual_jet, final_jet)

        # Δh = max(Δh_b, Δh_m)
        return jnp.maximum(buoyant_rise, momentum_rise)


def briggs_plume_rise(
    x: Float[Array, "*obs"],
    *,
    exit_velocity: float | Float[Array, ""],
    exit_temperature: float | Float[Array, ""],
    stack_diameter: float | Float[Array, ""],
    ambient_temperature: float | Float[Array, ""],
    wind_speed: float | Float[Array, ""],
    stability_class: str = "D",
) -> Float[Array, "*obs"]:
    """Briggs plume rise Δh(x) [m]; see :class:`BriggsPlumeRise`.

    Args:
        x: Downwind distance [m].
        exit_velocity: Stack exit velocity v_s [m s⁻¹].
        exit_temperature: Stack gas temperature T_s [K].
        stack_diameter: Stack diameter d [m].
        ambient_temperature: Ambient temperature T_a [K].
        wind_speed: Wind speed u at the stack top [m s⁻¹].
        stability_class: Pasquill–Gifford class ``'A'`` … ``'F'``.

    Returns:
        Δh(x), with the final rise past x_f.

    Examples:
        The final rise of a buoyant plume in neutral air (F_b < 55):

        >>> import jax.numpy as jnp
        >>> rise = briggs_plume_rise(
        ...     jnp.array([1.0e5]), exit_velocity=10.0, exit_temperature=400.0,
        ...     stack_diameter=2.0, ambient_temperature=290.0, wind_speed=5.0,
        ... )
        >>> flux = 9.80665 * 10.0 * 4.0 * 110.0 / 1600.0
        >>> final = 1.6 * flux ** (1 / 3) * (49.0 * flux ** 0.625) ** (2 / 3) / 5.0
        >>> bool(jnp.allclose(rise, final))
        True
    """
    plume_rise = BriggsPlumeRise(
        exit_velocity=exit_velocity,
        exit_temperature=exit_temperature,
        stack_diameter=stack_diameter,
        ambient_temperature=ambient_temperature,
        stability_class=stability_class,
    )
    return plume_rise(x, wind_speed=wind_speed)


def effective_height(
    x: Float[Array, "*obs"],
    *,
    stack_height: float | Float[Array, ""],
    exit_velocity: float | Float[Array, ""],
    exit_temperature: float | Float[Array, ""],
    stack_diameter: float | Float[Array, ""],
    ambient_temperature: float | Float[Array, ""],
    wind_speed: float | Float[Array, ""],
    stability_class: str = "D",
) -> Float[Array, "*obs"]:
    """Effective release height H_eff(x) = H_stack + Δh(x) [m].

    Args:
        x: Downwind distance [m].
        stack_height: Physical stack height H_stack [m].
        exit_velocity: Stack exit velocity v_s [m s⁻¹].
        exit_temperature: Stack gas temperature T_s [K].
        stack_diameter: Stack diameter d [m].
        ambient_temperature: Ambient temperature T_a [K].
        wind_speed: Wind speed u at the stack top [m s⁻¹].
        stability_class: Pasquill–Gifford class ``'A'`` … ``'F'``.

    Returns:
        H_eff at each x.

    Examples:
        At the stack itself the plume has not risen yet:

        >>> import jax.numpy as jnp
        >>> effective_height(
        ...     jnp.array([0.0]), stack_height=30.0, exit_velocity=10.0,
        ...     exit_temperature=400.0, stack_diameter=2.0,
        ...     ambient_temperature=290.0, wind_speed=5.0,
        ... ).tolist()
        [30.0]
    """
    rise = briggs_plume_rise(
        x,
        exit_velocity=exit_velocity,
        exit_temperature=exit_temperature,
        stack_diameter=stack_diameter,
        ambient_temperature=ambient_temperature,
        wind_speed=wind_speed,
        stability_class=stability_class,
    )
    # H_eff = H_stack + Δh
    return stack_height + rise


__all__ = [
    "STABLE_THETA_GRADIENT",
    "UNSTABLE_NEUTRAL_CLASSES",
    "BriggsPlumeRise",
    "briggs_plume_rise",
    "effective_height",
]
