"""Monin–Obukhov surface-layer similarity in JAX: stability functions, u*, L, w*.

Tier I's Monin–Obukhov dispersion, Tier II's Hanna turbulence and Tier III's
eddy diffusivity all need the friction velocity ``u*``, the Obukhov length
``L`` and the convective velocity ``w*``. Everything here is
``jit``/``grad``/``vmap``-safe: the coupled ``u*``–``L`` system is solved by a
fixed number of fixed-point iterations, and every branch is evaluated on a
clipped argument so the unused side of a ``where`` never produces NaN
gradients.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from plumax._src.errors import PlumaxError
from plumax.met.field import MetField


#: von Kármán constant κ.
KARMAN = 0.4
#: Gravitational acceleration g [m s⁻²].
GRAVITY = 9.80665
#: Dry-air gas constant R_d [J kg⁻¹ K⁻¹].
R_DRY = 287.04
#: Specific heat of dry air at constant pressure c_p [J kg⁻¹ K⁻¹].
CP_DRY = 1004.5
#: Reference pressure of potential temperature p_ref [Pa].
P_REFERENCE = 1.0e5
#: Floor on ``u*`` [m s⁻¹] where it divides ``1/L``, so calm columns stay finite.
U_STAR_FLOOR = 1.0e-3


def _unstable_root(zeta: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    # x = (1 − 16 ζ⁻)^{1/4},  ζ⁻ = min(ζ, 0), so ζ ≥ 0 gives x = 1
    unstable_part = jnp.minimum(zeta, 0.0)
    return (1.0 - 16.0 * unstable_part) ** 0.25


def phi_m(zeta: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Dimensionless wind shear φ_m(ζ) = (κ z / u*) ∂U/∂z.

    φ_m = (1 − 16ζ)^{−1/4} for ζ < 0,  1 + 5ζ for ζ ≥ 0  (Businger–Dyer).

    Args:
        zeta: Stability parameter ζ = z / L.

    Returns:
        φ_m at each ζ.

    Examples:
        Neutral, unstable and stable:

        >>> import jax.numpy as jnp
        >>> values = phi_m(jnp.array([0.0, -1.0, 0.2]))
        >>> bool(jnp.allclose(values, jnp.array([1.0, 17.0**-0.25, 2.0])))
        True

        Works under ``jit`` and keeps the input's dtype:

        >>> import jax
        >>> jax.jit(phi_m)(jnp.zeros(3, dtype=jnp.float32)).dtype
        dtype('float32')

    References:
        Dyer (1974), Boundary-Layer Meteorology 7, 363–372.
    """
    stable_part = jnp.maximum(zeta, 0.0)
    # (1 − 16ζ)^{−1/4}
    unstable = 1.0 / _unstable_root(zeta)
    # 1 + 5ζ
    stable = 1.0 + 5.0 * stable_part
    return jnp.where(zeta < 0.0, unstable, stable)


def phi_h(zeta: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Dimensionless temperature gradient φ_h(ζ).

    φ_h = (1 − 16ζ)^{−1/2} for ζ < 0,  1 + 5ζ for ζ ≥ 0  (Businger–Dyer).

    Args:
        zeta: Stability parameter ζ = z / L.

    Returns:
        φ_h at each ζ.

    Examples:
        Neutral, unstable and stable:

        >>> import jax.numpy as jnp
        >>> values = phi_h(jnp.array([0.0, -1.0, 0.2]))
        >>> bool(jnp.allclose(values, jnp.array([1.0, 17.0**-0.5, 2.0])))
        True
    """
    stable_part = jnp.maximum(zeta, 0.0)
    # (1 − 16ζ)^{−1/2} = x⁻²
    unstable = 1.0 / _unstable_root(zeta) ** 2
    # 1 + 5ζ
    stable = 1.0 + 5.0 * stable_part
    return jnp.where(zeta < 0.0, unstable, stable)


def psi_m(zeta: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Integrated momentum stability correction ψ_m(ζ), zero at neutral.

    ψ_m = 2 ln((1+x)/2) + ln((1+x²)/2) − 2 atan x + π/2 for ζ < 0 (x as in
    φ_m), and −5ζ for ζ ≥ 0. It satisfies φ_m = 1 − ζ dψ_m/dζ.

    Args:
        zeta: Stability parameter ζ = z / L.

    Returns:
        ψ_m at each ζ.

    Examples:
        Zero at neutral, −5ζ when stable, and φ_m = 1 − ζ ψ_m′ when unstable:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> bool(jnp.allclose(psi_m(jnp.array([0.0, 0.2])), jnp.array([0.0, -1.0])))
        True
        >>> zeta = jnp.array(-1.0)
        >>> bool(jnp.isclose(phi_m(zeta), 1.0 - zeta * jax.grad(psi_m)(zeta)))
        True

    References:
        Paulson (1970), Journal of Applied Meteorology 9, 857–861.
    """
    root = _unstable_root(zeta)
    stable_part = jnp.maximum(zeta, 0.0)
    # 2 ln((1 + x)/2)
    first = 2.0 * jnp.log((1.0 + root) / 2.0)
    # ln((1 + x²)/2)
    second = jnp.log((1.0 + root**2) / 2.0)
    # −2 atan x + π/2
    third = -2.0 * jnp.arctan(root) + jnp.pi / 2.0
    unstable = first + second + third
    # −5ζ
    stable = -5.0 * stable_part
    return jnp.where(zeta < 0.0, unstable, stable)


def psi_h(zeta: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Integrated heat stability correction ψ_h(ζ), zero at neutral.

    ψ_h = 2 ln((1 + x²)/2) for ζ < 0,  −5ζ for ζ ≥ 0.

    Args:
        zeta: Stability parameter ζ = z / L.

    Returns:
        ψ_h at each ζ.

    Examples:
        Zero at neutral, −5ζ when stable, 2 ln((1 + √17)/2) at ζ = −1:

        >>> import jax.numpy as jnp
        >>> values = psi_h(jnp.array([0.0, -1.0, 0.2]))
        >>> expected = jnp.array([0.0, 2.0 * jnp.log((1.0 + 17.0**0.5) / 2.0), -1.0])
        >>> bool(jnp.allclose(values, expected))
        True
    """
    root = _unstable_root(zeta)
    stable_part = jnp.maximum(zeta, 0.0)
    # 2 ln((1 + x²)/2)
    unstable = 2.0 * jnp.log((1.0 + root**2) / 2.0)
    # −5ζ
    stable = -5.0 * stable_part
    return jnp.where(zeta < 0.0, unstable, stable)


def friction_velocity(
    wind_speed: Float[Array, "*#batch"],
    z: float | Float[Array, "*#batch"],
    z0: float | Float[Array, "*#batch"],
    inverse_obukhov_length: Float[Array, "*#batch"],
) -> Float[Array, ...]:
    """Friction velocity from the stability-corrected log law.

    u* = κ U / (ln(z/z₀) − ψ_m(z/L) + ψ_m(z₀/L)).

    Args:
        wind_speed: Wind speed U at height z [m s⁻¹].
        z: Reference height above ground [m].
        z0: Roughness length z₀ [m].
        inverse_obukhov_length: 1/L [m⁻¹].

    Returns:
        u* [m s⁻¹].

    Examples:
        Neutral: the plain log law κU / ln(z/z₀):

        >>> import jax.numpy as jnp
        >>> u_star = friction_velocity(jnp.array(5.0), 10.0, 0.1, jnp.array(0.0))
        >>> bool(jnp.isclose(u_star, 0.4 * 5.0 / jnp.log(100.0)))
        True

        Unstable air mixes more momentum down, so u* grows:

        >>> unstable = friction_velocity(jnp.array(5.0), 10.0, 0.1, jnp.array(-0.05))
        >>> bool(unstable > u_star)
        True
    """
    # ln(z / z₀)
    log_ratio = jnp.log(z / z0)
    # ψ_m(z/L)
    correction_top = psi_m(z * inverse_obukhov_length)
    # ψ_m(z₀/L)
    correction_bottom = psi_m(z0 * inverse_obukhov_length)
    denominator = log_ratio - correction_top + correction_bottom
    # u* = κ U / (…)
    return KARMAN * wind_speed / denominator


class SurfaceLayer(eqx.Module):
    """Surface-layer scales on the analysis grid's horizontal T-points.

    Attributes:
        u_star: Friction velocity u* [m s⁻¹].
        inverse_obukhov_length: 1/L [m⁻¹]; zero is neutral, negative unstable.
            Stored as 1/L because it is finite at neutral.
        w_star: Convective velocity w* [m s⁻¹]; zero unless H > 0.
    """

    u_star: Float[Array, "time y x"]
    inverse_obukhov_length: Float[Array, "time y x"]
    w_star: Float[Array, "time y x"]

    @property
    def obukhov_length(self) -> Float[Array, "time y x"]:
        """L [m]; ±inf at neutral."""
        # L = 1 / (1/L)
        return 1.0 / self.inverse_obukhov_length


class MoninObukhov(eqx.Module):
    """Diagnose u*, 1/L and w* from a MetField's lowest level (dry air).

    u* = κ U / (ln(z/z₀) − ψ_m(z/L) + ψ_m(z₀/L)),
    1/L = −κ g H / (ρ c_p T u*³),
    w* = (g H h / (ρ c_p T))^{1/3}  for H > 0, else 0.

    The usual forms carry g w'θ'/θ. H/(ρ c_p) is the actual-temperature flux
    w'T' and w'θ' = (θ/T) w'T', so the potential-temperature factor cancels
    and T remains.

    Algorithm:
        1. Take the lowest T-level (height z = Δz/2 above the grid's surface):
           wind speed from the neighbouring U- and V-points, T, p and ρ.
        2. Start neutral (1/L = 0) and iterate u* ← log law, 1/L ← −b/u*³
           ``n_iterations`` times, with u* floored at :data:`U_STAR_FLOOR`
           where it divides.
        3. Evaluate w* where H > 0.

    Calm and neutral gives u* = 0, 1/L = 0, w* = 0. Very stable cases
    (z/L ≳ 1) converge slowly; check the residual there.

    Attributes:
        n_iterations: Fixed-point iterations of the u*–L system (≥ 1).

    Call arguments: the ``MetField``; ``z0``, the roughness length z₀ [m]; and
    ``sensible_heat_flux``, H [W m⁻²] positive upward. ``z0`` and H are scalars
    or broadcast against ``(time, y, x)``.

    Raises:
        PlumaxError: If ``n_iterations`` is below 1, or (under ``jit`` too) if
            ``z0`` is not positive.

    Examples:
        Neutral air over 5 m s⁻¹ wind: the log law, and 1/L = 0:

        >>> import jax.numpy as jnp
        >>> from plumax.les_fvm.grid import make_grid
        >>> from plumax.met.field import MetField
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
        >>> layer.u_star.shape
        (1, 4, 4)
        >>> bool(jnp.allclose(layer.u_star, 0.4 * 5.0 / jnp.log(25.0 / 0.1)))
        True

        Daytime heating makes the column unstable and convective:

        >>> sunny = MoninObukhov()(field, z0=0.1, sensible_heat_flux=200.0)
        >>> bool((sunny.inverse_obukhov_length < 0).all() & (sunny.w_star > 0).all())
        True

        The configuration round-trips:

        >>> MoninObukhov.from_config(MoninObukhov(n_iterations=8).get_config())
        MoninObukhov(n_iterations=8)

    References:
        Monin & Obukhov (1954); Businger et al. (1971), JAS 28, 181–189.
    """

    n_iterations: int = eqx.field(static=True, default=20)

    def __check_init__(self) -> None:
        if self.n_iterations < 1:
            raise PlumaxError(f"n_iterations must be ≥ 1, got {self.n_iterations}")

    def __call__(
        self,
        field: MetField,
        *,
        z0: float | Float[Array, "*#columns"],
        sensible_heat_flux: float | Float[Array, "*#columns"],
    ) -> SurfaceLayer:
        z0 = jnp.asarray(z0, dtype=field.u.dtype)
        z0 = eqx.error_if(z0, jnp.any(z0 <= 0.0), "z0 must be positive")

        # --- lowest level ---
        # z = Δz / 2 above the grid's surface
        reference_height = 0.5 * field.plume_grid.dz
        u_lowest = field.u[:, 0]
        v_lowest = field.v[:, 0]
        # U-points sit at x + Δx/2: pair each with its western neighbour (the
        # edge column repeats itself), and V-points likewise along y
        u_west = einx.id(
            "t y first, t y rest -> t y (first + rest)",
            u_lowest[..., :1],
            u_lowest[..., :-1],
        )
        v_south = einx.id(
            "t first x, t rest x -> t (first + rest) x",
            v_lowest[..., :1, :],
            v_lowest[..., :-1, :],
        )
        # u_T = (u_{i−1/2} + u_{i+1/2}) / 2
        u_centre = 0.5 * (u_lowest + u_west)
        # v_T = (v_{j−1/2} + v_{j+1/2}) / 2
        v_centre = 0.5 * (v_lowest + v_south)
        # U² = u_T² + v_T²
        speed_squared = u_centre**2 + v_centre**2
        # U = √(U²), with ∂U/∂u_T = ∂U/∂v_T = 0 at calm: √ has an infinite
        # slope at 0, so take it of 1 there (the double-where trick)
        is_calm = speed_squared == 0.0
        safe_squared = jnp.where(is_calm, 1.0, speed_squared)
        wind_speed = jnp.where(is_calm, 0.0, jnp.sqrt(safe_squared))
        temperature = field.temperature[:, 0]
        pressure = field.pressure[:, 0]
        # ρ = p / (R_d T)
        density = pressure / (R_DRY * temperature)
        heat_flux = jnp.broadcast_to(
            jnp.asarray(sensible_heat_flux, dtype=wind_speed.dtype), wind_speed.shape
        )
        # b = κ g H / (ρ c_p T)
        buoyancy = KARMAN * GRAVITY * heat_flux / (density * CP_DRY * temperature)

        # --- u*–L fixed point ---
        def step(_, carry):
            _, inverse_length = carry
            u_star = friction_velocity(wind_speed, reference_height, z0, inverse_length)
            # 1/L = −b / max(u*, u*_min)³
            floored_cube = jnp.maximum(u_star, U_STAR_FLOOR) ** 3
            return u_star, -buoyancy / floored_cube

        neutral = jnp.zeros_like(wind_speed)
        initial = (
            friction_velocity(wind_speed, reference_height, z0, neutral),
            neutral,
        )
        _, inverse_length = jax.lax.fori_loop(0, self.n_iterations, step, initial)
        u_star = friction_velocity(wind_speed, reference_height, z0, inverse_length)

        # --- convective velocity ---
        # g H⁺ h / (ρ c_p T)
        convective_cube = (
            GRAVITY
            * jnp.maximum(heat_flux, 0.0)
            * field.pbl_height
            / (density * CP_DRY * temperature)
        )
        # cbrt has an infinite slope at 0: evaluate it only where H > 0 so the
        # stable/neutral side of the `where` keeps finite gradients
        is_convective = convective_cube > 0.0
        safe_cube = jnp.where(is_convective, convective_cube, 1.0)
        # w* = (…)^{1/3}
        w_star = jnp.where(is_convective, jnp.cbrt(safe_cube), 0.0)
        return SurfaceLayer(
            u_star=u_star, inverse_obukhov_length=inverse_length, w_star=w_star
        )

    def with_n_iterations(self, n_iterations: int) -> MoninObukhov:
        """A copy with another iteration count (re-runs the field check)."""
        return dataclasses.replace(self, n_iterations=n_iterations)

    def get_config(self) -> dict[str, Any]:
        """The hyperparameters, for ``from_config``."""
        return {"n_iterations": self.n_iterations}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MoninObukhov:
        """The diagnosis described by ``get_config``'s output."""
        return cls(**config)
