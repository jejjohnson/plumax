"""Monin–Obukhov surface-layer similarity, in pure JAX.

Tier I's Monin–Obukhov dispersion, Tier II's Hanna turbulence and Tier III's
eddy diffusivity all need the same surface-layer scales: the friction
velocity ``u*``, the Obukhov length ``L`` and the convective velocity
``w*``. This module diagnoses them from a :class:`~plumax.met.MetField`'s
lowest level, a roughness length and a sensible heat flux, and provides the
stability functions themselves. Everything is ``jit``/``grad``-safe: the
coupled ``u*``–``L`` system is solved by a fixed number of fixed-point
iterations, and every branch is evaluated on a clipped argument so the
unused side of a ``where`` never produces NaN gradients.

Notation
--------
- ``κ = 0.4``: von Kármán constant.
- ``z``: height above ground of the reference level [m]; ``z₀``: roughness
  length [m].
- ``ζ = z / L``: stability parameter (``ζ < 0`` unstable, ``> 0`` stable).
- ``H``: surface sensible heat flux [W m⁻²], positive upward (heating the
  air, unstable).
- ``θ``: potential temperature [K]; ``ρ``: air density [kg m⁻³];
  ``c_p``: specific heat of dry air [J kg⁻¹ K⁻¹].

Stability functions (Businger–Dyer, in Dyer's 1974 form, integrated by
Paulson 1970):

- unstable, ``x = (1 − 16ζ)^{1/4}``: ``φ_m = x⁻¹``, ``φ_h = x⁻²``,
  ``ψ_m = 2 ln((1 + x)/2) + ln((1 + x²)/2) − 2 atan x + π/2``,
  ``ψ_h = 2 ln((1 + x²)/2)``;
- stable: ``φ_m = φ_h = 1 + 5ζ``, ``ψ_m = ψ_h = −5ζ``.

They satisfy ``φ = 1 − ζ dψ/dζ``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from plumax.met.field import MetField


#: von Kármán constant.
KARMAN = 0.4
#: Gravitational acceleration [m s⁻²].
GRAVITY = 9.80665
#: Dry-air gas constant [J kg⁻¹ K⁻¹].
R_DRY = 287.04
#: Specific heat of dry air at constant pressure [J kg⁻¹ K⁻¹].
CP_DRY = 1004.5
#: Reference pressure of potential temperature [Pa].
P_REFERENCE = 1.0e5


def _unstable_x(zeta: Array) -> Array:
    # x = (1 − 16ζ)^{1/4} on the unstable side; ζ ≥ 0 is clipped to x = 1.
    return (1.0 - 16.0 * jnp.minimum(zeta, 0.0)) ** 0.25


def phi_m(zeta: Float[Array, ...]) -> Float[Array, ...]:
    """Dimensionless wind shear ``φ_m(ζ) = (κz/u*) ∂U/∂z``.

    Args:
        zeta: Stability parameter ``ζ = z/L``.

    Returns:
        ``φ_m`` of ``zeta``'s shape.
    """
    zeta = jnp.asarray(zeta)
    return jnp.where(
        zeta < 0.0, 1.0 / _unstable_x(zeta), 1.0 + 5.0 * jnp.maximum(zeta, 0.0)
    )


def phi_h(zeta: Float[Array, ...]) -> Float[Array, ...]:
    """Dimensionless temperature gradient ``φ_h(ζ)``.

    Args:
        zeta: Stability parameter ``ζ = z/L``.

    Returns:
        ``φ_h`` of ``zeta``'s shape.
    """
    zeta = jnp.asarray(zeta)
    return jnp.where(
        zeta < 0.0, 1.0 / _unstable_x(zeta) ** 2, 1.0 + 5.0 * jnp.maximum(zeta, 0.0)
    )


def psi_m(zeta: Float[Array, ...]) -> Float[Array, ...]:
    """Integrated momentum stability correction ``ψ_m(ζ)``.

    Args:
        zeta: Stability parameter ``ζ = z/L``.

    Returns:
        ``ψ_m`` of ``zeta``'s shape (zero at neutral).
    """
    zeta = jnp.asarray(zeta)
    x = _unstable_x(zeta)
    unstable = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x**2) / 2.0)
        - 2.0 * jnp.arctan(x)
        + jnp.pi / 2.0
    )
    return jnp.where(zeta < 0.0, unstable, -5.0 * jnp.maximum(zeta, 0.0))


def psi_h(zeta: Float[Array, ...]) -> Float[Array, ...]:
    """Integrated heat stability correction ``ψ_h(ζ)``.

    Args:
        zeta: Stability parameter ``ζ = z/L``.

    Returns:
        ``ψ_h`` of ``zeta``'s shape (zero at neutral).
    """
    zeta = jnp.asarray(zeta)
    x = _unstable_x(zeta)
    return jnp.where(
        zeta < 0.0, 2.0 * jnp.log((1.0 + x**2) / 2.0), -5.0 * jnp.maximum(zeta, 0.0)
    )


class SurfaceLayer(eqx.Module):
    """Surface-layer scales on the analysis grid's horizontal T-points.

    Attributes:
        u_star: Friction velocity ``u*`` [m s⁻¹], ``(n_time, ny, nx)``.
        inverse_obukhov_length: ``1/L`` [m⁻¹]; zero is neutral, negative
            unstable. Stored as ``1/L`` because it is finite at neutral.
        w_star: Convective velocity ``w*`` [m s⁻¹]; zero unless ``H > 0``.
    """

    u_star: Float[Array, "n_time ny nx"]
    inverse_obukhov_length: Float[Array, "n_time ny nx"]
    w_star: Float[Array, "n_time ny nx"]

    @property
    def obukhov_length(self) -> Float[Array, "n_time ny nx"]:
        """``L`` [m]; ``±inf`` at neutral."""
        return 1.0 / self.inverse_obukhov_length


def friction_velocity(
    wind_speed: Float[Array, ...],
    z: float | Float[Array, ...],
    z0: float | Float[Array, ...],
    inverse_obukhov_length: Float[Array, ...],
) -> Float[Array, ...]:
    """``u* = κ U / (ln(z/z₀) − ψ_m(z/L) + ψ_m(z₀/L))``.

    Args:
        wind_speed: Wind speed ``U`` at height ``z`` [m s⁻¹].
        z: Reference height above ground [m].
        z0: Roughness length [m].
        inverse_obukhov_length: ``1/L`` [m⁻¹].

    Returns:
        ``u*`` [m s⁻¹].
    """
    log_term = (
        jnp.log(z / z0)
        - psi_m(z * inverse_obukhov_length)
        + psi_m(z0 * inverse_obukhov_length)
    )
    return KARMAN * wind_speed / log_term


def surface_layer_from_met(
    field: MetField,
    *,
    z0: float | Float[Array, "ny nx"],
    sensible_heat_flux: float | Float[Array, "n_time ny nx"],
    n_iterations: int = 20,
) -> SurfaceLayer:
    """Diagnose ``u*``, ``1/L`` and ``w*`` from a MetField's lowest level.

    The reference level is the analysis grid's lowest T-level, at height
    ``z = dz/2`` above its surface; the wind speed there averages ``u``
    and ``v`` from the neighbouring U- and V-points. With
    ``θ = T (p_ref/p)^{R_d/c_p}`` and ``ρ = p/(R_d T)`` the system

    - ``u* = κ U / (ln(z/z₀) − ψ_m(z/L) + ψ_m(z₀/L))``,
    - ``1/L = −κ g H / (ρ c_p θ u*³)``

    is iterated ``n_iterations`` times from neutral (``1/L = 0``), and
    ``w* = (g H h / (ρ c_p θ))^{1/3}`` for ``H > 0`` with ``h`` the PBL
    height, else zero. Very stable cases (large ``z/L``) converge slowly;
    check the residual if ``ζ`` exceeds about 1.

    Args:
        field: Met on the analysis grid.
        z0: Roughness length [m], scalar or per column.
        sensible_heat_flux: Surface sensible heat flux ``H`` [W m⁻²],
            positive upward; scalar or ``(n_time, ny, nx)``.
        n_iterations: Fixed-point iterations (static).

    Returns:
        The :class:`SurfaceLayer`.
    """
    grid = field.plume_grid
    z_ref = 0.5 * grid.dz
    u0 = field.u[:, 0]
    v0 = field.v[:, 0]
    # U-points sit at x + dx/2: average each with its western neighbour
    # (the edge column repeats itself), likewise V-points along y.
    u_t = 0.5 * (u0 + jnp.concatenate([u0[..., :1], u0[..., :-1]], axis=-1))
    v_t = 0.5 * (v0 + jnp.concatenate([v0[..., :1, :], v0[..., :-1, :]], axis=-2))
    speed = jnp.sqrt(u_t**2 + v_t**2)
    temperature = field.temperature[:, 0]
    pressure = field.pressure[:, 0]
    theta = temperature * (P_REFERENCE / pressure) ** (R_DRY / CP_DRY)
    rho = pressure / (R_DRY * temperature)
    heat = jnp.broadcast_to(jnp.asarray(sensible_heat_flux, speed.dtype), speed.shape)
    buoyancy = KARMAN * GRAVITY * heat / (rho * CP_DRY * theta)

    def step(_, carry):
        _, inv_l = carry
        u_star = friction_velocity(speed, z_ref, z0, inv_l)
        return u_star, -buoyancy / u_star**3

    u_star0 = friction_velocity(speed, z_ref, z0, jnp.zeros_like(speed))
    u_star, inv_l = jax.lax.fori_loop(
        0, n_iterations, step, (u_star0, jnp.zeros_like(speed))
    )
    u_star = friction_velocity(speed, z_ref, z0, inv_l)
    convective = (
        GRAVITY * jnp.maximum(heat, 0.0) * field.pbl_height / (rho * CP_DRY * theta)
    )
    # cbrt has an infinite slope at 0: evaluate it only where H > 0 so the
    # stable/neutral side of the `where` keeps finite gradients.
    positive = convective > 0.0
    w_star = jnp.where(positive, jnp.cbrt(jnp.where(positive, convective, 1.0)), 0.0)
    return SurfaceLayer(u_star=u_star, inverse_obukhov_length=inv_l, w_star=w_star)
