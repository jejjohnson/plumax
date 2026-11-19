"""Turbulence parameterisations for the Lagrangian particle model.

The Markov-1 Langevin model (see :mod:`plumax.lagrangian.particles`) needs two
fields at every particle location: the turbulent velocity standard deviations
``σ = (σ_u, σ_v, σ_w)`` [m/s] and the Lagrangian decorrelation timescales
``τ_L = (τ_u, τ_v, τ_w)`` [s]. Together they define the Ornstein–Uhlenbeck
velocity process whose stationary variance is ``σ²`` and whose autocorrelation
decays as ``exp(-Δt / τ_L)``.

Two parameterisations are provided:

- :class:`HomogeneousTurbulence` — constant ``σ`` and ``τ_L`` everywhere. This
  is the well-mixed-exact baseline used by the validation suite (a constant-σ
  OU process is exactly well mixed, so no density-gradient drift correction is
  needed) and a reasonable first approximation inside a well-mixed boundary
  layer.
- :func:`hanna_profiles` — Monin–Obukhov-similarity profiles after Hanna
  (1982), giving height-dependent ``σ(z)`` and ``τ_L(z)`` from surface-layer
  scales ``(u_*, L, w_*, h)``. This is the operational parameterisation; height
  dependence makes the turbulence inhomogeneous, which the integrator handles
  with the well-mixed drift correction.

References:
    Hanna, S. R. (1982). Applications in air pollution modeling. In
    *Atmospheric Turbulence and Air Pollution Modelling* (pp. 275–310).
"""

from __future__ import annotations

from dataclasses import dataclass

import einx
import jax
import jax.numpy as jnp

from plumax._src.tracing import require_positive


# Below this magnitude a turbulent σ is treated as zero (deterministic limit).
_SIGMA_FLOOR: float = 1e-12

#: Default Coriolis parameter f [1/s] for the neutral Hanna profiles: the
#: value FLEXPART's ``hanna.f90`` hard-codes (latitude ≈ 43°).
DEFAULT_CORIOLIS_PARAMETER: float = 1e-4


@dataclass(frozen=True)
class HomogeneousTurbulence:
    """Spatially uniform, well-mixed-exact turbulence statistics.

    Attributes:
        sigma_u: Along-wind velocity standard deviation [m/s].
        sigma_v: Cross-wind velocity standard deviation [m/s].
        sigma_w: Vertical velocity standard deviation [m/s].
        tau_u: Along-wind Lagrangian timescale [s].
        tau_v: Cross-wind Lagrangian timescale [s].
        tau_w: Vertical Lagrangian timescale [s].
    """

    sigma_u: float
    sigma_v: float
    sigma_w: float
    tau_u: float
    tau_v: float
    tau_w: float

    def __post_init__(self) -> None:
        for name in ("sigma_u", "sigma_v", "sigma_w"):
            require_positive(
                name,
                getattr(self, name),
                caller="HomogeneousTurbulence",
                allow_zero=True,
            )
        for name in ("tau_u", "tau_v", "tau_w"):
            require_positive(name, getattr(self, name), caller="HomogeneousTurbulence")

    @classmethod
    def isotropic(cls, sigma: float, tau: float) -> HomogeneousTurbulence:
        """Build isotropic turbulence: equal ``σ`` and ``τ_L`` on all axes."""
        return cls(sigma, sigma, sigma, tau, tau, tau)

    @property
    def sigma(self) -> jax.Array:
        """Velocity standard deviations ``(σ_u, σ_v, σ_w)`` [m/s], shape ``(3,)``."""
        return jnp.array([self.sigma_u, self.sigma_v, self.sigma_w])

    @property
    def tau(self) -> jax.Array:
        """Lagrangian timescales ``(τ_u, τ_v, τ_w)`` [s], shape ``(3,)``."""
        return jnp.array([self.tau_u, self.tau_v, self.tau_w])

    def at(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return ``(σ, τ_L)`` at height(s) ``z`` — constant, so ``z`` is ignored.

        The signature matches :func:`hanna_profiles` so the integrator can treat
        any turbulence model uniformly. ``σ`` and ``τ_L`` broadcast to the shape
        of ``z`` along a trailing axis of size 3.
        """
        z = jnp.asarray(z)
        ones = jnp.ones((*z.shape, 3))
        return ones * self.sigma, ones * self.tau


def hanna_profiles(
    z: jax.Array,
    *,
    u_star: float,
    pbl_height: float,
    obukhov_length: float,
    w_star: float = 0.0,
    coriolis_parameter: float = DEFAULT_CORIOLIS_PARAMETER,
) -> tuple[jax.Array, jax.Array]:
    """Hanna (1982) Monin–Obukhov-similarity turbulence profiles.

    Returns height-dependent velocity standard deviations and Lagrangian
    timescales for three stability regimes, chosen as in FLEXPART's
    ``hanna.f90``:

    - **Neutral** (``h/|L| < 1``, or ``L = 0``): Stohl et al. (2005) eqs. 26–28,
      σ_u = 2.0 u_* exp(−3fz/u_*), σ_v = σ_w = 1.3 u_* exp(−2fz/u_*),
      τ_L = 0.5 z / σ_w / (1 + 15fz/u_*) on all three axes.
    - **Unstable / convective** (``h/|L| ≥ 1`` and ``L < 0``): convective
      scaling with ``w_*``.
    - **Stable** (``h/|L| ≥ 1`` and ``L > 0``): eqs. 29–33,
      σ_u = 2.0 u_* (1 − z/h), σ_v = σ_w = 1.3 u_* (1 − z/h),
      τ_L,u = 0.15 (h/σ_u)(z/h)^0.5, τ_L,v = 0.07 (h/σ_v)(z/h)^0.5,
      τ_L,w = 0.1 (h/σ_w)(z/h)^0.8.

    The stable τ_L,w exponent is 0.8, as in Hanna (1982) and ``hanna.f90``;
    Stohl et al. (2005) print 0.5 in eq. 33. ``L = 0`` is the neutral sentinel
    (``1/L = 0``) that :func:`plumax.met.hanna_turbulence_from` passes.

    Args:
        z: Height(s) above ground [m]; any shape.
        u_star: Friction velocity [m/s].
        pbl_height: Boundary-layer height ``h`` [m].
        obukhov_length: Monin–Obukhov length ``L`` [m] (``<0`` unstable,
            ``0`` neutral).
        w_star: Convective velocity scale [m/s] (used when unstable).
        coriolis_parameter: Coriolis parameter f = 2Ω sin φ [1/s], used by the
            neutral profiles; only ``|f|`` enters. The default ``1e-4`` s⁻¹
            (latitude ≈ 43°) is the value FLEXPART's ``hanna.f90`` hard-codes;
            pass ``2 · 7.292e-5 · sin(latitude)`` for a specific site.

    Returns:
        ``(sigma, tau)`` arrays, each of shape ``z.shape + (3,)``: velocity
        standard deviations [m/s] and Lagrangian timescales [s] for
        ``(u, v, w)``.

    References:
        Hanna, S. R. (1982). Applications in air pollution modeling.
        Stohl, A. et al. (2005). Technical note: The Lagrangian particle
        dispersion model FLEXPART version 6.2. *Atmos. Chem. Phys.* 5,
        2461–2474, eqs. 20–33.
    """
    height = jnp.asarray(z, dtype=float)
    # ζ = z / h, held inside [0, 1]
    relative_height = jnp.clip(height / pbl_height, 0.0, 1.0)
    # u_* ← max(u_*, ε)
    u_star_safe = jnp.maximum(u_star, _SIGMA_FLOOR)
    # |f|
    coriolis = jnp.abs(coriolis_parameter)

    # --- Unstable (convective), L < 0 ---
    # σ_u² = (12 + h / 2|L|)^(2/3) u_*²
    sig_u2_conv = (12.0 + 0.5 * pbl_height / jnp.maximum(-obukhov_length, 1e-6)) ** (
        2.0 / 3.0
    ) * u_star**2
    # σ_w² = 1.74 u_*² (1 − 0.8 ζ)² + 0.35 w_*² ζ^(2/3) (1 − 0.8 ζ)²
    sig_w2_conv = 1.74 * u_star**2 * (1.0 - 0.8 * relative_height) ** 2 + (
        0.35
        * w_star**2
        * (relative_height ** (2.0 / 3.0))
        * (1.0 - 0.8 * relative_height) ** 2
    )
    sig_w_conv = jnp.sqrt(jnp.maximum(sig_w2_conv, 0.0))
    # σ_u,conv = √σ_u²
    sig_uv_conv = jnp.sqrt(jnp.maximum(sig_u2_conv, 0.0)) * jnp.ones_like(height)
    # τ_L,w = 0.15 h / σ_w
    timescale_w_conv = 0.15 * pbl_height / jnp.maximum(sig_w_conv, _SIGMA_FLOOR)
    # τ_L,uv = 0.15 h / σ_uv
    timescale_uv_conv = 0.15 * pbl_height / jnp.maximum(sig_uv_conv, _SIGMA_FLOOR)

    # --- Stable, L > 0: Stohl et al. (2005) eqs. 29–33 ---
    # 1 − ζ, floored
    decay = jnp.maximum(1.0 - relative_height, 1e-3)
    # σ_u = 2.0 u_* (1 − z/h)
    sig_u_stab = 2.0 * u_star * decay
    # σ_v = σ_w = 1.3 u_* (1 − z/h)
    sig_vw_stab = 1.3 * u_star * decay
    # τ_L,u = 0.15 (h/σ_u)(z/h)^0.5
    timescale_u_stable = (
        0.15
        * pbl_height
        / jnp.maximum(sig_u_stab, _SIGMA_FLOOR)
        * jnp.sqrt(jnp.maximum(relative_height, 1e-3))
    )
    # τ_L,v = 0.07 (h/σ_v)(z/h)^0.5
    timescale_v_stable = (
        0.07
        * pbl_height
        / jnp.maximum(sig_vw_stab, _SIGMA_FLOOR)
        * jnp.sqrt(jnp.maximum(relative_height, 1e-3))
    )
    # τ_L,w = 0.1 (h/σ_w)(z/h)^0.8
    timescale_w_stable = (
        0.1
        * pbl_height
        / jnp.maximum(sig_vw_stab, _SIGMA_FLOOR)
        * jnp.maximum(relative_height, 1e-3) ** 0.8
    )

    # --- Neutral: Stohl et al. (2005) eqs. 26–28 ---
    # σ_u = 2.0 u_* exp(−3fz/u_*)
    sig_u_neut = 2.0 * u_star * jnp.exp(-3.0 * coriolis * height / u_star_safe)
    # σ_v = σ_w = 1.3 u_* exp(−2fz/u_*)
    sig_vw_neut = 1.3 * u_star * jnp.exp(-2.0 * coriolis * height / u_star_safe)
    # τ_L = 0.5 z / σ_w / (1 + 15fz/u_*)
    timescale_neutral = (
        0.5
        * height
        / jnp.maximum(sig_vw_neut, _SIGMA_FLOOR)
        / (1.0 + 15.0 * coriolis * height / u_star_safe)
    )
    # τ_L ← max(τ_L, 1)
    timescale_neutral = jnp.maximum(timescale_neutral, 1.0)

    # --- regime: neutral when h/|L| < 1 (FLEXPART hanna.f90); L = 0 is neutral ---
    is_neutral_sentinel = obukhov_length == 0.0
    safe_length = jnp.where(is_neutral_sentinel, 1.0, obukhov_length)
    # h/|L|, with 1/L = 0 at the sentinel
    stability_ratio = jnp.where(
        is_neutral_sentinel, 0.0, pbl_height / jnp.abs(safe_length)
    )
    neutral = stability_ratio < 1.0
    unstable = jnp.logical_and(~neutral, obukhov_length < 0.0)
    stable = jnp.logical_and(~neutral, obukhov_length > 0.0)

    def pick(conv: jax.Array, stab: jax.Array, neut: jax.Array) -> jax.Array:
        return jnp.where(unstable, conv, jnp.where(stable, stab, neut))

    std_u = pick(sig_uv_conv, sig_u_stab, sig_u_neut)
    std_v = pick(0.8 * sig_uv_conv, sig_vw_stab, sig_vw_neut)
    std_w = pick(sig_w_conv, sig_vw_stab, sig_vw_neut)
    timescale_u = pick(timescale_uv_conv, timescale_u_stable, timescale_neutral)
    timescale_v = pick(timescale_uv_conv, timescale_v_stable, timescale_neutral)
    timescale_w = pick(timescale_w_conv, timescale_w_stable, timescale_neutral)

    if height.size == 0:
        # einx rejects empty axes; no heights have empty profiles
        empty_std = jnp.zeros((*height.shape, 3), dtype=std_u.dtype)
        empty_timescale = jnp.zeros((*height.shape, 3), dtype=timescale_u.dtype)
        return empty_std, empty_timescale
    # σ = [σ_u, σ_v, σ_w]
    std = einx.id("..., ..., ... -> ... (1 + 1 + 1)", std_u, std_v, std_w)
    # τ_L = [τ_u, τ_v, τ_w]
    timescale = einx.id(
        "..., ..., ... -> ... (1 + 1 + 1)", timescale_u, timescale_v, timescale_w
    )
    # τ_L ← max(τ_L, 1)
    timescale = jnp.maximum(timescale, 1.0)
    return std, timescale


@dataclass(frozen=True)
class HannaTurbulence:
    """Height-dependent Hanna (1982) turbulence as a ``TurbulenceModel``.

    Thin adapter that binds the surface-layer scales required by
    :func:`hanna_profiles` and exposes the ``at(z) -> (sigma, tau)`` interface
    the integrator (:func:`plumax.lagrangian.langevin_step`,
    :func:`~plumax.lagrangian.integrate_particles`,
    :func:`~plumax.lagrangian.compute_footprint`) expects. Because ``σ_w`` varies
    with height, the well-mixed vertical drift correction ``½(1+v_w²/σ_w²)
    ∂σ_w²/∂z`` is non-zero — the inhomogeneous path that
    :class:`HomogeneousTurbulence` never exercises.

    To drive it from met rather than hand-typed scales, build it with
    :func:`plumax.met.hanna_turbulence_from` from a
    :class:`plumax.met.MoninObukhov` diagnosis.

    Attributes:
        u_star: Friction velocity ``u_*`` [m/s].
        pbl_height: Boundary-layer height ``h`` [m].
        obukhov_length: Monin–Obukhov length ``L`` [m] (``< 0`` unstable,
            ``0`` neutral; any ``h/|L| < 1`` is neutral).
        w_star: Convective velocity scale ``w_*`` [m/s] (used when unstable).
        z_floor: Minimum height [m] at which the profiles are evaluated. The
            convective ``σ_w²`` term carries a ``(z/h)^{2/3}`` factor whose
            ``∂/∂z`` is singular at the ground, so the integrator's well-mixed
            drift ``½(1+v_w²/σ_w²)∂σ_w²/∂z`` would be ``inf``/``NaN`` for a
            ground-level source or a particle reflected exactly to ``z = 0``.
            Flooring ``z`` keeps the drift finite; the default ``0.1 m`` sits at
            the roughness / surface-layer scale where the similarity profiles are
            not valid anyway.
        coriolis_parameter: Coriolis parameter ``f`` [1/s] for the neutral
            profiles; defaults to FLEXPART's ``1e-4`` (see :func:`hanna_profiles`).
    """

    u_star: float
    pbl_height: float
    obukhov_length: float
    w_star: float = 0.0
    z_floor: float = 0.1
    coriolis_parameter: float = DEFAULT_CORIOLIS_PARAMETER

    def __post_init__(self) -> None:
        require_positive("z_floor", self.z_floor, caller="HannaTurbulence")

    def at(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return ``(σ, τ_L)`` at height(s) ``z`` from the Hanna (1982) profiles.

        ``z`` is floored at :attr:`z_floor` before evaluation so the vertical
        drift stays finite at the ground (see the class docstring). ``σ`` and
        ``τ_L`` have a trailing axis of size 3, matching
        :meth:`HomogeneousTurbulence.at`.
        """
        z = jnp.maximum(jnp.asarray(z, dtype=float), self.z_floor)
        return hanna_profiles(
            z,
            u_star=self.u_star,
            pbl_height=self.pbl_height,
            obukhov_length=self.obukhov_length,
            w_star=self.w_star,
            coriolis_parameter=self.coriolis_parameter,
        )


__all__ = ["HannaTurbulence", "HomogeneousTurbulence", "hanna_profiles"]
