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

import jax
import jax.numpy as jnp


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
            if getattr(self, name) < 0.0:
                raise ValueError(f"HomogeneousTurbulence: `{name}` must be ≥ 0")
        for name in ("tau_u", "tau_v", "tau_w"):
            if not (getattr(self, name) > 0.0):
                raise ValueError(f"HomogeneousTurbulence: `{name}` must be > 0")

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
    z = jnp.asarray(z, dtype=float)
    h = pbl_height
    zh = jnp.clip(z / h, 0.0, 1.0)
    us = u_star
    ws = w_star
    us_safe = jnp.maximum(us, _SIGMA_FLOOR)
    f_c = jnp.abs(coriolis_parameter)

    # --- Unstable (convective), L < 0 ---
    sig_u2_conv = (12.0 + 0.5 * h / jnp.maximum(-obukhov_length, 1e-6)) ** (
        2.0 / 3.0
    ) * us**2
    sig_w2_conv = 1.74 * us**2 * (1.0 - 0.8 * zh) ** 2 + (
        0.35 * ws**2 * (zh ** (2.0 / 3.0)) * (1.0 - 0.8 * zh) ** 2
    )
    sig_w_conv = jnp.sqrt(jnp.maximum(sig_w2_conv, 0.0))
    sig_uv_conv = jnp.sqrt(jnp.maximum(sig_u2_conv, 0.0)) * jnp.ones_like(z)
    tau_w_conv = 0.15 * h / jnp.maximum(sig_w_conv, _SIGMA_FLOOR)
    tau_uv_conv = 0.15 * h / jnp.maximum(sig_uv_conv, _SIGMA_FLOOR)

    # --- Stable, L > 0: Stohl et al. (2005) eqs. 29–33 ---
    decay = jnp.maximum(1.0 - zh, 1e-3)
    # σ_u = 2.0 u_* (1 − z/h)
    sig_u_stab = 2.0 * us * decay
    # σ_v = σ_w = 1.3 u_* (1 − z/h)
    sig_vw_stab = 1.3 * us * decay
    # τ_L,u = 0.15 (h/σ_u)(z/h)^0.5
    tau_u_stab = (
        0.15
        * h
        / jnp.maximum(sig_u_stab, _SIGMA_FLOOR)
        * jnp.sqrt(jnp.maximum(zh, 1e-3))
    )
    # τ_L,v = 0.07 (h/σ_v)(z/h)^0.5
    tau_v_stab = (
        0.07
        * h
        / jnp.maximum(sig_vw_stab, _SIGMA_FLOOR)
        * jnp.sqrt(jnp.maximum(zh, 1e-3))
    )
    # τ_L,w = 0.1 (h/σ_w)(z/h)^0.8
    tau_w_stab = (
        0.1 * h / jnp.maximum(sig_vw_stab, _SIGMA_FLOOR) * jnp.maximum(zh, 1e-3) ** 0.8
    )

    # --- Neutral: Stohl et al. (2005) eqs. 26–28 ---
    # σ_u = 2.0 u_* exp(−3fz/u_*)
    sig_u_neut = 2.0 * us * jnp.exp(-3.0 * f_c * z / us_safe)
    # σ_v = σ_w = 1.3 u_* exp(−2fz/u_*)
    sig_vw_neut = 1.3 * us * jnp.exp(-2.0 * f_c * z / us_safe)
    # τ_L = 0.5 z / σ_w / (1 + 15fz/u_*)
    tau_neut = (
        0.5
        * z
        / jnp.maximum(sig_vw_neut, _SIGMA_FLOOR)
        / (1.0 + 15.0 * f_c * z / us_safe)
    )
    tau_neut = jnp.maximum(tau_neut, 1.0)

    # --- regime: neutral when h/|L| < 1 (FLEXPART hanna.f90); L = 0 is neutral ---
    is_neutral_sentinel = obukhov_length == 0.0
    safe_length = jnp.where(is_neutral_sentinel, 1.0, obukhov_length)
    # h/|L|, with 1/L = 0 at the sentinel
    stability_ratio = jnp.where(is_neutral_sentinel, 0.0, h / jnp.abs(safe_length))
    neutral = stability_ratio < 1.0
    unstable = jnp.logical_and(~neutral, obukhov_length < 0.0)
    stable = jnp.logical_and(~neutral, obukhov_length > 0.0)

    def pick(conv: jax.Array, stab: jax.Array, neut: jax.Array) -> jax.Array:
        return jnp.where(unstable, conv, jnp.where(stable, stab, neut))

    sigma_u = pick(sig_uv_conv, sig_u_stab, sig_u_neut)
    sigma_v = pick(0.8 * sig_uv_conv, sig_vw_stab, sig_vw_neut)
    sigma_w = pick(sig_w_conv, sig_vw_stab, sig_vw_neut)
    tau_u = pick(tau_uv_conv, tau_u_stab, tau_neut)
    tau_v = pick(tau_uv_conv, tau_v_stab, tau_neut)
    tau_w = pick(tau_w_conv, tau_w_stab, tau_neut)

    sigma = jnp.stack([sigma_u, sigma_v, sigma_w], axis=-1)
    tau = jnp.stack([tau_u, tau_v, tau_w], axis=-1)
    tau = jnp.maximum(tau, 1.0)
    return sigma, tau


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
        if not (self.z_floor > 0.0):
            raise ValueError("HannaTurbulence: `z_floor` must be > 0")

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
