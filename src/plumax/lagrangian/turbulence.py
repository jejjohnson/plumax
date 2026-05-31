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
        ones = jnp.ones(z.shape + (3,))
        return ones * self.sigma, ones * self.tau


def hanna_profiles(
    z: jax.Array,
    *,
    u_star: float,
    pbl_height: float,
    obukhov_length: float,
    w_star: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Hanna (1982) Monin–Obukhov-similarity turbulence profiles.

    Returns height-dependent velocity standard deviations and Lagrangian
    timescales for the three stability regimes selected by the sign of the
    Obukhov length ``L``:

    - **Unstable / convective** (``L < 0``): convective scaling with ``w_*``.
    - **Stable** (``L > 0``): surface-layer scaling decaying with ``z / h``.
    - **Neutral** (``|L|`` large): ``σ ∝ u_*`` with an exponential PBL decay.

    Args:
        z: Height(s) above ground [m]; any shape.
        u_star: Friction velocity [m/s].
        pbl_height: Boundary-layer height ``h`` [m].
        obukhov_length: Monin–Obukhov length ``L`` [m] (``<0`` unstable).
        w_star: Convective velocity scale [m/s] (used when ``L < 0``).

    Returns:
        ``(sigma, tau)`` arrays, each of shape ``z.shape + (3,)``: velocity
        standard deviations [m/s] and Lagrangian timescales [s] for
        ``(u, v, w)``.
    """
    z = jnp.asarray(z, dtype=float)
    h = pbl_height
    zh = jnp.clip(z / h, 0.0, 1.0)
    us = u_star
    ws = w_star

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

    # --- Stable, L > 0 ---
    decay = jnp.maximum(1.0 - zh, 1e-3)
    sig_uv_stab = 2.0 * us * decay
    sig_w_stab = 1.3 * us * decay
    tau_uv_stab = 0.07 * h / jnp.maximum(us, _SIGMA_FLOOR) * jnp.sqrt(zh + 1e-3)
    tau_w_stab = 0.1 * h / jnp.maximum(us, _SIGMA_FLOOR) * (zh + 1e-3) ** 0.8

    # --- Neutral ---
    exp_decay = jnp.exp(-2.0 * zh)
    sig_uv_neut = 2.0 * us * exp_decay
    sig_w_neut = 1.3 * us * exp_decay
    f_c = 1e-4  # Coriolis parameter [1/s], mid-latitude default
    tau_neut = (
        0.5
        * z
        / jnp.maximum(sig_w_neut, _SIGMA_FLOOR)
        / (1.0 + 15.0 * f_c * z / jnp.maximum(us, _SIGMA_FLOOR))
    )
    tau_neut = jnp.maximum(tau_neut, 1.0)

    unstable = obukhov_length < 0.0
    stable = obukhov_length > 0.0

    def pick(conv: jax.Array, stab: jax.Array, neut: jax.Array) -> jax.Array:
        return jnp.where(unstable, conv, jnp.where(stable, stab, neut))

    sigma_u = pick(sig_uv_conv, sig_uv_stab, sig_uv_neut)
    sigma_v = pick(0.8 * sig_uv_conv, sig_uv_stab, sig_uv_neut)
    sigma_w = pick(sig_w_conv, sig_w_stab, sig_w_neut)
    tau_u = pick(tau_uv_conv, tau_uv_stab, tau_neut)
    tau_v = pick(tau_uv_conv, tau_uv_stab, tau_neut)
    tau_w = pick(tau_w_conv, tau_w_stab, tau_neut)

    sigma = jnp.stack([sigma_u, sigma_v, sigma_w], axis=-1)
    tau = jnp.stack([tau_u, tau_v, tau_w], axis=-1)
    tau = jnp.maximum(tau, 1.0)
    return sigma, tau


__all__ = ["HomogeneousTurbulence", "hanna_profiles"]
