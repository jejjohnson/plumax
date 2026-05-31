"""Markov-1 Langevin particle integrator (the Tier II forward dynamics).

Operational Lagrangian particle dispersion models (FLEXPART, STILT, HYSPLIT)
integrate a **Markov-1** process: a Langevin equation on the turbulent
*velocity* perturbation, with position advected by mean wind plus that
perturbation,

    dv = a(x, v, t) dt + b(x, t) dW,     dx = (u(x, t) + v) dt.

For Gaussian turbulence the well-mixed drift (Thomson 1987) is

    a_i = -v_i / τ_i  +  ½ (1 + v_i² / σ_i²) ∂σ_i²/∂z   (vertical inhomogeneity),
    b_i = sqrt(2 σ_i² / τ_i).

This module integrates the *linear* (OU) part with its **exact** transition —
``v ← e^{-Δt/τ} v + σ√(1 - e^{-2Δt/τ}) ξ`` — so a homogeneous run is exactly
well mixed (stationary velocity variance ``σ²`` with no time-step bias), and
adds the inhomogeneous vertical drift correction with an Euler step. The
correction's ``∂σ_w²/∂z`` is obtained by autodiff of the turbulence profile, so
it is zero for :class:`~plumax.lagrangian.turbulence.HomogeneousTurbulence` and
nonzero for :func:`~plumax.lagrangian.turbulence.hanna_profiles` automatically.

Vertical motion reflects off the ground (``z = 0``) and, when a boundary-layer
height is supplied, off the PBL top — flipping both ``z`` and ``v_w``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable


class TurbulenceModel(Protocol):
    """Structural interface for a turbulence parameterisation.

    ``at(z)`` returns ``(sigma, tau)`` with a trailing axis of size 3, matching
    :class:`~plumax.lagrangian.turbulence.HomogeneousTurbulence`.
    """

    def at(self, z: jax.Array) -> tuple[jax.Array, jax.Array]: ...


@dataclass(frozen=True)
class ParticleState:
    """State of an ensemble of Lagrangian particles.

    Registered as a JAX pytree so it can be carried through ``lax.scan`` /
    ``vmap``; ``position`` and ``velocity`` are the (array) children.

    Attributes:
        position: Particle positions ``(x, y, z)`` [m], shape ``(n, 3)``.
        velocity: Turbulent velocity perturbations [m/s], shape ``(n, 3)``.
    """

    position: jax.Array
    velocity: jax.Array

    @property
    def n_particles(self) -> int:
        return int(self.position.shape[0])


jax.tree_util.register_dataclass(
    ParticleState, data_fields=["position", "velocity"], meta_fields=[]
)


def _sigma_w_sq(turbulence: TurbulenceModel, z: jax.Array) -> jax.Array:
    sigma, _ = turbulence.at(z)
    return sigma[..., 2] ** 2


def langevin_step(
    state: ParticleState,
    wind: jax.Array,
    turbulence: TurbulenceModel,
    dt: float,
    key: jax.Array,
    *,
    pbl_height: float | None = None,
) -> ParticleState:
    """Advance the ensemble one step ``dt`` with the Markov-1 update.

    Args:
        state: Current particle state.
        wind: Mean wind ``(u, v, w)`` [m/s] at this step, shape ``(3,)``.
        turbulence: Turbulence model supplying ``σ(z)`` and ``τ_L(z)``.
        dt: Time step [s].
        key: PRNG key for the Wiener increment.
        pbl_height: If given, reflect particles off the PBL top [m].

    Returns:
        The advanced :class:`ParticleState`.
    """
    pos, vel = state.position, state.velocity
    z = pos[:, 2]

    sigma, tau = turbulence.at(z)  # each (n, 3)
    sigma = jnp.maximum(sigma, 1e-12)

    decay = jnp.exp(-dt / tau)  # (n, 3)
    ou_std = sigma * jnp.sqrt(jnp.clip(1.0 - decay**2, 0.0, None))
    noise = jax.random.normal(key, vel.shape)
    vel_ou = decay * vel + ou_std * noise

    # Well-mixed vertical drift correction: ½(1 + v_w²/σ_w²) ∂σ_w²/∂z · dt.
    dsigw2_dz = jax.vmap(jax.grad(lambda zz: _sigma_w_sq(turbulence, zz)))(z)
    vw = vel_ou[:, 2]
    drift_w = 0.5 * (1.0 + vw**2 / sigma[:, 2] ** 2) * dsigw2_dz * dt
    vel_new = vel_ou.at[:, 2].add(drift_w)

    pos_new = pos + (wind[None, :] + vel_new) * dt

    # Ground reflection at z = 0.
    zc = pos_new[:, 2]
    vc = vel_new[:, 2]
    below = zc < 0.0
    zc = jnp.where(below, -zc, zc)
    vc = jnp.where(below, -vc, vc)
    # PBL-top reflection at z = h.
    if pbl_height is not None:
        above = zc > pbl_height
        zc = jnp.where(above, 2.0 * pbl_height - zc, zc)
        vc = jnp.where(above, -vc, vc)
    pos_new = pos_new.at[:, 2].set(zc)
    vel_new = vel_new.at[:, 2].set(vc)

    return ParticleState(position=pos_new, velocity=vel_new)


def step_durations(horizon: float, dt: float, n_steps: int) -> jax.Array:
    """Per-step durations summing to ``horizon``; the final step takes the rest.

    All steps are ``dt`` except the last, which is the remainder
    ``horizon - (n_steps - 1) * dt`` (equal to ``dt`` when ``horizon`` is an
    exact multiple). Returns an empty array when ``n_steps == 0``. Used by the
    integrator / residence / footprint paths so a non-divisible horizon advances
    over exactly ``horizon`` rather than ``n_steps * dt``.
    """
    if n_steps == 0:
        return jnp.zeros((0,))
    dts = np.full(n_steps, dt, dtype=float)
    dts[-1] = horizon - dt * (n_steps - 1)
    return jnp.asarray(dts)


def integrate_particles(
    state: ParticleState,
    wind: Callable[[jax.Array], jax.Array],
    turbulence: TurbulenceModel,
    *,
    t0: float,
    t1: float,
    dt: float,
    key: jax.Array,
    pbl_height: float | None = None,
    save_trajectory: bool = False,
) -> tuple[ParticleState, jax.Array | None]:
    """Integrate the ensemble from ``t0`` to ``t1`` with fixed step ``dt``.

    Args:
        state: Initial particle state.
        wind: Mean-wind field ``t -> (u, v, w)`` [m/s], spatially uniform.
        turbulence: Turbulence model.
        t0: Start time [s].
        t1: End time [s].
        dt: Time step [s].
        key: PRNG key.
        pbl_height: Optional PBL-top reflecting lid [m].
        save_trajectory: If ``True``, also return the stacked positions at every
            step, shape ``(n_steps + 1, n, 3)``.

    Returns:
        ``(final_state, trajectory)`` where ``trajectory`` is ``None`` unless
        ``save_trajectory`` is set.
    """
    n_steps = max(int(np.ceil((t1 - t0) / dt)), 0)
    keys = jax.random.split(key, n_steps)
    # Step start times and per-step durations. When ``t1 - t0`` is not an exact
    # multiple of ``dt`` the final step is shortened to the remainder so the
    # ensemble is advanced to exactly ``t1`` (not ``t0 + n_steps * dt``).
    times = t0 + dt * jnp.arange(n_steps)
    dts = step_durations(t1 - t0, dt, n_steps)

    def body(
        carry: ParticleState, inputs: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[ParticleState, jax.Array]:
        t, k, dt_i = inputs
        nxt = langevin_step(
            carry, jnp.asarray(wind(t)), turbulence, dt_i, k, pbl_height=pbl_height
        )
        return nxt, nxt.position

    final, positions = jax.lax.scan(body, state, (times, keys, dts))
    if not save_trajectory:
        return final, None
    trajectory = jnp.concatenate([state.position[None], positions], axis=0)
    return final, trajectory


def uniform_wind(
    u: float, v: float, w: float = 0.0
) -> Callable[[jax.Array], jax.Array]:
    """Return a constant mean-wind field ``t -> (u, v, w)``."""
    vec = jnp.array([u, v, w])

    def field(t: jax.Array) -> jax.Array:
        del t
        return vec

    return field


def wind_from_speed_direction(
    speed: float, direction_deg: float, w: float = 0.0
) -> Callable[[jax.Array], jax.Array]:
    """Constant wind from speed + meteorological direction (degrees *from* North).

    ``direction_deg = 270`` is a wind from the west (flowing east), matching the
    convention used by :func:`plumax.gauss_plume.simulate_plume`.
    """
    theta = jnp.deg2rad(direction_deg)
    u = -speed * jnp.sin(theta)
    v = -speed * jnp.cos(theta)
    return uniform_wind(float(u), float(v), w)


__all__ = [
    "ParticleState",
    "TurbulenceModel",
    "integrate_particles",
    "langevin_step",
    "uniform_wind",
    "wind_from_speed_direction",
]
