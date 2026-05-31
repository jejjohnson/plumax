"""Multi-instrument fusion: closed-form joint inversion of ``(Q, bias_inst)``.

This is the Tier IV v1 inference (design build-order step 1 + the
[linear-conditional-Gaussian limit](https://github.com/jejjohnson/plumax/blob/main/docs/design/05_tier4_coupled.md#tier4-validation)).
Conditioned on the source location, wind and stability, the Gaussian-plume
forward is **linear in the emission rate ``Q``**, and the per-instrument bias
enters additively, so the joint state

    θ = (Q, bias₁, …, bias_M)

has a closed-form Gaussian posterior given a Gaussian prior and Gaussian
observation errors. Stacking every instrument's receptors, the forward is the
affine map

    y_stacked = G θ + c_bg,     G[:, 0] = responses,  G[:, 1+i] = 1 on instrument i,

and the BLUE / Kalman update gives ``θ*`` and its covariance ``P*``. Per the
design, the per-instrument **bias is a first-class state element** (documented
inter-instrument biases are O(±10 ppb); ignoring them double-counts agreement),
which is why it is solved jointly with ``Q`` here rather than pre-subtracted.

The estimator is pure JAX and differentiable; it reuses
:func:`plumax.lagrangian.inversion.observation_covariance` semantics implicitly
via the per-instrument :meth:`Instrument.observation_variance`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from plumax.coupled.forward import CoupledForward


@dataclass(frozen=True)
class FusionPosterior:
    """Closed-form joint posterior over ``(Q, bias₁ … bias_M)``.

    All numeric fields are kept as JAX arrays (the scalars as 0-d arrays) so the
    whole posterior stays differentiable / jittable — the design advertises the
    estimator for gradient-based workflows, and forcing ``float(...)`` here would
    raise a concretization error under ``jax.grad`` / ``jax.jit``. Use
    ``float(post.emission_rate)`` at the call site when a Python scalar is wanted.

    Attributes:
        emission_rate: Posterior mean emission ``Q*`` [kg/s], 0-d array.
        emission_std: Posterior standard deviation of ``Q``, 0-d array.
        biases: Posterior mean per-instrument biases, shape ``(n_inst,)``.
        bias_std: Posterior standard deviations of the biases, shape
            ``(n_inst,)``.
        mean: Full posterior mean state ``[Q, bias₁ … bias_M]``, shape
            ``(1 + n_inst,)``.
        covariance: Full posterior covariance ``P*``, shape
            ``(1 + n_inst, 1 + n_inst)``.
        instrument_names: Names aligned with the bias entries.
    """

    emission_rate: jax.Array
    emission_std: jax.Array
    biases: jax.Array
    bias_std: jax.Array
    mean: jax.Array
    covariance: jax.Array
    instrument_names: tuple[str, ...]


def _stack_design(
    forward: CoupledForward,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the stacked design matrix ``G``, observation variance and background.

    ``G`` has shape ``(n_total_obs, 1 + n_inst)``: column 0 is the concatenated
    unit-``Q`` responses; column ``1 + i`` is the indicator (ones) for the rows
    belonging to instrument ``i`` (its additive bias). Returns ``(G, r, bg)``
    with ``r`` the stacked observation variance and ``bg`` the stacked
    background offset.
    """
    n_inst = len(forward.instruments)
    cols_bias = []
    response_blocks = []
    var_blocks = []
    for i, (inst, resp) in enumerate(
        zip(forward.instruments, forward.responses, strict=True)
    ):
        n = inst.n_obs
        response_blocks.append(resp)
        var_blocks.append(inst.observation_variance())
        indicator = jnp.zeros((n, n_inst)).at[:, i].set(1.0)
        cols_bias.append(indicator)
    response = jnp.concatenate(response_blocks)[:, None]  # (n_total, 1)
    bias_cols = jnp.concatenate(cols_bias, axis=0)  # (n_total, n_inst)
    g = jnp.concatenate([response, bias_cols], axis=1)  # (n_total, 1 + n_inst)
    r = jnp.concatenate(var_blocks)  # (n_total,)
    bg = jnp.asarray(forward.source.background, dtype=float)
    return g, r, bg


def fuse_observations(
    forward: CoupledForward,
    observations: Sequence[jax.Array],
    *,
    prior_mean: jax.Array,
    prior_covariance: jax.Array,
) -> FusionPosterior:
    """Closed-form joint posterior over ``(Q, bias_inst)`` from fused observations.

    Solves the BLUE update for the affine forward ``y = G θ + c_bg`` with a
    Gaussian prior ``θ ~ N(θ_b, B)`` and diagonal observation covariance ``R``
    (the per-instrument ``R_retr + R_repr + R_align``)::

        θ* = θ_b + B Gᵀ (G B Gᵀ + R)⁻¹ (y − G θ_b − c_bg)
        P* = B − B Gᵀ (G B Gᵀ + R)⁻¹ G B

    solved in the ``(n_total_obs × n_total_obs)`` innovation form via Cholesky.

    Args:
        forward: The assembled multi-instrument coupled forward.
        observations: Observed enhancement vector per instrument, aligned with
            ``forward.instruments`` (each ``(n_obs_i,)``).
        prior_mean: Prior mean state ``[Q_b, bias_b₁ … bias_b_M]``, shape
            ``(1 + n_inst,)`` (biases typically prior-zero).
        prior_covariance: Prior covariance ``B``, shape
            ``(1 + n_inst, 1 + n_inst)`` — e.g. ``diag(σ_Q², σ_bias², …)``.

    Returns:
        The :class:`FusionPosterior`.

    Raises:
        ValueError: on a per-instrument observation-count mismatch or a
            mis-shaped prior.
    """
    n_inst = len(forward.instruments)
    n_state = 1 + n_inst
    theta_b = jnp.asarray(prior_mean, dtype=float)
    b = jnp.asarray(prior_covariance, dtype=float)
    if theta_b.shape != (n_state,):
        raise ValueError(
            f"fuse_observations: `prior_mean` must be (1 + n_inst = {n_state},); "
            f"got {theta_b.shape}."
        )
    if b.shape != (n_state, n_state):
        raise ValueError(
            f"fuse_observations: `prior_covariance` must be ({n_state}, {n_state}); "
            f"got {b.shape}."
        )
    if len(observations) != n_inst:
        raise ValueError(
            f"fuse_observations: got {len(observations)} observation vectors for "
            f"{n_inst} instruments."
        )
    obs_blocks = []
    for inst, obs in zip(forward.instruments, observations, strict=True):
        o = jnp.asarray(obs, dtype=float)
        if o.shape != (inst.n_obs,):
            raise ValueError(
                f"fuse_observations: observation for {inst.name!r} must be "
                f"({inst.n_obs},); got {o.shape}."
            )
        obs_blocks.append(o)
    y = jnp.concatenate(obs_blocks)

    g, r, bg = _stack_design(forward)
    innovation = y - (g @ theta_b + bg)

    # Information-form BLUE, solved in the tiny (1 + n_inst)-dim state space.
    # For fusion n_obs ≫ n_state, so the innovation form (n_obs × n_obs) would be
    # large *and* ill-conditioned (G B Gᵀ has rank ≤ n_state, so adding a small
    # R gives a near-singular system). The information form inverts only the
    # (n_state × n_state) posterior precision, which is well-conditioned:
    #     P*⁻¹ = B⁻¹ + Gᵀ R⁻¹ G,   θ* = θ_b + P* Gᵀ R⁻¹ (y − G θ_b − c_bg).
    r_inv = 1.0 / r
    gt_rinv = g.T * r_inv[None, :]  # Gᵀ R⁻¹, (n_state, n_obs)
    precision = jnp.linalg.inv(b) + gt_rinv @ g  # B⁻¹ + Gᵀ R⁻¹ G
    precision = 0.5 * (precision + precision.T)
    posterior_cov = jnp.linalg.inv(precision)
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
    increment = posterior_cov @ (gt_rinv @ innovation)
    theta = theta_b + increment

    std = jnp.sqrt(jnp.clip(jnp.diag(posterior_cov), 0.0, None))
    names = tuple(inst.name for inst in forward.instruments)
    return FusionPosterior(
        emission_rate=theta[0],
        emission_std=std[0],
        biases=theta[1:],
        bias_std=std[1:],
        mean=theta,
        covariance=posterior_cov,
        instrument_names=names,
    )


def default_prior(
    *,
    n_instruments: int,
    emission_prior_mean: float,
    emission_prior_std: float,
    bias_prior_std: float,
) -> tuple[jax.Array, jax.Array]:
    """Build a diagonal prior ``(θ_b, B)`` for ``(Q, bias₁ … bias_M)``.

    A convenience for the common case: a Gaussian prior on ``Q`` and zero-mean
    Gaussian biases (the design's ``bias_inst ~ N(0, σ²_inst)``).

    Args:
        n_instruments: Number of instruments (bias slots).
        emission_prior_mean: Prior mean for ``Q`` [kg/s].
        emission_prior_std: Prior std for ``Q`` (``> 0``).
        bias_prior_std: Prior std for each per-instrument bias (``> 0``).

    Returns:
        ``(prior_mean, prior_covariance)`` ready for :func:`fuse_observations`.
    """
    if emission_prior_std <= 0.0 or bias_prior_std <= 0.0:
        raise ValueError("default_prior: prior standard deviations must be > 0.")
    mean = jnp.concatenate([jnp.array([emission_prior_mean]), jnp.zeros(n_instruments)])
    variances = np.concatenate(
        [[emission_prior_std**2], np.full(n_instruments, bias_prior_std**2)]
    )
    return mean, jnp.diag(jnp.asarray(variances, dtype=float))


__all__ = ["FusionPosterior", "default_prior", "fuse_observations"]
