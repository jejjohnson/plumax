"""Tier II — model-based inversion of the source vector from observations.

Given a source–receptor sensitivity (footprint) matrix ``F`` — the linear map
from a discretised emission field ``q`` to predicted observations
``y_pred = F q`` (see :func:`plumax.lagrangian.compute_footprint`) — this module
recovers a Bayesian posterior over ``q`` from observed column enhancements.

Two estimators, both following the Tier II design note
([design](https://github.com/jejjohnson/plumax/blob/main/docs/design/02_tier2_lagrangian.md#tier2-inference)):

- :func:`linear_gaussian_inversion` — the closed-form Gaussian–Gaussian
  (BLUE / Kalman) update for a Gaussian prior ``q ~ N(q_a, B)`` and Gaussian
  likelihood ``y = F q + ε``, ``ε ~ N(0, R)``::

      q* = q_a + B Fᵀ (F B Fᵀ + R)⁻¹ (y − F q_a)
      P* = B − B Fᵀ (F B Fᵀ + R)⁻¹ F B

- :func:`lognormal_inversion` — the sign-constrained ``log q ~ N(log q_a,
  B_log)`` variant, linearised about ``log q_a`` (the design's recommended v1
  prior). With the tangent-linear Jacobian ``F̃ = F · diag(q_a)`` the log-space
  increment is the same BLUE update and ``q* = q_a · exp(δ*) ≥ 0``.

The spatial prior is built with :func:`matern32_covariance` — a Matérn-3/2
kernel with a real correlation length, as the design note insists (a diagonal
``B`` produces single-cell-spike posteriors). The likelihood covariance is
assembled by :func:`observation_covariance`, which adds the **representation
error** to the retrieval error rather than using the retrieval error alone.

The estimators solve the ``(n_obs × n_obs)`` innovation system, which is the
efficient ordering when ``n_obs ≪ n_grid`` (the usual overpass regime). They
are pure JAX, so they compose under ``jit`` / ``vmap`` and are differentiable
through the posterior mean (e.g. for hyperparameter tuning of ``ℓ`` or ``σ``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GaussianPosterior:
    """Closed-form Gaussian posterior over the source vector.

    Attributes:
        mean: Posterior mean emission field ``q*`` [same units as the prior
            mean], shape ``(n_grid,)``.
        covariance: Posterior covariance ``P*`` [units²], shape
            ``(n_grid, n_grid)``.
    """

    mean: jax.Array
    covariance: jax.Array

    @property
    def std(self) -> jax.Array:
        """Posterior per-cell standard deviation ``sqrt(diag(P*))``."""
        return jnp.sqrt(jnp.clip(jnp.diag(self.covariance), 0.0, None))


def matern32_covariance(
    coordinates: jax.Array,
    *,
    variance: float,
    length_scale: float,
) -> jax.Array:
    """Build a Matérn-3/2 prior covariance over grid-cell coordinates.

    The Matérn-3/2 kernel

        k(r) = σ² (1 + √3 r / ℓ) exp(−√3 r / ℓ),   r = ‖xᵢ − xⱼ‖

    gives a once-differentiable random field — smoother than the exponential
    (AR(1) / Matérn-1/2) kernel but far less rigid than a squared-exponential.
    It is the design note's recommended spatial prior for regional emission
    inversions: a real correlation length ``ℓ`` regularises the posterior so
    unresolved structure spreads over neighbouring cells instead of collapsing
    into single-cell spikes.

    Args:
        coordinates: Cell-centre coordinates [m], shape ``(n_grid, n_dim)``
            (1-D coordinates of shape ``(n_grid,)`` are accepted and promoted).
        variance: Marginal prior variance ``σ²`` (per cell), ``> 0``.
        length_scale: Correlation length ``ℓ`` [m], ``> 0``.

    Returns:
        The ``(n_grid, n_grid)`` covariance matrix, symmetric positive-definite.
    """
    if variance <= 0.0:
        raise ValueError("matern32_covariance: `variance` must be > 0.")
    if length_scale <= 0.0:
        raise ValueError("matern32_covariance: `length_scale` must be > 0.")
    coords = jnp.asarray(coordinates, dtype=float)
    if coords.ndim == 1:
        coords = coords[:, None]
    diff = coords[:, None, :] - coords[None, :, :]
    r = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-300)
    scaled = jnp.sqrt(3.0) * r / length_scale
    return variance * (1.0 + scaled) * jnp.exp(-scaled)


def observation_covariance(
    retrieval_variance: jax.Array,
    *,
    representation_variance: float | jax.Array = 0.0,
) -> jax.Array:
    """Assemble the diagonal observation-error covariance ``R``.

    ``R = R_retr + R_repr``: the per-pixel L2 **retrieval** error plus a
    **representation** error that accounts for model-vs-observation footprint
    mismatch. The design note is explicit that omitting ``R_repr`` (using the
    retrieval error alone) overweights the observations and yields overconfident
    posteriors, so the representation term is a first-class argument here.

    Args:
        retrieval_variance: Per-observation retrieval variance ``σ²_retr``,
            shape ``(n_obs,)``. Must be ``> 0``.
        representation_variance: Representation variance ``σ²_repr`` added to
            every observation — a scalar or a ``(n_obs,)`` array. Default ``0``.

    Returns:
        The diagonal of ``R``, shape ``(n_obs,)``.
    """
    retr = jnp.asarray(retrieval_variance, dtype=float)
    if jnp.ndim(retr) != 1:
        raise ValueError("observation_covariance: `retrieval_variance` must be 1-D.")
    repr_ = jnp.asarray(representation_variance, dtype=float)
    return retr + repr_


def _blue_update(
    jacobian: jax.Array,
    innovation: jax.Array,
    prior_cov: jax.Array,
    obs_var: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve the BLUE increment and posterior covariance via the innovation form.

    Returns ``(increment, posterior_cov)`` where
    ``increment = B Fᵀ (F B Fᵀ + R)⁻¹ innovation`` and
    ``posterior_cov = B − B Fᵀ (F B Fᵀ + R)⁻¹ F B``. ``R`` is diagonal, given by
    ``obs_var``. The ``(n_obs × n_obs)`` system is solved with a Cholesky factor
    for symmetry and speed (the efficient ordering when ``n_obs ≪ n_grid``).
    """
    bft = prior_cov @ jacobian.T  # B Fᵀ, (n_grid, n_obs)
    s = jacobian @ bft  # F B Fᵀ, (n_obs, n_obs)
    s = s + jnp.diag(obs_var)  # + R
    # Symmetrise to kill round-off asymmetry before the Cholesky solve.
    s = 0.5 * (s + s.T)
    chol = jax.scipy.linalg.cho_factor(s, lower=True)
    gain_t = jax.scipy.linalg.cho_solve(chol, bft.T)  # (F B Fᵀ + R)⁻¹ F B
    increment = bft @ jax.scipy.linalg.cho_solve(chol, innovation)
    posterior_cov = prior_cov - bft @ gain_t
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
    return increment, posterior_cov


def linear_gaussian_inversion(
    footprint: jax.Array,
    observation: jax.Array,
    *,
    prior_mean: jax.Array,
    prior_cov: jax.Array,
    obs_variance: jax.Array,
) -> GaussianPosterior:
    """Closed-form Gaussian–Gaussian (BLUE) inversion of the source vector.

    For a Gaussian prior ``q ~ N(q_a, B)`` and Gaussian likelihood
    ``y = F q + ε``, ``ε ~ N(0, R)`` (``R`` diagonal), returns the exact
    posterior ``N(q*, P*)`` with::

        q* = q_a + B Fᵀ (F B Fᵀ + R)⁻¹ (y − F q_a)
        P* = B − B Fᵀ (F B Fᵀ + R)⁻¹ F B

    Args:
        footprint: Source–receptor matrix ``F``, shape ``(n_obs, n_grid)``.
        observation: Observed values ``y``, shape ``(n_obs,)``.
        prior_mean: Prior mean ``q_a``, shape ``(n_grid,)``.
        prior_cov: Prior covariance ``B``, shape ``(n_grid, n_grid)`` (e.g. from
            :func:`matern32_covariance`).
        obs_variance: Diagonal of ``R``, shape ``(n_obs,)`` (e.g. from
            :func:`observation_covariance`).

    Returns:
        The :class:`GaussianPosterior` over ``q``.
    """
    f = jnp.asarray(footprint, dtype=float)
    y = jnp.asarray(observation, dtype=float)
    qa = jnp.asarray(prior_mean, dtype=float)
    b = jnp.asarray(prior_cov, dtype=float)
    r = jnp.asarray(obs_variance, dtype=float)
    _check_inversion_shapes(f, y, qa, b, r)

    innovation = y - f @ qa
    increment, posterior_cov = _blue_update(f, innovation, b, r)
    return GaussianPosterior(mean=qa + increment, covariance=posterior_cov)


@dataclass(frozen=True)
class LognormalPosterior:
    """Linearised lognormal posterior over a non-negative source vector.

    Attributes:
        mean: Posterior mean emission field ``q* = q_a · exp(δ*) ≥ 0``, shape
            ``(n_grid,)``.
        log_increment: Log-space MAP increment ``δ* = log(q*/q_a)``, shape
            ``(n_grid,)``.
        log_covariance: Posterior covariance in log-space ``P*_log``, shape
            ``(n_grid, n_grid)``.
    """

    mean: jax.Array
    log_increment: jax.Array
    log_covariance: jax.Array


def lognormal_inversion(
    footprint: jax.Array,
    observation: jax.Array,
    *,
    prior_mean: jax.Array,
    prior_log_cov: jax.Array,
    obs_variance: jax.Array,
) -> LognormalPosterior:
    """Sign-constrained inversion via a lognormal prior linearised in log-space.

    With ``log q ~ N(log q_a, B_log)`` and the forward linearised about the
    prior mean, the tangent-linear Jacobian is ``F̃ = F · diag(q_a)`` and the
    log-space increment ``δ = log(q / q_a)`` follows the same BLUE update::

        δ* = B_log F̃ᵀ (F̃ B_log F̃ᵀ + R)⁻¹ (y − F q_a)
        q* = q_a · exp(δ*)          (≥ 0 by construction)

    This is the design note's recommended v1 prior: it enforces non-negative
    emissions while keeping the update closed-form (no projected-gradient or
    NNLS inner loop). The innovation ``y − F q_a`` is the observation minus the
    forward evaluated at the prior mean (``δ = 0 ⇒ q = q_a``).

    Args:
        footprint: Source–receptor matrix ``F``, shape ``(n_obs, n_grid)``.
        observation: Observed values ``y``, shape ``(n_obs,)``.
        prior_mean: Prior median emission field ``q_a > 0``, shape
            ``(n_grid,)``.
        prior_log_cov: Log-space prior covariance ``B_log``, shape
            ``(n_grid, n_grid)``.
        obs_variance: Diagonal of ``R``, shape ``(n_obs,)``.

    Returns:
        The :class:`LognormalPosterior` over ``q``.
    """
    f = jnp.asarray(footprint, dtype=float)
    y = jnp.asarray(observation, dtype=float)
    qa = jnp.asarray(prior_mean, dtype=float)
    b_log = jnp.asarray(prior_log_cov, dtype=float)
    r = jnp.asarray(obs_variance, dtype=float)
    _check_inversion_shapes(f, y, qa, b_log, r)
    if bool(jnp.any(qa <= 0.0)):
        raise ValueError("lognormal_inversion: `prior_mean` (q_a) must be > 0.")

    jac_tilde = f * qa[None, :]  # F̃ = F diag(q_a), (n_obs, n_grid)
    innovation = y - f @ qa
    log_increment, log_cov = _blue_update(jac_tilde, innovation, b_log, r)
    mean = qa * jnp.exp(log_increment)
    return LognormalPosterior(
        mean=mean, log_increment=log_increment, log_covariance=log_cov
    )


def _check_inversion_shapes(
    footprint: jax.Array,
    observation: jax.Array,
    prior_mean: jax.Array,
    prior_cov: jax.Array,
    obs_variance: jax.Array,
) -> None:
    if footprint.ndim != 2:
        raise ValueError("inversion: `footprint` must be 2-D (n_obs, n_grid).")
    n_obs, n_grid = footprint.shape
    if observation.shape != (n_obs,):
        raise ValueError(
            f"inversion: `observation` must have shape ({n_obs},), "
            f"got {observation.shape}."
        )
    if prior_mean.shape != (n_grid,):
        raise ValueError(
            f"inversion: `prior_mean` must have shape ({n_grid},), "
            f"got {prior_mean.shape}."
        )
    if prior_cov.shape != (n_grid, n_grid):
        raise ValueError(
            f"inversion: prior covariance must have shape ({n_grid}, {n_grid}), "
            f"got {prior_cov.shape}."
        )
    if obs_variance.shape != (n_obs,):
        raise ValueError(
            f"inversion: `obs_variance` must have shape ({n_obs},), "
            f"got {obs_variance.shape}."
        )


__all__ = [
    "GaussianPosterior",
    "LognormalPosterior",
    "linear_gaussian_inversion",
    "lognormal_inversion",
    "matern32_covariance",
    "observation_covariance",
]
