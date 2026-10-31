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

plumax builds the design matrix, the background, the prior and the instrument
error budget; the solve is a ``vardax.OptimalInterpolation`` analysis in its
state-space (information) form (the ``assim`` extra, ``pip install
'plumax[assim]'``). The estimator stays pure JAX and differentiable; it reuses
:func:`plumax.lagrangian.inversion.observation_covariance` semantics implicitly
via the per-instrument :meth:`Instrument.observation_variance`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from plumax._src.errors import PayloadError
from plumax._src.scalar_emission import GaussianEmission
from plumax._vardax import require_vardax
from plumax.coupled.forward import CoupledForward
from plumax.coupled.quality import QualityPolicy, apply_quality, full_weight_pixels


@dataclass(frozen=True)
class FusionPosterior:
    """Closed-form joint posterior over ``(Q, bias₁ … bias_M)``.

    All numeric fields are kept as JAX arrays (the scalars as 0-d arrays) so the
    whole posterior stays differentiable / jittable — the design advertises the
    estimator for gradient-based workflows, and forcing ``float(...)`` here would
    raise a concretization error under ``jax.grad`` / ``jax.jit``. Use
    ``float(post.emission_rate)`` at the call site when a Python scalar is wanted.

    With design-matrix responses (``n_param`` emission parameters, e.g. one per
    cell of a Tier II source) the emission fields are ``(n_param,)`` vectors and
    the state is ``[q₁ … q_P, bias₁ … bias_M]``.

    Attributes:
        emission_rate: Posterior mean emission ``Q*`` [kg/s], 0-d array
            (``(n_param,)`` for design-matrix responses).
        emission_std: Posterior standard deviation of ``Q``, 0-d array
            (``(n_param,)`` for design-matrix responses).
        biases: Posterior mean per-instrument biases, shape ``(n_inst,)``.
        bias_std: Posterior standard deviations of the biases, shape
            ``(n_inst,)``.
        mean: Full posterior mean state ``[Q, bias₁ … bias_M]``, shape
            ``(n_param + n_inst,)``.
        covariance: Full posterior covariance ``P*``, shape
            ``(n_param + n_inst, n_param + n_inst)``.
        instrument_names: Names aligned with the bias entries.
        pixels_used: Pixels each instrument contributed at full weight
            (unmasked, not inflated), shape ``(n_inst,)``, aligned with
            ``instrument_names``.
        prior_mean: Prior mean state θ_b the fusion used (``None`` when built
            by hand).
        prior_covariance: Prior covariance ``B`` (``None`` when built by hand).

    For a scalar Q it also carries the Tier V per-event payload
    (``population.PerEventPosterior``): Q's posterior marginal
    N(Q*, P*₀₀) and its prior marginal N(θ_b,0, B₀₀) are exact.
    """

    emission_rate: jax.Array
    emission_std: jax.Array
    biases: jax.Array
    bias_std: jax.Array
    mean: jax.Array
    covariance: jax.Array
    instrument_names: tuple[str, ...]
    pixels_used: jax.Array | None = None
    prior_mean: jax.Array | None = None
    prior_covariance: jax.Array | None = None

    def samples(self, key: jax.Array, n: int) -> jax.Array:
        """n exact draws of Q from its posterior marginal N(Q*, P*₀₀).

        Raises:
            PayloadError: For design-matrix (vector) emissions.
        """
        self._require_scalar("samples")
        return GaussianEmission(self.emission_rate, self.emission_std).sample(key, n)

    def prior_logpdf(self, q: jax.Array) -> jax.Array:
        """log π(Q) of Q's prior marginal N(θ_b,0, B₀₀).

        Raises:
            PayloadError: For vector emissions, or without the prior.
        """
        self._require_scalar("prior_logpdf")
        if self.prior_mean is None or self.prior_covariance is None:
            raise PayloadError(
                "FusionPosterior.prior_logpdf: this posterior carries no prior; "
                "build it with fuse_observations."
            )
        prior_std = jnp.sqrt(self.prior_covariance[0, 0])
        return GaussianEmission(self.prior_mean[0], prior_std).log_prob(q)

    def _require_scalar(self, method: str) -> None:
        if jnp.ndim(self.emission_rate) != 0:
            raise PayloadError(
                f"FusionPosterior.{method}: the Tier V payload is for a scalar Q; "
                "this posterior has one emission per design-matrix column."
            )


def _stack_design(
    forward: CoupledForward,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the stacked design matrix ``G``, observation variance and background.

    ``G`` has shape ``(n_total_obs, n_param + n_inst)``: the first ``n_param``
    columns are the concatenated unit-``Q`` responses (one column for a scalar
    ``Q``); column ``n_param + i`` is the indicator (ones) for the rows
    belonging to instrument ``i`` (its additive bias). Returns
    ``(G, r, m, bg)`` with ``r`` the stacked observation variance, ``m`` the
    stacked observation mask and ``bg`` the stacked background offset.
    """
    n_inst = len(forward.instruments)
    cols_bias = []
    response_blocks = []
    var_blocks = []
    mask_blocks = []
    for i, (inst, resp) in enumerate(
        zip(forward.instruments, forward.responses, strict=True)
    ):
        n = inst.n_obs
        response_blocks.append(resp[:, None] if resp.ndim == 1 else resp)
        var_blocks.append(inst.observation_variance())
        mask_blocks.append(inst.observation_mask())
        indicator = jnp.zeros((n, n_inst)).at[:, i].set(1.0)
        cols_bias.append(indicator)
    response = jnp.concatenate(response_blocks, axis=0)  # (n_total, n_param)
    bias_cols = jnp.concatenate(cols_bias, axis=0)  # (n_total, n_inst)
    g = jnp.concatenate([response, bias_cols], axis=1)  # (n_total, n_param + n_inst)
    r = jnp.concatenate(var_blocks)  # (n_total,)
    m = jnp.concatenate(mask_blocks)  # (n_total,)
    bg = jnp.asarray(forward.source.background, dtype=float)
    return g, r, m, bg


def _n_emission_params(forward: CoupledForward) -> int:
    """Emission parameters per response: 1 for a vector, its columns otherwise."""
    widths = {1 if r.ndim == 1 else int(r.shape[1]) for r in forward.responses}
    if len(widths) != 1:
        raise ValueError(
            "fuse_observations: every instrument's response must have the same "
            f"number of emission parameters; got {sorted(widths)}."
        )
    return widths.pop()


def _information_form_analysis(
    design: jax.Array,
    shifted_obs: jax.Array,
    mask: jax.Array,
    obs_variance: jax.Array,
    *,
    prior_mean: jax.Array,
    prior_covariance: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """BLUE of the stacked fusion problem as a vardax state-space OI analysis.

    θ* = θ_b + (B⁻¹ + Gᵀ R⁻¹ G)⁻¹ Gᵀ R⁻¹ (y − c − G θ_b),  P* = (B⁻¹ + Gᵀ R⁻¹ G)⁻¹.

    For fusion D_y ≫ D_x, and G B Gᵀ has rank at most D_x, so the
    observation-space system G B Gᵀ + R is large and close to singular for a
    small R. ``OptimalInterpolation(form="state")`` solves the small, well
    conditioned D_x × D_x information system instead, with a Cholesky
    factorisation; its default posterior is the inverse of that precision.

    Args:
        design: Stacked design matrix G, shape ``(n_obs, n_state)``.
        shifted_obs: y − c, the observations minus the stacked background; a
            masked pixel may hold anything, NaN included.
        mask: Stacked 0/1 observation mask m; vardax zeroes the masked rows
            of G and of the innovation, so they carry no weight.
        obs_variance: Diagonal of R, shape ``(n_obs,)``.
        prior_mean: θ_b, shape ``(n_state,)``.
        prior_covariance: B, shape ``(n_state, n_state)``.

    Returns:
        ``(θ*, P*)``, the posterior mean and the materialised, symmetrised
        posterior covariance.

    Raises:
        ImportError: Without the ``assim`` extra.
    """
    vdx = require_vardax("fuse_observations")
    psd = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    oi = vdx.OptimalInterpolation(
        obs_op=vdx.LinearObs(operator=lx.MatrixLinearOperator(design)),
        prior_mean=prior_mean,
        prior_cov_op=lx.MatrixLinearOperator(prior_covariance, psd),
        obs_cov_op=lx.DiagonalLinearOperator(obs_variance),
        form="state",
        solver=lx.Cholesky(),
    )
    batch = vdx.Batch1D(input=shifted_obs[None], mask=mask[None])
    posterior = oi.posterior(batch)
    n_state = prior_mean.shape[0]
    # P* = (B⁻¹ + Gᵀ R⁻¹ G)⁻¹, one lazy mat-vec per basis vector
    covariance = jax.vmap(posterior.cov.mv)(jnp.eye(n_state, dtype=prior_mean.dtype))
    return posterior.mean, gx.symmetrize(covariance)


def fuse_observations(
    forward: CoupledForward,
    observations: Sequence[jax.Array],
    *,
    prior_mean: jax.Array,
    prior_covariance: jax.Array,
    quality: QualityPolicy | None = None,
) -> FusionPosterior:
    """Closed-form joint posterior over ``(Q, bias_inst)`` from fused observations.

    Solves the BLUE update for the affine forward ``y = G θ + c_bg`` with a
    Gaussian prior ``θ ~ N(θ_b, B)`` and diagonal observation covariance ``R``
    (the per-instrument ``R_retr + R_repr + R_align``)::

        P*⁻¹ = B⁻¹ + Gᵀ R⁻¹ G
        θ*   = θ_b + P* Gᵀ R⁻¹ (y − G θ_b − c_bg)

    solved as a ``vardax.OptimalInterpolation`` analysis in its state-space
    (information) form, which factorises only the small
    ``(n_param + n_inst) × (n_param + n_inst)`` posterior precision.

    Each instrument's ``mask`` zeroes its pixels' weight in ``R⁻¹`` and its
    ``variance_inflation`` scales their variance (see
    :mod:`plumax.coupled.quality`); a masked observation may hold anything,
    NaN included.

    Args:
        forward: The assembled multi-instrument coupled forward.
        observations: Observed enhancement vector per instrument, aligned with
            ``forward.instruments`` (each ``(n_obs_i,)``).
        prior_mean: Prior mean state ``[Q_b, bias_b₁ … bias_b_M]``, shape
            ``(n_param + n_inst,)`` with ``n_param = 1`` for a scalar ``Q``
            (biases typically prior-zero).
        prior_covariance: Prior covariance ``B``, shape
            ``(n_param + n_inst, n_param + n_inst)`` — e.g.
            ``diag(σ_Q², σ_bias², …)``.
        quality: Optional quality policy, applied to every instrument that
            carries ``flags`` (:func:`~plumax.coupled.quality.apply_quality`).
            Default ``None``: use the instruments' masks and inflation as
            they are.

    Returns:
        The :class:`FusionPosterior`.

    Raises:
        ValueError: on a per-instrument observation-count mismatch or a
            mis-shaped prior.
        ImportError: Without the ``assim`` extra (vardax).
    """
    if quality is not None:
        screened = tuple(
            inst if inst.flags is None else apply_quality(inst, inst.flags, quality)
            for inst in forward.instruments
        )
        forward = dataclasses.replace(forward, instruments=screened)
    n_inst = len(forward.instruments)
    n_param = _n_emission_params(forward)
    n_state = n_param + n_inst
    theta_b = jnp.asarray(prior_mean, dtype=float)
    b = jnp.asarray(prior_covariance, dtype=float)
    if theta_b.shape != (n_state,):
        raise ValueError(
            f"fuse_observations: `prior_mean` must be (n_param + n_inst = "
            f"{n_state},); "
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

    g, r, m, bg = _stack_design(forward)
    theta, posterior_cov = _information_form_analysis(
        g, y - bg, m, r, prior_mean=theta_b, prior_covariance=b
    )

    std = jnp.sqrt(jnp.clip(jnp.diag(posterior_cov), 0.0, None))
    names = tuple(inst.name for inst in forward.instruments)
    emission = theta[0] if n_param == 1 else theta[:n_param]
    emission_std = std[0] if n_param == 1 else std[:n_param]
    return FusionPosterior(
        emission_rate=emission,
        emission_std=emission_std,
        biases=theta[n_param:],
        bias_std=std[n_param:],
        mean=theta,
        covariance=posterior_cov,
        instrument_names=names,
        pixels_used=jnp.stack(
            [full_weight_pixels(inst) for inst in forward.instruments]
        ),
        prior_mean=theta_b,
        prior_covariance=b,
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
