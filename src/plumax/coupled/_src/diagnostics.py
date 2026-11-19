"""What the data determined in a coupled fit, and how well the model fits.

At the MAP θ* of a coupled problem, with J_h the Jacobian of the stacked
prediction and Σ_ε the diagonal observation-error covariance (masks applied),
the Fisher information of the control is

    F = J_hᵀ Σ_ε⁻¹ J_h.

:func:`fit_diagnostics` reports the prior-free Cramér–Rao bound √[F⁻¹]_aa of
each parameter, how much of each the data determined against the prior, the
degrees of freedom for signal, how far the prior was pulled, and each
instrument's residual statistics (Rodgers 2000, ch. 2).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

from plumax._src.vardax import PSD_TAGS
from plumax.coupled._src.inversion import CoupledProblem, CoupledResult


#: An eigenvalue of F below this fraction of the largest carries no information.
_NULL_EIGENVALUE = 1e-10
#: A parameter with more than this weight on the null space of F has no bound.
_NULL_WEIGHT = 1e-8


@dataclasses.dataclass(frozen=True)
class FitDiagnostics:
    """Information content and residual statistics of a coupled fit.

    Attributes:
        fisher_information: F = J_hᵀ Σ_ε⁻¹ J_h at θ*, in the prior's layout.
        cramer_rao_std: Prior-free bound √[F⁻¹]_aa by block; ``inf`` where F
            is singular along the parameter.
        data_fraction: 1 − [Σ_θ|y]_aa / [Σ_θ]_aa by block: 0 when the prior
            set the parameter, near 1 when the data did.
        dofs: Degrees of freedom for signal d_s = tr A, with A the averaging
            kernel (``gaussx.information_diagnostics``).
        prior_pull: (θ*_a − μ_θ,a) / √[Σ_θ]_aa by block.
        reduced_chi2: χ²_red = Σₙ ωₙ rₙ² / σₙ² / (N_i − d_s) by instrument.
        rms_residual: √(Σₙ ωₙ rₙ² / N_i) by instrument.
    """

    fisher_information: Float[np.ndarray, "param param"]
    cramer_rao_std: dict[str, np.ndarray]
    data_fraction: dict[str, np.ndarray]
    dofs: float
    prior_pull: dict[str, np.ndarray]
    reduced_chi2: dict[str, float]
    rms_residual: dict[str, float]


def fit_diagnostics(problem: CoupledProblem, result: CoupledResult) -> FitDiagnostics:
    """Diagnostics of a solved coupled problem.

    F = J_hᵀ Σ_ε⁻¹ J_h;  V[a] ≥ [F⁻¹]_aa;  Σ_θ|y = (Σ_θ⁻¹ + F)⁻¹.

    Algorithm:
        1. Evaluate the Jacobian J_h of the stacked prediction at θ*.
        2. Form F = J_hᵀ Σ_ε⁻¹ J_h with the masked, inflated variances.
        3. Eigendecompose F; bound each parameter, or report inf.
        4. Compute the posterior and d_s with ``gaussx.information_diagnostics``
           and compare the posterior with the prior, block by block.
        5. Compute the residual statistics of each instrument.

    The posterior in step 4 is the Gauss–Newton one at θ*, the covariance
    ``solve_coupled`` reports, recomputed here so a solve with
    ``laplace=False`` can be diagnosed too. Each instrument's χ²_red uses the
    d_s of the whole fit.

    Args:
        problem: The coupled problem.
        result: Its solution, from ``solve_coupled``.

    Returns:
        The :class:`FitDiagnostics`.

    Examples:
        A parameter the data never see has no prior-free bound and keeps its
        prior: here the bias of a second, fully masked instrument.

        >>> import jax.numpy as jnp
        >>> import plumax.coupled as cpl
        >>> from plumax.gauss_plume import ColumnPlume, ColumnPlumeSource, PlumeGeometry
        >>> receptors = jnp.stack(
        ...     [jnp.linspace(100.0, 1000.0, 20), jnp.zeros(20)], axis=-1
        ... )
        >>> seen = cpl.Instrument("EMIT", receptors, retrieval_variance=1e-8)
        >>> masked = cpl.Instrument(
        ...     "GHGSat", receptors, retrieval_variance=1e-8, mask=jnp.zeros(20)
        ... )
        >>> geometry = PlumeGeometry(
        ...     location=jnp.zeros(2),
        ...     direction=0.0,
        ...     reference_width=150.0,
        ...     width_exponent=0.9,
        ... )
        >>> source = ColumnPlumeSource(plume=ColumnPlume(), geometry=geometry)
        >>> forward = cpl.build_coupled_forward(source, [seen, masked])
        >>> observation = 0.1 * source.column_response(seen)
        >>> prior = cpl.CoupledPrior.from_std(
        ...     {"line_density": 0.0, "biases": jnp.zeros(2)},
        ...     {"line_density": 1.0, "biases": 1e-3},
        ... )
        >>> problem = cpl.CoupledProblem(
        ...     forward=forward,
        ...     prior=prior,
        ...     observations={"EMIT": observation, "GHGSat": observation},
        ... )
        >>> report = fit_diagnostics(problem, cpl.solve_coupled(problem))
        >>> report.cramer_rao_std["biases"].tolist()[1]
        inf
        >>> float(report.data_fraction["biases"][1])
        0.0
    """
    prior = problem.prior
    prior_mean, unflatten = prior.flatten()
    params = jnp.asarray(result.mean, dtype=prior_mean.dtype)

    # --- Jacobian at the MAP ---
    # J_h = ∂h / ∂θ at θ*
    obs_jacobian = _stacked_jacobian(problem, unflatten)(params)

    # --- Fisher information ---
    stacked = problem.stacked_observations()
    # Σ_ε⁻¹ with masked pixels at zero weight: ωₙ / σₙ²
    obs_precision = stacked.mask / stacked.variance
    # Σ_ε⁻¹ J_h
    weighted_jacobian = einx.multiply(
        "obs param, obs -> obs param", obs_jacobian, obs_precision
    )
    # F = J_hᵀ Σ_ε⁻¹ J_h
    fisher = einx.dot("obs row, obs col -> row col", obs_jacobian, weighted_jacobian)
    transposed = einx.id("row col -> col row", fisher)
    fisher = 0.5 * (fisher + transposed)

    # --- prior-free bound ---
    # F = V Λ Vᵀ
    eigenvalues, eigenvectors = jnp.linalg.eigh(fisher)
    largest = jnp.maximum(jnp.max(eigenvalues), 0.0)
    informative = eigenvalues > _NULL_EIGENVALUE * largest
    safe_eigenvalues = jnp.where(informative, eigenvalues, 1.0)
    # 1 / λ_k on the informative eigenvalues, 0 elsewhere
    inverse_eigenvalues = jnp.where(informative, 1.0 / safe_eigenvalues, 0.0)
    # [F⁻¹]_aa = Σ_k V_ak² / λ_k
    squared_loadings = jnp.square(eigenvectors)
    bound_variance = einx.dot(
        "param k, k -> param", squared_loadings, inverse_eigenvalues
    )
    # Σ_k V_ak² over λ_k ≈ 0: the share of parameter a the data cannot see
    null_weight = einx.dot(
        "param k, k -> param", squared_loadings, (~informative).astype(fisher.dtype)
    )
    bound_std = jnp.where(null_weight > _NULL_WEIGHT, jnp.inf, jnp.sqrt(bound_variance))

    # --- posterior against prior ---
    # Σ_ε^{-1/2} J_h, so the noise is white; a masked pixel is a zero row
    whitened_jacobian = einx.multiply(
        "obs param, obs -> obs param", obs_jacobian, jnp.sqrt(obs_precision)
    )
    # Σ_θ = L Lᵀ
    cholesky = prior.cholesky()
    prior_covariance = einx.dot("row a, col a -> row col", cholesky, cholesky)
    # Σ_θ|y, d_s = tr A, by a QR of [Σ_ε^{-1/2} J_h; L⁻¹] (Rodgers 2000)
    information = gx.information_diagnostics(
        whitened_jacobian,
        lx.DiagonalLinearOperator(jnp.ones(whitened_jacobian.shape[0])),
        lx.MatrixLinearOperator(prior_covariance, PSD_TAGS),
    )
    dofs = information.dofs
    posterior_covariance = information.posterior_cov
    prior_variance = jnp.diagonal(prior_covariance)
    # 1 − [Σ_θ|y]_aa / [Σ_θ]_aa
    data_fraction = 1.0 - jnp.diagonal(posterior_covariance) / prior_variance
    # (θ* − μ_θ) / √[Σ_θ]_aa
    prior_pull = (params - prior_mean) / jnp.sqrt(prior_variance)

    # --- residuals ---
    predictions = problem.predict(unflatten(params))
    reduced_chi2, rms_residual = {}, {}
    start = 0
    for instrument in problem.forward.instruments:
        size = int(np.size(predictions[instrument.name]))
        stop = start + size
        # rₙ = yₙ − hₙ(θ*)
        predicted = einx.id("... -> (...)", predictions[instrument.name])
        residual = stacked.values[start:stop] - predicted
        # ωₙ, σₙ²
        weight = stacked.mask[start:stop]
        variance = stacked.variance[start:stop]
        start = stop
        # N_i = Σₙ ωₙ
        num_kept = einx.sum("[obs]", weight)
        # Σₙ ωₙ rₙ² / σₙ²
        chi2 = einx.sum("[obs]", weight * jnp.square(residual) / variance)
        # χ²_red = Σₙ ωₙ rₙ² / σₙ² / (N_i − d_s)
        reduced_chi2[instrument.name] = _ratio(chi2, num_kept - dofs)
        # rms = √(Σₙ ωₙ rₙ² / N_i)
        squared = einx.sum("[obs]", weight * jnp.square(residual))
        rms_residual[instrument.name] = float(np.sqrt(_ratio(squared, num_kept)))

    return FitDiagnostics(
        fisher_information=np.asarray(fisher),
        cramer_rao_std=_blocks(unflatten, bound_std),
        data_fraction=_blocks(unflatten, data_fraction),
        dofs=float(dofs),
        prior_pull=_blocks(unflatten, prior_pull),
        reduced_chi2=reduced_chi2,
        rms_residual=rms_residual,
    )


def _stacked_jacobian(
    problem: CoupledProblem,
    unflatten: Callable[[jax.Array], dict[str, jax.Array]],
) -> Callable[[jax.Array], Float[Array, "obs param"]]:
    """θ ↦ J_h(θ), the Jacobian of the stacked prediction, by forward mode."""

    def predict_flat(state: Float[Array, " param"]) -> Float[Array, " obs"]:
        # h(θ), every instrument stacked
        return problem.predict_stacked(unflatten(state))

    # D_θ forward-mode products
    return jax.jacfwd(predict_flat)


def _blocks(
    unflatten: Callable[[jax.Array], dict[str, jax.Array]], flat: jax.Array
) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in unflatten(flat).items()}


def _ratio(numerator: jax.Array, denominator: jax.Array) -> float:
    bottom = float(denominator)
    return float(numerator) / bottom if bottom > 0.0 else float("nan")


__all__ = ["FitDiagnostics", "fit_diagnostics"]
