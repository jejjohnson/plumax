"""Nonlinear end-to-end coupled inversion on vardax, with a Laplace posterior.

The control θ stacks whichever of these blocks the prior names: the emission
rate (``emission_rate``, or ``log_emission_rate`` for a lognormal Q), the
source position (``location``), the mean wind (``wind``: speed and
meteorological direction), the background column (``background``) and one bias
per instrument (``biases``). Conditioned on everything else, the cost is

    𝒥(θ) = ½ ‖θ − μ_θ‖²_{Σ_θ⁻¹} + ½ Σ_i ‖m_i ⊙ (y_i − h_i(θ))‖²_{Σ_ε,i⁻¹},

with h_i the coupled forward of instrument i: the transport response times Q,
plus the background and the bias, then the instrument's radiance operator when
it observes radiance. m_i is the quality mask of ``coupled.quality``.

plumax builds the forward, the prior and the masks; vardax does the rest.
:func:`solve_coupled` hands ``vardax.ThreeDVar`` the stacked forward as a
``vardax.CallableObs``, runs L-BFGS in the whitened control θ = μ_θ + L χ
(Σ_θ = L Lᵀ, through ``vardax.LinearParameterization``), and takes the
Gauss–Newton Laplace covariance from ``vardax.LaplaceCovariance``,

    Σ_θ|y = (Σ_θ⁻¹ + J_hᵀ Σ_ε⁻¹ J_h)⁻¹,   J_h = ∂h/∂θ at θ*.

Σ_θ is dense or diagonal and Σ_ε diagonal; block-structured operators wait on
jejjohnson/vardax#141, and the Gauss–Newton minimiser on jejjohnson/vardax#72.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Literal, Protocol, runtime_checkable

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float

from plumax._vardax import require_vardax
from plumax.coupled.forward import CoupledForward, PlumeSource, _scale_response
from plumax.coupled.instrument import Instrument
from plumax.coupled.quality import StackedObservations, contribution_report
from plumax.coupled.rtm import RadianceObservationOperator
from plumax.gauss_plume.dispersion import get_dispersion_params
from plumax.gauss_plume.observation import ColumnObservation
from plumax.gauss_plume.plume import plume_concentration_vmap


_EMISSION_BLOCKS = ("emission_rate", "log_emission_rate")
_BLOCK_SIZES = {"location": 2, "wind": 2, "background": 0}
_MET_BLOCKS = ("location", "wind")
_LOG_EMISSION_CAP = 50.0


@runtime_checkable
class RadianceOperator(Protocol):
    """A JAX radiance operator: column enhancement → per-receptor channels.

    ``LinearisedRadianceOperator`` and ``LayeredRadianceOperator`` qualify.
    """

    def predict(self, column_enhancement: jax.Array) -> jax.Array:
        """Radiance per receptor and channel, shape ``(n_obs, n_channels)``."""
        ...


@dataclasses.dataclass(frozen=True)
class CoupledPrior:
    """Gaussian prior N(μ_θ, Σ_θ) on the stacked control, by named block.

    Attributes:
        mean: Prior mean μ_θ of each block. Exactly one of ``emission_rate``
            and ``log_emission_rate`` (a scalar, or one value per emission
            parameter); optionally ``location`` ``(x, y)`` [m], ``wind``
            ``(speed [m/s], direction [deg from North])``, ``background``
            (scalar) and ``biases`` (one per instrument).
        covariance: Σ_θ over the flattened control, in :attr:`layout` order:
            a vector of variances (diagonal) or a dense matrix.

    Raises:
        ValueError: On an unknown or missing block, a mis-shaped block or a
            covariance that does not match the control.
    """

    mean: Mapping[str, Float[Array, "*shape"]]
    covariance: Float[Array, " param"] | Float[Array, "param param"]

    def __post_init__(self) -> None:
        known = (*_EMISSION_BLOCKS, *_BLOCK_SIZES, "biases")
        unknown = set(self.mean) - set(known)
        if unknown:
            raise ValueError(
                f"CoupledPrior: unknown blocks {sorted(unknown)}; expected some of "
                f"{list(known)}."
            )
        emission = [name for name in _EMISSION_BLOCKS if name in self.mean]
        if len(emission) != 1:
            raise ValueError(
                "CoupledPrior: give exactly one of `emission_rate` and "
                f"`log_emission_rate`; got {emission}."
            )
        for name, size in _BLOCK_SIZES.items():
            if name in self.mean:
                expected = () if size == 0 else (size,)
                if np.shape(self.mean[name]) != expected:
                    raise ValueError(
                        f"CoupledPrior: `{name}` must have shape {expected}; got "
                        f"{np.shape(self.mean[name])}."
                    )
        size = self.size
        covariance_shape = np.shape(self.covariance)
        if covariance_shape not in ((size,), (size, size)):
            raise ValueError(
                f"CoupledPrior: `covariance` must be ({size},) or ({size}, {size}) "
                f"for this control; got {covariance_shape}."
            )

    @classmethod
    def from_std(
        cls,
        mean: Mapping[str, Float[Array, "*shape"] | float],
        std: Mapping[str, Float[Array, "*shape"] | float],
    ) -> CoupledPrior:
        """A diagonal prior from per-block means and standard deviations.

        Σ_θ = diag(σ²), each block's σ broadcast to the block's shape.

        Args:
            mean: Prior mean μ_θ of each block.
            std: Prior standard deviation σ of each block, same keys.

        Returns:
            The prior.

        Raises:
            ValueError: If the keys differ or a deviation is not positive.
        """
        if set(mean) != set(std):
            raise ValueError("CoupledPrior.from_std: `mean` and `std` keys differ.")
        means = {name: jnp.asarray(value, dtype=float) for name, value in mean.items()}
        stds = {
            name: jnp.broadcast_to(jnp.asarray(std[name], dtype=float), value.shape)
            for name, value in means.items()
        }
        flat_std, _ = ravel_pytree(stds)
        if not bool(jnp.all(flat_std > 0.0)):
            raise ValueError("CoupledPrior.from_std: standard deviations must be > 0.")
        return cls(mean=means, covariance=flat_std**2)

    @property
    def size(self) -> int:
        """Length D_θ of the flattened control."""
        flat, _ = self.flatten()
        return int(flat.shape[0])

    @property
    def layout(self) -> tuple[tuple[str, int], ...]:
        """``(block, length)`` pairs in the flattened order (sorted by name)."""
        return tuple(
            (name, int(np.size(self.mean[name]))) for name in sorted(self.mean)
        )

    @property
    def log_emission(self) -> bool:
        """Whether the emission is lognormal (controlled through log Q)."""
        return "log_emission_rate" in self.mean

    def flatten(
        self,
    ) -> tuple[Float[Array, " param"], Callable[[jax.Array], dict[str, jax.Array]]]:
        """μ_θ as one vector, and the map from such a vector back to blocks."""
        means = {
            name: jnp.asarray(value, dtype=float) for name, value in self.mean.items()
        }
        return ravel_pytree(means)

    def cholesky(self) -> Float[Array, "param param"]:
        """L with Σ_θ = L Lᵀ (√σ² on the diagonal for a diagonal prior)."""
        covariance = jnp.asarray(self.covariance, dtype=float)
        if covariance.ndim == 1:
            # L = diag(σ)
            return jnp.diag(jnp.sqrt(covariance))
        psd = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
        return gx.safe_cholesky(lx.MatrixLinearOperator(covariance, psd))

    def row_operator(self) -> lx.AbstractLinearOperator:
        """Σ_θ on a ``(1, D_θ)`` control row, the shape vardax applies it to."""
        covariance = jnp.asarray(self.covariance, dtype=float)
        if covariance.ndim == 1:
            return lx.DiagonalLinearOperator(einx.rearrange("p -> 1 p", covariance))
        psd = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
        return lx.FunctionLinearOperator(
            lambda row: einx.dot("row p, p q -> row q", row, covariance),
            jax.ShapeDtypeStruct((1, covariance.shape[0]), covariance.dtype),
            psd,
        )


@dataclasses.dataclass(frozen=True)
class CoupledProblem:
    """The coupled forward, its prior and every instrument's observations.

    Column instruments are observed through their averaging kernel (the
    ``CoupledForward`` responses); an instrument named in ``radiance_ops`` is
    observed through that operator instead, applied to its column (bias
    included). Masks and variance inflation come from each instrument
    (``coupled.quality.apply_quality``).

    Attributes:
        forward: The coupled forward. ``location`` and ``wind`` blocks need
            the Tier I ``PlumeSource``, whose response is recomputed at each
            control; other backends keep their precomputed responses.
        prior: Prior on the control.
        observations: Observations by instrument name: ``(n_obs,)`` columns,
            or ``(n_obs, n_channels)`` radiances.
        radiance_ops: JAX radiance operators by instrument name.
        radiance_variance: Radiance error variance by instrument name, a
            scalar or ``(n_obs, n_channels)``.

    Raises:
        ValueError: On missing or extra observations, a met block without a
            ``PlumeSource``, mis-shaped biases or a radiance operator without
            a variance.
        TypeError: For a ``RadianceObservationOperator``, which runs in NumPy
            and so cannot be differentiated.
    """

    forward: CoupledForward
    prior: CoupledPrior
    observations: Mapping[str, Float[Array, "*shape"]]
    radiance_ops: Mapping[str, RadianceOperator] = dataclasses.field(
        default_factory=dict
    )
    radiance_variance: Mapping[str, Float[Array, "*shape"] | float] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        names = [instrument.name for instrument in self.forward.instruments]
        if len(set(names)) != len(names):
            raise ValueError(f"CoupledProblem: instrument names repeat: {names}.")
        if set(self.observations) != set(names):
            raise ValueError(
                f"CoupledProblem: observations for {sorted(self.observations)} but "
                f"the instruments are {sorted(names)}."
            )
        unknown = set(self.radiance_ops) - set(names)
        if unknown:
            raise ValueError(
                f"CoupledProblem: radiance operators for unknown instruments "
                f"{sorted(unknown)}."
            )
        for name, operator in self.radiance_ops.items():
            if isinstance(operator, RadianceObservationOperator):
                raise TypeError(
                    f"CoupledProblem: the radiance operator of {name!r} runs in "
                    "NumPy and cannot be differentiated; pass its "
                    "LayeredRadianceOperator, or `linearise(operator)`."
                )
            if name not in self.radiance_variance:
                raise ValueError(
                    f"CoupledProblem: no `radiance_variance` for {name!r}."
                )
        met = [name for name in _MET_BLOCKS if name in self.prior.mean]
        if met and not isinstance(self.forward.source, PlumeSource):
            raise ValueError(
                f"CoupledProblem: the {met} blocks re-run the Tier I plume and "
                "need a PlumeSource; other backends invert the emission, the "
                "background and the biases."
            )
        if "biases" in self.prior.mean:
            expected = (len(names),)
            if np.shape(self.prior.mean["biases"]) != expected:
                raise ValueError(
                    f"CoupledProblem: `biases` must have shape {expected}; got "
                    f"{np.shape(self.prior.mean['biases'])}."
                )

    def responses(self, control: Mapping[str, jax.Array]) -> list[jax.Array]:
        """Unit-emission response of each instrument at this control."""
        if not any(name in control for name in _MET_BLOCKS):
            return list(self.forward.responses)
        source = self.forward.source
        assert isinstance(source, PlumeSource)
        location = control.get("location", jnp.asarray(source.location[:2]))
        wind = control.get(
            "wind", jnp.asarray([source.wind_speed, source.wind_direction])
        )
        return [
            plume_response(source, instrument, location, wind)
            for instrument in self.forward.instruments
        ]

    def predict(self, control: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        """h_i(θ) for every instrument, by name.

        Args:
            control: The control, by block (as :meth:`CoupledPrior.flatten`
                unflattens it).

        Returns:
            Columns ``(n_obs,)``, or radiances ``(n_obs, n_channels)``.
        """
        if "log_emission_rate" in control:
            # A line-search trial far from the data can overflow exp, and
            # lineax raises on the non-finite residual; capping log Q at 50
            # (Q ≈ 5e21 kg/s) keeps the trial cost finite, so the search
            # backtracks instead.
            log_emission = jnp.minimum(control["log_emission_rate"], _LOG_EMISSION_CAP)
            # Q = exp(log Q)
            emission = jnp.exp(log_emission)
        else:
            emission = control["emission_rate"]
        background = control.get(
            "background", jnp.asarray(self.forward.source.background, dtype=float)
        )
        n_instruments = len(self.forward.instruments)
        biases = control.get("biases", jnp.zeros(n_instruments))
        predictions = {}
        for index, (instrument, response) in enumerate(
            zip(self.forward.instruments, self.responses(control), strict=True)
        ):
            # c_i = Q r_i + c_bg + bias_i
            column = _scale_response(response, emission) + background + biases[index]
            operator = self.radiance_ops.get(instrument.name)
            predictions[instrument.name] = (
                column if operator is None else operator.predict(column)
            )
        return predictions

    def predict_stacked(self, control: Mapping[str, jax.Array]) -> jax.Array:
        """Every instrument's prediction, flattened and stacked, shape ``(D_y,)``."""
        predictions = self.predict(control)
        return jnp.concatenate(
            [
                jnp.ravel(predictions[instrument.name])
                for instrument in self.forward.instruments
            ]
        )

    def stacked_observations(self) -> StackedObservations:
        """Observations, masks and Σ_ε stacked like :meth:`predict_stacked`.

        A radiance pixel's mask and variance inflation apply to all its
        channels. Masked values are zero (jejjohnson/vardax#147).
        """
        values, masks, variances = [], [], []
        for instrument in self.forward.instruments:
            observation = jnp.asarray(self.observations[instrument.name], dtype=float)
            mask = instrument.observation_mask()
            if instrument.name in self.radiance_ops:
                n_channels = observation.shape[-1]
                variance = jnp.broadcast_to(
                    jnp.asarray(self.radiance_variance[instrument.name], dtype=float),
                    observation.shape,
                )
                if instrument.variance_inflation is not None:
                    inflation = jnp.asarray(instrument.variance_inflation, dtype=float)
                    variance = einx.multiply(
                        "obs channel, obs -> obs channel", variance, inflation
                    )
                mask = einx.rearrange("obs -> obs channel", mask, channel=n_channels)
            else:
                variance = instrument.observation_variance()
            if observation.shape != mask.shape:
                raise ValueError(
                    f"CoupledProblem: observation for {instrument.name!r} has shape "
                    f"{observation.shape}; expected {mask.shape}."
                )
            # y = 0 where m = 0
            values.append(jnp.ravel(jnp.where(mask > 0.0, observation, 0.0)))
            masks.append(jnp.ravel(mask))
            variances.append(jnp.ravel(variance))
        return StackedObservations(
            values=jnp.concatenate(values),
            mask=jnp.concatenate(masks),
            variance=jnp.concatenate(variances),
        )


def plume_response(
    source: PlumeSource,
    instrument: Instrument,
    location: Float[Array, " 2"],
    wind: Float[Array, " 2"],
) -> Float[Array, " obs"]:
    """Tier I unit-emission column at a traced source position and wind.

    ``coupled.column_response`` with the horizontal position and the wind
    (speed, meteorological direction) as JAX arrays, so the coupled cost can
    differentiate through them. The source height, stability class and levels
    stay those of ``source``.

    Args:
        source: The Tier I source (height, stability, levels).
        instrument: The observing instrument.
        location: Source ``(x, y)`` [m].
        wind: ``(speed [m/s], direction [deg from North])``.

    Returns:
        The unit-emission column through the averaging kernel.
    """
    levels = jnp.asarray(source.column_z, dtype=float)
    receptors = jnp.asarray(instrument.receptors, dtype=float)
    n_levels = levels.shape[0]
    # every receptor at every level, level fastest
    x = einx.rearrange("obs -> (obs level)", receptors[:, 0], level=n_levels)
    y = einx.rearrange("obs -> (obs level)", receptors[:, 1], level=n_levels)
    z = einx.rearrange("level -> (obs level)", levels, obs=receptors.shape[0])
    direction = jnp.deg2rad(wind[1])
    # (u, v) = −s (sin φ, cos φ)
    wind_u = -wind[0] * jnp.sin(direction)
    wind_v = -wind[0] * jnp.cos(direction)
    concentration = plume_concentration_vmap(
        x,
        y,
        z,
        location[0],
        location[1],
        source.location[2],
        wind_u,
        wind_v,
        1.0,
        get_dispersion_params(source.stability_class),
    )
    profiles = einx.rearrange("(obs level) -> obs level", concentration, level=n_levels)
    observation = ColumnObservation.trapezoid(levels, averaging_kernel=instrument.ak)
    return observation.column(profiles)


@dataclasses.dataclass(frozen=True)
class CoupledResult:
    """MAP and Laplace posterior of the coupled inversion.

    ``emission_rate`` and ``emission_std`` are the posterior mean and standard
    deviation of Q (``population.PerEventPosterior``): for a lognormal Q the
    lognormal moments of the Laplace marginal on log Q,
    E[Q] = exp(μ + σ²/2) and sd[Q] = E[Q] √(exp(σ²) − 1).

    Attributes:
        map_estimate: MAP θ* by block (``log_emission_rate`` stays in logs).
        std: Laplace posterior standard deviation by block (``None`` without
            the posterior).
        mean: MAP θ* flattened, in ``CoupledPrior.layout`` order.
        covariance: Laplace covariance Σ_θ|y, ``None`` without the posterior.
        emission_rate: Posterior mean of Q.
        emission_std: Posterior standard deviation of Q (NaN without the
            posterior).
        converged: optimistix's success flag for the L-BFGS solve.
        n_steps: Iterations taken.
        cost: 𝒥 at the MAP, in the whitened control.
        per_instrument_residuals: y_i − h_i(θ*) by instrument, NaN where
            masked.
        pixels_used: Pixels each instrument contributed at full weight.
    """

    map_estimate: dict[str, jax.Array]
    std: dict[str, jax.Array] | None
    mean: jax.Array
    covariance: jax.Array | None
    emission_rate: jax.Array
    emission_std: jax.Array
    converged: bool
    n_steps: int
    cost: float
    per_instrument_residuals: dict[str, jax.Array]
    pixels_used: dict[str, int]


def solve_coupled(
    problem: CoupledProblem,
    *,
    method: Literal["lbfgs", "gauss_newton"] = "lbfgs",
    laplace: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 200,
) -> CoupledResult:
    """MAP and Laplace posterior of the coupled problem, on vardax.

    θ* = argmin 𝒥(θ),   Σ_θ|y = (Σ_θ⁻¹ + J_hᵀ Σ_ε⁻¹ J_h)⁻¹ at θ*.

    Algorithm:
        1. Flatten the prior to μ_θ, factor Σ_θ = L Lᵀ.
        2. Stack observations, masks and Σ_ε over the instruments.
        3. ``vardax.ThreeDVar`` with the stacked forward as a ``CallableObs``
           and the whitened control θ = μ_θ + L χ; minimise with L-BFGS.
        4. ``vardax.LaplaceCovariance`` at θ*, materialised.

    Args:
        problem: The coupled problem.
        method: ``"lbfgs"``. ``"gauss_newton"`` waits on vardax's
            Gauss–Newton minimiser (jejjohnson/vardax#72).
        laplace: Also compute the Laplace posterior.
        rtol: L-BFGS relative tolerance.
        atol: L-BFGS absolute tolerance.
        max_steps: Iteration cap.

    Returns:
        The :class:`CoupledResult`.

    Raises:
        NotImplementedError: For ``method="gauss_newton"``.
        ImportError: Without the ``assim`` extra.
    """
    if method == "gauss_newton":
        raise NotImplementedError(
            "solve_coupled: the Gauss–Newton minimiser waits on vardax "
            "(jejjohnson/vardax#72); use method='lbfgs'."
        )
    if method != "lbfgs":
        raise ValueError(f"solve_coupled: unknown method {method!r}.")
    import optimistix as optx

    vdx = require_vardax("solve_coupled")

    # --- prior and whitened control ---
    prior = problem.prior
    prior_mean, unflatten = prior.flatten()
    n_param = prior_mean.shape[0]
    # Σ_θ = L Lᵀ
    cholesky = prior.cholesky()
    # θ = μ_θ + L χ, ½ ‖χ‖²
    basis = vdx.linear_basis(cholesky, variance=1.0)
    control = vdx.LinearParameterization(basis, x_b=prior_mean[None])

    # --- observations ---
    stacked = problem.stacked_observations()

    def stacked_prediction(state: jax.Array) -> jax.Array:
        # h(θ) on a (1, D_θ) row
        return einx.rearrange(
            "obs -> 1 obs", problem.predict_stacked(unflatten(state[0]))
        )

    model = vdx.ThreeDVar(
        obs_op=vdx.CallableObs(fn=stacked_prediction, materialise=True),
        prior_mean=prior_mean[None],
        prior_cov_op=prior.row_operator(),
        obs_cov_op=stacked.obs_cov_op(),
        minimiser=vdx.OptimistixMinimiser(
            optx.LBFGS(rtol=rtol, atol=atol), max_steps=max_steps
        ),
        parameterization=control,
        cov_solver=lx.AutoLinearSolver(well_posed=True),
    )
    batch = stacked.batch()

    # --- MAP ---
    # θ* = argmin 𝒥(θ)
    analysis, outcome = model.analyse(batch)
    theta = analysis[0, 0]
    map_estimate = unflatten(theta)

    # --- Laplace posterior ---
    covariance = None
    std = None
    if laplace:
        # Σ_θ|y = (Σ_θ⁻¹ + J_hᵀ Σ_ε⁻¹ J_h)⁻¹
        posterior = vdx.LaplaceCovariance(solver=lx.AutoLinearSolver(well_posed=True))(
            theta[None], model, batch
        )
        columns = jax.vmap(
            lambda basis_vector: posterior.cov.mv(basis_vector[None])[0]
        )(jnp.eye(n_param, dtype=theta.dtype))
        covariance = gx.symmetrize(columns)
        marginal_std = jnp.sqrt(jnp.clip(jnp.diagonal(covariance), 0.0, None))
        std = unflatten(marginal_std)

    emission_rate, emission_std = _emission_moments(prior, map_estimate, std)

    # --- diagnostics ---
    predictions = problem.predict(map_estimate)
    residuals = {}
    for instrument in problem.forward.instruments:
        observation = jnp.asarray(problem.observations[instrument.name], dtype=float)
        residual = observation - predictions[instrument.name]
        mask = instrument.observation_mask()
        if residual.ndim == 2:
            mask = einx.rearrange("obs -> obs channel", mask, channel=1)
        residuals[instrument.name] = jnp.where(mask > 0.0, residual, jnp.nan)
    return CoupledResult(
        map_estimate=map_estimate,
        std=std,
        mean=theta,
        covariance=covariance,
        emission_rate=emission_rate,
        emission_std=emission_std,
        converged=bool(outcome.converged[0]),
        n_steps=int(outcome.num_steps[0]),
        cost=float(outcome.cost[0]),
        per_instrument_residuals=residuals,
        pixels_used=contribution_report(problem.forward.instruments),
    )


def _emission_moments(
    prior: CoupledPrior,
    map_estimate: Mapping[str, jax.Array],
    std: Mapping[str, jax.Array] | None,
) -> tuple[jax.Array, jax.Array]:
    """Posterior mean and standard deviation of Q from the Laplace marginal."""
    if not prior.log_emission:
        emission = map_estimate["emission_rate"]
        if std is None:
            return emission, jnp.full_like(emission, jnp.nan)
        return emission, std["emission_rate"]
    log_mean = map_estimate["log_emission_rate"]
    if std is None:
        # no posterior: report the MAP, Q* = exp(log Q*)
        return jnp.exp(log_mean), jnp.full_like(log_mean, jnp.nan)
    log_variance = std["log_emission_rate"] ** 2
    # E[Q] = exp(μ + σ²/2)
    mean = jnp.exp(log_mean + 0.5 * log_variance)
    # sd[Q] = E[Q] √(exp(σ²) − 1)
    deviation = mean * jnp.sqrt(jnp.expm1(log_variance))
    return mean, deviation


__all__ = [
    "CoupledPrior",
    "CoupledProblem",
    "CoupledResult",
    "RadianceOperator",
    "plume_response",
    "solve_coupled",
]
