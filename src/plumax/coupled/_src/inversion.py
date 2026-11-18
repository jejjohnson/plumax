"""Nonlinear end-to-end coupled inversion on vardax, with a Laplace posterior.

The control θ stacks whichever blocks the prior names. The transport backend
says which it can fit: an ``AbstractControlledSource`` declares its own
blocks and its emission blocks (the Tier I ``PlumeSource``: ``location`` and
``wind``, scaled by ``emission_rate`` or ``log_emission_rate``; the column
plume: ``location``, ``direction``, ``log_reference_width`` and
``width_exponent``, scaled by the line density ``line_density`` or
``log_line_density``), and any other backend fits only ``emission_rate`` or
``log_emission_rate``. Every backend also fits the background column
(``background``) and one bias per instrument (``biases``). Conditioned on
everything else, the cost is

    𝒥(θ) = ½ ‖θ − μ_θ‖²_{Σ_θ⁻¹} + ½ Σ_i ‖m_i ⊙ (y_i − h_i(θ))‖²_{Σ_ε,i⁻¹},

with h_i the coupled forward of instrument i: the transport response times Q
(or the line density q = Q / U for the column plume), plus the background and
the bias, then the instrument's radiance operator when it observes radiance.
m_i is the quality mask of ``coupled.quality``.

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
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float

from plumax._src.controlled import AbstractControlledSource
from plumax._src.errors import ControlBlockError, ParameterError, PayloadError
from plumax._src.scalar_emission import GaussianEmission, LognormalEmission
from plumax._vardax import require_vardax
from plumax.coupled._src.forward import (
    CoupledForward,
    _scale_response,
    plume_response,
)
from plumax.coupled.quality import StackedObservations, contribution_report
from plumax.coupled.rtm import RadianceObservationOperator


if TYPE_CHECKING:
    from plumax.coupled.backends import TransportBackend


# the emission blocks of a backend that declares none of its own
_EMISSION_BLOCKS = ("emission_rate", "log_emission_rate")
_LINE_DENSITY_BLOCKS = ("line_density", "log_line_density")
_SHARED_BLOCKS = ("background", "biases")
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

    The prior accepts any block name; :class:`CoupledProblem` checks the names
    and shapes against what its transport backend declares.

    Attributes:
        mean: Prior mean μ_θ of each block. Exactly one emission block (for
            the Tier I plume ``emission_rate`` or ``log_emission_rate``, a
            scalar or one value per emission parameter), the backend's own
            blocks (for the Tier I plume ``location`` ``(x, y)`` [m] and
            ``wind`` ``(speed [m/s], direction [deg from North])``),
            ``background`` (scalar) and ``biases`` (one per instrument).
        covariance: Σ_θ over the flattened control, in :attr:`layout` order:
            a vector of variances (diagonal) or a dense matrix.

    Raises:
        ValueError: On a covariance that does not match the control.
    """

    mean: Mapping[str, Float[Array, "*shape"]]
    covariance: Float[Array, " param"] | Float[Array, "param param"]

    def __post_init__(self) -> None:
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
        forward: The coupled forward. A backend that is an
            ``AbstractControlledSource`` (the Tier I ``PlumeSource``, the
            column plume) recomputes its response at each control that holds
            one of its declared blocks; other backends keep their precomputed
            responses.
        prior: Prior on the control.
        observations: Observations by instrument name: ``(n_obs,)`` columns,
            or ``(n_obs, n_channels)`` radiances.
        radiance_ops: JAX radiance operators by instrument name.
        radiance_variance: Radiance error variance by instrument name, a
            scalar or ``(n_obs, n_channels)``.

    Raises:
        ControlBlockError: If the prior names a block the backend does not
            declare, holds other than one emission block, or gives a block the
            wrong shape.
        ValueError: On missing or extra observations or a radiance operator
            without a variance.
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
        self._check_blocks(n_instruments=len(names))

    def _check_blocks(self, n_instruments: int) -> None:
        """The prior's blocks against the backend's declarations."""
        source = self.forward.source
        declared, emission = _declarations(source)
        allowed = (*declared, *emission, *_SHARED_BLOCKS)
        unknown = sorted(set(self.prior.mean) - set(allowed))
        if unknown:
            hint = (
                ""
                if isinstance(source, AbstractControlledSource)
                else " Only an AbstractControlledSource, such as PlumeSource, "
                "fits a position or a wind."
            )
            raise ControlBlockError(
                f"CoupledProblem: the prior names {unknown}, which "
                f"{type(source).__name__} does not declare. It fits "
                f"{list(declared)}, one of {list(emission)}, and "
                f"{list(_SHARED_BLOCKS)}.{hint}"
            )
        present = [name for name in emission if name in self.prior.mean]
        if len(present) != 1:
            raise ControlBlockError(
                f"CoupledProblem: give exactly one of {list(emission)} for "
                f"{type(source).__name__}; got {present}."
            )
        shapes = {**declared, "background": (), "biases": (n_instruments,)}
        for name, expected in shapes.items():
            if name not in self.prior.mean:
                continue
            shape = np.shape(self.prior.mean[name])
            if shape != tuple(expected):
                raise ControlBlockError(
                    f"CoupledProblem: `{name}` must have shape {tuple(expected)}; "
                    f"got {shape}."
                )

    @property
    def emission_block(self) -> str:
        """The emission block the prior holds, one of the backend's."""
        _, emission = _declarations(self.forward.source)
        return next(name for name in emission if name in self.prior.mean)

    def responses(self, control: Mapping[str, jax.Array]) -> list[jax.Array]:
        """Unit-emission response of each instrument at this control."""
        source = self.forward.source
        if not isinstance(source, AbstractControlledSource):
            return list(self.forward.responses)
        if not any(name in control for name in source.control_blocks):
            return list(self.forward.responses)
        return [
            source.response_at(instrument, control)
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
        block = self.emission_block
        if block.startswith("log_"):
            # A line-search trial far from the data can overflow exp, and
            # lineax raises on the non-finite residual; capping log Q at 50
            # (Q ≈ 5e21 kg/s) keeps the trial cost finite, so the search
            # backtracks instead.
            log_emission = jnp.minimum(control[block], _LOG_EMISSION_CAP)
            # Q = exp(log Q)
            emission = jnp.exp(log_emission)
        else:
            emission = control[block]
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


def _declarations(
    source: TransportBackend,
) -> tuple[Mapping[str, tuple[int, ...]], tuple[str, ...]]:
    """A backend's own blocks and its emission blocks."""
    if isinstance(source, AbstractControlledSource):
        return source.control_blocks, source.emission_blocks
    return {}, _EMISSION_BLOCKS


@dataclasses.dataclass(frozen=True)
class CoupledResult:
    """MAP and Laplace posterior of the coupled inversion.

    ``emission_rate`` and ``emission_std`` are the posterior mean and standard
    deviation of Q (``population.PerEventPosterior``): for a lognormal Q the
    lognormal moments of the Laplace marginal on log Q,
    E[Q] = exp(μ + σ²/2) and sd[Q] = E[Q] √(exp(σ²) − 1).

    A column-plume solve retrieves the line density q = Q / U instead, with
    the same moments in ``line_density`` and ``line_density_std``; its
    ``emission_rate`` and ``emission_std`` are NaN until :meth:`with_wind`
    supplies the wind.

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
        prior: The prior the solve used (``None`` when built by hand).
        line_density: Posterior mean of q [kg m⁻¹]; ``None`` for a backend
            that retrieves Q.
        line_density_std: Posterior standard deviation of q (NaN without the
            posterior).
        wind: ``(U, s_U)`` [m s⁻¹] that :meth:`with_wind` applied, or ``None``.
        emission_block: The emission block the solve fitted, as its backend
            declared it; ``None`` for a result built by hand, whose block is
            then read from the names in ``map_estimate`` and ``prior``. A
            name that starts with ``log_`` is lognormal in the payload.

    For a scalar Q it also carries the Tier V per-event payload
    (``population.PerEventPosterior``): draws of Q from the Laplace marginal
    of its block, N(Q*, σ²) or log Q ~ N(log Q*, σ²) for a lognormal Q, and the
    log-density of that block's prior marginal, both exact for the Gaussian
    laws on θ. For a line density these need the wind: log Q ~ N(μ_q + μ_U,
    σ_q² + σ_U²) for the lognormal wind of :meth:`with_wind`.
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
    prior: CoupledPrior | None = None
    line_density: jax.Array | None = None
    line_density_std: jax.Array | None = None
    wind: tuple[float, float] | None = None
    emission_block: str | None = None

    def with_wind(self, *, speed: float, speed_std: float) -> CoupledResult:
        """The same result with the emission rate filled in, Q = q U.

        For a lognormal wind independent of the fit, with
        σ_U² = log(1 + s_U² / U²) and μ_U = log U − σ_U² / 2, a lognormal line
        density gives log Q ~ N(μ_q + μ_U, σ_q² + σ_U²). The moments are those
        of the product of independent q and U, which equal that law's:

            E[Q] = E[q] U,
            Var[Q] = sd[q]² s_U² + sd[q]² U² + E[q]² s_U².

        Args:
            speed: Mean wind speed U [m s⁻¹].
            speed_std: Its standard deviation s_U [m s⁻¹].

        Returns:
            The result with ``emission_rate``, ``emission_std`` and ``wind``
            set.

        Raises:
            ParameterError: If ``speed`` is not positive or ``speed_std`` is
                negative.
            PayloadError: If the result holds no line density.

        Examples:
            With no spread in the wind, Q is q times the speed:

            >>> import jax.numpy as jnp
            >>> result = CoupledResult(
            ...     map_estimate={"log_line_density": jnp.log(0.1)},
            ...     std={"log_line_density": jnp.asarray(0.15)},
            ...     mean=jnp.log(0.1)[None],
            ...     covariance=None,
            ...     emission_rate=jnp.asarray(jnp.nan),
            ...     emission_std=jnp.asarray(jnp.nan),
            ...     converged=True,
            ...     n_steps=1,
            ...     cost=0.0,
            ...     per_instrument_residuals={},
            ...     pixels_used={},
            ...     line_density=jnp.asarray(0.1),
            ...     line_density_std=jnp.asarray(0.015),
            ... )
            >>> rate = result.with_wind(speed=3.5, speed_std=0.0)
            >>> bool(jnp.allclose(rate.emission_rate, 0.35))
            True
        """
        if speed <= 0.0:
            raise ParameterError(f"with_wind: speed must be > 0, got {speed}")
        if speed_std < 0.0:
            raise ParameterError(f"with_wind: speed_std must be ≥ 0, got {speed_std}")
        if self.line_density is None or self.line_density_std is None:
            raise PayloadError(
                "CoupledResult.with_wind: this result retrieved Q, not the line "
                "density q = Q / U; only a column-plume solve needs a wind."
            )
        # E[Q] = E[q] U
        mean = self.line_density * speed
        # sd[q]² (s_U² + U²)
        spread_of_line_density = jnp.square(self.line_density_std) * (
            speed_std**2 + speed**2
        )
        # E[q]² s_U²
        spread_of_wind = jnp.square(self.line_density) * speed_std**2
        # sd[Q] = √(sd[q]² s_U² + sd[q]² U² + E[q]² s_U²)
        deviation = jnp.sqrt(spread_of_line_density + spread_of_wind)
        return dataclasses.replace(
            self,
            emission_rate=mean,
            emission_std=deviation,
            wind=(float(speed), float(speed_std)),
        )

    def samples(self, key: jax.Array, n: int) -> Float[Array, " member"]:
        """n draws of Q from the Laplace marginal of its block.

        For a line density, Q = q U with U from the lognormal wind of
        :meth:`with_wind`.

        Raises:
            PayloadError: Without the Laplace posterior, for a vector Q, or
                for a line density with no wind applied.
        """
        block = self._payload_block(self.map_estimate)
        if self.std is None:
            raise PayloadError(
                "CoupledResult.samples needs the Laplace posterior; solve with "
                "laplace=True."
            )
        center = self.map_estimate[block]
        spread = self.std[block]
        _require_scalar(center, "samples")
        if block in _LINE_DENSITY_BLOCKS:
            wind = self._wind_law("samples")
            if block == "log_line_density":
                # log Q ~ N(μ_q + μ_U, σ_q² + σ_U²)
                law = LognormalEmission(
                    center + wind.loc, jnp.hypot(spread, wind.scale)
                )
                return law.sample(key, n)
            line_key, wind_key = jax.random.split(key)
            # Q = q U
            line_density = GaussianEmission(center, spread).sample(line_key, n)
            return line_density * wind.sample(wind_key, n)
        if block.startswith("log_"):
            return LognormalEmission(center, spread).sample(key, n)
        return GaussianEmission(center, spread).sample(key, n)

    def _payload_block(self, blocks: Mapping[str, jax.Array]) -> str:
        """The fitted emission block, or the one ``blocks`` names."""
        if self.emission_block is not None:
            return self.emission_block
        return _emission_block(blocks)

    def _wind_law(self, method: str) -> LognormalEmission:
        """The lognormal wind :meth:`with_wind` applied."""
        if self.wind is None:
            raise PayloadError(
                f"CoupledResult.{method}: this result carries the line density "
                "q = Q / U; apply a wind with with_wind(speed=..., speed_std=...)."
            )
        speed, speed_std = self.wind
        return LognormalEmission.from_moments(speed, speed_std)

    def prior_logpdf(self, q: jax.Array) -> jax.Array:
        """log π(Q) of the prior marginal of Q's block.

        For a log line density the prior's (μ_q,0, σ²_q,0) take the place of
        the posterior's: log Q ~ N(μ_q,0 + μ_U, σ²_q,0 + σ_U²).

        Raises:
            PayloadError: Without the prior, for a vector Q, for a line density
                with no wind applied, or for a linear line density, whose
                product with a lognormal wind has no closed-form density.
        """
        if self.prior is None:
            raise PayloadError(
                "CoupledResult.prior_logpdf: this result carries no prior; build "
                "it with solve_coupled."
            )
        block = self._payload_block(self.prior.mean)
        center = jnp.asarray(self.prior.mean[block], dtype=float)
        _require_scalar(center, "prior_logpdf")
        # Q's diagonal entry of Σ_θ, at its offset in the flattened layout
        offset = 0
        for name, length in self.prior.layout:
            if name == block:
                break
            offset += length
        covariance = jnp.asarray(self.prior.covariance, dtype=float)
        if covariance.ndim == 1:
            variance = covariance[offset]
        else:
            variance = covariance[offset, offset]
        if block == "line_density":
            raise PayloadError(
                "CoupledResult.prior_logpdf: a Gaussian line density times a "
                "lognormal wind has no closed-form density; fit "
                "log_line_density instead."
            )
        if block == "log_line_density":
            wind = self._wind_law("prior_logpdf")
            # log Q ~ N(μ_q,0 + μ_U, σ²_q,0 + σ_U²)
            law = LognormalEmission(
                center + wind.loc, jnp.hypot(jnp.sqrt(variance), wind.scale)
            )
            return law.log_prob(q)
        if block.startswith("log_"):
            return LognormalEmission(center, jnp.sqrt(variance)).log_prob(q)
        return GaussianEmission(center, jnp.sqrt(variance)).log_prob(q)


def _emission_block(blocks: Mapping[str, jax.Array]) -> str:
    """The emission block a hand-built result's control or prior holds."""
    for name in ("log_emission_rate", "log_line_density", "line_density"):
        if name in blocks:
            return name
    return "emission_rate"


def _require_scalar(value: jax.Array, method: str) -> None:
    if jnp.ndim(value) != 0:
        raise PayloadError(
            f"CoupledResult.{method}: the Tier V payload is for a scalar Q; this "
            "result has one emission per parameter."
        )


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
    control = vdx.LinearParameterization(basis, background=prior_mean[None])

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

    block = problem.emission_block
    moments = _emission_moments(block, map_estimate, std)
    line_density = None
    line_density_std = None
    if block in _LINE_DENSITY_BLOCKS:
        # Q and U enter only as q = Q / U: Q waits on the wind (with_wind)
        line_density, line_density_std = moments
        emission_rate = jnp.full_like(line_density, jnp.nan)
        emission_std = jnp.full_like(line_density, jnp.nan)
    else:
        emission_rate, emission_std = moments

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
        prior=prior,
        line_density=line_density,
        line_density_std=line_density_std,
        emission_block=block,
    )


def _emission_moments(
    block: str,
    map_estimate: Mapping[str, jax.Array],
    std: Mapping[str, jax.Array] | None,
) -> tuple[jax.Array, jax.Array]:
    """Posterior mean and standard deviation of the scale from its Laplace marginal."""
    if not block.startswith("log_"):
        emission = map_estimate[block]
        if std is None:
            return emission, jnp.full_like(emission, jnp.nan)
        return emission, std[block]
    log_mean = map_estimate[block]
    if std is None:
        # no posterior: report the MAP, Q* = exp(log Q*)
        return jnp.exp(log_mean), jnp.full_like(log_mean, jnp.nan)
    log_variance = std[block] ** 2
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
