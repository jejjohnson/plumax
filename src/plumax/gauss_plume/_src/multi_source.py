"""Masked-K sources: K_max slots with an on/off indicator each, and their posterior.

A basin overpass can hold several plumes. The masked-K formulation keeps a
fixed number K_max of source slots, each with an emission rate Q_k, a location
(x_k, y_k) and an activity indicator a_k ∈ {0, 1}, and sums the active ones:

    c(𝐬) = Σ_k a_k Q_k c₁(𝐬; x_k, y_k),

with c₁ the unit-rate plume of one source. The active count K = Σ_k a_k varies
per overpass inside one fixed-shape model, which is what NumPyro needs.

:class:`MaskedSources` holds one configuration and evaluates that sum;
:class:`MultiSourcePosterior` holds posterior draws of the slots and reduces
them to a posterior over K and to per-source summaries that
:func:`plumax.population.event_from_posterior` accepts. Neither needs NumPyro;
the model and its runner are :func:`plumax.gauss_plume.inference.multi_source_plume_model`
and :func:`plumax.gauss_plume.inference.infer_multi_source`. The Tier IV
coupled version (#104) reuses both carriers.
"""

from __future__ import annotations

import dataclasses

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from plumax._src.errors import ParameterError
from plumax.gauss_plume.plume import plume_concentration


class MaskedSources(eqx.Module):
    """K_max source slots with emission rates, locations and activity indicators.

    c(𝐬) = Σ_k a_k Q_k c₁(𝐬; x_k, y_k).

    Attributes:
        emission_rate: Q_k [kg s⁻¹] per slot.
        x: Source x_k [m] per slot.
        y: Source y_k [m] per slot.
        active: Activity a_k per slot, 1 on and 0 off (a float, so a relaxed
            indicator in [0, 1] works too).

    Examples:
        Two slots, one on, one off: the field is the first source's alone.

        >>> import jax.numpy as jnp
        >>> from plumax.gauss_plume.dispersion import BRIGGS_DISPERSION_PARAMS
        >>> sources = MaskedSources(
        ...     emission_rate=jnp.array([0.1, 5.0]), x=jnp.array([0.0, 50.0]),
        ...     y=jnp.array([0.0, 50.0]), active=jnp.array([1.0, 0.0]),
        ... )
        >>> sources.active_count().tolist()
        1.0
        >>> receptors = (jnp.array([300.0]), jnp.array([0.0]), jnp.array([1.0]))
        >>> params = BRIGGS_DISPERSION_PARAMS["D"]
        >>> field = sources.concentration(
        ...     receptors, source_z=2.0, wind_u=5.0, wind_v=0.0, dispersion_params=params
        ... )
        >>> single = plume_concentration(*receptors, 0.0, 0.0, 2.0, 5.0, 0.0, 0.1, params)
        >>> bool(jnp.allclose(field, single))
        True
    """

    emission_rate: Float[Array, " source"]
    x: Float[Array, " source"]
    y: Float[Array, " source"]
    active: Float[Array, " source"]

    def __check_init__(self) -> None:
        shapes = {
            jnp.shape(self.emission_rate),
            jnp.shape(self.x),
            jnp.shape(self.y),
            jnp.shape(self.active),
        }
        if len(shapes) != 1:
            raise ParameterError(
                f"emission_rate, x, y and active must share one shape, got {shapes}"
            )

    @property
    def k_max(self) -> int:
        """Number of slots K_max."""
        return int(jnp.shape(self.emission_rate)[0])

    def active_count(self) -> Float[Array, ""]:
        """K = Σ_k a_k."""
        # K = Σ_k a_k
        return einx.sum("source ->", self.active)

    def unit_responses(
        self,
        receptor_coords: tuple[
            Float[Array, " obs"], Float[Array, " obs"], Float[Array, " obs"]
        ],
        *,
        source_z: float | Float[Array, ""],
        wind_u: float | Float[Array, ""],
        wind_v: float | Float[Array, ""],
        dispersion_params: Float[Array, " param"],
    ) -> Float[Array, "source obs"]:
        """Unit-rate plume c₁(𝐬_i; x_k, y_k) of every slot at every receptor."""
        x_obs, y_obs, z_obs = receptor_coords

        def one_source(source_x, source_y):
            return plume_concentration(
                x_obs,
                y_obs,
                z_obs,
                source_x,
                source_y,
                source_z,
                wind_u,
                wind_v,
                1.0,
                dispersion_params,
            )

        return jax.vmap(one_source)(self.x, self.y)

    def concentration(
        self,
        receptor_coords: tuple[
            Float[Array, " obs"], Float[Array, " obs"], Float[Array, " obs"]
        ],
        *,
        source_z: float | Float[Array, ""],
        wind_u: float | Float[Array, ""],
        wind_v: float | Float[Array, ""],
        dispersion_params: Float[Array, " param"],
    ) -> Float[Array, " obs"]:
        """c(𝐬_i) = Σ_k a_k Q_k c₁(𝐬_i; x_k, y_k) [kg m⁻³]."""
        responses = self.unit_responses(
            receptor_coords,
            source_z=source_z,
            wind_u=wind_u,
            wind_v=wind_v,
            dispersion_params=dispersion_params,
        )
        # a_k Q_k
        weights = self.active * self.emission_rate
        # Σ_k a_k Q_k c₁(𝐬_i; x_k, y_k)
        return einx.dot("source, source obs -> obs", weights, responses)


@dataclasses.dataclass(frozen=True)
class SourceSummary:
    """Posterior summary of one slot, conditioned on the slot being active.

    It satisfies :class:`plumax.population.PerEventPosterior`, so
    ``event_from_posterior(summary, x=summary.x, y=summary.y, …)`` builds the
    slot's catalogue event.

    Attributes:
        emission_rate: Posterior mean of Q_k over draws with a_k = 1 [kg s⁻¹];
            NaN when the slot is never active.
        emission_std: Posterior standard deviation of Q_k over those draws.
        x: Posterior mean of x_k over those draws [m].
        y: Posterior mean of y_k over those draws [m].
        probability_active: Posterior probability P(a_k = 1 | y).
    """

    emission_rate: float
    emission_std: float
    x: float
    y: float
    probability_active: float


@dataclasses.dataclass(frozen=True)
class MultiSourcePosterior:
    """Posterior draws of a masked-K source model.

    Attributes:
        emission_rate: Q_k per draw and slot [kg s⁻¹], shape ``(sample, source)``.
        x: x_k per draw and slot [m].
        y: y_k per draw and slot [m].
        active: a_k per draw and slot (bool).
        background: Background per draw [kg m⁻³], shape ``(sample,)``.

    Examples:
        Three draws of two slots:

        >>> import numpy as np
        >>> posterior = MultiSourcePosterior(
        ...     emission_rate=np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]]),
        ...     x=np.zeros((3, 2)), y=np.zeros((3, 2)),
        ...     active=np.array([[True, False], [True, True], [True, False]]),
        ...     background=np.zeros(3),
        ... )
        >>> posterior.count_probabilities().round(3).tolist()
        [0.0, 0.667, 0.333]
        >>> summary = posterior.per_source(1)
        >>> summary.probability_active, summary.emission_rate
        (0.3333333333333333, 0.6)
    """

    emission_rate: np.ndarray
    x: np.ndarray
    y: np.ndarray
    active: np.ndarray
    background: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            np.shape(self.emission_rate),
            np.shape(self.x),
            np.shape(self.y),
            np.shape(self.active),
        }
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ParameterError(
                "emission_rate, x, y and active must share one (sample, source) "
                f"shape, got {shapes}"
            )

    @property
    def k_max(self) -> int:
        """Number of slots K_max."""
        return int(np.shape(self.active)[1])

    def active_count(self) -> np.ndarray:
        """K = Σ_k a_k per draw, shape ``(sample,)``."""
        # K = Σ_k a_k
        return einx.sum("sample source -> sample", np.asarray(self.active, dtype=int))

    def count_probabilities(self) -> np.ndarray:
        """P(K = k | y) for k = 0, …, K_max, shape ``(K_max + 1,)``."""
        counts = np.bincount(self.active_count(), minlength=self.k_max + 1)
        return counts / counts.sum()

    def saturated_fraction(self) -> float:
        """P(K = K_max | y): the mass at the cap, which drives the RJMCMC promotion."""
        return float(self.count_probabilities()[-1])

    def per_source(self, k: int) -> SourceSummary:
        """Summary of slot ``k`` over the draws where it is active.

        Raises:
            ParameterError: If ``k`` is not a slot index.
        """
        if not 0 <= k < self.k_max:
            raise ParameterError(f"slot k must be in [0, {self.k_max}), got {k}")
        on = np.asarray(self.active[:, k], dtype=bool)
        probability = float(on.mean())
        if not on.any():
            nan = float("nan")
            return SourceSummary(nan, nan, nan, nan, probability)
        rates = np.asarray(self.emission_rate[on, k])
        return SourceSummary(
            emission_rate=float(rates.mean()),
            emission_std=float(rates.std()),
            x=float(np.asarray(self.x[on, k]).mean()),
            y=float(np.asarray(self.y[on, k]).mean()),
            probability_active=probability,
        )


__all__ = ["MaskedSources", "MultiSourcePosterior", "SourceSummary"]
