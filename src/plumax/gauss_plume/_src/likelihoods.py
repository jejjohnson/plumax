"""Observation-noise models for the Tier I inference: Gaussian, Student-t, heteroscedastic.

Every Tier I model writes its likelihood as y = ŷ + ε, with ŷ the forward
prediction (plume or puff plus background) and ε the observation noise. A
:class:`Noise` holds the parameters of p(ε) and does two things with them:

- ``noise(name, mean, obs)`` draws or observes the NumPyro site ``name``
  inside a model, so the plume and puff models swap noise without editing the
  model.
- ``noise.log_prob(obs, mean)`` returns the pointwise log-density log p(y | ŷ)
  in plain JAX, so a training loss (the Step 5 predictor) uses the same
  likelihood without NumPyro.

NumPyro is the ``inference`` extra, so it is imported inside ``__call__`` and
``distribution`` only; building a noise model or calling ``log_prob`` never
needs it.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
from jaxtyping import Array, Float

from plumax._src.errors import ParameterError


if TYPE_CHECKING:
    import numpyro.distributions as dist


def _require_positive(name: str, value: object) -> None:
    # Traced values (a noise built inside jit or a NumPyro trace) skip the
    # eager check; concrete ones get a clear error at construction.
    if isinstance(value, jax.core.Tracer):
        return
    if not bool(np.all(np.asarray(value) > 0.0)):
        raise ParameterError(f"{name} must be > 0, got {value!r}")


class Noise(eqx.Module):
    """Observation-noise model p(y | ŷ) for a Tier I likelihood.

    Subclasses define :meth:`distribution` (the NumPyro distribution, for
    models) and :meth:`log_prob` (plain JAX, for losses); they must agree.
    """

    @abc.abstractmethod
    def distribution(self, mean: Float[Array, "*obs"]) -> dist.Distribution:
        """The NumPyro distribution of y given the prediction ŷ = ``mean``."""

    @abc.abstractmethod
    def log_prob(
        self, obs: Float[Array, "*obs"], mean: Float[Array, "*obs"]
    ) -> Float[Array, "*obs"]:
        """Pointwise log-density log p(y | ŷ), without NumPyro."""

    def __call__(
        self,
        name: str,
        mean: Float[Array, "*obs"],
        obs: Float[Array, "*obs"] | None = None,
    ) -> Float[Array, "*obs"]:
        """Sample (``obs=None``) or observe the NumPyro site ``name`` around ``mean``.

        Args:
            name: NumPyro site name, ``"obs"`` in the Tier I models.
            mean: Forward prediction ŷ.
            obs: Observations y, or ``None`` for predictive draws.

        Returns:
            The site's value: ``obs`` when given, else a draw.
        """
        import numpyro

        return numpyro.sample(name, self.distribution(mean), obs=obs)


class GaussianNoise(Noise):
    """Homoscedastic Gaussian noise, ε ~ 𝒩(0, σ²).

    log p(y | ŷ) = −½ ((y − ŷ)/σ)² − log σ − ½ log 2π.

    The Tier I models default to this with σ = ``obs_noise_std``, so their
    numbers are unchanged.

    Attributes:
        std: Noise standard deviation σ > 0, in the units of y; a scalar or an
            array that broadcasts against the observations.

    Raises:
        ParameterError: If ``std`` is not positive.

    Examples:
        The log-density at the mean of a unit Gaussian is −½ log 2π:

        >>> import jax.numpy as jnp
        >>> noise = GaussianNoise(std=1.0)
        >>> value = noise.log_prob(jnp.zeros(3), jnp.zeros(3))
        >>> value.shape
        (3,)
        >>> bool(jnp.allclose(value, -0.5 * jnp.log(2.0 * jnp.pi)))
        True

        A non-positive scale is rejected:

        >>> GaussianNoise(std=0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ParameterError: std must be > 0, got 0.0
    """

    std: float | Float[Array, "*#obs"]

    def __check_init__(self) -> None:
        _require_positive("std", self.std)

    def distribution(self, mean: Float[Array, "*obs"]) -> dist.Distribution:
        import numpyro.distributions as dist

        return dist.Normal(mean, self.std)

    def log_prob(
        self, obs: Float[Array, "*obs"], mean: Float[Array, "*obs"]
    ) -> Float[Array, "*obs"]:
        # log 𝒩(y; ŷ, σ²)
        return jstats.norm.logpdf(obs, loc=mean, scale=self.std)


class StudentTNoise(Noise):
    """Heavy-tailed Student-t noise, ε ~ t_ν(0, s), for outliers near the detection floor.

    log p(y | ŷ) = log Γ((ν+1)/2) − log Γ(ν/2) − ½ log(ν π) − log s
                   − (ν+1)/2 · log(1 + ((y − ŷ)/s)² / ν).

    ν ≈ 5 is the Tier I design's default for low-SNR retrievals; as ν → ∞ it
    tends to :class:`GaussianNoise` with σ = s.

    Attributes:
        scale: Scale s > 0, in the units of y; a scalar or an array that
            broadcasts against the observations. The standard deviation is
            s √(ν/(ν − 2)) for ν > 2.
        df: Degrees of freedom ν > 0.

    Raises:
        ParameterError: If ``scale`` or ``df`` is not positive.

    Examples:
        A 10-scale outlier costs far less than under a Gaussian:

        >>> import jax.numpy as jnp
        >>> obs, mean = jnp.array([10.0]), jnp.array([0.0])
        >>> student = StudentTNoise(scale=1.0).log_prob(obs, mean)
        >>> gaussian = GaussianNoise(std=1.0).log_prob(obs, mean)
        >>> bool(student[0] > gaussian[0] + 30.0)
        True

        Large ν recovers the Gaussian:

        >>> near = StudentTNoise(scale=1.0, df=1.0e8).log_prob(obs, mean)
        >>> bool(jnp.allclose(near, gaussian, rtol=1.0e-6))
        True
    """

    scale: float | Float[Array, "*#obs"]
    df: float = 5.0

    def __check_init__(self) -> None:
        _require_positive("scale", self.scale)
        _require_positive("df", self.df)

    def distribution(self, mean: Float[Array, "*obs"]) -> dist.Distribution:
        import numpyro.distributions as dist

        return dist.StudentT(self.df, mean, self.scale)

    def log_prob(
        self, obs: Float[Array, "*obs"], mean: Float[Array, "*obs"]
    ) -> Float[Array, "*obs"]:
        # log t_ν(y; ŷ, s)
        return jstats.t.logpdf(obs, self.df, loc=mean, scale=self.scale)


class HeteroscedasticNoise(Noise):
    """Per-observation Gaussian noise, ε_i ~ 𝒩(0, σ_i²), from a retrieval-error map.

    log p(y_i | ŷ_i) = −½ ((y_i − ŷ_i)/σ_i)² − log σ_i − ½ log 2π.

    Unlike :class:`GaussianNoise`, σ must give one value per observation, and
    a mismatch with the observations is an error rather than a broadcast.

    Attributes:
        std: Per-observation standard deviation σ_i > 0, in the units of y.

    Raises:
        ParameterError: If any ``std`` is not positive, or (from ``log_prob``
            and the NumPyro site) if its shape differs from the observations'.

    Examples:
        Doubling σ_i halves the weight of a residual:

        >>> import jax.numpy as jnp
        >>> noise = HeteroscedasticNoise(std=jnp.array([1.0, 2.0]))
        >>> value = noise.log_prob(jnp.array([2.0, 4.0]), jnp.zeros(2))
        >>> bool(jnp.allclose(value[0] - value[1], jnp.log(2.0)))
        True

        Build it from variances, e.g. ``Instrument.observation_variance()``:

        >>> HeteroscedasticNoise.from_variance(jnp.array([4.0, 9.0])).std.tolist()
        [2.0, 3.0]
    """

    std: Float[Array, " obs"]

    def __check_init__(self) -> None:
        _require_positive("std", self.std)

    @classmethod
    def from_variance(cls, variance: Float[Array, " obs"]) -> HeteroscedasticNoise:
        """The noise with σ_i = √(variance_i)."""
        # σ_i = √(R_ii)
        std = jnp.sqrt(jnp.asarray(variance))
        return cls(std=std)

    def _check_shape(self, mean: Float[Array, "*obs"]) -> None:
        std_shape = jnp.shape(self.std)
        mean_shape = jnp.shape(mean)
        if std_shape != mean_shape:
            raise ParameterError(
                f"std has shape {std_shape} but the observations have {mean_shape}"
            )

    def distribution(self, mean: Float[Array, "*obs"]) -> dist.Distribution:
        import numpyro.distributions as dist

        self._check_shape(mean)
        return dist.Normal(mean, self.std)

    def log_prob(
        self, obs: Float[Array, "*obs"], mean: Float[Array, "*obs"]
    ) -> Float[Array, "*obs"]:
        self._check_shape(mean)
        # log 𝒩(y_i; ŷ_i, σ_i²)
        return jstats.norm.logpdf(obs, loc=mean, scale=self.std)


__all__ = ["GaussianNoise", "HeteroscedasticNoise", "Noise", "StudentTNoise"]
