"""Persistency metrics of a fitted intensity, and their posterior versions.

The operational layer of Tier V (06c): from λ(t), the expected wait until the next
event, the probability of an event in a horizon and the expected count, each as a
point value of one kernel or as posterior samples through :func:`posterior_metric`.
All three rest on one compensator, Λ(a, b) = ∫ₐᵇ λ(t) dt, from
:class:`~plumax.population.processes.LogIntensity`: the kernel's closed form when it
has one, Simpson quadrature through xtremax otherwise.

Time is in days and λ in events day⁻¹, as in
:mod:`plumax.population.intensity`.

The metrics hold for Poisson-family intensities, whose future does not depend on
the past. A Hawkes process's are conditional on the event history, so it is not
supported yet; the Weibull renewal hazard is rejected for the same reason.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from plumax.population._src.fitting_iw import TMTPPPosterior
from plumax.population._src.processes import (
    NODES_PER_DAY,
    LogIntensity,
    _quadrature_points,
    poisson_kernel,
)
from plumax.population.intensity import ConstantIntensity
from plumax.population.point_process import (
    InhomogeneousIntensityPosterior,
    PoissonRatePosterior,
)


#: Shortest and longest wait the log-spaced survival grid spans directly [days].
MIN_WAIT = 1e-6
MAX_WAIT = 1e4


class _LogLinearIntensity(eqx.Module):
    # the log-linear fit's intensity as a kernel: λ(t) = c exp(β₀ + z(t) · β)
    beta0: Float[Array, ""]
    beta: Float[Array, " covariate"]
    covariates: Callable[[Array], Array]
    rate_scale: float

    def __call__(self, t: Array) -> Array:
        # z(t) · β
        linear = einx.dot(
            "... covariate, covariate -> ...", self.covariates(t), self.beta
        )
        # c exp(β₀ + z(t) · β)
        return self.rate_scale * jnp.exp(self.beta0 + linear)


def _compensator(
    kernel: Any, start: ArrayLike, stop: ArrayLike, *, n_points: int
) -> Float[Array, "*time"]:
    # Λ(a, b) = ∫ₐᵇ λ(t) dt
    log_intensity = LogIntensity(kernel, n_integration_points=n_points)
    return log_intensity.integrate(start, stop)


def expected_count(
    intensity: str | eqx.Module,
    t: ArrayLike,
    horizon: float,
    *,
    nodes_per_day: int = NODES_PER_DAY,
    **theta: Any,
) -> Float[Array, ""]:
    """Expected number of events in the horizon, E[N(t, t + h)] = Λ(t, t + h).

    Args:
        intensity: A registry name or a kernel instance.
        t: Start time t [days].
        horizon: Horizon h > 0 [days], a concrete number since it sets the node count.
        nodes_per_day: Quadrature density for kernels without a closed form.
        **theta: Kernel parameters, when ``intensity`` is a name.

    Returns:
        The expected count.

    Raises:
        ValueError: For the Weibull renewal hazard, or parameters with an instance.

    Examples:
        >>> import jax.numpy as jnp
        >>> count = expected_count("constant", 3.0, 2.0, lambda_0=1.5)
        >>> bool(jnp.allclose(count, 3.0))
        True

        A diurnal cycle integrates to its mean over a whole day:

        >>> count = expected_count(
        ...     "diurnal", 0.0, 1.0, lambda_0=2.0, amplitude=0.5, phase_hours=14.0
        ... )
        >>> bool(jnp.allclose(count, 2.0))
        True
    """
    kernel = poisson_kernel(intensity, theta, caller="expected_count")
    n_points = _quadrature_points(float(horizon), nodes_per_day)
    start = jnp.asarray(t, dtype=float)
    return _compensator(kernel, start, start + horizon, n_points=n_points)


def occurrence_probability(
    intensity: str | eqx.Module,
    t: ArrayLike,
    horizon: float,
    *,
    nodes_per_day: int = NODES_PER_DAY,
    **theta: Any,
) -> Float[Array, ""]:
    """Probability of at least one event, P(N(t, t + h) ≥ 1) = 1 − exp(−Λ(t, t + h)).

    Args:
        intensity: A registry name or a kernel instance.
        t: Start time t [days].
        horizon: Horizon h > 0 [days], a concrete number since it sets the node count.
        nodes_per_day: Quadrature density for kernels without a closed form.
        **theta: Kernel parameters, when ``intensity`` is a name.

    Returns:
        The probability.

    Raises:
        ValueError: For the Weibull renewal hazard, or parameters with an instance.

    Examples:
        >>> import jax.numpy as jnp
        >>> probability = occurrence_probability("constant", 0.0, 2.0, lambda_0=0.5)
        >>> bool(jnp.allclose(probability, 1.0 - jnp.exp(-1.0)))
        True
    """
    count = expected_count(intensity, t, horizon, nodes_per_day=nodes_per_day, **theta)
    # 1 − exp(−Λ), accurate for small Λ
    return -jnp.expm1(-count)


def expected_wait_time(
    intensity: str | eqx.Module,
    t: ArrayLike,
    *,
    n_cells: int = 2048,
    nodes_per_cell: int = 9,
    **theta: Any,
) -> Float[Array, ""]:
    """Expected wait until the next event, E[Δt | t] = ∫₀^∞ exp(−Λ(t, t + s)) ds.

    Exactly 1/λ₀ for :class:`~plumax.population.intensity.ConstantIntensity`, whatever
    t is.

    Algorithm:
        1. Lay out a survival grid: s = 0, then ``n_cells`` log-spaced cells up to
           ``MAX_WAIT``, so the first cells resolve short waits and the last ones long
           waits at a fixed cost.
        2. Accumulate Λ(t, t + s) cell by cell with Simpson's rule.
        3. Trapezoid the survival function S(s) = exp(−Λ(t, t + s)) over the grid.
        4. Close the tail beyond s_max with the mean rate λ̄ of the last decade,
           ∫ S ds ≈ S(s_max) / λ̄, which is exact for a constant rate.

    Late cells outgrow a diurnal cycle, which is safe because S only has weight out
    there when λ is small enough that the ripple in Λ is small too.

    Args:
        intensity: A registry name or a kernel instance.
        t: Start time t [days].
        n_cells: Log-spaced cells of the survival grid.
        nodes_per_cell: Simpson nodes of Λ inside one cell (odd).
        **theta: Kernel parameters, when ``intensity`` is a name.

    Returns:
        The expected wait [days]; infinite for λ ≡ 0.

    Raises:
        ValueError: For the Weibull renewal hazard, or parameters with an instance.

    Examples:
        The memoryless wait of a constant rate:

        >>> import jax.numpy as jnp
        >>> wait = expected_wait_time("constant", 7.0, lambda_0=4.0)
        >>> bool(jnp.allclose(wait, 0.25))
        True

        A diurnal source is quicker to emit near its 14:00 peak than at 02:00:

        >>> diurnal = dict(lambda_0=4.0, amplitude=3.6, phase_hours=8.0)
        >>> at_peak = expected_wait_time("diurnal", 14.0 / 24.0, **diurnal)
        >>> at_trough = expected_wait_time("diurnal", 2.0 / 24.0, **diurnal)
        >>> bool(at_peak < at_trough)
        True
    """
    kernel = poisson_kernel(intensity, theta, caller="expected_wait_time")
    start = jnp.asarray(t, dtype=float)
    if isinstance(kernel, ConstantIntensity):
        # E[Δt] = 1 / λ₀
        return 1.0 / jnp.asarray(kernel.lambda_0, dtype=start.dtype)

    # --- survival grid ---
    log_spaced = jnp.geomspace(MIN_WAIT, MAX_WAIT, n_cells, dtype=start.dtype)
    lags = einx.id("first, rest -> (first + rest)", jnp.zeros(1), log_spaced)

    # --- compensator ---
    # ΔΛₖ = ∫ λ(t + s) ds over cell k
    increments = jax.vmap(
        lambda low, high: _compensator(
            kernel, start + low, start + high, n_points=nodes_per_cell
        )
    )(lags[:-1], lags[1:])
    # Λ(t, t + sₖ) = Σ_{j ≤ k} ΔΛⱼ
    cumulative = einx.id(
        "first, rest -> (first + rest)", jnp.zeros(1), jnp.cumsum(increments)
    )

    # --- survival ---
    # S(sₖ) = exp(−Λ(t, t + sₖ))
    survival = jnp.exp(-cumulative)
    body = trapezoid(survival, lags)
    # λ̄ over the last decade [s_max / 10, s_max]
    last_decade = lags >= 0.1 * MAX_WAIT
    decade_start = jnp.argmax(last_decade)
    mean_rate = (cumulative[-1] - cumulative[decade_start]) / (
        lags[-1] - lags[decade_start]
    )
    # ∫_{s_max}^∞ S ds ≈ S(s_max) / λ̄
    tail = survival[-1] / mean_rate
    return body + tail


def _poisson_draws(
    posterior: PoissonRatePosterior,
    *,
    key: PRNGKeyArray | None,
    num_samples: int,
    rate_scale: float,
) -> Float[Array, " member"]:
    if key is None:
        raise ValueError("posterior_metric: a Gamma posterior needs a key to draw.")
    # λ⁽ᵉ⁾ ~ Gamma(α, β), per unit of exposure
    shape = jnp.asarray(posterior.alpha, dtype=float)
    draws = jax.random.gamma(key, shape, (num_samples,), dtype=shape.dtype)
    return rate_scale * draws / posterior.beta


def posterior_metric(
    metric: Callable[..., Array],
    posterior: PoissonRatePosterior | InhomogeneousIntensityPosterior | TMTPPPosterior,
    *args: Any,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    rate_scale: float = 1.0,
    covariates: Callable[[Array], Array] | None = None,
    num_samples: int = 1000,
    key: PRNGKeyArray | None = None,
    **metric_kwargs: Any,
) -> dict[str, Array]:
    """Posterior samples of a persistency metric, and their quantiles.

    m⁽ᵉ⁾ = m(λ⁽ᵉ⁾, …) for each draw λ⁽ᵉ⁾ of the intensity, vmapped over the draws.

    The two rate posteriors of :mod:`~plumax.population.point_process` are per unit of
    their exposure, and their time and area units are the caller's, so ``rate_scale``
    turns them into events day⁻¹ (the area, times the time unit per day, when the
    exposure was area × duration).

    Args:
        metric: ``metric(kernel, *args, **metric_kwargs)``, such as
            :func:`expected_wait_time`.
        posterior: The fitted posterior.
        *args: The metric's positional arguments after the intensity.
        quantiles: Quantile levels to report.
        rate_scale: Multiplier from the posterior's rate to events day⁻¹, for
            :class:`~plumax.population.point_process.PoissonRatePosterior` and
            :class:`~plumax.population.point_process.InhomogeneousIntensityPosterior`.
        covariates: z(t), the covariates of an
            :class:`~plumax.population.point_process.InhomogeneousIntensityPosterior`
            at time t, with a trailing covariate axis.
        num_samples: Draws from a Gamma posterior (the others carry their draws).
        key: PRNG key of a Gamma posterior.
        **metric_kwargs: The metric's keyword arguments.

    Returns:
        ``{"samples": m⁽ᵉ⁾, "levels": the quantile levels, "quantiles": their values}``.

    Raises:
        ValueError: Without ``key`` for a Gamma posterior, or without ``covariates``
            for a log-linear one.

    Examples:
        Wait time under a Gamma posterior on a constant rate:

        >>> import jax
        >>> from plumax.population.point_process import PoissonRatePosterior
        >>> posterior = PoissonRatePosterior(
        ...     alpha=201.0, beta=100.0, n_events=200, exposure=100.0
        ... )
        >>> result = posterior_metric(
        ...     expected_wait_time, posterior, 0.0, key=jax.random.PRNGKey(0)
        ... )
        >>> result["samples"].shape
        (1000,)
        >>> bool(result["quantiles"][0] < 0.5 < result["quantiles"][2])
        True
    """
    if isinstance(posterior, PoissonRatePosterior):
        draws = _poisson_draws(
            posterior, key=key, num_samples=num_samples, rate_scale=rate_scale
        )

        def member(draw: Array) -> Array:
            kernel = ConstantIntensity(lambda_0=draw)
            return metric(kernel, *args, **metric_kwargs)

    elif isinstance(posterior, InhomogeneousIntensityPosterior):
        if covariates is None:
            raise ValueError(
                "posterior_metric: a log-linear posterior needs covariates z(t)."
            )
        draws = (
            jnp.asarray(posterior.beta0_samples),
            jnp.asarray(posterior.beta_samples),
        )

        def member(draw: Any) -> Array:
            beta0, beta = draw
            kernel = _LogLinearIntensity(beta0, beta, covariates, rate_scale)
            return metric(kernel, *args, **metric_kwargs)

    else:
        draws = {name: jnp.asarray(value) for name, value in posterior.samples.items()}

        def member(draw: Any) -> Array:
            return metric(posterior.kernel(draw), *args, **metric_kwargs)

    samples = jax.vmap(member)(draws)
    levels = jnp.asarray(quantiles, dtype=samples.dtype)
    return {
        "samples": samples,
        "levels": levels,
        "quantiles": jnp.quantile(samples, levels),
    }
