"""Registry intensities and POD curves as xtremax point processes.

The Tier V generative layer runs on ``xtremax.point_processes``: plumax owns the
methane-domain pieces (the λ(t) kernels of :mod:`plumax.population.intensity`
and the P_d(Q) curves of :mod:`plumax.population.pod`) and this module adapts
them to xtremax's protocols, so the Poisson log-likelihood, the thinning
sampler, the marked process and the detection-thinned process are xtremax's.

* :class:`LogIntensity` is a registry kernel as an xtremax log-intensity:
  log λ(t₀ + s) on window time s, with the compensator Λ(b) − Λ(a) and the
  thinning envelope λ_max.
* :func:`make_process` builds the ``InhomogeneousPoissonProcess`` on a window.
* :class:`PODRetention` is a POD curve as a ``ThinningProcess`` retention,
  p(t, H, Q) = P_d(Q).
* :func:`detection_process` composes the ground process, a mark distribution
  and the retention into ``ThinningProcess(MarkedTemporalPointProcess(...))``.
* :func:`padded_event_buffer` lays detection times out in xtremax's padded
  buffer: a contiguous mask prefix, padding at the window length T.

xtremax is the ``population`` extra and is imported only when a process is
built or integrated numerically.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float

from plumax.population._src._xtremax import require_xtremax
from plumax.population.intensity import INTENSITY_REGISTRY, WeibullRenewalIntensity


#: Default quadrature density for kernels without a closed-form compensator:
#: one Simpson node a minute. The sigmoid-gated kernels switch over about a
#: minute and the clipped sinusoids have kinks; at this density every registry
#: kernel's compensator is within 1e-6 (relative) of dense trapezoid
#: quadrature. Lower it for long windows when the kernel is smooth.
NODES_PER_DAY = 1440


class LogIntensity(eqx.Module):
    """A registry intensity as an xtremax log-intensity on window time.

    log λ̃(s) = log λ(t₀ + s), with s ∈ [0, T] the time since the window start
    t₀ in days, so the seasonal and diurnal phases of the kernel stay tied to
    calendar days.

    Args:
        intensity: A kernel from :mod:`plumax.population.intensity`.
        origin: Window start t₀ [days].
        n_integration_points: Simpson nodes for the compensator when the kernel
            has no closed form.

    Examples:
        >>> import jax.numpy as jnp
        >>> from plumax.population.intensity import ConstantIntensity
        >>> log_intensity = LogIntensity(ConstantIntensity(lambda_0=2.0), origin=0.0)
        >>> bool(jnp.allclose(log_intensity(jnp.zeros(3)), jnp.log(2.0)))
        True
        >>> float(log_intensity.integrate(0.0, 3.0))
        6.0
    """

    intensity: Callable[[Array], Array]
    origin: Float[Array, ""]
    n_integration_points: int = eqx.field(static=True, default=NODES_PER_DAY + 1)

    def __init__(
        self,
        intensity: Callable[[Array], Array],
        origin: ArrayLike = 0.0,
        n_integration_points: int = NODES_PER_DAY + 1,
    ) -> None:
        self.intensity = intensity
        self.origin = jnp.asarray(origin, dtype=float)
        self.n_integration_points = n_integration_points

    def __call__(self, time: Float[Array, "*time"]) -> Float[Array, "*time"]:
        # λ(t₀ + s)
        rate = self.intensity(self.origin + time)
        # a clipped kernel is exactly 0 between bursts; floor it at the
        # smallest normal number so log λ stays finite
        floor = jnp.finfo(rate.dtype).tiny
        # log λ̃(s)
        return jnp.log(jnp.maximum(rate, floor))

    def integrate(self, start: ArrayLike, stop: ArrayLike) -> Float[Array, "*time"]:
        """Compensator Λ̃(b) − Λ̃(a) = ∫ₐᵇ λ(t₀ + s) ds.

        The kernel's own ``integrate`` when it has one (exact), else composite
        Simpson quadrature through ``xtremax.point_processes``.

        Args:
            start: Lower limit a [days since t₀].
            stop: Upper limit b [days since t₀].

        Returns:
            Expected event count on ``[a, b]``.

        Raises:
            ImportError: Without the ``population`` extra, for kernels without
                a closed form.
        """
        closed_form = getattr(self.intensity, "integrate", None)
        if closed_form is not None:
            return closed_form(self.origin + start, self.origin + stop)
        point_processes = require_xtremax("LogIntensity.integrate")
        return point_processes.integrate_log_intensity(
            self,
            jnp.asarray(start, dtype=float),
            jnp.asarray(stop, dtype=float),
            n_points=self.n_integration_points,
            method="simpson",
        )

    def max_intensity(self) -> Float[Array, ""]:
        """Thinning envelope λ_max ≥ λ(t) for all t, from the kernel."""
        return self.intensity.max_intensity()


def _quadrature_points(window_length: float, nodes_per_day: int) -> int:
    # odd, so Simpson's rule uses the nodes as given
    nodes = max(math.ceil(window_length * nodes_per_day), 2)
    return nodes + 1 if nodes % 2 == 0 else nodes


def make_process(
    intensity: str | eqx.Module,
    window: tuple[float, float],
    *,
    nodes_per_day: int = NODES_PER_DAY,
    **params: Any,
) -> Any:
    """An xtremax inhomogeneous Poisson process for a registry intensity.

    N(t) on s ∈ [0, T], T = t₁ − t₀, with log-intensity log λ(t₀ + s).

    Args:
        intensity: A registry name (see
            :data:`~plumax.population.intensity.INTENSITY_REGISTRY`) or a
            kernel instance.
        window: ``(t₀, t₁)``, the observation window [days].
        nodes_per_day: Quadrature density for kernels without a closed-form
            compensator.
        **params: Kernel parameters, when ``intensity`` is a name.

    Returns:
        ``xtremax.point_processes.operators.InhomogeneousPoissonProcess`` over
        :class:`LogIntensity`; its ``log_prob(times, mask)`` takes times since
        t₀ (see :func:`padded_event_buffer`).

    Raises:
        KeyError: For an unknown registry name.
        ValueError: For an empty window, parameters passed with an instance,
            or the Weibull renewal hazard (a renewal process, not a Poisson
            intensity: use ``xtremax.point_processes.operators.RenewalProcess``).
        ImportError: Without the ``population`` extra.

    Examples:
        >>> import jax.numpy as jnp
        >>> process = make_process("constant", (0.0, 10.0), lambda_0=0.5)
        >>> times = jnp.array([1.0, 4.0, 10.0])
        >>> mask = jnp.array([True, True, False])
        >>> bool(jnp.allclose(process.log_prob(times, mask), 2 * jnp.log(0.5) - 5.0))
        True
    """
    point_processes = require_xtremax("make_process")
    if isinstance(intensity, str):
        kernel = INTENSITY_REGISTRY[intensity](**params)
    elif params:
        raise ValueError("make_process: pass parameters only with a registry name.")
    else:
        kernel = intensity
    if isinstance(kernel, WeibullRenewalIntensity):
        raise ValueError(
            "make_process: the Weibull renewal hazard depends on the time since "
            "the last event, so it is a renewal process, not a Poisson intensity; "
            "use xtremax.point_processes.operators.RenewalProcess."
        )
    start, stop = (float(edge) for edge in window)
    if not stop > start:
        raise ValueError(f"make_process: window must have t₁ > t₀, got {window}.")
    window_length = stop - start
    n_points = _quadrature_points(window_length, nodes_per_day)
    log_intensity = LogIntensity(kernel, origin=start, n_integration_points=n_points)
    return point_processes.operators.InhomogeneousPoissonProcess(
        log_intensity,
        observation_window=window_length,
        n_integration_points=n_points,
    )


class PODRetention(eqx.Module):
    """A probability-of-detection curve as an xtremax retention callable.

    p(t, H, Q) = P_d(c · Q; covariates), with c converting the mark units to
    the POD's flux units (the registry PODs take kg/h, so c = 3600 for marks in
    kg/s). Time and history are ignored: detection depends on the mark alone.

    Args:
        pod: A model from :mod:`plumax.population.pod`.
        covariates: Fixed keyword covariates of the POD beyond the flux (wind
            speed ``U``, pixel size ``p``, …), by its argument names.
        mark_scale: The factor c from mark units to the POD's flux units.

    Examples:
        >>> import jax.numpy as jnp
        >>> from plumax.population.pod import LogLogisticPOD
        >>> retention = PODRetention(LogLogisticPOD(beta_0=0.0, beta_1=2.0))
        >>> float(retention(0.0, None, jnp.asarray(1.0)))
        0.5
    """

    pod: Callable[..., Array]
    covariates: dict[str, Array]
    mark_scale: float = eqx.field(static=True, default=1.0)

    def __init__(
        self,
        pod: Callable[..., Array],
        covariates: Mapping[str, ArrayLike] | None = None,
        mark_scale: float = 1.0,
    ) -> None:
        self.pod = pod
        self.covariates = {
            name: jnp.asarray(value) for name, value in (covariates or {}).items()
        }
        self.mark_scale = mark_scale

    def __call__(
        self,
        time: ArrayLike,
        history: Any,
        proposed_mark: Float[Array, "*mark"] | None = None,
    ) -> Float[Array, "*mark"]:
        del time, history
        if proposed_mark is None:
            raise ValueError(
                "PODRetention depends on the mark: pass `marks` to log_prob and a "
                "`mark_sample_key` for the compensator."
            )
        # flux in the POD's units, c Q
        flux = self.mark_scale * proposed_mark
        # P_d(c Q)
        return self.pod(flux, **self.covariates)


def detection_process(
    ground: Any,
    mark_distribution: Any,
    retention: Callable[..., Array],
    *,
    n_integration_points: int | None = None,
) -> Any:
    """The detected-event process: POD-thinned, marked ground process.

    λ_det(t, Q) = p(t, H, Q) · λ(t) · f(Q), as
    ``ThinningProcess(MarkedTemporalPointProcess(ground, f), p)``.

    Args:
        ground: The ground process, e.g. from :func:`make_process`.
        mark_distribution: The mark law f(Q), a NumPyro distribution.
        retention: The retention p, e.g. a :class:`PODRetention`.
        n_integration_points: Trapezoid nodes of the retention compensator.
            Default: the ground process's own.

    Returns:
        ``xtremax.point_processes.operators.ThinningProcess``. Its
        ``log_prob(times, mask, marks, mark_sample_key=key)`` averages the
        retention over f by Monte Carlo inside the compensator (xtremax's
        ``n_mark_samples``, see jejjohnson/xtremax#103).

    Raises:
        ImportError: Without the ``population`` extra.
    """
    point_processes = require_xtremax("detection_process")
    operators = point_processes.operators
    marked = operators.MarkedTemporalPointProcess(
        ground, point_processes.constant_mark_distribution(mark_distribution)
    )
    if n_integration_points is None:
        n_integration_points = getattr(ground, "n_integration_points", 100)
    return operators.ThinningProcess(
        marked, retention, n_integration_points=n_integration_points
    )


def padded_event_buffer(
    times: ArrayLike | Sequence[float],
    window: tuple[float, float],
    *,
    max_events: int | None = None,
) -> tuple[Float[Array, " event"], Bool[Array, " event"]]:
    """Detection times as xtremax's padded buffer on window time.

    Sorts s_i = t_i − t₀, pads to ``max_events`` with T = t₁ − t₀ and returns
    the mask, which is a contiguous prefix of ``True``.

    Args:
        times: Detection times t_i [days].
        window: ``(t₀, t₁)`` [days]; every time must lie inside it.
        max_events: Buffer length. Default: the number of events.

    Returns:
        ``(event_times, mask)`` for xtremax ``log_prob`` calls.

    Raises:
        ValueError: If a time lies outside the window or the buffer is too
            short.

    Examples:
        >>> event_times, mask = padded_event_buffer([3.0, 1.0], (0.0, 5.0), max_events=3)
        >>> event_times.tolist(), mask.tolist()
        ([1.0, 3.0, 5.0], [True, True, False])
    """
    start, stop = (float(edge) for edge in window)
    host_times = np.sort(np.asarray(times, dtype=float).ravel())
    if host_times.size and (host_times[0] < start or host_times[-1] > stop):
        raise ValueError(
            f"padded_event_buffer: every time must lie in the window {window}."
        )
    n_events = host_times.size
    length = n_events if max_events is None else max_events
    if length < n_events:
        raise ValueError(
            f"padded_event_buffer: max_events={length} < {n_events} events."
        )
    window_time = np.full(length, stop - start)
    window_time[:n_events] = host_times - start
    mask = np.arange(length) < n_events
    return jnp.asarray(window_time), jnp.asarray(mask)
