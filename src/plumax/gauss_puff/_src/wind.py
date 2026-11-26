"""Time-varying wind schedules and their cumulative integrals.

A Gaussian puff released at time ``t_r`` under a time-varying wind field
``V(t) = (u(t), v(t))`` satisfies the trivial ODE ``dX/dt = V(t)``, whose
integral is

    X(t) − X(t_r) = ∫_{t_r}^{t} V(τ) dτ .

Because every puff sees the same wind, the per-puff offsets are differences
of a single *cumulative* integral evaluated at puff release times and the
current simulation time:

    I_u(t) = ∫_{t_0}^{t} u(τ) dτ                          [m]
    I_v(t) = ∫_{t_0}^{t} v(τ) dτ                          [m]
    S(t)   = ∫_{t_0}^{t} √(u(τ)² + v(τ)²) dτ              [m]   (travel distance)

The wind is linear between its knots, so on the knots and the save times
together the trapezoid rule gives I_u and I_v exactly (integrix's running
trapezoid). The speed |V| is smooth between those nodes except where the wind
passes through zero, so S is integrated per interval by adaptive
Gauss–Kronrod (integrix). Both are differentiable in reverse mode.

Public surface
--------------
- ``WindSchedule``              : equinox.Module wrapping (t, u, v) arrays
- ``cumulative_wind_integrals`` : running integrals ``(I_u, I_v, S)``
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from functax import quad as functax_quad
from functax.interp import Linear, Policies
from geophysix.wind import wind_speed as speed_of
from integrix import adapt as integrix_adapt, scan as integrix_scan
from integrix.march import SampledPath
from jaxtyping import Array, Float
from numpy.typing import DTypeLike

from plumax._src.errors import ParameterError
from plumax._src.tracing import require
from plumax._src.wind import wind_components


def _check_time_resolution(
    times: object, step: float, dtype: DTypeLike, *, caller: str
) -> None:
    """Refuse times too far from their epoch for ``dtype`` to resolve a step.

    ε_t = spacing of ``dtype`` at max |t|; raise when ε_t > Δt / 100, the rule of
    tracerax's ``check_epoch``.

    Args:
        times: Times [s], counted from some epoch.
        step: Δt, the step the times must resolve [s].
        dtype: The dtype the times are kept in.
        caller: The public function the check guards, for the message.

    Raises:
        ValueError: If the dtype's spacing at the largest |t| exceeds Δt / 100.

    Examples:
        >>> import numpy as np
        >>> _check_time_resolution(np.array([0.0, 60.0]), 1.0, np.float32, caller="f")
        >>> _check_time_resolution(  # doctest: +ELLIPSIS
        ...     np.array([1.7e9]), 1.0, np.float32, caller="f"
        ... )
        Traceback (most recent call last):
        ...
        ValueError: f: the times reach 1.7e+09 s from their epoch, where float32 ...
    """
    # max |t|
    largest = float(np.max(np.abs(np.asarray(times, dtype=np.float64)), initial=0.0))
    # ε_t = spacing of the dtype at max |t|
    spacing = float(np.spacing(np.asarray(largest, dtype=dtype)))
    if spacing > step / 100.0:
        raise ValueError(
            f"{caller}: the times reach {largest:.6g} s from their epoch, where "
            f"{np.dtype(dtype).name} resolves {spacing:.3g} s, more than a "
            f"hundredth of the {step:g} s step; count times from an epoch near "
            "the run, or enable 64-bit JAX"
        )


class WindSchedule(eqx.Module):
    """Piecewise-linear time-varying wind field.

    Attributes
    ----------
    times : Float[Array, "T"]
        Monotone increasing time knots [s].
    u_wind : Float[Array, "T"]
        x-component of wind velocity at the knots [m/s].
    v_wind : Float[Array, "T"]
        y-component of wind velocity at the knots [m/s].

    Notes
    -----
    The schedule behaves as a piecewise-linear interpolant between knots.
    Queries outside ``[times[0], times[-1]]`` linearly extrapolate — consumers
    that need strict clamping should range-check times before calling. A
    single knot is a constant wind.
    """

    times: Float[Array, " T"]
    u_wind: Float[Array, " T"]
    v_wind: Float[Array, " T"]

    @classmethod
    def from_speed_direction(
        cls,
        times: np.ndarray | jnp.ndarray,
        wind_speed: np.ndarray | jnp.ndarray,
        wind_direction: np.ndarray | jnp.ndarray,
    ) -> WindSchedule:
        """Build a schedule from (speed, direction) in meteorological "from" convention.

        ``wind_direction = 270°`` is a wind *from* the west (u > 0);
        ``wind_direction = 0°`` is a wind *from* the north (v < 0).

            u = −|V| · sin(θ),   v = −|V| · cos(θ),   θ = direction (radians).

        Parameters
        ----------
        times : array-like, shape (T,)
            Monotone time knots [s].
        wind_speed : array-like, shape (T,)
            Wind speed at each knot [m/s]. Must be ≥ 0.
        wind_direction : array-like, shape (T,)
            Wind direction at each knot [degrees from North].

        Returns
        -------
        WindSchedule
            In the default float dtype (float64 with 64-bit JAX).

        Raises
        ------
        ValueError
            If the knot times are too far from their epoch for the dtype to
            resolve a hundredth of their finest spacing.
        """
        # the default float dtype: float64 with 64-bit JAX, else float32
        times_j = jnp.asarray(times, dtype=float)
        speed_j = jnp.asarray(wind_speed, dtype=float)
        wind_east, wind_north = wind_components(
            speed_j, jnp.asarray(wind_direction, dtype=float)
        )
        knots = np.asarray(times, dtype=np.float64)
        if knots.size > 1:
            # Δt_min = min_k (t_{k+1} − t_k), the finest knot spacing
            finest = float(np.min(np.abs(np.diff(knots))))
            _check_time_resolution(
                knots, finest, times_j.dtype, caller="WindSchedule.from_speed_direction"
            )
        return cls(times=times_j, u_wind=wind_east, v_wind=wind_north)

    def _path(self) -> SampledPath:
        """û(t) = (u(t), v(t)): linear between knots, continued linearly past them.

        A single knot is a constant wind, read through a second knot 1 s later.
        """
        times, u_wind, v_wind = self.times, self.u_wind, self.v_wind
        if times.shape[0] == 1:
            # one knot: û(t) = û₀ for every t
            times = einx.id("one, one -> (one + one)", times, times + 1.0)
            u_wind = einx.id("one, one -> (one + one)", u_wind, u_wind)
            v_wind = einx.id("one, one -> (one + one)", v_wind, v_wind)
        # û_k = (u_k, v_k)
        samples = einx.id("knot, knot -> knot (1 + 1)", u_wind, v_wind)
        return SampledPath.fit(
            Linear(), times, samples, policies=Policies(outside="extrapolate")
        )

    def wind_at(self, t: Float[Array, ""]) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """The wind (u(t), v(t)) at time ``t``, linear between knots.

        u(t) = (1 − λ) u_k + λ u_{k+1},  λ = (t − t_k) / (t_{k+1} − t_k),
        continued along the first or last segment outside the knots.
        """
        wind_east, wind_north = self._path().evaluate(t)
        return wind_east, wind_north

    def speed_at(self, t: Float[Array, ""]) -> Float[Array, ""]:
        """The wind speed |V(t)| = √(u(t)² + v(t)²), with a zero gradient in calm air."""
        wind_east, wind_north = self.wind_at(t)
        # |V| = √(u² + v²), by geophysix's double where
        return speed_of(wind_east, wind_north)


def cumulative_wind_integrals(
    schedule: WindSchedule,
    save_at: Float[Array, " M"],
    rtol: float = 1e-8,
    atol: float = 1e-6,
) -> tuple[Float[Array, " M"], Float[Array, " M"], Float[Array, " M"]]:
    """Running integrals of the wind and of its speed from the first knot.

    I_u(s) = ∫_{t_0}^{s} u dτ,  I_v(s) = ∫_{t_0}^{s} v dτ,  S(s) = ∫_{t_0}^{s} |V| dτ.

    Algorithm:
        1. Put the knots and the save times together and sort them: the nodes
           t_n, between which the wind is linear.
        2. I_u and I_v: the running trapezoid sum on the nodes,
           I(t_n) = Σ_{m ≤ n} ½ (t_m − t_{m−1}) (û_{m−1} + û_m), exact for a
           linear wind (``integrix.scan.cumulative``).
        3. S: each interval [t_{n−1}, t_n] by adaptive Gauss–Kronrod, summed
           (``integrix.adapt.cumulative_adaptive``); |V| has a kink where the
           wind passes through zero, which the bisection resolves.
        4. Read each save time at its node.

    The save times may come in any order and may repeat. Times past the last
    knot read the wind continued along the last segment. Every output has a
    finite reverse-mode gradient, in calm air too.

    Parameters
    ----------
    schedule : WindSchedule
        Time-varying wind field.
    save_at : Float[Array, "M"]
        Times [s] at which to return the integrals, none before
        ``schedule.times[0]``.
    rtol, atol : float
        Relative and absolute tolerance [m] of the Gauss–Kronrod rule on each
        interval of S; ``rtol`` is held at 10 ε of the wind's dtype or above
        (1.2e-6 in float32), which the rule can still meet.

    Returns
    -------
    I_u, I_v, S : Float[Array, "M"]
        Cumulative integrals of u, v, and wind speed [m], each at ``save_at``.

    Raises
    ------
    ParameterError
        If a save time is before the first knot (when ``save_at`` is traced,
        when the compiled function runs).

    Examples
    --------
    A wind from 5 to 3 m/s over 10 s, read halfway: I_u = 5·5 − ½·0.2·5² = 22.5 m.

    >>> import jax.numpy as jnp
    >>> schedule = WindSchedule(
    ...     times=jnp.array([0.0, 10.0]),
    ...     u_wind=jnp.array([5.0, 3.0]),
    ...     v_wind=jnp.zeros(2),
    ... )
    >>> east, north, travel = cumulative_wind_integrals(schedule, jnp.array([5.0, 10.0]))
    >>> [round(float(value), 6) for value in east]
    [22.5, 40.0]
    >>> bool(jnp.allclose(travel, east))
    True
    """
    save_at = jnp.asarray(save_at)
    first_knot = schedule.times[0]
    save_at = require(
        save_at >= first_knot,
        "cumulative_wind_integrals: a save time is before the wind schedule's "
        "first knot; the integrals start there",
        value=save_at,
        error=ParameterError,
    )
    return _integrals(schedule, save_at, rtol, atol)


@eqx.filter_jit
def _integrals(
    schedule: WindSchedule,
    save_at: Float[Array, " M"],
    rtol: float,
    atol: float,
) -> tuple[Float[Array, " M"], Float[Array, " M"], Float[Array, " M"]]:
    """The integrals of :func:`cumulative_wind_integrals`, compiled once per shape."""
    path = schedule._path()
    # t_n = sort(knots ∪ save times); a repeat is an interval of zero width
    nodes = jnp.sort(einx.id("knot, save -> (knot + save)", schedule.times, save_at))
    # û(t_n), one row per component
    wind = einx.id("node component -> component node", jax.vmap(path.evaluate)(nodes))
    # I(t_n) = Σ_{m ≤ n} ½ (t_m − t_{m−1}) (û_{m−1} + û_m)
    drift = integrix_scan.cumulative(functax_quad.trapezoid(nodes), wind)
    # a tolerance below the dtype's rounding can never be met, and the rule
    # would bisect to its buffer's end: hold rtol at 10 ε of the wind's dtype
    resolution = 10.0 * float(jnp.finfo(schedule.u_wind.dtype).eps)
    rule = integrix_adapt.GaussKronrod(rtol=max(rtol, resolution), atol=atol)

    def speed(time: Float[Array, ""]) -> Float[Array, ""]:
        # |V(t)| = √(u² + v²), by geophysix's double where
        wind_east, wind_north = path.evaluate(time)
        return speed_of(wind_east, wind_north)

    # S(t_n) = Σ_{m ≤ n} ∫_{t_{m−1}}^{t_m} |û(τ)| dτ
    travel_integral = integrix_adapt.cumulative_adaptive(rule, speed, nodes)
    # an interval that ran out of subintervals would come back less accurate
    travel = eqx.error_if(
        travel_integral.value,
        ~travel_integral.converged,
        "cumulative_wind_integrals: the travel distance did not converge to "
        "rtol and atol; loosen them or add knots",
    )
    # n(s): the last node at or before the save time s
    where = jnp.searchsorted(nodes, save_at, side="right") - 1
    east, north = drift[0], drift[1]
    return east[where], north[where], travel[where]
