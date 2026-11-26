"""Gaussian puff forward model (JAX).

A continuous source releasing at rate ``Q(t)`` is represented by a sequence
of instantaneous puffs at release times ``t_r^{(i)}`` carrying mass
``m_i = Q(t_r^{(i)}) · Δt_release``. Each puff is advected by the (possibly
time-varying) wind and spreads as a 3-D Gaussian:

    C_i(x, y, z; t) = m_i / [(2π)^{3/2} σ_x σ_y σ_z]
                      · exp[−(x − x_i(t))² / (2 σ_x²)]
                      · exp[−(y − y_i(t))² / (2 σ_y²)]
                      · [exp(−(z − z_s)² / (2 σ_z²))
                         + exp(−(z + z_s)² / (2 σ_z²))]          (ground reflection)

The total field is the superposition ``C = Σ_i C_i`` over all active puffs.
Dispersion coefficients are evaluated as functions of puff travel distance
``s_i(t) = ∫_{t_r^{(i)}}^{t} |V(τ)| dτ``.

Public surface
--------------
- ``release_interval_to_frequency`` / ``frequency_to_release_interval`` : helpers
- ``make_release_times``                 : build evenly spaced release times
- ``PuffState``                           : eqx.Module bundling puff positions/mass
- ``puff_concentration`` / ``puff_concentration_vmap`` : single-puff kernel
- ``evolve_puffs``                        : release_times + wind → PuffState via the wind integrals
- ``simulate_puff_field``                : JIT orchestrator, field at one time
- ``simulate_puff``                       : xarray wrapper with ``time`` axis
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from functax import quad as functax_quad
from integrix import rule as integrix_rule
from jax import jit, vmap
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from plumax._src.dataset import label_concentrations
from plumax._src.domain import check_domain, check_source, flat_mesh, node_axis
from plumax._src.tracing import require_positive
from plumax.gauss_plume._src.dispersion import check_stability
from plumax.gauss_puff._src.wind import _check_time_resolution
from plumax.gauss_puff.dispersion import (
    get_dispersion_scheme,
)
from plumax.gauss_puff.wind import WindSchedule, cumulative_wind_integrals


if TYPE_CHECKING:  # xarray is imported lazily inside simulate_puff
    import xarray as xr

    from plumax.gauss_puff.turbulence import OUTurbulence


# ── Release cadence helpers ──────────────────────────────────────────────────


def release_interval_to_frequency(release_interval: float) -> float:
    """Convert a release interval Δt [s] into a release frequency f [Hz].

    ``f = 1 / Δt``. Both Δt and f are common ways to parametrise puff cadence;
    this pair of helpers makes the conversion explicit.
    """
    require_positive(
        "release_interval", release_interval, caller="release_interval_to_frequency"
    )
    return 1.0 / release_interval


def frequency_to_release_interval(release_frequency: float) -> float:
    """Convert a release frequency f [Hz] into a release interval Δt [s].

    ``Δt = 1 / f``.
    """
    require_positive(
        "release_frequency", release_frequency, caller="frequency_to_release_interval"
    )
    return 1.0 / release_frequency


def _count_steps(span: float, step: float) -> int:
    """K = ⌈T/Δt⌉, snapped to the nearest integer within 1e-9 relative.

    Plain ⌈T/Δt⌉ adds a step of zero length when the quotient is an integer that
    floating point rounds just above: 2.1 / 0.15 is 14 steps and not 15. This is
    tracerax's ``n_steps_for_horizon``; its future home is jejjohnson/integrix#110.

    Examples:
        >>> _count_steps(2.1, 0.15), _count_steps(10.5, 1.0), _count_steps(0.0, 5.0)
        (14, 11, 0)
    """
    # q = T / Δt
    quotient = span / step
    nearest = round(quotient)
    if abs(quotient - nearest) <= 1e-9 * max(1.0, nearest):
        # a span much shorter than the step is still one step
        return nearest if span == 0.0 else max(nearest, 1)
    # K = ⌈q⌉
    return math.ceil(quotient)


def make_release_times(
    t_start: float,
    t_end: float,
    release_frequency: float,
) -> jnp.ndarray:
    """Evenly spaced puff release times in ``[t_start, t_end)``.

    Parameters
    ----------
    t_start, t_end : float
        Simulation time window [s]. ``t_start < t_end`` required.
    release_frequency : float
        Puff release rate [Hz]. Must be > 0.

    Returns
    -------
    release_times : jnp.ndarray, shape (N,)
        Times ``[t_start, t_start + Δt, t_start + 2Δt, ...]`` with
        ``Δt = 1 / release_frequency``, all strictly less than ``t_end``, in
        the default float dtype (float64 with 64-bit JAX).

    Raises
    ------
    ValueError
        If the window is empty, or the times are too far from their epoch for
        the dtype to resolve a hundredth of ``Δt`` (float32 at a Unix time).

    Notes
    -----
    The count is ``ceil((t_end − t_start) / Δt)``, snapped to the nearest
    integer within 1e-9 relative, so a window of K whole intervals gets K
    releases even where floating point rounds the quotient just above K. When
    the window is not an exact multiple of ``Δt`` the **final, partial interval
    still gets a release** (e.g. ``[0, 0.5)`` at 1 Hz yields one puff at ``t = 0``, and
    ``[0, 10.5)`` yields eleven). That trailing puff represents a shorter slice
    of the continuous source, so its emitted mass must be weighted by its actual
    interval duration — see :func:`release_durations` and how ``simulate_puff``
    assigns ``puff_mass``. All release times remain strictly ``< t_end``.
    """
    if not (t_end > t_start):
        raise ValueError(
            f"make_release_times: `t_end` must be > `t_start` "
            f"(got t_start={t_start!r}, t_end={t_end!r})"
        )
    release_interval = frequency_to_release_interval(release_frequency)
    # N = ⌈(t_end − t_start) / Δt⌉, snapped within 1e-9 relative
    num_puffs = _count_steps(t_end - t_start, release_interval)
    # t_i = t_start + i Δt
    times = t_start + np.arange(num_puffs) * release_interval
    release_times = jnp.asarray(times, dtype=float)
    _check_time_resolution(
        times, release_interval, release_times.dtype, caller="make_release_times"
    )
    return release_times


def release_durations(
    release_times: jnp.ndarray,
    t_end: float,
) -> jnp.ndarray:
    """Per-puff emission-interval durations [s] for a continuous source.

    Each puff carries the mass emitted over its interval, i.e. up to the next
    release (or ``t_end`` for the last puff). The durations therefore sum to
    ``t_end − release_times[0]`` exactly: interior intervals equal the release
    spacing ``Δt`` while the final, possibly partial interval is the remainder.
    Multiply by the emission rate ``Q`` to get the per-puff mass.

    Parameters
    ----------
    release_times : jnp.ndarray, shape (N,)
        Monotone increasing release times [s], as from
        :func:`make_release_times`.
    t_end : float
        End of the emission window [s]; must exceed the last release time.

    Returns
    -------
    durations : jnp.ndarray, shape (N,)
        Per-puff interval durations [s], all > 0, in the default float dtype.
    """
    rel = np.asarray(release_times, dtype=np.float64)
    if rel.size == 0:
        return jnp.zeros((0,), dtype=float)
    edges = np.append(rel, float(t_end))
    # Δt_i = t_{i+1} − t_i, with t_N = t_end
    return jnp.asarray(np.diff(edges), dtype=float)


# ── Puff state container ─────────────────────────────────────────────────────


class PuffState(eqx.Module):
    """Container for the Lagrangian state of a puff ensemble at a single time.

    Attributes
    ----------
    release_times : Float[Array, "N"]
        Puff release times [s].
    x, y, z : Float[Array, "N"]
        Puff-centre positions at the evaluation time [m].
    travel_distance : Float[Array, "N"]
        Cumulative travel distance since release [m].
    mass : Float[Array, "N"]
        Puff mass [kg]. Inactive puffs (not yet released) have mass 0.
    """

    release_times: Float[Array, " N"]
    x: Float[Array, " N"]
    y: Float[Array, " N"]
    z: Float[Array, " N"]
    travel_distance: Float[Array, " N"]
    mass: Float[Array, " N"]


# ── Single-puff concentration kernel ─────────────────────────────────────────


@jit
def puff_concentration(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    puff_x: Float[ArrayLike, ""],
    puff_y: Float[ArrayLike, ""],
    puff_z: Float[ArrayLike, ""],
    sigma_x: Float[ArrayLike, ""],
    sigma_y: Float[ArrayLike, ""],
    sigma_z: Float[ArrayLike, ""],
    puff_mass: Float[ArrayLike, ""],
) -> jnp.ndarray:
    """Evaluate the concentration from a single Gaussian puff (JIT).

    Includes ground-reflection via an image source at ``z = −puff_z``.

    Parameters
    ----------
    x, y, z : jnp.ndarray
        Receptor coordinates [m]. Broadcast-compatible.
    puff_x, puff_y, puff_z : float
        Puff-centre position [m]. ``puff_z ≥ 0``.
    sigma_x, sigma_y, sigma_z : float
        Dispersion coefficients [m]. All > 0.
    puff_mass : float
        Mass of the puff [kg]. Must be ≥ 0.

    Returns
    -------
    concentration : jnp.ndarray
        Mass concentration [kg/m³], same shape as the broadcast of the input
        coordinates.
    """
    # Δx = x − x_p
    offset_x = x - puff_x
    # Δy = y − y_p
    offset_y = y - puff_y
    # Δz₋ = z − z_p
    offset_z_direct = z - puff_z
    # Δz₊ = z + z_p
    offset_z_reflected = z + puff_z

    normalization = jnp.power(2.0 * jnp.pi, 1.5) * sigma_x * sigma_y * sigma_z

    # exp(−Δx² / 2σ_x²)
    exp_x = jnp.exp(-0.5 * jnp.square(offset_x / sigma_x))
    # exp(−Δy² / 2σ_y²)
    exp_y = jnp.exp(-0.5 * jnp.square(offset_y / sigma_y))
    # exp(−Δz₋² / 2σ_z²)
    exp_z_direct = jnp.exp(-0.5 * jnp.square(offset_z_direct / sigma_z))
    # exp(−Δz₊² / 2σ_z²)
    exp_z_reflected = jnp.exp(-0.5 * jnp.square(offset_z_reflected / sigma_z))

    return (
        (puff_mass / normalization) * exp_x * exp_y * (exp_z_direct + exp_z_reflected)
    )


puff_concentration_vmap = vmap(
    puff_concentration,
    in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0),
    out_axes=0,
)
"""Vectorised over puffs (per-puff params batched along axis 0, receptor shared)."""


# ── Diffrax-driven puff evolution ────────────────────────────────────────────


def _puffs_at(
    current_time: Float[Array, ""],
    release_times: Float[Array, " N"],
    at_release: tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, " N"]],
    now: tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]],
    source_location: tuple[float, float, float],
    puff_mass: Float[Array, " N"] | float,
    position_disturbance: tuple[Float[Array, " N"], Float[Array, " N"]] | None = None,
) -> PuffState:
    """The puff ensemble at ``current_time`` from the wind integrals.

    x_i = x_s + I_u(t) − I_u(t_i) + Δx_i,  y_i = y_s + I_v(t) − I_v(t_i) + Δy_i,
    s_i = S(t) − S(t_i), for released puffs (t_i ≤ t); a puff not yet released
    sits at the source with no mass and no travel.

    Args:
        current_time: Evaluation time t [s].
        release_times: Release times t_i [s].
        at_release: ``(I_u, I_v, S)`` at the release times [m].
        now: ``(I_u, I_v, S)`` at ``current_time`` [m].
        source_location: ``(x, y, z)`` of the source [m].
        puff_mass: Per-puff mass [kg], or one mass for all puffs.
        position_disturbance: Per-puff ``(Δx, Δy)`` offsets [m], or ``None``.

    Returns:
        The puff state at ``current_time``.
    """
    src_x, src_y, src_z = source_location
    east_release, north_release, travel_release = at_release
    east_now, north_now, travel_now = now
    active = release_times <= current_time
    # x_i = x_src + I_u(t) − I_u(t_i)
    position_x = src_x + jnp.where(active, east_now - east_release, 0.0)
    # y_i = y_src + I_v(t) − I_v(t_i)
    position_y = src_y + jnp.where(active, north_now - north_release, 0.0)
    if position_disturbance is not None:
        offset_x, offset_y = position_disturbance
        # x_i = x_i + Δx_i
        position_x = position_x + jnp.where(active, offset_x, 0.0)
        # y_i = y_i + Δy_i
        position_y = position_y + jnp.where(active, offset_y, 0.0)
    position_z = jnp.full_like(release_times, src_z)
    # s_i = S(t) − S(t_i)
    travel = jnp.where(active, travel_now - travel_release, 0.0)
    mass_arr = jnp.broadcast_to(jnp.asarray(puff_mass), release_times.shape)
    mass = jnp.where(active, mass_arr, 0.0)
    return PuffState(
        release_times=release_times,
        x=position_x,
        y=position_y,
        z=position_z,
        travel_distance=travel,
        mass=mass,
    )


@eqx.filter_jit
def evolve_puffs(
    schedule: WindSchedule,
    release_times: Float[Array, " N"],
    current_time: Float[Array, ""],
    source_location: tuple[float, float, float],
    puff_mass: Float[Array, " N"] | float,
    position_disturbance: tuple[Float[Array, " N"], Float[Array, " N"]] | None = None,
) -> PuffState:
    """Advance a puff ensemble to ``current_time`` under a time-varying wind.

    Uses :func:`cumulative_wind_integrals` to compute ``(I_u, I_v, S)`` at the
    release times and at ``current_time`` in one call, then derives per-puff
    positions and travel distances as differences.

    Puffs with ``release_times > current_time`` are "not yet released" — their
    mass is set to 0 and position is placed at the source, so they contribute
    nothing to the downstream field.

    Parameters
    ----------
    schedule : WindSchedule
        Time-varying wind field.
    release_times : Float[Array, "N"]
        Puff release times [s], none before the schedule's first knot.
    current_time : Float[Array, ""]
        Evaluation time [s].
    source_location : tuple of float, (3,)
        ``(x, y, z)`` source coordinates [m]. ``z ≥ 0``.
    puff_mass : Float[Array, "N"] or float
        Per-puff mass [kg], or a scalar applied to all puffs.
    position_disturbance : tuple of Float[Array, "N"], optional
        Per-puff ``(Δx, Δy)`` offsets [m] applied on top of the wind advection.
        Intended for OU-process turbulence samples (see
        :func:`plumax.gauss_puff.turbulence.sample_ou_offsets`),
        but any caller-provided offset array is accepted. Offsets for
        not-yet-released puffs are ignored.

    Returns
    -------
    puff_state : PuffState
        Puff positions, travel distances, and masses at ``current_time``.
    """
    current_time_array = jnp.asarray(current_time)
    # the release times, then t
    save_at = jnp.concatenate([release_times, einx.id(" -> 1", current_time_array)])
    east, north, travel = cumulative_wind_integrals(schedule, save_at)
    at_release = (east[:-1], north[:-1], travel[:-1])
    now = (east[-1], north[-1], travel[-1])
    return _puffs_at(
        current_time_array,
        release_times,
        at_release,
        now,
        source_location,
        puff_mass,
        position_disturbance,
    )


# ── Field assembly ───────────────────────────────────────────────────────────


def simulate_puff_field(
    receptor_coords: tuple[Float[Array, " P"], Float[Array, " P"], Float[Array, " P"]],
    puff_state: PuffState,
    dispersion_params: Float[Array, " 6"],
    dispersion_fn: Callable,
) -> Float[Array, " P"]:
    """Sum the contributions of all puffs at each receptor.

    Parameters
    ----------
    receptor_coords : tuple of Float[Array, "P"]
        ``(x, y, z)`` receptor coordinates [m].
    puff_state : PuffState
        Per-puff positions, travel distances, and masses.
    dispersion_params : Float[Array, " 6"]
        Dispersion coefficient vector for the chosen scheme and stability class.
    dispersion_fn : callable
        A function with signature ``(distance, params) → (σ_x, σ_y, σ_z)``.
        See :func:`plumax.gauss_puff.dispersion.calculate_pg_dispersion`
        and :func:`calculate_briggs_dispersion_xyz`.

    Returns
    -------
    concentration : Float[Array, "P"]
        Total concentration [kg/m³] at each receptor.
    """
    receptor_x, receptor_y, receptor_z = receptor_coords

    spread_x, crosswind_spread, vertical_spread = dispersion_fn(
        puff_state.travel_distance, dispersion_params
    )

    per_puff = puff_concentration_vmap(
        receptor_x,
        receptor_y,
        receptor_z,
        puff_state.x,
        puff_state.y,
        puff_state.z,
        spread_x,
        crosswind_spread,
        vertical_spread,
        puff_state.mass,
    )
    if per_puff.shape[0] == 0:
        # einx rejects a zero-size axis; no puffs add no concentration
        return jnp.zeros(per_puff.shape[1:], dtype=per_puff.dtype)
    # C(r) = Σ_i C_i(r)
    return einx.sum("[puff] P", per_puff)


# ── Xarray wrapper ───────────────────────────────────────────────────────────


def simulate_puff(
    emission_rate: float | np.ndarray,
    source_location: tuple[float, float, float],
    wind_speed: np.ndarray,
    wind_direction: np.ndarray,
    stability_class: str,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    time_array: np.ndarray,
    release_frequency: float = 1.0,
    scheme: str = "pg",
    background_conc: float = 0.0,
    turbulence: OUTurbulence | None = None,
    turbulence_seed: int | np.random.Generator | None = None,
) -> xr.Dataset:
    """Simulate a time-resolved Gaussian-puff dispersion field on a 3-D grid.

    Wraps :func:`evolve_puffs` + :func:`simulate_puff_field` with grid
    construction, the cumulative integrals of a time-varying wind, and optional
    column integration. Returns an :class:`xarray.Dataset` with variables
    on a ``(time, x, y, z)`` grid.

    Parameters
    ----------
    emission_rate : float or ndarray
        Continuous emission rate Q [kg/s]. Scalar constant Q, or an array of
        length ``n_puffs`` specifying Q at each puff release time. The puff
        mass is ``Q · Δt_release`` with ``Δt_release = 1/release_frequency``.
        Entries must be ≥ 0 (scalar or array); ``Q = 0`` yields a zero field.
    source_location : tuple of float, (3,)
        ``(x, y, z)`` source coordinates [m]. ``z ≥ 0``.
    wind_speed : ndarray, shape (T_w,)
        Wind speed [m/s] at each entry in ``time_array``. Must be same length.
    wind_direction : ndarray, shape (T_w,)
        Wind direction [deg from North]. Same length as ``time_array``.
    stability_class : str
        One of ``'A'``, …, ``'F'``.
    domain_x, domain_y, domain_z : tuple of (float, float, int)
        ``(start, stop, n_points)`` for each axis.
    time_array : ndarray, shape (T,)
        Monotone increasing evaluation times [s].
    release_frequency : float
        Puff release rate [Hz]. Δt_release = 1/release_frequency. Default 1 Hz.
    scheme : str
        Dispersion scheme: ``'pg'`` (Pasquill-Gifford, default) or
        ``'briggs'`` (sharing coefficients with the steady plume model).
    background_conc : float
        Additive background concentration [kg/m³]. Default 0.
    turbulence : OUTurbulence, optional
        Ornstein-Uhlenbeck sub-grid turbulence applied as per-puff ``(Δx, Δy)``
        offsets at release time. Adds realistic meander on top of the
        wind-driven advection. Defaults to ``None`` (deterministic puffs).
    turbulence_seed : int or numpy.random.Generator, optional
        Seed for the OU sampler; required for bit-exact reproducibility
        when ``turbulence`` is set.

    Returns
    -------
    ds : xarray.Dataset
        Variables: ``concentration`` ``(time, x, y, z)`` [kg/m³],
        ``column_concentration`` ``(time, x, y)`` [kg/m²],
        ``wind_speed``/``wind_direction`` ``(time,)``.

    Raises
    ------
    ValueError
        On invalid stability class, non-positive emission rate, shape
        mismatches, unknown scheme, or malformed domain tuples.
    """
    import xarray as xr  # lazy — only this entry point requires xarray

    scheme_params_dict, dispersion_fn = get_dispersion_scheme(scheme)

    check_stability(stability_class, caller="simulate_puff", table=scheme_params_dict)
    check_source(source_location, caller="simulate_puff")
    # the default float dtype: float64 with 64-bit JAX, else float32
    dtype = jnp.result_type(float)
    time_array = np.asarray(time_array, dtype=dtype)
    wind_speed = np.asarray(wind_speed, dtype=dtype)
    wind_direction = np.asarray(wind_direction, dtype=dtype)
    if time_array.ndim != 1 or time_array.size < 2:
        raise ValueError("simulate_puff: `time_array` must be 1-D with ≥ 2 entries")
    if wind_speed.shape != time_array.shape:
        raise ValueError(
            f"simulate_puff: `wind_speed` shape {wind_speed.shape} must match "
            f"`time_array` shape {time_array.shape}"
        )
    if wind_direction.shape != time_array.shape:
        raise ValueError(
            f"simulate_puff: `wind_direction` shape {wind_direction.shape} "
            f"must match `time_array` shape {time_array.shape}"
        )
    require_positive("wind_speed", wind_speed, caller="simulate_puff", allow_zero=True)

    check_domain("domain_x", domain_x, caller="simulate_puff")
    check_domain("domain_y", domain_y, caller="simulate_puff")
    check_domain("domain_z", domain_z, caller="simulate_puff")
    require_positive("release_frequency", release_frequency, caller="simulate_puff")

    dispersion_params = scheme_params_dict[stability_class]

    x_grid = node_axis(domain_x)
    y_grid = node_axis(domain_y)
    z_grid = node_axis(domain_z)
    num_x, num_y, num_levels = len(x_grid), len(y_grid), len(z_grid)
    num_times = len(time_array)

    mesh_x_flat, mesh_y_flat, mesh_z_flat = flat_mesh(x_grid, y_grid, z_grid)
    x_flat = jnp.asarray(mesh_x_flat, dtype=dtype)
    y_flat = jnp.asarray(mesh_y_flat, dtype=dtype)
    z_flat = jnp.asarray(mesh_z_flat, dtype=dtype)

    release_interval = frequency_to_release_interval(release_frequency)
    release_times = make_release_times(
        float(time_array[0]), float(time_array[-1]), release_frequency
    )
    num_puffs = int(release_times.shape[0])
    # Per-puff emission-interval durations: full Δt for interior puffs, the
    # remainder for the final (possibly partial) interval. Weighting mass by
    # these durations conserves the total emitted mass Q·(t_end − t_start) even
    # when the window is not an exact multiple of the release interval.
    durations = release_durations(release_times, float(time_array[-1]))

    if np.ndim(emission_rate) == 0:
        q_scalar = float(emission_rate)
        require_positive(
            "emission_rate", q_scalar, caller="simulate_puff", allow_zero=True
        )
        # m_i = Q Δt_i
        puff_mass = jnp.asarray(q_scalar * durations, dtype=dtype)
    else:
        emission_array = np.asarray(emission_rate, dtype=dtype)
        if emission_array.shape != (num_puffs,):
            raise ValueError(
                f"simulate_puff: array `emission_rate` must have shape "
                f"({num_puffs},) matching the number of releases in "
                f"[t_start, t_end) at release_frequency={release_frequency}, "
                f"got {emission_array.shape}"
            )
        require_positive(
            "emission_rate", emission_array, caller="simulate_puff", allow_zero=True
        )
        # m_i = Q_i Δt_i
        puff_mass = jnp.asarray(emission_array * np.asarray(durations), dtype=dtype)

    schedule = WindSchedule.from_speed_direction(time_array, wind_speed, wind_direction)

    if turbulence is not None:
        from plumax.gauss_puff.turbulence import sample_ou_offsets

        offset_x_np, offset_y_np = sample_ou_offsets(
            turbulence,
            np.asarray(release_times),
            seed=turbulence_seed,
        )
        position_disturbance = (
            jnp.asarray(offset_x_np, dtype=dtype),
            jnp.asarray(offset_y_np, dtype=dtype),
        )
    else:
        position_disturbance = None

    # (I_u, I_v, S) at every release and every output time, in one call; the
    # xarray entry point compiles its per-time field once
    output_times = jnp.asarray(time_array, dtype=dtype)
    save_at = jnp.concatenate([release_times, output_times])
    east, north, travel = cumulative_wind_integrals(schedule, save_at)
    at_release = (east[:num_puffs], north[:num_puffs], travel[:num_puffs])

    @eqx.filter_jit
    def field_at(current_time, east_now, north_now, travel_now):
        puff_state = _puffs_at(
            current_time,
            release_times,
            at_release,
            (east_now, north_now, travel_now),
            source_location,
            puff_mass,
            position_disturbance,
        )
        return simulate_puff_field(
            (x_flat, y_flat, z_flat),
            puff_state,
            dispersion_params,
            dispersion_fn,
        )

    concentration = np.zeros((num_times, num_x, num_y, num_levels), dtype=dtype)
    for time_index in range(num_times):
        output_index = num_puffs + time_index
        field_flat = field_at(
            output_times[time_index],
            east[output_index],
            north[output_index],
            travel[output_index],
        )
        field_np = np.asarray(field_flat)
        concentration[time_index] = einx.id(
            "(x y z) -> x y z", field_np, x=num_x, y=num_y
        )

    # C = C + C_bg
    concentration = concentration + background_conc

    # Trapezoidal vertical integration over the (endpoint-inclusive) z grid;
    # a plain ``sum * dz`` over-counts a vertically uniform component (such as
    # the background) by one grid interval. See gauss_plume.simulate_plume.
    # h_k, the trapezoid weights on the levels z_k
    levels = functax_quad.trapezoid(z_grid)
    # Ω(t, x, y) = Σ_k h_k C(t, x, y, z_k)
    integral = integrix_rule.integrate(levels, concentration, axis=3)
    column = np.asarray(integral.value)

    dataset = xr.Dataset(
        data_vars={
            "concentration": (["time", "x", "y", "z"], concentration),
            "column_concentration": (["time", "x", "y"], column),
            "wind_speed": (["time"], np.asarray(wind_speed)),
            "wind_direction": (["time"], np.asarray(wind_direction)),
        },
        coords={
            "time": np.asarray(time_array),
            "x": x_grid,
            "y": y_grid,
            "z": z_grid,
        },
        attrs={
            "title": "Gaussian puff dispersion (JAX)",
            "source_x": float(source_location[0]),
            "source_y": float(source_location[1]),
            "source_z": float(source_location[2]),
            "stability_class": stability_class,
            "dispersion_scheme": scheme,
            "release_frequency": release_frequency,
            "release_interval": release_interval,
            "n_puffs": num_puffs,
            "background_concentration": background_conc,
            "ou_sigma_fluctuations": (
                float(turbulence.sigma_fluctuations) if turbulence else 0.0
            ),
            "ou_correlation_time": (
                float(turbulence.correlation_time) if turbulence else 0.0
            ),
        },
    )
    label_concentrations(dataset)
    dataset["wind_speed"].attrs = {"long_name": "Wind speed", "units": "m/s"}
    dataset["wind_direction"].attrs = {
        "long_name": "Wind direction",
        "units": "degrees from North (meteorological)",
    }
    return dataset
